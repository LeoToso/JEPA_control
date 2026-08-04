"""Reproducible closed-loop control evaluation for cartpole JEPA checkpoints.

Evaluates every checkpoint on identical initial conditions.  LQR uses the
learned local Jacobian with the normalized-action derivative converted back to
physical Newtons.  CEM uses the nonlinear learned predictor directly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import yaml

from control.cem import CEMLatentPlanner
from control.jacobian import compute_jacobian_np
from control.lqr import solve_discrete_lqr
from control.rollout import evaluate_stabilization_mpc
from data.dataset import load_discrete_dataset_meta
from envs.cartpole_visual import ContinuousCartpoleVisual
from models.jepa import make_jepa


class LearnedLQRPlanner:
    def __init__(self, A, B, K, lb, ub):
        self.A, self.B, self.K = A, B, K
        self.action_lb, self.action_ub = float(lb), float(ub)

    def reset(self):
        pass

    def plan(self, z, z_star):
        u = np.clip(-self.K @ (z - z_star), self.action_lb, self.action_ub)
        z1 = self.A @ (z - z_star) + self.B @ u + z_star
        return [np.asarray(u).reshape(-1)], np.stack([z, z1])


class DecodedStateLQRPlanner:
    """Exact physical LQR driven by the checkpoint's decoded visual state."""
    def __init__(self, latent_dim, K_phys, state_head_state, state_mean,
                 state_std, lb, ub):
        head = nn.Linear(latent_dim, 4)
        head.load_state_dict(state_head_state)
        self.W = head.weight.detach().cpu().numpy()
        self.b = head.bias.detach().cpu().numpy()
        self.mean = np.asarray(state_mean)
        self.std = np.asarray(state_std)
        self.K = K_phys
        self.action_lb, self.action_ub = float(lb), float(ub)
        self.A = np.eye(latent_dim)
        self.B = np.zeros((latent_dim, 1))

    def reset(self):
        pass

    def plan(self, z, z_star):
        # LQR regulates deviations from the upright equilibrium.  Subtract the
        # decoder's own equilibrium output so constant decoder bias (including
        # normalization mean and head bias) cannot create a persistent force.
        state_delta_norm = self.W @ (z - z_star)
        state_delta_phys = state_delta_norm * self.std
        u = np.clip(-self.K @ state_delta_phys, self.action_lb, self.action_ub)
        return [np.asarray(u).reshape(-1)], np.stack([z, z])


def physical_macro_jacobian(env, eps_x=1e-4, eps_u=1e-3):
    """Finite-difference the exact frame-skip environment at equilibrium."""
    A = np.zeros((4, 4))
    B = np.zeros((4, 1))
    for j in range(4):
        dx = np.zeros(4)
        dx[j] = eps_x
        env.reset_to_state(dx)
        _, xp, _, _, _ = env.step(0.0)
        env.reset_to_state(-dx)
        _, xm, _, _, _ = env.step(0.0)
        A[:, j] = (xp - xm) / (2.0 * eps_x)
    env.reset_to_state(np.zeros(4))
    _, xp, _, _, _ = env.step(eps_u)
    env.reset_to_state(np.zeros(4))
    _, xm, _, _, _ = env.step(-eps_u)
    B[:, 0] = (xp - xm) / (2.0 * eps_u)
    return A, B


def build_model(model_cfg, checkpoint, device):
    arch = dict(model_cfg)
    arch.update(checkpoint.get('config', {}))
    model = make_jepa(
        variant=arch.get('variant', 'E-full'),
        latent_dim=int(arch.get('latent_dim', 8)),
        action_dim=int(arch.get('action_dim', 1)),
        action_latent_dim=int(arch.get('action_latent_dim', 1)),
        action_encoder=arch.get('action_encoder', 'none'),
        encoder_type=arch.get('encoder_type', 'vit'),
        image_size=int(arch.get('image_size', 64)),
        patch_size=int(arch.get('patch_size', 8)),
        frame_stack=int(arch.get('frame_stack', 1)),
        use_frame_diff=bool(arch.get('use_frame_diff', False)),
        vit_embed_dim=int(arch.get('vit_embed_dim', 128)),
        vit_depth=int(arch.get('vit_depth', 4)),
        vit_num_heads=int(arch.get('vit_num_heads', 4)),
        vit_mlp_ratio=float(arch.get('vit_mlp_ratio', 2.0)),
        predictor_type=arch.get('predictor_type', 'mlp'),
        predictor_window=int(arch.get('predictor_window', 1)),
        predictor_hidden_dim=int(arch.get('predictor_hidden_dim', 256)),
        predictor_n_layers=int(arch.get('predictor_n_layers', 2)),
        predictor_activation=arch.get('predictor_activation', 'elu'),
        predictor_embed_dim=int(arch.get('predictor_embed_dim', 128)),
        predictor_depth=int(arch.get('predictor_depth', 4)),
        predictor_num_heads=int(arch.get('predictor_num_heads', 4)),
        predictor_mlp_ratio=float(arch.get('predictor_mlp_ratio', 4.0)),
    ).to(device)
    state = checkpoint.get('model_state', checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model, arch


def make_env(cfg, seed):
    e, d = cfg['environment'], cfg.get('data', {})
    return ContinuousCartpoleVisual(
        frame_skip=int(e.get('frame_skip', 1)),
        image_size=int(e.get('image_size', 64)),
        action_range=tuple(e.get('action_range', [-10, 10])),
        mass_cart=float(e.get('mass_cart', 1.0)),
        mass_pole=float(e.get('mass_pole', 0.1)),
        pole_length=float(e.get('pole_length', 0.5)),
        gravity=float(e.get('gravity', 9.8)), dt=float(e.get('dt', 0.02)),
        friction_cart=float(e.get('friction_cart', 0.0)),
        friction_pole=float(e.get('friction_pole', 0.0)),
        # Match the dataset's visible region; the environment default is only 12 degrees.
        theta_threshold=float(d.get('max_abs_theta', 1.2)), seed=seed)


def equilibrium_latent(model, env, device):
    obs, _, _ = env.reset_to_state(np.zeros(4, dtype=np.float32))
    x = torch.from_numpy(obs).float().permute(2, 0, 1)[None].to(device) / 255.0
    with torch.no_grad():
        return model.encode_obs(x, x).cpu().numpy()[0]


def physical_action_jacobian(model, B_encoded, action_scale, device):
    """Convert dz_next/d(action embedding) to dz_next/d(raw Newtons)."""
    u = torch.zeros(1, model.config.action_dim, device=device, requires_grad=True)
    a = model.action_encoder(u)[0]
    rows = []
    for j in range(a.numel()):
        rows.append(torch.autograd.grad(a[j], u, retain_graph=True)[0][0])
    J = torch.stack(rows).detach().cpu().numpy()
    return B_encoded @ J / float(action_scale)


def lifted_state_cost(checkpoint, latent_dim, state_std, device):
    state = checkpoint.get('state_head_state')
    if not state:
        print('[warn] checkpoint has no state head; using identity latent cost')
        return np.eye(latent_dim)
    head = nn.Linear(latent_dim, 4).to(device)
    head.load_state_dict(state)
    W = head.weight.detach().cpu().numpy()
    C = np.diag(np.asarray(state_std, dtype=np.float64)) @ W
    Qx = np.diag([1.0, 1.0, 10.0, 1.0])
    return C.T @ Qx @ C + 1e-4 * np.eye(latent_dim)


def compact(result):
    return {k: v for k, v in result.items() if k != 'vis_result'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--checkpoints', nargs='+', required=True)
    p.add_argument('--controller', choices=['lqr', 'cem', 'decoded_lqr', 'both', 'all'], default='lqr')
    p.add_argument('--trials', type=int, default=50)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', default='results/cartpole_control_eval.json')
    p.add_argument('--cem-samples', type=int, default=None)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    meta = load_discrete_dataset_meta(args.data)
    action_scale = float(meta.get('action_scale', 1.0))
    state_mean = np.asarray(meta.get('state_mean', np.zeros(4)))
    state_std = np.asarray(meta.get('state_std', np.ones(4)))
    ctrl = cfg.get('control', {})
    bounds = cfg['environment'].get('action_range', [-10, 10])
    if args.controller == 'both':
        controllers = ['lqr', 'cem']
    elif args.controller == 'all':
        controllers = ['decoded_lqr', 'lqr', 'cem']
    else:
        controllers = [args.controller]
    rows = []

    for ckpt_path in args.checkpoints:
        checkpoint = torch.load(ckpt_path, map_location=device)
        model, arch = build_model(cfg['model'], checkpoint, device)
        env = make_env(cfg, args.seed)
        z_star = equilibrium_latent(model, env, device)
        Qz = lifted_state_cost(checkpoint, model.latent_dim, state_std, device)

        A, B_encoded = compute_jacobian_np(model, z_star, device=device)
        B_raw = physical_action_jacobian(model, B_encoded, action_scale, device)
        rho = float(np.max(np.abs(np.linalg.eigvals(A))))
        print(f'\n[{Path(ckpt_path).name}] rho(A)={rho:.4f}  '
              f'||B_raw||={np.linalg.norm(B_raw):.5f}  action_scale={action_scale:g}')

        for kind in controllers:
            if kind == 'decoded_lqr':
                if 'state_head_state' not in checkpoint:
                    raise RuntimeError(
                        f'{ckpt_path} has no state_head_state for decoded LQR')
                A_phys, B_phys = physical_macro_jacobian(env)
                R = np.array([[float(ctrl.get('R_lqr', 0.01))]])
                K_phys, _, poles = solve_discrete_lqr(
                    A_phys, B_phys, np.diag([1., 1., 10., 1.]), R)
                planner = DecodedStateLQRPlanner(
                    model.latent_dim, K_phys, checkpoint['state_head_state'],
                    state_mean, state_std, bounds[0], bounds[1])
                print(f'[decoded_lqr] rho(A_phys-B_phys K)='
                      f'{np.max(np.abs(poles)):.4f}')
            elif kind == 'lqr':
                R = np.array([[float(ctrl.get('R_lqr', 0.01))]])
                K, _, poles = solve_discrete_lqr(A, B_raw, Qz, R)
                planner = LearnedLQRPlanner(A, B_raw, K, bounds[0], bounds[1])
                print(f'[lqr] rho(A-BK)={np.max(np.abs(poles)):.4f}')
            else:
                cem = cfg.get('cem', {})
                planner = CEMLatentPlanner(
                    predictor=model, action_encoder=model.action_encoder,
                    predictor_window=int(arch.get('predictor_window', 1)),
                    Q=Qz, R=float(ctrl.get('R_lqr', 0.01)),
                    horizon=int(cem.get('horizon', 15)),
                    chunk_size=int(cem.get('chunk_size', 1)),
                    n_samples=int(args.cem_samples or cem.get('n_samples', 2000)),
                    n_elites=int(cem.get('n_elites', 100)),
                    n_iter=int(cem.get('n_iter', 15)),
                    init_std=float(cem.get('init_std', 1.0)),
                    action_lb=float(bounds[0]), action_ub=float(bounds[1]),
                    action_scale=action_scale, action_dim=model.config.action_dim,
                    device=device)

            result = evaluate_stabilization_mpc(
                encoder=model.encoder, mpc=planner, env=env,
                n_trials=args.trials, T=args.steps,
                init_scale=float(ctrl.get('init_scale', 0.05)),
                stabilization_threshold=float(ctrl.get('stabilization_threshold', 0.1)),
                settling_threshold=float(ctrl.get('settling_threshold', 0.05)),
                success_hold_steps=int(ctrl.get('success_hold_steps', 10)),
                failure_penalty=float(ctrl.get('failure_penalty', 1e4)),
                seed=args.seed, device=device, z_star=z_star,
                frame_stack=int(arch.get('frame_stack', 1)),
                use_frame_diff=bool(arch.get('use_frame_diff', False)))
            row = {'checkpoint': ckpt_path, 'epoch': checkpoint.get('epoch'),
                   'controller': kind, 'rho_A': rho,
                   'B_raw_norm': float(np.linalg.norm(B_raw)), **compact(result)}
            rows.append(row)
            print(f"[{kind}] success={row['success_rate']:.1%}  "
                  f"stable={row['mean_fraction_stable']:.1%}  cost={row['mean_cost']:.1f}")
        env.close()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f'\n[done] {out}')


if __name__ == '__main__':
    main()
