"""Diagnose whether cartpole control failures come from physics or learned dynamics."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from control.lqr import solve_discrete_lqr
from data.dataset import load_discrete_dataset_meta
from experiments.evaluate_cartpole_control import (
    build_model, equilibrium_latent, make_env, physical_action_jacobian)
from control.jacobian import compute_jacobian_np


def physical_jacobian(env, eps_x=1e-4, eps_u=1e-3):
    A = np.zeros((4, 4)); B = np.zeros((4, 1))
    for j in range(4):
        dx = np.zeros(4); dx[j] = eps_x
        env.reset_to_state(dx); _, xp, _, _, _ = env.step(0.0)
        env.reset_to_state(-dx); _, xm, _, _, _ = env.step(0.0)
        A[:, j] = (xp - xm) / (2 * eps_x)
    env.reset_to_state(np.zeros(4)); _, xp, _, _, _ = env.step(eps_u)
    env.reset_to_state(np.zeros(4)); _, xm, _, _, _ = env.step(-eps_u)
    B[:, 0] = (xp - xm) / (2 * eps_u)
    return A, B


def oracle_eval(env, K, trials, steps, init_scale, seed, threshold, hold=10):
    rng = np.random.RandomState(seed)
    success, terminated, costs, sat = [], [], [], []
    Q = np.diag([1., 1., 10., 1.])
    for _ in range(trials):
        _, x, _ = env.reset_to_state(rng.uniform(-init_scale, init_scale, 4))
        xs, us, done = [], [], False
        for _ in range(steps):
            u = float(np.clip((-K @ x).item(), env.action_low, env.action_high))
            xs.append(x.copy()); us.append(u)
            _, x, _, done, _ = env.step(u)
            if done:
                break
        errs = np.linalg.norm(np.vstack(xs + [x]), axis=1)
        success.append(bool(not done and np.all(errs[-min(hold, len(errs)):] < threshold)))
        terminated.append(done)
        stage = sum(float(xx @ Q @ xx + .01 * uu * uu) for xx, uu in zip(xs, us))
        costs.append(stage + (1e4 if done else 0.0))
        sat.append(np.mean(np.abs(us) >= .99 * env.action_high))
    return dict(success_rate=float(np.mean(success)),
                termination_rate=float(np.mean(terminated)),
                mean_cost=float(np.mean(costs)), saturation_rate=float(np.mean(sat)))


def empirical_impulse_response(model, env, device, action_scale,
                               forces=(0.5, 1., 3.), horizon=3):
    obs0, _, _ = env.reset_to_state(np.zeros(4))
    x0 = torch.from_numpy(obs0).float().permute(2, 0, 1)[None].to(device) / 255.
    with torch.no_grad():
        z0 = model.encode_obs(x0, x0)
    rows = []
    W = int(model.config.predictor_window)
    for force in forces:
        real_branches, pred_branches = [], []
        for sign in (1., -1.):
            env.reset_to_state(np.zeros(4))
            prev_obs = obs0
            real_zs = []
            z = z0.clone()
            z_hist = [z] * W
            u_hist = [torch.zeros(1, model.config.action_dim, device=device)] * W
            pred_zs = []
            for h in range(horizon):
                u_raw = sign * force if h == 0 else 0.0
                obs1, _, _, _, _ = env.step(u_raw)
                prev_t = torch.from_numpy(prev_obs).float().permute(2, 0, 1)[None].to(device) / 255.
                x1 = torch.from_numpy(obs1).float().permute(2, 0, 1)[None].to(device) / 255.
                u = torch.tensor([[u_raw / action_scale]], device=device)
                u_hist.append(u)
                with torch.no_grad():
                    real_zs.append(model.encode_obs(x1, prev_t).cpu().numpy()[0])
                    z = model.predict(
                        torch.stack(z_hist[-W:], dim=1),
                        torch.stack(u_hist[-W:], dim=1))
                z_hist.append(z)
                pred_zs.append(z.cpu().numpy()[0])
                prev_obs = obs1
            real_branches.append(np.stack(real_zs))
            pred_branches.append(np.stack(pred_zs))
        B_real = (real_branches[0] - real_branches[1]) / (2 * force)
        B_pred = (pred_branches[0] - pred_branches[1]) / (2 * force)
        for h in range(horizon):
            denom = np.linalg.norm(B_real[h]) * np.linalg.norm(B_pred[h])
            rows.append((
                force, h + 1, np.linalg.norm(B_real[h]), np.linalg.norm(B_pred[h]),
                float(B_real[h] @ B_pred[h] / denom) if denom > 1e-12 else np.nan,
                np.linalg.norm(B_pred[h] - B_real[h])
                / max(np.linalg.norm(B_real[h]), 1e-12)))
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True); p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', required=True); p.add_argument('--trials', type=int, default=50)
    p.add_argument('--steps', type=int, default=200); p.add_argument('--seed', type=int, default=123)
    p.add_argument('--device', default='cuda')
    a = p.parse_args(); device = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    with open(a.config) as f: cfg = yaml.safe_load(f)
    env = make_env(cfg, a.seed); ctrl = cfg.get('control', {})
    Ap, Bp = physical_jacobian(env)
    K, _, poles = solve_discrete_lqr(Ap, Bp, np.diag([1., 1., 10., 1.]), np.array([[.01]]))
    print(f'[oracle] rho(A)={max(abs(np.linalg.eigvals(Ap))):.4f}  '
          f'rho(A-BK)={max(abs(poles)):.4f}  ||B||={np.linalg.norm(Bp):.5f}')
    print('[oracle]', oracle_eval(env, K, a.trials, a.steps,
          float(ctrl.get('init_scale', .05)), a.seed,
          float(ctrl.get('stabilization_threshold', .1))))

    ck = torch.load(a.checkpoint, map_location=device)
    model, _ = build_model(cfg['model'], ck, device)
    zstar = equilibrium_latent(model, env, device)
    Al, Be = compute_jacobian_np(model, zstar, device=device)
    scale = float(load_discrete_dataset_meta(a.data).get('action_scale', 1.))
    Bl = physical_action_jacobian(model, Be, scale, device)
    print(f'[learned] rho(A)={max(abs(np.linalg.eigvals(Al))):.4f}  ||B_raw||={np.linalg.norm(Bl):.5f}')
    print('[impulse test] force step  ||real||    ||pred||   cosine  relative_error')
    for row in empirical_impulse_response(model, env, device, scale):
        print(f'               {row[0]:4.1f}   {row[1]:2d}   {row[2]:10.5f}  '
              f'{row[3]:10.5f}  {row[4]:+7.3f}  {row[5]:10.3f}')
    env.close()


if __name__ == '__main__': main()
