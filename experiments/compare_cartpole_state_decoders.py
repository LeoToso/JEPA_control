#!/usr/bin/env python
"""Compare global ridge, local ridge, and nonlinear frozen-latent decoders."""
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

from control.lqr import solve_discrete_lqr
from data.dataset import load_discrete_dataset_meta
from experiments.evaluate_cartpole_control import (
    build_model, make_env, physical_macro_jacobian,
)
from experiments.evaluate_cartpole_encoder_gt_cem import (
    GroundTruthCEM, encode_online,
)
from experiments.probe_cartpole_nonlinear_instability import (
    decode, encode_split, fit_ridge_probe, probe_metrics,
)


STATE_NAMES = ('x', 'x_dot', 'theta', 'theta_dot')


def fit_weighted_ridge(z, states, state_mean, state_std, ridge, weights):
    y = (states - state_mean) / state_std
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.mean()
    z_mean = np.average(z, axis=0, weights=weights)
    y_mean = np.average(y, axis=0, weights=weights)
    zc, yc = z - z_mean, y - y_mean
    zw = zc * np.sqrt(weights[:, None])
    yw = yc * np.sqrt(weights[:, None])
    gram = zw.T @ zw
    w = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), zw.T @ yw)
    b = y_mean - z_mean @ w
    return w, b


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 4))

    def forward(self, z):
        return self.net(z)


def fit_mlp(z, states, state_mean, state_std, hidden, epochs, batch_size,
            lr, seed, device):
    rng = np.random.RandomState(seed)
    z_mean = z.mean(0).astype(np.float32)
    z_std = np.maximum(z.std(0), 1e-4).astype(np.float32)
    zn = ((z - z_mean) / z_std).astype(np.float32)
    yn = ((states - state_mean) / state_std).astype(np.float32)
    model = MLPDecoder(z.shape[1], hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    zt = torch.from_numpy(zn)
    yt = torch.from_numpy(yn)
    model.train()
    for epoch in range(epochs):
        order = rng.permutation(len(z))
        total = 0.
        for start in range(0, len(z), batch_size):
            idx = order[start:start + batch_size]
            xb, yb = zt[idx].to(device), yt[idx].to(device)
            loss = torch.mean((model(xb) - yb) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss) * len(idx)
        if epoch in (0, epochs - 1) or (epoch + 1) % 50 == 0:
            print(f'[mlp epoch {epoch + 1:03d}] loss={total / len(z):.6f}')
    model.eval()
    return model, z_mean, z_std


def mlp_decode(model, z, z_mean, z_std, state_mean, state_std, device):
    zn = ((np.asarray(z) - z_mean) / z_std).astype(np.float32)
    with torch.no_grad():
        y = model(torch.from_numpy(zn).to(device)).cpu().numpy()
    pred = y * state_std + state_mean
    pred[..., 2] = np.arctan2(np.sin(pred[..., 2]), np.cos(pred[..., 2]))
    return pred


def metric_groups(truth, pred, region, K):
    local = (np.abs(truth) <= region[None]).all(1)
    very_local = (np.abs(truth) <= (.5 * region)[None]).all(1)
    groups = {'global': np.ones(len(truth), dtype=bool),
              'local': local, 'very_local': very_local}
    out = {}
    for name, mask in groups.items():
        if not np.any(mask):
            out[name] = {'n': 0}
            continue
        row = probe_metrics(truth[mask], pred[mask])
        u_true = -(truth[mask] @ K.ravel())
        u_pred = -(pred[mask] @ K.ravel())
        row['action_mae'] = float(np.mean(np.abs(u_pred - u_true)))
        row['action_bias'] = float(np.mean(u_pred - u_true))
        row['action_sign_agreement'] = float(np.mean(
            np.sign(u_pred) == np.sign(u_true)))
        out[name] = row
    return out


def evaluate_cem(cfg, args, encoder, decoder_fn, terminal_q, initial_states,
                 device):
    bounds = cfg['environment'].get('action_range', [-10., 10.])
    ctrl = cfg.get('control', {})
    rows = []
    for trial, x0 in enumerate(initial_states):
        env = make_env(cfg, args.seed + trial)
        planner = GroundTruthCEM(
            cfg, args.cem_horizon, args.cem_samples, args.cem_elites,
            args.cem_iters, args.cem_init_std, args.cem_warm_start_std,
            1., terminal_q, bounds[0], bounds[1], device)
        obs, state, _ = env.reset_to_state(x0)
        prev_obs = obs
        norms = []
        terminated = False
        for step in range(args.steps):
            z = encode_online(encoder, obs, prev_obs, device)
            estimate = decoder_fn(z[None])[0]
            torch.manual_seed(args.seed * 10000 + trial * 1000 + step)
            action = planner.plan_state(estimate)
            norms.append(float(np.linalg.norm(state)))
            old_obs = obs
            obs, state, _, done, _ = env.step(action)
            prev_obs = old_obs
            if done:
                terminated = True
                break
        env.close()
        norms.append(float(np.linalg.norm(state)))
        hold = min(int(ctrl.get('success_hold_steps', 10)), len(norms))
        success = bool(not terminated and np.all(
            np.asarray(norms[-hold:]) <
            float(ctrl.get('stabilization_threshold', .1))))
        rows.append({'success': success, 'terminated': terminated,
                     'steps': len(norms) - 1, 'final_error': norms[-1],
                     'max_error': max(norms)})
    return {
        'success_rate': float(np.mean([r['success'] for r in rows])),
        'termination_rate': float(np.mean([r['terminated'] for r in rows])),
        'mean_final_error': float(np.mean([r['final_error'] for r in rows])),
        'trials': rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--train-samples', type=int, default=6000)
    p.add_argument('--test-samples', type=int, default=1500)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--ridge', type=float, default=1e-3)
    p.add_argument('--local-weight', type=float, default=25.)
    p.add_argument('--mlp-hidden', type=int, default=128)
    p.add_argument('--mlp-epochs', type=int, default=200)
    p.add_argument('--mlp-batch-size', type=int, default=256)
    p.add_argument('--mlp-lr', type=float, default=1e-3)
    p.add_argument('--control-trials', type=int, default=3)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--cem-horizon', type=int, default=2)
    p.add_argument('--cem-samples', type=int, default=2048)
    p.add_argument('--cem-elites', type=int, default=128)
    p.add_argument('--cem-iters', type=int, default=8)
    p.add_argument('--cem-init-std', type=float, default=3.)
    p.add_argument('--cem-warm-start-std', type=float, default=1.)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', required=True)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    encoder, _ = build_model(cfg['model'], checkpoint, device)
    meta = load_discrete_dataset_meta(args.data)
    state_mean = np.asarray(meta.get('state_mean', np.zeros(4)))
    state_std = np.asarray(meta.get('state_std', np.ones(4)))
    region = np.asarray(cfg.get('training', {}).get(
        'local_sampling_region', [.1, .25, .05, .5]), dtype=np.float64)

    print('[probe] encoding frozen train/test latents ...')
    z_train, s_train = encode_split(
        encoder, args.data, 'train', args.train_samples,
        args.batch_size, args.seed, device)
    z_test, s_test = encode_split(
        encoder, args.data, 'test', args.test_samples,
        args.batch_size, args.seed + 1, device)
    local_train = (np.abs(s_train) <= region[None]).all(1)
    print(f'[probe] local train={local_train.sum()}/{len(local_train)}  '
          f'region={region.tolist()}  weight={args.local_weight:g}')

    global_params = fit_ridge_probe(
        z_train, s_train, state_mean, state_std, args.ridge)
    weights = np.where(local_train, args.local_weight, 1.)
    local_params = fit_weighted_ridge(
        z_train, s_train, state_mean, state_std, args.ridge, weights)
    mlp, z_mean, z_std = fit_mlp(
        z_train, s_train, state_mean, state_std, args.mlp_hidden,
        args.mlp_epochs, args.mlp_batch_size, args.mlp_lr, args.seed, device)

    eq_env = make_env(cfg, args.seed + 999)
    eq_obs, _, _ = eq_env.reset_to_state(np.zeros(4, dtype=np.float32))
    z_eq = encode_online(encoder, eq_obs, eq_obs, device)
    A, B = physical_macro_jacobian(eq_env)
    K, terminal_q, poles = solve_discrete_lqr(
        A, B, np.diag([1., 1., 10., 1.]), np.array([[.01]]))
    eq_env.close()

    raw_decoders = {
        'global_ridge': lambda z: decode(
            z, *global_params, state_mean, state_std),
        'local_weighted_ridge': lambda z: decode(
            z, *local_params, state_mean, state_std),
        'mlp': lambda z: mlp_decode(
            mlp, z, z_mean, z_std, state_mean, state_std, device),
    }
    result = {'checkpoint': args.checkpoint, 'state_names': STATE_NAMES,
              'local_region': region.tolist(), 'local_weight': args.local_weight,
              'rho_lqr': float(np.max(np.abs(poles))), 'decoders': {}}
    rng = np.random.RandomState(args.seed)
    scale = float(cfg.get('control', {}).get('init_scale', .05))
    initial_states = [rng.uniform(-scale, scale, 4).astype(np.float32)
                      for _ in range(args.control_trials)]
    for name, raw_fn in raw_decoders.items():
        pred = raw_fn(z_test)
        eq = raw_fn(z_eq[None])[0]
        centered_fn = lambda z, fn=raw_fn, offset=eq: fn(z) - offset
        metrics = metric_groups(s_test, pred, region, K)
        cem = evaluate_cem(
            cfg, args, encoder, centered_fn, terminal_q,
            initial_states, device)
        result['decoders'][name] = {
            'equilibrium_decode': eq.tolist(),
            'metrics': metrics, 'cem': cem,
        }
        local = metrics['local']
        print(f'[{name}] global_R2={np.round(metrics["global"]["r2"], 3).tolist()}')
        print(f'  local n={local["n"]}  RMSE={np.round(local["rmse"], 4).tolist()}  '
              f'action_MAE={local["action_mae"]:.3f}')
        print(f'  CEM success={cem["success_rate"]:.1%}  '
              f'term={cem["termination_rate"]:.1%}  '
              f'final={cem["mean_final_error"]:.3f}')

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f'[done] {out}')


if __name__ == '__main__':
    main()
