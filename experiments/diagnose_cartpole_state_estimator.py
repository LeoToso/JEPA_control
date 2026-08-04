"""Diagnose near-equilibrium state-decoding errors that corrupt LQR actions.

Trajectories are driven by the oracle state-feedback controller.  At every
visited state we also compute, without applying it, the action that the same
LQR gain would choose from the decoded visual state.  The two controllers are
therefore compared on exactly the same stable trajectories.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import yaml

from control.lqr import solve_discrete_lqr
from data.dataset import load_discrete_dataset_meta
from experiments.evaluate_cartpole_control import build_model, make_env


STATE_NAMES = ('x', 'x_dot', 'theta', 'theta_dot')


def physical_jacobian(env, eps_x=1e-4, eps_u=1e-3):
    """Finite-difference the true macro-step dynamics at upright."""
    A = np.zeros((4, 4), dtype=np.float64)
    B = np.zeros((4, 1), dtype=np.float64)
    for j in range(4):
        dx = np.zeros(4); dx[j] = eps_x
        env.reset_to_state(dx); _, xp, _, _, _ = env.step(0.0)
        env.reset_to_state(-dx); _, xm, _, _, _ = env.step(0.0)
        A[:, j] = (xp - xm) / (2.0 * eps_x)
    env.reset_to_state(np.zeros(4)); _, xp, _, _, _ = env.step(eps_u)
    env.reset_to_state(np.zeros(4)); _, xm, _, _, _ = env.step(-eps_u)
    B[:, 0] = (xp - xm) / (2.0 * eps_u)
    return A, B


def to_tensor(obs, device):
    return (torch.from_numpy(obs).float().permute(2, 0, 1)
            .unsqueeze(0).to(device) / 255.0)


def summarize(mask, true_states, pred_states, u_oracle, u_decoded, K):
    n = int(mask.sum())
    if n == 0:
        return {'n': 0}
    truth = true_states[mask]
    pred = pred_states[mask]
    err = pred - truth
    uo, ud = u_oracle[mask], u_decoded[mask]
    active = (np.abs(uo) > 1e-3) | (np.abs(ud) > 1e-3)
    sign_agreement = float(np.mean(np.sign(uo[active]) == np.sign(ud[active]))) \
        if np.any(active) else float('nan')
    corr = float(np.corrcoef(uo, ud)[0, 1]) \
        if n > 1 and np.std(uo) > 1e-12 and np.std(ud) > 1e-12 else float('nan')
    # Since u_dec-u_oracle = -K(pred-true), these values show which decoded
    # coordinate contributes most strongly to action disagreement.
    action_error_contrib = np.mean(np.abs(err * K.reshape(1, 4)), axis=0)
    return {
        'n': n,
        'rmse': np.sqrt(np.mean(err ** 2, axis=0)).tolist(),
        'bias': np.mean(err, axis=0).tolist(),
        'mae': np.mean(np.abs(err), axis=0).tolist(),
        'action_mae': float(np.mean(np.abs(ud - uo))),
        'action_bias': float(np.mean(ud - uo)),
        'action_correlation': corr,
        'action_sign_agreement': sign_agreement,
        'decoded_saturation_rate': float(np.mean(np.abs(ud) >= 9.9)),
        'mean_abs_action_contribution': action_error_contrib.tolist(),
    }


def fmt_vec(v):
    return '[' + ', '.join(f'{x:.4f}' for x in v) + ']'


def print_summary(label, row):
    if row['n'] == 0:
        print(f'[{label}] n=0')
        return
    print(f'[{label}] n={row["n"]}  RMSE={fmt_vec(row["rmse"])}  '
          f'bias={fmt_vec(row["bias"])}')
    print(f'  action: MAE={row["action_mae"]:.4f}  bias={row["action_bias"]:+.4f}  '
          f'corr={row["action_correlation"]:+.3f}  '
          f'sign={row["action_sign_agreement"]:.1%}  '
          f'sat={row["decoded_saturation_rate"]:.1%}')
    print(f'  mean |K_i error_i|: {fmt_vec(row["mean_abs_action_contribution"])} '
          f'for {STATE_NAMES}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--trials', type=int, default=50)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', default=None)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'state_head_state' not in checkpoint:
        raise RuntimeError('Checkpoint does not contain state_head_state')
    model, arch = build_model(cfg['model'], checkpoint, device)
    head = nn.Linear(model.latent_dim, 4).to(device)
    head.load_state_dict(checkpoint['state_head_state'])
    head.eval()

    meta = load_discrete_dataset_meta(args.data)
    state_mean = np.asarray(meta.get('state_mean', np.zeros(4)), dtype=np.float64)
    state_std = np.asarray(meta.get('state_std', np.ones(4)), dtype=np.float64)
    env = make_env(cfg, args.seed)
    A, B = physical_jacobian(env)
    K, _, poles = solve_discrete_lqr(
        A, B, np.diag([1., 1., 10., 1.]), np.array([[.01]]))
    print(f'[model] checkpoint={Path(args.checkpoint).name}  '
          f'rho(A-BK)={np.max(np.abs(poles)):.4f}  K={fmt_vec(K.ravel())}')

    eq_obs, _, _ = env.reset_to_state(np.zeros(4, dtype=np.float32))
    eq_t = to_tensor(eq_obs, device)
    with torch.no_grad():
        eq_norm = head(model.encode_obs(eq_t, eq_t)).cpu().numpy()[0]
    decoded_eq = eq_norm * state_std + state_mean
    decoded_eq[2] = np.arctan2(np.sin(decoded_eq[2]), np.cos(decoded_eq[2]))
    print(f'[equilibrium decoder] D(z*)={fmt_vec(decoded_eq)}')

    rng = np.random.RandomState(args.seed)
    bounds = cfg['environment'].get('action_range', [-10., 10.])
    # Reprint the equilibrium force now that controller bounds are available.
    print(f'[equilibrium action] raw={float(np.clip((-K @ decoded_eq).item(), bounds[0], bounds[1])):+.4f}  '
          f'centered=+0.0000')
    true_rows, raw_pred_rows, pred_rows = [], [], []
    uo_rows, raw_ud_rows, ud_rows, time_rows = [], [], [], []
    init_scale = float(cfg.get('control', {}).get('init_scale', .05))

    for _ in range(args.trials):
        obs, state, _ = env.reset_to_state(
            rng.uniform(-init_scale, init_scale, 4).astype(np.float32))
        prev_obs = obs
        for t in range(args.steps):
            obs_t, prev_t = to_tensor(obs, device), to_tensor(prev_obs, device)
            with torch.no_grad():
                z = model.encode_obs(obs_t, prev_t)
                pred_norm = head(z).cpu().numpy()[0]
            pred_state = pred_norm * state_std + state_mean
            # Keep theta on the same principal branch used during training.
            pred_state[2] = np.arctan2(np.sin(pred_state[2]), np.cos(pred_state[2]))
            centered_state = pred_state - decoded_eq
            u_oracle = float(np.clip((-K @ state).item(), bounds[0], bounds[1]))
            raw_u_decoded = float(np.clip((-K @ pred_state).item(), bounds[0], bounds[1]))
            u_decoded = float(np.clip((-K @ centered_state).item(), bounds[0], bounds[1]))

            true_rows.append(state.copy()); raw_pred_rows.append(pred_state.copy())
            pred_rows.append(centered_state.copy()); uo_rows.append(u_oracle)
            raw_ud_rows.append(raw_u_decoded); ud_rows.append(u_decoded); time_rows.append(t)

            old_obs = obs
            obs, state, _, done, _ = env.step(u_oracle)
            prev_obs = old_obs
            if done:
                break
    env.close()

    truth = np.asarray(true_rows); raw_pred = np.asarray(raw_pred_rows)
    pred = np.asarray(pred_rows); uo = np.asarray(uo_rows)
    raw_ud = np.asarray(raw_ud_rows); ud = np.asarray(ud_rows); times = np.asarray(time_rows)
    abs_theta = np.abs(truth[:, 2])
    masks = {
        'all': np.ones(len(truth), dtype=bool),
        'first_frame': times == 0,
        'after_first': times > 0,
        'theta_le_0.02': abs_theta <= .02,
        'theta_0.02_to_0.05': (abs_theta > .02) & (abs_theta <= .05),
        'theta_0.05_to_0.10': (abs_theta > .05) & (abs_theta <= .10),
    }
    result = {
        'checkpoint': args.checkpoint,
        'state_names': STATE_NAMES,
        'K': K.ravel().tolist(),
        'decoded_equilibrium': decoded_eq.tolist(),
        'rho_closed_loop': float(np.max(np.abs(poles))),
        'groups': {},
    }
    raw_all = summarize(masks['all'], truth, raw_pred, uo, raw_ud, K)
    result['raw_uncentered_all'] = raw_all
    print_summary('raw_uncentered_all', raw_all)
    print('[centered decoder: D(z)-D(z*)]')
    for label, mask in masks.items():
        row = summarize(mask, truth, pred, uo, ud, K)
        result['groups'][label] = row
        print_summary(label, row)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f'[done] {out}')


if __name__ == '__main__':
    main()
