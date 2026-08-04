#!/usr/bin/env python
"""Diagnose which decoded state coordinates break exact-dynamics CEM."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
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
    decode, encode_split, fit_ridge_probe,
)


STATE_NAMES = ('x', 'x_dot', 'theta', 'theta_dot')
MODES = ('oracle', 'decoded_all', 'true_x', 'true_x_dot',
         'true_theta', 'true_theta_dot')


def make_planner(cfg, args, terminal_q, device):
    bounds = cfg['environment'].get('action_range', [-10., 10.])
    return GroundTruthCEM(
        cfg, args.cem_horizon, args.cem_samples, args.cem_elites,
        args.cem_iters, args.cem_init_std, args.cem_warm_start_std,
        1.0, terminal_q, bounds[0], bounds[1], device)


def feedback_state(mode, truth, decoded):
    if mode == 'oracle':
        return truth.copy()
    state = decoded.copy()
    if mode.startswith('true_'):
        name = mode[5:]
        state[STATE_NAMES.index(name)] = truth[STATE_NAMES.index(name)]
    return state


def run_trial(cfg, args, mode, model, ridge, decoded_eq, terminal_q,
              x0, trial_seed, device):
    env = make_env(cfg, trial_seed)
    planner = make_planner(cfg, args, terminal_q, device)
    oracle_planner = make_planner(cfg, args, terminal_q, device)
    obs, truth, _ = env.reset_to_state(x0)
    prev_obs = obs
    w, b, state_mean, state_std = ridge
    rows = []
    terminated = False
    for step in range(args.steps):
        z = encode_online(model, obs, prev_obs, device)
        decoded_raw = decode(z[None], w, b, state_mean, state_std)[0]
        decoded = decoded_raw - decoded_eq
        estimate = feedback_state(mode, truth, decoded)

        # Use identical CEM perturbations for the applied and counterfactual
        # oracle actions at this time step.
        torch.manual_seed(trial_seed * 1000 + step)
        rng_state = torch.cuda.get_rng_state(device) if device.type == 'cuda' \
            else torch.get_rng_state()
        action = planner.plan_state(estimate)
        if device.type == 'cuda':
            torch.cuda.set_rng_state(rng_state, device)
        else:
            torch.set_rng_state(rng_state)
        oracle_action = oracle_planner.plan_state(truth)

        rows.append({
            'step': step,
            'true_state': truth.tolist(),
            'decoded_state': decoded.tolist(),
            'feedback_state': estimate.tolist(),
            'state_error': (decoded - truth).tolist(),
            'action': action,
            'oracle_action_same_state': oracle_action,
            'action_error': action - oracle_action,
        })
        old_obs = obs
        obs, truth, _, done, _ = env.step(action)
        prev_obs = old_obs
        if done:
            terminated = True
            break
    env.close()

    true = np.asarray([r['true_state'] for r in rows])
    pred = np.asarray([r['decoded_state'] for r in rows])
    actions = np.asarray([r['action'] for r in rows])
    oracle_actions = np.asarray([r['oracle_action_same_state'] for r in rows])
    final_error = float(np.linalg.norm(truth))
    ctrl = cfg.get('control', {})
    hold = min(int(ctrl.get('success_hold_steps', 10)), len(true))
    state_norms = np.linalg.norm(true, axis=1)
    success = bool(not terminated and hold > 0 and
                   np.all(state_norms[-hold:] <
                          float(ctrl.get('stabilization_threshold', .1))))
    return {
        'initial_state': x0.tolist(),
        'success': success,
        'terminated': terminated,
        'steps': len(rows),
        'final_error': final_error,
        'max_state_norm': float(max(state_norms.max(), final_error)),
        'decoded_rmse': np.sqrt(np.mean((pred - true) ** 2, axis=0)).tolist(),
        'decoded_bias': np.mean(pred - true, axis=0).tolist(),
        'action_mae_vs_oracle': float(np.mean(np.abs(actions - oracle_actions))),
        'action_sign_agreement': float(np.mean(
            np.sign(actions) == np.sign(oracle_actions))),
        'trace': rows,
    }


def summarize(trials):
    return {
        'success_rate': float(np.mean([r['success'] for r in trials])),
        'termination_rate': float(np.mean([r['terminated'] for r in trials])),
        'mean_final_error': float(np.mean([r['final_error'] for r in trials])),
        'mean_steps': float(np.mean([r['steps'] for r in trials])),
        'decoded_rmse': np.mean([r['decoded_rmse'] for r in trials], axis=0).tolist(),
        'action_mae_vs_oracle': float(np.mean(
            [r['action_mae_vs_oracle'] for r in trials])),
        'action_sign_agreement': float(np.mean(
            [r['action_sign_agreement'] for r in trials])),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--probe-train-samples', type=int, default=4000)
    p.add_argument('--probe-batch-size', type=int, default=128)
    p.add_argument('--ridge', type=float, default=1e-3)
    p.add_argument('--trials', type=int, default=3)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--cem-horizon', type=int, default=2)
    p.add_argument('--cem-samples', type=int, default=2048)
    p.add_argument('--cem-elites', type=int, default=128)
    p.add_argument('--cem-iters', type=int, default=8)
    p.add_argument('--cem-init-std', type=float, default=3.)
    p.add_argument('--cem-warm-start-std', type=float, default=1.)
    p.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', required=True)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model, _ = build_model(cfg['model'], checkpoint, device)
    meta = load_discrete_dataset_meta(args.data)
    state_mean = np.asarray(meta.get('state_mean', np.zeros(4)))
    state_std = np.asarray(meta.get('state_std', np.ones(4)))
    print('[probe] fitting frozen ridge decoder ...')
    z_train, s_train = encode_split(
        model, args.data, 'train', args.probe_train_samples,
        args.probe_batch_size, args.seed, device)
    w, b = fit_ridge_probe(
        z_train, s_train, state_mean, state_std, args.ridge)

    eq_env = make_env(cfg, args.seed + 999)
    eq_obs, _, _ = eq_env.reset_to_state(np.zeros(4, dtype=np.float32))
    z_eq = encode_online(model, eq_obs, eq_obs, device)
    decoded_eq = decode(z_eq[None], w, b, state_mean, state_std)[0]
    A, B = physical_macro_jacobian(eq_env)
    _, terminal_q, poles = solve_discrete_lqr(
        A, B, np.diag([1., 1., 10., 1.]), np.array([[.01]]))
    eq_env.close()
    print('[equilibrium] decoded=' + np.array2string(decoded_eq, precision=4))
    print(f'[terminal] rho(A-BK)={np.max(np.abs(poles)):.4f}')

    ctrl = cfg.get('control', {})
    scale = float(ctrl.get('init_scale', .05))
    rng = np.random.RandomState(args.seed)
    initial_states = [rng.uniform(-scale, scale, 4).astype(np.float32)
                      for _ in range(args.trials)]
    result = {'checkpoint': args.checkpoint,
              'state_names': STATE_NAMES,
              'decoded_equilibrium': decoded_eq.tolist(),
              'modes': {}}
    ridge_params = (w, b, state_mean, state_std)
    for mode in args.modes:
        trials = []
        for i, x0 in enumerate(initial_states):
            trials.append(run_trial(
                cfg, args, mode, model, ridge_params, decoded_eq,
                terminal_q, x0, args.seed + i, device))
        row = summarize(trials)
        result['modes'][mode] = {'summary': row, 'trials': trials}
        print(f'[{mode}] success={row["success_rate"]:.1%}  '
              f'term={row["termination_rate"]:.1%}  '
              f'final={row["mean_final_error"]:.3f}  '
              f'action_MAE={row["action_mae_vs_oracle"]:.3f}  '
              f'sign={row["action_sign_agreement"]:.1%}  '
              f'RMSE={np.round(row["decoded_rmse"], 4).tolist()}')

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f'[done] {out}')


if __name__ == '__main__':
    main()
