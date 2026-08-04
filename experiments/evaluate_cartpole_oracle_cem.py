#!/usr/bin/env python
"""Standalone oracle-state, exact-dynamics CEM validation for CartPole."""
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
from experiments.evaluate_cartpole_control import make_env, physical_macro_jacobian
from experiments.evaluate_cartpole_encoder_gt_cem import GroundTruthCEM


def evaluate_trial(env, planner, x0, steps, stabilization_threshold,
                   settling_threshold, success_hold_steps, failure_penalty):
    planner.reset()
    _, state, _ = env.reset_to_state(x0)
    states, actions = [], []
    terminated = False
    for _ in range(steps):
        states.append(state.copy())
        action = planner.plan_state(state)
        actions.append(action)
        _, state, _, done, _ = env.step(action)
        if done:
            terminated = True
            break
    states = np.asarray(states)
    actions = np.asarray(actions)
    errors = np.concatenate((np.linalg.norm(states, axis=1),
                             [float(np.linalg.norm(state))]))
    hold = min(max(int(success_hold_steps), 1), len(errors))
    success = bool(not terminated and
                   np.all(errors[-hold:] < stabilization_threshold))
    Q = np.diag([1., 1., 10., 1.])
    cost = sum(float(x @ Q @ x + .01 * u * u)
               for x, u in zip(states, actions))
    if terminated:
        cost += failure_penalty
    return {
        'initial_state': x0.tolist(),
        'success': success,
        'terminated': terminated,
        'steps': int(len(actions)),
        'final_error': float(errors[-1]),
        'max_error': float(errors.max()),
        'fraction_stable': float(np.mean(errors < settling_threshold)),
        'cost': float(cost),
        'action_rms': float(np.sqrt(np.mean(actions ** 2))),
        'action_max_abs': float(np.max(np.abs(actions))),
        'first_actions': actions[:10].tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--trials', type=int, default=10)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--cem-horizon', type=int, default=2)
    p.add_argument('--cem-samples', type=int, default=2048)
    p.add_argument('--cem-elites', type=int, default=128)
    p.add_argument('--cem-iters', type=int, default=8)
    p.add_argument('--cem-init-std', type=float, default=3.0)
    p.add_argument('--cem-warm-start-std', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', required=True)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env = make_env(cfg, args.seed)
    A, B = physical_macro_jacobian(env)
    K, P, poles = solve_discrete_lqr(
        A, B, np.diag([1., 1., 10., 1.]), np.array([[.01]]))
    print(f'[oracle] rho(A)={np.max(np.abs(np.linalg.eigvals(A))):.4f}  '
          f'rho(A-BK)={np.max(np.abs(poles)):.4f}')

    # Verify that the planner's vectorized dynamics are the actual environment.
    bounds = cfg['environment'].get('action_range', [-10., 10.])
    planner = GroundTruthCEM(
        cfg, args.cem_horizon, args.cem_samples, args.cem_elites,
        args.cem_iters, args.cem_init_std, args.cem_warm_start_std,
        1.0, P, bounds[0], bounds[1], device)
    rng = np.random.RandomState(args.seed)
    consistency = []
    for _ in range(5):
        x = rng.uniform(-.05, .05, 4).astype(np.float32)
        u = float(rng.uniform(-3., 3.))
        env.reset_to_state(x)
        _, x_env, _, _, _ = env.step(u)
        with torch.no_grad():
            x_model = planner._macro_step(
                torch.tensor(x, device=device)[None],
                torch.tensor([u], device=device))[0].cpu().numpy()
        consistency.append(float(np.max(np.abs(x_env - x_model))))
    print(f'[dynamics] max one-step discrepancy={max(consistency):.3e}')

    ctrl = cfg.get('control', {})
    init_scale = float(ctrl.get('init_scale', .05))
    # Restart the RNG so initial states match all other control evaluations.
    rng = np.random.RandomState(args.seed)
    rows = []
    for trial in range(args.trials):
        x0 = rng.uniform(-init_scale, init_scale, 4).astype(np.float32)
        torch.manual_seed(args.seed + trial)
        row = evaluate_trial(
            env, planner, x0, args.steps,
            float(ctrl.get('stabilization_threshold', .1)),
            float(ctrl.get('settling_threshold', .05)),
            int(ctrl.get('success_hold_steps', 10)),
            float(ctrl.get('failure_penalty', 1e4)))
        rows.append(row)
        print(f'[trial {trial:02d}] success={row["success"]}  '
              f'terminated={row["terminated"]}  '
              f'final={row["final_error"]:.4f}  '
              f'max={row["max_error"]:.4f}  cost={row["cost"]:.4f}  '
              f'u_rms={row["action_rms"]:.3f}')
    env.close()

    summary = {
        'success_rate': float(np.mean([r['success'] for r in rows])),
        'termination_rate': float(np.mean([r['terminated'] for r in rows])),
        'mean_final_error': float(np.mean([r['final_error'] for r in rows])),
        'mean_fraction_stable': float(np.mean([r['fraction_stable'] for r in rows])),
        'mean_cost': float(np.mean([r['cost'] for r in rows])),
        'n_trials': len(rows),
    }
    print('[oracle_gt_cem] '
          f'success={summary["success_rate"]:.1%}  '
          f'terminated={summary["termination_rate"]:.1%}  '
          f'final={summary["mean_final_error"]:.4f}  '
          f'stable={summary["mean_fraction_stable"]:.1%}  '
          f'cost={summary["mean_cost"]:.4f}')
    result = {
        'config': args.config,
        'cem': vars(args),
        'dynamics_max_discrepancy': max(consistency),
        'lqr_gain': K.tolist(),
        'closed_loop_rho': float(np.max(np.abs(poles))),
        'summary': summary,
        'trials': rows,
    }
    # vars(args) contains only JSON-compatible scalars and strings.
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f'[done] {out}')


if __name__ == '__main__':
    main()
