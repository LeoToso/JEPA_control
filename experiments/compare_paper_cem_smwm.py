#!/usr/bin/env python
"""Compare paper-style latent goal CEM across multiple SMWM checkpoints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from sensorimotor_probe_utils import make_env
from cartpole_cem_utils import (PaperLearnedCEM, encode_online, evaluate_trial,
                                summarize, make_frame_buffer)
from models.sensorimotor_world_model import SensorimotorWorldModel


def checkpoint_label(checkpoint):
    path = Path(checkpoint)
    run_name = (path.parent.parent.name
                if path.parent.name == 'checkpoints' else path.parent.name)
    r = run_name.lower()
    if 'sigreg_ms' in r or 'sigreg-ms' in r:
        return 'SIGReg MS'
    if 'sigreg' in r:
        return 'SIGReg'
    if '_sr_' in r or r.endswith('_sr'):
        return 'State recon.'
    if 'ar_ms' in r or 'ar-ms' in r:
        return 'AR MS'
    if 'dinov2' in r:
        return 'DINOv2'
    if 'proprio' in r:
        return 'Action recon.'
    return run_name


def load_model(checkpoint_path, config_path, device):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False)
    model_cfg = dict(checkpoint['model_config'])
    model = SensorimotorWorldModel(model_cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    state_mean = np.asarray(checkpoint['state_mean'], dtype=np.float32)
    state_std = np.asarray(checkpoint['state_std'], dtype=np.float32)
    action_scale = float(checkpoint.get('action_scale', 10.))
    return cfg, checkpoint, model, state_mean, state_std, action_scale


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpts', nargs='+', required=True)
    p.add_argument('--cfgs', nargs='+', required=True)
    p.add_argument('--trials', type=int, default=100)
    p.add_argument('--primitive-budget', type=int, default=50)
    p.add_argument('--planning-horizon', type=int, default=5)
    p.add_argument('--executed-steps', type=int, default=5)
    p.add_argument('--cem-population', type=int, default=300)
    p.add_argument('--cem-elites', type=int, default=30)
    p.add_argument('--cem-iters', type=int, default=30)
    p.add_argument('--cem-initial-variance', type=float, default=1.)
    p.add_argument('--success-threshold', type=float, default=None,
                   help='Physical-state norm for success; '
                        'defaults to control.stabilization_threshold in cfg')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if len(args.cfgs) == 1:
        args.cfgs *= len(args.ckpts)
    if len(args.cfgs) != len(args.ckpts):
        raise ValueError('--cfgs must have length 1 or match --ckpts')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Shared trial setup — use first config for defaults.
    with open(args.cfgs[0]) as f:
        ref_cfg = yaml.safe_load(f)
    ctrl = ref_cfg.get('control', {})
    success_threshold = (float(args.success_threshold)
                         if args.success_threshold is not None
                         else float(ctrl.get('stabilization_threshold', .1)))
    init_scale = float(ctrl.get('init_scale', .05))
    success_hold_steps = int(ctrl.get('success_hold_steps', 10))

    # Generate all initial states once so every model faces the same trials.
    rng = np.random.RandomState(args.seed)
    initial_states = [
        rng.uniform(-init_scale, init_scale, 4).astype(np.float32)
        for _ in range(args.trials)]

    all_results = []

    for ckpt, cfg_path in zip(args.ckpts, args.cfgs):
        label = checkpoint_label(ckpt)
        print(f'\n[{label}] loading {ckpt}')
        cfg, checkpoint, model, state_mean, state_std, action_scale = \
            load_model(ckpt, cfg_path, device)

        if 'environment' not in cfg:
            model_cfg = dict(checkpoint['model_config'])
            cfg['environment'] = {
                'frame_skip': int(model_cfg.get('frame_skip', 5)),
                'image_size': int(model_cfg.get('image_size', 128)),
                'action_range': [-10, 10],
                'mass_cart': 1.0, 'mass_pole': 0.1,
                'pole_length': 0.5, 'gravity': 9.8, 'dt': 0.02,
            }

        bounds = cfg['environment'].get('action_range', [-10., 10.])
        planner = PaperLearnedCEM(
            model, action_scale,
            args.planning_horizon, args.executed_steps,
            args.cem_population, args.cem_elites, args.cem_iters,
            args.cem_initial_variance, bounds[0], bounds[1], device)
        frame_skip = int(cfg['environment'].get('frame_skip', 1))

        goal_state = np.zeros(4, dtype=np.float32)
        goal_env = make_env(cfg, args.seed + 999)
        goal_obs, _, _ = goal_env.reset_to_state(goal_state)
        goal_buf = make_frame_buffer(model, goal_obs)
        z_goal = encode_online(
            model, goal_buf, goal_obs, goal_state,
            state_mean, state_std, device)
        goal_env.close()

        print(f'[{label}] epoch={checkpoint.get("epoch", "?")} '
              f'H={planner.horizon} K={planner.executed_steps} '
              f'frame_skip={frame_skip} '
              f'success_threshold={success_threshold:g}')

        rows = []
        for trial, x0 in enumerate(initial_states):
            torch.manual_seed(args.seed + trial)
            env = make_env(cfg, args.seed + trial)
            row = evaluate_trial(
                env, planner, model, x0, goal_state, z_goal,
                state_mean, state_std, args.primitive_budget,
                success_threshold, success_threshold,
                success_hold_steps, device)
            env.close()
            rows.append(row)
            print(f'[{label}] trial {trial:03d}  '
                  f'success={row["success"]}  '
                  f'held={row["held_stable"]}  '
                  f'term={row["terminated"]}  '
                  f'final={row["final_error"]:.5f}  '
                  f'zgoal={row["latent_goal_errors_at_replans"][-1]:.4f}')

        summary = summarize(rows)
        print(f'[{label}] SUMMARY  '
              f'success={summary["success_rate"]:.1%}  '
              f'held={summary["held_stable_rate"]:.1%}  '
              f'term={summary["termination_rate"]:.1%}  '
              f'final={summary["mean_final_error"]:.5f}  '
              f'Lbar={summary["mean_stabilization_macro_steps"]:.2f}  '
              f'invL2={summary["inverse_squared_mean_stabilization_macro_length"]}')
        all_results.append({
            'label': label,
            'checkpoint': ckpt,
            'summary': summary,
            'trials': rows,
        })

    # Comparison table.
    print('\n' + '=' * 70)
    print(f'{"Model":<28} {"Success":>8} {"Held":>8} '
          f'{"Term":>6} {"FinalErr":>10} {"Lbar":>8}')
    print('-' * 70)
    for r in all_results:
        s = r['summary']
        print(f'{r["label"]:<28} '
              f'{s["success_rate"]:>8.1%} '
              f'{s["held_stable_rate"]:>8.1%} '
              f'{s["termination_rate"]:>6.1%} '
              f'{s["mean_final_error"]:>10.5f} '
              f'{s["mean_stabilization_macro_steps"]:>8.2f}')
    print('=' * 70)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'protocol': {
            'planning_horizon': args.planning_horizon,
            'executed_steps': args.executed_steps,
            'cem_population': args.cem_population,
            'cem_elites': args.cem_elites,
            'cem_iters': args.cem_iters,
            'cem_initial_variance': args.cem_initial_variance,
            'primitive_budget': args.primitive_budget,
            'success_threshold': success_threshold,
            'seed': args.seed,
            'n_trials': args.trials,
        },
        'models': all_results,
    }, indent=2))
    print(f'[done] {out}')


if __name__ == '__main__':
    main()
