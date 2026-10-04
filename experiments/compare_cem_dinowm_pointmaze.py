#!/usr/bin/env python
"""CEM evaluation for PointMaze using the DINO-WM evaluation protocol.

Matches arXiv 2411.04983 exactly:
  - Per-trial independent random start AND goal (both from valid maze positions)
  - CEM horizon = executed steps = 5  (re-plans every 5 steps, no limit)
  - Population=300, elites=30, iterations=10, initial variance=1.0
  - Success threshold = 0.5 maze units (same as D4RL sparse reward)
  - 50 trials, seeds: start=99*n+1, goal=99*n+2 (per-trial, independent)

Key differences from compare_cem_pointmaze.py:
  - Goal is a randomly sampled maze position, not the fixed desired_goal
  - Horizon=5 (not 20), so replanning happens every 5 steps
  - Unlimited replanning within the n_steps budget

Usage
-----
  MUJOCO_GL=egl python experiments/compare_cem_dinowm_pointmaze.py \\
      --ckpts /path/to/ckpt1.pt /path/to/ckpt2.pt \\
      --cfgs  configs/config1.yaml configs/config2.yaml \\
      --trials 50 --n-steps 200 \\
      --half \\
      --output results/cem_dinowm_pointmaze.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from experiments.probe_utils import encode_obs, load_bundle
from envs.pointmaze_visual import PointMazeVisual


# ── helpers ───────────────────────────────────────────────────────────────────

def checkpoint_label(checkpoint):
    path = Path(checkpoint)
    run_name = (path.parent.parent.name
                if path.parent.name == 'checkpoints' else path.parent.name)
    r = run_name.lower()
    if 'sigreg' in r and 'rollout' in r: return 'SIGreg+Rollout'
    if 'ar_1step_ms' in r or 'ar-1step-ms' in r: return 'AR 1step+MS'
    if 'sigreg' in r: return 'SIGreg'
    return run_name


def make_env(bundle, seed):
    env_cfg = bundle['env_cfg']
    if 'environment' not in env_cfg:
        env_cfg['environment'] = {
            'maze_map':    bundle['model_cfg'].get('maze_map', 'U'),
            'image_size':  int(bundle['model_cfg'].get('image_size', 64)),
            'action_scale': 1.0,
        }
    return PointMazeVisual(env_cfg, seed=seed)


def sample_random_state(env, seed):
    """Return a random valid state [x, y, vx, vy] from env reset."""
    obs_dict, _ = env._env.reset(seed=int(seed))
    return obs_dict['observation'].astype(np.float32)  # [x, y, vx, vy]


def encode_state_as_goal(bundle, env, state):
    """Render an observation at `state` (zero velocity) and encode it."""
    goal_state = np.array([state[0], state[1], 0., 0.], dtype=np.float32)
    obs, actual_state, _ = env.reset_to_state(goal_state)
    z = encode_obs(bundle, obs, obs, actual_state)
    return z, goal_state


# ── CEM planner (identical to compare_cem_pointmaze.py) ───────────────────────

class PointMazeCEM:
    """CEM planner with terminal latent goal cost for 2D action spaces."""

    def __init__(self, model, action_scale, horizon, executed_steps,
                 population, elites, iterations, initial_variance_scale,
                 action_lb, action_ub, device, use_half=False):
        self.model = model
        self.action_scale = float(action_scale)
        self.horizon = int(horizon)
        self.executed_steps = min(int(executed_steps), self.horizon)
        self.population = int(population)
        self.elites = min(int(elites), self.population)
        self.iterations = int(iterations)
        self.initial_variance_scale = float(initial_variance_scale)
        self.action_lb = float(action_lb)
        self.action_ub = float(action_ub)
        self.device = device
        self.use_half = use_half and (device.type == 'cuda')

        ctx = self.model.action_context_dim
        self._raw_d = 2
        self._k = ctx // self._raw_d

    @torch.no_grad()
    def _terminal_goal_cost(self, z0, z_goal, actions, prev_hist=None):
        pop = actions.shape[0]
        ctx = self.model.action_context_dim
        k, raw_d = self._k, self._raw_d

        z = z0.expand(pop, -1)

        if self.model.use_action_history and k > 1:
            if prev_hist is not None:
                hist = prev_hist.unsqueeze(0).expand(pop, -1, -1).clone()
            else:
                hist = z0.new_zeros(pop, k - 1, raw_d)
        else:
            hist = None

        for t in range(self.horizon):
            a_t = actions[:, t, :] / self.action_scale

            if hist is not None:
                window = torch.cat([hist, a_t.unsqueeze(1)], dim=1)
                a_ctx = window.reshape(pop, ctx).unsqueeze(1)
                hist = torch.cat([hist[:, 1:, :], a_t.unsqueeze(1)], dim=1)
            else:
                a_ctx = self.model.expand_action(a_t).unsqueeze(1)

            z = self.model.predict(z[:, None], a_ctx)[:, 0]

        return (z - z_goal).square().sum(-1)

    @torch.no_grad()
    def plan_sequence(self, z0, z_goal, prev_hist=None):
        """Return (executed_steps, 2) action array in raw action units."""
        mean     = torch.zeros(self.horizon, 2, device=self.device)
        variance = torch.full((self.horizon, 2), self.initial_variance_scale,
                              device=self.device)
        with torch.autocast(device_type='cuda', enabled=self.use_half):
            for _ in range(self.iterations):
                noise   = torch.randn(self.population, self.horizon, 2,
                                      device=self.device)
                actions = mean.unsqueeze(0) + variance.sqrt().unsqueeze(0) * noise
                actions.clamp_(self.action_lb, self.action_ub)
                costs  = self._terminal_goal_cost(z0, z_goal, actions, prev_hist)
                elite  = actions[torch.argsort(costs)[:self.elites]]
                mean     = elite.mean(0)
                variance = elite.var(0, unbiased=False).clamp(min=1e-6)
        return mean[:self.executed_steps].clamp(
            self.action_lb, self.action_ub).cpu().numpy()


# ── trial evaluation ──────────────────────────────────────────────────────────

def cem_trial(bundle, env, planner, z_goal, goal_xy,
              start_state, n_steps, success_threshold, success_hold_steps,
              frame_skip: int = 1):
    """Run one trial starting from start_state, trying to reach goal_xy / z_goal.

    frame_skip: execute each planned action this many times in the env.
    Set to 5 to match DINO-WM's dynamics depth (each action = 5 mj_steps).
    """
    obs, state, _ = env.reset_to_state(start_state)
    prev_obs = obs.copy()
    states = [state.copy()]

    ctx   = planner.model.action_context_dim
    raw_d = 2
    k     = ctx // raw_d

    if planner.model.use_action_history and k > 1:
        action_hist = torch.zeros(k - 1, raw_d, device=planner.device)
    else:
        action_hist = None

    step          = 0
    consec_stable = 0

    while step < n_steps:
        with torch.no_grad(), torch.autocast(device_type='cuda',
                                              enabled=planner.use_half):
            z = encode_obs(bundle, obs, prev_obs, state)

        latent_dist = float(torch.linalg.vector_norm(z - z_goal))
        sequence = planner.plan_sequence(z, z_goal, action_hist)

        for a in sequence:
            if step >= n_steps:
                break
            a_clipped = np.clip(a, -bundle['action_scale'], bundle['action_scale'])
            prev_obs = obs.copy()
            # Execute frame_skip sub-steps; only render on the last one
            done = False
            for sub in range(frame_skip):
                if sub < frame_skip - 1:
                    state, _, done, _ = env.step_no_render(a_clipped)
                else:
                    obs, state, _, done, _ = env.step(a_clipped)
                if done:
                    break
            states.append(state.copy())
            step += 1

            if action_hist is not None:
                a_norm = torch.as_tensor(
                    a_clipped / bundle['action_scale'],
                    dtype=torch.float32, device=planner.device)
                action_hist = torch.cat(
                    [action_hist[1:], a_norm.unsqueeze(0)], dim=0)

            xy_dist = float(np.linalg.norm(state[:2] - goal_xy))
            if xy_dist < success_threshold:
                consec_stable += 1
                if consec_stable >= success_hold_steps:
                    break
            else:
                consec_stable = 0

            if done:
                break

        if consec_stable >= success_hold_steps:
            break

    with torch.no_grad(), torch.autocast(device_type='cuda',
                                          enabled=planner.use_half):
        z_final = encode_obs(bundle, obs, prev_obs, state)

    states  = np.array(states)
    xy_err  = np.linalg.norm(states[:, :2] - goal_xy[None], axis=1)
    stable  = xy_err < success_threshold

    success = bool(stable[-1])
    tail    = stable[-success_hold_steps:]
    held    = bool(len(tail) == success_hold_steps and tail.all())

    return {
        'success':            success,
        'held':               held,
        'final_error':        float(xy_err[-1]),
        'max_error':          float(np.max(xy_err)),
        'fraction_stable':    float(np.mean(stable[1:])) if len(stable) > 1 else 0.,
        'final_latent_error': float(torch.linalg.vector_norm(z_final - z_goal)),
        'goal_xy':            goal_xy.tolist(),
        'start_xy':           start_state[:2].tolist(),
        'states':             states.tolist(),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='CEM eval matching the DINO-WM protocol (random start+goal per trial).')
    p.add_argument('--ckpts', nargs='+', required=True)
    p.add_argument('--cfgs',  nargs='+', required=True)
    p.add_argument('--trials', type=int, default=50)
    p.add_argument('--n-steps', type=int, default=200,
                   help='Max env steps per trial (DINO-WM uses effectively unlimited; '
                        '200 is generous for a 5-step horizon with replanning)')
    p.add_argument('--planning-horizon', type=int, default=5,
                   help='CEM horizon (DINO-WM: 5)')
    p.add_argument('--executed-steps', type=int, default=5,
                   help='Steps executed before replan (DINO-WM: 5 = full plan)')
    p.add_argument('--cem-population', type=int, default=300,
                   help='CEM population (DINO-WM: 300)')
    p.add_argument('--cem-elites', type=int, default=30,
                   help='CEM top-k elites (DINO-WM: 30)')
    p.add_argument('--cem-iters', type=int, default=10,
                   help='CEM optimisation iterations (DINO-WM: 10)')
    p.add_argument('--cem-initial-variance', type=float, default=1.0)
    p.add_argument('--success-threshold', type=float, default=0.5,
                   help='XY distance threshold (DINO-WM: 0.5)')
    p.add_argument('--success-hold-steps', type=int, default=1,
                   help='Consecutive steps within threshold to count as success '
                        '(DINO-WM checks final step only → 1)')
    p.add_argument('--seed', type=int, default=0,
                   help='Base seed; trial n uses start seed 99*n+1, goal seed 99*n+2')
    p.add_argument('--frame-skip', type=int, default=1,
                   help='Execute each planned action this many times in the env. '
                        'Set 5 to match DINO-WM dynamics depth (each action = 5 mj_steps). '
                        'Use the same value as the oracle for a fair comparison.')
    p.add_argument('--device', default='cuda')
    p.add_argument('--half', action='store_true',
                   help='Use FP16 autocast for model inference')
    p.add_argument('--output', required=True)
    args = p.parse_args()

    if len(args.cfgs) == 1:
        args.cfgs = args.cfgs * len(args.ckpts)

    results = {'protocol': vars(args), 'models': []}

    for ckpt, cfg in zip(args.ckpts, args.cfgs):
        label = checkpoint_label(ckpt)
        print(f'\n[{label}] loading {ckpt}')
        bundle = load_bundle(ckpt, cfg, args.device)

        env = make_env(bundle, seed=0)

        planner = PointMazeCEM(
            model=bundle['model'],
            action_scale=bundle['action_scale'],
            horizon=args.planning_horizon,
            executed_steps=args.executed_steps,
            population=args.cem_population,
            elites=args.cem_elites,
            iterations=args.cem_iters,
            initial_variance_scale=args.cem_initial_variance,
            action_lb=-bundle['action_scale'],
            action_ub= bundle['action_scale'],
            device=bundle['device'],
            use_half=args.half,
        )
        print(f'[{label}] H={planner.horizon} K={planner.executed_steps} '
              f'pop={planner.population} elites={planner.elites} '
              f'iters={planner.iterations} half={planner.use_half} '
              f'frame_skip={args.frame_skip}')

        trials, n_success = [], 0
        for i in range(args.trials):
            # DINO-WM seeds: start=99*n+1, goal=99*n+2
            start_seed = args.seed + 99 * i + 1
            goal_seed  = args.seed + 99 * i + 2

            start_state = sample_random_state(env, start_seed)
            goal_raw    = sample_random_state(env, goal_seed)
            goal_xy     = goal_raw[:2]

            z_goal, goal_state = encode_state_as_goal(bundle, env, goal_raw)

            row = cem_trial(bundle, env, planner, z_goal, goal_xy,
                            start_state, args.n_steps,
                            args.success_threshold, args.success_hold_steps,
                            frame_skip=args.frame_skip)
            n_success += int(row['success'])
            print(f'[{label}] trial {i:03d}  success={row["success"]}  '
                  f'err={row["final_error"]:.4f}  '
                  f'start={np.round(start_state[:2], 2)}  '
                  f'goal={np.round(goal_xy, 2)}')
            trials.append(row)

        sr = n_success / max(len(trials), 1)
        print(f'[{label}] success rate: {sr:.1%}')
        env.close()

        results['models'].append({
            'label':        label,
            'success_rate': sr,
            'trials':       trials,
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n[done] {out}')


if __name__ == '__main__':
    main()
