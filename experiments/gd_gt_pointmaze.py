#!/usr/bin/env python
"""Gradient-based planning with GROUND-TRUTH PointMaze dynamics (oracle baseline).

Since MuJoCo is not differentiable, gradients are estimated via SPSA
(Simultaneous Perturbation Stochastic Approximation): at each optimisation
step, two simulator rollouts with ±ε random perturbations give an unbiased
gradient estimate.  The rest of the loop — Adam optimiser, noise injection,
warm-start, multi-restart — is identical to compare_gd_pointmaze.py.

This is the oracle upper bound: same planner structure as GBP, but with
perfect dynamics instead of a learned world model.

Usage
-----
  MUJOCO_GL=egl python experiments/gd_gt_pointmaze.py \\
      --trials 10 --n-steps 200 \\
      --planning-horizon 25 --executed-steps 25 \\
      --gd-steps 50 --lr 0.1 --action-noise 0.05 \\
      --fd-eps 0.1 \\
      --objective last \\
      --frame-skip 5 \\
      --output results/gd_gt_pointmaze.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from envs.pointmaze_visual import PointMazeVisual

FRAMESKIP   = 5
ACTION_SCALE = 1.0
MAZE_MAP     = 'U'
IMAGE_SIZE   = 64


def make_env(seed=0):
    env_cfg = {'environment': {'maze_map': MAZE_MAP, 'image_size': IMAGE_SIZE,
                               'action_scale': ACTION_SCALE}}
    return PointMazeVisual(env_cfg, seed=seed)


def sample_random_state(env, seed):
    obs_dict, _ = env._env.reset(seed=int(seed))
    return obs_dict['observation'].astype(np.float32)


# ── cost function ─────────────────────────────────────────────────────────────

def simulate_cost(env, start_state, u, goal_xy, frame_skip, objective):
    """Roll out action sequence u in the true simulator; return scalar cost."""
    env.reset_to_state(start_state.copy())
    positions = []
    H = u.shape[0]
    for t in range(H):
        a = np.clip(u[t], -ACTION_SCALE, ACTION_SCALE)
        for sub in range(frame_skip):
            if sub < frame_skip - 1:
                state, _, done, _ = env.step_no_render(a)
            else:
                obs, state, _, done, _ = env.step(a)
            if done:
                break
        positions.append(state[:2].copy())
        if done:
            break
    if not positions:
        return float('inf')
    if objective == 'last':
        return float(np.sum((positions[-1] - goal_xy) ** 2))
    else:
        return float(np.mean([np.sum((p - goal_xy) ** 2) for p in positions]))


# ── SPSA gradient + Adam ──────────────────────────────────────────────────────

class AdamState:
    def __init__(self, shape, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.m     = np.zeros(shape, dtype=np.float64)
        self.v     = np.zeros(shape, dtype=np.float64)
        self.t     = 0

    def step(self, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat  = self.m / (1 - self.beta1 ** self.t)
        v_hat  = self.v / (1 - self.beta2 ** self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def spsa_gradient(env, start_state, u, goal_xy, frame_skip, objective, fd_eps):
    """Two-rollout SPSA gradient estimate of the cost w.r.t. u."""
    delta     = np.random.choice([-1., 1.], size=u.shape).astype(np.float64)
    cost_plus  = simulate_cost(env, start_state, u + fd_eps * delta,
                               goal_xy, frame_skip, objective)
    cost_minus = simulate_cost(env, start_state, u - fd_eps * delta,
                               goal_xy, frame_skip, objective)
    return ((cost_plus - cost_minus) / (2.0 * fd_eps)) * delta


# ── GT-GD planner ─────────────────────────────────────────────────────────────

class PointMazeGTGDPlanner:
    """GD planner using SPSA gradients through the ground-truth simulator."""

    def __init__(self, horizon, executed_steps, gd_steps, lr, action_noise,
                 fd_eps, objective, warm_start, n_restarts):
        self.horizon        = int(horizon)
        self.executed_steps = min(int(executed_steps), self.horizon)
        self.gd_steps       = int(gd_steps)
        self.lr             = float(lr)
        self.action_noise   = float(action_noise)
        self.fd_eps         = float(fd_eps)
        self.objective      = objective
        self.warm_start     = warm_start
        self.n_restarts     = max(1, int(n_restarts))
        self._prev_u        = None

    def _run_one(self, env, start_state, init_u, goal_xy, frame_skip):
        """SPSA + Adam from one initialisation; return (u_clamped, final_cost)."""
        u     = init_u.copy().astype(np.float64)
        adam  = AdamState((self.horizon, 2), self.lr)
        for _ in range(self.gd_steps):
            grad = spsa_gradient(env, start_state, u, goal_xy,
                                 frame_skip, self.objective, self.fd_eps)
            u   -= adam.step(grad)
            u   += np.random.randn(*u.shape) * self.action_noise
        u_clamped = np.clip(u, -ACTION_SCALE, ACTION_SCALE)
        cost = simulate_cost(env, start_state, u_clamped,
                             goal_xy, frame_skip, 'last')
        return u_clamped, cost

    def plan(self, env, start_state, goal_xy, frame_skip):
        """Return (executed_steps, 2) action array."""
        inits = []
        if self.warm_start and self._prev_u is not None:
            shifted = np.concatenate([
                self._prev_u[self.executed_steps:],
                np.zeros((self.executed_steps, 2), dtype=np.float64)
            ], axis=0)
            inits.append(shifted)
        for _ in range(self.n_restarts - len(inits)):
            inits.append(np.random.randn(self.horizon, 2))

        best_u, best_cost = None, float('inf')
        for init_u in inits:
            u_clamped, cost = self._run_one(
                env, start_state, init_u, goal_xy, frame_skip)
            if cost < best_cost:
                best_cost = cost
                best_u    = u_clamped

        self._prev_u = best_u
        return best_u[:self.executed_steps]


# ── trial evaluation ──────────────────────────────────────────────────────────

def gd_gt_trial(env, planner, goal_xy, start_state,
                n_steps, success_threshold, success_hold_steps, frame_skip):
    obs, state, _ = env.reset_to_state(start_state.copy())
    states  = [state.copy()]
    step    = 0
    consec  = 0
    planner._prev_u = None

    while step < n_steps:
        sequence = planner.plan(env, state.copy(), goal_xy, frame_skip)
        # After planning, reset env back to current state
        obs, state, _ = env.reset_to_state(state.copy())

        for a in sequence:
            if step >= n_steps:
                break
            a_clipped = np.clip(a, -ACTION_SCALE, ACTION_SCALE)
            done      = False
            for sub in range(frame_skip):
                if sub < frame_skip - 1:
                    state, _, done, _ = env.step_no_render(a_clipped)
                else:
                    obs, state, _, done, _ = env.step(a_clipped)
                if done:
                    break
            states.append(state.copy())
            step += 1

            xy_dist = float(np.linalg.norm(state[:2] - goal_xy))
            if xy_dist < success_threshold:
                consec += 1
                if consec >= success_hold_steps:
                    break
            else:
                consec = 0
            if done:
                break

        if consec >= success_hold_steps:
            break

    states  = np.array(states)
    xy_err  = np.linalg.norm(states[:, :2] - goal_xy[None], axis=1)
    stable  = xy_err < success_threshold
    success = bool(stable[-1])
    tail    = stable[-success_hold_steps:]
    held    = bool(len(tail) == success_hold_steps and tail.all())

    return {
        'success':         success,
        'held':            held,
        'final_error':     float(xy_err[-1]),
        'max_error':       float(np.max(xy_err)),
        'fraction_stable': float(np.mean(stable[1:])) if len(stable) > 1 else 0.,
        'goal_xy':         goal_xy.tolist(),
        'start_xy':        start_state[:2].tolist(),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='GBP oracle: SPSA gradient descent through GT PointMaze dynamics.')
    p.add_argument('--trials',             type=int,   default=10)
    p.add_argument('--n-steps',            type=int,   default=200)
    p.add_argument('--planning-horizon',   type=int,   default=25)
    p.add_argument('--executed-steps',     type=int,   default=25)
    p.add_argument('--gd-steps',           type=int,   default=50)
    p.add_argument('--lr',                 type=float, default=0.1)
    p.add_argument('--action-noise',       type=float, default=0.05)
    p.add_argument('--fd-eps',             type=float, default=0.1,
                   help='SPSA perturbation size for finite-difference gradient')
    p.add_argument('--objective',          choices=['last', 'all'], default='last')
    p.add_argument('--n-restarts',         type=int,   default=1)
    p.add_argument('--no-warm-start',      action='store_true')
    p.add_argument('--success-threshold',  type=float, default=0.5)
    p.add_argument('--success-hold-steps', type=int,   default=1)
    p.add_argument('--frame-skip',         type=int,   default=5)
    p.add_argument('--seed',               type=int,   default=0)
    p.add_argument('--output',             required=True)
    args = p.parse_args()

    env = make_env(seed=args.seed)

    planner = PointMazeGTGDPlanner(
        horizon        = args.planning_horizon,
        executed_steps = args.executed_steps,
        gd_steps       = args.gd_steps,
        lr             = args.lr,
        action_noise   = args.action_noise,
        fd_eps         = args.fd_eps,
        objective      = args.objective,
        warm_start     = not args.no_warm_start,
        n_restarts     = args.n_restarts,
    )
    print(f'[GT-GBP] H={planner.horizon} K={planner.executed_steps} '
          f'gd_steps={planner.gd_steps} lr={planner.lr} '
          f'noise={planner.action_noise} fd_eps={planner.fd_eps} '
          f'objective={planner.objective} restarts={planner.n_restarts} '
          f'frame_skip={args.frame_skip}')
    print(f'[GT-GBP] ~{2 * planner.gd_steps} simulator rollouts per plan call '
          f'(SPSA: 2 per step)')

    trials, n_success = [], 0
    for i in range(args.trials):
        start_seed = args.seed + 99 * i + 1
        goal_seed  = args.seed + 99 * i + 2

        np.random.seed(args.seed + i)
        start_state = sample_random_state(env, start_seed)
        goal_raw    = sample_random_state(env, goal_seed)
        goal_xy     = goal_raw[:2]

        print(f'[trial {i:03d}] start=[{start_state[0]:.2f},{start_state[1]:.2f}]  '
              f'goal=[{goal_xy[0]:.2f},{goal_xy[1]:.2f}]  ', end='', flush=True)

        t0  = time.time()
        row = gd_gt_trial(env, planner, goal_xy, start_state,
                          args.n_steps, args.success_threshold,
                          args.success_hold_steps, args.frame_skip)
        elapsed = time.time() - t0

        n_success += int(row['success'])
        trials.append(row)
        print(f'success={row["success"]}  final={row["final_error"]:.3f}  '
              f't={elapsed:.1f}s')

    sr = n_success / max(len(trials), 1)
    print(f'\n[GT-GBP] success rate: {sr:.1%}  ({n_success}/{len(trials)})')
    env.close()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'protocol': {
            'planner':           'gt_spsa_gd',
            'n_steps':           args.n_steps,
            'planning_horizon':  args.planning_horizon,
            'executed_steps':    args.executed_steps,
            'gd_steps':          args.gd_steps,
            'lr':                args.lr,
            'action_noise':      args.action_noise,
            'fd_eps':            args.fd_eps,
            'objective':         args.objective,
            'n_restarts':        args.n_restarts,
            'warm_start':        not args.no_warm_start,
            'success_threshold': args.success_threshold,
            'seed':              args.seed,
            'n_trials':          args.trials,
        },
        'success_rate': sr,
        'trials':       trials,
    }, indent=2))
    print(f'[done] {out}')


if __name__ == '__main__':
    main()
