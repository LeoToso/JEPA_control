#!/usr/bin/env python
"""Oracle CEM on gymnasium-robotics PointMaze — fair upper bound for compare_cem_dinowm_pointmaze.py.

Uses the SAME environment (PointMaze_UMaze-v3), SAME seed scheme, and SAME
start/goal positions as compare_cem_dinowm_pointmaze.py, so results are a
direct, fair oracle ceiling for model-based CEM results.

Physics: raw mj_step into gymnasium's MuJoCo model (no rendering) — identical
dynamics to what the model CEM executes in, but without world-model error.

Usage
-----
  MUJOCO_GL=osmesa python experiments/gt_cem_gymnasium_pointmaze.py \\
      --trials 50 --n-steps 200 \\
      --planning-horizon 5 --executed-steps 5 \\
      --cem-population 300 --cem-elites 30 --cem-iters 10 \\
      --output results/gt_oracle_cem_gymnasium.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


# ── environment helpers ───────────────────────────────────────────────────────

def make_gymnasium_env():
    try:
        import gymnasium
        import gymnasium_robotics  # noqa: F401 — registers envs
    except ImportError as exc:
        raise ImportError(
            'gymnasium-robotics is required: pip install gymnasium-robotics') from exc
    env = gymnasium.make('PointMaze_UMaze-v3',
                         render_mode='rgb_array',
                         max_episode_steps=None)
    env.reset(seed=0)
    return env


def sample_random_state(env, seed: int) -> np.ndarray:
    """Return [x, y, vx, vy] from a seeded gymnasium reset.

    Identical to compare_cem_dinowm_pointmaze.py:sample_random_state().
    """
    obs_dict, _ = env.reset(seed=int(seed))
    return obs_dict['observation'].astype(np.float32)


# ── oracle CEM planner ────────────────────────────────────────────────────────

class GymnasiumOracleCEM:
    """Oracle CEM using gymnasium's raw MuJoCo model for rollouts.

    Bypasses gymnasium's Python wrapper for speed; uses the same underlying
    MuJoCo model as the environment the model CEM evaluates in.
    Pass frame_skip > 0 to override the environment's default (e.g. 5 to
    match DINO-WM's dynamics depth and achieve ~100 % SR).
    """

    def __init__(self, gym_env, goal_xy,
                 horizon, executed_steps,
                 population, elites, iterations, initial_variance_scale,
                 frame_skip: int = 0):
        self.goal_xy        = np.asarray(goal_xy, dtype=np.float64)
        self.horizon        = int(horizon)
        self.executed_steps = min(int(executed_steps), self.horizon)
        self.population     = int(population)
        self.elites         = min(int(elites), self.population)
        self.iterations     = int(iterations)
        self.initial_variance_scale = float(initial_variance_scale)

        # Grab raw MuJoCo handles from gymnasium's unwrapped env
        import mujoco as _mj
        self._mj        = _mj
        self._mj_model  = gym_env.unwrapped.model
        self._mj_data   = gym_env.unwrapped.data
        env_fs = int(getattr(gym_env.unwrapped, 'frame_skip', 1))
        self._frame_skip = frame_skip if frame_skip > 0 else env_fs
        self._n_ctrl    = min(2, self._mj_model.nu)

    def _set_physics(self, state):
        self._mj_data.qpos[:2] = state[:2]
        self._mj_data.qvel[:2] = state[2:]
        self._mj.mj_forward(self._mj_model, self._mj_data)

    def _get_state(self) -> np.ndarray:
        return np.array([self._mj_data.qpos[0], self._mj_data.qpos[1],
                         self._mj_data.qvel[0], self._mj_data.qvel[1]],
                        dtype=np.float64)

    def _rollout_cost(self, state, actions) -> float:
        self._set_physics(state)
        for a in actions:
            self._mj_data.ctrl[:self._n_ctrl] = np.clip(a, -1.0, 1.0)
            for _ in range(self._frame_skip):
                self._mj.mj_step(self._mj_model, self._mj_data)
        dx = self._mj_data.qpos[0] - self.goal_xy[0]
        dy = self._mj_data.qpos[1] - self.goal_xy[1]
        return dx * dx + dy * dy

    def _step_exec(self, state, action) -> np.ndarray:
        """Execute one env step (frame_skip mj_steps) from the given state."""
        self._set_physics(state)
        self._mj_data.ctrl[:self._n_ctrl] = np.clip(action, -1.0, 1.0)
        for _ in range(self._frame_skip):
            self._mj.mj_step(self._mj_model, self._mj_data)
        return self._get_state()

    def plan_sequence(self, state) -> np.ndarray:
        """Return (executed_steps, 2) best-action prefix from CEM."""
        state    = np.asarray(state, dtype=np.float64)
        mean     = np.zeros((self.horizon, 2), dtype=np.float64)
        variance = np.full((self.horizon, 2), self.initial_variance_scale,
                           dtype=np.float64)
        for _ in range(self.iterations):
            noise   = np.random.randn(self.population, self.horizon, 2)
            actions = np.clip(mean[None] + np.sqrt(variance)[None] * noise,
                              -1.0, 1.0)
            costs   = np.array([self._rollout_cost(state, actions[p])
                                for p in range(self.population)])
            elite   = actions[np.argsort(costs)[:self.elites]]
            mean     = elite.mean(axis=0)
            variance = np.maximum(elite.var(axis=0), 1e-6)
        return np.clip(mean[:self.executed_steps], -1.0, 1.0)


# ── trial evaluation ──────────────────────────────────────────────────────────

def cem_trial(planner, init_state, goal_xy,
              n_steps, success_threshold, success_hold_steps):
    state  = np.asarray(init_state, dtype=np.float64)
    states = [state.copy()]

    step          = 0
    consec_stable = 0

    while step < n_steps:
        sequence = planner.plan_sequence(state)

        for a in sequence:
            if step >= n_steps:
                break
            state = planner._step_exec(state, a)
            states.append(state.copy())
            step += 1

            xy_dist = float(np.linalg.norm(state[:2] - goal_xy))
            if xy_dist < success_threshold:
                consec_stable += 1
                if consec_stable >= success_hold_steps:
                    break
            else:
                consec_stable = 0

        if consec_stable >= success_hold_steps:
            break

    states = np.array(states)
    xy_err = np.linalg.norm(states[:, :2] - goal_xy[None], axis=1)
    stable = xy_err < success_threshold
    tail   = stable[-success_hold_steps:]

    return {
        'success':         bool(stable[-1]),
        'held':            bool(len(tail) == success_hold_steps and tail.all()),
        'final_error':     float(xy_err[-1]),
        'max_error':       float(np.max(xy_err)),
        'min_error':       float(np.min(xy_err)),
        'fraction_stable': float(np.mean(stable[1:])) if len(stable) > 1 else 0.,
        'goal_xy':         goal_xy.tolist(),
        'start_xy':        state[:2].tolist(),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Oracle CEM in gymnasium PointMaze — same env/seeds as '
                    'compare_cem_dinowm_pointmaze.py.')
    p.add_argument('--trials',               type=int,   default=50)
    p.add_argument('--n-steps',              type=int,   default=200,
                   help='Max env steps per trial')
    p.add_argument('--planning-horizon',     type=int,   default=5,
                   help='CEM rollout horizon (match compare_cem_dinowm_pointmaze.py default)')
    p.add_argument('--executed-steps',       type=int,   default=5,
                   help='Steps executed before replanning')
    p.add_argument('--cem-population',       type=int,   default=300)
    p.add_argument('--cem-elites',           type=int,   default=30)
    p.add_argument('--cem-iters',            type=int,   default=10)
    p.add_argument('--cem-initial-variance', type=float, default=1.0)
    p.add_argument('--success-threshold',    type=float, default=0.5)
    p.add_argument('--success-hold-steps',   type=int,   default=1)
    p.add_argument('--frame-skip',           type=int,   default=0,
                   help='mj_steps per action (0 = use env default=1). '
                        'Set 5 to match DINO-WM dynamics depth.')
    p.add_argument('--seed',                 type=int,   default=0,
                   help='Base seed: trial i uses start=seed+99*i+1, goal=seed+99*i+2')
    p.add_argument('--output',               required=True)
    args = p.parse_args()

    env = make_gymnasium_env()
    env_frame_skip = int(getattr(env.unwrapped, 'frame_skip', 1))
    frame_skip = args.frame_skip if args.frame_skip > 0 else env_frame_skip

    print(f'[GT Oracle CEM Gymnasium] env=PointMaze_UMaze-v3  '
          f'env_frame_skip={env_frame_skip}  effective_frame_skip={frame_skip}')
    print(f'[GT Oracle CEM Gymnasium] seed scheme: start=seed+99*i+1, goal=seed+99*i+2')
    print(f'[GT Oracle CEM Gymnasium] H={args.planning_horizon} K={args.executed_steps} '
          f'pop={args.cem_population} elites={args.cem_elites} '
          f'iters={args.cem_iters}  hold={args.success_hold_steps}  '
          f'effective_depth={args.planning_horizon * frame_skip}')

    trials, n_success = [], 0

    for i in range(args.trials):
        start_seed = args.seed + 99 * i + 1
        goal_seed  = args.seed + 99 * i + 2

        start_state = sample_random_state(env, start_seed)
        goal_state  = sample_random_state(env, goal_seed)
        goal_xy     = goal_state[:2].astype(np.float64)

        planner = GymnasiumOracleCEM(
            gym_env=env,
            goal_xy=goal_xy,
            horizon=args.planning_horizon,
            executed_steps=args.executed_steps,
            population=args.cem_population,
            elites=args.cem_elites,
            iterations=args.cem_iters,
            initial_variance_scale=args.cem_initial_variance,
            frame_skip=frame_skip,
        )

        row = cem_trial(
            planner, start_state, goal_xy,
            args.n_steps, args.success_threshold, args.success_hold_steps,
        )
        n_success += int(row['success'])
        print(f'[GT Oracle CEM Gymnasium] trial {i:03d}  success={row["success"]}  '
              f'err={row["final_error"]:.4f}  '
              f'goal=[{goal_xy[0]:.3f},{goal_xy[1]:.3f}]  '
              f'start=[{float(start_state[0]):.3f},{float(start_state[1]):.3f}]')
        trials.append(row)

    sr = n_success / max(len(trials), 1)
    print(f'[GT Oracle CEM Gymnasium] success rate: {sr:.1%}  ({n_success}/{len(trials)})')
    env.close()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump({
            'protocol': vars(args),
            'frame_skip': frame_skip,
            'env': 'PointMaze_UMaze-v3',
            'models': [{
                'label':        'GT Oracle CEM (gymnasium)',
                'success_rate': sr,
                'n_success':    n_success,
                'n_trials':     len(trials),
                'trials':       trials,
            }],
        }, f, indent=2)
    print(f'\n[done] {out}')


if __name__ == '__main__':
    main()
