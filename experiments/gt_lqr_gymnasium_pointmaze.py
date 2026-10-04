#!/usr/bin/env python
"""Oracle LQR on gymnasium-robotics PointMaze — fair upper bound for model-based LQR.

Uses the SAME environment (PointMaze_UMaze-v3), SAME seed scheme, and SAME
start/goal positions as gt_cem_gymnasium_pointmaze.py, so results are a
directly comparable oracle ceiling for model-based controllers.

Physics: raw MuJoCo finite-difference Jacobians at the goal, DARE → linear
feedback K.  Controller: u_t = -K (s_t - s*), clipped to [-1, 1].

Limitation: LQR is a linear controller — it cannot plan around the U-maze wall.
Cross-arm trials (goal on the opposite side of the wall from start) will mostly
fail because the ball drives straight toward the goal and collides with the wall.

Usage
-----
  MUJOCO_GL=osmesa python experiments/gt_lqr_gymnasium_pointmaze.py \\
      --trials 50 --n-steps 200 \\
      --frame-skip 5 \\
      --output results/gt_oracle_lqr_gymnasium_H25_fs5.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.linalg


# ── environment helpers ───────────────────────────────────────────────────────

def make_gymnasium_env():
    try:
        import gymnasium
        import gymnasium_robotics  # noqa: F401
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

    Identical to gt_cem_gymnasium_pointmaze.py:sample_random_state().
    """
    obs_dict, _ = env.reset(seed=int(seed))
    return obs_dict['observation'].astype(np.float32)


# ── finite-difference Jacobians ───────────────────────────────────────────────

def jacobians_fd(gym_env, state, frame_skip: int,
                 eps_s: float = 1e-3, eps_u: float = 1e-2):
    """Central finite-difference A (4×4) and B (4×2) at state, zero action."""
    import mujoco as _mj
    mj_model = gym_env.unwrapped.model
    mj_data  = gym_env.unwrapped.data
    n_ctrl   = min(2, mj_model.nu)

    def _rollout(s, u):
        mj_data.qpos[:2] = s[:2]
        mj_data.qvel[:2] = s[2:]
        _mj.mj_forward(mj_model, mj_data)
        mj_data.ctrl[:n_ctrl] = np.clip(u, -1.0, 1.0)
        for _ in range(frame_skip):
            _mj.mj_step(mj_model, mj_data)
        return np.array([mj_data.qpos[0], mj_data.qpos[1],
                         mj_data.qvel[0], mj_data.qvel[1]], dtype=np.float64)

    n, m = 4, 2
    u0 = np.zeros(m)
    A  = np.zeros((n, n))
    for j in range(n):
        sp = state.copy(); sp[j] += eps_s
        sm = state.copy(); sm[j] -= eps_s
        A[:, j] = (_rollout(sp, u0) - _rollout(sm, u0)) / (2.0 * eps_s)

    B = np.zeros((n, m))
    for j in range(m):
        up = u0.copy(); up[j] += eps_u
        um = u0.copy(); um[j] -= eps_u
        B[:, j] = (_rollout(state, up) - _rollout(state, um)) / (2.0 * eps_u)

    return A, B


def solve_dare(A, B, Q, R):
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K


# ── trial evaluation ──────────────────────────────────────────────────────────

def lqr_trial(gym_env, K, init_state, goal_state, goal_xy,
              frame_skip: int, n_steps: int,
              success_threshold: float, success_hold_steps: int,
              relinearize_every: int = 0, Q=None, R=None,
              eps_s: float = 1e-3, eps_u: float = 1e-2):
    import mujoco as _mj
    mj_model = gym_env.unwrapped.model
    mj_data  = gym_env.unwrapped.data
    n_ctrl   = min(2, mj_model.nu)

    def _set(s):
        mj_data.qpos[:2] = s[:2]
        mj_data.qvel[:2] = s[2:]
        _mj.mj_forward(mj_model, mj_data)

    def _step(u):
        mj_data.ctrl[:n_ctrl] = np.clip(u, -1.0, 1.0)
        for _ in range(frame_skip):
            _mj.mj_step(mj_model, mj_data)
        return np.array([mj_data.qpos[0], mj_data.qpos[1],
                         mj_data.qvel[0], mj_data.qvel[1]], dtype=np.float64)

    state = np.asarray(init_state, dtype=np.float64)
    _set(state)
    states        = [state.copy()]
    goal          = np.asarray(goal_state, dtype=np.float64)
    K_cur         = K.copy()
    consec_stable = 0
    relin_fails   = 0

    for step in range(n_steps):
        if relinearize_every > 0 and step % relinearize_every == 0:
            try:
                A_t, B_t = jacobians_fd(gym_env, state, frame_skip, eps_s, eps_u)
                K_cur = solve_dare(A_t, B_t, Q, R)
            except Exception:
                relin_fails += 1
                K_cur = K

        u = -(K_cur @ (state - goal))
        u = np.clip(u, -1.0, 1.0)
        _set(state)
        state = _step(u)
        states.append(state.copy())

        xy_dist = float(np.linalg.norm(state[:2] - goal_xy))
        if xy_dist < success_threshold:
            consec_stable += 1
            if consec_stable >= success_hold_steps:
                break
        else:
            consec_stable = 0

    states = np.array(states)
    xy_err = np.linalg.norm(states[:, :2] - goal_xy[None], axis=1)
    stable = xy_err < success_threshold
    tail   = stable[-success_hold_steps:]

    return {
        'success':              bool(stable[-1]),
        'held':                 bool(len(tail) == success_hold_steps and tail.all()),
        'final_error':          float(xy_err[-1]),
        'max_error':            float(np.max(xy_err)),
        'min_error':            float(np.min(xy_err)),
        'fraction_stable':      float(np.mean(stable[1:])) if len(stable) > 1 else 0.,
        'relinearize_failures': relin_fails,
        'states':  states.tolist(),
        'goal_xy': goal_xy.tolist(),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Oracle LQR in gymnasium PointMaze — same env/seeds as '
                    'gt_cem_gymnasium_pointmaze.py.')
    p.add_argument('--trials',               type=int,   default=50)
    p.add_argument('--n-steps',              type=int,   default=200)
    p.add_argument('--success-threshold',    type=float, default=0.5)
    p.add_argument('--success-hold-steps',   type=int,   default=1)
    p.add_argument('--q-scale',              type=float, default=1.0,
                   help='Q = q_scale * I_4')
    p.add_argument('--r-scale',              type=float, default=1.0,
                   help='R = r_scale * I_2')
    p.add_argument('--eps-state',            type=float, default=1e-3)
    p.add_argument('--eps-action',           type=float, default=1e-2)
    p.add_argument('--relinearize-every',    type=int,   default=0,
                   help='Re-linearise at current state every k steps (0 = static).')
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

    mode = ('static' if args.relinearize_every == 0
            else f'receding-horizon (every {args.relinearize_every} steps)')

    print(f'[GT Oracle LQR Gymnasium] env=PointMaze_UMaze-v3  '
          f'env_frame_skip={env_frame_skip}  effective_frame_skip={frame_skip}')
    print(f'[GT Oracle LQR Gymnasium] seed scheme: start=seed+99*i+1, goal=seed+99*i+2')
    print(f'[GT Oracle LQR Gymnasium] n_steps={args.n_steps}  '
          f'hold={args.success_hold_steps}  mode={mode}  '
          f'effective_depth={frame_skip}')
    print(f'[GT Oracle LQR Gymnasium] Q={args.q_scale}*I  R={args.r_scale}*I')

    Q = args.q_scale * np.eye(4)
    R = args.r_scale * np.eye(2)

    trials, n_success = [], 0

    for i in range(args.trials):
        start_seed = args.seed + 99 * i + 1
        goal_seed  = args.seed + 99 * i + 2

        start_state = sample_random_state(env, start_seed)
        goal_state  = sample_random_state(env, goal_seed)
        goal_xy     = goal_state[:2].astype(np.float64)
        goal_s      = goal_state.astype(np.float64)

        # Linearise at the goal state (needed for static K and RH fallback)
        try:
            A, B = jacobians_fd(env, goal_s, frame_skip,
                                args.eps_state, args.eps_action)
            K = solve_dare(A, B, Q, R)
            rho_cl = float(np.max(np.abs(np.linalg.eigvals(A - B @ K))))
            dare_ok = True
        except Exception as exc:
            print(f'[GT Oracle LQR Gymnasium] trial {i:03d}  DARE failed: {exc}')
            trials.append({'success': False, 'held': False,
                           'final_error': float('nan'), 'max_error': float('nan'),
                           'min_error': float('nan'), 'fraction_stable': 0.,
                           'relinearize_failures': 1})
            continue

        row = lqr_trial(
            env, K, start_state, goal_s, goal_xy,
            frame_skip, args.n_steps,
            args.success_threshold, args.success_hold_steps,
            args.relinearize_every, Q, R,
            args.eps_state, args.eps_action,
        )
        n_success += int(row['success'])
        print(f'[GT Oracle LQR Gymnasium] trial {i:03d}  success={row["success"]}  '
              f'err={row["final_error"]:.4f}  '
              f'rho_cl={rho_cl:.4f}  '
              f'goal=[{goal_xy[0]:.3f},{goal_xy[1]:.3f}]  '
              f'start=[{float(start_state[0]):.3f},{float(start_state[1]):.3f}]')
        trials.append(row)

    sr = n_success / max(len(trials), 1)
    print(f'[GT Oracle LQR Gymnasium] success rate: {sr:.1%}  ({n_success}/{len(trials)})')
    env.close()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump({
            'protocol': vars(args),
            'frame_skip': frame_skip,
            'mode': mode,
            'env': 'PointMaze_UMaze-v3',
            'models': [{
                'label':        'GT Oracle LQR (gymnasium)',
                'success_rate': sr,
                'n_success':    n_success,
                'n_trials':     len(trials),
                'trials':       trials,
            }],
        }, f, indent=2)
    print(f'\n[done] {out}')


if __name__ == '__main__':
    main()

