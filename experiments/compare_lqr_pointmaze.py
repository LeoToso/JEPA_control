#!/usr/bin/env python
"""LQR control in latent space for PointMaze models.

For each model and each trial:
  1. Sample random start/goal using the DINO-WM seed scheme
     (start_seed = seed+99*i+1, goal_seed = seed+99*i+2) — same as
     gt_cem_gymnasium_pointmaze.py and gt_lqr_gymnasium_pointmaze.py.
  2. Encode goal state z* at the trial's goal position
  3. Linearise the predictor at z*: A_z (d×d), B_z (d×2)
  4. Solve DARE with Q = q_scale*I, R = r_scale*I_2
  5. Run closed-loop trial: u_t = -K @ (z_t - z*)  [2D action]
  6. Report success rate (agent reaches within threshold of goal)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.linalg
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
    if 'rollout_ms' in r or 'rollout-ms' in r: return 'Rollout+MS'
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


def encode_goal(bundle, env, goal_xy):
    """Encode observation at the goal position with zero velocity."""
    goal_state = np.array([goal_xy[0], goal_xy[1], 0., 0.], dtype=np.float32)
    obs, state, _ = env.reset_to_state(goal_state)
    z = encode_obs(bundle, obs, obs, state)
    return z, goal_state


def local_jacobians_2d(bundle, z_goal):
    """A_z (d×d) and B_z (d×2) at z_goal via autograd."""
    model  = bundle['model']
    scale  = bundle['action_scale']
    z0 = z_goal.detach().reshape(-1).requires_grad_(True)
    u0 = torch.zeros(2, device=z0.device, dtype=z0.dtype, requires_grad=True)

    def fz(zz):
        a_raw = u0.reshape(1, 2).detach() / scale
        a_ctx = model.expand_action(a_raw).unsqueeze(1)
        return model.predict(zz[None, None], a_ctx)[0, 0]

    def fu(uu):
        a_raw = uu.reshape(1, 2) / scale
        a_ctx = model.expand_action(a_raw).unsqueeze(1)
        return model.predict(z0.detach()[None, None], a_ctx)[0, 0]

    A = torch.autograd.functional.jacobian(fz, z0, vectorize=True)
    B = torch.autograd.functional.jacobian(fu, u0, vectorize=True)

    with torch.no_grad():
        a_raw = torch.zeros(1, 2, device=z0.device, dtype=z0.dtype) / scale
        a_ctx = model.expand_action(a_raw).unsqueeze(1)
        z_next = model.predict(z0.detach()[None, None], a_ctx)[0, 0]
        fp_err = float(torch.linalg.vector_norm(z_next - z0.detach()).cpu())

    return (A.detach().cpu().numpy(),
            B.detach().cpu().numpy(),
            fp_err)


def solve_dare_2d(A, B, Q, R):
    """DARE for 2D action → K (2×d)."""
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K


def lqr_trial(bundle, env, K, z_goal, goal_xy, init_state,
              frame_skip, n_steps, success_threshold, success_hold_steps):
    obs, state, _ = env.reset_to_state(init_state)
    prev_obs = obs.copy()
    states = [state.copy()]

    for _ in range(n_steps):
        with torch.no_grad():
            z = encode_obs(bundle, obs, prev_obs, state)
            z_np = z.cpu().numpy().flatten()

        u = -(K @ (z_np - z_goal))
        u = np.clip(u, -bundle['action_scale'], bundle['action_scale'])

        prev_obs = obs.copy()
        done = False
        for _ in range(frame_skip - 1):
            state, _, done, _ = env.step_no_render(u)
            if done:
                break
        if not done:
            obs, state, _, done, _ = env.step(u)
        else:
            obs = prev_obs.copy()

        states.append(state.copy())
        if done:
            break

    states = np.array(states)
    xy_err = np.linalg.norm(states[:, :2] - goal_xy[None], axis=1)
    stable = xy_err < success_threshold
    tail   = stable[-success_hold_steps:]
    held   = bool(len(tail) == success_hold_steps and tail.all())
    success = bool(stable[-1])

    run = 0
    for i in range(len(stable) - 1, -1, -1):
        if stable[i]: run += 1
        else: break
    settling = len(stable) - run if run >= success_hold_steps else None

    return {
        'success': success,
        'held': held,
        'final_error': float(xy_err[-1]),
        'max_error':   float(np.max(xy_err)),
        'fraction_stable': float(np.mean(stable)),
        'settling_step': settling,
        'states':  states.tolist(),
        'goal_xy': goal_xy.tolist(),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Model-based latent LQR for PointMaze. Uses same DINO-WM seed '
                    'scheme as gt_cem/gt_lqr oracle scripts for fair comparison.')
    p.add_argument('--ckpts', nargs='+', required=True)
    p.add_argument('--cfgs',  nargs='+', required=True)
    p.add_argument('--trials', type=int, default=50)
    p.add_argument('--n-steps', type=int, default=200)
    p.add_argument('--success-threshold', type=float, default=0.5,
                   help='XY distance to goal for success (maze units)')
    p.add_argument('--success-hold-steps', type=int, default=1)
    p.add_argument('--q-scale', type=float, default=1.0)
    p.add_argument('--r-scale', type=float, default=1.0)
    p.add_argument('--frame-skip', type=int, default=1,
                   help='PointMazeVisual steps per action (1=no skip). '
                        'Set 5 to match fs5 model training.')
    p.add_argument('--static-k', action='store_true', default=False,
                   help='Compute K once at trial-0 goal and reuse for all trials. '
                        'Default: recompute K per trial (since goal changes each trial).')
    p.add_argument('--seed', type=int, default=0,
                   help='Base seed: trial i uses start=seed+99*i+1, goal=seed+99*i+2')
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', required=True)
    args = p.parse_args()

    if len(args.cfgs) == 1:
        args.cfgs = args.cfgs * len(args.ckpts)

    print(f'[Model LQR] seed scheme: start=seed+99*i+1, goal=seed+99*i+2')
    print(f'[Model LQR] frame_skip={args.frame_skip}  n_steps={args.n_steps}  '
          f'hold={args.success_hold_steps}  static_k={args.static_k}')

    results = {'protocol': vars(args), 'models': []}

    for ckpt, cfg in zip(args.ckpts, args.cfgs):
        label = checkpoint_label(ckpt)
        print(f'\n[{label}] loading {ckpt}')
        bundle = load_bundle(ckpt, cfg, args.device)

        env = make_env(bundle, args.seed)

        Q = None  # set lazily after first Jacobian computation (need d)
        R = args.r_scale * np.eye(2)
        K_cached = None

        rho_Az_list, rho_cl_list, fp_err_list = [], [], []
        trials, n_success, n_dare_fail = [], 0, 0

        for i in range(args.trials):
            start_seed = args.seed + 99 * i + 1
            goal_seed  = args.seed + 99 * i + 2

            # Sample positions via seeded gymnasium resets (same protocol as oracle)
            obs_dict, _ = env._env.reset(seed=int(start_seed))
            start_state  = obs_dict['observation'][:4].astype(np.float32)

            obs_dict, _ = env._env.reset(seed=int(goal_seed))
            goal_xy = obs_dict['observation'][:2].astype(np.float32)

            # Encode goal in latent space
            z_goal_t, _ = encode_goal(bundle, env, goal_xy)
            z_goal = z_goal_t.detach().cpu().numpy().flatten()

            # Compute Jacobians and K (per-trial unless --static-k)
            if K_cached is None or not args.static_k:
                A_z, B_z, fp_err = local_jacobians_2d(bundle, z_goal_t)
                if Q is None:
                    d = A_z.shape[0]
                    Q = args.q_scale * np.eye(d)
                rho_Az = float(np.max(np.abs(np.linalg.eigvals(A_z))))
                try:
                    K = solve_dare_2d(A_z, B_z, Q, R)
                except Exception as e:
                    print(f'[{label}] trial {i:03d}  DARE failed: {e}')
                    n_dare_fail += 1
                    trials.append({'success': False, 'held': False,
                                   'final_error': float('nan'),
                                   'max_error': float('nan'),
                                   'fraction_stable': 0., 'settling_step': None})
                    continue
                rho_cl = float(np.max(np.abs(np.linalg.eigvals(A_z - B_z @ K))))
                rho_Az_list.append(rho_Az)
                rho_cl_list.append(rho_cl)
                fp_err_list.append(fp_err)
                if args.static_k:
                    K_cached = K
            else:
                K = K_cached

            row = lqr_trial(bundle, env, K, z_goal, goal_xy, start_state,
                            args.frame_skip, args.n_steps,
                            args.success_threshold, args.success_hold_steps)
            n_success += int(row['success'])
            print(f'[{label}] trial {i:03d}  success={row["success"]}  '
                  f'err={row["final_error"]:.4f}  '
                  f'goal=[{goal_xy[0]:.3f},{goal_xy[1]:.3f}]  '
                  f'start=[{start_state[0]:.3f},{start_state[1]:.3f}]')
            trials.append(row)

        sr = n_success / max(len(trials), 1)
        print(f'[{label}] success rate: {sr:.1%}  ({n_success}/{len(trials)})  '
              f'DARE_fails={n_dare_fail}')
        env.close()

        results['models'].append({
            'label': label,
            'success_rate': sr,
            'n_success': n_success,
            'n_trials': len(trials),
            'n_dare_fail': n_dare_fail,
            'mean_rho_Az': float(np.mean(rho_Az_list)) if rho_Az_list else float('nan'),
            'mean_rho_closed_loop': float(np.mean(rho_cl_list)) if rho_cl_list else float('nan'),
            'mean_fp_err': float(np.mean(fp_err_list)) if fp_err_list else float('nan'),
            'trials': trials,
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n[done] {out}')


if __name__ == '__main__':
    main()
