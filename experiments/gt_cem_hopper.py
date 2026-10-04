#!/usr/bin/env python
"""GT CEM-MPC oracle for Hopper-v4 — true MuJoCo dynamics, no learned model.

Hopper is a monopod with 3 actuators and simpler dynamics than Walker2d.
Same objective and planner structure as gt_cem_walker.py:

    J = -wx*(x_H - x_0) + Cf*(H - T_alive) + wu * sum_{t} ||u_t||^2

Health condition (Gymnasium Hopper-v4 defaults):
    z > 0.7   AND   |torso_angle| < 0.2

Action blocking: CEM optimises n_blocks = ceil(H / block_size) × 3-D parameters.

Usage
-----
  python experiments/gt_cem_hopper.py \\
      --trials 10 --n-steps 500 \\
      --planning-horizon 100 --action-block-size 4 \\
      --cem-population 1000 --cem-elites 50 --cem-iters 5 \\
      --executed-steps 1 \\
      --wx 5.0 --cf 10.0 --wu 1e-3 \\
      --success-min-velocity 0.5 \\
      --render-dir results/hopper_gt_cem_frames \\
      --output results/gt_cem_hopper.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import gymnasium as gym

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

ACTION_LOW  = -1.0
ACTION_HIGH =  1.0
ACTION_DIM  =  3      # Hopper: hip, knee, ankle

# Hopper-v4 health bounds (Gymnasium defaults)
HEALTHY_Z_MIN   = 0.7
HEALTHY_ANG_MAX = 0.2   # |torso_angle| < 0.2 rad  (much tighter than Walker)


# ── env state helpers ─────────────────────────────────────────────────────────

def get_state(env):
    d = env.unwrapped.data
    return d.qpos.copy(), d.qvel.copy()


def set_state(env, qpos, qvel):
    env.unwrapped.set_state(qpos, qvel)   # gymnasium calls mj_forward internally


def is_healthy(data):
    """Match Gymnasium Hopper-v4 health condition.

    qpos layout: [x_pos, z_height, torso_angle, thigh, leg, foot]
    """
    z   = float(data.qpos[1])
    ang = float(data.qpos[2])
    return z > HEALTHY_Z_MIN and abs(ang) < HEALTHY_ANG_MAX


def failure_reason(data):
    z   = float(data.qpos[1])
    ang = float(data.qpos[2])
    if z <= HEALTHY_Z_MIN:
        return 'low_height'
    if abs(ang) >= HEALTHY_ANG_MAX:
        return 'excessive_angle'
    return 'other'


# ── CEM rollout ───────────────────────────────────────────────────────────────

def rollout_cost(plan_env, start_qpos, start_qvel, actions, frame_skip,
                 wx, cf, wu, wv=0.0, wvt=0.0):
    """Evaluate one action sequence under:

        J = -wv  * sum(x_vel_t)                   ← dense per-step velocity
            - wx  * (x_H - x_0)                   ← endpoint displacement (optional)
            + cf  * (H - T_alive)                  ← early-termination penalty
            + wu  * sum(||u_t||^2)                 ← action regulariser
            - wvt * max(0, x_vel_H) * [T_alive==H] ← terminal velocity bonus (IHMPC)

    The terminal bonus (wvt) is an approximation of the Erez et al. IHMPC
    terminal value function.  It gives a large reward only if the hopper
    survives the FULL H-step plan with positive forward velocity.  This
    breaks the warm-start trap where CEM locks onto "run fast and fall":

      - Fall at T=60 at v=1.6 m/s → no terminal bonus
      - Survive all H=100 at v=1.0 m/s → gets -wvt*1.0 terminal bonus

    With wvt >> wv*H, surviving is overwhelmingly better than fast-fall.

    Health checked AFTER each full frame_skip macro-step (matches Gymnasium).
    """
    set_state(plan_env, start_qpos, start_qvel)
    model = plan_env.unwrapped.model
    data  = plan_env.unwrapped.data

    x0       = float(data.qpos[0])
    H        = len(actions)
    t_alive  = 0
    vel_sum  = 0.0   # accumulated forward velocity over alive steps

    for t in range(H):
        a = np.clip(actions[t], ACTION_LOW, ACTION_HIGH)
        if HAS_MUJOCO:
            data.ctrl[:] = a
            for _ in range(frame_skip):
                mujoco.mj_step(model, data)
            if not is_healthy(data):
                break
        else:
            plan_env.step(a)
            if not is_healthy(plan_env.unwrapped.data):
                break
        t_alive += 1
        vel_sum += float(data.qvel[0])   # qvel[0] = forward x-velocity

    x_H            = float(data.qpos[0])
    action_penalty = float(np.sum(actions[:t_alive + 1] ** 2)) * wu

    # Terminal value: approximate IHMPC bonus for surviving the full horizon
    terminal_bonus = 0.0
    if t_alive == H:
        terminal_bonus = wvt * max(0.0, float(data.qvel[0]))

    cost = (-wv * vel_sum
            - wx * (x_H - x0)
            + cf * (H - t_alive)
            + action_penalty
            - terminal_bonus)
    return cost


# ── CEM planner with action blocking ─────────────────────────────────────────

class HopperGTCEM:
    """GT CEM-MPC planner with action blocking and warm start for Hopper-v4."""

    def __init__(self, horizon, block_size, executed_steps,
                 population, elites, iterations, initial_variance,
                 plan_env, frame_skip, wx, cf, wu, wv=5.0, wvt=0.0,
                 no_warmstart=False, min_variance=1e-6):
        self.horizon        = int(horizon)
        self.block_size     = int(block_size)
        self.n_blocks       = math.ceil(horizon / block_size)
        self.executed_steps = min(int(executed_steps), horizon)
        self.population     = int(population)
        self.elites         = min(int(elites), population)
        self.iterations     = int(iterations)
        self.initial_var    = float(initial_variance)
        self.plan_env       = plan_env
        self.frame_skip     = int(frame_skip)
        self.wx             = float(wx)
        self.cf             = float(cf)
        self.wu             = float(wu)
        self.wv             = float(wv)
        self.wvt            = float(wvt)
        self.no_warmstart   = bool(no_warmstart)
        self.min_variance   = float(min_variance)
        self._prev_blocks   = None

    def _expand(self, blocks):
        return np.repeat(blocks, self.block_size, axis=0)[:self.horizon]

    def _shift_blocks(self, blocks):
        n_exec = max(1, math.ceil(self.executed_steps / self.block_size))
        return np.concatenate([
            blocks[n_exec:],
            np.zeros((n_exec, ACTION_DIM))
        ], axis=0)

    def plan(self, start_qpos, start_qvel):
        if self.no_warmstart or self._prev_blocks is None:
            mean = np.zeros((self.n_blocks, ACTION_DIM))
        else:
            mean = self._shift_blocks(self._prev_blocks)
        variance = np.full((self.n_blocks, ACTION_DIM), self.initial_var)

        for _ in range(self.iterations):
            noise  = np.random.randn(self.population, self.n_blocks, ACTION_DIM)
            blocks = np.clip(
                mean[None] + np.sqrt(variance)[None] * noise,
                ACTION_LOW, ACTION_HIGH)

            costs = np.empty(self.population)
            for k in range(self.population):
                costs[k] = rollout_cost(
                    self.plan_env, start_qpos, start_qvel,
                    self._expand(blocks[k]), self.frame_skip,
                    self.wx, self.cf, self.wu, self.wv, self.wvt)

            elite_idx = np.argsort(costs)[:self.elites]
            elite     = blocks[elite_idx]
            mean      = elite.mean(0)
            variance  = elite.var(0).clip(min=self.min_variance)

        self._prev_blocks = mean
        actions = self._expand(mean)
        return np.clip(actions[:self.executed_steps], ACTION_LOW, ACTION_HIGH)


# ── trial evaluation ──────────────────────────────────────────────────────────

def run_trial(planner, eval_env, initial_qpos, initial_qvel,
              n_steps, do_render):
    set_state(eval_env, initial_qpos, initial_qvel)
    planner._prev_blocks = None   # fresh warm start per trial

    step        = 0
    x_vels      = []
    frames      = []
    terminated  = False
    truncated   = False

    while step < n_steps and not (terminated or truncated):
        qpos, qvel = get_state(eval_env)
        sequence   = planner.plan(qpos, qvel)

        for a in sequence:
            if step >= n_steps or terminated or truncated:
                break
            obs, reward, terminated, truncated, info = eval_env.step(a)
            step += 1
            x_vels.append(float(info.get('x_velocity', 0.0)))
            if do_render:
                frames.append(eval_env.render())

    data           = eval_env.unwrapped.data
    final_height   = float(data.qpos[1])
    final_angle    = float(data.qpos[2])
    forward_disp   = float(data.qpos[0]) - float(initial_qpos[0])
    survived_full  = (step >= n_steps) and not terminated
    avg_vel        = float(np.mean(x_vels)) if x_vels else 0.0

    if terminated:
        fail = failure_reason(data)
    elif truncated:
        fail = 'truncated'
    else:
        fail = 'timeout'

    return {
        'survived_full':  survived_full,
        'steps':          step,
        'forward_disp':   forward_disp,
        'avg_x_velocity': avg_vel,
        'final_height':   final_height,
        'final_angle':    final_angle,
        'fail_reason':    fail,
        'x_velocities':   x_vels,
    }, frames


# ── visualization helpers ─────────────────────────────────────────────────────

def save_gif(frames, out_path, fps=30):
    if not frames:
        return
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio
        imageio.mimsave(str(out), frames, fps=fps)
    except ImportError:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(str(out), save_all=True, append_images=imgs[1:],
                     loop=0, duration=int(1000 / fps))
    print(f'[gif saved] {out}')


def save_frame_grid(frames, out_path, every=10, title='GT CEM Hopper'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    subset = frames[::every]
    n      = len(subset)
    if n == 0:
        return
    cols = min(8, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.array(axes).reshape(rows, cols)
    for idx, frame in enumerate(subset):
        r, c = divmod(idx, cols)
        axes[r, c].imshow(frame)
        axes[r, c].axis('off')
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis('off')
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'[grid saved] {out}')


def pick_viz_trial(trials_data, success_mask):
    successes = [i for i, s in enumerate(success_mask) if s]
    if successes:
        return max(successes, key=lambda i: trials_data[i]['avg_x_velocity'])
    return max(range(len(trials_data)),
               key=lambda i: (trials_data[i]['steps'],
                              trials_data[i]['forward_disp']))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='GT oracle CEM-MPC for Hopper-v4 — true MuJoCo dynamics.')
    p.add_argument('--trials',               type=int,   default=10)
    p.add_argument('--n-steps',              type=int,   default=500)
    p.add_argument('--planning-horizon',     type=int,   default=100)
    p.add_argument('--action-block-size',    type=int,   default=4)
    p.add_argument('--executed-steps',       type=int,   default=1)
    p.add_argument('--cem-population',       type=int,   default=1000)
    p.add_argument('--cem-elites',           type=int,   default=50)
    p.add_argument('--cem-iters',            type=int,   default=5)
    p.add_argument('--cem-initial-variance', type=float, default=0.5)
    p.add_argument('--wv',  type=float, default=5.0,
                   help='Dense per-step x-velocity reward weight (primary Hopper objective)')
    p.add_argument('--wx',  type=float, default=0.0,
                   help='Endpoint displacement bonus weight (secondary; 0 = off)')
    p.add_argument('--cf',  type=float, default=0.5,
                   help='Cost per wasted step after fall (keep small so CEM takes risks)')
    p.add_argument('--wu',  type=float, default=1e-3,
                   help='Action regularisation weight')
    p.add_argument('--wvt', type=float, default=0.0,
                   help='Terminal velocity bonus weight — awarded only when the hopper '
                        'survives the full H-step plan with positive forward velocity. '
                        'Approximates the IHMPC terminal value function (Erez et al.). '
                        'Rule of thumb: wvt ≈ wv*H to dominate intermediate reward.')
    p.add_argument('--no-warmstart', action='store_true', default=False,
                   help='Disable warm-start: CEM initialises from zero at every MPC step. '
                        'Slower but escapes the "fast-hop-fall" local basin that warm-start '
                        'locks into. Recommended when --wvt is set.')
    p.add_argument('--cem-min-variance', type=float, default=1e-6,
                   help='Minimum per-dim CEM variance floor (default 1e-6 = near-zero). '
                        'Raising to 0.05-0.1 keeps diversity across iterations.')
    p.add_argument('--success-min-velocity', type=float, default=0.5)
    p.add_argument('--frame-skip',           type=int,   default=4)
    p.add_argument('--render-dir',           default='')
    p.add_argument('--render-every',         type=int,   default=10)
    p.add_argument('--gif-fps',              type=int,   default=30)
    p.add_argument('--seed',                 type=int,   default=42)
    p.add_argument('--output',               required=True)
    args = p.parse_args()

    np.random.seed(args.seed)

    plan_env = gym.make('Hopper-v4')
    eval_env = gym.make('Hopper-v4',
                        render_mode='rgb_array' if args.render_dir else None)
    plan_env.reset(seed=args.seed)
    eval_env.reset(seed=args.seed)

    n_blocks = math.ceil(args.planning_horizon / args.action_block_size)
    planner = HopperGTCEM(
        horizon          = args.planning_horizon,
        block_size       = args.action_block_size,
        executed_steps   = args.executed_steps,
        population       = args.cem_population,
        elites           = args.cem_elites,
        iterations       = args.cem_iters,
        initial_variance = args.cem_initial_variance,
        plan_env         = plan_env,
        frame_skip       = args.frame_skip,
        wx               = args.wx,
        cf               = args.cf,
        wu               = args.wu,
        wv               = args.wv,
        wvt              = args.wvt,
        no_warmstart     = args.no_warmstart,
        min_variance     = args.cem_min_variance,
    )
    print(f'[GT-CEM Hopper] H={args.planning_horizon} block={args.action_block_size} '
          f'n_blocks={n_blocks} K={args.executed_steps} '
          f'pop={args.cem_population} elites={args.cem_elites} iters={args.cem_iters}')
    print(f'  objective: wv={args.wv} wx={args.wx} cf={args.cf} wu={args.wu} wvt={args.wvt}')
    print(f'  warmstart={not args.no_warmstart}  min_var={args.cem_min_variance}')
    print(f'  health:    z > {HEALTHY_Z_MIN}  |angle| < {HEALTHY_ANG_MAX}')
    print(f'  success:   survived_full AND avg_vel > {args.success_min_velocity} m/s')

    trials_data  = []
    frames_list  = []
    success_mask = []

    for i in range(args.trials):
        np.random.seed(args.seed + i)
        obs, _ = eval_env.reset(seed=args.seed + i)
        initial_qpos, initial_qvel = get_state(eval_env)
        plan_env.reset(seed=args.seed + i)
        set_state(plan_env, initial_qpos, initial_qvel)

        print(f'[trial {i:03d}] ', end='', flush=True)
        t0 = time.time()

        row, frames = run_trial(
            planner, eval_env, initial_qpos, initial_qvel,
            args.n_steps, bool(args.render_dir))

        elapsed = time.time() - t0
        success = row['survived_full'] and row['avg_x_velocity'] >= args.success_min_velocity
        row['success'] = success
        success_mask.append(success)
        trials_data.append(row)
        frames_list.append(frames)

        status = 'SUCCESS' if success else f'FAIL({row["fail_reason"]})'
        print(f'{status}  survived={row["survived_full"]}  '
              f'steps={row["steps"]}  disp={row["forward_disp"]:.2f}m  '
              f'avg_vel={row["avg_x_velocity"]:.3f}m/s  '
              f'h={row["final_height"]:.3f}  ang={row["final_angle"]:.3f}  '
              f't={elapsed:.1f}s')

    n_success = sum(success_mask)
    sr        = n_success / max(args.trials, 1)
    mean_vel  = float(np.mean([r['avg_x_velocity'] for r in trials_data]))
    mean_disp = float(np.mean([r['forward_disp']   for r in trials_data]))

    print(f'\nSuccess: {n_success}/{args.trials} ({sr:.1%})')
    print(f'Mean avg_x_velocity: {mean_vel:.3f} m/s  |  Mean forward disp: {mean_disp:.2f} m')

    if any(not s for s in success_mask):
        from collections import Counter
        reasons = [r['fail_reason'] for r in trials_data if not r['success']]
        print('Failure breakdown:', dict(Counter(reasons)))

    if args.render_dir:
        viz_idx    = pick_viz_trial(trials_data, success_mask)
        viz_frames = frames_list[viz_idx]
        row        = trials_data[viz_idx]
        tag  = 'success' if success_mask[viz_idx] else 'best_failed'
        stem = f'{tag}_trial_{viz_idx:03d}'
        title = (f'GT CEM Hopper — trial {viz_idx} [{tag}] '
                 f'vel={row["avg_x_velocity"]:.3f} disp={row["forward_disp"]:.2f}m')
        if viz_frames:
            save_gif(viz_frames, Path(args.render_dir) / f'{stem}.gif',
                     fps=args.gif_fps)
            save_frame_grid(viz_frames, Path(args.render_dir) / f'{stem}.png',
                            every=args.render_every, title=title)

    for r in trials_data:
        r.pop('x_velocities', None)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'protocol': {
            'planner':              'gt_cem_mpc',
            'env':                  'Hopper-v4',
            'n_steps':              args.n_steps,
            'planning_horizon':     args.planning_horizon,
            'action_block_size':    args.action_block_size,
            'n_blocks':             n_blocks,
            'executed_steps':       args.executed_steps,
            'cem_population':       args.cem_population,
            'cem_elites':           args.cem_elites,
            'cem_iters':            args.cem_iters,
            'cem_initial_variance': args.cem_initial_variance,
            'wx':                   args.wx,
            'cf':                   args.cf,
            'wu':                   args.wu,
            'frame_skip':           args.frame_skip,
            'success_min_velocity': args.success_min_velocity,
            'seed':                 args.seed,
            'n_trials':             args.trials,
        },
        'success_rate':      sr,
        'n_success':         n_success,
        'mean_avg_velocity': mean_vel,
        'mean_forward_disp': mean_disp,
        'trials':            trials_data,
    }, indent=2))
    print(f'[done] {out}')

    plan_env.close()
    eval_env.close()


if __name__ == '__main__':
    main()

