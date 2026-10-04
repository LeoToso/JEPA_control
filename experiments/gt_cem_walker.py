#!/usr/bin/env python
"""GT CEM-MPC oracle for Walker2d-v4 — true MuJoCo dynamics, no learned model.

Objective (minimised by CEM):
    J = -wx*(x_H - x_0)
        + cf*(H - T_alive)
        + wu*sum(||u_t||^2)
        + wz*sum((h_t - h_target)^2)
        + wang*sum(angle_t^2)

  x_H - x_0          forward displacement over the horizon
  T_alive             steps survived before fall (= H if survived)
  H - T_alive         steps wasted → early falls cost far more than late ones
  ||u_t||^2           action regulariser
  (h_t - h_target)^2  height-tracking term (keeps torso near 1.2 m)
  angle_t^2           upright-posture term (penalises forward lean)

Action blocking: CEM optimises n_blocks = ceil(H/block_size) distinct 6-D parameters;
each is repeated block_size times, so the full (H, 6) sequence collapses to a much
smaller search space.

Success criterion: survived the full episode AND mean x-velocity > --success-min-velocity.

Usage
-----
  python experiments/gt_cem_walker.py \\
      --trials 10 --n-steps 500 \\
      --planning-horizon 100 --action-block-size 4 \\
      --cem-population 2000 --cem-elites 100 --cem-iters 7 \\
      --executed-steps 1 \\
      --wx 5.0 --cf 10.0 --wu 1e-3 \\
      --success-min-velocity 0.5 \\
      --render-dir results/walker_gt_cem_frames \\
      --output results/gt_cem_walker.json
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
ACTION_DIM  =  6

# Walker2d-v4 health bounds (must match gymnasium defaults exactly)
HEALTHY_Z_MIN   = 0.8
HEALTHY_Z_MAX   = 2.0
HEALTHY_ANG_MAX = 1.0


# ── env state helpers ─────────────────────────────────────────────────────────

def get_state(env):
    """Return (qpos, qvel) copies — includes x-position in qpos[0]."""
    d = env.unwrapped.data
    return d.qpos.copy(), d.qvel.copy()


def set_state(env, qpos, qvel):
    """Restore full kinematic state and call mj_forward to update derived quantities."""
    env.unwrapped.set_state(qpos, qvel)   # gymnasium calls mj_forward internally


def is_healthy(data):
    """Match gymnasium Walker2d-v4 health condition exactly."""
    z   = float(data.qpos[1])
    ang = float(data.qpos[2])
    return HEALTHY_Z_MIN < z < HEALTHY_Z_MAX and abs(ang) < HEALTHY_ANG_MAX


def failure_reason(data):
    z   = float(data.qpos[1])
    ang = float(data.qpos[2])
    if z <= HEALTHY_Z_MIN:
        return 'low_height'
    if z >= HEALTHY_Z_MAX:
        return 'high_height'
    if abs(ang) >= HEALTHY_ANG_MAX:
        return 'excessive_angle'
    return 'other'


# ── CEM rollout with new objective ────────────────────────────────────────────

def rollout_cost(plan_env, start_qpos, start_qvel, actions, frame_skip,
                 wx, cf, wu, wz=0.0, wang=0.0, height_target=1.2):
    """Evaluate one action sequence (H, 6) under the full posture-aware objective.

    J = -wx*(x_H - x_0)
        + cf*(H - T_alive)
        + wu*sum(||u_t||^2)
        + wz*sum((h_t - height_target)^2)
        + wang*sum(angle_t^2)

    Health is checked AFTER each full frame_skip macro-step, consistent
    with how gymnasium checks termination during eval.
    """
    set_state(plan_env, start_qpos, start_qvel)
    model = plan_env.unwrapped.model
    data  = plan_env.unwrapped.data

    x0             = float(data.qpos[0])
    H              = len(actions)
    t_alive        = 0
    posture_cost   = 0.0

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
        h   = float(data.qpos[1])
        ang = float(data.qpos[2])
        posture_cost += wz * (h - height_target) ** 2 + wang * ang ** 2

    x_H            = float(data.qpos[0])
    action_penalty = float(np.sum(actions[:t_alive + 1] ** 2)) * wu
    cost = (-wx * (x_H - x0)
            + cf * (H - t_alive)
            + action_penalty
            + posture_cost)
    return cost


# ── CEM planner with action blocking ─────────────────────────────────────────

class WalkerGTCEM:
    """GT CEM-MPC planner with action blocking and warm start.

    CEM optimises (n_blocks, ACTION_DIM) in block space.
    Each block is repeated block_size times to form the full (H, 6) sequence.
    Warm start shifts the previous block mean forward by one block after each
    MPC call; reset _prev_blocks = None between trials.
    """

    def __init__(self, horizon, block_size, executed_steps,
                 population, elites, iterations, initial_variance,
                 plan_env, frame_skip, wx, cf, wu,
                 wz=0.0, wang=0.0, height_target=1.2):
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
        self.wz             = float(wz)
        self.wang           = float(wang)
        self.height_target  = float(height_target)
        self._prev_blocks   = None   # (n_blocks, ACTION_DIM) warm-start mean

    # ── helper: expand block mean → full action sequence ──────────────────────
    def _expand(self, blocks):
        """(n_blocks, 6) → (H, 6) by repeating each block block_size times."""
        return np.repeat(blocks, self.block_size, axis=0)[:self.horizon]

    # ── warm-start shift in block space ───────────────────────────────────────
    def _shift_blocks(self, blocks):
        n_exec = max(1, math.ceil(self.executed_steps / self.block_size))
        shifted = np.concatenate([
            blocks[n_exec:],
            np.zeros((n_exec, ACTION_DIM))
        ], axis=0)
        return shifted

    def plan(self, start_qpos, start_qvel):
        """Run CEM; return (executed_steps, ACTION_DIM) actions to execute."""
        if self._prev_blocks is not None:
            mean = self._shift_blocks(self._prev_blocks)
        else:
            mean = np.zeros((self.n_blocks, ACTION_DIM))
        variance = np.full((self.n_blocks, ACTION_DIM), self.initial_var)

        for _ in range(self.iterations):
            noise  = np.random.randn(self.population, self.n_blocks, ACTION_DIM)
            blocks = np.clip(
                mean[None] + np.sqrt(variance)[None] * noise,
                ACTION_LOW, ACTION_HIGH)                         # (pop, n_blocks, 6)

            costs = np.empty(self.population)
            for k in range(self.population):
                actions = self._expand(blocks[k])
                costs[k] = rollout_cost(
                    self.plan_env, start_qpos, start_qvel,
                    actions, self.frame_skip, self.wx, self.cf, self.wu,
                    self.wz, self.wang, self.height_target)

            elite_idx   = np.argsort(costs)[:self.elites]
            elite       = blocks[elite_idx]
            mean        = elite.mean(0)
            variance    = elite.var(0).clip(min=1e-6)

        self._prev_blocks = mean
        actions = self._expand(mean)
        return np.clip(actions[:self.executed_steps], ACTION_LOW, ACTION_HIGH)


# ── trial evaluation ──────────────────────────────────────────────────────────

def run_trial(planner, eval_env, initial_qpos, initial_qvel,
              n_steps, do_render):
    """Execute one MPC trial; return (metrics_dict, frames_list)."""
    set_state(eval_env, initial_qpos, initial_qvel)
    planner._prev_blocks = None   # fresh warm start for each trial

    step         = 0
    x_vels       = []
    frames       = []
    terminated   = False
    truncated    = False
    fail_reason  = 'timeout'

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

    # collect final state for diagnostics
    data            = eval_env.unwrapped.data
    final_height    = float(data.qpos[1])
    final_angle     = float(data.qpos[2])
    final_x         = float(data.qpos[0])
    initial_x       = float(initial_qpos[0])
    forward_disp    = final_x - initial_x
    survived_full   = (step >= n_steps) and not terminated
    avg_vel         = float(np.mean(x_vels)) if x_vels else 0.0

    if terminated:
        fail_reason = failure_reason(data)
    elif truncated:
        fail_reason = 'truncated'
    else:
        fail_reason = 'timeout'   # completed normally

    return {
        'survived_full':    survived_full,
        'steps':            step,
        'forward_disp':     forward_disp,
        'avg_x_velocity':   avg_vel,
        'final_height':     final_height,
        'final_angle':      final_angle,
        'fail_reason':      fail_reason,
        'x_velocities':     x_vels,
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


def save_frame_grid(frames, out_path, every=10, title='GT CEM Walker2d'):
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


def pick_viz_trial(trials_data, frames_list, success_mask, min_vel):
    """Return index of trial to visualise.

    Priority: best successful trial (highest avg_vel).
    Fallback: trial with greatest forward displacement among those that
              survived the longest.
    """
    successes = [i for i, s in enumerate(success_mask) if s]
    if successes:
        return max(successes, key=lambda i: trials_data[i]['avg_x_velocity'])
    # fallback: sort by (steps, forward_disp)
    return max(range(len(trials_data)),
               key=lambda i: (trials_data[i]['steps'],
                              trials_data[i]['forward_disp']))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='GT oracle CEM-MPC for Walker2d-v4 — true MuJoCo dynamics.')
    p.add_argument('--trials',              type=int,   default=10)
    p.add_argument('--n-steps',             type=int,   default=500)
    # planner
    p.add_argument('--planning-horizon',    type=int,   default=100)
    p.add_argument('--action-block-size',   type=int,   default=4,
                   help='Repeat each CEM action this many times (reduces search dim)')
    p.add_argument('--executed-steps',      type=int,   default=1)
    # CEM
    p.add_argument('--cem-population',      type=int,   default=2000)
    p.add_argument('--cem-elites',          type=int,   default=100)
    p.add_argument('--cem-iters',           type=int,   default=7)
    p.add_argument('--cem-initial-variance',type=float, default=0.5)
    # objective
    p.add_argument('--wx',  type=float, default=5.0,
                   help='Weight on forward displacement')
    p.add_argument('--cf',  type=float, default=10.0,
                   help='Cost per wasted step after fall (early falls cost more)')
    p.add_argument('--wu',   type=float, default=1e-3,
                   help='Action regularisation weight')
    p.add_argument('--wz',   type=float, default=0.0,
                   help='Per-step height-tracking weight (penalises deviation from --height-target)')
    p.add_argument('--wang', type=float, default=0.0,
                   help='Per-step torso-angle penalty weight (penalises forward lean)')
    p.add_argument('--height-target', type=float, default=1.2,
                   help='Target torso height in metres for the wz term')
    # success criterion
    p.add_argument('--success-min-velocity', type=float, default=0.5,
                   help='Minimum mean x-velocity (m/s) for a trial to count as success')
    # misc
    p.add_argument('--frame-skip',          type=int,   default=4)
    p.add_argument('--render-dir',          default='',
                   help='Directory to save GIF + frame grid (empty = no render)')
    p.add_argument('--render-every',        type=int,   default=10,
                   help='Subsample every N frames for the PNG grid')
    p.add_argument('--gif-fps',             type=int,   default=30)
    p.add_argument('--seed',                type=int,   default=42)
    p.add_argument('--output',              required=True)
    args = p.parse_args()

    np.random.seed(args.seed)

    plan_env = gym.make('Walker2d-v4')
    eval_env = gym.make('Walker2d-v4',
                        render_mode='rgb_array' if args.render_dir else None)
    plan_env.reset(seed=args.seed)
    eval_env.reset(seed=args.seed)

    n_blocks = math.ceil(args.planning_horizon / args.action_block_size)
    planner = WalkerGTCEM(
        horizon        = args.planning_horizon,
        block_size     = args.action_block_size,
        executed_steps = args.executed_steps,
        population     = args.cem_population,
        elites         = args.cem_elites,
        iterations     = args.cem_iters,
        initial_variance = args.cem_initial_variance,
        plan_env       = plan_env,
        frame_skip     = args.frame_skip,
        wx             = args.wx,
        cf             = args.cf,
        wu             = args.wu,
        wz             = args.wz,
        wang           = args.wang,
        height_target  = args.height_target,
    )
    print(f'[GT-CEM Walker2d] H={args.planning_horizon} block={args.action_block_size} '
          f'n_blocks={n_blocks} K={args.executed_steps} '
          f'pop={args.cem_population} elites={args.cem_elites} iters={args.cem_iters}')
    print(f'  objective: wx={args.wx} cf={args.cf} wu={args.wu} '
          f'wz={args.wz} wang={args.wang} h*={args.height_target}')
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

    # failure breakdown
    reasons = [r['fail_reason'] for r in trials_data if not r['success']]
    if reasons:
        from collections import Counter
        print('Failure breakdown:', dict(Counter(reasons)))

    if args.render_dir:
        viz_idx = pick_viz_trial(trials_data, frames_list, success_mask,
                                 args.success_min_velocity)
        viz_frames = frames_list[viz_idx]
        row = trials_data[viz_idx]
        tag = 'success' if success_mask[viz_idx] else 'best_failed'
        stem  = f'{tag}_trial_{viz_idx:03d}'
        title = (f'GT CEM Walker2d — trial {viz_idx} [{tag}] '
                 f'vel={row["avg_x_velocity"]:.3f} disp={row["forward_disp"]:.2f}m')
        if viz_frames:
            save_gif(viz_frames, Path(args.render_dir) / f'{stem}.gif',
                     fps=args.gif_fps)
            save_frame_grid(viz_frames, Path(args.render_dir) / f'{stem}.png',
                            every=args.render_every, title=title)

    # strip per-step velocity list from JSON (can be large)
    for r in trials_data:
        r.pop('x_velocities', None)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'protocol': {
            'planner':              'gt_cem_mpc',
            'env':                  'Walker2d-v4',
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

