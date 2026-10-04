#!/usr/bin/env python
"""Gradient-based planning (SPSA) with GROUND-TRUTH Walker2d-v4 dynamics (oracle baseline).

MuJoCo is not differentiable, so gradients are approximated with SPSA
(Simultaneous Perturbation Stochastic Approximation): each planning step
performs two rollouts with ±ε random perturbations and estimates the gradient
from the cost difference. Adam then steps on the action sequence.

Cost: negative mean x-velocity across the horizon (maximise walking speed).

Visualization: if --render-dir is given, the best trial's frames are saved
as a PNG grid (one frame every --render-every steps).

Usage
-----
  python experiments/gd_gt_walker.py \\
      --trials 5 --n-steps 500 \\
      --planning-horizon 10 --executed-steps 1 \\
      --gd-steps 20 --lr 0.05 --spsa-eps 0.1 --action-noise 0.02 \\
      --render-dir results/walker_gt_gd_frames \\
      --output results/gd_gt_walker.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
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


# ── env state helpers ─────────────────────────────────────────────────────────

def get_state(env):
    d = env.unwrapped.data
    return d.qpos.copy(), d.qvel.copy()


def set_state(env, qpos, qvel):
    env.unwrapped.set_state(qpos, qvel)


# ── single sequence rollout (returns scalar cost) ─────────────────────────────

def simulate_cost(plan_env, start_qpos, start_qvel, actions, frame_skip=4,
                  survival_penalty=0.0, height_target=1.2,
                  alpha_height=1.0, alpha_angle=0.5, height_threshold=1.0):
    """Roll out action sequence.

    cost = -mean(x_vel * upright_mask)          # only credit velocity while standing
           + alpha_height * mean((height - height_target)^2)
           + alpha_angle  * mean(trunk_angle^2)
           + survival_penalty  if the rollout terminated early

    upright_mask = 1 when height > height_threshold, else 0.
    """
    set_state(plan_env, start_qpos, start_qvel)
    model = plan_env.unwrapped.model
    data  = plan_env.unwrapped.data
    x_velocities, height_costs, angle_costs = [], [], []
    done = False

    for a in actions:
        a_clipped = np.clip(a, ACTION_LOW, ACTION_HIGH)
        if HAS_MUJOCO:
            data.ctrl[:] = a_clipped
            for _ in range(frame_skip):
                mujoco.mj_step(model, data)
                if data.qpos[1] < 0.8 or data.qpos[1] > 2.0 or abs(data.qpos[2]) > 1.0:
                    done = True
                    break
        else:
            plan_env.unwrapped.data.ctrl[:] = a_clipped
            plan_env.step(a_clipped)
        upright = float(data.qpos[1]) > height_threshold
        x_velocities.append(float(data.qvel[0]) if upright else 0.0)
        height_costs.append(float((data.qpos[1] - height_target) ** 2))
        angle_costs.append(float(data.qpos[2] ** 2))
        if done:
            break

    if not x_velocities:
        return survival_penalty
    cost  = -float(np.mean(x_velocities))
    cost += alpha_height * float(np.mean(height_costs))
    cost += alpha_angle  * float(np.mean(angle_costs))
    if done:
        cost += survival_penalty
    return cost


# ── SPSA gradient estimate ────────────────────────────────────────────────────

def spsa_gradient(plan_env, start_qpos, start_qvel, actions, eps, frame_skip,
                  survival_penalty=0.0, height_target=1.2,
                  alpha_height=1.0, alpha_angle=0.5, height_threshold=1.0):
    """SPSA gradient of cost w.r.t. actions."""
    delta   = (np.random.randint(0, 2, size=actions.shape) * 2 - 1).astype(np.float64)
    u_plus  = np.clip(actions + eps * delta, ACTION_LOW, ACTION_HIGH)
    u_minus = np.clip(actions - eps * delta, ACTION_LOW, ACTION_HIGH)
    kw = dict(frame_skip=frame_skip, survival_penalty=survival_penalty,
              height_target=height_target, alpha_height=alpha_height,
              alpha_angle=alpha_angle, height_threshold=height_threshold)
    c_plus  = simulate_cost(plan_env, start_qpos, start_qvel, u_plus,  **kw)
    c_minus = simulate_cost(plan_env, start_qpos, start_qvel, u_minus, **kw)
    return (c_plus - c_minus) / (2.0 * eps * delta + 1e-12)


# ── Adam state ────────────────────────────────────────────────────────────────

class AdamState:
    def __init__(self, shape, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.m     = np.zeros(shape)
        self.v     = np.zeros(shape)
        self.t     = 0

    def step(self, grad):
        self.t  += 1
        self.m   = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v   = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat    = self.m / (1 - self.beta1 ** self.t)
        v_hat    = self.v / (1 - self.beta2 ** self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ── GBP planner ───────────────────────────────────────────────────────────────

class WalkerGTGDPlanner:
    """SPSA gradient-based planner using ground-truth Walker2d-v4 dynamics."""

    def __init__(self, horizon, executed_steps, gd_steps, lr,
                 spsa_eps, action_noise, warm_start, n_restarts,
                 plan_env, frame_skip, survival_penalty=0.0,
                 height_target=1.2, alpha_height=1.0, alpha_angle=0.5,
                 height_threshold=1.0):
        self.horizon           = int(horizon)
        self.executed_steps    = min(int(executed_steps), self.horizon)
        self.gd_steps          = int(gd_steps)
        self.lr                = float(lr)
        self.spsa_eps          = float(spsa_eps)
        self.action_noise      = float(action_noise)
        self.warm_start        = warm_start
        self.n_restarts        = max(1, int(n_restarts))
        self.plan_env          = plan_env
        self.frame_skip        = int(frame_skip)
        self.survival_penalty  = float(survival_penalty)
        self.height_target     = float(height_target)
        self.alpha_height      = float(alpha_height)
        self.alpha_angle       = float(alpha_angle)
        self.height_threshold  = float(height_threshold)
        self._prev_u           = None

    @property
    def _cost_kw(self):
        return dict(frame_skip=self.frame_skip,
                    survival_penalty=self.survival_penalty,
                    height_target=self.height_target,
                    alpha_height=self.alpha_height,
                    alpha_angle=self.alpha_angle,
                    height_threshold=self.height_threshold)

    def _run_one(self, start_qpos, start_qvel, init_u):
        """GD from one initialisation; return (u_clamped, terminal_cost)."""
        u    = init_u.copy()
        adam = AdamState((self.horizon, ACTION_DIM), self.lr)

        for _ in range(self.gd_steps):
            grad = spsa_gradient(self.plan_env, start_qpos, start_qvel,
                                 u, self.spsa_eps, **self._cost_kw)
            u   -= adam.step(grad)
            u   += np.random.randn(*u.shape) * self.action_noise
            u    = np.clip(u, ACTION_LOW, ACTION_HIGH)

        cost = simulate_cost(self.plan_env, start_qpos, start_qvel,
                             u, **self._cost_kw)
        return u, cost

    def plan(self, start_qpos, start_qvel):
        """Return (executed_steps, action_dim) action array."""
        inits = []
        if self.warm_start and self._prev_u is not None:
            shifted = np.concatenate([
                self._prev_u[self.executed_steps:],
                np.zeros((self.executed_steps, ACTION_DIM))
            ], axis=0)
            inits.append(shifted)
        for _ in range(self.n_restarts - len(inits)):
            inits.append(np.random.randn(self.horizon, ACTION_DIM) * 0.3)

        best_u, best_cost = None, float('inf')
        for init_u in inits:
            u, cost = self._run_one(start_qpos, start_qvel, init_u)
            if cost < best_cost:
                best_cost = cost
                best_u    = u

        self._prev_u = best_u
        return np.clip(best_u[:self.executed_steps], ACTION_LOW, ACTION_HIGH)


# ── trial evaluation ──────────────────────────────────────────────────────────

def gd_walker_trial(planner, eval_env, initial_qpos, initial_qvel,
                    n_steps, do_render=False):
    set_state(eval_env, initial_qpos, initial_qvel)
    planner._prev_u = None
    step   = 0
    x_vels = []
    frames = []
    done   = False

    while step < n_steps and not done:
        qpos, qvel = get_state(eval_env)
        sequence   = planner.plan(qpos, qvel)

        for a in sequence:
            if step >= n_steps or done:
                break
            obs, reward, terminated, truncated, info = eval_env.step(a)
            step += 1
            x_vels.append(float(info.get('x_velocity', 0.0)))
            done = terminated or truncated
            if do_render:
                frames.append(eval_env.render())

    avg_vel = float(np.mean(x_vels)) if x_vels else 0.0
    return {
        'avg_x_velocity': avg_vel,
        'total_steps':    step,
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


def save_frame_grid(frames, out_path, every=10, title='GT GBP Walker2d'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    subset = frames[::every]
    n      = len(subset)
    if n == 0:
        return
    cols = min(8, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 2.5, rows * 2.5))
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


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='GT oracle GBP (SPSA) for Walker2d-v4 — true MuJoCo dynamics.')
    p.add_argument('--trials',             type=int,   default=5)
    p.add_argument('--n-steps',            type=int,   default=500)
    p.add_argument('--planning-horizon',   type=int,   default=20)
    p.add_argument('--executed-steps',     type=int,   default=3)
    p.add_argument('--gd-steps',           type=int,   default=50)
    p.add_argument('--lr',                 type=float, default=0.05)
    p.add_argument('--spsa-eps',           type=float, default=0.1)
    p.add_argument('--action-noise',       type=float, default=0.02)
    p.add_argument('--n-restarts',         type=int,   default=1)
    p.add_argument('--no-warm-start',      action='store_true')
    p.add_argument('--survival-penalty',   type=float, default=5.0,
                   help='Added to cost when rollout terminates early (falling)')
    p.add_argument('--height-target',      type=float, default=1.2,
                   help='Target torso height for posture cost')
    p.add_argument('--alpha-height',       type=float, default=1.0,
                   help='Weight for height deviation penalty')
    p.add_argument('--alpha-angle',        type=float, default=0.5,
                   help='Weight for trunk angle penalty')
    p.add_argument('--height-threshold',   type=float, default=1.0,
                   help='Min height to count x-velocity toward cost (blocks fall exploit)')
    p.add_argument('--frame-skip',         type=int,   default=4)
    p.add_argument('--render-dir',         default='',
                   help='Directory to save GIF + frame grid (empty = no render)')
    p.add_argument('--render-every',       type=int,   default=10,
                   help='Frame subsampling for the PNG grid (GIF uses every frame)')
    p.add_argument('--gif-fps',            type=int,   default=30)
    p.add_argument('--seed',               type=int,   default=42)
    p.add_argument('--output',             required=True)
    args = p.parse_args()

    np.random.seed(args.seed)

    plan_env = gym.make('Walker2d-v4')
    eval_env = gym.make('Walker2d-v4',
                        render_mode='rgb_array' if args.render_dir else None)

    plan_env.reset(seed=args.seed)
    eval_env.reset(seed=args.seed)

    planner = WalkerGTGDPlanner(
        horizon           = args.planning_horizon,
        executed_steps    = args.executed_steps,
        gd_steps          = args.gd_steps,
        lr                = args.lr,
        spsa_eps          = args.spsa_eps,
        action_noise      = args.action_noise,
        warm_start        = not args.no_warm_start,
        n_restarts        = args.n_restarts,
        plan_env          = plan_env,
        frame_skip        = args.frame_skip,
        survival_penalty  = args.survival_penalty,
        height_target     = args.height_target,
        alpha_height      = args.alpha_height,
        alpha_angle       = args.alpha_angle,
        height_threshold  = args.height_threshold,
    )
    print(f'[GT-GBP Walker2d] H={planner.horizon} K={planner.executed_steps} '
          f'gd_steps={planner.gd_steps} lr={planner.lr} '
          f'spsa_eps={planner.spsa_eps} noise={planner.action_noise} '
          f'restarts={planner.n_restarts} frame_skip={args.frame_skip}')

    trials_data    = []
    best_vel       = -float('inf')
    best_frames    = []
    best_trial_idx = -1

    for i in range(args.trials):
        np.random.seed(args.seed + i)
        obs, _ = eval_env.reset(seed=args.seed + i)
        initial_qpos, initial_qvel = get_state(eval_env)
        plan_env.reset(seed=args.seed + i)
        set_state(plan_env, initial_qpos, initial_qvel)

        print(f'[trial {i:03d}] ', end='', flush=True)
        t0 = time.time()

        do_render = bool(args.render_dir)
        row, frames = gd_walker_trial(
            planner, eval_env, initial_qpos, initial_qvel,
            args.n_steps, do_render)

        elapsed = time.time() - t0
        trials_data.append(row)
        print(f'avg_vel={row["avg_x_velocity"]:.3f}  steps={row["total_steps"]}  t={elapsed:.1f}s')

        if row['avg_x_velocity'] > best_vel:
            best_vel       = row['avg_x_velocity']
            best_frames    = frames
            best_trial_idx = i

    mean_vel = float(np.mean([r['avg_x_velocity'] for r in trials_data]))
    print(f'\n[GT-GBP Walker2d] mean avg_x_velocity: {mean_vel:.3f} m/s '
          f'(best={best_vel:.3f}, trial {best_trial_idx})')

    if args.render_dir and best_frames:
        stem  = f'best_trial_{best_trial_idx:03d}'
        title = f'GT GBP Walker2d — trial {best_trial_idx} (avg_vel={best_vel:.3f})'
        save_gif(best_frames,
                 Path(args.render_dir) / f'{stem}.gif',
                 fps=args.gif_fps)
        save_frame_grid(best_frames,
                        Path(args.render_dir) / f'{stem}.png',
                        every=args.render_every,
                        title=title)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'protocol': {
            'planner':          'gt_spsa_gbp',
            'env':              'Walker2d-v4',
            'gradient_method':  'spsa',
            'n_steps':          args.n_steps,
            'planning_horizon': args.planning_horizon,
            'executed_steps':   args.executed_steps,
            'gd_steps':         args.gd_steps,
            'lr':               args.lr,
            'spsa_eps':         args.spsa_eps,
            'action_noise':     args.action_noise,
            'n_restarts':       args.n_restarts,
            'warm_start':       not args.no_warm_start,
            'frame_skip':       args.frame_skip,
            'seed':             args.seed,
            'n_trials':         args.trials,
        },
        'mean_avg_x_velocity': mean_vel,
        'best_avg_x_velocity': best_vel,
        'best_trial_idx':      best_trial_idx,
        'trials':              trials_data,
    }, indent=2))
    print(f'[done] {out}')

    plan_env.close()
    eval_env.close()


if __name__ == '__main__':
    main()

