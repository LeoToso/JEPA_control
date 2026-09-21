#!/usr/bin/env python
"""Replay a Walker2d dataset episode in the GT environment, holding each
stored action for `--frame-skip` gym steps, and render the frames.

This visualises what "frame_skip=5" means physically: the SAC policy issues
one torque command that is held constant for 5 consecutive gym steps
(= 5 × 4 = 20 MuJoCo micro-steps).  The resulting gait should be stable
and upright, confirming that the SAC actions in walker2d_sac_fs5_64 are
high-quality even under holding.

Usage
-----
  MUJOCO_GL=egl python experiments/replay_walker2d_dataset.py \\
      --dataset data/walker2d_sac_fs5_64 \\
      --split train --episode 0 \\
      --frame-skip 5 \\
      --render-every 1 \\
      --out-png  results/replay_fs5_episode0.png \\
      --out-gif  results/replay_fs5_episode0.gif
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import h5py
import numpy as np
import gymnasium as gym


def replay_episode(actions: np.ndarray, initial_state: np.ndarray | None,
                   frame_skip: int, render_every: int, seed: int):
    """Apply actions to a Walker2d env, holding each for frame_skip gym steps.

    frame_skip=1  → each stored action applied once (exact replay of fs=1 data)
    frame_skip=N  → hold each action for N consecutive gym steps

    If initial_state (17-D gym obs = qpos[1:9] + qvel) is provided, the env
    is set to that exact state before replay.
    qpos[0] (x position) is set to 0 — it has no effect on dynamics.

    Returns (frames, returns, heights, angles).
    """
    env = gym.make('Walker2d-v4', render_mode='rgb_array')
    env.reset(seed=seed)

    if initial_state is not None:
        # stored state = qpos[1:9] (8,) + qvel[0:9] (9,) — x excluded from obs
        qpos     = np.zeros(9, dtype=np.float64)
        qpos[1:] = initial_state[:8].astype(np.float64)
        qvel     = initial_state[8:].astype(np.float64)
        env.unwrapped.set_state(qpos, qvel)

    frames  = []
    returns = []
    heights = []
    angles  = []

    total_reward = 0.0
    terminated   = False
    step_idx     = 0

    for macro_idx, a in enumerate(actions):
        if terminated:
            break
        a = np.clip(a, -1.0, 1.0)

        for micro in range(frame_skip):
            obs, reward, terminated, truncated, info = env.step(a)
            total_reward += reward
            if terminated or truncated:
                terminated = True
                break

        step_idx += 1
        data = env.unwrapped.data
        h    = float(data.qpos[1])
        ang  = float(data.qpos[2])
        heights.append(h)
        angles.append(ang)
        returns.append(total_reward)

        if macro_idx % render_every == 0:
            frames.append(env.render())

    env.close()
    return frames, returns, heights, angles


def save_frame_grid(frames, out_path, every=1,
                    title='Walker2d — action replay (frame_skip=5)'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    subset = frames[::every]
    n      = len(subset)
    if n == 0:
        print('[warn] no frames to save')
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


def save_gif(frames, out_path, fps=15):
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
    print(f'[gif  saved] {out}')


def save_stats_plot(heights, angles, returns, out_path, frame_skip):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t = np.arange(len(heights)) * frame_skip   # micro-steps on x-axis

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t, heights, color='steelblue', lw=1.5)
    axes[0].axhline(1.35, ls='--', color='gray', lw=1, label='standing h*=1.35')
    axes[0].axhline(0.8,  ls=':',  color='red',  lw=1, label='min healthy z')
    axes[0].set_ylabel('Torso height (m)')
    axes[0].legend(fontsize=8)

    axes[1].plot(t, np.degrees(angles), color='darkorange', lw=1.5)
    axes[1].axhline(0, ls='--', color='gray', lw=1)
    axes[1].set_ylabel('Torso angle (deg)')

    axes[2].plot(t, returns, color='seagreen', lw=1.5)
    axes[2].set_ylabel('Cumulative return')
    axes[2].set_xlabel(f'Micro-steps (frame_skip={frame_skip} per action)')

    fig.suptitle(f'Walker2d SAC replay — action held {frame_skip} gym steps each',
                 fontsize=10)
    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[stats saved] {out}')


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataset',      default='data/walker2d_sac_fs5_64')
    p.add_argument('--split',        default='train')
    p.add_argument('--episode',      type=int, default=0,
                   help='Episode index to replay')
    p.add_argument('--frame-skip',   type=int, default=5,
                   help='How many gym steps to hold each stored action')
    p.add_argument('--render-every', type=int, default=1,
                   help='Render every N macro-steps (1 = every step)')
    p.add_argument('--grid-every',   type=int, default=5,
                   help='Show every N-th rendered frame in the grid')
    p.add_argument('--seed',         type=int, default=42)
    p.add_argument('--out-png',      default='results/replay_walker2d_fs5.png')
    p.add_argument('--out-gif',      default='results/replay_walker2d_fs5.gif')
    p.add_argument('--out-stats',    default='results/replay_walker2d_fs5_stats.png')
    p.add_argument('--fps',          type=int, default=15)
    args = p.parse_args()

    hdf5 = Path(args.dataset) / f'{args.split}.hdf5'
    print(f'Loading episode {args.episode} from {hdf5}')
    with h5py.File(hdf5, 'r') as f:
        ep            = f[f'episodes/{args.episode}']
        actions       = ep['actions'][:]          # (T, 6)
        rewards       = ep['rewards'][:]          # (T,)
        initial_state = ep['states'][0]           # (17,) — qpos[1:9] + qvel
        print(f'  Stored: {len(actions)} macro-steps, '
              f'dataset return = {rewards.sum():.1f}')
        print(f'  Initial state: h={initial_state[0]:.3f} m  '
              f'ang={np.degrees(initial_state[1]):.1f}°')

    print(f'Replaying with frame_skip={args.frame_skip} …')
    frames, returns, heights, angles = replay_episode(
        actions, initial_state, args.frame_skip, args.render_every, args.seed)

    alive    = sum(1 for h in heights if h > 0.8)
    mean_h   = float(np.mean(heights))
    ep_return = returns[-1] if returns else 0.0

    print(f'  Replay return : {ep_return:.1f}')
    print(f'  Macro-steps   : {len(heights)}  (alive={alive})')
    print(f'  Mean height   : {mean_h:.3f} m')
    print(f'  Final height  : {heights[-1]:.3f} m  angle: {np.degrees(angles[-1]):.1f}°')

    title = (f'Walker2d SAC — episode {args.episode} replayed '
             f'(action held {args.frame_skip}× per step)  '
             f'return={ep_return:.0f}  mean_h={mean_h:.2f}m')

    save_frame_grid(frames, args.out_png, every=args.grid_every, title=title)
    save_gif(frames, args.out_gif, fps=args.fps)
    save_stats_plot(heights, angles, returns, args.out_stats, args.frame_skip)


if __name__ == '__main__':
    main()
