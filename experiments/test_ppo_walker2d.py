#!/usr/bin/env python
"""Quick visual test: load PPO policy, roll out in Walker2d-v4, save GIF.

Usage
-----
  MUJOCO_GL=egl python experiments/test_ppo_walker2d.py \
      --out results/ppo_walker2d_test.gif

  # Or with a local checkpoint:
  MUJOCO_GL=egl python experiments/test_ppo_walker2d.py \
      --ppo-ckpt /path/to/agent.pt \
      --out results/ppo_walker2d_test.gif
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch
import torch.nn.functional as F_
import gymnasium as gym

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from experiments.walker2d_ppo_utils import download_and_load_ppo, load_ppo_from_local


def run_episode(policy, seed: int = 0, max_steps: int = 1000,
                image_size: int = 128) -> tuple[list[np.ndarray], float, int]:
    """Roll out policy; return (frames, total_return, episode_length)."""
    env = gym.make('Walker2d-v4', render_mode='rgb_array')
    obs, _ = env.reset(seed=seed)
    obs = obs.astype(np.float32)

    frames, total_ret, step = [], 0.0, 0
    while step < max_steps:
        frame = np.ascontiguousarray(env.render())   # copy removes negative strides
        if frame.shape[0] != image_size or frame.shape[1] != image_size:
            t = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0)
            t = F_.interpolate(t, (image_size, image_size), mode='bilinear',
                               align_corners=False)
            frame = t[0].permute(1, 2, 0).byte().numpy()
        frames.append(frame)

        action = policy.act(obs)
        obs, rew, term, trunc, _ = env.step(action)
        obs = obs.astype(np.float32)
        total_ret += rew
        step += 1
        if term or trunc:
            break

    env.close()
    return frames, total_ret, step


def save_gif(frames: list[np.ndarray], path: str, fps: int = 30) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=100)
    ax.axis('off')
    fig.subplots_adjust(0, 0, 1, 1)
    im = ax.imshow(frames[0])
    step_txt = ax.text(0.01, 0.97, 't=0', transform=ax.transAxes,
                       color='white', fontsize=9, va='top',
                       bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.5))

    def update(i):
        im.set_data(frames[i])
        step_txt.set_text(f't={i}')
        return [im, step_txt]

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000 // fps, blit=True)
    anim.save(str(out), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f'[saved] {out}  ({len(frames)} frames)')


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--ppo-repo',
                   default='sdpkjc/Walker2d-v4-ppo_fix_continuous_action-seed3')
    p.add_argument('--ppo-ckpt', default=None,
                   help='Local checkpoint (skip HF download)')
    p.add_argument('--device',   default='cpu')
    p.add_argument('--seed',     type=int, default=0)
    p.add_argument('--n-episodes', type=int, default=1)
    p.add_argument('--max-steps', type=int, default=1000)
    p.add_argument('--image-size', type=int, default=128)
    p.add_argument('--fps',      type=int, default=30)
    p.add_argument('--out', default='results/ppo_walker2d_test.gif')
    args = p.parse_args()

    # Load policy
    print('Loading PPO policy …')
    if args.ppo_ckpt:
        policy = load_ppo_from_local(args.ppo_ckpt, device=args.device)
    else:
        policy = download_and_load_ppo(args.ppo_repo, device=args.device)

    returns = []
    all_frames = []
    for ep in range(args.n_episodes):
        print(f'\n--- Episode {ep} (seed={args.seed + ep}) ---')
        frames, ret, length = run_episode(
            policy,
            seed=args.seed + ep,
            max_steps=args.max_steps,
            image_size=args.image_size,
        )
        print(f'  return={ret:.1f}  length={length}')
        returns.append(ret)
        all_frames.extend(frames)

    print(f'\nMean return: {np.mean(returns):.1f}  '
          f'(over {args.n_episodes} episode(s))')

    # Save GIF (all episodes concatenated)
    if all_frames:
        save_gif(all_frames, args.out, fps=args.fps)


if __name__ == '__main__':
    main()
