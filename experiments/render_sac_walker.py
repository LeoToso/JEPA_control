#!/usr/bin/env python
"""Render a Walker2d-v4 rollout with the SAC policy and save as GIF.

Usage:
  python experiments/render_sac_walker.py \
      --sac-repo sdpkjc/Walker2d-v4-sac_continuous_action-seed4 \
      --out results/sac_walker.gif \
      --n-steps 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from experiments.walker2d_ppo_utils import download_and_load_sac, load_sac_from_local


def render_rollout(policy, n_steps: int = 500, seed: int = 0) -> list[np.ndarray]:
    import gymnasium as gym
    env = gym.make('Walker2d-v4', render_mode='rgb_array')
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    total_reward = 0.0
    for t in range(n_steps):
        action = policy.act(obs.astype('float32'))
        obs, rew, term, trunc, _ = env.step(action)
        total_reward += rew
        frames.append(env.render())
        if term or trunc:
            print(f'  episode ended at step {t+1}  return={total_reward:.1f}')
            break
    else:
        print(f'  ran {n_steps} steps  return={total_reward:.1f}')
    env.close()
    return frames


def save_gif(frames: list[np.ndarray], path: Path, fps: int = 30) -> None:
    try:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(
            path, save_all=True, append_images=imgs[1:],
            duration=int(1000 / fps), loop=0,
        )
        print(f'[saved] {path}  ({len(frames)} frames @ {fps} fps)')
    except ImportError:
        # Fallback: save as individual PNGs
        png_dir = path.with_suffix('')
        png_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            from PIL import Image
            Image.fromarray(f).save(png_dir / f'{i:04d}.png')
        print(f'[saved] {len(frames)} PNGs in {png_dir} (PIL not available for GIF)')


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--sac-repo', default='sdpkjc/Walker2d-v4-sac_continuous_action-seed4')
    p.add_argument('--sac-ckpt', default=None, help='Local SAC checkpoint (skip download)')
    p.add_argument('--out',      default='results/sac_walker.gif')
    p.add_argument('--n-steps',  type=int, default=500)
    p.add_argument('--fps',      type=int, default=30)
    p.add_argument('--seed',     type=int, default=0)
    p.add_argument('--device',   default='cpu')
    args = p.parse_args()

    print('=== Load SAC policy ===')
    if args.sac_ckpt:
        policy = load_sac_from_local(args.sac_ckpt, device=args.device)
    else:
        policy = download_and_load_sac(args.sac_repo, device=args.device)

    print('=== Render rollout ===')
    frames = render_rollout(policy, n_steps=args.n_steps, seed=args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_gif(frames, out, fps=args.fps)


if __name__ == '__main__':
    main()

