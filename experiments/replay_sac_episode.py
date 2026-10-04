#!/usr/bin/env python
"""Open-loop replay of a SAC episode from the HDF5 dataset.

Executes the stored actions directly in Walker2d-v4 without any planning.
This is the ground truth for "does SAC work on this model?"

Usage
-----
python experiments/replay_sac_episode.py \
    --hdf5-dir /path/to/walker2d_mixed_sac_fs5_64 \
    --episode 0 \
    --render-dir results/sac_replay_frames
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('MUJOCO_GL', 'egl')

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hdf5-dir',  required=True)
    p.add_argument('--episode',   type=int, default=0)
    p.add_argument('--n-episodes', type=int, default=5, help='Replay this many episodes')
    p.add_argument('--render-dir', default='')
    p.add_argument('--gif-fps',   type=int, default=30)
    args = p.parse_args()

    import gymnasium as gym
    train_hdf5 = Path(args.hdf5_dir) / 'train.hdf5'
    if not train_hdf5.exists():
        import glob
        files = sorted(glob.glob(str(Path(args.hdf5_dir) / '*.hdf5')))
        if not files:
            raise FileNotFoundError(f'No HDF5 files in {args.hdf5_dir}')
        train_hdf5 = files[0]

    hf = h5py.File(train_hdf5, 'r')
    ep_group = hf['episodes'] if 'episodes' in hf else hf
    eps = sorted(ep_group.keys(), key=lambda k: int(k))
    print(f'Dataset: {train_hdf5.name}  episodes={len(eps)}')

    env = gym.make('Walker2d-v4', render_mode='rgb_array' if args.render_dir else None)

    for ep_offset in range(args.n_episodes):
        ep_idx = (args.episode + ep_offset) % len(eps)
        ep_key = eps[ep_idx]
        actions = np.array(ep_group[ep_key]['actions'])  # (T, 6)
        T = len(actions)

        obs, info = env.reset(seed=ep_idx * 100)
        x_vels = []
        frames = []
        terminated = False

        for t, a in enumerate(actions):
            obs, reward, term, trunc, info = env.step(a)
            x_vels.append(float(info.get('x_velocity', 0.0)))
            if args.render_dir:
                frames.append(env.render())
            terminated = term or trunc
            if terminated:
                break

        avg_vel = float(np.mean(x_vels)) if x_vels else 0.0
        steps   = len(x_vels)
        fwd     = float(env.unwrapped.data.qpos[0])
        h       = float(env.unwrapped.data.qpos[1])
        print(f'  ep={ep_key:>4s}  steps={steps:4d}/{T}  '
              f'avg_vel={avg_vel:+.3f}  disp={fwd:+.2f}m  h={h:.3f}  '
              f'survived={not terminated}')

        if args.render_dir and frames:
            out_dir = Path(args.render_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            gif_path = out_dir / f'sac_ep{ep_key}.gif'
            try:
                import imageio
                try:
                    imageio.mimsave(str(gif_path), frames, fps=args.gif_fps)
                except TypeError:
                    imageio.mimsave(str(gif_path), frames,
                                    duration=int(1000 / args.gif_fps))
                print(f'    [gif saved] {gif_path}')
            except ImportError:
                pass

    hf.close()
    env.close()


if __name__ == '__main__':
    main()
