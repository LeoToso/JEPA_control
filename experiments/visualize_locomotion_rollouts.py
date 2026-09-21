#!/usr/bin/env python
"""Visualize rollouts from a locomotion (Walker2d / Hopper) HDF5 dataset.

Reads the stored RGB observations and renders a grid:
  rows  = episodes
  cols  = uniformly-sampled frames from each episode

Saves the result as a PDF (or PNG if the path ends in .png).

Usage
-----
  python experiments/visualize_locomotion_rollouts.py \\
      --dataset data/walker2d_rnd_64/train.hdf5 \\
      --n-episodes 6 --n-frames 8 \\
      --out results/walker2d_rollouts.pdf

  python experiments/visualize_locomotion_rollouts.py \\
      --dataset data/hopper_rnd_64/train.hdf5 \\
      --n-episodes 6 --n-frames 8 \\
      --out results/hopper_rollouts.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_episodes(hdf5_path: str, max_episodes: int) -> list[dict]:
    episodes = []
    with h5py.File(hdf5_path, 'r') as f:
        grp = f['episodes']
        keys = sorted(grp.keys(), key=int)[:max_episodes]
        for k in keys:
            g = grp[k]
            ep = {
                'observations': g['observations'][:],   # (T+1, h, w, 3) uint8
                'rewards':      g['rewards'][:],         # (T,)
                'terminated':   g['terminated'][:],      # (T,)
            }
            episodes.append(ep)
    return episodes


def pick_frames(obs: np.ndarray, n_frames: int) -> tuple[np.ndarray, list[int]]:
    """Uniformly sample n_frames from the observation sequence."""
    T = len(obs)
    if T <= n_frames:
        idx = list(range(T))
    else:
        idx = np.linspace(0, T - 1, n_frames, dtype=int).tolist()
    return obs[idx], idx


def main():
    p = argparse.ArgumentParser(
        description='Visualize locomotion dataset rollouts as an image grid.')
    p.add_argument('--dataset',     required=True,
                   help='Path to train/val/test.hdf5')
    p.add_argument('--n-episodes',  type=int, default=6,
                   help='Number of episodes to display (rows)')
    p.add_argument('--n-frames',    type=int, default=8,
                   help='Number of frames per episode (columns)')
    p.add_argument('--out',         required=True,
                   help='Output file (.pdf or .png)')
    p.add_argument('--title',       default=None,
                   help='Optional figure title (default: dataset filename)')
    args = p.parse_args()

    print(f'Loading {args.n_episodes} episodes from {args.dataset} …')
    episodes = load_episodes(args.dataset, args.n_episodes)
    n_eps    = len(episodes)
    n_frames = args.n_frames

    title = args.title or Path(args.dataset).parent.name

    fig = plt.figure(figsize=(1.6 * n_frames, 1.8 * n_eps + 0.5))
    fig.suptitle(title, fontsize=11, y=0.99)
    gs = gridspec.GridSpec(n_eps, n_frames, figure=fig, hspace=0.08, wspace=0.04)

    for row, ep in enumerate(episodes):
        obs    = ep['observations']   # (T+1, h, w, 3)
        rews   = ep['rewards']        # (T,)
        terms  = ep['terminated']     # (T,)
        T      = len(rews)
        total_r = float(rews.sum())
        n_resets = int(terms.sum())

        frames, frame_idx = pick_frames(obs, n_frames)

        for col, (frame, idx) in enumerate(zip(frames, frame_idx)):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(frame, interpolation='nearest')
            ax.axis('off')

            if col == 0:
                # Left label: episode stats
                ax.set_ylabel(
                    f'ep {row}  |  T={T}  R={total_r:.1f}  resets={n_resets}',
                    fontsize=6, rotation=0, labelpad=60, va='center')

            # Top label: step index + terminated marker
            is_term = idx < T and terms[idx]
            step_label = f't={idx}' + (' ✗' if is_term else '')
            ax.set_title(step_label, fontsize=5, pad=1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'[done] → {out_path}')


if __name__ == '__main__':
    main()
