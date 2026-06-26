"""Visualize sample frames from an HDF5 dataset.

Shows a grid of observations with their corresponding states,
giving a quick sanity check of the data before training.

Usage:
    python experiments/visualize_dataset_frames.py \\
        --data data/cartpole_random_seed42.h5 \\
        --out results/dataset_frames.png

    python experiments/visualize_dataset_frames.py \\
        --data data/cartpole_random_seed42.h5 \\
        --out results/dataset_frames.png --n-rows 4 --n-cols 6
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(
        description='Visualize sample frames from an h5 dataset')
    p.add_argument('--data',      required=True, help='Path to .h5 dataset')
    p.add_argument('--out',       required=True, help='Output PNG path')
    p.add_argument('--split',     default='train', choices=['train', 'val', 'test'])
    p.add_argument('--n-rows',    type=int, default=4)
    p.add_argument('--n-cols',    type=int, default=6)
    p.add_argument('--seed',      type=int, default=0)
    p.add_argument('--sequential', action='store_true',
                   help='Show sequential frames from one episode instead of random samples')
    args = p.parse_args()

    from data.dataset import load_dataset
    print(f'[data] Loading {args.data}')
    data = load_dataset(args.data)

    idx = data['splits'][args.split]
    obs_all    = data['obs'][idx]
    states_all = data['states'][idx]
    actions_all = data['actions'][idx]
    ep_ids     = data['episode_ids'][idx] if 'episode_ids' in data else None

    N = len(obs_all)
    n_frames = args.n_rows * args.n_cols
    rng = np.random.default_rng(args.seed)

    if args.sequential and ep_ids is not None:
        unique_eps = np.unique(ep_ids)
        ep = rng.choice(unique_eps)
        ep_mask = np.where(ep_ids == ep)[0]
        start = ep_mask[0]
        sel = ep_mask[:n_frames]
        title_extra = f' — Episode {ep} (sequential)'
    else:
        sel = rng.choice(N, size=min(n_frames, N), replace=False)
        sel.sort()
        title_extra = ' — random samples'

    fig, axes = plt.subplots(args.n_rows, args.n_cols,
                              figsize=(2.5 * args.n_cols, 3.0 * args.n_rows))
    axes = axes.flatten() if args.n_rows * args.n_cols > 1 else [axes]

    dataset_name = Path(args.data).stem
    fig.suptitle(f'{dataset_name} ({args.split}){title_extra}',
                 fontsize=12, fontweight='bold')

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= len(sel):
            ax.axis('off')
            continue
        i = sel[ax_idx]
        img = obs_all[i]  # (H, W, 3) uint8
        state = states_all[i]  # (4,) [x, xdot, theta, thetadot]
        action = actions_all[i] if i < len(actions_all) else None

        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

        theta_deg = np.degrees(state[2])
        label = f'θ={theta_deg:.1f}°'
        if action is not None:
            label += f'  u={action[0]:.1f}N'
        ax.set_title(label, fontsize=8)

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'[viz] Saved {len(sel)} frames → {out_path}')
    print(f'[viz] Dataset: {N} transitions in {args.split} split')
    if 'action_scale' in data:
        print(f'[viz] Action scale: {data["action_scale"]}')
    if 'state_mean' in data:
        print(f'[viz] State mean: {np.round(data["state_mean"], 4)}')
        print(f'[viz] State std:  {np.round(data["state_std"], 4)}')

    theta_all = states_all[:, 2]
    print(f'[viz] θ range: [{np.degrees(theta_all.min()):.1f}°, '
          f'{np.degrees(theta_all.max()):.1f}°]')
    print(f'[viz] Action range: [{actions_all.min():.2f}, {actions_all.max():.2f}]')


if __name__ == '__main__':
    main()
