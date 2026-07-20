"""Visualize frames from a discrete CartPole HDF5 dataset (per-episode format).

Shows a grid of sampled frames with state annotations, and optionally a side-by-side
comparison of two datasets (e.g. original 224x224 vs resized 64x64).

Usage — single dataset:
    python experiments/visualize_discrete_dataset.py \
        --data data/cartpole_visual_64 \
        --out results/frames_64.png

Side-by-side comparison (original vs resized):
    python experiments/visualize_discrete_dataset.py \
        --data data/cartpole_visual \
        --compare data/cartpole_visual_64 \
        --out results/compare_resize.png

Sequential frames from one episode:
    python experiments/visualize_discrete_dataset.py \
        --data data/cartpole_visual_64 \
        --out results/episode.png --sequential --n-cols 10
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _load_episode_sample(hdf5_path: str, n: int, seed: int = 0,
                         sequential: bool = False) -> list[dict]:
    """Return up to n frames (obs + state + action) from random episodes."""
    rng = np.random.default_rng(seed)
    frames = []
    with h5py.File(hdf5_path, 'r') as f:
        ep_grp = f['episodes']
        ep_keys = sorted(ep_grp.keys(), key=int)
        if sequential:
            ep_key = ep_keys[rng.integers(len(ep_keys))]
            ep = ep_grp[ep_key]
            obs    = ep['observations'][:]   # (T+1, H, W, C)
            states = ep['states'][:]         # (T+1, 4)
            acts   = ep['actions'][:]        # (T,)
            step   = max(1, len(acts) // n)
            for t in range(0, min(n * step, len(acts)), step):
                frames.append({'obs': obs[t], 'state': states[t],
                               'action': int(acts[t]), 'ep': int(ep_key), 't': t})
        else:
            indices = rng.integers(0, len(ep_keys), size=n)
            for ep_key in [ep_keys[i] for i in indices]:
                ep = ep_grp[ep_key]
                T  = len(ep['actions'])
                t  = int(rng.integers(T))
                frames.append({
                    'obs':    ep['observations'][t],
                    'state':  ep['states'][t],
                    'action': int(ep['actions'][t]),
                    'ep':     int(ep_key),
                    't':      t,
                })
    return frames[:n]


def _draw_grid(frames: list[dict], title: str, axes, n_rows: int, n_cols: int):
    for ax in axes.flat:
        ax.axis('off')
    for idx, fr in enumerate(frames):
        if idx >= n_rows * n_cols:
            break
        ax  = axes[idx // n_cols, idx % n_cols]
        obs = fr['obs']
        ax.imshow(obs if obs.ndim == 3 else obs[..., 0], cmap='gray' if obs.ndim == 2 else None)
        s = fr['state']
        ax.set_title(
            f"ep{fr['ep']} t{fr['t']}\na={fr['action']}  θ={s[2]:.2f}",
            fontsize=6, pad=2)
        ax.axis('off')
    axes[0, 0].figure.suptitle(title, fontsize=9, y=1.01)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',       required=True, help='HDF5 dataset directory')
    p.add_argument('--compare',    default=None,  help='Second dataset for side-by-side')
    p.add_argument('--split',      default='train', choices=['train', 'val', 'test'])
    p.add_argument('--out',        default='results/frames.png')
    p.add_argument('--n-rows',     type=int, default=3)
    p.add_argument('--n-cols',     type=int, default=6)
    p.add_argument('--seed',       type=int, default=0)
    p.add_argument('--sequential', action='store_true',
                   help='Show sequential frames from a single episode')
    args = p.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n = args.n_rows * args.n_cols
    src_hdf5 = str(Path(args.data) / f'{args.split}.hdf5')

    if args.compare is None:
        frames = _load_episode_sample(src_hdf5, n, args.seed, args.sequential)
        h, w   = frames[0]['obs'].shape[:2]
        fig, axes = plt.subplots(args.n_rows, args.n_cols,
                                 figsize=(args.n_cols * 1.4, args.n_rows * 1.6))
        _draw_grid(frames, f'{args.data}  ({args.split})  {h}×{w}', axes,
                   args.n_rows, args.n_cols)
    else:
        cmp_hdf5 = str(Path(args.compare) / f'{args.split}.hdf5')
        frames_a = _load_episode_sample(src_hdf5, n, args.seed, args.sequential)
        frames_b = _load_episode_sample(cmp_hdf5, n, args.seed, args.sequential)
        h_a, w_a = frames_a[0]['obs'].shape[:2]
        h_b, w_b = frames_b[0]['obs'].shape[:2]
        fig, (top, bot) = plt.subplots(
            2 * args.n_rows, args.n_cols,
            figsize=(args.n_cols * 1.4, 2 * args.n_rows * 1.6))
        top_axes = top.reshape(args.n_rows, args.n_cols) if hasattr(top, 'reshape') else \
                   np.array(fig.axes[:args.n_rows * args.n_cols]).reshape(args.n_rows, args.n_cols)
        # Simpler: just use gridspec
        plt.close(fig)
        fig, all_axes = plt.subplots(
            2 * args.n_rows, args.n_cols,
            figsize=(args.n_cols * 1.4, 2 * args.n_rows * 1.6 + 0.5))
        axes_a = all_axes[:args.n_rows]
        axes_b = all_axes[args.n_rows:]
        _draw_grid(frames_a, f'{args.data}  {h_a}×{w_a}', axes_a, args.n_rows, args.n_cols)
        _draw_grid(frames_b, f'{args.compare}  {h_b}×{w_b}', axes_b, args.n_rows, args.n_cols)
        fig.suptitle(f'Original vs Resized  ({args.split} split)', fontsize=10, y=1.005)

    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches='tight')
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()
