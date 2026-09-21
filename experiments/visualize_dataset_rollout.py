#!/usr/bin/env python
"""Visualize a dataset rollout and the exact input fed to the ViT encoder.

Usage
-----
  python experiments/visualize_dataset_rollout.py \
      --data data/pointmaze_u_fs5_ma_64 \
      --traj 0 --horizon 8 \
      --out results/rollout_vis.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from data.dataset import make_discrete_dataloaders


def to_uint8(t):
    """Float [0,1] or uint8 tensor → HWC numpy uint8."""
    arr = t.cpu().numpy()
    if arr.max() <= 1.0 + 1e-3:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 9):
        arr = arr.transpose(1, 2, 0)
    return arr


def visualize(data_path, traj_idx, horizon, use_frame_diff, out_path):
    loaders = make_discrete_dataloaders(
        data_path, batch_size=1, num_workers=0,
        horizon=horizon, frame_stack=1,
        state_mean=np.zeros(4), state_std=np.ones(4),
        action_scale=1.0, preload_obs=True)

    loader = loaders['train']
    for i, batch in enumerate(loader):
        if i == traj_idx:
            break

    # obs_seq: (1, horizon+1, 3, H, W) uint8
    obs_seq  = batch['obs_seq'][0].float() / 255.   # (H+1, 3, h, w)
    prev_obs = batch['prev_obs'][0].float() / 255.  # (3, h, w)
    states   = batch['states'][0]                   # (H+1, 4)
    steps    = obs_seq.shape[0]

    # Build prev frames: prev_obs, then each frame is the previous obs_seq
    prevs = torch.cat([prev_obs.unsqueeze(0), obs_seq[:-1]], dim=0)  # (H+1, 3, h, w)

    if use_frame_diff:
        # Exactly what encode() does: cat([prev, curr, curr-prev], dim=1)
        vit_inputs = torch.cat([prevs, obs_seq, obs_seq - prevs], dim=1)  # (H+1, 9, h, w)
        n_rows = 4   # raw_prev | raw_curr | diff | vit_input_preview
    else:
        vit_inputs = obs_seq                                              # (H+1, 3, h, w)
        n_rows = 2

    ncols  = steps
    fig    = plt.figure(figsize=(2.5 * ncols, 2.5 * n_rows + 0.6))
    fig.suptitle(f'Trajectory {traj_idx} — {steps} steps  '
                 f'({"frame-diff 9ch" if use_frame_diff else "RGB 3ch"} → ViT)',
                 fontsize=10, y=0.98)
    gs = gridspec.GridSpec(n_rows, ncols, figure=fig, hspace=0.35, wspace=0.05)

    state_labels = ['x', 'y', 'vx', 'vy']

    for t in range(steps):
        # Row 0: previous frame (raw RGB)
        ax = fig.add_subplot(gs[0, t])
        ax.imshow(to_uint8(prevs[t]))
        ax.set_title(f't={t}\nprev', fontsize=7)
        ax.axis('off')

        # Row 1: current frame (raw RGB)
        ax = fig.add_subplot(gs[1, t])
        ax.imshow(to_uint8(obs_seq[t]))
        s = states[t].numpy()
        lbl = '\n'.join(f'{k}={v:.2f}' for k, v in zip(state_labels, s))
        ax.set_title(f'curr\n{lbl}', fontsize=6)
        ax.axis('off')

        if use_frame_diff:
            # Row 2: frame difference (clipped and rescaled for display)
            diff = obs_seq[t] - prevs[t]
            diff_vis = (diff * 0.5 + 0.5).clamp(0, 1)  # shift [-1,1]→[0,1]
            ax = fig.add_subplot(gs[2, t])
            ax.imshow(to_uint8(diff_vis))
            ax.set_title('diff\n(±1→gray)', fontsize=6)
            ax.axis('off')

            # Row 3: ViT 9-ch input shown as 3 side-by-side RGB patches
            ax = fig.add_subplot(gs[3, t])
            vi = vit_inputs[t]          # (9, h, w)
            h, w = vi.shape[1], vi.shape[2]
            strip = np.concatenate([
                to_uint8(vi[:3]),                         # prev
                to_uint8(vi[3:6]),                        # curr
                to_uint8((vi[6:9] * 0.5 + 0.5).clamp(0,1)),  # diff
            ], axis=1)  # (h, 3w, 3)
            ax.imshow(strip)
            ax.set_title('→ViT\n[prev|curr|diff]', fontsize=6)
            ax.axis('off')

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f'[done] {out}')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data',            required=True)
    p.add_argument('--traj',            type=int, default=0,
                   help='Which trajectory (batch index) to visualize')
    p.add_argument('--horizon',         type=int, default=8,
                   help='Number of steps to show (horizon)')
    p.add_argument('--no-frame-diff',   action='store_true',
                   help='Show raw RGB only (skip frame-diff reconstruction)')
    p.add_argument('--out',             default='results/rollout_vis.png')
    args = p.parse_args()

    visualize(
        data_path=args.data,
        traj_idx=args.traj,
        horizon=args.horizon,
        use_frame_diff=not args.no_frame_diff,
        out_path=args.out)


if __name__ == '__main__':
    main()
