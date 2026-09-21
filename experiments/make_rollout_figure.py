#!/usr/bin/env python
"""Assemble saved rollout frames into a 3-row × N-col figure.

Rows: GT MuJoCo | FWD+EP-AR | MS+SR
Cols: N evenly-spaced frames selected from each sequence.

Usage
-----
  python experiments/make_rollout_figure.py \
      --rollout-dir results/rollout_comparison \
      --n-frames 20 \
      --out results/rollout_comparison/rollout_figure.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image


ROW_DIRS   = ['gt',     'fwd_ar',   'ms_sr']
ROW_LABELS = ['GT MuJoCo', 'FWD+EP-AR', 'MS+SR']


def load_frames(seq_dir: Path) -> list[np.ndarray]:
    paths = sorted(seq_dir.glob('*.png'))
    if not paths:
        raise FileNotFoundError(f'No PNG frames found in {seq_dir}')
    return [np.asarray(Image.open(p).convert('RGB')) for p in paths]


def pick_frames(frames: list[np.ndarray], n: int,
                start: int = 0, mode: str = 'evenly') -> list[np.ndarray]:
    """Select n frames from `frames[start:]`.

    mode='evenly'  — evenly spaced across the remaining sequence (default)
    mode='first'   — first n frames starting at `start`
    """
    sub = frames[start:]
    if mode == 'first':
        return sub[:n]
    total = len(sub)
    indices = [round(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
    return [sub[i] for i in indices]


def make_figure(rollout_dir: Path, n_frames: int, out_path: Path,
                dpi: int = 150, start: int = 0,
                mode: str = 'evenly') -> None:
    rows = []
    for d in ROW_DIRS:
        all_frames = load_frames(rollout_dir / d)
        rows.append(pick_frames(all_frames, n_frames, start=start, mode=mode))
        print(f'  {d}: {len(all_frames)} total → selected {n_frames} '
              f'(start={start}, mode={mode})')

    n_rows = len(rows)
    h, w   = rows[0][0].shape[:2]
    aspect = w / h

    # Cell size in inches; tight layout handles margins
    cell_w = 1.5
    cell_h = cell_w / aspect
    label_w = 1.0   # left margin for row labels

    fig_w = label_w + n_frames * cell_w
    fig_h = n_rows  * cell_h

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    # GridSpec: one extra column on the left for row labels
    gs = gridspec.GridSpec(
        n_rows, n_frames + 1,
        figure=fig,
        left=label_w / fig_w,
        right=1.0,
        top=1.0,
        bottom=0.0,
        wspace=0.02,
        hspace=0.02,
    )

    for r, (frame_seq, label) in enumerate(zip(rows, ROW_LABELS)):
        # Row label axis (leftmost column)
        ax_lbl = fig.add_subplot(gs[r, 0])
        ax_lbl.set_axis_off()
        ax_lbl.text(0.5, 0.5, label,
                    ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    rotation=90,
                    transform=ax_lbl.transAxes)

        for c, frame in enumerate(frame_seq):
            ax = fig.add_subplot(gs[r, c + 1])
            ax.imshow(frame)
            ax.set_axis_off()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'[saved] {out_path}')


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--rollout-dir', default='results/rollout_comparison')
    p.add_argument('--n-frames',   type=int, default=20)
    p.add_argument('--out',        default=None,
                   help='Output path (.pdf or .png). '
                        'Defaults to <rollout-dir>/rollout_figure.pdf')
    p.add_argument('--dpi',        type=int, default=150)
    p.add_argument('--start',      type=int, default=0,
                   help='Index of first frame to consider (0-based)')
    p.add_argument('--mode',       default='evenly',
                   choices=['evenly', 'first'],
                   help='evenly: pick N evenly-spaced frames; '
                        'first: pick the first N frames after --start')
    return p.parse_args()


def main():
    args = parse_args()
    rollout_dir = Path(args.rollout_dir)
    out_path    = Path(args.out) if args.out else rollout_dir / 'rollout_figure.pdf'
    print(f'Building {args.n_frames}-frame figure from {rollout_dir} '
          f'(start={args.start}, mode={args.mode})')
    make_figure(rollout_dir, args.n_frames, out_path,
                dpi=args.dpi, start=args.start, mode=args.mode)


if __name__ == '__main__':
    main()
