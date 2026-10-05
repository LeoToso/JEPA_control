#!/usr/bin/env python
"""Build a side-by-side trajectory figure from three GIFs.

Layout: 1 row of 3 columns, each column = one model.
Each column shows N evenly-spaced frames stacked vertically.
Model names appear in bold above each column.

Usage
-----
python experiments/make_trajectory_figure.py \
    --gif1 results/renders/latent_icem_H3_success/success_trial_004.gif \
    --gif2 results/renders/latent_icem_H3_ms_sr_success/best_failed_trial_009.gif \
    --gif3 results/renders/latent_icem_H3_fwd_sr/best_failed_trial_000.gif \
    --out  results/trajectory_figure.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_gif_frames(path: str) -> list[np.ndarray]:
    img = Image.open(path)
    frames = []
    try:
        while True:
            frames.append(np.array(img.convert('RGB')))
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return frames


def pick_frames(frames: list[np.ndarray], n: int,
                start: int = 0, end_offset: int = 0) -> list[np.ndarray]:
    T = len(frames)
    i0 = min(start, T - 1)
    i1 = max(T - 1 - end_offset, i0)
    indices = np.linspace(i0, i1, n).round().astype(int).tolist()
    return [frames[i] for i in indices]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gif1', required=True, help='GIF for column 1 (1SP+EP-IDM)')
    p.add_argument('--gif2', required=True, help='GIF for column 2 (MSP+SIG)')
    p.add_argument('--gif3', required=True, help='GIF for column 3 (1SP+SIG)')
    p.add_argument('--label1', default='1SP+EP-IDM')
    p.add_argument('--label2', default='MSP+SIG')
    p.add_argument('--label3', default='1SP+SIG')
    p.add_argument('--n-frames',    type=int, default=5,
                   help='Number of frames to show per model')
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--end-offset',  type=int, default=0)
    p.add_argument('--out', default='results/trajectory_figure.pdf')
    p.add_argument('--dpi', type=int, default=200)
    args = p.parse_args()

    cols = [
        (args.gif1, args.label1),
        (args.gif2, args.label2),
        (args.gif3, args.label3),
    ]

    n = args.n_frames
    n_cols = len(cols)

    cell_size = 2.0          # each frame is cell_size × cell_size inches
    label_height = 0.55      # inches reserved for the bold label row
    fig_w = n_cols * cell_size
    fig_h = label_height + n * cell_size

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=args.dpi)
    fig.patch.set_facecolor('white')

    # GridSpec: first row = labels (thin), remaining rows = frames
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(
        n + 1, n_cols,
        height_ratios=[label_height / cell_size] + [1.0] * n,
        hspace=0.04, wspace=0.04,
        left=0.0, right=1.0, top=1.0, bottom=0.0,
    )

    for col_idx, (gif_path, label) in enumerate(cols):
        print(f'[fig] loading {gif_path} …')
        all_frames = load_gif_frames(gif_path)
        print(f'      {len(all_frames)} frames total')
        selected = pick_frames(all_frames, n,
                               start=args.start_frame,
                               end_offset=args.end_offset)

        # Bold label in the top row
        ax_label = fig.add_subplot(gs[0, col_idx])
        ax_label.axis('off')
        ax_label.text(0.5, 0.5, label,
                      ha='center', va='center',
                      fontsize=16, fontweight='bold',
                      transform=ax_label.transAxes)

        # Frames stacked below
        for row_idx, frame in enumerate(selected):
            ax = fig.add_subplot(gs[row_idx + 1, col_idx])
            ax.imshow(frame)
            ax.axis('off')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', dpi=args.dpi)
    png_path = out_path.with_suffix('.png')
    fig.savefig(png_path, bbox_inches='tight', dpi=args.dpi)
    print(f'[fig] saved → {out_path}')
    print(f'[fig] saved → {png_path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
