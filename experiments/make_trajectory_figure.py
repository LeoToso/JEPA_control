#!/usr/bin/env python
"""Build a 3-row × 15-frame trajectory figure from saved GIFs.

Each row = one model, best trajectory.
First column = first frame, last column = last frame, rest evenly subsampled.

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


def pick_frames(frames: list[np.ndarray], n: int = 10,
                start: int = 100, end_offset: int = 50) -> list[np.ndarray]:
    """Select n frames evenly spaced from frame[start] to frame[T-1-end_offset]."""
    T = len(frames)
    i0 = min(start, T - 1)
    i1 = max(T - 1 - end_offset, i0)
    indices = np.linspace(i0, i1, n).round().astype(int).tolist()
    return [frames[i] for i in indices]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gif1', required=True, help='GIF for row 1 (1SP+EP-IDM)')
    p.add_argument('--gif2', required=True, help='GIF for row 2 (MSP+SIG)')
    p.add_argument('--gif3', required=True, help='GIF for row 3 (1SP+SIG)')
    p.add_argument('--label1', default='1SP+EP-IDM')
    p.add_argument('--label2', default='1SP+SIG')
    p.add_argument('--label3', default='MSP+SIG')
    p.add_argument('--n-frames',    type=int, default=10)
    p.add_argument('--start-frame', type=int, default=150,
                   help='Index of the first frame to show')
    p.add_argument('--end-offset',  type=int, default=50,
                   help='Number of frames to trim from the end')
    p.add_argument('--out', default='results/trajectory_figure.pdf')
    p.add_argument('--dpi', type=int, default=200)
    args = p.parse_args()

    rows = [
        (args.gif1, args.label1),
        (args.gif2, args.label2),
        (args.gif3, args.label3),
    ]

    n = args.n_frames
    cell_w = cell_h = 2.2
    left_margin = 1.2
    fig_w = left_margin + n * cell_w
    fig_h = len(rows) * cell_h
    fig, axes = plt.subplots(len(rows), n, figsize=(fig_w, fig_h), dpi=args.dpi)
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=left_margin / fig_w,
                        right=1.0, top=1.0, bottom=0.0,
                        hspace=0.04, wspace=0.04)
    if len(rows) == 1:
        axes = [axes]

    for row_idx, (gif_path, label) in enumerate(rows):
        print(f'[fig] loading {gif_path} …')
        all_frames = load_gif_frames(gif_path)
        print(f'      {len(all_frames)} frames total')
        selected = pick_frames(all_frames, n,
                               start=args.start_frame,
                               end_offset=args.end_offset)

        for col_idx, frame in enumerate(selected):
            ax = axes[row_idx][col_idx]
            ax.imshow(frame)
            ax.axis('off')

        # vertical label centred on this row (matches visualize_walker_trajectory_frames.py)
        row_axes = axes[row_idx]
        ys = [ax.get_position().y0 + ax.get_position().height / 2
              for ax in row_axes]
        y_mid = sum(ys) / len(ys)
        xs = [ax.get_position().x0 for ax in row_axes]
        x_left = min(xs) - 0.01
        fig.text(x_left, y_mid, label,
                 ha='right', va='center', rotation='vertical',
                 fontsize=18, fontweight='bold')


    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', dpi=args.dpi)
    # also save PNG for easy preview
    png_path = out_path.with_suffix('.png')
    fig.savefig(png_path, bbox_inches='tight', dpi=args.dpi)
    print(f'[fig] saved → {out_path}')
    print(f'[fig] saved → {png_path}')
    plt.close(fig)


if __name__ == '__main__':
    main()

