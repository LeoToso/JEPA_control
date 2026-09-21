#!/usr/bin/env python
"""Plot LQR trajectory frames for multiple models.

Each row shows N evenly-spaced frames from one trial, first frame = step 0,
last frame = last step of the trajectory.

Usage:
  python experiments/plot_lqr_trajectories.py \
      --jsons  results/lqr_1sp_AR.json results/lqr_ms_sigreg.json results/lqr_ms_dinov2.json \
      --subtitles "1sp AR" "MS SIGReg" "MS DINOv2" \
      --out    results/lqr_trajectories.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif':  ['Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
})

from envs.cartpole_visual import ContinuousCartpoleVisual


def make_render_env(image_size, seed=0):
    return ContinuousCartpoleVisual(
        frame_skip=5,
        image_size=image_size,
        action_range=(-10., 10.),
        mass_cart=1.0, mass_pole=0.1,
        pole_length=0.5, gravity=9.8, dt=0.02,
        theta_threshold=1.2,
        seed=seed)


def render_states(states, image_size, n_frames):
    """Render n_frames evenly-spaced frames; first=states[0], last=states[-1]."""
    indices = np.round(np.linspace(0, len(states) - 1, n_frames)).astype(int)
    env = make_render_env(image_size)
    frames = []
    for idx in indices:
        state = np.asarray(states[idx], dtype=np.float32)
        obs, _, _ = env.reset_to_state(state)
        frames.append(obs)
    env.close()
    return frames, indices


def pick_trial(trials, trial_idx):
    if trial_idx >= len(trials):
        raise ValueError(f'trial {trial_idx} out of range (only {len(trials)} trials)')
    return trials[trial_idx]


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--jsons', nargs='+', required=True,
                   help='LQR JSON result files, one per row')
    p.add_argument('--subtitles', nargs='+', required=True,
                   help='Row label for each JSON (same order)')
    p.add_argument('--out', required=True, help='Output image path')
    p.add_argument('--trial', type=int, default=0,
                   help='Which trial index to visualise from each JSON')
    p.add_argument('--n-frames', type=int, default=10,
                   help='Number of frames per row')
    p.add_argument('--image-size', type=int, default=224,
                   help='Render resolution (square)')
    p.add_argument('--subtitle-fontsize', type=float, default=13)
    p.add_argument('--show-step', action='store_true',
                   help='Annotate each frame with its step index')
    p.add_argument('--row-height', type=float, default=2.4,
                   help='Figure height per row in inches')
    p.add_argument('--hspace', type=float, default=0.25,
                   help='Vertical gap between rows (gridspec hspace)')
    args = p.parse_args()

    if len(args.jsons) != len(args.subtitles):
        p.error('--jsons and --subtitles must have the same number of entries')

    n_rows = len(args.jsons)
    n_cols = args.n_frames

    fig = plt.figure(figsize=(n_cols * 1.7, n_rows * args.row_height))
    gs = gridspec.GridSpec(
        n_rows, n_cols,
        hspace=args.hspace, wspace=0.04,
        left=0.10, right=0.97, top=0.97, bottom=0.02)

    for row, (json_path, subtitle) in enumerate(zip(args.jsons, args.subtitles)):
        with open(json_path) as f:
            data = json.load(f)
        # compare_lqr_smwm.py wraps in {"models": [{..., "trials": [...]}]};
        # lqr_dinowm_cartpole.py writes {"trials": [...]} directly.
        model_entry = data['models'][0] if 'models' in data else data
        trial = pick_trial(model_entry['trials'], args.trial)
        states = trial['states']
        success = trial.get('success', None)
        held = trial.get('held_stable', trial.get('held', None))

        frames, indices = render_states(states, args.image_size, n_cols)

        is_last = [col == n_cols - 1 for col in range(n_cols)]
        for col, (frame, step_idx, last) in enumerate(zip(frames, indices, is_last)):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(frame)
            if last and success is not None:
                border_color = '#2ca02c' if success else '#d62728'
                lw = 3.5
            else:
                border_color = 'black'
                lw = 1.5
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(border_color)
                spine.set_linewidth(lw)
            ax.set_xticks([])
            ax.set_yticks([])
            if args.show_step:
                ax.set_title(f't={step_idx}', fontsize=7, pad=2)

        # Row subtitle on the left (no status symbol)
        fig.text(
            0.01, 1. - (row + 0.5) / n_rows,
            subtitle,
            va='center', ha='left',
            fontsize=args.subtitle_fontsize,
            fontweight='bold',
            rotation=90)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight', pad_inches=0.05)
    print(f'[done] {out}')


if __name__ == '__main__':
    main()
