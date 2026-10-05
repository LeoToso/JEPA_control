#!/usr/bin/env python
"""Animated GIF of LQR trajectories — one column per model, time plays out.

Each column corresponds to one model; the animation shows all models in sync,
stepping through the episode frame by frame.

Usage:
  python experiments/make_lqr_gif.py \
      --jsons  results/lqr_a.json results/lqr_b.json \
      --subtitles "Model A" "Model B" \
      --trial 5 \
      --out results/lqr_comparison.gif
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif':  ['Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
})

from envs.cartpole_visual import ContinuousCartpoleVisual


def make_render_env(image_size, seed=0):
    return ContinuousCartpoleVisual(
        frame_skip=5, image_size=image_size,
        action_range=(-10., 10.),
        mass_cart=1.0, mass_pole=0.1,
        pole_length=0.5, gravity=9.8, dt=0.02,
        theta_threshold=1.2, seed=seed)


def render_all_states(states, image_size):
    """Render every state in the trajectory."""
    env = make_render_env(image_size)
    frames = []
    for state in states:
        obs, _, _ = env.reset_to_state(np.asarray(state, dtype=np.float32))
        frames.append(obs)
    env.close()
    return frames


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--jsons', nargs='+', required=True)
    p.add_argument('--subtitles', nargs='+', required=True)
    p.add_argument('--out', required=True, help='Output .gif path')
    p.add_argument('--trial', type=int, default=0)
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--fps', type=int, default=8,
                   help='Frames per second in the GIF')
    p.add_argument('--subtitle-fontsize', type=float, default=11)
    p.add_argument('--col-width', type=float, default=2.2,
                   help='Figure width per column in inches')
    p.add_argument('--fig-height', type=float, default=2.8,
                   help='Figure height in inches')
    args = p.parse_args()

    if len(args.jsons) != len(args.subtitles):
        p.error('--jsons and --subtitles must have the same length')

    # Load all trials and render frames
    all_frames = []
    successes = []
    n_steps_list = []
    fail_steps = []          # frame index where episode terminated (-1 = never)
    THETA_IDX       = 2      # pole angle in state vector
    THETA_THRESHOLD = 1.2    # rad — matches ContinuousCartpoleVisual default

    for json_path, subtitle in zip(args.jsons, args.subtitles):
        with open(json_path) as f:
            data = json.load(f)
        model_entry = data['models'][0] if 'models' in data else data
        trial = model_entry['trials'][args.trial]
        states = trial['states']
        success = trial.get('success', None)
        successes.append(success)

        # Find first frame where |theta| >= threshold (episode terminated)
        fail_t = -1
        for t_idx, s in enumerate(states):
            if abs(s[THETA_IDX]) >= THETA_THRESHOLD:
                fail_t = t_idx
                break
        fail_steps.append(fail_t)

        print(f'[render] {subtitle}  ({len(states)} states, '
              f'fail_t={fail_t}) …')
        frames = render_all_states(states, args.image_size)
        all_frames.append(frames)
        n_steps_list.append(len(frames))

    n_models = len(args.jsons)
    n_steps = max(n_steps_list)  # pad shorter trajectories with last frame

    # Pad shorter trajectories
    for i, frames in enumerate(all_frames):
        if len(frames) < n_steps:
            all_frames[i] = frames + [frames[-1]] * (n_steps - len(frames))

    # Build figure
    fig = plt.figure(figsize=(n_models * args.col_width, args.fig_height))
    gs = gridspec.GridSpec(1, n_models, hspace=0.0, wspace=0.04,
                           left=0.02, right=0.98, top=0.82, bottom=0.02)

    axes = [fig.add_subplot(gs[0, c]) for c in range(n_models)]
    im_objs = []
    for ax, frames, subtitle, success in zip(axes, all_frames, args.subtitles, successes):
        im = ax.imshow(frames[0])
        im_objs.append(im)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(subtitle, fontsize=args.subtitle_fontsize,
                     fontweight='bold', pad=4)

    # Step counter text
    step_text = fig.text(0.5, 0.88, 't = 0', ha='center', va='bottom',
                         fontsize=10)

    def update(t):
        for i, (im, frames) in enumerate(zip(im_objs, all_frames)):
            im.set_data(frames[t])
            terminated = fail_steps[i] >= 0 and t >= fail_steps[i]
            success_end = (not terminated) and (t == n_steps - 1) and successes[i]
            if terminated:
                c, lw = '#d62728', 3.5   # red from failure step onward
            elif success_end:
                c, lw = '#2ca02c', 3.5   # green on final frame if succeeded
            else:
                c, lw = 'black', 1.5
            for spine in axes[i].spines.values():
                spine.set_edgecolor(c)
                spine.set_linewidth(lw)
        step_text.set_text(f't = {t}')
        return im_objs

    anim = FuncAnimation(fig, update, frames=n_steps,
                         interval=1000 // args.fps, blit=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f'[save] writing {n_steps} frames → {out} …')
    anim.save(str(out), writer=PillowWriter(fps=args.fps))
    print(f'[done] {out}')


if __name__ == '__main__':
    main()

