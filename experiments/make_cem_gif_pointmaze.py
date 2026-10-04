#!/usr/bin/env python
"""Animated GIF of PointMaze CEM trajectories — one column per model, time plays out.

Each column shows one model; all models animate in sync through the episode.
The last frame's border turns green (success) or red (failure).

json_spec syntax:
  "path/to/results.json"    → reads models[0]
  "path/to/results.json:2"  → reads models[2]

Usage
-----
  MUJOCO_GL=egl python experiments/make_cem_gif_pointmaze.py \\
      --jsons \\
          results/cem_pointmaze_ms_ep_ar.json \\
          results/cem_pointmaze_fwd_ep_ar.json \\
          results/cem_pointmaze_ms_sr.json \\
          results/cem_dinowm_pointmaze.json \\
      --labels "MS+EP-AR" "FWD+EP-AR" "MS+SR" "DINO-WM" \\
      --trial 0 --image-size 128 \\
      --out results/cem_trajectories_4models_pointmaze.gif
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

from envs.pointmaze_visual import PointMazeVisual


def load_trial(json_spec: str, trial_idx: int) -> tuple[dict, str]:
    parts = json_spec.rsplit(':', 1)
    try:
        model_idx = int(parts[-1])
        json_path = parts[0]
    except ValueError:
        json_path, model_idx = json_spec, 0

    with open(json_path) as f:
        data = json.load(f)
    model_entry = data['models'][model_idx]
    trial = model_entry['trials'][trial_idx]
    if 'goal_xy' not in trial and 'goal_xy' in model_entry:
        trial = dict(trial)
        trial['goal_xy'] = model_entry['goal_xy']
    return trial, model_entry.get('label', Path(json_path).stem)


def make_render_env(image_size: int, seed: int = 0) -> PointMazeVisual:
    env_cfg = {'environment': {'maze_map': 'U', 'image_size': image_size,
                               'action_scale': 1.0}}
    return PointMazeVisual(env_cfg, seed=seed)


def render_all_states(states, goal_xy, image_size: int) -> list:
    env = make_render_env(image_size)
    frames = []
    for state in states:
        obs, _, _ = env.reset_to_state(np.asarray(state, dtype=np.float32),
                                       goal_xy=goal_xy)
        frames.append(obs)
    env.close()
    return frames


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Animated GIF of PointMaze CEM trajectories.')
    p.add_argument('--jsons',   nargs='+', required=True,
                   help='JSON specs: "path.json" or "path.json:model_idx"')
    p.add_argument('--labels',  nargs='+', required=True,
                   help='Column label for each JSON spec (same order)')
    p.add_argument('--trial',   type=int, default=0)
    p.add_argument('--image-size', type=int, default=128)
    p.add_argument('--fps',     type=int, default=8)
    p.add_argument('--col-width',  type=float, default=2.2,
                   help='Figure width per column in inches')
    p.add_argument('--fig-height', type=float, default=2.8,
                   help='Figure height in inches')
    p.add_argument('--subtitle-fontsize', type=float, default=11)
    p.add_argument('--out',     required=True, help='Output .gif path')
    args = p.parse_args()

    if len(args.jsons) != len(args.labels):
        p.error('--jsons and --labels must have the same length')

    # Load trials and render all frames
    all_frames = []
    successes  = []
    n_steps_list = []

    for json_spec, label in zip(args.jsons, args.labels):
        trial, _ = load_trial(json_spec, args.trial)
        states  = trial['states']
        goal_xy = trial.get('goal_xy', None)
        success = trial.get('success', None)
        successes.append(success)

        goal_np = np.array(goal_xy, dtype=np.float32) if goal_xy is not None else None
        print(f'[render] {label}  ({len(states)} states) …')
        frames = render_all_states(states, goal_np, args.image_size)
        all_frames.append(frames)
        n_steps_list.append(len(frames))

    n_models = len(args.jsons)
    n_steps  = max(n_steps_list)

    # Pad shorter trajectories with their last frame
    for i, frames in enumerate(all_frames):
        if len(frames) < n_steps:
            all_frames[i] = frames + [frames[-1]] * (n_steps - len(frames))

    # Build figure
    fig = plt.figure(figsize=(n_models * args.col_width, args.fig_height))
    gs  = gridspec.GridSpec(1, n_models, hspace=0.0, wspace=0.04,
                            left=0.02, right=0.98, top=0.82, bottom=0.02)

    axes    = [fig.add_subplot(gs[0, c]) for c in range(n_models)]
    im_objs = []
    for ax, frames, label in zip(axes, all_frames, args.labels):
        im = ax.imshow(frames[0])
        im_objs.append(im)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, fontsize=args.subtitle_fontsize,
                     fontweight='bold', pad=4)

    step_text = fig.text(0.5, 0.88, 't = 0', ha='center', va='bottom', fontsize=10)

    def update(t):
        for i, (im, frames) in enumerate(zip(im_objs, all_frames)):
            im.set_data(frames[t])
            if t == n_steps - 1 and successes[i] is not None:
                c  = '#2ca02c' if successes[i] else '#d62728'
                lw = 3.5
            else:
                c  = 'black'
                lw = 1.5
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
    plt.close(fig)
    print(f'[done] {out}')


if __name__ == '__main__':
    main()

