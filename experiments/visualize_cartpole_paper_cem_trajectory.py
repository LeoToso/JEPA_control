#!/usr/bin/env python
"""Visualize one successful trajectory saved by paper-style CartPole CEM."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import yaml

from probe_utils import make_env


STATE_NAMES = (r'$x$', r'$\dot{x}$', r'$\theta$', r'$\dot{\theta}$')
STATE_UNITS = ('m', 'm/s', 'rad', 'rad/s')


def choose_trial(rows, index):
    if index is not None:
        if index < 0 or index >= len(rows):
            raise IndexError(f'trial index {index} outside [0, {len(rows) - 1}]')
        if not rows[index]['success']:
            print(f'[warning] requested trial {index} is not marked successful')
        return index, rows[index]
    successful = [(i, row) for i, row in enumerate(rows) if row['success']]
    if not successful:
        raise RuntimeError('result contains no successful trajectory')
    # Choose the successful trajectory with the smallest terminal error.
    return min(successful, key=lambda item: item[1]['final_error'])


def save_plot(row, protocol, config, path):
    states = np.asarray(row['states'], dtype=np.float64)
    goal = np.asarray(row['goal_state'], dtype=np.float64)
    actions = np.asarray(row['actions'], dtype=np.float64)
    frame_skip = int(protocol['primitive_steps_per_model_step'])
    dt = float(config['environment'].get('dt', .02))
    macro_dt = frame_skip * dt
    time = np.arange(len(states)) * macro_dt
    action_time = np.arange(len(actions) + 1) * macro_dt
    error = np.linalg.norm(states - goal[None], axis=1)
    executed = int(protocol['executed_model_steps'])
    threshold = float(protocol.get('success_threshold_physical_norm', .1))
    terminal_cost = protocol.get('terminal_cost', '')
    controller_name = ('CEM oracle' if terminal_cost ==
                       'squared_physical_state_goal_distance'
                       else 'CEM learned dynamics')

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    axes = axes.ravel()
    for i in range(4):
        axes[i].plot(time, states[:, i], marker='o', ms=3, lw=2)
        axes[i].axhline(goal[i], color='black', ls='--', lw=1, label='goal')
        axes[i].set_ylabel(f'{STATE_NAMES[i]} [{STATE_UNITS[i]}]')
        axes[i].grid(alpha=.25)
    axes[4].step(action_time, np.r_[actions, actions[-1]], where='post', lw=2)
    axes[4].axhline(0., color='black', ls='--', lw=1)
    axes[4].set_ylabel('force [N]')
    axes[4].grid(alpha=.25)
    axes[5].plot(time, error, marker='o', ms=3, lw=2, color='tab:red')
    axes[5].axhline(threshold, color='black', ls='--', lw=1,
                    label=f'success/stable threshold ({threshold:g})')
    axes[5].set_ylabel(r'$\|x_t-x_g\|_2$')
    axes[5].legend(frameon=False)
    axes[5].grid(alpha=.25)
    for ax in axes:
        for boundary in range(executed, len(actions), executed):
            ax.axvline(boundary * macro_dt, color='tab:purple', ls=':', lw=1)
        ax.set_xlabel('time [s]')
    fig.suptitle(
        f'{controller_name} successful trajectory\n'
        f'final error={row["final_error"]:.5f}, '
        f'max error={row["max_error"]:.5f}, replans={row["replans"]}')
    fig.tight_layout()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] plot -> {path}')


def annotate(frame, primitive_step, action, error):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    text = f'step {primitive_step:02d}'
    box = draw.textbbox((0, 0), text)
    width, height = box[2] - box[0], box[3] - box[1]
    draw.rectangle((4, 4, width + 12, height + 10), fill=(255, 255, 255))
    draw.text((8, 7), text, fill=(0, 0, 0))
    return image


def save_gif(row, config, path, fps):
    env = make_env(config, 0)
    initial = np.asarray(row['initial_state'], dtype=np.float64)
    goal = np.asarray(row['goal_state'], dtype=np.float64)
    actions = np.asarray(row['actions'], dtype=np.float64)
    frame_skip = int(config['environment'].get('frame_skip', 1))
    frame, state, _ = env.reset_to_state(initial)
    images = [annotate(frame, 0, 0., float(np.linalg.norm(state - goal)))]
    primitive_step = 0
    for action in actions:
        for _ in range(frame_skip):
            state, _ = env._physics_step(float(action))
            env._state = state.copy()
            primitive_step += 1
            images.append(annotate(
                env._render_obs(), primitive_step, float(action),
                float(np.linalg.norm(state - goal))))
    env.close()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(int(round(1000 / fps)), 1)
    images[0].save(
        path, save_all=True, append_images=images[1:], duration=duration,
        loop=0, optimize=False)
    print(f'[saved] gif  -> {path} ({len(images)} frames, {fps:g} fps)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--result', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--trial-index', type=int)
    p.add_argument('--plot', required=True)
    p.add_argument('--gif')
    p.add_argument('--fps', type=float, default=10.)
    args = p.parse_args()

    with open(args.result) as f:
        result = json.load(f)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    index, row = choose_trial(result['trials'], args.trial_index)
    print(f'[trajectory] trial={index} success={row["success"]} '
          f'final={row["final_error"]:.5f} max={row["max_error"]:.5f}')
    save_plot(row, result['protocol'], config, args.plot)
    if args.gif:
        save_gif(row, config, args.gif, args.fps)


if __name__ == '__main__':
    main()
