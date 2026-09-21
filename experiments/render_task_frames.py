#!/usr/bin/env python
"""Render a 3-row × 10-column task frame strip.

Each row shows 10 frames from trial 0 of the given JSON.
First frame: blue border (initial state).
Last frame:  green border (final state).
Row labels (bold, left): CartPole | Walker2D | PointMaze.

Requirements
------------
Walker2D JSON must contain 'qpos_seq' and 'qvel_seq' per trial.
Re-generate with:
    python experiments/gt_icem_walker.py ... --save-states --trials 1 \\
        --output results/gt_icem_upright5_states.json

Usage
-----
python experiments/render_task_frames.py \\
    --cartpole  results/lqr_1steppred_endpointAR.json \\
    --walker    results/gt_icem_upright5_states.json \\
    --pointmaze results/cem_fs5_pointmaze_mspred_endpoint_AR.json \\
    --out       results/task_frames.pdf
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent))
os.environ.setdefault('MUJOCO_GL', 'egl')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

COLOR_FIRST  = '#2166ac'   # blue  — initial state
COLOR_LAST   = '#2ca02c'   # green — final state
BORDER_LW    = 5


# ── frame helpers ─────────────────────────────────────────────────────────────

def _subsample(n_total: int, n_frames: int) -> list[int]:
    """Return n_frames indices spanning [0, n_total-1], always including both ends."""
    return list(np.round(np.linspace(0, n_total - 1, n_frames)).astype(int))


def render_cartpole(json_path: str, n_frames: int, image_size: int,
                    trial_idx: int = -1) -> list:
    with open(json_path) as f:
        data = json.load(f)
    # handle both {trials:[...]} and {models:[{trials:[...]}]}
    trials = (data['models'][0]['trials'] if 'models' in data
              else data['trials'])

    if trial_idx < 0:
        # pick trial with largest ||initial state||
        norms = [np.linalg.norm(t['states'][0]) for t in trials]
        trial_idx = int(np.argmax(norms))
        print(f'  CartPole: using trial {trial_idx} '
              f'(largest initial state norm={norms[trial_idx]:.3f})')

    trial  = trials[trial_idx]
    states = trial['states']           # list of [x, xdot, theta, thetadot]
    idx    = _subsample(len(states), n_frames)

    from envs.cartpole_visual import ContinuousCartpoleVisual
    # protocol may be nested under models[0] or at top level
    cfg = (data['models'][0].get('protocol', {})
           if 'models' in data else data.get('protocol', {}))
    env = ContinuousCartpoleVisual(
        image_size   = image_size,
        frame_skip   = int(cfg.get('frame_skip', 1)),
        action_range = tuple(cfg.get('action_range', [-10, 10])),
        mass_cart    = float(cfg.get('mass_cart', 1.0)),
        mass_pole    = float(cfg.get('mass_pole', 0.1)),
        pole_length  = float(cfg.get('pole_length', 0.5)),
        gravity      = float(cfg.get('gravity', 9.8)),
        dt           = float(cfg.get('dt', 0.02)),
    )
    frames = []
    for i in idx:
        obs = env.reset_to_state(np.array(states[i], dtype=np.float32))
        if isinstance(obs, tuple):
            obs = obs[0]
        frames.append(np.array(obs))
    env.close()
    return frames


def render_walker_from_gif(gif_path: str, n_frames: int) -> list:
    """Extract n_frames evenly-spaced frames from a GIF."""
    from PIL import Image
    gif = Image.open(gif_path)
    raw = []
    try:
        while True:
            raw.append(np.array(gif.convert('RGB')))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    idx = _subsample(len(raw), n_frames)
    return [raw[i] for i in idx]


def render_walker(json_path: str, n_frames: int) -> list:
    import gymnasium as gym

    with open(json_path) as f:
        data = json.load(f)
    trial = data['trials'][0]

    if 'qpos_seq' not in trial:
        raise RuntimeError(
            f"Walker JSON '{json_path}' has no 'qpos_seq'.\n"
            "Re-run gt_icem_walker.py with --save-states, or pass "
            "--walker-gif path/to/trajectory.gif instead.")

    qpos_seq = trial['qpos_seq']
    qvel_seq = trial['qvel_seq']
    idx      = _subsample(len(qpos_seq), n_frames)

    env = gym.make('Walker2d-v4', render_mode='rgb_array')
    env.reset(seed=42)

    frames = []
    for i in idx:
        env.unwrapped.set_state(
            np.array(qpos_seq[i], dtype=np.float64),
            np.array(qvel_seq[i], dtype=np.float64))
        frames.append(env.render())
    env.close()
    return frames


def render_pointmaze(json_path: str, n_frames: int, image_size: int) -> list:
    with open(json_path) as f:
        data = json.load(f)

    # handle both formats: {trials:[...]} and {models:[{trials:[...]}]}
    if 'models' in data:
        trial = data['models'][0]['trials'][0]
    else:
        trial = data['trials'][0]

    states  = trial['states']          # list of [x, y, vx, vy]
    goal_xy = np.array(trial.get('goal_xy', [0.0, 0.0]), dtype=np.float32)
    idx     = _subsample(len(states), n_frames)

    from envs.pointmaze_visual import PointMazeVisual
    # build env_cfg from protocol or fall back to defaults
    proto = (data['models'][0].get('protocol', {})
             if 'models' in data else data.get('protocol', {}))
    env_section = proto.get('environment', proto)
    env_cfg = {'environment': {
        'maze_map':   env_section.get('maze_map', 'U'),
        'image_size': image_size,
        'action_scale': float(env_section.get('action_scale', 1.0)),
    }}
    env = PointMazeVisual(env_cfg)

    frames = []
    for i in idx:
        obs = env.reset_to_state(
            np.array(states[i], dtype=np.float32), goal_xy=goal_xy)
        if isinstance(obs, tuple):
            obs = obs[0]
        frames.append(np.array(obs))
    env.close()
    return frames


# ── figure assembly ───────────────────────────────────────────────────────────

def _add_border(ax, color: str, lw: float):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(color)
        sp.set_linewidth(lw)


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--cartpole',    required=True)
    p.add_argument('--cartpole-trial', type=int, default=9,
                   help='Trial index for CartPole (-1 = auto: largest initial state norm)')
    p.add_argument('--walker',      default=None,
                   help='Walker2D JSON with qpos_seq (from --save-states)')
    p.add_argument('--walker-gif',  default=None,
                   help='Walker2D GIF to subsample frames from directly')
    p.add_argument('--pointmaze',   required=True)
    p.add_argument('--out',         required=True)
    p.add_argument('--n-frames',    type=int,   default=10)
    p.add_argument('--image-size',  type=int,   default=128,
                   help='Render size for PointMaze env')
    p.add_argument('--cartpole-size', type=int, default=256,
                   help='Render size for CartPole env')
    p.add_argument('--cell-size',   type=float, default=1.5,
                   help='Inches per image cell')
    p.add_argument('--label-width', type=float, default=0.9,
                   help='Inches for the row-label column')
    args = p.parse_args()

    N = args.n_frames

    if args.walker is None and args.walker_gif is None:
        p.error('Provide either --walker (JSON) or --walker-gif (GIF path)')

    print('[render] CartPole …')
    cp_frames = render_cartpole(args.cartpole, N, args.cartpole_size,
                                trial_idx=args.cartpole_trial)
    print('[render] Walker2D …')
    if args.walker_gif:
        wk_frames = render_walker_from_gif(args.walker_gif, N)
    else:
        wk_frames = render_walker(args.walker, N)
    print('[render] PointMaze …')
    pm_frames = render_pointmaze(args.pointmaze, N, args.image_size)

    task_rows = [
        ('CartPole',  cp_frames),
        ('Walker2D',  wk_frames),
        ('PointMaze', pm_frames),
    ]

    cell  = args.cell_size
    lw    = args.label_width
    n_rows = len(task_rows)

    fig_w = lw + cell * N
    fig_h = cell * n_rows
    fig   = plt.figure(figsize=(fig_w, fig_h))

    gs = GridSpec(n_rows, N + 1, figure=fig,
                  width_ratios=[lw / cell] + [1] * N,
                  wspace=0.025, hspace=0.025,
                  left=0.0, right=1.0, top=1.0, bottom=0.0)

    for row_i, (label, frames) in enumerate(task_rows):
        # row label
        ax_lbl = fig.add_subplot(gs[row_i, 0])
        ax_lbl.text(0.5, 0.5, label,
                    ha='center', va='center',
                    fontweight='bold', fontsize=13,
                    rotation=90, rotation_mode='anchor',
                    transform=ax_lbl.transAxes)
        ax_lbl.axis('off')

        for col_i, frame in enumerate(frames):
            ax = fig.add_subplot(gs[row_i, col_i + 1])
            img = np.array(frame)
            if img.ndim == 2:                       # grayscale → RGB
                img = np.stack([img] * 3, axis=-1)
            ax.imshow(img, interpolation='bilinear')
            ax.set_xticks([])
            ax.set_yticks([])

            if col_i == 0:
                _add_border(ax, COLOR_FIRST, BORDER_LW)
            elif col_i == N - 1:
                _add_border(ax, COLOR_LAST, BORDER_LW)
            else:
                for sp in ax.spines.values():
                    sp.set_visible(False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f'[done] {out}')


if __name__ == '__main__':
    main()
