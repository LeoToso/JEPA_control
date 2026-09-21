#!/usr/bin/env python
"""Filmstrip visualisation of PointMaze LQR trajectories.

Each row shows N frames for one trial:
  - Frame 1   : always states[0]  (same initial state across all models)
  - Frames 2…N-1 : evenly spaced through the trajectory
  - Frame N   : the state closest to the goal (argmin xy-distance)

Works with JSON files produced by:
  - gt_lqr_gymnasium_pointmaze.py
  - lqr_dinowm_pointmaze.py
  - compare_lqr_pointmaze.py
(all use the {"models": [{"label": ..., "trials": [...]}]} schema
 and must have been run with the updated scripts that save 'states'.)

json_spec syntax:
  "path/to/results.json"    → reads models[0]
  "path/to/results.json:2"  → reads models[2]

Usage
-----
  MUJOCO_GL=egl python experiments/plot_lqr_trajectories_pointmaze.py \\
      --jsons   results/gt_oracle_lqr_gymnasium_fs5.json \\
                results/lqr_fs5_pointmaze.json:0 \\
      --labels  "GT Oracle LQR" "SIGreg+Rollout" \\
      --trial   5 --image-size 128 \\
      --out     results/lqr_traj_pointmaze.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from envs.pointmaze_visual import PointMazeVisual


# ── helpers ───────────────────────────────────────────────────────────────────

def load_trial(json_spec: str, trial_idx: int) -> tuple[dict, str]:
    """Return (trial_dict, model_label).

    json_spec may be:
      "path/to/results.json"        → reads models[0]
      "path/to/results.json:2"      → reads models[2]
    """
    parts = json_spec.rsplit(':', 1)
    try:
        model_idx = int(parts[-1])
        json_path = parts[0]
    except ValueError:
        json_path, model_idx = json_spec, 0

    with open(json_path) as f:
        data = json.load(f)
    model_entry = data['models'][model_idx]
    trials = model_entry['trials']
    if trial_idx >= len(trials):
        raise ValueError(
            f'{json_path}[{model_idx}]: trial {trial_idx} out of range '
            f'(only {len(trials)} trials)')
    trial = trials[trial_idx]
    if 'states' not in trial:
        raise KeyError(
            f"'states' missing in {json_path} model[{model_idx}] trial {trial_idx}. "
            "Re-run the LQR/CEM script with the updated version that saves states.")
    # goal_xy may be at trial level (LQR scripts) or model level (CEM scripts)
    if 'goal_xy' not in trial and 'goal_xy' in model_entry:
        trial = dict(trial)
        trial['goal_xy'] = model_entry['goal_xy']
    return trial, model_entry.get('label', Path(json_path).stem)


def make_render_env(image_size: int = 128, seed: int = 0) -> PointMazeVisual:
    env_cfg = {'environment': {'maze_map': 'U', 'image_size': image_size,
                               'action_scale': 1.0}}
    return PointMazeVisual(env_cfg, seed=seed)


def set_topdown_camera(env: PointMazeVisual,
                       distance: float = 6.0, azimuth: float = 90.0,
                       lookat: tuple = (0.0, 0.0)) -> None:
    """Set MuJoCo free camera to a bird's-eye view (elevation = -90°).

    Must be called after at least one render so the viewer is initialised.
    """
    try:
        import mujoco
        renderer = env._env.unwrapped.point_env.mujoco_renderer
        viewer = renderer._viewers.get('rgb_array') or renderer.viewer
        if viewer is None:
            return
        cam = viewer.cam
        cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
        cam.elevation = -90.0
        cam.azimuth   = float(azimuth)
        cam.distance  = float(distance)
        cam.lookat[0] = float(lookat[0])
        cam.lookat[1] = float(lookat[1])
        cam.lookat[2] = 0.0
    except Exception as e:
        print(f'[camera] warning: {e}')


def render_frames(states: list, goal_xy, env: PointMazeVisual,
                  n_frames: int) -> tuple[list, list[int], int]:
    """Render n_frames frames; first = states[0], last = closest to goal.

    Returns (frames, indices, best_idx).
    All models share the same states[0] (deterministic seed scheme), so
    the first column is guaranteed to show identical initial conditions.
    """
    n = len(states)
    states_arr = np.array(states, dtype=np.float32)

    # Index of the state closest to the goal
    if goal_xy is not None:
        goal = np.array(goal_xy, dtype=np.float32)[:2]
        xy_err = np.linalg.norm(states_arr[:, :2] - goal[None], axis=1)
        best_idx = int(np.argmin(xy_err))
    else:
        best_idx = n - 1

    # All frames span [0, best_idx] so the row reads chronologically and
    # the last column is always the closest-to-goal moment.
    indices = list(np.round(np.linspace(0, best_idx, n_frames)).astype(int))

    frames = []
    for idx in indices:
        state = states_arr[idx]
        obs, _, _ = env.reset_to_state(state, goal_xy=goal_xy)
        frames.append(obs)
    return frames, indices, best_idx


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='PointMaze LQR trajectory filmstrip.')
    p.add_argument('--jsons',   nargs='+', required=True,
                   help='JSON specs: "path.json" or "path.json:model_idx"')
    p.add_argument('--labels',  nargs='+', required=True,
                   help='Row label for each JSON spec (same order)')
    p.add_argument('--trial',   type=int, default=0,
                   help='Trial index to visualise from each JSON')
    p.add_argument('--n-frames', type=int, default=10,
                   help='Number of rendered frames per row')
    p.add_argument('--image-size', type=int, default=128,
                   help='Render resolution (square). 128 or 256 recommended.')
    p.add_argument('--label-fontsize', type=float, default=11)
    p.add_argument('--show-step', action='store_true',
                   help='Annotate each frame with its step index')
    p.add_argument('--top-view', action='store_true',
                   help='Render from a top-down (bird\'s-eye) camera at -90° elevation')
    p.add_argument('--cam-distance', type=float, default=6.0,
                   help='Camera distance for top-view (tune to fit maze in frame)')
    p.add_argument('--cam-azimuth',  type=float, default=90.0,
                   help='Camera azimuth for top-view')
    p.add_argument('--cam-lookat',   type=float, nargs=2, default=[0.0, 0.0],
                   metavar=('X', 'Y'), help='Camera lookat XY for top-view')
    p.add_argument('--out',     required=True, help='Output image path')
    args = p.parse_args()

    if len(args.jsons) != len(args.labels):
        p.error('--jsons and --labels must have the same number of entries')

    n_rows = len(args.jsons)
    n_cols = args.n_frames

    env = make_render_env(image_size=args.image_size)

    if args.top_view:
        # Warm up the viewer with one render so the camera object exists
        env._env.reset(seed=0)
        env._env.render()
        set_topdown_camera(env, distance=args.cam_distance,
                           azimuth=args.cam_azimuth,
                           lookat=tuple(args.cam_lookat))

    fig = plt.figure(figsize=(n_cols * 1.8, n_rows * 2.1))
    gs  = gridspec.GridSpec(
        n_rows, n_cols,
        hspace=0.06, wspace=0.03,
        left=0.12, right=0.98, top=0.97, bottom=0.03)

    for row, (json_spec, label) in enumerate(zip(args.jsons, args.labels)):
        trial, auto_label = load_trial(json_spec, args.trial)
        display_label = label or auto_label
        states  = trial['states']
        goal_xy = trial.get('goal_xy', None)
        success = trial.get('success', None)

        frames, indices, best_idx = render_frames(states, goal_xy, env, n_cols)

        for col, (frame, step_idx) in enumerate(zip(frames, indices)):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(frame)

            is_first = (col == 0)
            is_last  = (col == n_cols - 1)   # closest-to-goal frame
            border_lw    = 2.5 if (is_first or is_last) else 0.8
            border_color = ('#44ff88' if (is_last and success)
                            else '#ff4444' if (is_last and success is False)
                            else '#888888')
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(border_color)
                spine.set_linewidth(border_lw)
            ax.set_xticks([])
            ax.set_yticks([])

            if args.show_step:
                tag = f't={step_idx}*' if step_idx == best_idx else f't={step_idx}'
                ax.set_title(tag, fontsize=6, pad=1.5, color='gray')

        fig.text(
            0.005, 1. - (row + 0.5) / n_rows,
            display_label,
            va='center', ha='left',
            fontsize=args.label_fontsize,
            fontweight='bold',
            rotation=90)

    env.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'[done] {out}')


if __name__ == '__main__':
    main()
