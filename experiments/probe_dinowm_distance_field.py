#!/usr/bin/env python
"""Latent goal-distance landscape for the original DINO-WM checkpoint.

Evaluates ‖z_mean − z_goal_mean‖ on a regular (x,y) grid and plots as
a smooth heatmap — same panel D logic as probe_pointmaze_landscapes.py
but using DINO-WM's encoder / proprio / action modules.

Usage
-----
  MUJOCO_GL=egl python experiments/probe_dinowm_distance_field.py \\
      --dino-wm-dir ~/dino_wm \\
      --ckpt /mnt/.../model_latest.pth \\
      --goal-xy -0.20 1.05 \\
      --output results/probe_distance_field_dinowm.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from envs.pointmaze_visual import PointMazeVisual


def _build_wall_checker(env):
    """Return is_wall(x, y) -> bool using maze geometry (no physics needed).

    Mirrors the same helper in probe_pointmaze_landscapes.py.
    """
    import math
    try:
        inner    = env._env.unwrapped
        maze     = inner.maze
        maze_map = maze.maze_map
        scaling  = maze.maze_size_scaling
        x_ctr    = maze.x_map_center
        y_ctr    = maze.y_map_center
        nrows    = maze.map_length
        ncols    = maze.map_width

        def is_wall(x, y):
            i = math.floor((y_ctr - y) / scaling)
            j = math.floor((x + x_ctr) / scaling)
            if i < 0 or i >= nrows or j < 0 or j >= ncols:
                return True
            return maze_map[i][j] == 1

        return is_wall
    except Exception:
        return None


# ── DINO-WM constants (must match training) ───────────────────────────────────
FRAMESKIP   = 5
D_VIS       = 384
D_PROP_OUT  = 10
D_ACT_OUT   = 10
D_TOTAL     = D_VIS + D_PROP_OUT + D_ACT_OUT   # 404
ACT_IN_DIM  = FRAMESKIP * 2                     # 10
PROP_IN_DIM = 4

_NORM_MEAN = torch.tensor([0.5, 0.5, 0.5])
_NORM_STD  = torch.tensor([0.5, 0.5, 0.5])


# ── model loading ─────────────────────────────────────────────────────────────

def setup_dinowm(dino_wm_dir: str):
    p = str(Path(dino_wm_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def load_model(ckpt_path: str, dino_wm_dir: str, device: torch.device) -> dict:
    setup_dinowm(dino_wm_dir)
    from models.dino import DinoV2Encoder
    print('[DINO-WM] loading checkpoint …')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    parts = {}
    for k in ['predictor', 'proprio_encoder', 'action_encoder']:
        if k not in ckpt:
            raise KeyError(f"'{k}' missing — keys: {list(ckpt.keys())}")
        parts[k] = ckpt[k].to(device).eval()
    encoder = DinoV2Encoder(name='dinov2_vits14', feature_key='x_norm_patchtokens')
    parts['encoder'] = encoder.to(device).eval()
    n = sum(p.numel() for p in parts['predictor'].parameters())
    print(f'[DINO-WM] predictor params={n/1e6:.1f}M')
    return parts


# ── encoding ──────────────────────────────────────────────────────────────────

def preprocess_obs(obs_hwc: np.ndarray, device: torch.device,
                   target_size: int = 196) -> torch.Tensor:
    x = torch.from_numpy(obs_hwc.copy()).float().permute(2, 0, 1) / 255.0
    if x.shape[-2] != target_size or x.shape[-1] != target_size:
        x = F.interpolate(x.unsqueeze(0), size=(target_size, target_size),
                          mode='bilinear', align_corners=False).squeeze(0)
    x = (x - _NORM_MEAN[:, None, None]) / _NORM_STD[:, None, None]
    return x.unsqueeze(0).to(device)


@torch.no_grad()
def encode_mean(parts: dict, obs_hwc: np.ndarray,
                state: np.ndarray, device: torch.device) -> np.ndarray:
    """Encode (obs, state) → mean-pooled latent (D_TOTAL,) as numpy array."""
    vis = parts['encoder'](preprocess_obs(obs_hwc, device))   # (1, N, 384)
    N   = vis.shape[1]

    dtype_pe = next(parts['proprio_encoder'].parameters()).dtype
    prop_in  = torch.as_tensor(
        state[:PROP_IN_DIM].astype(np.float32), device=device
    ).to(dtype_pe).unsqueeze(0).unsqueeze(0)
    prop_emb = parts['proprio_encoder'](prop_in)               # (1, 1, D_PROP_OUT)
    prop_t   = prop_emb[:, 0:1, :].expand(-1, N, -1)

    dtype_ae = next(parts['action_encoder'].parameters()).dtype
    zero_act = torch.zeros(1, 1, ACT_IN_DIM, device=device, dtype=dtype_ae)
    act_emb  = parts['action_encoder'](zero_act)               # (1, 1, D_ACT_OUT)
    act_t    = act_emb[:, 0:1, :].expand(-1, N, -1)

    tokens = torch.cat([vis, prop_t, act_t], dim=2)            # (1, N, D_TOTAL)
    return tokens[0].mean(dim=0).cpu().numpy()                 # (D_TOTAL,)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='DINO-WM latent goal-distance landscape.')
    p.add_argument('--dino-wm-dir',  required=True,
                   help='Path to cloned gaoyuezhou/dino_wm repo')
    p.add_argument('--ckpt',         required=True,
                   help='DINO-WM checkpoint (model_latest.pth)')
    p.add_argument('--goal-xy',      type=float, nargs=2, required=True,
                   metavar=('X', 'Y'))
    p.add_argument('--maze-bounds',  type=float, nargs=4,
                   default=[-2.0, 2.0, -2.0, 2.0],
                   metavar=('X0', 'X1', 'Y0', 'Y1'),
                   help='Grid extent (default covers PointMaze U-maze).')
    p.add_argument('--grid-size',    type=int,   default=45)
    p.add_argument('--wall-tol',     type=float, default=0.18,
                   help='Max displacement to count a cell as free space.')
    p.add_argument('--image-size',   type=int,   default=196,
                   help='Render size (DINOv2 needs 196).')
    p.add_argument('--cmap',         default='viridis_r')
    p.add_argument('--bg',           default='white')
    p.add_argument('--device',       default='cuda')
    p.add_argument('--visual-only',  action='store_true', default=True,
                   help='Compare only visual (first D_VIS=384) dims — matches CEM alpha=0.')
    p.add_argument('--output',       required=True)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    parts  = load_model(args.ckpt, args.dino_wm_dir, device)

    env_cfg = {'environment': {'maze_map': 'U',
                               'image_size': args.image_size,
                               'action_scale': 1.0}}
    env = PointMazeVisual(env_cfg, seed=0)

    # Encode goal
    goal_xy    = np.array(args.goal_xy, dtype=np.float32)
    goal_state = np.array([goal_xy[0], goal_xy[1], 0., 0.], dtype=np.float32)
    obs_g, st_g, _ = env.reset_to_state(goal_state, goal_xy=goal_xy)
    z_goal = encode_mean(parts, obs_g, st_g, device)
    print(f'[probe] goal encoded  xy={np.round(goal_xy, 3)}')

    # Grid evaluation
    x0, x1, y0, y1 = args.maze_bounds
    xs = np.linspace(x0, x1, args.grid_size)
    ys = np.linspace(y0, y1, args.grid_size)
    grid_dist = np.full((args.grid_size, args.grid_size), np.nan)
    n_valid   = 0

    is_wall = _build_wall_checker(env)
    if is_wall is not None:
        print('  wall detection: geometry-based (maze_map)')
    else:
        print(f'  wall detection: physics fallback (wall_tol={args.wall_tol})')

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            if is_wall is not None and is_wall(x, y):
                continue   # skip wall cells without resetting env

            target = np.array([x, y, 0., 0.], dtype=np.float32)
            obs, actual, _ = env.reset_to_state(target, goal_xy=goal_xy)

            if is_wall is None:
                if np.linalg.norm(actual[:2] - target[:2]) > args.wall_tol:
                    continue

            z = encode_mean(parts, obs, actual, device)
            d = z[:D_VIS] - z_goal[:D_VIS] if args.visual_only else z - z_goal
            grid_dist[i, j] = float(np.linalg.norm(d))
            n_valid += 1
        if (i + 1) % 5 == 0:
            print(f'  row {i+1:3d}/{args.grid_size}  valid={n_valid}', flush=True)

    valid_vals = grid_dist[~np.isnan(grid_dist)]
    vmin = float(np.percentile(valid_vals, 2))
    vmax = float(np.percentile(valid_vals, 98))
    print(f'  {n_valid}/{args.grid_size**2} valid cells  '
          f'dist=[{vmin:.2f}, {vmax:.2f}]')

    # Plot
    is_dark  = args.bg.lower() not in ('white', '#ffffff', '#fff', 'w')
    line_clr = 'white'   if is_dark else '#555555'
    edge_clr = 'white'   if is_dark else '#333333'
    txt_clr  = '#cccccc' if is_dark else '#333333'
    out_clr  = '#555555' if is_dark else '#cccccc'

    masked = np.ma.masked_invalid(grid_dist)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5.6), facecolor=args.bg)
    ax.set_facecolor(args.bg)
    fig.subplots_adjust(left=0.04, right=0.86, top=0.97, bottom=0.03)

    im = ax.imshow(masked, origin='lower',
                   extent=[x0, x1, y0, y1],
                   cmap=args.cmap, vmin=vmin, vmax=vmax,
                   interpolation='bilinear', aspect='equal')
    try:
        levels = np.linspace(vmin, vmax, 7)
        ax.contour(xs, ys, masked, levels=levels,
                   colors=line_clr, linewidths=0.4, alpha=0.35)
    except Exception:
        pass

    ax.scatter(*goal_xy, s=280, marker='*', c='#FFD60A', zorder=10,
               edgecolors=edge_clr, linewidths=0.8)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.axis('off')

    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label('latent distance to goal', color=txt_clr, fontsize=9, labelpad=8)
    cb.ax.yaxis.set_tick_params(color=txt_clr, labelsize=7)
    plt.setp(cb.ax.get_yticklabels(), color=txt_clr)
    cb.outline.set_edgecolor(out_clr)
    cb.outline.set_linewidth(0.5)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches='tight', facecolor=args.bg)
    plt.close(fig)
    print(f'[done] → {out}')
    env.close()


if __name__ == '__main__':
    main()

