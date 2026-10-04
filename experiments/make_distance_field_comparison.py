#!/usr/bin/env python
"""4-panel row figure comparing latent goal-distance fields across models.

Panels (left → right):
  MSP + EP-IDM  |  1SP + EP-IDM  |  MSP + SIG  |  DINO-WM

Usage
-----
  MUJOCO_GL=egl python experiments/make_distance_field_comparison.py \\
      --smwm-ckpts \\
          /mnt/t7shield/jepa_results/pointmaze_smwm_rollout_endpoint_inv_act1_seed42/model_final.pt \\
          /mnt/t7shield/jepa_results/pointmaze_smwm_fwd_endpoint_inv_act1_seed42/model_final.pt \\
          /mnt/t7shield/jepa_results/pointmaze_sigreg_rollout_fs5_act1_200ep_seed42/model_final.pt \\
      --smwm-cfgs \\
          configs/pointmaze_jepa_rollout_endpoint_inverse_fs5_act1.yaml \\
          configs/pointmaze_jepa_fwd_endpoint_inverse_fs5_act1.yaml \\
          configs/pointmaze_jepa_sigreg_rollout_fs5_act1.yaml \\
      --smwm-titles "MSP + EP-IDM" "1SP + EP-IDM" "MSP + SIG" \\
      --dino-wm-dir ~/dino_wm \\
      --dino-ckpt /mnt/t7shield/jepa_results/dinowm_checkpoints/outputs/point_maze/checkpoints/model_latest.pth \\
      --goal-xy -0.20 1.05 \\
      --output results/distance_field_comparison.pdf
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

from experiments.probe_utils import encode_obs, load_bundle
from envs.pointmaze_visual import PointMazeVisual

# ── reuse panel_distance_field_grid and wall-checker from existing script ──────
from probe_pointmaze_landscapes import panel_distance_field_grid, _build_wall_checker


# ── DINO-WM constants (must match training) ───────────────────────────────────
D_VIS      = 384
D_PROP_OUT = 10
D_ACT_OUT  = 10
D_TOTAL    = D_VIS + D_PROP_OUT + D_ACT_OUT
ACT_IN_DIM = 10
PROP_IN_DIM = 4
_NORM_MEAN = torch.tensor([0.5, 0.5, 0.5])
_NORM_STD  = torch.tensor([0.5, 0.5, 0.5])


def load_dinowm(ckpt_path: str, dino_wm_dir: str, device: torch.device) -> dict:
    p = str(Path(dino_wm_dir).expanduser().resolve())
    if p not in sys.path:
        sys.path.insert(0, p)
    # Clear any cached 'models' package imported by SMWM utilities so that
    # dino_wm's models.dino is found instead.
    for key in list(sys.modules.keys()):
        if key == 'models' or key.startswith('models.'):
            del sys.modules[key]
    from models.dino import DinoV2Encoder
    print('[DINO-WM] loading …')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    parts = {}
    for k in ['predictor', 'proprio_encoder', 'action_encoder']:
        parts[k] = ckpt[k].to(device).eval()
    parts['encoder'] = DinoV2Encoder(
        name='dinov2_vits14', feature_key='x_norm_patchtokens').to(device).eval()
    return parts


@torch.no_grad()
def encode_dinowm(parts: dict, obs_hwc: np.ndarray,
                  state: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(obs_hwc.copy()).float().permute(2, 0, 1) / 255.0
    if x.shape[-2] != 196 or x.shape[-1] != 196:
        x = F.interpolate(x.unsqueeze(0), (196, 196),
                          mode='bilinear', align_corners=False).squeeze(0)
    x = (x - _NORM_MEAN[:, None, None]) / _NORM_STD[:, None, None]
    x = x.unsqueeze(0).to(device)
    vis  = parts['encoder'](x)                                      # (1, N, 384)
    N    = vis.shape[1]
    dtype_pe = next(parts['proprio_encoder'].parameters()).dtype
    prop = torch.as_tensor(state[:PROP_IN_DIM].astype(np.float32),
                           device=device).to(dtype_pe).unsqueeze(0).unsqueeze(0)
    prop_emb = parts['proprio_encoder'](prop)[:, 0:1, :].expand(-1, N, -1)
    dtype_ae = next(parts['action_encoder'].parameters()).dtype
    zero = torch.zeros(1, 1, ACT_IN_DIM, device=device, dtype=dtype_ae)
    act_emb  = parts['action_encoder'](zero)[:, 0:1, :].expand(-1, N, -1)
    tokens   = torch.cat([vis, prop_emb, act_emb], dim=2)          # (1, N, D_TOTAL)
    return tokens[0].mean(dim=0).cpu().numpy()                      # (D_TOTAL,)


@torch.no_grad()
def dinowm_distance_field(env, parts, bounds, goal_xy,
                           grid_size: int, wall_tol: float,
                           cmap: str, device: torch.device):
    """Compute and plot DINO-WM distance field — same logic as panel D."""
    (x0, x1), (y0, y1) = bounds
    is_wall = _build_wall_checker(env)

    goal_state = np.array([goal_xy[0], goal_xy[1], 0., 0.], dtype=np.float32)
    obs_g, st_g, _ = env.reset_to_state(goal_state, goal_xy=goal_xy)
    z_goal = encode_dinowm(parts, obs_g, st_g, device)
    print(f'  [DINO-WM] goal encoded  xy={np.round(goal_xy, 3)}')

    xs = np.linspace(x0, x1, grid_size)
    ys = np.linspace(y0, y1, grid_size)
    grid_dist = np.full((grid_size, grid_size), np.nan)
    n_valid = 0

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            if is_wall is not None and is_wall(x, y):
                continue
            target = np.array([x, y, 0., 0.], dtype=np.float32)
            obs, actual, _ = env.reset_to_state(target, goal_xy=goal_xy)
            if is_wall is None and np.linalg.norm(actual[:2] - target[:2]) > wall_tol:
                continue
            z = encode_dinowm(parts, obs, actual, device)
            grid_dist[i, j] = float(np.linalg.norm(z - z_goal))
            n_valid += 1
        if (i + 1) % 5 == 0:
            print(f'  [DINO-WM] row {i+1:3d}/{grid_size}  valid={n_valid}', flush=True)

    valid_vals = grid_dist[~np.isnan(grid_dist)]
    vmin = float(np.percentile(valid_vals, 2))
    vmax = float(np.percentile(valid_vals, 98))
    print(f'  [DINO-WM] done: {n_valid}/{grid_size**2} cells  '
          f'dist=[{vmin:.2f}, {vmax:.2f}]')
    return grid_dist, xs, ys, vmin, vmax


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='4-panel latent distance-field comparison figure.')
    p.add_argument('--smwm-ckpts', nargs=3, required=True,
                   metavar='CKPT',
                   help='3 SMWM checkpoint paths: MSP+EP-IDM, 1SP+EP-IDM, MSP+SIG')
    p.add_argument('--smwm-cfgs',  nargs=3, required=True,
                   metavar='CFG',
                   help='Matching YAML configs for the 3 SMWM checkpoints')
    p.add_argument('--smwm-titles', nargs=3,
                   default=['MSP + EP-IDM', '1SP + EP-IDM', 'MSP + SIG'])
    p.add_argument('--dino-wm-dir', required=True)
    p.add_argument('--dino-ckpt',   required=True)
    p.add_argument('--dino-title',  default='DINO-WM')
    p.add_argument('--goal-xy',     type=float, nargs=2, default=[-0.20, 1.05])
    p.add_argument('--grid-size',   type=int,   default=45)
    p.add_argument('--wall-tol',    type=float, default=0.18)
    p.add_argument('--cmap',        default='viridis_r')
    p.add_argument('--maze-bounds', type=float, nargs=4,
                   default=[-2.0, 2.0, -2.0, 2.0],
                   help='x0 x1 y0 y1 — matches probe_pointmaze_landscapes.py default')
    p.add_argument('--device',      default='cuda')
    p.add_argument('--output',      required=True)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    goal_xy = args.goal_xy
    x0, x1, y0, y1 = args.maze_bounds
    bounds = ((x0, x1), (y0, y1))

    all_titles = list(args.smwm_titles) + [args.dino_title]
    n_panels   = len(all_titles)   # 4

    fig, axes = plt.subplots(1, n_panels,
                             figsize=(4.5 * n_panels, 4.5),
                             constrained_layout=True)

    env = None   # created from first bundle's env_cfg

    # ── SMWM panels ───────────────────────────────────────────────────────────
    for idx, (ckpt, cfg, title) in enumerate(
            zip(args.smwm_ckpts, args.smwm_cfgs, args.smwm_titles)):
        print(f'\n[{title}] loading bundle …')
        bundle = load_bundle(ckpt, cfg, args.device)

        if env is None:
            env_cfg = bundle.get('env_cfg', {})
            if 'environment' not in env_cfg:
                model_img_sz = int(bundle.get('model_cfg', {}).get('image_size', 64))
                env_cfg['environment'] = {'maze_map': 'U', 'image_size': model_img_sz,
                                          'frame_skip': 5}
            env = PointMazeVisual(env_cfg, seed=0)

        ax     = axes[idx]
        ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
        print(f'[{title}] computing {args.grid_size}×{args.grid_size} distance field …')
        im = panel_distance_field_grid(
            ax, env, bundle, bounds, goal_xy,
            grid_size=args.grid_size,
            cmap=args.cmap,
            wall_tol=args.wall_tol,
            bg='white',
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label('latent distance to goal', fontsize=9)
        cb.ax.tick_params(labelsize=8)

    # ── DINO-WM panel ─────────────────────────────────────────────────────────
    print(f'\n[{args.dino_title}] loading …')
    parts = load_dinowm(args.dino_ckpt, args.dino_wm_dir, device)
    ax    = axes[3]
    ax.set_title(args.dino_title, fontsize=13, fontweight='bold', pad=8)
    print(f'[{args.dino_title}] computing {args.grid_size}×{args.grid_size} distance field …')

    grid_dist, xs, ys, vmin, vmax = dinowm_distance_field(
        env, parts, bounds, goal_xy,
        grid_size=args.grid_size,
        wall_tol=args.wall_tol,
        cmap=args.cmap,
        device=device,
    )
    masked = np.ma.masked_invalid(grid_dist)
    ax.set_facecolor('white')
    im = ax.imshow(masked, origin='lower',
                   extent=[x0, x1, y0, y1],
                   cmap=args.cmap, vmin=vmin, vmax=vmax,
                   interpolation='bilinear', aspect='equal')
    try:
        levels = np.linspace(vmin, vmax, 7)
        ax.contour(xs, ys, masked, levels=levels,
                   colors='#555555', linewidths=0.4, alpha=0.35)
    except Exception:
        pass
    ax.scatter(*goal_xy, s=280, marker='*', c='#FFD60A', zorder=10,
               edgecolors='#333333', linewidths=0.8)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.axis('off')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label('latent distance to goal', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    env.close()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n[saved] {out}')


if __name__ == '__main__':
    main()
