#!/usr/bin/env python
"""PointMaze JEPA probe landscapes.

Panels
------
  A  Imagination rollout fan   — K predicted futures from diverse starts
  B  Latent UMAP / PCA         — coloured by 2-D maze position (topology test)
  C  Latent interpolation       — decoded paths between start↔goal pairs
  D  Latent distance field      — each state coloured by ‖z − z_goal‖₂

Usage
-----
  # All three classic panels
  MUJOCO_GL=egl python experiments/probe_pointmaze_landscapes.py \\
      --ckpt  /mnt/.../model_final.pt \\
      --cfg   configs/pointmaze_smwm_sigreg_rollout_fs5.yaml \\
      --output results/probe_landscapes_sigreg_rollout_fs5.pdf

  # Single distance-field panel (no probe fitting needed)
  MUJOCO_GL=egl python experiments/probe_pointmaze_landscapes.py \\
      --ckpt  /mnt/.../model_final.pt \\
      --cfg   configs/pointmaze_smwm_sigreg_rollout_fs5.yaml \\
      --panel D --goal-xy -0.20 1.05 \\
      --output results/probe_distance_field.pdf

Optional: pip install umap-learn  (falls back to PCA for panel B if absent)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score

from sensorimotor_probe_utils import encode_obs, load_bundle
from envs.pointmaze_visual import PointMazeVisual


# ── data & probe ─────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_data(env, bundle, n_episodes: int = 80, n_steps: int = 150):
    """Random rollouts → aligned (latents, xy) arrays for probe + UMAP."""
    zs, xys = [], []
    for _ in range(n_episodes):
        obs, state, _ = env.reset()
        prev_obs = obs.copy()
        for _ in range(n_steps):
            z = encode_obs(bundle, obs, prev_obs, state)
            zs.append(z.cpu().numpy().flatten())
            xys.append(state[:2].copy())
            u = np.random.uniform(-1, 1, 2).astype(np.float32)
            prev_obs = obs.copy()
            obs, state, _, done, _ = env.step(u)
            if done:
                break
    return np.array(zs), np.array(xys)


def fit_probe(zs, xys):
    probe = Ridge(alpha=1.0)
    probe.fit(zs, xys)
    r2 = r2_score(xys, probe.predict(zs))
    print(f'[probe] ridge R²={r2:.3f}  (1.0 = perfect position decoding)')
    return probe


# ── helper: data-driven positions ────────────────────────────────────────────

def diverse_positions(xys, n=6):
    """n positions spread across the collected maze states via KMeans."""
    km = KMeans(n_clusters=n, random_state=42, n_init=10).fit(xys)
    positions = []
    for c in km.cluster_centers_:
        i = np.argmin(np.linalg.norm(xys - c, axis=1))
        positions.append(xys[i])
    return np.array(positions)


def data_bounds(xys, pad=0.15):
    """Return ((x0,x1), (y0,y1)) with padding around collected data."""
    xp = (xys[:, 0].max() - xys[:, 0].min()) * pad
    yp = (xys[:, 1].max() - xys[:, 1].min()) * pad
    return ((xys[:, 0].min() - xp, xys[:, 0].max() + xp),
            (xys[:, 1].min() - yp, xys[:, 1].max() + yp))


# ── latent rollout ────────────────────────────────────────────────────────────

@torch.no_grad()
def latent_rollout(bundle, z0, actions):
    """Pure latent rollout.  z0: (d,) tensor  |  actions: (H, 2) array."""
    model, scale = bundle['model'], bundle['action_scale']
    z = z0.flatten()
    zs = [z]
    for u in actions:
        a_raw = torch.tensor(u, dtype=z.dtype, device=z.device).reshape(1, 2) / scale
        a_ctx = model.expand_action(a_raw).unsqueeze(1)   # (1, 1, ctx)
        z = model.predict(z[None, None], a_ctx)[0, 0]     # (d,)
        zs.append(z)
    return zs


def decode_zs(zs, probe, bounds=None):
    """List of (d,) tensors → (N, 2) xy positions; optionally clipped to bounds."""
    Z = np.stack([z.detach().cpu().numpy().flatten() for z in zs])
    xy = probe.predict(Z)
    if bounds is not None:
        (x0, x1), (y0, y1) = bounds
        xy[:, 0] = np.clip(xy[:, 0], x0, x1)
        xy[:, 1] = np.clip(xy[:, 1], y0, y1)
    return xy


# ── maze density background ───────────────────────────────────────────────────

def maze_bg(ax, xys, bounds, surface='#10101a'):
    (x0, x1), (y0, y1) = bounds
    counts, xe, ye = np.histogram2d(
        xys[:, 0], xys[:, 1], bins=40,
        range=[[x0, x1], [y0, y1]])
    counts = np.log1p(counts).T
    ax.pcolormesh(xe, ye, counts, cmap='Blues', alpha=0.28, zorder=0,
                  vmin=0, vmax=counts.max())
    ax.set_facecolor(surface)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


# ── geometry-based wall checker ───────────────────────────────────────────────

def _build_wall_checker(env):
    """Return is_wall(x, y) -> bool using maze geometry (no physics needed).

    Uses gymnasium-robotics' coordinate system:
      i = floor((y_map_center - y) / scaling)
      j = floor((x + x_map_center) / scaling)
    Wall cells have maze_map[i][j] == 1.
    Returns None if the maze cannot be introspected.
    """
    import math
    try:
        inner = env._env.unwrapped
        maze  = inner.maze
        maze_map     = maze.maze_map          # list[list[int|str]], 1=wall
        scaling      = maze.maze_size_scaling  # units per cell (usually 1.0)
        x_ctr        = maze.x_map_center
        y_ctr        = maze.y_map_center
        nrows        = maze.map_length
        ncols        = maze.map_width

        def is_wall(x, y):
            i = math.floor((y_ctr - y) / scaling)
            j = math.floor((x + x_ctr) / scaling)
            if i < 0 or i >= nrows or j < 0 or j >= ncols:
                return True   # out of bounds → treat as wall
            return maze_map[i][j] == 1

        return is_wall
    except Exception:
        return None


# ── panel D — latent distance field (grid) ───────────────────────────────────

@torch.no_grad()
def panel_distance_field_grid(ax, env, bundle, bounds, goal_xy,
                               grid_size: int = 45,
                               cmap: str = 'viridis_r',
                               wall_tol: float = 0.18,
                               bg: str = 'white'):
    """Evaluate ‖z − z_goal‖ on a regular (x,y) grid and display as imshow.

    Wall positions are detected geometrically from the maze map (fast, exact).
    Physics-based displacement fallback is used only when geometry is unavailable.
    Result: smooth filled landscape with walls correctly masked.
    """
    (x0, x1), (y0, y1) = bounds

    is_wall = _build_wall_checker(env)
    if is_wall is not None:
        print('  wall detection: geometry-based (maze_map)')
    else:
        print(f'  wall detection: physics fallback (wall_tol={wall_tol})')

    goal_state = np.array([goal_xy[0], goal_xy[1], 0., 0.], dtype=np.float32)
    obs_g, st_g, _ = env.reset_to_state(goal_state, goal_xy=goal_xy)
    z_goal = encode_obs(bundle, obs_g, obs_g, st_g).cpu().numpy().flatten()
    print(f'  goal encoded  xy={np.round(goal_xy, 3)}')

    xs = np.linspace(x0, x1, grid_size)
    ys = np.linspace(y0, y1, grid_size)
    grid_dist = np.full((grid_size, grid_size), np.nan)
    n_valid = 0

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            if is_wall is not None and is_wall(x, y):
                continue   # geometry-based: skip wall cells without env reset

            target = np.array([x, y, 0., 0.], dtype=np.float32)
            obs, actual, _ = env.reset_to_state(target, goal_xy=goal_xy)

            if is_wall is None:
                # physics fallback: check if ball was displaced into a wall
                if np.linalg.norm(actual[:2] - target[:2]) > wall_tol:
                    continue

            z = encode_obs(bundle, obs, obs, actual).cpu().numpy().flatten()
            grid_dist[i, j] = float(np.linalg.norm(z - z_goal))
            n_valid += 1
        if (i + 1) % 5 == 0:
            print(f'  row {i+1:3d}/{grid_size}  valid={n_valid}', flush=True)

    valid_vals = grid_dist[~np.isnan(grid_dist)]
    vmin = float(np.percentile(valid_vals, 2))
    vmax = float(np.percentile(valid_vals, 98))
    print(f'  done: {n_valid}/{grid_size**2} valid cells  '
          f'dist range=[{vmin:.2f}, {vmax:.2f}]')

    is_dark   = bg.lower() not in ('white', '#ffffff', '#fff', 'w')
    line_clr  = 'white'    if is_dark else '#555555'
    edge_clr  = 'white'    if is_dark else '#333333'

    masked = np.ma.masked_invalid(grid_dist)
    ax.set_facecolor(bg)
    im = ax.imshow(masked, origin='lower',
                   extent=[x0, x1, y0, y1],
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='bilinear', aspect='equal')

    # Topographic contour lines
    try:
        levels = np.linspace(vmin, vmax, 7)
        ax.contour(xs, ys, masked,
                   levels=levels, colors=line_clr,
                   linewidths=0.4, alpha=0.35)
    except Exception:
        pass

    ax.scatter(*goal_xy, s=280, marker='*', c='#FFD60A', zorder=10,
               edgecolors=edge_clr, linewidths=0.8)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.axis('off')
    return im


# ── panel A — imagination fan ─────────────────────────────────────────────────

COLORS = ['#4CC9F0', '#F72585', '#7209B7', '#3A0CA3', '#4361EE', '#48CAE4']


@torch.no_grad()
def panel_a(ax, env, bundle, probe, xys, bounds, rng, K=15, H=12):
    maze_bg(ax, xys, bounds)
    (x0, x1), (y0, y1) = bounds

    starts_xy = diverse_positions(xys, n=6)
    # goal = mean of top-10% positions by x+y (far corner of U-maze)
    score   = xys[:, 0] + xys[:, 1]
    top_idx = np.argsort(score)[-max(1, len(xys) // 10):]
    goal_xy = xys[top_idx].mean(axis=0)

    for start_xy, color in zip(starts_xy, COLORS):
        state_s = np.array([start_xy[0], start_xy[1], 0., 0.], dtype=np.float32)
        obs0, state0, _ = env.reset_to_state(state_s.copy())
        z0 = encode_obs(bundle, obs0, obs0, state0).flatten()

        for _ in range(K):
            actions = rng.uniform(-1, 1, size=(H, 2)).astype(np.float32)
            zs  = latent_rollout(bundle, z0, actions)
            xy  = decode_zs(zs, probe, bounds)
            dist = np.linalg.norm(xy[-1] - goal_xy)
            alpha = float(np.clip(0.7 - dist / (x1 - x0), 0.07, 0.7))
            ax.plot(xy[:, 0], xy[:, 1], color=color, alpha=alpha,
                    lw=0.8, zorder=2, solid_capstyle='round')

        ax.scatter(*start_xy, s=50, c=color, zorder=5,
                   edgecolors='white', linewidths=0.7)

    ax.scatter(*goal_xy, s=120, c='#FFD60A', marker='*', zorder=6,
               edgecolors='white', linewidths=0.7)
    ax.set_aspect('equal', adjustable='datalim')
    ax.axis('off')


# ── panel B — UMAP / PCA topology ────────────────────────────────────────────

def panel_b(ax, zs, xys):
    try:
        import umap as _umap
        emb = _umap.UMAP(n_components=2, n_neighbors=25, min_dist=0.06,
                         random_state=42, verbose=False).fit_transform(zs)
    except ImportError:
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2, random_state=42).fit_transform(zs)

    xn = (xys[:, 0] - xys[:, 0].min()) / (xys[:, 0].ptp() + 1e-8)
    yn = (xys[:, 1] - xys[:, 1].min()) / (xys[:, 1].ptp() + 1e-8)

    h   = xn * 0.72
    s   = np.full_like(h, 0.88)
    v   = 0.45 + yn * 0.55
    rgb = hsv_to_rgb(np.stack([h, s, v], axis=-1))

    ax.scatter(emb[:, 0], emb[:, 1], c=rgb, s=5, alpha=0.6,
               linewidths=0, rasterized=True)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_facecolor('#10101a')
    ax.axis('off')


# ── panel C — latent interpolation ───────────────────────────────────────────

@torch.no_grad()
def panel_c(ax, env, bundle, probe, xys, bounds, n_steps=30):
    maze_bg(ax, xys, bounds)

    positions = diverse_positions(xys, n=8)
    pairs = []
    for i in range(min(5, len(positions) // 2)):
        pairs.append((positions[i], positions[-(i + 1)]))

    for (xy_s, xy_g), color in zip(pairs, COLORS):
        s0 = np.array([xy_s[0], xy_s[1], 0., 0.], dtype=np.float32)
        g0 = np.array([xy_g[0], xy_g[1], 0., 0.], dtype=np.float32)

        obs_s, st_s, _ = env.reset_to_state(s0.copy())
        obs_g, st_g, _ = env.reset_to_state(g0.copy())

        z_s = encode_obs(bundle, obs_s, obs_s, st_s).flatten()
        z_g = encode_obs(bundle, obs_g, obs_g, st_g).flatten()

        alphas    = np.linspace(0., 1., n_steps)
        z_interps = [(1 - a) * z_s + a * z_g for a in alphas]
        xy        = decode_zs(z_interps, probe, bounds)

        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=2.0, alpha=0.9,
                zorder=3, solid_capstyle='round')
        ax.scatter(*xy_s, s=58, c=color, zorder=5, marker='o',
                   edgecolors='white', linewidths=0.7)
        ax.scatter(*xy_g, s=58, c=color, zorder=5, marker='s',
                   edgecolors='white', linewidths=0.7)

    ax.set_aspect('equal', adjustable='datalim')
    ax.axis('off')


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='PointMaze JEPA probe landscapes (panels A / B / C / D).')
    p.add_argument('--ckpt',             required=True)
    p.add_argument('--cfg',              required=True)
    p.add_argument('--output',           required=True)
    p.add_argument('--panel',            choices=['A', 'B', 'C', 'D', 'all'],
                   default='all',
                   help='Which panel to produce. D = grid distance field.')
    p.add_argument('--goal-xy',          type=float, nargs=2, default=None,
                   metavar=('X', 'Y'),
                   help='Goal position for panel D.')
    p.add_argument('--maze-bounds',      type=float, nargs=4,
                   default=[-2.0, 2.0, -2.0, 2.0],
                   metavar=('X0', 'X1', 'Y0', 'Y1'),
                   help='Grid extent for panel D (default covers PointMaze U-maze).')
    p.add_argument('--grid-size',        type=int,   default=45,
                   help='Grid resolution for panel D (default 45 → 2025 evals).')
    p.add_argument('--wall-tol',         type=float, default=0.18,
                   help='Max displacement to count a grid cell as free space.')
    p.add_argument('--cmap',             default='viridis_r',
                   help='Colormap for panel D.')
    p.add_argument('--bg',               default='white',
                   help='Figure background colour for panel D (default: white).')
    p.add_argument('--n-probe-episodes', type=int,   default=80,
                   help='Episodes for data collection (panels A/B/C).')
    p.add_argument('--n-steps',          type=int,   default=150,
                   help='Steps per collection episode (panels A/B/C).')
    p.add_argument('--seed',             type=int,   default=0)
    p.add_argument('--device',           default='cuda')
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('[probe] loading model …')
    bundle = load_bundle(args.ckpt, args.cfg, args.device)

    env_cfg = bundle.get('env_cfg', {})
    if 'environment' not in env_cfg:
        # Use model's image_size so frozen encoders (DINOv2 patch=14, iBOT patch=16)
        # receive correctly-sized images without a separate resize step.
        model_img_sz = int(bundle['model_cfg'].get('image_size', 64))
        env_cfg['environment'] = {'maze_map': 'U', 'image_size': model_img_sz,
                                  'action_scale': 1.0}
    env = PointMazeVisual(env_cfg, seed=args.seed)

    BG = '#10101a'
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # ── panel D — grid distance field (no data collection needed) ────────────
    if args.panel == 'D':
        if args.goal_xy is None:
            p.error('--goal-xy X Y is required for --panel D')
        bounds = ((args.maze_bounds[0], args.maze_bounds[1]),
                  (args.maze_bounds[2], args.maze_bounds[3]))
        goal_xy = np.array(args.goal_xy, dtype=np.float32)

        panel_bg  = args.bg
        is_dark   = panel_bg.lower() not in ('white', '#ffffff', '#fff', 'w')
        txt_clr   = '#cccccc' if is_dark else '#333333'
        out_clr   = '#555555' if is_dark else '#cccccc'

        fig, ax = plt.subplots(1, 1, figsize=(5, 5.6), facecolor=panel_bg)
        ax.set_facecolor(panel_bg)
        fig.subplots_adjust(left=0.04, right=0.86, top=0.97, bottom=0.03)

        print(f'[probe] panel D — {args.grid_size}×{args.grid_size} grid …')
        im = panel_distance_field_grid(
            ax, env, bundle, bounds, goal_xy,
            grid_size=args.grid_size, cmap=args.cmap,
            wall_tol=args.wall_tol, bg=panel_bg)

        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label('latent distance to goal', color=txt_clr,
                     fontsize=9, labelpad=8)
        cb.ax.yaxis.set_tick_params(color=txt_clr, labelsize=7)
        plt.setp(cb.ax.get_yticklabels(), color=txt_clr)
        cb.outline.set_edgecolor(out_clr)
        cb.outline.set_linewidth(0.5)

        fig.savefig(out, dpi=220, bbox_inches='tight', facecolor=panel_bg)
        plt.close(fig)
        print(f'[done] → {out}')
        env.close()
        return

    # ── panels A / B / C (or all) — collect data + fit probe ─────────────────
    print(f'[probe] collecting {args.n_probe_episodes} episodes × {args.n_steps} steps …')
    zs, xys = collect_data(env, bundle,
                           n_episodes=args.n_probe_episodes,
                           n_steps=args.n_steps)
    bounds = data_bounds(xys, pad=0.12)
    print(f'[probe] maze bounds  x={bounds[0]}  y={bounds[1]}')

    probe = fit_probe(zs, xys)

    if args.panel == 'all':
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), facecolor=BG,
                                 gridspec_kw={'wspace': 0.03})
        for ax in axes:
            ax.set_facecolor(BG)
        fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)

        print('[probe] panel A — imagination fan …')
        panel_a(axes[0], env, bundle, probe, xys, bounds, rng)
        print('[probe] panel B — UMAP topology …')
        panel_b(axes[1], zs, xys)
        print('[probe] panel C — latent interpolation …')
        panel_c(axes[2], env, bundle, probe, xys, bounds)

    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor=BG)
        ax.set_facecolor(BG)
        fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)
        if args.panel == 'A':
            print('[probe] panel A — imagination fan …')
            panel_a(ax, env, bundle, probe, xys, bounds, rng)
        elif args.panel == 'B':
            print('[probe] panel B — UMAP topology …')
            panel_b(ax, zs, xys)
        elif args.panel == 'C':
            print('[probe] panel C — latent interpolation …')
            panel_c(ax, env, bundle, probe, xys, bounds)

    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'[done] → {out}')
    env.close()


if __name__ == '__main__':
    main()
