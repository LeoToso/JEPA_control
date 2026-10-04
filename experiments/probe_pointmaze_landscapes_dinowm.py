#!/usr/bin/env python
"""PointMaze probe landscapes for a DINO-WM checkpoint.

Three panels (no titles):
  A  Imagination rollout fan   — DINO-WM latent rollouts decoded via ridge probe
  B  Latent UMAP               — mean-pooled tokens coloured by maze position
  C  Latent interpolation       — decoded paths between start/goal pairs

Usage
-----
  MUJOCO_GL=egl python experiments/probe_pointmaze_landscapes_dinowm.py \\
      --dino-wm-dir ~/dino_wm \\
      --ckpt /mnt/.../model_latest.pth \\
      --output results/probe_landscapes_dinowm.pdf
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
from matplotlib.colors import hsv_to_rgb
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score

from envs.pointmaze_visual import PointMazeVisual

# ── DINO-WM constants ─────────────────────────────────────────────────────────
FRAMESKIP  = 5
D_VIS      = 384
D_PROP_OUT = 10
D_ACT_OUT  = 10
D_TOTAL    = D_VIS + D_PROP_OUT + D_ACT_OUT   # 404
ACT_IN_DIM = FRAMESKIP * 2                     # 10
PROP_IN_DIM = 4
IMG_SIZE   = 196


# ── model loading ─────────────────────────────────────────────────────────────

def setup_dinowm(dino_wm_dir: str):
    p = str(Path(dino_wm_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def load_model(ckpt_path: str, dino_wm_dir: str, device):
    setup_dinowm(dino_wm_dir)
    from models.dino import DinoV2Encoder
    print('[probe] loading checkpoint …')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    parts = {}
    for k in ['predictor', 'proprio_encoder', 'action_encoder']:
        if k not in ckpt:
            raise KeyError(f"'{k}' not in checkpoint. Keys: {list(ckpt.keys())}")
        parts[k] = ckpt[k].to(device).eval()
    parts['encoder'] = DinoV2Encoder(
        name='dinov2_vits14', feature_key='x_norm_patchtokens').to(device).eval()
    n = sum(p.numel() for p in parts['predictor'].parameters())
    print(f'[probe] predictor params={n/1e6:.1f}M  D_TOTAL={D_TOTAL}')
    return parts


# ── frame encoding ────────────────────────────────────────────────────────────

_NORM_MEAN = torch.tensor([0.5, 0.5, 0.5])
_NORM_STD  = torch.tensor([0.5, 0.5, 0.5])


def _preprocess(obs_hwc: np.ndarray, device) -> torch.Tensor:
    x = torch.from_numpy(obs_hwc.copy()).float().permute(2, 0, 1) / 255.0
    if x.shape[-1] != IMG_SIZE or x.shape[-2] != IMG_SIZE:
        x = F.interpolate(x.unsqueeze(0), (IMG_SIZE, IMG_SIZE),
                          mode='bilinear', align_corners=False).squeeze(0)
    x = (x - _NORM_MEAN[:, None, None]) / _NORM_STD[:, None, None]
    return x.unsqueeze(0).to(device)


@torch.no_grad()
def encode_frame(parts, obs_hwc, state, action_10d, device) -> torch.Tensor:
    """(obs, state, action) → (1, N_patches, D_TOTAL=404)."""
    if action_10d is None:
        action_10d = np.zeros(ACT_IN_DIM, dtype=np.float32)
    vis_emb = parts['encoder'](_preprocess(obs_hwc, device))          # (1,N,384)
    N = vis_emb.shape[1]
    _pe = next(parts['proprio_encoder'].parameters())
    prop_in = torch.as_tensor(state[:PROP_IN_DIM].astype(np.float32),
                               device=device).to(_pe.dtype).unsqueeze(0).unsqueeze(0)
    prop_emb   = parts['proprio_encoder'](prop_in)                    # (1,1,10)
    prop_tiled = prop_emb[:, 0:1, :].expand(-1, N, -1)
    _ae = next(parts['action_encoder'].parameters())
    act_in = torch.as_tensor(action_10d.astype(np.float32),
                              device=device).to(_ae.dtype).unsqueeze(0).unsqueeze(0)
    act_emb   = parts['action_encoder'](act_in)                       # (1,1,10)
    act_tiled = act_emb[:, 0:1, :].expand(-1, N, -1)
    return torch.cat([vis_emb, prop_tiled, act_tiled], dim=2)         # (1,N,404)


@torch.no_grad()
def build_context(parts, env, state, num_hist, device):
    """Encode a state → initial context (1, num_hist, N, D_TOTAL)."""
    obs, st, _ = env.reset_to_state(state.copy())
    frame = encode_frame(parts, obs, st, None, device)
    return frame.unsqueeze(1).expand(-1, num_hist, -1, -1).clone()


# ── latent rollout ────────────────────────────────────────────────────────────

@torch.no_grad()
def latent_rollout(parts, z_ctx, actions, num_hist, device):
    """Pure predictor rollout with action injection.

    z_ctx:   (1, T, N, D_TOTAL)
    actions: (H, ACT_IN_DIM=10)
    Returns list of (D_TOTAL,) mean-pooled tensors (H+1, including t=0).
    """
    _ae = next(parts['action_encoder'].parameters())
    N   = z_ctx.shape[2]
    means = [z_ctx[0, -1, :, :].mean(dim=0)]

    for u in actions:
        z_flat  = z_ctx.reshape(1, num_hist * N, D_TOTAL)
        z_pred  = parts['predictor'](z_flat).reshape(1, num_hist, N, D_TOTAL)
        nxt     = z_pred[:, -1:, :, :].clone()                       # (1,1,N,D)
        # inject chosen action into predicted patches' action slice
        act_t   = torch.as_tensor(u.astype(np.float32), device=device
                                   ).to(_ae.dtype).unsqueeze(0).unsqueeze(0)
        act_emb = parts['action_encoder'](act_t)                      # (1,1,10)
        nxt[0, 0, :, D_VIS + D_PROP_OUT:] = act_emb[:, 0, :].expand(N, -1).to(nxt.dtype)
        z_ctx   = torch.cat([z_ctx[:, 1:], nxt], dim=1)
        means.append(nxt[0, 0, :, :].mean(dim=0))

    return means


def decode_means(means, probe, bounds=None):
    """List of (D_TOTAL,) tensors → (N, 2) xy via proprio-slice probe."""
    # Use proprio slice only (indices D_VIS:D_VIS+D_PROP_OUT) — R²≈1.0 by
    # construction since it inverts the proprio encoder.
    Z  = np.stack([m.detach().cpu().numpy()[D_VIS:D_VIS + D_PROP_OUT] for m in means])
    xy = probe.predict(Z)
    if bounds is not None:
        (x0, x1), (y0, y1) = bounds
        xy[:, 0] = np.clip(xy[:, 0], x0, x1)
        xy[:, 1] = np.clip(xy[:, 1], y0, y1)
    return xy


# ── data collection & probe ───────────────────────────────────────────────────

@torch.no_grad()
def collect_data(env, parts, num_hist, device, n_episodes=80, n_steps=150):
    """Random rollouts → (latents, xy) covering as much of the maze as possible."""
    zs, xys = [], []
    for ep in range(n_episodes):
        obs, state, _ = env.reset()
        frame = encode_frame(parts, obs, state, None, device)
        z_ctx = frame.unsqueeze(1).expand(-1, num_hist, -1, -1).clone()

        for _ in range(n_steps):
            zs.append(z_ctx[0, -1, :, :].mean(dim=0).cpu().numpy())
            xys.append(state[:2].copy())

            u_2d  = np.random.uniform(-1, 1, 2).astype(np.float32)
            u_10d = np.tile(u_2d, FRAMESKIP).astype(np.float32)
            obs, state, _, done, _ = env.step(u_2d)

            new_frame = encode_frame(parts, obs, state, u_10d, device)
            z_ctx = torch.cat([z_ctx[:, 1:], new_frame.unsqueeze(1)], dim=1)
            if done:
                break

    return np.array(zs), np.array(xys)


def fit_probe(zs, xys):
    """Fit Ridge probe on the proprio slice only (D_VIS:D_VIS+D_PROP_OUT).

    The proprio encoder maps [x,y,vx,vy]→10-D, so probing this 10-D slice
    back to (x,y) gives R²≈1.0 and reliable position decoding from latent
    rollout predictions (whose proprio slice the predictor updates).
    """
    zs_prop = zs[:, D_VIS:D_VIS + D_PROP_OUT]
    probe = Ridge(alpha=1.0)
    probe.fit(zs_prop, xys)
    r2 = r2_score(xys, probe.predict(zs_prop))
    print(f'[probe] proprio-slice R²={r2:.3f}  '
          f'(probing {D_PROP_OUT}-D proprio embedding → (x,y))')
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
    """Return ((x0,x1), (y0,y1)) with a small padding around collected data."""
    xp = (xys[:, 0].max() - xys[:, 0].min()) * pad
    yp = (xys[:, 1].max() - xys[:, 1].min()) * pad
    return ((xys[:, 0].min() - xp, xys[:, 0].max() + xp),
            (xys[:, 1].min() - yp, xys[:, 1].max() + yp))


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


# ── panels ────────────────────────────────────────────────────────────────────

COLORS = ['#4CC9F0', '#F72585', '#7209B7', '#3A0CA3', '#4361EE', '#48CAE4']


@torch.no_grad()
def panel_a(ax, env, parts, probe, xys, bounds, rng, num_hist, device,
            K=15, H=12):
    maze_bg(ax, xys, bounds)
    (x0, x1), (y0, y1) = bounds

    starts_xy = diverse_positions(xys, n=6)
    # goal = mean of top-10% positions by x+y (opposite arm)
    score    = xys[:, 0] + xys[:, 1]
    top_idx  = np.argsort(score)[-max(1, len(xys) // 10):]
    goal_xy  = xys[top_idx].mean(axis=0)

    for xy_s, color in zip(starts_xy, COLORS):
        state_s = np.array([xy_s[0], xy_s[1], 0., 0.], dtype=np.float32)
        z_ctx   = build_context(parts, env, state_s, num_hist, device)

        for _ in range(K):
            actions = rng.uniform(-1, 1, size=(H, ACT_IN_DIM)).astype(np.float32)
            means   = latent_rollout(parts, z_ctx, actions, num_hist, device)
            xy      = decode_means(means, probe, bounds)
            dist    = np.linalg.norm(xy[-1] - goal_xy)
            alpha   = float(np.clip(0.7 - dist / (x1 - x0), 0.07, 0.7))
            ax.plot(xy[:, 0], xy[:, 1], color=color, alpha=alpha,
                    lw=0.8, zorder=2, solid_capstyle='round')

        ax.scatter(*xy_s, s=50, c=color, zorder=5,
                   edgecolors='white', linewidths=0.7)

    ax.scatter(*goal_xy, s=120, c='#FFD60A', marker='*', zorder=6,
               edgecolors='white', linewidths=0.7)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')


def panel_b(ax, zs, xys):
    # Use visual features only (DINOv2 patch means) — removes proprio/action
    # that would trivially organise by position, making the topology test real.
    from sklearn.decomposition import PCA
    zs_vis = zs[:, :D_VIS]
    # PCA to 64-D first: stabilises UMAP on high-dimensional inputs
    zs_pca = PCA(n_components=min(64, zs_vis.shape[1]),
                 random_state=42).fit_transform(zs_vis)
    try:
        import umap as _umap
        emb = _umap.UMAP(n_components=2, n_neighbors=50, min_dist=0.15,
                         random_state=42, verbose=False).fit_transform(zs_pca)
    except ImportError:
        emb = PCA(n_components=2, random_state=42).fit_transform(zs_pca)

    xn  = (xys[:, 0] - xys[:, 0].min()) / (xys[:, 0].max() - xys[:, 0].min() + 1e-8)
    yn  = (xys[:, 1] - xys[:, 1].min()) / (xys[:, 1].max() - xys[:, 1].min() + 1e-8)
    h   = xn * 0.72
    s   = np.full_like(h, 0.88)
    v   = 0.45 + yn * 0.55
    rgb = hsv_to_rgb(np.stack([h, s, v], axis=-1))

    ax.scatter(emb[:, 0], emb[:, 1], c=rgb, s=5, alpha=0.6,
               linewidths=0, rasterized=True)
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('#10101a')
    ax.axis('off')


@torch.no_grad()
def panel_c(ax, env, parts, probe, xys, bounds, num_hist, device,
            n_steps=30):
    maze_bg(ax, xys, bounds)

    # Pick 5 diverse positions and form pairs (each with the "opposite" cluster)
    positions = diverse_positions(xys, n=8)
    # Pair: each odd index with the furthest-apart even index
    pairs = []
    for i in range(min(5, len(positions) // 2)):
        s = positions[i]
        g = positions[-(i + 1)]
        pairs.append((s, g))

    for (xy_s, xy_g), color in zip(pairs, COLORS):
        s0 = np.array([xy_s[0], xy_s[1], 0., 0.], dtype=np.float32)
        g0 = np.array([xy_g[0], xy_g[1], 0., 0.], dtype=np.float32)

        z_ctx_s = build_context(parts, env, s0, num_hist, device)
        z_ctx_g = build_context(parts, env, g0, num_hist, device)
        z_s = z_ctx_s[0, -1, :, :].mean(dim=0)
        z_g = z_ctx_g[0, -1, :, :].mean(dim=0)

        alphas    = np.linspace(0., 1., n_steps)
        z_interps = [(1 - a) * z_s + a * z_g for a in alphas]
        xy        = decode_means(z_interps, probe, bounds)

        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=2.0, alpha=0.9,
                zorder=3, solid_capstyle='round')
        ax.scatter(*xy_s, s=58, c=color, zorder=5, marker='o',
                   edgecolors='white', linewidths=0.7)
        ax.scatter(*xy_g, s=58, c=color, zorder=5, marker='s',
                   edgecolors='white', linewidths=0.7)

    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='PointMaze probe landscapes for DINO-WM checkpoint.')
    p.add_argument('--dino-wm-dir',      required=True)
    p.add_argument('--ckpt',             required=True)
    p.add_argument('--output',           required=True)
    p.add_argument('--num-hist',         type=int, default=3)
    p.add_argument('--n-probe-episodes', type=int, default=120,
                   help='More episodes = better maze coverage for the probe.')
    p.add_argument('--n-steps',          type=int, default=200)
    p.add_argument('--seed',             type=int, default=0)
    p.add_argument('--device',           default='cuda')
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    parts  = load_model(args.ckpt, args.dino_wm_dir, device)

    env_cfg = {'environment': {'maze_map': 'U', 'image_size': IMG_SIZE,
                               'action_scale': 1.0}}
    env = PointMazeVisual(env_cfg, seed=args.seed)

    print(f'[probe] collecting {args.n_probe_episodes} episodes × {args.n_steps} steps …')
    zs, xys = collect_data(env, parts, args.num_hist, device,
                           n_episodes=args.n_probe_episodes,
                           n_steps=args.n_steps)
    probe  = fit_probe(zs, xys)
    bounds = data_bounds(xys, pad=0.12)
    print(f'[probe] maze bounds  x={bounds[0]}  y={bounds[1]}')

    BG = '#10101a'
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), facecolor=BG,
                             gridspec_kw={'wspace': 0.03})
    for ax in axes:
        ax.set_facecolor(BG)
    fig.subplots_adjust(left=0.004, right=0.996, top=0.996, bottom=0.004)

    print('[probe] panel A — imagination fan …')
    panel_a(axes[0], env, parts, probe, xys, bounds, rng, args.num_hist, device)

    print('[probe] panel B — UMAP topology …')
    panel_b(axes[1], zs, xys)

    print('[probe] panel C — latent interpolation …')
    panel_c(axes[2], env, parts, probe, xys, bounds, args.num_hist, device)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'[done] → {out}')
    env.close()


if __name__ == '__main__':
    main()

