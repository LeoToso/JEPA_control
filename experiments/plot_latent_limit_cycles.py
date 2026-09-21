#!/usr/bin/env python
"""Latent-space limit cycle comparison: GT (encoder) vs teacher-forced prediction.

For each SMWM model we compute two trajectories **in the model's own latent space**:

  GT-encoded  : z_t = encode(frame_t, frame_{t-1}, state_t)   [blue]
  TF-predicted: z_0 = encode(frame_0, ...) then z_{t+1} = predict(z_t, action_t)  [model colour]

If the predictor has captured the gait dynamics, the predicted trajectory should
trace the same closed loop as the encoded one.  No probe / decoding needed.

Figure layout: 1 row × 3 columns (phase portraits only)
  Col 0: GT MuJoCo — right-hip angle vs right-hip angular velocity (position-velocity phase portrait)
  Col 1: FWD+EP-AR — TF-predicted latent trajectory in GT-encoded PCA space (PC1 vs PC2)
  Col 2: MS+SR     — same

Usage
-----
MUJOCO_GL=egl python experiments/plot_latent_limit_cycles.py \\
    --fwd-ar-ckpt results/walker2d_smwm_fwd_endpoint_inverse_act1_seed42/model_final.pt \\
    --ms-sr-ckpt  results/walker2d_smwm_sigreg_rollout_act1_seed42/model_final.pt \\
    --hdf5-dir    data/walker2d_fs5_64 \\
    --output-dir  results/poincare_walker2d
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('MUJOCO_GL', 'egl')

import h5py
import numpy as np
import torch
import torch.nn.functional as F_
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from experiments.walker2d_smwm_utils import load_walker_bundle, latent_step
from experiments.sensorimotor_probe_utils import encode_obs

# ── style constants ───────────────────────────────────────────────────────────

GT_COLOR  = '#1f77b4'   # blue  — GT
FWD_COLOR = '#2ca02c'   # green — FWD+EP-AR
MS_COLOR  = '#ff7f0e'   # orange — MS+SR

plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Computer Modern Roman', 'DejaVu Serif'],
    'mathtext.fontset':  'cm',
    'axes.labelsize':    10,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
})


# ── GT gait trajectory ────────────────────────────────────────────────────────

def collect_gt_gait_trajectory(
    hdf5_path:  str,
    n_episodes: int = 10,
    image_size: int = 64,
) -> list[np.ndarray]:
    """Load gait states from HDF5, one array per episode.

    Returns list of (T_ep, 16) float32 arrays.
    """
    eps: list[np.ndarray] = []
    with h5py.File(hdf5_path, 'r') as f:
        ep_grp  = f['episodes']
        ep_keys = sorted(ep_grp.keys(), key=lambda k: int(k))[:n_episodes]
        for ek in ep_keys:
            st = ep_grp[ek]['states'][:]   # (T+1, 17)
            eps.append(st[:, :16].astype(np.float32))
    return eps


# ── latent trajectories via teacher forcing ───────────────────────────────────

@torch.no_grad()
def collect_latent_trajectories(
    bundle:     dict,
    hdf5_path:  str,
    n_episodes: int = 10,
    image_size: int = 64,
    skip_steps: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """For each episode compute GT-encoded and teacher-forced latent trajectories.

    GT-encoded:  z_t  = encode(frame_t, frame_{t-1}, state_t)  at every step
    TF-predicted:z_{t+1} = latent_step(z_t, action_t) starting from z_0 = encode(frame_0)

    Returns
    -------
    gt_zs   : list[np.ndarray]  len=n_episodes, each (T-skip_steps, latent_dim)
    pred_zs : list[np.ndarray]  len=n_episodes, each (T-skip_steps, latent_dim)
    """
    gt_zs:   list[np.ndarray] = []
    pred_zs: list[np.ndarray] = []

    with h5py.File(hdf5_path, 'r') as f:
        ep_grp  = f['episodes']
        ep_keys = sorted(ep_grp.keys(), key=lambda k: int(k))[:n_episodes]
        print(f'[latent-tf] encoding {len(ep_keys)} episodes …')

        for ek in ep_keys:
            ep    = ep_grp[ek]
            obs_t = ep['observations'][:]   # (T+1, H, W, 3) uint8
            act_t = ep['actions'][:]        # (T, 6)
            st_t  = ep['states'][:]         # (T+1, 17)
            T     = obs_t.shape[0] - 1

            if T < 5:
                continue

            # Resize to model's expected image size
            if obs_t.shape[1] != image_size or obs_t.shape[2] != image_size:
                t_obs = torch.from_numpy(obs_t).permute(0, 3, 1, 2).float()
                t_obs = F_.interpolate(t_obs, (image_size, image_size),
                                       mode='bilinear', align_corners=False)
                obs_t = t_obs.permute(0, 2, 3, 1).byte().numpy()

            # ── GT-encoded trajectory ─────────────────────────────────────────
            gt_ep: list[np.ndarray] = []
            for t in range(T + 1):
                prev = obs_t[max(t - 1, 0)]
                z    = encode_obs(bundle, obs_t[t], prev, st_t[t])
                gt_ep.append(z[0].cpu().numpy())
            gt_zs.append(np.stack(gt_ep[skip_steps:]))

            # ── Teacher-forced trajectory ─────────────────────────────────────
            pred_ep: list[np.ndarray] = []
            z = encode_obs(bundle, obs_t[0], obs_t[0], st_t[0])  # (1, D)
            pred_ep.append(z[0].cpu().numpy())
            for t in range(T):
                z = latent_step(bundle, z, act_t[t])
                pred_ep.append(z[0].cpu().numpy())
            pred_zs.append(np.stack(pred_ep[skip_steps:]))

    return gt_zs, pred_zs


# ── PCA helper ────────────────────────────────────────────────────────────────

def fit_pca(vecs: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Zero-mean PCA.  Returns (mean, V) where V columns are principal axes."""
    mu   = vecs.mean(0)
    _, _, Vt = np.linalg.svd(vecs - mu, full_matrices=False)
    return mu, Vt[:n_components].T   # (D, n_components)


def project(vecs: np.ndarray, mu: np.ndarray, V: np.ndarray) -> np.ndarray:
    return (vecs - mu) @ V


# ── plotting helpers ──────────────────────────────────────────────────────────

def _traj_lc(ax, traj: np.ndarray, xi: int, yi: int,
             color_or_cmap, label: str | None = None,
             lw: float = 0.9, alpha: float = 0.85) -> None:
    """Add a time-colored trajectory line to ax."""
    if len(traj) < 2:
        return
    pts  = np.column_stack([traj[:, xi], traj[:, yi]])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    t_n  = np.linspace(0, 1, len(segs))

    if isinstance(color_or_cmap, str) and color_or_cmap.startswith('#'):
        from matplotlib.colors import to_rgba
        rgba = to_rgba(color_or_cmap)
        colors = np.array([(*rgba[:3], 0.3 + 0.7 * v) for v in t_n])
        lc = LineCollection(segs, colors=colors, linewidths=lw)
    else:
        lc = LineCollection(segs, cmap=color_or_cmap,
                            norm=plt.Normalize(0, 1),
                            linewidths=lw, alpha=alpha)
        lc.set_array(t_n)

    ax.add_collection(lc)
    if label:
        ax.plot([], [], color=color_or_cmap if isinstance(color_or_cmap, str)
                else plt.get_cmap(color_or_cmap)(0.5),
                lw=lw, label=label)


def _autolim(ax, *trajs, xi: int, yi: int, pad: float = 0.08) -> None:
    all_x = np.concatenate([t[:, xi] for t in trajs if len(t)])
    all_y = np.concatenate([t[:, yi] for t in trajs if len(t)])
    rx, ry = all_x.max() - all_x.min(), all_y.max() - all_y.min()
    ax.set_xlim(all_x.min() - pad * rx, all_x.max() + pad * rx)
    ax.set_ylim(all_y.min() - pad * ry, all_y.max() + pad * ry)


def _style(ax: plt.Axes, xl: str, yl: str) -> None:
    ax.set_xlabel(xl, fontsize=9)
    ax.set_ylabel(yl, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.grid(True, lw=0.3, alpha=0.3)


# ── main figure ───────────────────────────────────────────────────────────────

def plot_latent_limit_cycles(
    gt_gait_eps:  list[np.ndarray],    # episodes × (T_ep, 16) gait states
    fwd_gt_zs:    list[np.ndarray],    # episodes × (T, D) — GT-encoded for FWD
    fwd_pred_zs:  list[np.ndarray],    # TF-predicted for FWD
    ms_gt_zs:     list[np.ndarray],    # GT-encoded for MS
    ms_pred_zs:   list[np.ndarray],    # TF-predicted for MS
    out_path:     Path,
    skip_steps:   int = 80,
    n_show:       int = 100,
) -> None:
    """Single row of 3 phase portraits, trajectory lines per episode.

    All panels: within-episode-centred PCA → PC1 vs PC2 trajectory.
    Centring removes between-episode offsets so PCA captures within-episode
    dynamics.  A periodic orbit traces a closed ellipse in PC1-PC2; a linear
    drift traces a line from one end to the other.

    GT col:    16D physical gait state (centred PCA) — directly encodes the
               mechanical gait cycle; should give a clean ellipse.
    FWD col:   FWD TF-predicted latent (centred PCA).  A gait attractor →
               orbit stays close to origin / traces loops.
    MS+SR col: MS+SR TF-predicted latent (centred PCA).  Divergence →
               elongated line in PC1 direction.
    """
    TORSO_Z     = 0     # obs[0] = qpos[1] = torso height (m)
    MIN_TORSO_Z = 0.9   # m — filter fallen episodes

    def _trim_gt(eps: list[np.ndarray]) -> list[np.ndarray]:
        return [ep[skip_steps: skip_steps + n_show]
                for ep in eps if len(ep) > skip_steps + 5]

    def _trim_lat(eps: list[np.ndarray]) -> list[np.ndarray]:
        return [ep[:n_show] for ep in eps if len(ep) > 5]

    # ── Fall filter ───────────────────────────────────────────────────────────
    gt_eps_raw = _trim_gt(gt_gait_eps)
    keep       = [ep[:, TORSO_Z].min() >= MIN_TORSO_Z for ep in gt_eps_raw]
    n_kept     = sum(keep)
    print(f'[filter] upright episodes: {n_kept}/{len(keep)}')

    gt_eps_flt   = [ep for ep, k in zip(gt_eps_raw,   keep) if k]
    fwd_pred_flt = [ep for ep, k in zip(fwd_pred_zs,  keep) if k]
    ms_pred_flt  = [ep for ep, k in zip(ms_pred_zs,   keep) if k]

    def _global_pc1(traj_list: list[np.ndarray]) -> list[np.ndarray]:
        """Global PCA (no centering per episode); return PC1 per trajectory."""
        all_z = np.concatenate(traj_list, axis=0)
        mu    = all_z.mean(axis=0)
        _, _, Vt = np.linalg.svd(all_z - mu, full_matrices=False)
        return [(ep - mu) @ Vt[0] for ep in traj_list]

    # ── Scalar time series ────────────────────────────────────────────────────
    DT = 0.01   # s per observation step

    gt_series  = [ep[:n_show, 0]   for ep in gt_eps_flt]  # torso height
    fwd_series = _global_pc1(_trim_lat(fwd_pred_flt))
    ms_series  = _global_pc1(_trim_lat(ms_pred_flt))

    t_ax = np.arange(n_show) * DT   # shared time axis (seconds)

    for lbl, ser in [('GT-torso (m)', gt_series),
                     ('FWD-PC1',      fwd_series),
                     ('MS+SR-PC1',    ms_series)]:
        vals = np.concatenate(ser)
        print(f'[{lbl}]  range [{vals.min():.2f}, {vals.max():.2f}]')

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), facecolor='white')

    panels = [
        (axes[0], gt_series,  GT_COLOR,  'GT MuJoCo',  'Torso height (m)',  False),
        (axes[1], fwd_series, FWD_COLOR, 'FWD+EP-AR',  'Latent PC 1',       True),
        (axes[2], ms_series,  MS_COLOR,  'MS+SR',       'Latent PC 1',       True),
    ]
    for ax, series, color, title, ylabel, show_zero in panels:
        for s in series:
            T = min(len(s), n_show)
            ax.plot(t_ax[:T], s[:T], lw=0.9, alpha=0.55, color=color)
        if show_zero:
            ax.axhline(0, color='#aaaaaa', lw=0.6, ls='--', zorder=0)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_title(title, fontsize=12, color=color, fontweight='bold')
        # auto-scale y to data range with a small pad
        all_vals = np.concatenate([s[:n_show] for s in series])
        vmin, vmax = all_vals.min(), all_vals.max()
        pad = (vmax - vmin) * 0.08
        ax.set_ylim(vmin - pad, vmax + pad)

    fig.suptitle(f'Teacher-forced latent dynamics vs GT gait  '
                 f'({n_kept} upright episodes, {n_show} steps = {n_show*DT:.1f} s)',
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'[saved] {out_path}')


# ── config auto-detection ─────────────────────────────────────────────────────

def find_cfg(ckpt_path: str) -> str:
    p = Path(ckpt_path).parent
    for name in ['env_config.yaml', 'config.yaml', 'cfg.yaml']:
        if (p / name).exists():
            return str(p / name)
    raise FileNotFoundError(
        f'No env config yaml found in {p}. Pass --fwd-ar-cfg / --ms-sr-cfg.')


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Latent-space limit cycle phase portraits (no probe needed).')
    p.add_argument('--fwd-ar-ckpt', required=True)
    p.add_argument('--fwd-ar-cfg',  default=None)
    p.add_argument('--ms-sr-ckpt',  required=True)
    p.add_argument('--ms-sr-cfg',   default=None)
    p.add_argument('--device',      default='cuda')
    p.add_argument('--hdf5-dir',    required=True)
    p.add_argument('--split',       default='train')
    p.add_argument('--n-episodes',  type=int, default=10)
    p.add_argument('--image-size',  type=int, default=64)
    p.add_argument('--skip-steps',  type=int, default=80,
                   help='Steps discarded at start of each episode (transient)')
    p.add_argument('--n-show',      type=int, default=100,
                   help='Steps shown per episode in the phase portrait')
    p.add_argument('--output-dir',  default='results/poincare_walker2d')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = str(Path(args.hdf5_dir) / f'{args.split}.hdf5')

    print('\n=== Load FWD+EP-AR bundle ===')
    fwd_cfg    = args.fwd_ar_cfg or find_cfg(args.fwd_ar_ckpt)
    fwd_bundle = load_walker_bundle(args.fwd_ar_ckpt, fwd_cfg, args.device)

    print('\n=== Load MS+SR bundle ===')
    ms_cfg    = args.ms_sr_cfg or find_cfg(args.ms_sr_ckpt)
    ms_bundle = load_walker_bundle(args.ms_sr_ckpt,  ms_cfg,  args.device)

    print('\n=== Collect GT gait trajectory ===')
    gt_gait_eps = collect_gt_gait_trajectory(hdf5_path, args.n_episodes, args.image_size)
    print(f'[GT] {len(gt_gait_eps)} episodes, '
          f'lengths={[len(e) for e in gt_gait_eps]}')

    print('\n=== Collect FWD+EP-AR latent trajectories ===')
    fwd_gt_zs, fwd_pred_zs = collect_latent_trajectories(
        fwd_bundle, hdf5_path, args.n_episodes, args.image_size, args.skip_steps)
    print(f'[FWD] episodes={len(fwd_gt_zs)}  '
          f'gt_len={sum(len(z) for z in fwd_gt_zs)}  '
          f'pred_len={sum(len(z) for z in fwd_pred_zs)}')

    print('\n=== Collect MS+SR latent trajectories ===')
    ms_gt_zs, ms_pred_zs = collect_latent_trajectories(
        ms_bundle, hdf5_path, args.n_episodes, args.image_size, args.skip_steps)
    print(f'[MS+SR] episodes={len(ms_gt_zs)}  '
          f'gt_len={sum(len(z) for z in ms_gt_zs)}  '
          f'pred_len={sum(len(z) for z in ms_pred_zs)}')

    print('\n=== Plotting ===')
    plot_latent_limit_cycles(
        gt_gait_eps=gt_gait_eps,
        fwd_gt_zs=fwd_gt_zs,   fwd_pred_zs=fwd_pred_zs,
        ms_gt_zs=ms_gt_zs,     ms_pred_zs=ms_pred_zs,
        out_path=out_dir / 'latent_limit_cycles.png',
        skip_steps=args.skip_steps,
        n_show=args.n_show,
    )
    print('\n=== Done ===')


if __name__ == '__main__':
    main()
