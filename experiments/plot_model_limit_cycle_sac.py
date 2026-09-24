#!/usr/bin/env python
"""Compare GT and model-decoded limit cycles on the right-hip phase portrait.

Figure: 4 panels (GT | 1SP+EP-IDM | 1SP+SIG | MSP+SIG).
Each model panel shows the model's decoded orbit in colour with the GT orbit
overlaid in light grey as a reference.

Pipeline per model
------------------
1. Fit MLP probe (z → 17-D obs) on the training dataset.
2. Seed z_0 = encode(frame_warmup, frame_warmup-1, state_warmup) from a
   real-env warm-up rollout (uses Walker2dVisual for frames).
3. Latent closed-loop rollout for n_steps:
     x̂_t  = mlp_probe(z_t)          # decoded 17-D state
     a_t   = SAC.act(x̂_t)            # policy on decoded state
     z_t+1 = latent_step(z_t, a_t)   # world-model prediction
4. Plot (x̂_t[2], x̂_t[11]) = (right-hip θ, right-hip θ̇).

Usage
-----
MUJOCO_GL=egl python experiments/plot_model_limit_cycle_sac.py \\
    --fwd-ar-ckpt  /mnt/t7shield/jepa_results/walker2d_mixed_sac_smwm_fwd_endpoint_inverse_act1_seed42/model_final.pt \\
    --fwd-ar-cfg   configs/walker2d_smwm_fwd_endpoint_inverse_act1.yaml \\
    --sig-fwd-ckpt /mnt/t7shield/jepa_results/walker2d_mixed_sac_smwm_sigreg_fwd_act1_seed42/model_final.pt \\
    --sig-fwd-cfg  configs/walker2d_smwm_sigreg_fwd_act1.yaml \\
    --ms-sr-ckpt   /mnt/t7shield/jepa_results/walker2d_mixed_sac_smwm_sigreg_rollout_act1_seed42/model_final.pt \\
    --ms-sr-cfg    configs/walker2d_smwm_sigreg_rollout_act1.yaml \\
    --hdf5-dir     data/walker2d_mixed_sac_fs5_64 \\
    --sac-repo     sdpkjc/Walker2d-v4-sac_continuous_action-seed4 \\
    --n-episodes   8 --n-steps 600 --warmup 150 \\
    --fwd-probe-path  results/probes/fwd_ep_ar_mlp_probe.pt \\
    --sig-fwd-probe-path results/probes/sig_fwd_mlp_probe.pt \\
    --ms-probe-path   results/probes/ms_sr_mlp_probe.pt \\
    --gt-cache    results/cache/gt_trajs.npz \\
    --fwd-cache   results/cache/fwd_trajs.npz \\
    --sig-fwd-cache results/cache/sig_fwd_trajs.npz \\
    --ms-cache    results/cache/ms_trajs.npz \\
    --fwd-show 2 --sig-fwd-show 2 --ms-show 2 --ms-steps 100 \\
    --out         results/model_limit_cycle_sac.pdf
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from experiments.walker2d_smwm_utils import (
    load_walker_bundle, fit_walker_mlp_probe,
    latent_step, is_healthy_obs,
)
from experiments.sensorimotor_probe_utils import encode_obs

# ── style ──────────────────────────────────────────────────────────────────────
PANEL_BG    = 'white'
GRID_KW     = dict(color='#cccccc', linewidth=0.8, alpha=0.9)
TICK_SIZE   = 13
TITLE_SIZE  = 16
LINEWIDTH   = 1.4

GT_CMAP      = 'Greens'    # GT → green
FWD_CMAP     = 'Oranges'   # 1SP+EP-IDM → orange
SIG_FWD_CMAP = 'GnBu'      # 1SP+SIG → green-blue
MS_CMAP      = 'Blues'     # MSP+SIG → blue
GT_REF_COLOR = '#aaaaaa'   # light grey GT reference in model panels

IDX_RIGHT_HIP_ANG = 2
IDX_RIGHT_HIP_VEL = 11


# ── GT rollout (no rendering) ──────────────────────────────────────────────────

def collect_gt_trajs(policy, n_episodes, n_steps, warmup, seed):
    import gymnasium as gym
    trajs = []
    for ep in range(n_episodes):
        env = gym.make('Walker2d-v4')
        obs, _ = env.reset(seed=seed + ep)
        buf = []
        for t in range(n_steps + warmup):
            a = policy.act(obs.astype(np.float32))
            obs, _, term, trunc, _ = env.step(a)
            if t >= warmup:
                buf.append(obs.astype(np.float32))
            if term or trunc:
                break
        env.close()
        if len(buf) >= 10:
            trajs.append(np.stack(buf))
    print(f'[GT] {len(trajs)}/{n_episodes} episodes OK')
    return trajs


# ── model latent rollout ───────────────────────────────────────────────────────

@torch.no_grad()
def collect_model_trajs(bundle, probe, policy,
                        n_episodes, n_steps, warmup, seed,
                        image_size=64):
    """Seed z_0 from real env, then close-loop in latent space."""
    from envs.walker2d_visual import Walker2dVisual

    trajs = []
    for ep in range(n_episodes):
        env = Walker2dVisual(image_size=image_size, seed=seed + ep)
        obs, state, _ = env.reset()
        prev_obs = obs.copy()

        # ── warm-up in real env to reach steady gait ──
        for t in range(warmup):
            a = policy.act(state.astype(np.float32))
            prev_obs = obs.copy()
            obs, state, _, done, _ = env.step(a)
            if done:
                break
        env.close()

        if done:
            print(f'  episode {ep}: fell during warm-up — skipped')
            continue

        # ── encode z_0 from the last warm-up frame ────
        z = encode_obs(bundle, obs, prev_obs, state)   # (1, D)

        # ── closed-loop latent rollout ─────────────────
        buf = []
        for _ in range(n_steps):
            x_hat = probe(z[0].cpu().numpy())           # decoded 17-D state
            a     = policy.act(x_hat)
            z     = latent_step(bundle, z, a)
            buf.append(x_hat.copy())

        trajs.append(np.stack(buf))
        print(f'  episode {ep}: {len(buf)} latent steps decoded')

    print(f'[model] {len(trajs)}/{n_episodes} episodes OK')
    return trajs


# ── drawing helpers ────────────────────────────────────────────────────────────

def _phase_xy(trajs):
    return [np.column_stack([t[:, IDX_RIGHT_HIP_ANG],
                              t[:, IDX_RIGHT_HIP_VEL]]) for t in trajs]


def _add_orbit(ax, xy, cmap, lw=LINEWIDTH, alpha=0.85, linestyle='solid'):
    if len(xy) < 2:
        return
    pts  = np.column_stack([xy[:, 0], xy[:, 1]])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    t_n  = np.linspace(0, 1, len(segs))
    lc   = LineCollection(segs, cmap=cmap, norm=Normalize(0, 1),
                          linewidths=lw, alpha=alpha, linestyle=linestyle)
    lc.set_array(t_n)
    ax.add_collection(lc)


def _gt_bbox(gt_xys, margin=0.15):
    """Return (xmin, xmax, ymin, ymax) of GT phase portrait with margin."""
    xs = np.concatenate([xy[:, 0] for xy in gt_xys])
    ys = np.concatenate([xy[:, 1] for xy in gt_xys])
    rx, ry = xs.ptp(), ys.ptp()
    return (xs.min() - margin * rx, xs.max() + margin * rx,
            ys.min() - margin * ry, ys.max() + margin * ry)


def _escapes_bbox(xy, bbox):
    """True if any point of xy lies outside the GT bounding box."""
    xmin, xmax, ymin, ymax = bbox
    return bool(np.any(xy[:, 0] < xmin) or np.any(xy[:, 0] > xmax) or
                np.any(xy[:, 1] < ymin) or np.any(xy[:, 1] > ymax))


def _autolim(ax, *xys, pad=0.08):
    xs = np.concatenate([xy[:, 0] for xy in xys if len(xy)])
    ys = np.concatenate([xy[:, 1] for xy in xys if len(xy)])
    rx, ry = xs.ptp(), ys.ptp()
    ax.set_xlim(xs.min() - pad * rx, xs.max() + pad * rx)
    ax.set_ylim(ys.min() - pad * ry, ys.max() + pad * ry)


def _style(ax, xlabel, ylabel, title=None):
    ax.set_facecolor(PANEL_BG)
    ax.set_axisbelow(True)
    ax.grid(True, **GRID_KW)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=TITLE_SIZE)
    ax.set_ylabel(ylabel, fontsize=TITLE_SIZE)
    if title is not None:
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight='bold', pad=6)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--fwd-ar-ckpt',  required=True)
    p.add_argument('--fwd-ar-cfg',   default=None)
    p.add_argument('--sig-fwd-ckpt', default=None,
                   help='1SP+SIG checkpoint (sigreg_fwd); omit to skip that panel')
    p.add_argument('--sig-fwd-cfg',  default=None)
    p.add_argument('--ms-sr-ckpt',   required=True)
    p.add_argument('--ms-sr-cfg',    default=None)
    p.add_argument('--hdf5-dir',     required=True)
    p.add_argument('--sac-repo',
                   default='sdpkjc/Walker2d-v4-sac_continuous_action-seed4')
    p.add_argument('--sac-ckpt',     default=None)
    p.add_argument('--device',       default='cuda')
    p.add_argument('--n-episodes',   type=int, default=8)
    p.add_argument('--n-steps',      type=int, default=600)
    p.add_argument('--warmup',       type=int, default=150)
    p.add_argument('--probe-episodes', type=int, default=200,
                   help='Max HDF5 episodes used to fit MLP probes')
    p.add_argument('--probe-epochs',   type=int, default=30)
    p.add_argument('--fwd-probe-path',     default=None,
                   help='Path to save/load 1SP+EP-IDM MLP probe (.pt)')
    p.add_argument('--sig-fwd-probe-path', default=None,
                   help='Path to save/load 1SP+SIG MLP probe (.pt)')
    p.add_argument('--ms-probe-path',      default=None,
                   help='Path to save/load MSP+SIG MLP probe (.pt)')
    p.add_argument('--gt-cache',      default=None,
                   help='Path to save/load GT trajectories (.npz)')
    p.add_argument('--fwd-cache',     default=None,
                   help='Path to save/load 1SP+EP-IDM trajectories (.npz)')
    p.add_argument('--sig-fwd-cache', default=None,
                   help='Path to save/load 1SP+SIG trajectories (.npz)')
    p.add_argument('--ms-cache',      default=None,
                   help='Path to save/load MSP+SIG trajectories (.npz)')
    p.add_argument('--seed',          type=int, default=0)
    p.add_argument('--image-size',    type=int, default=64)
    p.add_argument('--fwd-show',      type=int, default=3,
                   help='Max converging 1SP+EP-IDM episodes to plot')
    p.add_argument('--sig-fwd-show',  type=int, default=3,
                   help='Max converging 1SP+SIG episodes to plot')
    p.add_argument('--sig-fwd-steps', type=int, default=None,
                   help='Truncate each 1SP+SIG trajectory to first N steps')
    p.add_argument('--ms-show',       type=int, default=1,
                   help='Number of MSP+SIG episodes to plot (first N)')
    p.add_argument('--ms-steps',      type=int, default=None,
                   help='Truncate each MSP+SIG trajectory to first N steps')
    p.add_argument('--out',           default='results/model_limit_cycle_sac.pdf')
    args = p.parse_args()

    # ── trajectory cache helpers ──────────────────────────────────────────────
    def _save_trajs(path, trajs):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{f't{i}': t for i, t in enumerate(trajs)})
        print(f'[cache] saved {len(trajs)} episodes → {path}')

    def _load_trajs(path):
        d = np.load(path)
        trajs = [d[f't{i}'] for i in range(len(d))]
        print(f'[cache] loaded {len(trajs)} episodes from {path}')
        return trajs

    # ── SAC policy ────────────────────────────────────────────────────────────
    from experiments.walker2d_ppo_utils import download_and_load_sac, load_sac_from_local
    policy = (load_sac_from_local(args.sac_ckpt, device='cpu')
              if args.sac_ckpt else
              download_and_load_sac(args.sac_repo, device='cpu'))

    # ── GT trajectories ───────────────────────────────────────────────────────
    if args.gt_cache and Path(args.gt_cache).exists():
        gt_trajs = _load_trajs(args.gt_cache)
    else:
        print('\n=== GT rollout ===')
        gt_trajs = collect_gt_trajs(policy, args.n_episodes,
                                    args.n_steps, args.warmup, args.seed)
        if args.gt_cache:
            _save_trajs(args.gt_cache, gt_trajs)
    gt_xys = _phase_xy(gt_trajs)

    # ── load model bundles ────────────────────────────────────────────────────
    def _cfg(ckpt, override):
        if override:
            return override
        for name in ['env_config.yaml', 'config.yaml']:
            c = Path(ckpt).parent / name
            if c.exists():
                return str(c)
        raise FileNotFoundError(f'No config found near {ckpt}')

    print('\n=== Load 1SP+EP-IDM bundle ===')
    fwd_bundle = load_walker_bundle(args.fwd_ar_ckpt,
                                    _cfg(args.fwd_ar_ckpt, args.fwd_ar_cfg),
                                    args.device)
    sig_fwd_bundle = None
    if args.sig_fwd_ckpt:
        print('\n=== Load 1SP+SIG bundle ===')
        sig_fwd_bundle = load_walker_bundle(args.sig_fwd_ckpt,
                                            _cfg(args.sig_fwd_ckpt, args.sig_fwd_cfg),
                                            args.device)
    print('\n=== Load MSP+SIG bundle ===')
    ms_bundle  = load_walker_bundle(args.ms_sr_ckpt,
                                    _cfg(args.ms_sr_ckpt, args.ms_sr_cfg),
                                    args.device)

    # ── fit or load MLP probes ────────────────────────────────────────────────
    def _get_probe(bundle, label, save_path):
        from experiments.walker2d_smwm_utils import MLPStateProbe
        z_dim = int(bundle['model_cfg'].get('latent_dim', 192))
        if save_path and Path(save_path).exists():
            print(f'\n=== Load MLP probe for {label} from {save_path} ===')
            probe = MLPStateProbe(z_dim).to(bundle['device'])
            probe._net.load_state_dict(
                torch.load(save_path, map_location=bundle['device'],
                           weights_only=True))
            probe._net.eval()
            return probe
        print(f'\n=== Fit MLP probe for {label} ===')
        probe = fit_walker_mlp_probe(bundle, args.hdf5_dir,
                                     max_episodes=args.probe_episodes,
                                     n_epochs=args.probe_epochs)
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(probe._net.state_dict(), save_path)
            print(f'[probe] saved → {save_path}')
        return probe

    fwd_probe     = _get_probe(fwd_bundle, '1SP+EP-IDM', args.fwd_probe_path)
    sig_fwd_probe = (_get_probe(sig_fwd_bundle, '1SP+SIG', args.sig_fwd_probe_path)
                     if sig_fwd_bundle is not None else None)
    ms_probe      = _get_probe(ms_bundle,  'MSP+SIG',    args.ms_probe_path)

    # ── model latent rollouts ─────────────────────────────────────────────────
    if args.fwd_cache and Path(args.fwd_cache).exists():
        fwd_trajs = _load_trajs(args.fwd_cache)
    else:
        print('\n=== Latent rollout: 1SP+EP-IDM ===')
        fwd_trajs = collect_model_trajs(fwd_bundle, fwd_probe, policy,
                                        args.n_episodes, args.n_steps,
                                        args.warmup, args.seed, args.image_size)
        if args.fwd_cache:
            _save_trajs(args.fwd_cache, fwd_trajs)

    sig_fwd_trajs = []
    if sig_fwd_bundle is not None:
        if args.sig_fwd_cache and Path(args.sig_fwd_cache).exists():
            sig_fwd_trajs = _load_trajs(args.sig_fwd_cache)
        else:
            print('\n=== Latent rollout: 1SP+SIG ===')
            sig_fwd_trajs = collect_model_trajs(sig_fwd_bundle, sig_fwd_probe, policy,
                                                args.n_episodes, args.n_steps,
                                                args.warmup, args.seed, args.image_size)
            if args.sig_fwd_cache:
                _save_trajs(args.sig_fwd_cache, sig_fwd_trajs)

    if args.ms_cache and Path(args.ms_cache).exists():
        ms_trajs = _load_trajs(args.ms_cache)
    else:
        print('\n=== Latent rollout: MSP+SIG ===')
        ms_trajs = collect_model_trajs(ms_bundle, ms_probe, policy,
                                       args.n_episodes, args.n_steps,
                                       args.warmup, args.seed, args.image_size)
        if args.ms_cache:
            _save_trajs(args.ms_cache, ms_trajs)

    fwd_xys     = _phase_xy(fwd_trajs)
    sig_fwd_xys = _phase_xy(sig_fwd_trajs)
    ms_xys      = _phase_xy(ms_trajs)

    # ── classify converging vs escaping trajectories ─────────────────────────
    gt_bbox = _gt_bbox(gt_xys)

    fwd_escape = [xy for xy in fwd_xys if     _escapes_bbox(xy, gt_bbox)]
    fwd_stay   = [xy for xy in fwd_xys if not _escapes_bbox(xy, gt_bbox)]
    print(f'[1SP+EP-IDM] {len(fwd_escape)} escaping / {len(fwd_stay)} converging '
          f'(GT bbox margin=15%)')
    fwd_stay = fwd_stay[:args.fwd_show]

    if sig_fwd_xys:
        sig_fwd_escape = [xy for xy in sig_fwd_xys if     _escapes_bbox(xy, gt_bbox)]
        sig_fwd_stay   = [xy for xy in sig_fwd_xys if not _escapes_bbox(xy, gt_bbox)]
        print(f'[1SP+SIG]    {len(sig_fwd_escape)} escaping / {len(sig_fwd_stay)} converging '
              f'(GT bbox margin=15%)')

    # ── figure ────────────────────────────────────────────────────────────────
    XLABEL = r'Right hip $\theta$ (rad)'
    YLABEL = r'Right hip $\dot{\theta}$ (rad/s)'

    n_panels = 3 + (1 if sig_fwd_bundle is not None else 0)
    fig_width = 4.8 * n_panels
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 4.5))

    ax_idx = 0

    # Panel 0: GT — time-coloured line orbits
    for xy in gt_xys:
        _add_orbit(axes[ax_idx], xy, GT_CMAP)
    _autolim(axes[ax_idx], *gt_xys)
    _style(axes[ax_idx], XLABEL, YLABEL, 'GT')
    ax_idx += 1

    # Panel 1: 1SP+EP-IDM — converging trajectories only
    for xy in fwd_stay:
        _add_orbit(axes[ax_idx], xy, FWD_CMAP)
    _autolim(axes[ax_idx], *(gt_xys + fwd_stay)) if fwd_stay else _autolim(axes[ax_idx], *gt_xys)
    _style(axes[ax_idx], XLABEL, YLABEL, '1SP+EP-IDM')
    ax_idx += 1

    # Panel 2 (optional): 1SP+SIG — first sig_fwd_show episodes, truncated to sig_fwd_steps
    if sig_fwd_bundle is not None:
        sig_few = sig_fwd_xys[:args.sig_fwd_show]
        if args.sig_fwd_steps is not None:
            sig_few = [xy[:args.sig_fwd_steps] for xy in sig_few]
        for xy in sig_few:
            _add_orbit(axes[ax_idx], xy, SIG_FWD_CMAP, lw=LINEWIDTH, alpha=0.85)
        _autolim(axes[ax_idx], *(sig_few + gt_xys)) if sig_few else _autolim(axes[ax_idx], *gt_xys)
        _style(axes[ax_idx], XLABEL, YLABEL, '1SP+SIG')
        ax_idx += 1

    # Final panel: MSP+SIG — first ms_show episodes, truncated to ms_steps
    ms_few = ms_xys[:args.ms_show]
    if args.ms_steps is not None:
        ms_few = [xy[:args.ms_steps] for xy in ms_few]
    for xy in ms_few:
        _add_orbit(axes[ax_idx], xy, MS_CMAP, lw=LINEWIDTH, alpha=0.85)
    _autolim(axes[ax_idx], *(ms_few + gt_xys)) if ms_few else _autolim(axes[ax_idx], *gt_xys)
    _style(axes[ax_idx], XLABEL, YLABEL, 'MSP+SIG')

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    print(f'\n[saved] {out}')


if __name__ == '__main__':
    main()
