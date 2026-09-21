#!/usr/bin/env python
"""Single-checkpoint 5-panel summary figure.

Panels (left → right):
  1. Phase portrait — GT (green) vs learned (plasma) vector field on θ–θ̇ plane
  2. H-step prediction error landscape — ||f_H(E(s),0) − E(s_{t+H}^GT)||
  3. Planning cost landscape — log10||f_H(E(s),0) − z_goal||^2, H=horizon
  4. Latent norm divergence — encoded GT vs recursive predictor norm over time
  5. Cosine alignment — cos(Δz_GT, Δz_pred) vs rollout time

Usage:
  python experiments/plot_checkpoint_summary_smwm.py \\
      --ckpt  path/to/model.pt \\
      --cfg   path/to/config.yaml \\
      --title "1sp_AR" \\
      --out   results/summary_1sp_AR.png
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))           # experiments/ → sensorimotor_probe_utils
sys.path.insert(0, str(_here.parent))    # repo root    → models/, data/, …

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from sensorimotor_probe_utils import make_env
from sensorimotor_probe_utils import (
    RidgeStateProbe, encode_obs, encode_rendered_state,
    gt_rollout, learned_rollout, load_bundle,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _env_patch(bundle):
    """Fill in missing 'environment' key for result-only configs."""
    if 'environment' not in bundle['env_cfg']:
        bundle['env_cfg']['environment'] = {
            'frame_skip': int(bundle['model_cfg'].get('frame_skip', 5)),
            'image_size': int(bundle['model_cfg'].get('image_size', 128)),
            'action_range': [-10, 10],
            'mass_cart': 1.0, 'mass_pole': 0.1,
            'pole_length': 0.5, 'gravity': 9.8, 'dt': 0.02,
        }


def _macro_dt(bundle):
    dt = float(bundle['env_cfg']['environment'].get('dt', 0.02))
    fs = int(bundle['env_cfg']['environment'].get('frame_skip', 1))
    return dt * fs


def _build_probe(bundle, n_samples=200, seed=123):
    """Fit a ridge probe over randomly sampled states."""
    rng = np.random.RandomState(seed)
    states = rng.uniform(
        [-.2, -.5, -.15, -1.], [.2, .5, .15, 1.],
        size=(n_samples, 4)).astype(np.float32)
    env = make_env(bundle['env_cfg'], seed)
    zs = []
    for s in states:
        obs, _, _ = env.reset_to_state(s)
        zs.append(encode_obs(bundle, obs, obs, s).cpu().numpy()[0])
    env.close()
    probe = RidgeStateProbe().fit(np.asarray(zs), states, 1e-3)
    return probe


def _gt_unstable_eigvec(bundle, eps=1e-4):
    """Finite-difference GT linearisation → most unstable eigenvector."""
    n = 4
    A = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        d = np.zeros(n, np.float32); d[j] = eps
        xp, _ = gt_rollout(bundle, d, 1, 0.)
        xm, _ = gt_rollout(bundle, -d, 1, 0.)
        A[:, j] = (xp[1] - xm[1]) / (2. * eps)
    vals, vecs = np.linalg.eig(A)
    order = np.argsort(np.abs(vals))[::-1]
    vecs = vecs[:, order].real
    norms = np.linalg.norm(vecs, axis=0, keepdims=True)
    norms = np.where(norms < 1e-12, 1., norms)
    return (vecs / norms)[:, 0]


# ── Panel 1: phase portrait ───────────────────────────────────────────────────

def panel_phase_portrait(ax, bundle, probe,
                         theta_max, rate_max, n_theta, n_rate):
    theta_deg = np.linspace(-theta_max, theta_max, n_theta)
    rate_deg  = np.linspace(-rate_max,  rate_max,  n_rate)
    theta_mesh, rate_mesh = np.meshgrid(theta_deg, rate_deg)

    env = make_env(bundle['env_cfg'], 991)
    mdt = _macro_dt(bundle)
    gt_u = np.zeros_like(theta_mesh); gt_v = np.zeros_like(theta_mesh)
    lr_u = np.zeros_like(theta_mesh); lr_v = np.zeros_like(theta_mesh)

    for idx in np.ndindex(theta_mesh.shape):
        state = np.array([0., 0.,
                          np.deg2rad(theta_mesh[idx]),
                          np.deg2rad(rate_mesh[idx])], dtype=np.float32)
        obs, _, _ = env.reset_to_state(state)
        z0    = encode_obs(bundle, obs, obs, state)
        z_roll = learned_rollout(bundle, z0, 1, 0.)
        dec   = probe(z_roll)  # (2, 4) decoded states
        lr_u[idx] = np.rad2deg((dec[1, 2] - dec[0, 2]) / mdt)
        lr_v[idx] = np.rad2deg((dec[1, 3] - dec[0, 3]) / mdt)
        env.reset_to_state(state)
        _, ns, _, _, _ = env.step(0.)
        gt_u[idx] = np.rad2deg((ns[2] - state[2]) / mdt)
        gt_v[idx] = np.rad2deg((ns[3] - state[3]) / mdt)

    env.close()

    frac = 0.45
    dx = 2 * theta_max / max(n_theta - 1, 1)
    dy = 2 * rate_max  / max(n_rate  - 1, 1)

    def _n(u, v):
        n = np.sqrt(u**2 + v**2)
        s = np.maximum(n, 1e-12)
        return u / s, v / s, n

    gt_un, gt_vn, _ = _n(gt_u, gt_v)
    lr_un, lr_vn, speed = _n(lr_u, lr_v)

    ax.quiver(theta_mesh, rate_mesh,
              gt_un * frac * dx, gt_vn * frac * dy,
              color='#2ecc40', alpha=.7, pivot='mid',
              scale=1, scale_units='xy', angles='xy',
              width=.002, headwidth=4, headlength=4)
    ax.quiver(theta_mesh, rate_mesh,
              lr_un * frac * dx, lr_vn * frac * dy,
              color='black', alpha=0.75, pivot='mid',
              scale=1, scale_units='xy', angles='xy',
              width=.0025, headwidth=4, headlength=4)
    ax.scatter([0.], [0.], marker='*', s=200, color='red', zorder=5)
    ax.set_xlabel(r'$\theta$ [deg]', fontsize=14)
    ax.set_ylabel(r'$\dot{\theta}$ [deg/s]', fontsize=14)
    ax.legend(handles=[
        Line2D([0], [0], color='#2ecc40', lw=4, label='GT'),
        Line2D([0], [0], color='black',   lw=3, label='Learned'),
        Line2D([0], [0], marker='*', color='red', ls='None',
               ms=13, label='Equilibrium'),
    ], fontsize=14, loc='upper right')
    ax.tick_params(labelsize=14)
    ax.grid(alpha=.2)


# ── Panel 2: H-step prediction error ─────────────────────────────────────────

def panel_pred_error(ax, fig, bundle,
                     theta_max, rate_max, n_theta, n_rate, H):
    theta_deg = np.linspace(-theta_max, theta_max, n_theta)
    rate_deg  = np.linspace(-rate_max,  rate_max,  n_rate)
    err = np.zeros((n_rate, n_theta))
    env = make_env(bundle['env_cfg'], 993)

    for ri, r_deg in enumerate(rate_deg):
        for ti, t_deg in enumerate(theta_deg):
            state = np.array([0., 0., np.deg2rad(t_deg), np.deg2rad(r_deg)],
                             dtype=np.float32)
            obs, _, _ = env.reset_to_state(state)
            z0 = encode_obs(bundle, obs, obs, state)
            z_pred_H = learned_rollout(bundle, z0, H, 0.)[-1]
            obs_t, state_t = obs, state
            obs_prev = obs
            for _ in range(H):
                obs_next, state_next, _, done, _ = env.step(0.)
                obs_prev, obs_t = obs_t, obs_next
                state_t = state_next
                if done:
                    break
            z_gt_H = encode_obs(
                bundle, obs_t, obs_prev, state_t).cpu().numpy()[0]
            err[ri, ti] = float(np.linalg.norm(z_pred_H - z_gt_H))

    env.close()
    im = ax.pcolormesh(theta_deg, rate_deg, err,
                       cmap='YlOrRd', shading='auto', vmin=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                        label='prediction error')
    cbar.set_label('prediction error', fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    ax.contour(theta_deg, rate_deg, err, levels=6,
               colors='k', linewidths=0.5, alpha=0.35)
    ax.scatter([0.], [0.], marker='*', s=200, color='lime', zorder=5,
               edgecolors='darkgreen', lw=0.8, label='Equilibrium')
    ax.set_xlabel(r'$\theta$ [deg]', fontsize=14)
    ax.set_ylabel(r'$\dot{\theta}$ [deg/s]', fontsize=14)
    ax.legend(fontsize=14)
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.15)


# ── Panel 3: planning cost ────────────────────────────────────────────────────

def panel_planning_cost(ax, fig, bundle,
                        theta_max, rate_max, n_theta, n_rate, H):
    theta_deg = np.linspace(-theta_max, theta_max, n_theta)
    rate_deg  = np.linspace(-rate_max,  rate_max,  n_rate)

    goal_env = make_env(bundle['env_cfg'], 999)
    goal_obs, _, _ = goal_env.reset_to_state(np.zeros(4, np.float32))
    z_goal = encode_obs(
        bundle, goal_obs, goal_obs, np.zeros(4, np.float32)).cpu().numpy()[0]
    goal_env.close()

    cost = np.zeros((n_rate, n_theta))
    env = make_env(bundle['env_cfg'], 992)
    for ri, r_deg in enumerate(rate_deg):
        for ti, t_deg in enumerate(theta_deg):
            state = np.array([0., 0., np.deg2rad(t_deg), np.deg2rad(r_deg)],
                             dtype=np.float32)
            obs, _, _ = env.reset_to_state(state)
            z0 = encode_obs(bundle, obs, obs, state)
            orbit = learned_rollout(bundle, z0, H, 0.)
            diff = orbit[-1] - z_goal
            cost[ri, ti] = float(np.dot(diff, diff))
    env.close()

    log_cost = np.log10(np.maximum(cost, 1e-12))
    im = ax.pcolormesh(theta_deg, rate_deg, log_cost,
                       cmap='RdYlBu_r', shading='auto')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$\log_{10}\|\hat{z}_H - z_{\rm goal}\|^2$', fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    ax.contour(theta_deg, rate_deg, log_cost, levels=8,
               colors='k', linewidths=0.5, alpha=0.35)
    ax.scatter([0.], [0.], marker='*', s=200, color='lime', zorder=5,
               edgecolors='darkgreen', lw=0.8, label='Equilibrium')
    ax.set_xlabel(r'$\theta$ [deg]', fontsize=14)
    ax.set_ylabel(r'$\dot{\theta}$ [deg/s]', fontsize=14)
    ax.legend(fontsize=14)
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.15)


# ── Panel 4: latent norm divergence ──────────────────────────────────────────

def panel_latent_norm(ax, bundle, theta0_rad, steps, action=0.):
    state0 = np.array([0., 0., theta0_rad, 0.], dtype=np.float32)
    states, z_gt  = gt_rollout(bundle, state0, steps, action)
    z_pred = learned_rollout(
        bundle, encode_rendered_state(bundle, state0),
        len(states) - 1, action)

    gt_norm   = np.linalg.norm(z_gt,   axis=1)
    pred_norm = np.linalg.norm(z_pred, axis=1)
    time = np.arange(len(states)) * _macro_dt(bundle)

    ax.plot(time, gt_norm,   color='steelblue',   lw=2,   label='encoded GT')
    ax.plot(time, pred_norm, '--', color='darkorange', lw=2, label='predicted')
    ax.set_xlabel('time [s]', fontsize=14)
    ax.set_ylabel('latent norm', fontsize=14)
    ax.legend(fontsize=14)
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.25)


# ── Panel 5: cosine alignment Δz_GT vs Δz_pred ───────────────────────────────

def panel_cosine_alignment(ax, bundle, v_u, macro_dt,
                           H_max=20, n_starts=6, alpha_max=0.25):
    eps_vals = np.linspace(0.02, alpha_max, n_starts)
    color = '#2166ac'
    cos_curves = []

    for eps in eps_vals:
        x0 = (eps * v_u).astype(np.float32)
        _, gt_lat  = gt_rollout(bundle, x0, H_max, 0.)
        z0 = encode_rendered_state(bundle, x0)
        pr_lat = learned_rollout(bundle, z0, H_max, 0.)
        n = min(len(gt_lat), len(pr_lat))
        dz_gt   = gt_lat[:n]  - gt_lat[0][None]
        dz_pred = pr_lat[:n] - pr_lat[0][None]
        dot  = (dz_gt * dz_pred).sum(axis=1)
        norm = (np.linalg.norm(dz_gt,   axis=1) *
                np.linalg.norm(dz_pred, axis=1))
        cos = np.where(norm > 1e-6, dot / np.where(norm > 1e-6, norm, 1.), np.nan)
        cos_curves.append(cos)

    max_len = max(len(c) for c in cos_curves)
    Hs = np.arange(max_len) * macro_dt
    mat = np.array([np.pad(c.astype(float), (0, max_len - len(c)),
                           constant_values=np.nan)
                    for c in cos_curves])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mean = np.nanmean(mat, axis=0)
        std  = np.nanstd(mat,  axis=0)

    ax.plot(Hs, mean, color=color, lw=2.2)
    ax.fill_between(Hs,
                    np.clip(mean - std, -1, 1),
                    np.clip(mean + std, -1, 1),
                    color=color, alpha=0.15)
    ax.axhline(0., color='gray', ls='--', lw=0.8)
    ax.set_ylim(-0.25, 1.15)
    ax.set_xlabel('Rollout time [s]', fontsize=14)
    ax.set_ylabel(r'$\cos(\Delta z_{GT},\, \Delta z_{pred})$', fontsize=14)
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.25)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--ckpt',  required=True, help='Checkpoint path (.pt)')
    p.add_argument('--cfg',   required=True, help='Config path (.yaml)')
    p.add_argument('--title', required=True,
                   help='Figure suptitle, e.g. "1sp_AR"')
    p.add_argument('--out',   required=True, help='Output image path')

    # Panel 1: phase portrait
    p.add_argument('--phase-theta-max', type=float, default=5.,
                   help='Half-range of θ axis [deg]')
    p.add_argument('--phase-rate-max',  type=float, default=20.,
                   help='Half-range of θ̇ axis [deg/s]')
    p.add_argument('--phase-theta-pts', type=int,   default=13)
    p.add_argument('--phase-rate-pts',  type=int,   default=11)
    p.add_argument('--probe-samples',   type=int,   default=200,
                   help='States used to fit the ridge probe for panel 1')

    # Panels 2+3: prediction error and planning cost landscapes
    p.add_argument('--pred-theta-max',  type=float, default=25.)
    p.add_argument('--pred-rate-max',   type=float, default=100.)
    p.add_argument('--pred-theta-pts',  type=int,   default=13)
    p.add_argument('--pred-rate-pts',   type=int,   default=11)
    p.add_argument('--horizon',         type=int,   default=3,
                   help='H for panels 2 and 3')

    # Panel 4: latent norm divergence
    p.add_argument('--theta0',      type=float, default=0.1,
                   help='Initial θ [rad] for latent norm panel')
    p.add_argument('--norm-steps',  type=int,   default=20)

    # Panel 5: cosine alignment
    p.add_argument('--n-steps',   type=int,   default=20,
                   help='H_max for cosine alignment panel')
    p.add_argument('--n-starts',  type=int,   default=6,
                   help='Number of perturbation magnitudes')
    p.add_argument('--alpha-max', type=float, default=0.25,
                   help='Max perturbation magnitude along v_u')

    p.add_argument('--device', default='cuda')
    p.add_argument('--skip-panels', type=int, nargs='+', default=[],
                   metavar='N', help='Panel numbers to skip (1=phase portrait, '
                                     '2=pred error, 3=planning cost, '
                                     '4=latent norm, 5=cosine alignment)')
    args = p.parse_args()

    # ── load ──────────────────────────────────────────────────────────────────
    print('[load] bundle …')
    bundle = load_bundle(args.ckpt, args.cfg, args.device)
    _env_patch(bundle)
    mdt = _macro_dt(bundle)

    print('[probe] fitting ridge probe …')
    probe = _build_probe(bundle, n_samples=args.probe_samples)

    print('[GT] computing unstable eigenvector …')
    v_u = _gt_unstable_eigvec(bundle)

    # ── figure ────────────────────────────────────────────────────────────────
    skip = set(args.skip_panels)
    active = [i for i in (1, 2, 3, 4, 5) if i not in skip]
    n_panels = len(active)
    fig, axes_all = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6.5))
    if n_panels == 1:
        axes_all = [axes_all]
    ax_iter = iter(axes_all)
    if 1 not in skip:
        print('[panel 1] phase portrait …')
        ax = next(ax_iter)
        panel_phase_portrait(ax, bundle, probe,
                             args.phase_theta_max, args.phase_rate_max,
                             args.phase_theta_pts, args.phase_rate_pts)

    if 2 not in skip:
        print(f'[panel 2] H={args.horizon} prediction error …')
        ax = next(ax_iter)
        panel_pred_error(ax, fig, bundle,
                         args.pred_theta_max, args.pred_rate_max,
                         args.pred_theta_pts, args.pred_rate_pts, args.horizon)

    if 3 not in skip:
        print(f'[panel 3] planning cost H={args.horizon} …')
        ax = next(ax_iter)
        panel_planning_cost(ax, fig, bundle,
                            args.pred_theta_max, args.pred_rate_max,
                            args.pred_theta_pts, args.pred_rate_pts, args.horizon)

    if 4 not in skip:
        print('[panel 4] latent norm divergence …')
        ax = next(ax_iter)
        panel_latent_norm(ax, bundle, args.theta0, args.norm_steps)

    if 5 not in skip:
        print('[panel 5] cosine alignment …')
        ax = next(ax_iter)
        panel_cosine_alignment(ax, bundle, v_u, mdt,
                               H_max=args.n_steps,
                               n_starts=args.n_starts,
                               alpha_max=args.alpha_max)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    print(f'[done] {out}')


if __name__ == '__main__':
    main()
