#!/usr/bin/env python
"""Local stability probe: 3-panel diagnostic figure.

Panel 1  – Local vector-field comparison
           GT vs learned one-step transitions on a θ–θ̇ grid around equilibrium.
           Reports normalised mean vector error.

Panel 2  – Empirical region of attraction (ROA)
           Grid of (θ, θ̇) initial conditions; each run closed-loop on the GT
           environment using the LQR gain derived from the *learned* latent
           linearisation.  Background shows learned-LQR success; a dashed
           contour marks the GT-LQR success boundary for reference.

Panel 3  – Local Lyapunov decrease condition
           Quadratic Lyapunov function V(x) = xᵀ P x built from the GT
           physical DARE (Q=I₄, R=1).  One macro-step of the learned-LQR
           controller is applied on the GT system and ΔV = V(x') − V(x) is
           plotted; blue = decreasing (certificate holds), red = increasing.

Usage:
  python experiments/probe_local_stability_smwm.py \\
      --ckpt  path/to/model.pt \\
      --cfg   path/to/config.yaml \\
      --title "1sp AR" \\
      --out   results/stability_1sp_AR.png
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import scipy.linalg
import torch

from sensorimotor_probe_utils import make_env
from sensorimotor_probe_utils import (
    RidgeStateProbe,
    encode_obs,
    encode_rendered_state,
    equilibrium_latent,
    local_jacobians,
    load_bundle,
)


# ── shared helpers ─────────────────────────────────────────────────────────────

def _env_patch(bundle):
    if 'environment' not in bundle['env_cfg']:
        bundle['env_cfg']['environment'] = {
            'frame_skip': int(bundle['model_cfg'].get('frame_skip', 5)),
            'image_size': int(bundle['model_cfg'].get('image_size', 128)),
            'action_range': [-10, 10],
            'mass_cart': 1.0, 'mass_pole': 0.1,
            'pole_length': 0.5, 'gravity': 9.8, 'dt': 0.02,
        }


def _build_probe(bundle, n_samples=300, seed=7):
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
    return RidgeStateProbe().fit(np.asarray(zs), states, 1e-3)


def _solve_dare(A, B, Q, R):
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


def _gt_physical_linearization(bundle, eps=1e-3):
    """Finite-difference linearisation of GT macro-step dynamics at origin."""
    env = make_env(bundle['env_cfg'], seed=42)
    x0 = np.zeros(4, np.float32)
    n = 4
    A = np.zeros((n, n))
    B = np.zeros((n, 1))
    for j in range(n):
        d = np.zeros(n, np.float32); d[j] = eps
        _, sp, _ = env.reset_to_state((x0 + d).astype(np.float64))
        _, snp, _, _, _ = env.step(0.)
        xp = snp.copy()
        _, sm, _ = env.reset_to_state((x0 - d).astype(np.float64))
        _, snm, _, _, _ = env.step(0.)
        xm = snm.copy()
        A[:, j] = (xp - xm) / (2. * eps)
    ua = float(eps * bundle['action_scale'])
    env.reset_to_state(x0.astype(np.float64))
    _, snp, _, _, _ = env.step(ua)
    xp = snp.copy()
    env.reset_to_state(x0.astype(np.float64))
    _, snm, _, _, _ = env.step(-ua)
    xm = snm.copy()
    B[:, 0] = (xp - xm) / (2. * ua)
    env.close()
    return A, B


def _lqr_episode(bundle, K_z, z_eq, initial_state, n_steps,
                 threshold, hold_steps, seed):
    """Run one closed-loop episode with learned-latent LQR on GT env."""
    env = make_env(bundle['env_cfg'], seed)
    obs, state, _ = env.reset_to_state(
        np.asarray(initial_state, dtype=np.float64))
    prev_obs = obs.copy()
    as_ = bundle['action_scale']
    states = [state.copy()]
    terminated = False
    for _ in range(n_steps):
        with torch.no_grad():
            z = encode_obs(bundle, obs, prev_obs, state).cpu().numpy().flatten()
        u = float(-(K_z @ (z - z_eq)).item())
        u = float(np.clip(u, -as_, as_))
        prev_obs = obs.copy()
        obs, state, _, done, _ = env.step(u)
        states.append(state.copy())
        if done:
            terminated = True
            break
    env.close()
    states = np.asarray(states)
    errors = np.linalg.norm(states, axis=1)
    tail = errors[-hold_steps:]
    held = bool(not terminated and len(tail) == hold_steps and
                np.all(tail < threshold))
    return held, states


# ── Panel 1: local vector-field comparison ────────────────────────────────────

def panel_vector_field(ax, bundle, probe, theta_max, rate_max,
                       n_theta, n_rate, seed=11):
    theta_deg = np.linspace(-theta_max, theta_max, n_theta)
    rate_deg  = np.linspace(-rate_max,  rate_max,  n_rate)
    theta_mesh, rate_mesh = np.meshgrid(theta_deg, rate_deg)

    env = make_env(bundle['env_cfg'], seed)
    mdt = 1.  # arrows are per-macro-step, absolute scale via frac below

    gt_u = np.zeros_like(theta_mesh); gt_v = np.zeros_like(theta_mesh)
    lr_u = np.zeros_like(theta_mesh); lr_v = np.zeros_like(theta_mesh)
    err_norms = []

    for idx in np.ndindex(theta_mesh.shape):
        state = np.array([0., 0.,
                          np.deg2rad(theta_mesh[idx]),
                          np.deg2rad(rate_mesh[idx])], dtype=np.float32)
        obs, s0, _ = env.reset_to_state(state)
        z0 = encode_obs(bundle, obs, obs, state)

        # GT one-step
        _, s1, _, _, _ = env.step(0.)
        dth_gt  = np.rad2deg(s1[2] - s0[2])
        drate_gt = np.rad2deg(s1[3] - s0[3])
        gt_u[idx] = dth_gt; gt_v[idx] = drate_gt

        # Learned one-step decoded via probe
        from sensorimotor_probe_utils import predict_one
        z1 = predict_one(bundle, z0, 0.).detach().cpu().numpy()[0]
        dec = probe(np.stack([z0.cpu().numpy()[0], z1]))
        dth_lr  = np.rad2deg(dec[1, 2] - dec[0, 2])
        drate_lr = np.rad2deg(dec[1, 3] - dec[0, 3])
        lr_u[idx] = dth_lr; lr_v[idx] = drate_lr

        err_norms.append(np.sqrt((dth_gt - dth_lr)**2 + (drate_gt - drate_lr)**2))

    env.close()

    gt_mag = np.sqrt(gt_u**2 + gt_v**2)
    lr_mag = np.sqrt(lr_u**2 + lr_v**2)
    ref_mag = np.maximum(gt_mag, 1e-8)
    norm_err = float(np.mean(err_norms) / np.maximum(np.mean(gt_mag), 1e-8))

    frac = 0.45
    dx = 2 * theta_max / max(n_theta - 1, 1)
    dy = 2 * rate_max  / max(n_rate  - 1, 1)

    def _q(u, v, **kw):
        n = np.sqrt(u**2 + v**2 + 1e-14)
        ax.quiver(theta_mesh, rate_mesh,
                  u / n * frac * dx, v / n * frac * dy,
                  pivot='mid', scale=1, scale_units='xy', angles='xy',
                  width=.0025, headwidth=4, headlength=4, **kw)

    _q(gt_u, gt_v, color='#2ecc40', alpha=.75)
    _q(lr_u, lr_v, color='darkorange', alpha=.75)

    ax.scatter([0.], [0.], marker='*', s=200, color='red', zorder=5)
    ax.set_xlabel(r'$\theta$ [deg]', fontsize=14)
    ax.set_ylabel(r'$\dot\theta$ [deg/s]', fontsize=14)
    ax.legend(handles=[
        Line2D([0], [0], color='#2ecc40',   lw=3, label='GT'),
        Line2D([0], [0], color='darkorange', lw=3, label='Learned'),
    ], fontsize=14, loc='upper right')
    ax.tick_params(labelsize=14)
    ax.grid(alpha=.2)


# ── Panel 2: empirical ROA ─────────────────────────────────────────────────────

def panel_roa(ax, bundle, K_z, z_eq, K_phys,
              theta_max, rate_max, n_theta, n_rate,
              n_steps, threshold, hold_steps):
    theta_deg = np.linspace(-theta_max, theta_max, n_theta)
    rate_deg  = np.linspace(-rate_max,  rate_max,  n_rate)
    theta_mesh, rate_mesh = np.meshgrid(theta_deg, rate_deg)

    # Learned-LQR success map
    success_map = np.zeros(theta_mesh.shape, dtype=bool)
    for idx in np.ndindex(theta_mesh.shape):
        state = np.array([0., 0.,
                          np.deg2rad(theta_mesh[idx]),
                          np.deg2rad(rate_mesh[idx])], dtype=np.float32)
        held, _ = _lqr_episode(bundle, K_z, z_eq, state,
                                n_steps, threshold, hold_steps, seed=0)
        success_map[idx] = held

    ax.pcolormesh(theta_deg, rate_deg,
                  success_map.astype(float),
                  cmap='RdYlGn', vmin=0, vmax=1,
                  shading='auto', alpha=.85)

    # GT LQR success contour
    gt_success = np.zeros_like(success_map, dtype=float)
    for idx in np.ndindex(theta_mesh.shape):
        state = np.array([0., 0.,
                          np.deg2rad(theta_mesh[idx]),
                          np.deg2rad(rate_mesh[idx])], dtype=np.float32)
        env = make_env(bundle['env_cfg'], seed=0)
        obs0, s0, _ = env.reset_to_state(state.astype(np.float64))
        as_ = bundle['action_scale']
        states_gt = [s0.copy()]
        term = False
        for _ in range(n_steps):
            u = float(-(K_phys @ s0).item())
            u = float(np.clip(u, -as_, as_))
            _, s0, _, done, _ = env.step(u)
            states_gt.append(s0.copy())
            if done: term = True; break
        env.close()
        errs = np.linalg.norm(np.asarray(states_gt), axis=1)
        tail = errs[-hold_steps:]
        gt_success[idx] = float(not term and len(tail) == hold_steps and
                                 np.all(tail < threshold))

    try:
        ax.contour(theta_deg, rate_deg, gt_success,
                   levels=[0.5], colors='k', linewidths=1.5,
                   linestyles='--')
    except Exception:
        pass

    ax.scatter([0.], [0.], marker='*', s=200, color='blue', zorder=5)
    ax.set_xlabel(r'$\theta$ [deg]', fontsize=14)
    ax.set_ylabel(r'$\dot\theta$ [deg/s]', fontsize=14)
    ax.legend(handles=[
        mpatches.Patch(color='green',  alpha=.85, label='Learned LQR stabilises'),
        mpatches.Patch(color='red',    alpha=.85, label='Learned LQR fails'),
        Line2D([0], [0], color='k', lw=1.5, ls='--', label='GT LQR boundary'),
    ], fontsize=14, loc='upper right')
    ax.tick_params(labelsize=14)
    ax.grid(alpha=.15)


# ── Panel 3: Lyapunov decrease condition ──────────────────────────────────────

def panel_lyapunov(ax, bundle, K_z, z_eq, P_phys,
                   theta_max, rate_max, n_theta, n_rate):
    theta_deg = np.linspace(-theta_max, theta_max, n_theta)
    rate_deg  = np.linspace(-rate_max,  rate_max,  n_rate)
    theta_mesh, rate_mesh = np.meshgrid(theta_deg, rate_deg)

    dV_map = np.zeros(theta_mesh.shape)

    for idx in np.ndindex(theta_mesh.shape):
        state = np.array([0., 0.,
                          np.deg2rad(theta_mesh[idx]),
                          np.deg2rad(rate_mesh[idx])], dtype=np.float32)
        V0 = float(state @ P_phys @ state)

        # Apply one step of learned LQR on GT env
        env = make_env(bundle['env_cfg'], seed=0)
        obs, s0, _ = env.reset_to_state(state.astype(np.float64))
        with torch.no_grad():
            z = encode_obs(bundle, obs, obs, state).cpu().numpy().flatten()
        u = float(-(K_z @ (z - z_eq)).item())
        u = float(np.clip(u, -bundle['action_scale'], bundle['action_scale']))
        _, s1, _, _, _ = env.step(u)
        env.close()

        s1 = s1.astype(np.float64)
        V1 = float(s1 @ P_phys @ s1)
        dV_map[idx] = V1 - V0

    vabs = np.percentile(np.abs(dV_map), 95)
    im = ax.pcolormesh(theta_deg, rate_deg, dV_map,
                       cmap='RdBu_r', shading='auto',
                       vmin=-vabs, vmax=vabs)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$\Delta V = V(x^\prime) - V(x)$', fontsize=14)
    cbar.ax.tick_params(labelsize=14)
    ax.contour(theta_deg, rate_deg, dV_map,
               levels=[0.], colors='k', linewidths=1.2)

    ax.scatter([0.], [0.], marker='*', s=200, color='lime', zorder=5,
               edgecolors='darkgreen', lw=0.8)
    ax.set_xlabel(r'$\theta$ [deg]', fontsize=14)
    ax.set_ylabel(r'$\dot\theta$ [deg/s]', fontsize=14)
    ax.tick_params(labelsize=14)
    ax.grid(alpha=.15)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--ckpt',  required=True)
    p.add_argument('--cfg',   required=True)
    p.add_argument('--title', required=True)
    p.add_argument('--out',   required=True)

    # Grid sizes
    p.add_argument('--vf-theta-max',  type=float, default=8.,
                   help='Vector-field θ half-range [deg]')
    p.add_argument('--vf-rate-max',   type=float, default=30.,
                   help='Vector-field θ̇ half-range [deg/s]')
    p.add_argument('--vf-theta-pts',  type=int,   default=11)
    p.add_argument('--vf-rate-pts',   type=int,   default=9)

    p.add_argument('--roa-theta-max', type=float, default=25.)
    p.add_argument('--roa-rate-max',  type=float, default=80.)
    p.add_argument('--roa-theta-pts', type=int,   default=17)
    p.add_argument('--roa-rate-pts',  type=int,   default=13)

    p.add_argument('--lyap-theta-max', type=float, default=15.)
    p.add_argument('--lyap-rate-max',  type=float, default=50.)
    p.add_argument('--lyap-theta-pts', type=int,   default=17)
    p.add_argument('--lyap-rate-pts',  type=int,   default=13)

    # LQR / episode
    p.add_argument('--n-steps',   type=int,   default=60,
                   help='Macro-steps per ROA episode')
    p.add_argument('--threshold', type=float, default=0.3,
                   help='State-norm threshold for success')
    p.add_argument('--hold-steps', type=int,  default=10)
    p.add_argument('--q-scale',   type=float, default=1.0)
    p.add_argument('--r-scale',   type=float, default=1.0)

    # Probe
    p.add_argument('--probe-samples', type=int, default=300)

    p.add_argument('--device', default='cuda')
    p.add_argument('--skip-panels', type=int, nargs='+', default=[],
                   metavar='N', help='Panel numbers to skip (1=vector field, '
                                     '2=ROA, 3=Lyapunov)')
    args = p.parse_args()

    # ── load ──────────────────────────────────────────────────────────────────
    print('[load] bundle …')
    bundle = load_bundle(args.ckpt, args.cfg, args.device)
    _env_patch(bundle)

    print('[probe] fitting ridge probe …')
    probe = _build_probe(bundle, args.probe_samples)

    print('[jacobians] learned latent linearisation …')
    z_eq = equilibrium_latent(bundle)
    A_z, B_z, fp_err = local_jacobians(bundle, z=z_eq)
    z_eq_np = z_eq.cpu().numpy().flatten()
    print(f'  fp_err={fp_err:.4e}')

    d = A_z.shape[0]
    Q_z = args.q_scale * np.eye(d)
    R_z = np.array([[args.r_scale]])
    print('[DARE] learned latent …')
    try:
        K_z, P_z = _solve_dare(A_z, B_z, Q_z, R_z)
        print(f'  ρ(A_z - B_z K_z) = '
              f'{np.max(np.abs(np.linalg.eigvals(A_z - B_z @ K_z))):.4f}')
    except Exception as e:
        print(f'  DARE failed: {e}'); raise

    print('[GT linearisation] physical space …')
    A_p, B_p = _gt_physical_linearization(bundle)
    Q_p = args.q_scale * np.eye(4)
    R_p = np.array([[args.r_scale]])
    print('[DARE] GT physical …')
    K_phys, P_phys = _solve_dare(A_p, B_p, Q_p, R_p)
    print(f'  ρ(A_p - B_p K_p) = '
          f'{np.max(np.abs(np.linalg.eigvals(A_p - B_p @ K_phys))):.4f}')

    # ── figure ────────────────────────────────────────────────────────────────
    skip = set(args.skip_panels)
    active = [i for i in (1, 2, 3) if i not in skip]
    n_panels = len(active)
    fig, axes_all = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    if n_panels == 1:
        axes_all = [axes_all]
    ax_iter = iter(axes_all)
    if 1 not in skip:
        print('[panel 1] vector field …')
        panel_vector_field(next(ax_iter), bundle, probe,
                           args.vf_theta_max, args.vf_rate_max,
                           args.vf_theta_pts, args.vf_rate_pts)

    if 2 not in skip:
        print('[panel 2] empirical ROA …')
        panel_roa(next(ax_iter), bundle, K_z, z_eq_np, K_phys,
                  args.roa_theta_max, args.roa_rate_max,
                  args.roa_theta_pts, args.roa_rate_pts,
                  args.n_steps, args.threshold, args.hold_steps)

    if 3 not in skip:
        print('[panel 3] Lyapunov condition …')
        panel_lyapunov(next(ax_iter), bundle, K_z, z_eq_np, P_phys,
                       args.lyap_theta_max, args.lyap_rate_max,
                       args.lyap_theta_pts, args.lyap_rate_pts)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    print(f'[done] {out}')


if __name__ == '__main__':
    main()
