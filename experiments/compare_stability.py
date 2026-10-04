#!/usr/bin/env python
"""Four-panel stability comparison: 1SP+EP-IDM vs MSP+SIG.

Layout (1 × 4):
  [1SP+EP-IDM ROA] [MSP+SIG ROA] [1SP+EP-IDM Certificate] [MSP+SIG Certificate]

ROA panels show the empirical region of attraction for the learned-LQR
controller (green = stabilises, red = fails, dashed = GT-LQR boundary).

Certificate panels show the Lyapunov decrease condition ΔV = V(x') − V(x)
under one macro-step of the learned-LQR controller on the GT environment
(blue = decreasing, red = increasing).

Usage
-----
python experiments/compare_stability_smwm.py \\
    --fwd-ckpt /mnt/t7shield/jepa_results/cartpole_smwm_fwd_endpoint_inv_act1_seed42/model_final.pt \\
    --fwd-cfg  configs/cartpole_jepa_fwd_endpoint_inverse_act1.yaml \\
    --ms-ckpt  /mnt/t7shield/jepa_results/cartpole_smwm_sigreg_rollout_act1_seed42/model_final.pt \\
    --ms-cfg   configs/cartpole_jepa_sigreg_rollout_act1.yaml \\
    --out      results/stability_comparison.pdf
"""
from __future__ import annotations

import argparse
import sys
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

from experiments.probe_stability import (
    _env_patch, _build_probe, _solve_dare, _gt_physical_linearization,
    panel_roa, panel_lyapunov,
)
from experiments.probe_utils import load_bundle, equilibrium_latent, local_jacobians


def _setup_model(ckpt, cfg, device, probe_samples, q_scale, r_scale):
    """Load bundle, fit probe, solve DAREs. Returns a dict of everything needed."""
    print(f'\n[load] {Path(ckpt).parent.name}')
    bundle = load_bundle(ckpt, cfg, device)
    _env_patch(bundle)

    print('  [probe] fitting ridge probe …')
    probe = _build_probe(bundle, probe_samples)

    print('  [jacobians] learned latent linearisation …')
    z_eq = equilibrium_latent(bundle)
    A_z, B_z, fp_err = local_jacobians(bundle, z=z_eq)
    z_eq_np = z_eq.cpu().numpy().flatten()
    print(f'  fp_err={fp_err:.4e}')

    d = A_z.shape[0]
    print('  [DARE] learned latent …')
    K_z, _ = _solve_dare(A_z, B_z, q_scale * np.eye(d), np.array([[r_scale]]))
    print(f'  ρ(A_z - B_z K_z) = '
          f'{np.max(np.abs(np.linalg.eigvals(A_z - B_z @ K_z))):.4f}')

    print('  [GT linearisation] physical space …')
    A_p, B_p = _gt_physical_linearization(bundle)
    print('  [DARE] GT physical …')
    K_phys, P_phys = _solve_dare(A_p, B_p, q_scale * np.eye(4), np.array([[r_scale]]))
    print(f'  ρ(A_p - B_p K_p) = '
          f'{np.max(np.abs(np.linalg.eigvals(A_p - B_p @ K_phys))):.4f}')

    return dict(bundle=bundle, probe=probe, K_z=K_z, z_eq_np=z_eq_np,
                K_phys=K_phys, P_phys=P_phys)


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--fwd-ckpt', required=True, help='1SP+EP-IDM checkpoint')
    p.add_argument('--fwd-cfg',  required=True)
    p.add_argument('--ms-ckpt',  required=True, help='MSP+SIG checkpoint')
    p.add_argument('--ms-cfg',   required=True)

    p.add_argument('--fwd-label', default='1SP+EP-IDM')
    p.add_argument('--ms-label',  default='MSP+SIG')

    # ROA grid
    p.add_argument('--roa-theta-max', type=float, default=25.)
    p.add_argument('--roa-rate-max',  type=float, default=80.)
    p.add_argument('--roa-theta-pts', type=int,   default=17)
    p.add_argument('--roa-rate-pts',  type=int,   default=13)

    # Lyapunov grid
    p.add_argument('--lyap-theta-max', type=float, default=15.)
    p.add_argument('--lyap-rate-max',  type=float, default=50.)
    p.add_argument('--lyap-theta-pts', type=int,   default=17)
    p.add_argument('--lyap-rate-pts',  type=int,   default=13)

    # LQR / episode
    p.add_argument('--n-steps',    type=int,   default=60)
    p.add_argument('--threshold',  type=float, default=0.3)
    p.add_argument('--hold-steps', type=int,   default=10)
    p.add_argument('--q-scale',    type=float, default=1.0)
    p.add_argument('--r-scale',    type=float, default=1.0)

    p.add_argument('--probe-samples', type=int, default=300)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', default='results/stability_comparison.pdf')
    args = p.parse_args()

    kw = dict(probe_samples=args.probe_samples,
              q_scale=args.q_scale, r_scale=args.r_scale,
              device=args.device)

    fwd = _setup_model(args.fwd_ckpt, args.fwd_cfg, **kw)
    ms  = _setup_model(args.ms_ckpt,  args.ms_cfg,  **kw)

    roa_kw  = dict(theta_max=args.roa_theta_max,  rate_max=args.roa_rate_max,
                   n_theta=args.roa_theta_pts,     n_rate=args.roa_rate_pts,
                   n_steps=args.n_steps, threshold=args.threshold,
                   hold_steps=args.hold_steps)
    lyap_kw = dict(theta_max=args.lyap_theta_max, rate_max=args.lyap_rate_max,
                   n_theta=args.lyap_theta_pts,    n_rate=args.lyap_rate_pts)

    fig, axes = plt.subplots(1, 4, figsize=(28, 6))

    XLABEL = r'$\theta$ [deg]'
    YLABEL = r'$\dot\theta$ [deg/s]'
    FS = 20

    # ── Panels 0-1: ROA ──────────────────────────────────────────────────────
    for ax, m, label in [(axes[0], fwd, args.fwd_label),
                         (axes[1], ms,  args.ms_label)]:
        print(f'\n[ROA] {label} …')
        panel_roa(ax, m['bundle'], m['K_z'], m['z_eq_np'], m['K_phys'],
                  **roa_kw)
        ax.set_title(label, fontsize=FS, fontweight='bold', pad=6)
        ax.set_xlabel(XLABEL, fontsize=FS)
        ax.set_ylabel(YLABEL, fontsize=FS)
        ax.tick_params(labelsize=FS - 2)

    # shared ROA legend on the first ROA panel only
    axes[0].legend(handles=[
        mpatches.Patch(color='green', alpha=.85, label='Learned LQR stabilises'),
        mpatches.Patch(color='red',   alpha=.85, label='Learned LQR fails'),
        Line2D([0], [0], color='k', lw=1.5, ls='--', label='GT LQR boundary'),
    ], fontsize=16, loc='upper right')
    axes[1].get_legend().remove() if axes[1].get_legend() else None

    # ── Panels 2-3: Lyapunov certificate ─────────────────────────────────────
    for ax, m, label in [(axes[2], fwd, args.fwd_label),
                         (axes[3], ms,  args.ms_label)]:
        print(f'\n[Certificate] {label} …')
        panel_lyapunov(ax, m['bundle'], m['K_z'], m['z_eq_np'], m['P_phys'],
                       **lyap_kw, fontsize=FS)
        ax.set_title(label, fontsize=FS, fontweight='bold', pad=6)
        ax.set_xlabel(XLABEL, fontsize=FS)
        ax.set_ylabel(YLABEL, fontsize=FS)
        ax.tick_params(labelsize=FS - 2)


    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    print(f'\n[saved] {out}')


if __name__ == '__main__':
    main()
