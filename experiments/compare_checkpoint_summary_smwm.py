#!/usr/bin/env python
"""Two-model 2×3 summary comparison figure.

Layout:
  row 0 (top)    : 1SP+EP-IDM — phase portrait | pred error | planning cost
  row 1 (bottom) : MSP+SIG    — phase portrait | pred error | planning cost

Vertical model names are written on the left of each row.

Usage
-----
python experiments/compare_checkpoint_summary_smwm.py \\
    --fwd-ckpt /mnt/t7shield/jepa_results/cartpole_smwm_fwd_endpoint_inv_act1_seed42/model_final.pt \\
    --fwd-cfg  configs/cartpole_smwm_fwd_endpoint_inverse_act1.yaml \\
    --ms-ckpt  /mnt/t7shield/jepa_results/cartpole_smwm_sigreg_rollout_act1_seed42/model_final.pt \\
    --ms-cfg   configs/cartpole_smwm_sigreg_rollout_act1.yaml \\
    --out      results/summary_comparison.pdf
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
import numpy as np

from experiments.plot_checkpoint_summary_smwm import (
    _env_patch, _macro_dt, _build_probe, _gt_unstable_eigvec,
    panel_phase_portrait, panel_pred_error, panel_planning_cost,
)
from sensorimotor_probe_utils import load_bundle


def _setup(ckpt, cfg, device, probe_samples):
    print(f'\n[load] {Path(ckpt).parent.name}')
    bundle = load_bundle(ckpt, cfg, device)
    _env_patch(bundle)
    mdt = _macro_dt(bundle)
    print('  [probe] fitting ridge probe …')
    probe = _build_probe(bundle, n_samples=probe_samples)
    print('  [GT] computing unstable eigenvector …')
    v_u = _gt_unstable_eigvec(bundle)
    return bundle, mdt, probe, v_u


def _fill_row(axes_row, fig, bundle, probe,
              phase_kw, pred_kw, horizon, fontsize):
    print('  [panel 1] phase portrait …')
    panel_phase_portrait(axes_row[0], bundle, probe, **phase_kw)

    print(f'  [panel 2] H={horizon} prediction error …')
    panel_pred_error(axes_row[1], fig, bundle, **pred_kw, H=horizon)

    print(f'  [panel 3] planning cost H={horizon} …')
    panel_planning_cost(axes_row[2], fig, bundle, **pred_kw, H=horizon)

    # override font sizes set internally by the panel functions
    for ax in axes_row:
        ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
        ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)
        ax.tick_params(labelsize=fontsize - 2)
        leg = ax.get_legend()
        if leg:
            for t in leg.get_texts():
                t.set_fontsize(fontsize - 2)


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--fwd-ckpt', required=True, help='1SP+EP-IDM checkpoint')
    p.add_argument('--fwd-cfg',  required=True)
    p.add_argument('--ms-ckpt',  required=True, help='MSP+SIG checkpoint')
    p.add_argument('--ms-cfg',   required=True)

    p.add_argument('--fwd-label', default='1SP+EP-IDM')
    p.add_argument('--ms-label',  default='MSP+SIG')

    # phase-portrait grid
    p.add_argument('--phase-theta-max', type=float, default=5.)
    p.add_argument('--phase-rate-max',  type=float, default=20.)
    p.add_argument('--phase-theta-pts', type=int,   default=13)
    p.add_argument('--phase-rate-pts',  type=int,   default=11)

    # pred-error / planning-cost grid
    p.add_argument('--pred-theta-max', type=float, default=25.)
    p.add_argument('--pred-rate-max',  type=float, default=100.)
    p.add_argument('--pred-theta-pts', type=int,   default=13)
    p.add_argument('--pred-rate-pts',  type=int,   default=11)
    p.add_argument('--horizon',        type=int,   default=3)

    p.add_argument('--probe-samples', type=int,   default=200)
    p.add_argument('--fontsize',      type=int,   default=16)
    p.add_argument('--device',        default='cuda')
    p.add_argument('--out',           default='results/summary_comparison.pdf')
    args = p.parse_args()

    phase_kw = dict(theta_max=args.phase_theta_max, rate_max=args.phase_rate_max,
                    n_theta=args.phase_theta_pts,   n_rate=args.phase_rate_pts)
    pred_kw  = dict(theta_max=args.pred_theta_max,  rate_max=args.pred_rate_max,
                    n_theta=args.pred_theta_pts,     n_rate=args.pred_rate_pts)

    fwd_bundle, _, fwd_probe, _ = _setup(
        args.fwd_ckpt, args.fwd_cfg, args.device, args.probe_samples)
    ms_bundle,  _, ms_probe,  _ = _setup(
        args.ms_ckpt,  args.ms_cfg,  args.device, args.probe_samples)

    FS = args.fontsize
    fig, axes = plt.subplots(2, 3, figsize=(21, 13))
    fig.subplots_adjust(left=0.08, hspace=0.35, wspace=0.35)

    print(f'\n=== Row 0: {args.fwd_label} ===')
    _fill_row(axes[0], fig, fwd_bundle, fwd_probe,
              phase_kw, pred_kw, args.horizon, FS)

    print(f'\n=== Row 1: {args.ms_label} ===')
    _fill_row(axes[1], fig, ms_bundle, ms_probe,
              phase_kw, pred_kw, args.horizon, FS)

    # vertical row labels
    for row_idx, label in enumerate([args.fwd_label, args.ms_label]):
        # y-centre of the row in figure coordinates
        row_axes = axes[row_idx]
        ys = [ax.get_position().y0 + ax.get_position().height / 2
              for ax in row_axes]
        y_mid = sum(ys) / len(ys)
        fig.text(0.01, y_mid, label,
                 ha='center', va='center', rotation='vertical',
                 fontsize=FS + 2, fontweight='bold')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    print(f'\n[saved] {out}')


if __name__ == '__main__':
    main()
