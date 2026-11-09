#!/usr/bin/env python
"""Plot mid-rollout perturbation recovery curves from eval_walker_perturbation.py output.

Produces:
  1. x-velocity over time (GT vs model, mean ± stderr across trials)
  2. Torso angle over time (GT vs model)
  3. State MSE over recovery steps (per model)
  4. Summary bar chart: mean state MSE / survival accuracy

Usage
-----
python experiments/plot_walker_perturbation.py \\
    --json results/walker2d_perturbation.json \\
    --out  results/perturbation_fig.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.family':  'serif',
    'font.serif':   ['Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

MODEL_COLORS = [
    '#1f77b4',  # blue
    '#d62728',  # red
    '#2ca02c',  # green
    '#ff7f0e',  # orange
    '#9467bd',  # purple
]
GT_COLOR   = '#333333'
GT_STYLE   = dict(color=GT_COLOR, lw=1.4, ls='--', label='MuJoCo GT')
PERT_COLOR = '#e377c2'


def stderr(x: np.ndarray) -> np.ndarray:
    return x.std(axis=0) / np.sqrt(x.shape[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--json', required=True)
    p.add_argument('--out',  default='results/perturbation_fig.pdf')
    p.add_argument('--ci',   action='store_true', default=True,
                   help='Shade ±1 stderr confidence interval')
    args = p.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    cfg     = data['config']
    models  = data['models']
    T       = cfg['recovery_steps']
    steps   = np.arange(T)

    n_models = len(models)
    fig = plt.figure(figsize=(5 * n_models * 0.6 + 3, 10))
    gs  = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35,
                            left=0.09, right=0.97, top=0.92, bottom=0.07)

    ax_xvel   = fig.add_subplot(gs[0, :])   # x-velocity — full width
    ax_ang    = fig.add_subplot(gs[1, 0])   # torso angle
    ax_mse    = fig.add_subplot(gs[1, 1])   # state MSE curve
    ax_bar    = fig.add_subplot(gs[2, 0])   # summary bar: mean MSE
    ax_surv   = fig.add_subplot(gs[2, 1])   # survival rate: GT vs pred

    # ── per-model curves ──────────────────────────────────────────────────────
    for idx, model in enumerate(models):
        color   = MODEL_COLORS[idx % len(MODEL_COLORS)]
        trials  = model['trials']
        name    = model['name']

        # Collect arrays: (n_trials, T, 17) for gt and pred
        gt_arr   = np.array([t['gt_states']   for t in trials], dtype=np.float32)
        pred_arr = np.array([t['pred_states']  for t in trials], dtype=np.float32)
        mse_arr  = np.array([t['state_mse_per_step'] for t in trials], dtype=np.float32)

        # GT plotted once (same across models if same SAC + same seeds)
        if idx == 0:
            gt_xvel_m = gt_arr[:, :, 8].mean(axis=0)
            gt_xvel_e = stderr(gt_arr[:, :, 8])
            ax_xvel.plot(steps, gt_xvel_m, **GT_STYLE)
            if args.ci:
                ax_xvel.fill_between(steps,
                    gt_xvel_m - gt_xvel_e, gt_xvel_m + gt_xvel_e,
                    alpha=0.15, color=GT_COLOR)
            gt_ang_m = gt_arr[:, :, 1].mean(axis=0)
            gt_ang_e = stderr(gt_arr[:, :, 1])
            ax_ang.plot(steps, gt_ang_m, **GT_STYLE)
            if args.ci:
                ax_ang.fill_between(steps,
                    gt_ang_m - gt_ang_e, gt_ang_m + gt_ang_e,
                    alpha=0.15, color=GT_COLOR)

        # Model prediction curves
        pred_xvel_m = pred_arr[:, :, 8].mean(axis=0)
        pred_xvel_e = stderr(pred_arr[:, :, 8])
        ax_xvel.plot(steps, pred_xvel_m, color=color, lw=1.6, label=name)
        if args.ci:
            ax_xvel.fill_between(steps,
                pred_xvel_m - pred_xvel_e, pred_xvel_m + pred_xvel_e,
                alpha=0.15, color=color)

        pred_ang_m = pred_arr[:, :, 1].mean(axis=0)
        pred_ang_e = stderr(pred_arr[:, :, 1])
        ax_ang.plot(steps, pred_ang_m, color=color, lw=1.6, label=name)
        if args.ci:
            ax_ang.fill_between(steps,
                pred_ang_m - pred_ang_e, pred_ang_m + pred_ang_e,
                alpha=0.15, color=color)

        mse_m = mse_arr.mean(axis=0)
        mse_e = stderr(mse_arr)
        ax_mse.plot(steps, mse_m, color=color, lw=1.6, label=name)
        if args.ci:
            ax_mse.fill_between(steps, mse_m - mse_e, mse_m + mse_e,
                                alpha=0.15, color=color)

    # Mark perturbation step (always at step 0 of recovery, but label it)
    for ax in (ax_xvel, ax_ang):
        ax.axvline(0, color=PERT_COLOR, lw=1.2, ls=':', alpha=0.8,
                   label='perturbation')

    # ── axes cosmetics ────────────────────────────────────────────────────────
    ax_xvel.set_xlabel('Recovery step')
    ax_xvel.set_ylabel('x-velocity (m/s)')
    ax_xvel.set_title('x-Velocity: Model Prediction vs MuJoCo Ground Truth',
                      fontsize=10, fontweight='bold')
    ax_xvel.legend(fontsize=8, ncol=min(n_models + 2, 5), loc='best')

    ang_thresh = 1.0
    ax_ang.axhline( ang_thresh, color='#aaaaaa', lw=0.8, ls='--')
    ax_ang.axhline(-ang_thresh, color='#aaaaaa', lw=0.8, ls='--')
    ax_ang.set_xlabel('Recovery step')
    ax_ang.set_ylabel('Torso angle (rad)')
    ax_ang.set_title('Torso Angle', fontsize=10, fontweight='bold')
    ax_ang.legend(fontsize=7)

    ax_mse.set_xlabel('Recovery step')
    ax_mse.set_ylabel('State MSE')
    ax_mse.set_title('Prediction MSE over Recovery', fontsize=10, fontweight='bold')
    ax_mse.legend(fontsize=7)

    # ── bar charts ────────────────────────────────────────────────────────────
    names      = [m['name']                      for m in models]
    mse_means  = [m['summary']['mean_state_mse'] for m in models]
    mse_stds   = [m['summary']['std_state_mse']  for m in models]
    gt_survs   = [m['summary'].get('mean_gt_surv_rate',   m['summary'].get('mean_surv_acc', 0))
                  for m in models]
    pred_survs = [m['summary'].get('mean_pred_surv_rate', m['summary'].get('mean_surv_acc', 0))
                  for m in models]

    xs  = np.arange(len(names))
    w   = 0.35
    ax_bar.bar(xs, mse_means, yerr=mse_stds,
               color=MODEL_COLORS[:len(names)],
               capsize=4, error_kw=dict(elinewidth=1))
    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
    ax_bar.set_ylabel('Mean state MSE')
    ax_bar.set_title('Recovery State MSE\n(lower = model better tracks GT)',
                     fontsize=9, fontweight='bold')

    # Grouped bar: GT survival rate vs model predicted survival rate
    ax_surv.bar(xs - w/2, gt_survs,   width=w, color=GT_COLOR,   alpha=0.85, label='MuJoCo GT')
    ax_surv.bar(xs + w/2, pred_survs, width=w,
                color=MODEL_COLORS[:len(names)], alpha=0.85, label='Model (SAC on probe)')
    ax_surv.set_xticks(xs)
    ax_surv.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
    ax_surv.set_ylim(0, 1.15)
    ax_surv.axhline(1.0, color='#aaaaaa', lw=0.8, ls='--')
    ax_surv.set_ylabel('Survival rate')
    ax_surv.set_title('Survival Rate: GT vs Model SAC\n(should match for accurate dynamics)',
                      fontsize=9, fontweight='bold')
    ax_surv.legend(fontsize=7)

    # ── super-title ───────────────────────────────────────────────────────────
    pert_desc = (f'{cfg["perturb_type"]} kick  |  '
                 f'mag={cfg["perturb_mag"]}  |  '
                 f'at step {cfg["perturb_step"]}  |  '
                 f'{cfg["trials"]} trials × {cfg["recovery_steps"]} recovery steps')
    fig.suptitle(f'Walker2D Mid-Rollout Perturbation Ablation\n{pert_desc}',
                 fontsize=10, fontweight='bold', y=0.995)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches='tight')
    print(f'[done] {out}')


if __name__ == '__main__':
    main()
