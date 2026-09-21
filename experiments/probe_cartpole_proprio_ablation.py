#!/usr/bin/env python
"""Proprioception ablation probe for CartPole JEPA checkpoints.

For each checkpoint encode the dataset twice:
  (a) with-proprio  — normalized true state fed to the encoder (standard)
  (b) zero-proprio  — zero vector fed instead (pure visual features only)

Then fit a ridge regression onto the 4-D true state [x, dx, theta, dtheta]
and report R² per dimension and overall, revealing how much the encoder
relies on the injected proprio vs. what it learns from pixels.

Usage
-----
  python experiments/probe_cartpole_proprio_ablation.py \\
      --data  data/cartpole_excitation_depth4_fs5_128 \\
      --ckpts $R/cartpole/diff_proprio_ar/model_final.pt \\
              $R/cartpole/diff_proprio_sigreg/model_final.pt \\
              $R/cartpole/diff_proprio_ar_1step_ms/model_final.pt \\
              $R/cartpole/diff_proprio_sigreg_rollout/model_final.pt \\
              $R/cartpole/ibot_rollout/model_final.pt \\
      --cfgs  configs/cartpole_sensorimotor_world_model_diff_proprio.yaml \\
              configs/cartpole_sensorimotor_world_model_diff_proprio_sigreg.yaml \\
              configs/cartpole_sensorimotor_world_model_diff_proprio_ar_1step_ms.yaml \\
              configs/cartpole_sensorimotor_world_model_diff_proprio_sigreg_rollout.yaml \\
              configs/cartpole_sensorimotor_world_model_ibot_rollout.yaml \\
      --labels "AR-1step" "SIGReg" "AR-1step+MS" "SIGReg+rollout" "iBOT+rollout" \\
      --output results/probe_cartpole_proprio_ablation.pdf
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
from sklearn.metrics import r2_score

from data.dataset import make_discrete_dataloaders
from sensorimotor_probe_utils import RidgeStateProbe, load_bundle

STATE_LABELS = ['x', 'ẋ', 'θ', 'θ̇']


# ── data collection ───────────────────────────────────────────────────────────

@torch.no_grad()
def collect_latents(bundle, data_path, n_samples=4000, batch_size=128,
                    split='train'):
    """Encode the dataset in two modes.

    Returns
    -------
    z_with : (N, D) ndarray   — latents with true proprio
    z_zero : (N, D) ndarray   — latents with zero proprio
    y      : (N, 4) ndarray   — un-normalised state [x, dx, theta, dtheta]
    """
    # Guard against checkpoints that saved zero or NaN state statistics
    # (e.g. no-proprio models that never used the state during training).
    # Replacing zeros/NaN with 1.0 / 0.0 gives identity (un-)normalisation
    # and avoids propagating NaN into the state labels y.
    safe_std  = bundle['state_std'].copy()
    safe_mean = bundle['state_mean'].copy()
    bad_std  = (safe_std  == 0) | ~np.isfinite(safe_std)
    bad_mean = ~np.isfinite(safe_mean)
    if bad_std.any():
        print(f'  [info] state_std has zero/NaN in dims '
              f'{list(np.where(bad_std)[0])}; using 1.0 for those dims')
        safe_std[bad_std] = 1.0
    if bad_mean.any():
        safe_mean[bad_mean] = 0.0

    loaders = make_discrete_dataloaders(
        data_path, batch_size=batch_size, num_workers=0, horizon=1,
        frame_stack=1,
        state_mean=safe_mean, state_std=safe_std,
        action_scale=bundle['action_scale'],
        target_image_size=int(bundle['model_cfg'].get('image_size', 128)),
        preload_obs=True)

    model  = bundle['model']
    device = bundle['device']

    z_with_list, z_zero_list, y_list = [], [], []
    n_collected = 0

    for batch in loaders[split]:
        current  = batch['obs_seq'][:, 0].to(device).float() / 255.
        previous = batch['prev_obs'].to(device).float() / 255.
        y_norm   = batch['states'][:, 0].numpy()            # (B, 4) normalised
        B        = current.shape[0]

        # (a) with true proprio
        if model.use_proprio:
            proprio_true = batch['states'][:, 0].to(device).float()
            # apply proprio_indices if the model uses a subset of state dims
            if model.proprio_indices is not None:
                proprio_true = proprio_true[:, model.proprio_indices]
        else:
            proprio_true = None
        z_with = model.encode(current, previous, proprio_true)

        # (b) zero proprio — same shape, all zeros
        if model.use_proprio:
            proprio_zero = torch.zeros(B, model.proprio_dim,
                                       device=device, dtype=z_with.dtype)
        else:
            proprio_zero = None
        z_zero = model.encode(current, previous, proprio_zero)

        # un-normalise state for interpretable R²
        y = y_norm * safe_std[None] + safe_mean[None]

        z_with_list.append(z_with.cpu().numpy())
        z_zero_list.append(z_zero.cpu().numpy())
        y_list.append(y)
        n_collected += B
        if n_collected >= n_samples:
            break

    z_with = np.concatenate(z_with_list)[:n_samples]
    z_zero = np.concatenate(z_zero_list)[:n_samples]
    y      = np.concatenate(y_list)[:n_samples]

    # Check latents for NaN (indicates diverged model weights).
    n_nan_z = int(np.isnan(z_with).any(axis=1).sum())
    if n_nan_z:
        raise RuntimeError(
            f'{n_nan_z}/{len(z_with)} latent vectors contain NaN — '
            f'the checkpoint likely has diverged weights. Check the training log.')

    # Filter out samples whose state labels contain NaN (bad dataset entries
    # or a corrupted state_mean/state_std in the checkpoint).
    y_nan_mask = np.isnan(y).any(axis=1)
    n_nan_y = int(y_nan_mask.sum())
    if n_nan_y:
        nan_dims = [i for i in range(y.shape[1]) if np.isnan(y[:, i]).any()]
        print(f'  [warning] {n_nan_y}/{len(y)} state labels contain NaN '
              f'(dims {nan_dims}) — filtering them out')
        keep   = ~y_nan_mask
        z_with = z_with[keep]
        z_zero = z_zero[keep]
        y      = y[keep]
        if len(y) == 0:
            raise RuntimeError('No valid samples remain after filtering NaN state labels. '
                               'Check state_mean/state_std in the checkpoint.')

    print(f'  collected {len(y)} samples')
    return z_with, z_zero, y


# ── probe fit + R² ────────────────────────────────────────────────────────────

def probe_r2(z, y, ridge=1e-3):
    """Fit RidgeStateProbe and return per-dim and overall R²."""
    probe = RidgeStateProbe().fit(z, y, ridge)
    y_hat = probe(z)
    per_dim = [float(r2_score(y[:, i], y_hat[:, i])) for i in range(y.shape[1])]
    overall = float(r2_score(y, y_hat))
    return per_dim, overall


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_results(results, labels, output_path):
    """
    results : list of dicts, one per model, each with keys
              'with' and 'zero', each being (per_dim_r2, overall_r2).
    """
    n_models  = len(results)
    n_dims    = len(STATE_LABELS)
    cols_per  = n_dims + 1          # per-dim + overall
    bar_w     = 0.35
    x         = np.arange(n_models)

    # colour scheme
    clr_with = '#4CC9F0'
    clr_zero = '#F72585'

    fig, axes = plt.subplots(1, cols_per,
                             figsize=(2.8 * cols_per, 4.4),
                             sharey=False)
    fig.subplots_adjust(wspace=0.35, left=0.06, right=0.98,
                        top=0.88, bottom=0.22)

    col_titles = STATE_LABELS + ['overall']
    for col, (ax, title) in enumerate(zip(axes, col_titles)):
        is_overall = col == n_dims
        idx = None if is_overall else col

        r2_with = np.array([r['with'][0][idx] if not is_overall else r['with'][1]
                             for r in results])
        r2_zero = np.array([r['zero'][0][idx] if not is_overall else r['zero'][1]
                             for r in results])

        bars_w = ax.bar(x - bar_w / 2, r2_with, bar_w,
                        color=clr_with, label='with proprio',
                        edgecolor='white', linewidth=0.4)
        bars_z = ax.bar(x + bar_w / 2, r2_zero, bar_w,
                        color=clr_zero, label='zero proprio',
                        edgecolor='white', linewidth=0.4)

        ax.set_title(f'R²  [{title}]', fontsize=10, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0, color='#888888', linewidth=0.5, linestyle='--')
        ax.axhline(1, color='#888888', linewidth=0.5, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if col == 0:
            ax.set_ylabel('R²', fontsize=9)
        if col == 0:
            ax.legend(fontsize=8, framealpha=0.6)

    fig.suptitle('CartPole encoder: probe R²  with vs. without proprioception',
                 fontsize=11, y=0.97)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'[done] → {output_path}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Probe CartPole latents with and without proprioception.')
    p.add_argument('--ckpts',    nargs='+', required=True,
                   help='One or more checkpoint paths.')
    p.add_argument('--cfgs',     nargs='+', required=True,
                   help='Config YAML for each checkpoint (same order).')
    p.add_argument('--labels',   nargs='+', default=None,
                   help='Display names (defaults to checkpoint filename stems).')
    p.add_argument('--data',     required=True,
                   help='CartPole HDF5 dataset directory.')
    p.add_argument('--n-samples', type=int, default=4000,
                   help='Samples per model for probe fitting.')
    p.add_argument('--ridge',    type=float, default=1e-3)
    p.add_argument('--split',    default='train')
    p.add_argument('--device',   default='cuda')
    p.add_argument('--output',   required=True)
    args = p.parse_args()

    if len(args.ckpts) != len(args.cfgs):
        p.error('--ckpts and --cfgs must have the same number of entries')
    labels = args.labels or [Path(c).stem for c in args.ckpts]
    if len(labels) != len(args.ckpts):
        p.error('--labels length must match --ckpts')

    all_results = []
    for ckpt, cfg, label in zip(args.ckpts, args.cfgs, labels):
        print(f'\n[probe] {label}')
        bundle = load_bundle(ckpt, cfg, args.device)
        z_with, z_zero, y = collect_latents(
            bundle, args.data, n_samples=args.n_samples, split=args.split)

        r2_with = probe_r2(z_with, y, args.ridge)
        r2_zero = probe_r2(z_zero, y, args.ridge)

        print(f'  {"dim":<6} {"with-proprio":>14} {"zero-proprio":>14}  {"delta":>8}')
        for i, dim in enumerate(STATE_LABELS):
            w, z = r2_with[0][i], r2_zero[0][i]
            print(f'  {dim:<6} {w:14.3f} {z:14.3f}  {w-z:+8.3f}')
        print(f'  {"overall":<6} {r2_with[1]:14.3f} {r2_zero[1]:14.3f}  '
              f'{r2_with[1]-r2_zero[1]:+8.3f}')

        all_results.append({'with': r2_with, 'zero': r2_zero})

    plot_results(all_results, labels, args.output)


if __name__ == '__main__':
    main()
