"""Evaluate all checkpoints in a directory and plot metric curves over epochs.

Computes per-checkpoint:
  - rho(A_aug)          spectral radius of augmented Jacobian at equilibrium
  - ||B_eff||           effective control input norm
  - max|r| per state    Pearson correlation between latent dims and x, xdot, theta, thetadot

Usage:
  python experiments/eval_checkpoint_sweep.py \\
      --ckpt-dir results/jepa_v2_mixed2/checkpoints \\
      --config   configs/cartpole_jepa_pred_state_random.yaml \\
      --out      results/jepa_v2_mixed2/checkpoint_sweep.png \\
      --csv      results/jepa_v2_mixed2/checkpoint_sweep.csv \\
      --device   cuda:0
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _epoch_from_name(p: Path) -> int:
    """Extract epoch number from checkpoint filename."""
    stem = p.stem  # e.g. checkpoint_epoch0380
    for part in stem.split('_'):
        if part.startswith('epoch'):
            return int(part[5:])
    return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', required=True,
                        help='Directory containing checkpoint_epochNNNN.pt files')
    parser.add_argument('--config',   required=True,
                        help='Config yaml for model architecture')
    parser.add_argument('--out',      default=None,
                        help='Output plot path (default: <ckpt-dir>/sweep.png)')
    parser.add_argument('--csv',      default=None,
                        help='Output CSV path (default: <ckpt-dir>/sweep.csv)')
    parser.add_argument('--n-rollouts',  type=int, default=40)
    parser.add_argument('--rollout-len', type=int, default=50)
    parser.add_argument('--gramian-T',   type=int, default=20)
    parser.add_argument('--device',      default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--step',        type=int, default=1,
                        help='Evaluate every N-th checkpoint (1 = all)')
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    out_path  = Path(args.out)  if args.out  else ckpt_dir.parent / 'checkpoint_sweep.png'
    csv_path  = Path(args.csv)  if args.csv  else ckpt_dir.parent / 'checkpoint_sweep.csv'
    device    = torch.device(args.device)

    # ── Discover checkpoints ─────────────────────────────────────────────────
    ckpts = sorted(
        [p for p in ckpt_dir.glob('checkpoint_epoch*.pt')],
        key=_epoch_from_name)
    ckpts = ckpts[::args.step]
    if not ckpts:
        raise FileNotFoundError(f'No checkpoint_epoch*.pt found in {ckpt_dir}')
    print(f'Found {len(ckpts)} checkpoints  ({_epoch_from_name(ckpts[0])}–{_epoch_from_name(ckpts[-1])})')

    # ── GT spectral radius ───────────────────────────────────────────────────
    try:
        from ground_truth.cartpole_gt import CartpoleGroundTruth
        rho_gt = float(np.max(np.abs(np.linalg.eigvals(CartpoleGroundTruth().A_star))))
        print(f'GT rho(A_star) = {rho_gt:.4f}')
    except Exception:
        rho_gt = None

    # ── Shared environment ───────────────────────────────────────────────────
    from envs.cartpole_visual import ContinuousCartpoleVisual
    env = ContinuousCartpoleVisual(image_size=64, action_range=(-10, 10))
    obs_eq, _, _ = env.reset_to_state(np.zeros(4, dtype=np.float32))

    # Import helpers from compare_latent_dynamics
    from experiments.compare_latent_dynamics import (
        load_model, collect_rollouts, compute_pearson_matrix,
        compute_gramian_eigenvalues, _to_tensor,
    )

    # ── Sweep ────────────────────────────────────────────────────────────────
    rows = []
    for ckpt_path in ckpts:
        epoch = _epoch_from_name(ckpt_path)
        print(f'  ep{epoch:04d}  ', end='', flush=True)

        try:
            model, frame_stack = load_model(str(ckpt_path), args.config, device)
        except Exception as e:
            print(f'load failed: {e}')
            continue

        zs, _, states, _ = collect_rollouts(
            env, model, frame_stack, device,
            n_rollouts=args.n_rollouts, rollout_len=args.rollout_len)

        R = compute_pearson_matrix(zs, states)
        max_r = np.max(np.abs(R), axis=1)  # (4,)

        obs_eq_t = _to_tensor(obs_eq, device)
        if frame_stack > 1:
            obs_eq_t = torch.cat([obs_eq_t, obs_eq_t], dim=1)
        with torch.no_grad():
            z_star = model.encoder(obs_eq_t).cpu().numpy()[0]

        try:
            eigvals, A_aug, B_eff = compute_gramian_eigenvalues(
                model, z_star, device, T=args.gramian_T)
            rho  = float(np.max(np.abs(np.linalg.eigvals(A_aug))))
            b_norm = float(np.linalg.norm(B_eff[:len(z_star)]))
        except Exception:
            rho, b_norm = float('nan'), float('nan')

        print(f'rho={rho:.4f}  ||B||={b_norm:.4f}  '
              f'r_x={max_r[0]:.3f}  r_xd={max_r[1]:.3f}  '
              f'r_th={max_r[2]:.3f}  r_thd={max_r[3]:.3f}')

        rows.append({
            'epoch': epoch,
            'rho': rho,
            'b_eff_norm': b_norm,
            'r_x': max_r[0],
            'r_xdot': max_r[1],
            'r_theta': max_r[2],
            'r_thetadot': max_r[3],
        })

    env.close()

    if not rows:
        print('No results — exiting.')
        return

    # ── Save CSV ─────────────────────────────────────────────────────────────
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f'\nCSV saved → {csv_path}')

    # ── Plot ─────────────────────────────────────────────────────────────────
    epochs  = [r['epoch']      for r in rows]
    rhos    = [r['rho']        for r in rows]
    bnorms  = [r['b_eff_norm'] for r in rows]
    r_x     = [r['r_x']       for r in rows]
    r_xd    = [r['r_xdot']    for r in rows]
    r_th    = [r['r_theta']    for r in rows]
    r_thd   = [r['r_thetadot'] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: rho
    axes[0].plot(epochs, rhos, 'o-', color='steelblue', ms=4, lw=1.5, label='ρ(A_aug)')
    axes[0].axhline(1.0, color='red', linestyle='--', lw=1, label='ρ=1 threshold')
    if rho_gt is not None:
        axes[0].axhline(rho_gt, color='green', linestyle=':', lw=1, label=f'GT ρ={rho_gt:.3f}')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('ρ(A_aug)')
    axes[0].set_title('Spectral Radius at Equilibrium')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: ||B_eff||
    axes[1].plot(epochs, bnorms, 'o-', color='darkorange', ms=4, lw=1.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('||B_eff||')
    axes[1].set_title('Effective Control Input Norm')
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Pearson correlations
    axes[2].plot(epochs, r_th,  'o-', ms=4, lw=1.5, label=r'$\theta$')
    axes[2].plot(epochs, r_thd, 's-', ms=4, lw=1.5, label=r'$\dot\theta$')
    axes[2].plot(epochs, r_x,   '^-', ms=4, lw=1.5, label=r'$x$')
    axes[2].plot(epochs, r_xd,  'v-', ms=4, lw=1.5, label=r'$\dot{x}$')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('max|r|')
    axes[2].set_title('Pearson Correlation (latent vs state)')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f'Checkpoint Sweep — {ckpt_dir.parent.name}', fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f'Plot saved → {out_path}')


if __name__ == '__main__':
    main()
