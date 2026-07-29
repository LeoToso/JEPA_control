"""Sweep diagnose_instability_rollout over all checkpoints in a directory.

Produces:
  - A summary figure: Learned/GT growth-rate ratio vs epoch
  - A CSV with per-epoch metrics
  - Individual rollout plots (optional, via --individual-dir)

Usage:
    python experiments/diagnose_instability_rollout_sweep.py \\
        --ckpt-dir results/jepa_sf_w3_fs5_v9_phase2/checkpoints \\
        --config   configs/cartpole_jepa_sf_w3_fs5_v9_phase2.yaml \\
        --data     data/cartpole_visual_fs5_passive_long \\
        --state-head-ckpt results/jepa_sf_w3_fs5_v9/checkpoints/checkpoint_epoch0100.pt \\
        --out      results/jepa_sf_w3_fs5_v9_phase2/instability_sweep.png \\
        --csv      results/jepa_sf_w3_fs5_v9_phase2/instability_sweep.csv \\
        --individual-dir results/jepa_sf_w3_fs5_v9_phase2/instability_rollouts \\
        --device   cuda:2
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.diagnose_instability_rollout import (
    _load_norm_stats, _load_model, _encode, _predict, _decode, _fit_growth_rate,
)


def _epoch_from_path(p: Path) -> int:
    m = re.search(r'epoch(\d+)', p.stem)
    return int(m.group(1)) if m else -1


def _run_one(model, state_head, env, A_step, rho_gt, frame_skip, dt,
             eps, n_steps, fit_steps, state_mean, state_std, device,
             frame_stack, W, out_path=None):
    """Run instability rollout for one checkpoint. Returns (rate_gt_nonlin, rate_lrn, ratio)."""
    from envs.cartpole_visual import ContinuousCartpoleVisual

    # GT linear
    x0 = np.array([0.0, 0.0, eps, 0.0], dtype=np.float32)
    gt_lin = [x0.copy()]
    x = x0.copy().astype(float)
    for _ in range(n_steps):
        x = A_step @ x
        gt_lin.append(x.copy())
    gt_lin = np.array(gt_lin)

    # GT nonlinear
    obs0, s0, _ = env.reset_to_state(x0)
    gt_nlin = [s0.copy()]
    for _ in range(n_steps):
        obs_next, s_next, _, done, _ = env.step(0.0)
        gt_nlin.append(s_next.copy())
        if done:
            break
    gt_nlin = np.array(gt_nlin)

    # Learned: warm-start W steps with real env, then pure model prediction
    obs_cur, s_cur, _ = env.reset_to_state(x0)
    real_history = []
    lrn_warmup_states = [s_cur.copy()]
    for _ in range(W):
        z_cur = _encode(model, obs_cur, device, frame_stack)
        real_history.append(z_cur)
        obs_next, s_next, _, done, _ = env.step(0.0)
        lrn_warmup_states.append(s_next.copy())
        obs_cur = obs_next
        if done:
            break

    lrn_warmup = [_decode(state_head, z, state_mean, state_std) for z in real_history]
    history = list(real_history[-W:])
    lrn = lrn_warmup[:]
    for _ in range(n_steps):
        z_next = _predict(model, history, device)
        lrn.append(_decode(state_head, z_next, state_mean, state_std))
        history = history[1:] + [z_next]
    lrn = np.array(lrn)
    warmup_end = len(lrn_warmup)

    rate_gl  = _fit_growth_rate(gt_lin[:, 2],  fit_steps)
    rate_gn  = _fit_growth_rate(gt_nlin[:, 2], fit_steps)
    lrn_slice = lrn[warmup_end:warmup_end + fit_steps + 1, 2]
    rate_lrn = _fit_growth_rate(lrn_slice, fit_steps)
    ratio    = rate_lrn / rho_gt if rho_gt > 0 else float('nan')

    if out_path is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        t_gl  = np.arange(len(gt_lin))
        t_gn  = np.arange(len(gt_nlin))
        t_lrn = np.arange(len(lrn))
        ax.semilogy(t_gl,  np.abs(gt_lin[:, 2]),  'g-',  lw=2,   label=f'GT linear (λ={rate_gl:.3f})')
        ax.semilogy(t_gn,  np.abs(gt_nlin[:, 2]), 'b--', lw=2,   label=f'GT nonlinear (λ={rate_gn:.3f})')
        ax.semilogy(t_lrn[:warmup_end], np.abs(lrn[:warmup_end, 2]),
                    'r--', lw=1.2, alpha=0.5, label='Learned warm-start')
        ax.semilogy(t_lrn[warmup_end-1:], np.abs(lrn[warmup_end-1:, 2]),
                    'r-o', lw=1.5, ms=4, label=f'Learned pred (λ={rate_lrn:.3f})')
        ax.axvline(warmup_end - 1, color='orange', lw=0.8, linestyle=':')
        ax.set_xlabel(f'Latent step  (×{frame_skip} physics steps = ×{frame_skip * dt:.3f}s)')
        ax.set_ylabel('|θ| (rad)')
        ax.set_title(f'θ₀={np.degrees(eps):.1f}°  Learned/GT={ratio:.3f} ({ratio*100:.1f}%)')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which='both')
        fig.suptitle(Path(out_path).stem, fontsize=10)
        fig.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
        plt.close(fig)

    return rate_gn, rate_lrn, ratio


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt-dir',       required=True)
    p.add_argument('--config',         required=True)
    p.add_argument('--data',           required=True)
    p.add_argument('--state-head-ckpt', default=None,
                   help='Checkpoint to load state_head from (for phase-2 runs)')
    p.add_argument('--eps',            type=float, default=0.05)
    p.add_argument('--n-steps',        type=int,   default=20)
    p.add_argument('--fit-steps',      type=int,   default=5)
    p.add_argument('--out',            default=None, help='Summary figure path')
    p.add_argument('--csv',            default=None)
    p.add_argument('--individual-dir', default=None,
                   help='Directory to save per-checkpoint rollout figures')
    p.add_argument('--device',         default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg['environment']

    state_mean, state_std = _load_norm_stats(args.data)

    # Find and sort checkpoints
    ckpt_dir = Path(args.ckpt_dir)
    ckpts = sorted(ckpt_dir.glob('checkpoint_epoch*.pt'), key=_epoch_from_path)
    if not ckpts:
        raise FileNotFoundError(f'No checkpoints found in {ckpt_dir}')
    print(f'Found {len(ckpts)} checkpoints  (ep{_epoch_from_path(ckpts[0])}–ep{_epoch_from_path(ckpts[-1])})')

    # Load state head if provided
    sh_state = None
    if args.state_head_ckpt is not None:
        sh_raw = torch.load(args.state_head_ckpt, map_location=device, weights_only=False)
        if isinstance(sh_raw, dict) and 'state_head_state' in sh_raw:
            sh_state = sh_raw['state_head_state']
            print(f'[state_head] loaded from {args.state_head_ckpt}')

    # Build env and GT once
    from envs.cartpole_visual import ContinuousCartpoleVisual
    from ground_truth.cartpole_gt import CartpoleGroundTruth

    frame_skip = int(env_cfg.get('frame_skip', 1))
    env = ContinuousCartpoleVisual(
        frame_skip=frame_skip,
        image_size=int(env_cfg['image_size']),
        mass_cart=env_cfg['mass_cart'], mass_pole=env_cfg['mass_pole'],
        pole_length=env_cfg['pole_length'], gravity=env_cfg['gravity'],
        dt=env_cfg['dt'], seed=0,
    )
    gt = CartpoleGroundTruth(
        mass_cart=env_cfg['mass_cart'], mass_pole=env_cfg['mass_pole'],
        pole_length=env_cfg['pole_length'], gravity=env_cfg['gravity'],
        dt=env_cfg['dt'],
    )
    A_step = np.linalg.matrix_power(gt.A_star, frame_skip)
    rho_gt = float(np.max(np.abs(np.linalg.eigvals(A_step))))
    print(f'[GT] rho(A_star^{frame_skip}) = {rho_gt:.4f}\n')

    rows = []
    print(f'{"epoch":>7}  {"GT_rate":>8}  {"Lrn_rate":>9}  {"Lrn/GT":>7}  {"Lrn/GT %":>9}')
    print('-' * 52)

    for ckpt_path in ckpts:
        epoch = _epoch_from_path(ckpt_path)
        model, state_head, frame_stack, W = _load_model(str(ckpt_path), cfg, device)

        if state_head is None and sh_state is not None:
            d_lat = int(cfg['model']['latent_dim'])
            state_head = nn.Linear(d_lat, 4).to(device)
            state_head.load_state_dict(sh_state)
            state_head.eval()

        if state_head is None:
            print(f'  ep{epoch:04d}: skipped (no state head)')
            continue

        out_path = None
        if args.individual_dir is not None:
            out_path = Path(args.individual_dir) / f'instability_rollout_ep{epoch:04d}.png'

        rate_gn, rate_lrn, ratio = _run_one(
            model, state_head, env, A_step, rho_gt,
            frame_skip, float(env_cfg['dt']),
            args.eps, args.n_steps, args.fit_steps,
            state_mean, state_std, device, frame_stack, W,
            out_path=out_path,
        )
        rows.append((epoch, rate_gn, rate_lrn, ratio))
        print(f'  ep{epoch:04d}:  {rate_gn:8.4f}   {rate_lrn:9.4f}   {ratio:7.3f}   {ratio*100:8.1f}%')

    env.close()

    if not rows:
        print('No results to plot.')
        return

    epochs   = np.array([r[0] for r in rows])
    gt_rates = np.array([r[1] for r in rows])
    lrn_rates = np.array([r[2] for r in rows])
    ratios   = np.array([r[3] for r in rows])

    if args.csv:
        import csv
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['epoch', 'gt_rate', 'learned_rate', 'ratio', 'ratio_pct'])
            for r in rows:
                w.writerow([r[0], f'{r[1]:.4f}', f'{r[2]:.4f}', f'{r[3]:.4f}', f'{r[3]*100:.1f}'])
        print(f'\nCSV → {args.csv}')

    if args.out:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Top: raw growth rates
        ax = axes[0]
        ax.axhline(rho_gt, color='black', lw=1.5, linestyle='--', label=f'GT (ρ={rho_gt:.4f})')
        ax.plot(epochs, gt_rates,  'b-o', ms=4, lw=1.5, label='GT nonlinear rate')
        ax.plot(epochs, lrn_rates, 'r-o', ms=4, lw=1.5, label='Learned rate')
        ax.set_ylabel('Growth rate λ (per latent step)')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_title('Instability growth rates across phase-2 epochs')

        # Bottom: ratio
        ax = axes[1]
        ax.axhline(1.0, color='black', lw=1.5, linestyle='--', label='Perfect (ratio=1)')
        ax.plot(epochs, ratios, 'g-o', ms=4, lw=1.5, label='Learned/GT ratio')
        ax.fill_between(epochs, ratios, 1.0, alpha=0.15, color='green')
        ax.set_ylabel('Learned / GT growth rate')
        ax.set_xlabel('Phase-2 epoch')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(max(0, ratios.min() - 0.05), ratios.max() + 0.05)

        fig.tight_layout()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(args.out), dpi=150, bbox_inches='tight')
        print(f'Plot → {args.out}')


if __name__ == '__main__':
    main()
