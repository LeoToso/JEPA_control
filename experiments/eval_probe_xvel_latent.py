#!/usr/bin/env python
"""Diagnostic: does the MLP probe accurately decode x_vel (obs[8]) from
*predicted* latent states?

The concern is that during iCEM planning the probe sees z_{t+1} = f(z_t, a)
from the latent dynamics model — NOT z encoded directly from a real frame.
If the probe is only accurate for encoded z values, cost estimates in the
planner will be wrong.

This script tests probe accuracy in the actual planning setting:
  1. Encode real observation z_t from HDF5 val data
  2. Apply latent dynamics with the REAL action a_t → z_{t+1}^pred
  3. Decode x_vel from z_{t+1}^pred  and compare with ground-truth x_vel

Additionally generates a scatter plot of pred_xvel vs true_xvel and reports
per-step correlations to detect systematic bias (e.g., sign flip).

Usage
-----
python experiments/eval_probe_xvel_latent.py \\
    --ckpt  /path/to/model_final.pt \\
    --cfg   configs/walker2d_smwm_fwd_endpoint_inverse_act1.yaml \\
    --hdf5-dir data/walker2d_mixed_sac_fs5_64 \\
    --probe-path results/probes/fwd_ep_ar_mlp_probe.pt \\
    --out   results/probes/xvel_latent_eval.pdf \\
    --val-episodes 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent))
sys.path.insert(0, str(_here))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import h5py

from walker2d_smwm_utils import (
    MLPStateProbe, load_walker_bundle, latent_step_batch,
)
from sensorimotor_probe_utils import encode_obs


OBS_DIM = 17
XVEL_IDX = 8


def collect_predicted_xvel(bundle, probe_net, hdf5_dir, split, max_episodes,
                            image_size=64, max_steps_per_ep=None):
    """For each (z_t, a_t) pair in the dataset:
       - encode obs_t → z_t
       - apply latent_step(z_t, a_t) → z_{t+1}_pred
       - decode xvel from z_{t+1}_pred
       - record true xvel = states[t+1, XVEL_IDX]

    Returns:
       xvel_pred : (N,) float32
       xvel_true : (N,) float32
       xvel_enc  : (N,) float32  — probe on ENCODED z_{t+1}, upper-bound accuracy
    """
    hdf5_path = Path(hdf5_dir) / f'{split}.hdf5'
    device = bundle['device']

    xvel_pred_list, xvel_true_list, xvel_enc_list = [], [], []

    with torch.no_grad():
        with h5py.File(hdf5_path, 'r') as f:
            ep_grp  = f['episodes']
            ep_keys = sorted(ep_grp.keys(), key=lambda k: int(k))[:max_episodes]
            print(f'[xvel-eval]  {split}: evaluating {len(ep_keys)} episodes …')

            for ep_key in ep_keys:
                ep      = ep_grp[ep_key]
                obs_t   = ep['observations'][:]   # (T+1, H, W, 3)
                states  = ep['states'][:]          # (T+1, 17)
                actions = ep['actions'][:]         # (T, 6)
                T       = obs_t.shape[0] - 1
                if T < 2:
                    continue

                # Resize to model image_size if needed
                if obs_t.shape[1] != image_size or obs_t.shape[2] != image_size:
                    import torch.nn.functional as F_
                    t_ = torch.from_numpy(obs_t).permute(0, 3, 1, 2).float()
                    t_ = F_.interpolate(t_, (image_size, image_size),
                                        mode='bilinear', align_corners=False)
                    obs_t = t_.permute(0, 2, 3, 1).byte().numpy()

                max_t = T if max_steps_per_ep is None else min(T, max_steps_per_ep)
                for t in range(1, max_t + 1):
                    # Encode obs_t → z_t  (uses previous frame for frame-diff models)
                    z_t = encode_obs(bundle, obs_t[t], obs_t[t - 1], states[t])
                    # z_t : (1, D) on device

                    a_t = torch.as_tensor(
                        actions[t - 1].astype(np.float32),
                        device=device).unsqueeze(0)                   # (1, 6)

                    # ── predicted z_{t+1} via dynamics model ──────────────────
                    z_pred = latent_step_batch(bundle, z_t, a_t)      # (1, D)
                    xvel_p = probe_net(z_pred)[0, XVEL_IDX].item()
                    xvel_pred_list.append(xvel_p)

                    # ── encoded z_{t+1} → upper-bound accuracy ────────────────
                    if t < T:
                        z_enc = encode_obs(bundle, obs_t[t + 1], obs_t[t], states[t + 1])
                        xvel_e = probe_net(z_enc)[0, XVEL_IDX].item()
                    else:
                        xvel_e = float('nan')
                    xvel_enc_list.append(xvel_e)

                    # ── ground-truth x_vel ────────────────────────────────────
                    xvel_true_list.append(float(states[t + 1 if t + 1 <= T else T, XVEL_IDX]))

    xvel_pred = np.array(xvel_pred_list, dtype=np.float32)
    xvel_true = np.array(xvel_true_list, dtype=np.float32)
    xvel_enc  = np.array(xvel_enc_list,  dtype=np.float32)
    print(f'[xvel-eval]  collected N={len(xvel_pred)} (t+1) samples')
    return xvel_pred, xvel_true, xvel_enc


def r2_1d(y_hat, y):
    mask = np.isfinite(y_hat) & np.isfinite(y)
    if mask.sum() < 2:
        return float('nan')
    y_hat, y = y_hat[mask], y[mask]
    ss_res = ((y_hat - y) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def pearson(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float('nan')
    a, b = a[mask], b[mask]
    return float(np.corrcoef(a, b)[0, 1])


def print_report(xvel_pred, xvel_true, xvel_enc):
    print('\n' + '=' * 60)
    print('  x_vel (obs[8]) probe accuracy — predicted vs. encoded z')
    print('=' * 60)

    r2_p  = r2_1d(xvel_pred, xvel_true)
    r2_e  = r2_1d(xvel_enc,  xvel_true)
    cor_p = pearson(xvel_pred, xvel_true)
    cor_e = pearson(xvel_enc,  xvel_true)

    mae_p = float(np.nanmean(np.abs(xvel_pred - xvel_true)))
    mae_e = float(np.nanmean(np.abs(xvel_enc  - xvel_true)))

    # Sign-accuracy: fraction of cases where sign(pred) == sign(true)
    sa_p = float(np.mean(np.sign(xvel_pred) == np.sign(xvel_true)))
    sa_e = float(np.nanmean(np.sign(xvel_enc) == np.sign(xvel_true)))

    print(f'  Setting                R²        r (Pearson)   MAE    sign-acc')
    print(f'  {"Probe on predicted z":<22}  {r2_p:+.4f}   {cor_p:+.4f}       {mae_p:.4f}  {sa_p:.3f}')
    print(f'  {"Probe on encoded z":<22}  {r2_e:+.4f}   {cor_e:+.4f}       {mae_e:.4f}  {sa_e:.3f}')
    print()
    print(f'  true x_vel: mean={xvel_true.mean():.3f}  std={xvel_true.std():.3f}  '
          f'range=[{xvel_true.min():.2f}, {xvel_true.max():.2f}]')
    print(f'  pred x_vel: mean={xvel_pred.mean():.3f}  std={xvel_pred.std():.3f}  '
          f'range=[{xvel_pred.min():.2f}, {xvel_pred.max():.2f}]')
    print()

    if r2_p < 0.3:
        print('  !! LOW R² on predicted z — probe poorly decodes x_vel from '
              'dynamics-model outputs.')
        print('     iCEM cost estimates are unreliable; this explains backward walking.')
    elif r2_p < 0.7:
        print('  !  MODERATE R² on predicted z — some signal but likely noisy cost.')
    else:
        print('     Good R² on predicted z — probe is accurate in planning setting.')

    if cor_p < 0:
        print('  !! NEGATIVE CORRELATION — probe predicts x_vel with WRONG SIGN.')
        print('     iCEM will maximise backward walking (high predicted x_vel = '
              'actually large negative x_vel).')
    elif abs(cor_p) < 0.3:
        print('  !! Near-zero correlation — probe x_vel is essentially random '
              'relative to true x_vel.')

    print('=' * 60 + '\n')


def make_figure(xvel_pred, xvel_true, xvel_enc, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panel 1: predicted z ──────────────────────────────────────────────────
    ax = axes[0]
    lim = max(abs(xvel_true).max(), abs(xvel_pred).max()) * 1.1
    ax.hexbin(xvel_true, xvel_pred, gridsize=60, cmap='Blues', mincnt=1)
    ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=1, label='ideal')
    ax.set_xlabel('true x_vel', fontsize=12)
    ax.set_ylabel('probe(z_pred) x_vel', fontsize=12)
    r2_p  = r2_1d(xvel_pred, xvel_true)
    cor_p = pearson(xvel_pred, xvel_true)
    sa_p  = float(np.mean(np.sign(xvel_pred) == np.sign(xvel_true)))
    ax.set_title(f'Probe on predicted z\n'
                 f'R²={r2_p:.3f}  r={cor_p:.3f}  sign-acc={sa_p:.2f}', fontsize=11)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color='gray', linewidth=0.6, alpha=0.4)
    ax.axvline(0, color='gray', linewidth=0.6, alpha=0.4)
    ax.legend(fontsize=10)
    ax.grid(True, color='#eeeeee')
    ax.spines[['top', 'right']].set_visible(False)

    # ── Panel 2: encoded z (upper bound) ──────────────────────────────────────
    ax = axes[1]
    mask = np.isfinite(xvel_enc)
    xe, xt = xvel_enc[mask], xvel_true[mask]
    lim2 = max(abs(xt).max(), abs(xe).max()) * 1.1
    ax.hexbin(xt, xe, gridsize=60, cmap='Oranges', mincnt=1)
    ax.plot([-lim2, lim2], [-lim2, lim2], 'k--', linewidth=1, label='ideal')
    ax.set_xlabel('true x_vel', fontsize=12)
    ax.set_ylabel('probe(z_enc) x_vel', fontsize=12)
    r2_e  = r2_1d(xvel_enc, xvel_true)
    cor_e = pearson(xvel_enc, xvel_true)
    sa_e  = float(np.nanmean(np.sign(xvel_enc) == np.sign(xvel_true)))
    ax.set_title(f'Probe on encoded z (upper bound)\n'
                 f'R²={r2_e:.3f}  r={cor_e:.3f}  sign-acc={sa_e:.2f}', fontsize=11)
    ax.set_xlim(-lim2, lim2)
    ax.set_ylim(-lim2, lim2)
    ax.axhline(0, color='gray', linewidth=0.6, alpha=0.4)
    ax.axvline(0, color='gray', linewidth=0.6, alpha=0.4)
    ax.legend(fontsize=10)
    ax.grid(True, color='#eeeeee')
    ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle('Walker2d probe: x_vel (obs[8]) accuracy\n'
                 'left = planning setting (dynamics model output)  |  '
                 'right = upper bound (directly encoded frame)',
                 fontsize=11)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    print(f'[done]  {out_path}')
    plt.close(fig)


def collect_env_xvel(bundle, probe_net, n_episodes, steps_per_ep, image_size=64,
                     seed=0, action_policy='random'):
    """Collect x_vel accuracy by running the Walker2d-v4 env directly.

    No HDF5 dataset needed.  At each step:
      encode frame_t → z_t
      z_{t+1}^pred = latent_step(z_t, a_t)
      probe(z_{t+1}^pred)[8]  vs  real x_vel at t+1

    action_policy: 'random' (uniform [-1,1]) or 'forward' (hip bias)
    """
    import gymnasium
    from envs.walker2d_visual import Walker2dVisual

    device = bundle['device']
    rng    = np.random.default_rng(seed)

    # Forward-bias indices (same as generate_walker2d_dataset.py)
    FWD_IDX = [0, 3]
    FWD_BIAS = 0.4

    xvel_pred_list, xvel_true_list, xvel_enc_list = [], [], []

    env = Walker2dVisual(image_size=image_size, seed=seed)

    for ep in range(n_episodes):
        frame, state, _ = env.reset()
        prev_frame = frame.copy()

        for t in range(steps_per_ep):
            # Sample action
            a = rng.uniform(-1.0, 1.0, size=6).astype(np.float32)
            if action_policy == 'forward':
                a[FWD_IDX] = np.clip(a[FWD_IDX] + FWD_BIAS, -1.0, 1.0)

            # Encode current frame → z_t
            with torch.no_grad():
                z_t = encode_obs(bundle, frame, prev_frame, state)  # (1, D)

                # Predict z_{t+1} from latent dynamics
                a_t = torch.as_tensor(a, device=device).unsqueeze(0)  # (1, 6)
                z_pred = latent_step_batch(bundle, z_t, a_t)           # (1, D)
                xvel_p = probe_net(z_pred)[0, XVEL_IDX].item()

            # Step real env
            next_frame, next_state, _, done, _ = env.step(a)
            xvel_true = float(next_state[XVEL_IDX])

            # Encode next frame → z_{t+1} (upper-bound accuracy)
            with torch.no_grad():
                z_enc  = encode_obs(bundle, next_frame, frame, next_state)
                xvel_e = probe_net(z_enc)[0, XVEL_IDX].item()

            xvel_pred_list.append(xvel_p)
            xvel_true_list.append(xvel_true)
            xvel_enc_list.append(xvel_e)

            prev_frame = frame
            frame      = next_frame
            state      = next_state

            if done:
                break

        if (ep + 1) % 5 == 0:
            print(f'[xvel-eval]  env ep {ep+1}/{n_episodes}  '
                  f'N={len(xvel_pred_list)} samples so far')

    env.close()
    print(f'[xvel-eval]  collected N={len(xvel_pred_list)} (env) samples')
    return (np.array(xvel_pred_list, dtype=np.float32),
            np.array(xvel_true_list, dtype=np.float32),
            np.array(xvel_enc_list,  dtype=np.float32))


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--ckpt',            required=True)
    p.add_argument('--cfg',             default=None)
    p.add_argument('--probe-path',      required=True,
                   help='Saved .pt MLP probe file')
    p.add_argument('--device',          default='cuda')
    p.add_argument('--out',             default='results/probes/xvel_latent_eval.pdf')

    # ── HDF5 mode (original) ──────────────────────────────────────────────────
    p.add_argument('--hdf5-dir',        default=None,
                   help='HDF5 dataset dir. If omitted, --from-env is used.')
    p.add_argument('--val-episodes',    type=int, default=50)
    p.add_argument('--max-steps-per-ep', type=int, default=None)

    # ── Env mode (no dataset needed) ─────────────────────────────────────────
    p.add_argument('--from-env',        action='store_true',
                   help='Run Walker2d-v4 env directly; no HDF5 needed.')
    p.add_argument('--env-episodes',    type=int, default=20,
                   help='Number of env episodes for --from-env')
    p.add_argument('--env-steps',       type=int, default=200,
                   help='Max steps per episode for --from-env')
    p.add_argument('--env-policy',      default='random',
                   choices=['random', 'forward'],
                   help='Action policy for --from-env')
    p.add_argument('--image-size',      type=int, default=64)
    p.add_argument('--seed',            type=int, default=0)
    args = p.parse_args()

    # Default to --from-env if no hdf5-dir given
    if args.hdf5_dir is None:
        args.from_env = True

    bundle = load_walker_bundle(args.ckpt, args.cfg, args.device)
    z_dim  = int(bundle['model_cfg'].get('latent_dim', 192))

    # ── Load probe ────────────────────────────────────────────────────────────
    print(f'[xvel-eval]  loading probe from {args.probe_path}')
    ck    = torch.load(args.probe_path, map_location='cpu', weights_only=False)
    probe = MLPStateProbe(z_dim)
    probe._net.load_state_dict(ck['state_dict'] if 'state_dict' in ck else ck)
    probe_net = probe._net.to(bundle['device']).eval()

    # ── Collect predictions ───────────────────────────────────────────────────
    if args.from_env:
        print(f'[xvel-eval]  running {args.env_episodes} env episodes '
              f'({args.env_steps} steps, policy={args.env_policy}) …')
        xvel_pred, xvel_true, xvel_enc = collect_env_xvel(
            bundle, probe_net,
            n_episodes=args.env_episodes,
            steps_per_ep=args.env_steps,
            image_size=args.image_size,
            seed=args.seed,
            action_policy=args.env_policy,
        )
    else:
        xvel_pred, xvel_true, xvel_enc = collect_predicted_xvel(
            bundle, probe_net, args.hdf5_dir, 'val',
            max_episodes=args.val_episodes,
            max_steps_per_ep=args.max_steps_per_ep,
        )

    print_report(xvel_pred, xvel_true, xvel_enc)
    make_figure(xvel_pred, xvel_true, xvel_enc, args.out)


if __name__ == '__main__':
    main()
