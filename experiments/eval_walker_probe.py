#!/usr/bin/env python
"""Evaluate Walker2d MLP/ridge probe quality per state dimension.

Loads (or fits) a probe and reports R² per obs dimension on both train
and val splits.  Highlights the 5 quantities the latent iCEM cost uses:
  obs[0]   z_height   (wz, healthy_z_min)
  obs[1]   torso_ang  (wang)
  obs[2:8] joints×6   (wjoint)
  obs[8]   x_vel      (wx, wback)

Usage
-----
# Evaluate a saved probe
python experiments/eval_walker_probe.py \
    --ckpt  /mnt/t7shield/jepa_results/walker2d_mixed_sac_smwm_fwd_endpoint_inverse_act1_seed42/model_final.pt \
    --cfg   configs/walker2d_smwm_fwd_endpoint_inverse_act1.yaml \
    --hdf5-dir data/walker2d_mixed_sac_fs5_64 \
    --probe-path results/probes/fwd_ep_ar_mlp_probe.pt \
    --out   results/probes/fwd_ep_ar_mlp_probe_eval.pdf

# Fit a fresh probe and evaluate
python experiments/eval_walker_probe.py \
    --ckpt  ... --cfg ... --hdf5-dir ... \
    --probe-episodes 300 --probe-epochs 50 \
    --out   results/probes/probe_eval.pdf
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
    MLPStateProbe, fit_walker_mlp_probe, load_walker_bundle,
)
from sensorimotor_probe_utils import encode_obs

# ── obs-dimension labels and iCEM usage ───────────────────────────────────────

OBS_LABELS = [
    'z_height',        # obs[0]  — wz, healthy_z_min
    'torso_angle',     # obs[1]  — wang
    'knee_L',          # obs[2]  — wjoint
    'ankle_L',         # obs[3]  — wjoint
    'hip_R',           # obs[4]  — wjoint
    'knee_R',          # obs[5]  — wjoint
    'ankle_R',         # obs[6]  — wjoint
    'foot_R',          # obs[7]  — wjoint
    'x_vel',           # obs[8]  — wx / wback  ← critical
    'z_vel',           # obs[9]
    'torso_ang_vel',   # obs[10]
    'knee_L_vel',      # obs[11]
    'ankle_L_vel',     # obs[12]
    'hip_R_vel',       # obs[13]
    'knee_R_vel',      # obs[14]
    'ankle_R_vel',     # obs[15]
    'foot_R_vel',      # obs[16]
]

# Indices directly used by latent_icem_walker cost function
ICEM_CRITICAL = {0, 1, 2, 3, 4, 5, 6, 7, 8}


def collect_latents(bundle, hdf5_dir, split, max_episodes, image_size=64):
    """Encode all frames from a split; return (Z, Y) numpy arrays."""
    hdf5_path = Path(hdf5_dir) / f'{split}.hdf5'
    device = bundle['device']
    zs, ys = [], []
    with torch.no_grad():
        with h5py.File(hdf5_path, 'r') as f:
            ep_grp  = f['episodes']
            ep_keys = sorted(ep_grp.keys(), key=lambda k: int(k))[:max_episodes]
            print(f'[eval]  {split}: encoding {len(ep_keys)} episodes …')
            for ep_key in ep_keys:
                ep    = ep_grp[ep_key]
                obs_t = ep['observations'][:]
                st_t  = ep['states'][:]
                T     = obs_t.shape[0] - 1
                if T < 2:
                    continue
                if obs_t.shape[1] != image_size or obs_t.shape[2] != image_size:
                    import torch.nn.functional as F_
                    t_ = torch.from_numpy(obs_t).permute(0, 3, 1, 2).float()
                    t_ = F_.interpolate(t_, (image_size, image_size),
                                        mode='bilinear', align_corners=False)
                    obs_t = t_.permute(0, 2, 3, 1).byte().numpy()
                for t in range(1, T + 1):
                    z = encode_obs(bundle, obs_t[t], obs_t[t - 1], st_t[t])
                    zs.append(z[0].cpu().numpy())
                    ys.append(st_t[t])
    Z = np.stack(zs).astype(np.float32)
    Y = np.stack(ys).astype(np.float32)
    print(f'[eval]  {split}: N={len(Z)}')
    return Z, Y


def per_dim_r2(Y_hat, Y):
    """R² for each output dimension."""
    ss_res = ((Y_hat - Y) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def print_table(r2_train, r2_val):
    """Print a compact per-dimension table."""
    print(f'\n{"dim":>3}  {"name":<16}  {"train R²":>9}  {"val R²":>8}  critical')
    print('-' * 58)
    for i, (lbl, r2t, r2v) in enumerate(zip(OBS_LABELS, r2_train, r2_val)):
        star = ' ★' if i in ICEM_CRITICAL else ''
        print(f'{i:3d}  {lbl:<16}  {r2t:9.4f}  {r2v:8.4f}{star}')
    print()
    macro_t = float(1 - np.mean((r2_train < 0)))  # fraction of dims with R²>0
    print(f'  overall train R²  = {r2_train.mean():.4f}  (min={r2_train.min():.4f})')
    print(f'  overall val   R²  = {r2_val.mean():.4f}  (min={r2_val.min():.4f})')
    crit = list(ICEM_CRITICAL)
    print(f'  iCEM-critical dims train R² = {r2_train[crit].mean():.4f}  '
          f'(min={r2_train[crit].min():.4f})')
    print(f'  iCEM-critical dims val   R² = {r2_val[crit].mean():.4f}  '
          f'(min={r2_val[crit].min():.4f})')


def make_figure(r2_train, r2_val, out_path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(OBS_LABELS))
    w = 0.38
    bars_t = ax.bar(x - w / 2, r2_train, width=w, label='train R²',
                    color='#2166ac', alpha=0.85)
    bars_v = ax.bar(x + w / 2, r2_val,   width=w, label='val R²',
                    color='#e07b00', alpha=0.85)

    # Highlight iCEM-critical bars
    for i in ICEM_CRITICAL:
        for bar in (bars_t[i], bars_v[i]):
            bar.set_edgecolor('#d62728')
            bar.set_linewidth(1.8)

    ax.axhline(0, color='k', linewidth=0.6, alpha=0.5)
    ax.axhline(1, color='k', linewidth=0.6, linestyle=':', alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(OBS_LABELS, rotation=40, ha='right', fontsize=9)
    ax.set_ylabel('R²', fontsize=13)
    ax.set_title('Walker2d probe quality per obs dimension\n'
                 '(red border = used by latent iCEM cost)', fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(min(-0.1, r2_train.min() - 0.05), 1.05)
    ax.grid(axis='y', color='#cccccc', linewidth=0.8, alpha=0.7)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    print(f'[done]  {out_path}')


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--ckpt',          required=True)
    p.add_argument('--cfg',           default=None)
    p.add_argument('--hdf5-dir',      required=True)
    p.add_argument('--probe-path',    default=None,
                   help='Saved .pt probe to load (skip training if present)')
    p.add_argument('--probe-episodes', type=int, default=300)
    p.add_argument('--probe-epochs',   type=int, default=50)
    p.add_argument('--val-episodes',   type=int, default=50)
    p.add_argument('--device',        default='cuda')
    p.add_argument('--out',           required=True,
                   help='Output PDF path')
    args = p.parse_args()

    bundle = load_walker_bundle(args.ckpt, args.cfg, args.device)
    z_dim  = int(bundle['model_cfg'].get('latent_dim', 192))

    # ── Load or train probe ───────────────────────────────────────────────────
    probe = None
    if args.probe_path and Path(args.probe_path).exists():
        print(f'[eval]  loading probe from {args.probe_path}')
        ck = torch.load(args.probe_path, map_location='cpu', weights_only=False)
        probe = MLPStateProbe(z_dim)
        probe._net.load_state_dict(ck['state_dict'] if 'state_dict' in ck else ck)
    else:
        print('[eval]  fitting probe …')
        probe = fit_walker_mlp_probe(
            bundle, args.hdf5_dir,
            max_episodes=args.probe_episodes,
            n_epochs=args.probe_epochs,
        )
        if args.probe_path:
            Path(args.probe_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({'state_dict': probe._net.state_dict(),
                        'z_dim': z_dim}, args.probe_path)
            print(f'[eval]  probe saved to {args.probe_path}')

    probe.to(torch.device('cpu'))
    net = probe._net.eval()

    # ── Collect latents for train and val ─────────────────────────────────────
    Z_tr, Y_tr = collect_latents(bundle, args.hdf5_dir, 'train',
                                 args.probe_episodes)
    Z_val, Y_val = collect_latents(bundle, args.hdf5_dir, 'val',
                                   args.val_episodes)

    # ── Predict ───────────────────────────────────────────────────────────────
    with torch.no_grad():
        Y_hat_tr  = net(torch.from_numpy(Z_tr)).numpy()
        Y_hat_val = net(torch.from_numpy(Z_val)).numpy()

    r2_train = per_dim_r2(Y_hat_tr,  Y_tr)
    r2_val   = per_dim_r2(Y_hat_val, Y_val)

    print_table(r2_train, r2_val)
    make_figure(r2_train, r2_val, args.out)


if __name__ == '__main__':
    main()
