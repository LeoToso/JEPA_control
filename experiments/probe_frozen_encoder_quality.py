#!/usr/bin/env python
"""Linear-probe quality of frozen visual encoders on CartPole.

Compares iBOT-based vs DINOv2-based SMWM encoders by asking:
  "How much physical CartPole state can be linearly decoded from
   the visual features alone (no proprioception)?"

For each model:
  1. Encode frames from the HDF5 dataset with proprio=None (visual only)
  2. Fit ridge regression z → [x, ẋ, θ, θ̇] on train split
  3. Evaluate R² per variable on val split

Key diagnostic: R²(θ) — can the frozen backbone linearly decode the
pole angle?  High R²(θ) → A_z can capture the real unstable mode.

Also reports:
  - rho(A_z), ||B_z|| at equilibrium for each model (via local_jacobians)

Usage:
  python experiments/probe_frozen_encoder_quality.py \\
      --model "1SP+iBOT:/mnt/.../ibot_model.pt:configs/ibot.yaml" \\
      --model "1SP+DINOv2:/mnt/.../dinov2_model.pt:configs/dinov2.yaml" \\
      --data  data/cartpole_excitation_depth4_fs5_128 \\
      --n-train 3000 --n-val 500 \\
      --out results/encoder_probe.pdf
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
import torch

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

from experiments.probe_utils import (
    equilibrium_latent, load_bundle, local_jacobians,
)

STATE_NAMES = [r'$x$', r'$\dot{x}$', r'$\theta$', r'$\dot{\theta}$']
COLORS      = ['#2166ac', '#d6604d', '#4dac26', '#8856a7', '#f4a582']


# ── env patch ─────────────────────────────────────────────────────────────────

def _env_patch(bundle):
    if 'environment' not in bundle['env_cfg']:
        bundle['env_cfg']['environment'] = {
            'frame_skip': int(bundle['model_cfg'].get('frame_skip', 5)),
            'image_size': int(bundle['model_cfg'].get('image_size', 128)),
            'action_range': [-10, 10],
            'mass_cart': 1.0, 'mass_pole': 0.1,
            'pole_length': 0.5, 'gravity': 9.8, 'dt': 0.02,
        }


# ── dataset ───────────────────────────────────────────────────────────────────

def _sample_frames(data_dir, n_samples, seed, split):
    """Randomly sample (obs, prev_obs, state) tuples from an HDF5 split.

    Returns:
        obs_list      : list of N HWC uint8 arrays  (current frame)
        prev_obs_list : list of N HWC uint8 arrays  (previous frame)
        states        : (N, 4) float32 physical states
    """
    if not _HAS_H5PY:
        raise ImportError('h5py required')
    rng = np.random.RandomState(seed)
    hdf5_path = Path(data_dir) / f'{split}.hdf5'
    if not hdf5_path.exists():
        raise FileNotFoundError(hdf5_path)

    candidates = []
    with h5py.File(hdf5_path, 'r') as f:
        for ep_key in sorted(f['episodes'].keys(), key=int):
            # observations has T+1 entries; index 0 has no prev frame, skip it.
            n_obs = f['episodes'][ep_key]['observations'].shape[0]
            for t in range(1, n_obs):
                candidates.append((ep_key, t))

    chosen = [candidates[i]
              for i in rng.choice(len(candidates),
                                  min(n_samples, len(candidates)), replace=False)]

    obs_list, prev_obs_list, states = [], [], []
    with h5py.File(hdf5_path, 'r') as f:
        for ep_key, t in chosen:
            ep = f['episodes'][ep_key]
            obs_list.append(np.array(ep['observations'][t],     dtype=np.uint8))
            prev_obs_list.append(np.array(ep['observations'][t - 1], dtype=np.uint8))
            # state before this frame (same convention as training)
            states.append(np.array(ep['states'][t - 1], dtype=np.float32))

    return obs_list, prev_obs_list, np.stack(states)


# ── visual-only encoding ──────────────────────────────────────────────────────

@torch.no_grad()
def encode_visual(bundle, obs_list, prev_obs_list):
    """Encode frames with proprio=None → (N, D) numpy array.

    Resizes observations to the model's expected image_size (from model_cfg)
    before encoding, so iBOT (224) and learned encoders (128) both work.
    """
    from experiments.probe_utils import obs_tensor
    import torch.nn.functional as F
    model     = bundle['model']
    device    = bundle['device']
    img_size  = int(bundle['model_cfg'].get('image_size', 128))
    feats = []
    for obs, prev_obs in zip(obs_list, prev_obs_list):
        t_obs  = obs_tensor(obs,      device)   # (1, C, H, W)
        t_prev = obs_tensor(prev_obs, device)
        if t_obs.shape[-1] != img_size:
            t_obs  = F.interpolate(t_obs,  size=(img_size, img_size),
                                   mode='bilinear', align_corners=False)
            t_prev = F.interpolate(t_prev, size=(img_size, img_size),
                                   mode='bilinear', align_corners=False)
        # Call the backbone directly to skip the proprio-required check.
        # model.encode calls self.encoder(obs, proprio); we call it with
        # proprio=None but the encoder sub-modules raise if proprio_dim>0.
        # Instead, call self.encoder.backbone (or self.encoder) bypassing
        # the proprio branch.
        enc = model.encoder
        if hasattr(enc, 'backbone'):
            # DINOv2Encoder or IBOTEncoder: call backbone directly
            px = t_obs
            if model.use_frame_diff:
                px = torch.cat([t_prev, t_obs, t_obs - t_prev], dim=1)
            raw = enc.backbone.forward_features(px)
            if isinstance(raw, dict):
                feat = raw['x_norm_clstoken']          # DINOv2
            elif raw.ndim == 3:
                feat = raw[:, 0]                       # iBOT CLS token
            else:
                feat = raw
            if enc.projector is not None:
                feat = enc.projector(feat)
            z = feat
        else:
            z = model.encode(t_obs, t_prev, proprio=None)
        feats.append(z.cpu().float().numpy().reshape(-1))
    return np.stack(feats)   # (N, D)


# ── ridge probe ───────────────────────────────────────────────────────────────

def fit_ridge(Z, Y, alpha=1e-2):
    N, D = Z.shape
    Z_aug = np.concatenate([Z, np.ones((N, 1))], axis=1)
    A = Z_aug.T @ Z_aug + alpha * np.eye(D + 1)
    A[-1, -1] = 0.
    return np.linalg.solve(A, Z_aug.T @ Y)   # (D+1, 4)


def predict_ridge(W, Z):
    Z_aug = np.concatenate([Z, np.ones((len(Z), 1))], axis=1)
    return Z_aug @ W


def r2_score(Y_true, Y_pred):
    ss_res = ((Y_true - Y_pred) ** 2).sum(0)
    ss_tot = ((Y_true - Y_true.mean(0)) ** 2).sum(0)
    return 1. - ss_res / np.maximum(ss_tot, 1e-12)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_results(results, out_path):
    """
    results: list of (label, r2_array, rho, norm_Bz)
    """
    n_enc  = len(results)
    n_vars = 4
    x      = np.arange(n_vars)
    width  = 0.8 / n_enc

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, r2, rho, norm_Bz) in enumerate(results):
        offset = (i - n_enc / 2 + 0.5) * width
        full_label = f'{label}\nρ={rho:.3f}, ‖B_z‖={norm_Bz:.3f}'
        bars = ax.bar(x + offset, r2, width * 0.9,
                      label=full_label, color=COLORS[i % len(COLORS)])
        for bar, val in zip(bars, r2):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(STATE_NAMES, fontsize=14)
    ax.set_ylabel(r'$R^2$ on val split (visual features only)', fontsize=13)
    ax.set_ylim(0, 1.2)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    print(f'[done] {out_path}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model', action='append', dest='models',
                   metavar='LABEL:CKPT:CFG',
                   help='Repeat for each model: "label:ckpt_path:cfg_path"')
    p.add_argument('--data',    required=True,
                   help='HDF5 dataset dir with train.hdf5 / val.hdf5')
    p.add_argument('--n-train', type=int, default=3000)
    p.add_argument('--n-val',   type=int, default=500)
    p.add_argument('--ridge',   type=float, default=1e-2)
    p.add_argument('--seed',    type=int, default=42)
    p.add_argument('--device',  default='cuda')
    p.add_argument('--out',     required=True)
    args = p.parse_args()

    if not args.models:
        p.error('Provide at least one --model "label:ckpt:cfg"')

    print(f'[data] sampling frames …')
    obs_tr,  prev_tr,  Y_tr  = _sample_frames(
        args.data, args.n_train, args.seed,     'train')
    obs_val, prev_val, Y_val = _sample_frames(
        args.data, args.n_val,   args.seed + 1, 'val')
    print(f'  train theta [{Y_tr[:,2].min():.3f}, {Y_tr[:,2].max():.3f}] rad  '
          f'({len(obs_tr)} frames)')
    print(f'  val   theta [{Y_val[:,2].min():.3f}, {Y_val[:,2].max():.3f}] rad  '
          f'({len(obs_val)} frames)')

    results = []
    for model_spec in args.models:
        label, ckpt, cfg = model_spec.split(':', 2)
        print(f'\n[{label}] loading …')
        bundle = load_bundle(ckpt, cfg, args.device)
        _env_patch(bundle)

        print(f'[{label}] encoding train frames (visual only) …')
        Z_tr = encode_visual(bundle, obs_tr, prev_tr)
        print(f'[{label}] encoding val frames …')
        Z_val = encode_visual(bundle, obs_val, prev_val)
        print(f'  latent dim: {Z_tr.shape[1]}')

        W  = fit_ridge(Z_tr, Y_tr, alpha=args.ridge)
        r2 = r2_score(Y_val, predict_ridge(W, Z_val))
        for name, val in zip(STATE_NAMES, r2):
            print(f'  R²({name}) = {val:.4f}')

        print(f'[{label}] computing A_z, B_z at equilibrium …')
        z_eq = equilibrium_latent(bundle)
        A_z, B_z, fp_err = local_jacobians(bundle, z=z_eq)
        rho    = float(np.max(np.abs(np.linalg.eigvals(A_z))))
        norm_Bz = float(np.linalg.norm(B_z))
        print(f'  rho(A_z)={rho:.4f}  ||B_z||={norm_Bz:.4f}  fp_err={fp_err:.2e}')

        results.append((label, r2, rho, norm_Bz))

    plot_results(results, Path(args.out))


if __name__ == '__main__':
    main()
