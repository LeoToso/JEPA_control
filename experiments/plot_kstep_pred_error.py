#!/usr/bin/env python
"""K-step prediction error vs rollout step for SMWM models.

Two panels:
  Left  — full predictor latent error   ||ẑ_k^pred - z_k^GT||
  Right — linearized-system latent error ||ẑ_k^lin - z_k^GT||

GT latents z_0,...,z_K are obtained by encoding the actual dataset frames
(no re-simulation). Both the predictor and the linearized system are driven
by the real action sequence from the dataset.

Linearized rollout:
  δz_{k+1} = A_z @ δz_k + B_z * u_k
  ẑ_{k+1}  = z_eq + δz_{k+1}

Prints rho(A_z) and ||B_z|| per model.

Usage:
  python experiments/plot_kstep_pred_error_smwm.py \\
      --model "1SP+EP-IDM:/path/model.pt:configs/foo.yaml" \\
      --model "MSP+EP-IDM:/path/model.pt:configs/bar.yaml" \\
      --data  data/cartpole_excitation_depth4_fs5_128 \\
      --K 15 --n-states 80 \\
      --out results/kstep_pred_error.pdf
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
import numpy as np
import torch

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

from experiments.probe_utils import (
    encode_obs, equilibrium_latent,
    load_bundle, local_jacobians,
)
from experiments.probe_utils import predict_one


# ── dataset helpers ───────────────────────────────────────────────────────────

def _sample_trajectories(data_dir, n_states, K, seed, split='train'):
    """Sample n_states segments of length K from the HDF5 dataset.

    Returns
    -------
    padded_obs  : (N, K+2, H, W, C) uint8
        Frame sequence.  Index 0 is the frame just before t0 (used as
        prev_obs when encoding z_0).  Index k+1 is the frame at t0+k.
        So encode z_k  →  encode_obs(bundle, padded[k+1], padded[k], states[k]).
    state_seqs  : (N, K+1, 4)   physical states at t0, t0+1, …, t0+K
    action_seqs : (N, K)        actions u_0, …, u_{K-1}
    """
    if not _HAS_H5PY:
        raise ImportError('h5py required for --data')
    rng = np.random.RandomState(seed)
    hdf5_path = Path(data_dir) / f'{split}.hdf5'
    if not hdf5_path.exists():
        raise FileNotFoundError(hdf5_path)

    # Build candidate (ep_key, t0) pairs where t0 >= 1 so we always have a
    # previous frame, and t0+K < n_ep.
    candidates = []
    with h5py.File(hdf5_path, 'r') as f:
        for ep_key in sorted(f['episodes'].keys(), key=int):
            n_trans = len(f['episodes'][ep_key]['actions'])
            # t0 >= 1 (need prev frame) and t0+K <= n_trans (need K more frames)
            for t in range(1, n_trans - K + 1):
                candidates.append((ep_key, t))

    chosen = [candidates[i] for i in rng.choice(len(candidates), n_states, replace=False)]

    padded_obs_list, state_list, action_list = [], [], []
    with h5py.File(hdf5_path, 'r') as f:
        for ep_key, t0 in chosen:
            ep = f['episodes'][ep_key]
            # 'observations' has shape (T+1, H, W, C): index k is the frame
            # at time k.  'states' has shape (T+1, 4) with the same indexing.
            # Padded sequence: frame at t0-1, frame at t0, …, frame at t0+K.
            frames = ep['observations'][t0 - 1: t0 + K + 1]  # (K+2, H, W, C)
            padded_obs_list.append(np.array(frames, dtype=np.uint8))

            states = ep['states'][t0: t0 + K + 1]            # (K+1, 4)
            state_list.append(np.array(states, dtype=np.float32))

            acts = ep['actions'][t0: t0 + K].reshape(K).astype(np.float32)
            action_list.append(acts)

    padded_obs  = np.stack(padded_obs_list)   # (N, K+2, H, W, C)
    state_seqs  = np.stack(state_list)        # (N, K+1, 4)
    action_seqs = np.stack(action_list)       # (N, K)

    thetas = state_seqs[:, 0, 2]
    print(f'  theta range: [{thetas.min():.3f}, {thetas.max():.3f}] rad')
    print(f'  action range: [{action_seqs.min():.3f}, {action_seqs.max():.3f}]')
    return padded_obs, state_seqs, action_seqs


def _encode_gt_latents(bundle, padded_obs, state_seqs, K):
    """Encode GT latents z_0,...,z_K from dataset frames.

    padded_obs : (K+2, H, W, C) for a single trajectory
    state_seqs : (K+1, 4)       states at t0, t0+1, ..., t0+K
    Returns z_list: list of K+1 numpy arrays, each (latent_dim,)

    Encoding convention (matches training):
        z_k = encode(obs_k, obs_{k-1}, obs_k - obs_{k-1}, s_{k-1})
    where s_{k-1} is the physical state *before* the step that produced obs_k.
    For z_0 we use s_0 (no prior state available).
    """
    z_list = []

    # z_0: encode(obs_{t0}, obs_{t0-1}, s_{t0})
    z0 = encode_obs(bundle, padded_obs[1], padded_obs[0], state_seqs[0])
    z_list.append(z0.cpu().numpy()[0])

    # z_{k+1}: encode(obs_{t0+k+1}, obs_{t0+k}, s_{t0+k})  — state *before* step
    for k in range(K):
        obs      = padded_obs[k + 2]   # frame at t0+k+1
        prev_obs = padded_obs[k + 1]   # frame at t0+k
        state    = state_seqs[k]       # s_{t0+k}  (before the step)
        z = encode_obs(bundle, obs, prev_obs, state).cpu().numpy()[0]
        z_list.append(z)

    return z_list  # [z_0, z_1, ..., z_K]


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


# ── error functions ───────────────────────────────────────────────────────────

def kstep_predictor_errors(bundle, padded_obs, state_seqs, action_seqs, K):
    """(N, K) of ||ẑ_k^pred - z_k^GT||.

    GT latents are encoded directly from dataset frames.
    The predictor is unrolled one step at a time with the dataset actions.
    """
    N = len(padded_obs)
    all_errors = np.full((N, K), np.nan, dtype=np.float32)

    for i in range(N):
        gt = _encode_gt_latents(bundle, padded_obs[i], state_seqs[i], K)
        # encode z_0 to start the predictor
        z_cur = encode_obs(bundle, padded_obs[i][1], padded_obs[i][0],
                           state_seqs[i][0])
        for k in range(K):
            z_cur = predict_one(bundle, z_cur, float(action_seqs[i][k]))
            z_pred_k = z_cur.detach().cpu().numpy()[0]
            all_errors[i, k] = float(np.linalg.norm(z_pred_k - gt[k + 1]))

    return all_errors


def kstep_linear_errors(bundle, padded_obs, state_seqs, action_seqs, K):
    """(N, K) of ||ẑ_k^lin - z_k^GT||.

    GT latents are encoded directly from dataset frames.
    The linearized system is unrolled with the dataset actions:
        δz_{k+1} = A_z @ δz_k + B_z * u_k
        ẑ_{k+1}  = z_eq + δz_{k+1}
    """
    z_eq_t = equilibrium_latent(bundle)
    A_z, B_z, fp_err = local_jacobians(bundle, z=z_eq_t)
    z_eq = z_eq_t.cpu().numpy().flatten()
    B_z  = B_z.reshape(-1)
    rho  = np.max(np.abs(np.linalg.eigvals(A_z)))
    print(f'  rho(A_z)={rho:.4f}  ||B_z||={np.linalg.norm(B_z):.4f}  fp_err={fp_err:.2e}')

    N = len(padded_obs)
    all_errors = np.full((N, K), np.nan, dtype=np.float32)

    for i in range(N):
        gt = _encode_gt_latents(bundle, padded_obs[i], state_seqs[i], K)
        dz = gt[0] - z_eq  # initial deviation
        for k in range(K):
            u  = float(action_seqs[i][k])
            dz = A_z @ dz + B_z * u
            z_lin = z_eq + dz
            all_errors[i, k] = float(np.linalg.norm(z_lin - gt[k + 1]))

    return all_errors


# ── plotting ──────────────────────────────────────────────────────────────────

COLORS = ['#2166ac', '#d6604d', '#4dac26', '#8856a7', '#f4a582']


def _plot_curve(ax, ks, errs, color, label):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mean = np.nanmean(errs, axis=0)
        std  = np.nanstd(errs,  axis=0)
    ax.plot(ks, mean, color=color, lw=2.2, label=label)
    ax.fill_between(ks,
                    np.maximum(mean - std, 0.),
                    mean + std,
                    color=color, alpha=0.18)
    return mean


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model', action='append', dest='models',
                   metavar='LABEL:CKPT:CFG',
                   help='Repeat for each model: "label:ckpt_path:cfg_path"')
    p.add_argument('--data',       required=True,
                   help='HDF5 dataset dir containing train.hdf5')
    p.add_argument('--data-split', default='train')
    p.add_argument('--K',          type=int, default=15)
    p.add_argument('--n-states',   type=int, default=80)
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--device',     default='cuda')
    p.add_argument('--out',        required=True)
    args = p.parse_args()

    if not args.models:
        p.error('Provide at least one --model "label:ckpt:cfg"')

    print(f'[data] loading trajectories from {args.data} ({args.data_split}) …')
    padded_obs, state_seqs, action_seqs = _sample_trajectories(
        args.data, args.n_states, args.K, args.seed, args.data_split)

    fig, (ax_pred, ax_lin) = plt.subplots(1, 2, figsize=(13, 5))
    ks = np.arange(1, args.K + 1)

    for idx, model_spec in enumerate(args.models):
        label, ckpt, cfg = model_spec.split(':', 2)
        color = COLORS[idx % len(COLORS)]

        print(f'\n[{label}] loading …')
        bundle = load_bundle(ckpt, cfg, args.device)
        _env_patch(bundle)

        print(f'[{label}] full-predictor errors …')
        pred_errs = kstep_predictor_errors(
            bundle, padded_obs, state_seqs, action_seqs, args.K)
        m_pred = _plot_curve(ax_pred, ks, pred_errs, color, label)
        print(f'  k=1:{m_pred[0]:.3f}  k={args.K}:{m_pred[-1]:.3f}')

        print(f'[{label}] linearised-system errors …')
        lin_errs = kstep_linear_errors(
            bundle, padded_obs, state_seqs, action_seqs, args.K)
        m_lin = _plot_curve(ax_lin, ks, lin_errs, color, label)
        print(f'  k=1:{m_lin[0]:.3f}  k={args.K}:{m_lin[-1]:.3f}')

    for ax, ylabel in [
        (ax_pred, r'$\|\hat{z}_k^{\rm pred} - z_k^{\rm GT}\|$'),
        (ax_lin,  r'$\|\hat{z}_k^{\rm lin} - z_k^{\rm GT}\|$'),
    ]:
        ax.set_xlabel('Rollout step $k$', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(labelsize=14)
        ax.legend(fontsize=14)
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    print(f'\n[done] {out}')


if __name__ == '__main__':
    main()
