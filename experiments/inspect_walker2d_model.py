#!/usr/bin/env python
"""Inspect Walker2D world-model quality via a learned pixel decoder.

Trains a lightweight CNN decoder on encoded training frames, then shows a
3-row comparison for a test episode:
  Row 1  GT observations
  Row 2  Encoder quality  — encode each frame, decode back to pixels
  Row 3  Predictor quality — open-loop latent rollout from frame 0, decode

Usage
-----
  python experiments/inspect_walker2d_model.py \\
      --ckpt /mnt/t7shield/jepa_results/walker2d_mixed_sac_smwm_fwd_endpoint_inverse_act1_seed42/model_final.pt \\
      --cfg  configs/walker2d_smwm_fwd_endpoint_inverse_act1.yaml \\
      --data data/walker2d_mixed_sac_fs5_64 \\
      --test-episode 0 \\
      --decoder-epochs 100 \\
      --n-vis 8 \\
      --output results/inspect_walker2d_fwd_ep_ar.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.sensorimotor_world_model import SensorimotorWorldModel


# ── model loading ─────────────────────────────────────────────────────────────

def load_walker2d_bundle(ckpt_path: str, cfg_path: str,
                         device: torch.device) -> dict:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = dict(ckpt['model_config'])
    model = SensorimotorWorldModel(model_cfg).to(device).eval()
    model.load_state_dict(ckpt['model'])
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'[model] loaded  latent_dim={model_cfg.get("latent_dim")}  '
          f'params={n_params:.1f}M')
    state_mean   = np.asarray(ckpt.get('state_mean', np.zeros(17)), dtype=np.float32)
    state_std    = np.asarray(ckpt.get('state_std',  np.ones(17)),  dtype=np.float32)
    action_scale = float(ckpt.get('action_scale', 1.0))
    print(f'[model] action_scale={action_scale}  use_proprio={model.use_proprio}')
    return {
        'model': model, 'model_cfg': model_cfg, 'env_cfg': cfg,
        'device': device,
        'state_mean': state_mean, 'state_std': state_std,
        'action_scale': action_scale,
    }


# ── dataset helpers ───────────────────────────────────────────────────────────

def load_frames_for_decoder(data_dir: str, split: str,
                             n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (obs, prev_obs, states) arrays of shape (N, H, W, C) / (N, 17)."""
    hdf5 = Path(data_dir) / f'{split}.hdf5'
    obs_list, prev_list, st_list = [], [], []
    with h5py.File(hdf5, 'r') as f:
        for ep_idx in range(len(f['episodes'])):
            ep  = f[f'episodes/{ep_idx}']
            obs = ep['observations'][:]   # (T+1, H, W, C)
            sts = ep['states'][:]         # (T+1, 17)
            for t in range(1, len(obs)):
                obs_list.append(obs[t])
                prev_list.append(obs[t - 1])
                st_list.append(sts[t])
            if len(obs_list) >= n_frames:
                break
    return (np.stack(obs_list[:n_frames]),
            np.stack(prev_list[:n_frames]),
            np.stack(st_list[:n_frames]))


def load_test_episode(data_dir: str, split: str,
                      ep_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hdf5 = Path(data_dir) / f'{split}.hdf5'
    with h5py.File(hdf5, 'r') as f:
        ep  = f[f'episodes/{ep_idx}']
        obs = ep['observations'][:]   # (T+1, H, W, C)
        act = ep['actions'][:]         # (T, 6)
        sts = ep['states'][:]          # (T+1, 17)
    return obs, act, sts


# ── encoding ──────────────────────────────────────────────────────────────────

def to_chw(obs_hwc: np.ndarray, device: torch.device) -> torch.Tensor:
    return (torch.from_numpy(obs_hwc.copy()).float()
            .permute(2, 0, 1).unsqueeze(0).to(device) / 255.)


@torch.no_grad()
def encode_frame(model, obs_hwc, prev_hwc, state, s_mean, s_std, device):
    cur  = to_chw(obs_hwc,  device)
    prev = to_chw(prev_hwc, device)
    if model.use_proprio:
        norm = ((state - s_mean) / np.maximum(s_std, 1e-6)).astype(np.float32)
        prop = torch.from_numpy(norm).unsqueeze(0).to(device)
    else:
        prop = None
    return model.encode(cur, prev, prop)   # (1, D)


@torch.no_grad()
def predict_step(model, z, action_6d, action_scale, device):
    """One predictor step for a 6-D Walker2D macro-action."""
    a     = torch.as_tensor(action_6d, dtype=z.dtype, device=device).unsqueeze(0)  # (1,6)
    a_ctx = model.expand_action(a / action_scale).unsqueeze(1)   # (1,1,ctx)
    return model.predict(z.unsqueeze(1), a_ctx)[:, 0]            # (1, D)


# ── CNN pixel decoder ─────────────────────────────────────────────────────────

class PixelDecoder(nn.Module):
    """z ∈ ℝ^D → image ∈ [0,1]^{3×64×64}."""
    def __init__(self, z_dim: int = 192):
        super().__init__()
        self.proj = nn.Linear(z_dim, 256 * 4 * 4)
        self.net  = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),   # 4→8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128,  64, 4, 2, 1),   # 8→16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d( 64,  32, 4, 2, 1),   # 16→32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d( 32,   3, 4, 2, 1),   # 32→64
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(self.proj(z).view(-1, 256, 4, 4))


def train_decoder(z_train: torch.Tensor, img_train: torch.Tensor,
                  z_dim: int, n_epochs: int, lr: float, batch_size: int,
                  device: torch.device) -> PixelDecoder:
    decoder = PixelDecoder(z_dim).to(device)
    opt = torch.optim.Adam(decoder.parameters(), lr=lr)
    dl  = DataLoader(TensorDataset(z_train, img_train),
                     batch_size=batch_size, shuffle=True)
    for ep in range(n_epochs):
        total = 0.
        for zb, ob in dl:
            loss = F.mse_loss(decoder(zb), ob)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f'  decoder epoch {ep+1:4d}/{n_epochs}  '
                  f'loss={total/len(dl):.5f}', flush=True)
    return decoder.eval()


# ── figure ────────────────────────────────────────────────────────────────────

def to_img(arr: np.ndarray) -> np.ndarray:
    """Convert (3,H,W) or (H,W,3) float to (H,W,3) clipped to [0,1]."""
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    return np.clip(arr, 0., 1.)


def save_figure(gt, recon, pred, frame_idxs, title, out_path):
    n = len(gt)
    fig, axes = plt.subplots(3, n, figsize=(2.2 * n, 7),
                             constrained_layout=True)
    row_labels = ['GT', 'Encoder\nrecon', 'Predictor\nrollout']
    for r, (frames, label) in enumerate(zip([gt, recon, pred], row_labels)):
        for c, img in enumerate(frames):
            axes[r, c].imshow(to_img(img))
            axes[r, c].axis('off')
            if r == 0:
                axes[r, c].set_title(f't={frame_idxs[c]}', fontsize=8)
        axes[r, 0].set_ylabel(label, fontsize=10, rotation=0,
                               labelpad=65, va='center')
    fig.suptitle(title, fontsize=13, fontweight='bold')
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {out}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Walker2D world-model encoder/predictor inspection.')
    p.add_argument('--ckpt',           required=True)
    p.add_argument('--cfg',            required=True)
    p.add_argument('--data',           required=True)
    p.add_argument('--train-split',    default='train')
    p.add_argument('--test-split',     default='test',
                   help='Falls back to --train-split if test split missing.')
    p.add_argument('--test-episode',   type=int, default=0)
    p.add_argument('--n-train-frames', type=int, default=8000,
                   help='Frames used to train the pixel decoder.')
    p.add_argument('--decoder-epochs', type=int, default=100)
    p.add_argument('--decoder-lr',     type=float, default=1e-3)
    p.add_argument('--decoder-batch',  type=int, default=256)
    p.add_argument('--n-vis',          type=int, default=8,
                   help='Number of frames to show in the comparison figure.')
    p.add_argument('--device',         default='cuda')
    p.add_argument('--title',          default='Walker2D model inspection')
    p.add_argument('--output',         required=True)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')

    # ── load model ────────────────────────────────────────────────────────────
    bundle      = load_walker2d_bundle(args.ckpt, args.cfg, device)
    model       = bundle['model']
    z_dim       = int(bundle['model_cfg'].get('latent_dim', 192))
    act_scale   = bundle['action_scale']
    s_mean      = bundle['state_mean']
    s_std       = bundle['state_std']

    # ── encode training frames ────────────────────────────────────────────────
    print(f'[encode] loading up to {args.n_train_frames} frames '
          f'from {args.train_split} …')
    obs_tr, prev_tr, st_tr = load_frames_for_decoder(
        args.data, args.train_split, args.n_train_frames)
    print(f'  loaded {len(obs_tr)} frames')

    z_list, img_list = [], []
    BS = 64
    with torch.no_grad():
        for i in range(0, len(obs_tr), BS):
            cur_b  = torch.from_numpy(obs_tr[i:i+BS]).float().permute(0,3,1,2).to(device) / 255.
            prv_b  = torch.from_numpy(prev_tr[i:i+BS]).float().permute(0,3,1,2).to(device) / 255.
            prop   = None
            if model.use_proprio:
                norm = ((st_tr[i:i+BS] - s_mean) / np.maximum(s_std, 1e-6)).astype(np.float32)
                prop = torch.from_numpy(norm).to(device)
            z_list.append(model.encode(cur_b, prv_b, prop).cpu())
            img_list.append(cur_b.cpu())
            if (i // BS + 1) % 20 == 0:
                print(f'  encoded {min(i+BS, len(obs_tr))}/{len(obs_tr)}', flush=True)

    z_train   = torch.cat(z_list)
    img_train = torch.cat(img_list)
    print(f'  z_train {z_train.shape}  img_train {img_train.shape}')

    # ── train decoder ─────────────────────────────────────────────────────────
    print(f'[decoder] training {args.decoder_epochs} epochs …')
    decoder = train_decoder(z_train.to(device), img_train.to(device),
                            z_dim, args.decoder_epochs,
                            args.decoder_lr, args.decoder_batch, device)

    # ── load test episode ─────────────────────────────────────────────────────
    for split in [args.test_split, args.train_split]:
        try:
            obs_ep, act_ep, st_ep = load_test_episode(
                args.data, split, args.test_episode)
            print(f'[test] episode {args.test_episode} from {split}  '
                  f'T={len(act_ep)}  obs={obs_ep.shape}')
            break
        except Exception as e:
            print(f'[test] split={split} not found ({e}), trying next …')

    T = len(act_ep)
    stride = max(1, T // args.n_vis)
    frame_idxs = list(range(0, min(T + 1, stride * args.n_vis), stride))[:args.n_vis]
    print(f'  visualising frames {frame_idxs}')

    # ── encoder quality ───────────────────────────────────────────────────────
    gt_frames    = []
    recon_frames = []
    for t in frame_idxs:
        prev_t = max(0, t - 1)
        z = encode_frame(model, obs_ep[t], obs_ep[prev_t],
                         st_ep[t], s_mean, s_std, device)
        with torch.no_grad():
            recon = decoder(z)[0].cpu().numpy()
        gt_frames.append(obs_ep[t].astype(np.float32) / 255.)
        recon_frames.append(recon)

    # ── predictor quality: open-loop rollout from t=0 ─────────────────────────
    print('[pred rollout] open-loop from t=0 …')
    z_buf = {}
    z_buf[0] = encode_frame(model, obs_ep[0], obs_ep[0],
                             st_ep[0], s_mean, s_std, device)
    t_max = max(frame_idxs)
    with torch.no_grad():
        for t in range(t_max):
            z_buf[t + 1] = predict_step(model, z_buf[t], act_ep[t],
                                        act_scale, device)

    pred_frames = []
    with torch.no_grad():
        for t in frame_idxs:
            pred_frames.append(decoder(z_buf[t])[0].cpu().numpy())

    # ── save figure ───────────────────────────────────────────────────────────
    save_figure(gt_frames, recon_frames, pred_frames,
                frame_idxs, args.title, args.output)


if __name__ == '__main__':
    main()
