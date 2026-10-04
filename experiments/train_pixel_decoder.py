#!/usr/bin/env python
"""Train a convolutional pixel decoder: z (latent_dim) → (3, 64, 64) image.

Uses (z_t, frame_t) pairs collected from HDF5 training data via SMWM encoder.

Usage
-----
  python experiments/train_pixel_decoder.py \\
      --ckpt   results/.../model_final.pt \\
      --cfg    configs/walker2d_jepa_sigreg_rollout_act1.yaml \\
      --hdf5-dir data/walker2d_fs5_64 \\
      --out    results/pixel_decoder_ms_sr.pt \\
      --epochs 20 \\
      --max-episodes 500
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

from experiments.walker2d_utils import load_walker_bundle
from experiments.probe_utils import encode_obs


# ── decoder architecture ──────────────────────────────────────────────────────

class ConvDecoder(nn.Module):
    """z (latent_dim,) → (3, 64, 64) image in [0, 1]."""

    def __init__(self, latent_dim: int, base_ch: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_ch = base_ch
        # Project latent to 4×4 spatial start
        self.fc = nn.Linear(latent_dim, base_ch * 4 * 4)
        # 4 → 8 → 16 → 32 → 64
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_ch,       base_ch // 2, 4, 2, 1),  # 8×8
            nn.BatchNorm2d(base_ch // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch // 2,  base_ch // 4, 4, 2, 1),  # 16×16
            nn.BatchNorm2d(base_ch // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch // 4,  base_ch // 8, 4, 2, 1),  # 32×32
            nn.BatchNorm2d(base_ch // 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch // 8,  3,            4, 2, 1),   # 64×64
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), self.base_ch, 4, 4)
        return self.deconv(x)


# ── data collection ───────────────────────────────────────────────────────────

@torch.no_grad()
def collect_pairs(bundle: dict, hdf5_dir: str, split: str,
                  max_episodes: int, image_size: int,
                  device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect (z, frame) pairs from HDF5 data.

    Returns
    -------
    zs     : (N, latent_dim) float32 on CPU
    frames : (N, 3, H, W)   float32 [0,1] on CPU
    """
    hdf5_path = Path(hdf5_dir) / f'{split}.hdf5'
    model = bundle['model']

    zs_list:     list[np.ndarray] = []
    frames_list: list[np.ndarray] = []

    with h5py.File(hdf5_path, 'r') as f:
        ep_grp  = f['episodes']
        ep_keys = sorted(ep_grp.keys(), key=lambda k: int(k))[:max_episodes]
        print(f'[collect] {split}: {len(ep_keys)} episodes from {hdf5_path.name}')

        for ep_key in ep_keys:
            ep    = ep_grp[ep_key]
            obs_t = ep['observations'][:]   # (T+1, H, W, 3) uint8
            st_t  = ep['states'][:]         # (T+1, 17) float32
            T     = obs_t.shape[0] - 1
            if T < 2:
                continue

            # Resize frames to model's expected image_size (keep uint8 for encoding)
            H, W = obs_t.shape[1], obs_t.shape[2]
            if H != image_size or W != image_size:
                t_raw   = (torch.from_numpy(obs_t)
                           .permute(0, 3, 1, 2).float())       # (T+1, 3, H, W) [0-255]
                t_raw   = F.interpolate(t_raw, (image_size, image_size),
                                        mode='bilinear', align_corners=False)
                obs_arr = t_raw.permute(0, 2, 3, 1).byte().numpy()  # uint8 for encoder
            else:
                t_raw   = torch.from_numpy(obs_t).permute(0, 3, 1, 2).float()  # [0-255]
                obs_arr = obs_t                                                   # uint8

            t_frames = t_raw / 255.  # [0, 1] decoder targets, always

            # Encode each frame (skip first — no previous frame)
            for t in range(1, T + 1):
                z = encode_obs(bundle, obs_arr[t], obs_arr[t - 1], st_t[t])
                zs_list.append(z[0].cpu().numpy())
                frames_list.append(t_frames[t].numpy())

    Z = np.stack(zs_list)      # (N, latent_dim)
    F_ = np.stack(frames_list) # (N, 3, H, W)   float32 [0,1]
    print(f'[collect] {split}: N={len(Z)} pairs  '
          f'z_dim={Z.shape[1]}  frame shape={F_.shape[1:]}')
    return torch.from_numpy(Z), torch.from_numpy(F_)


# ── training ──────────────────────────────────────────────────────────────────

def train(decoder: ConvDecoder,
          z_tr: torch.Tensor, f_tr: torch.Tensor,
          z_va: torch.Tensor, f_va: torch.Tensor,
          epochs: int, batch_size: int, lr: float,
          device: torch.device) -> ConvDecoder:
    decoder = decoder.to(device)
    opt = torch.optim.Adam(decoder.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    tr_ds = TensorDataset(z_tr, f_tr)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                       num_workers=0, pin_memory=False)

    best_val  = float('inf')
    best_sd   = None

    for ep in range(1, epochs + 1):
        decoder.train()
        tr_loss = 0.
        for zb, fb in tr_dl:
            zb, fb = zb.to(device), fb.to(device)
            loss = F.mse_loss(decoder(zb), fb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(zb)
        tr_loss /= len(z_tr)

        decoder.eval()
        with torch.no_grad():
            n_va   = len(z_va)
            va_loss = 0.
            for i in range(0, n_va, batch_size):
                zb = z_va[i:i + batch_size].to(device)
                fb = f_va[i:i + batch_size].to(device)
                va_loss += F.mse_loss(decoder(zb), fb).item() * len(zb)
            va_loss /= n_va

        sched.step()
        print(f'  epoch {ep:3d}/{epochs}  '
              f'tr={tr_loss:.5f}  val={va_loss:.5f}')

        if va_loss < best_val:
            best_val = va_loss
            best_sd  = {k: v.cpu().clone() for k, v in decoder.state_dict().items()}

    decoder.load_state_dict(best_sd)
    return decoder


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Train z→pixel decoder for a Walker2d SMWM.')
    p.add_argument('--ckpt',           required=True)
    p.add_argument('--cfg',            default=None)
    p.add_argument('--hdf5-dir',       required=True)
    p.add_argument('--out',            required=True,
                   help='Output path for decoder checkpoint (.pt)')
    p.add_argument('--max-episodes',   type=int, default=500)
    p.add_argument('--image-size',     type=int, default=64)
    p.add_argument('--epochs',         type=int, default=20)
    p.add_argument('--batch-size',     type=int, default=256)
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--base-ch',        type=int, default=256)
    p.add_argument('--device',         default='cuda')
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')

    # Load SMWM bundle (encoder only)
    from experiments.probe_poincare_walker2d import find_cfg
    cfg = args.cfg or find_cfg(args.ckpt)
    bundle = load_walker_bundle(args.ckpt, cfg, args.device)
    latent_dim = bundle['model_cfg']['latent_dim']
    print(f'[bundle] latent_dim={latent_dim}')

    # Collect (z, frame) pairs from train and val splits
    print('\n=== Collect training pairs ===')
    z_tr, f_tr = collect_pairs(bundle, args.hdf5_dir, 'train',
                                args.max_episodes, args.image_size, device)
    print('\n=== Collect validation pairs ===')
    z_va, f_va = collect_pairs(bundle, args.hdf5_dir, 'val',
                                max(10, args.max_episodes // 10),
                                args.image_size, device)

    # Build and train decoder
    decoder = ConvDecoder(latent_dim=latent_dim, base_ch=args.base_ch)
    print(f'\n[decoder] params={sum(p.numel() for p in decoder.parameters()):,}')
    print('\n=== Training ===')
    decoder = train(decoder, z_tr, f_tr, z_va, f_va,
                    epochs=args.epochs, batch_size=args.batch_size,
                    lr=args.lr, device=device)

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'latent_dim': latent_dim,
        'base_ch':    args.base_ch,
        'image_size': args.image_size,
        'state_dict': decoder.state_dict(),
    }, out_path)
    print(f'\n[saved] {out_path}')


if __name__ == '__main__':
    main()
