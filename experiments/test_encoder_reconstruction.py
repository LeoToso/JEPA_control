"""Killer encoder test: can we reconstruct pixel frames from the 8-dim latent?

Procedure:
  1. Load checkpoint (encoder frozen).
  2. Train a small pixel decoder  z ∈ R^8 → 64×64×3  on the existing dataset.
  3. Generate a fresh cartpole rollout (pixel space).
  4. Encode each frame → z_t, decode back → obs_hat_t.
  5. Show original vs reconstructed side-by-side + PSNR per frame.

If reconstruction is faithful the encoder captures all visually relevant
information about the pole/cart state.  Any failure of the JEPA system is
then attributable to the predictor, not the encoder.

Usage:
    python experiments/test_encoder_reconstruction.py \\
        --checkpoint results/jepa_v11_difenc/checkpoints/checkpoint_epoch0270.pt \\
        --config     configs/cartpole_jepa_v11_difenc.yaml \\
        --data       data/cartpole_visual_fs5_v4 \\
        --out        results/encoder_reconstruction.png \\
        --decoder-epochs 200
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ── Pixel decoder ──────────────────────────────────────────────────────────────
class PixelDecoder(nn.Module):
    """z ∈ R^d  →  (3, H, W) image in [0, 1]."""
    def __init__(self, latent_dim: int = 8, image_size: int = 64):
        super().__init__()
        # 4×4 spatial base, 256 channels → upsample ×16 → 64×64
        self.proj   = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),   # 4  → 8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128,  64, 4, stride=2, padding=1),   # 8  → 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d( 64,  32, 4, stride=2, padding=1),   # 16 → 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d( 32,   3, 4, stride=2, padding=1),   # 32 → 64
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.proj(z).view(z.shape[0], 256, 4, 4)
        return self.decode(x)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, cfg: dict, device):
    from models.jepa import make_jepa
    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state   = raw['model_state'] if 'model_state' in raw else raw
    ckpt_cfg = raw.get('config', {})
    mc = cfg['model']
    for k in ('latent_dim', 'action_latent_dim', 'action_encoder', 'encoder_type',
              'patch_size', 'frame_stack', 'use_frame_diff',
              'vit_embed_dim', 'vit_depth', 'vit_num_heads',
              'predictor_type', 'predictor_window',
              'predictor_embed_dim', 'predictor_depth',
              'predictor_num_heads', 'predictor_mlp_ratio'):
        if k in ckpt_cfg:
            mc[k] = ckpt_cfg[k]
    model = make_jepa(
        variant='E-full',
        latent_dim=int(mc.get('latent_dim', 8)),
        action_latent_dim=int(mc.get('action_latent_dim', 8)),
        action_encoder=mc.get('action_encoder', 'linear'),
        encoder_type=mc.get('encoder_type', 'vit'),
        image_size=int(mc.get('image_size', 64)),
        patch_size=int(mc.get('patch_size', 8)),
        frame_stack=int(mc.get('frame_stack', 1)),
        use_frame_diff=bool(mc.get('use_frame_diff', False)),
        vit_embed_dim=int(mc.get('vit_embed_dim', 128)),
        vit_depth=int(mc.get('vit_depth', 3)),
        vit_num_heads=int(mc.get('vit_num_heads', 4)),
        predictor_type=mc.get('predictor_type', 'transformer'),
        predictor_window=int(mc.get('predictor_window', 1)),
        predictor_embed_dim=int(mc.get('predictor_embed_dim', 128)),
        predictor_depth=int(mc.get('predictor_depth', 3)),
        predictor_num_heads=int(mc.get('predictor_num_heads', 4)),
        predictor_mlp_ratio=float(mc.get('predictor_mlp_ratio', 4.0)),
        predictor_hidden_dim=int(mc.get('predictor_hidden_dim', 256)),
        predictor_n_layers=int(mc.get('predictor_n_layers', 2)),
    )
    model.load_state_dict(state)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def encode_frame(model, curr_np: np.ndarray, prev_np: np.ndarray, device) -> np.ndarray:
    """Encode a single (H,W,3) uint8 frame. prev_np = previous frame (for frame_diff)."""
    def to_t(x): return torch.from_numpy(x).float().permute(2,0,1).unsqueeze(0).to(device)/255.
    return model.encode_obs(to_t(curr_np), to_t(prev_np)).squeeze(0).cpu().numpy()


# ── Dataset loading ────────────────────────────────────────────────────────────
def _resize_np(img: np.ndarray, size: int) -> np.ndarray:
    """Resize (H,W,3) uint8 using PIL — no cv2 required."""
    from PIL import Image
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return np.array(Image.fromarray(img).resize((size, size), Image.BILINEAR))


def collect_z_obs_pairs(model, data_dir: str, device,
                        use_frame_diff: bool, max_pairs: int = 50_000,
                        target_size: int = 64, encode_batch: int = 512):
    """Scan train.hdf5, encode frames in batches, return (Z, OBS) arrays."""
    train_h5 = Path(data_dir) / 'train.hdf5'

    # Gather raw (curr, prev) numpy pairs first
    curr_buf, prev_buf, obs_buf = [], [], []
    with h5py.File(train_h5, 'r') as f:
        for ep_key in f['episodes']:
            ep  = f['episodes'][ep_key]
            obs = ep['observations'][:]   # (T, H, W, 3) uint8
            T   = len(obs)
            for t in range(T):
                curr = _resize_np(obs[t],             target_size)
                prev = _resize_np(obs[t-1] if t > 0 else obs[t], target_size)
                curr_buf.append(curr)
                prev_buf.append(prev)
                obs_buf.append(curr)
                if len(curr_buf) >= max_pairs:
                    break
            if len(curr_buf) >= max_pairs:
                break

    N = len(curr_buf)
    print(f'[data] encoding {N:,} frames in batches of {encode_batch} ...')

    # Batch-encode
    def to_batch(frames):
        arr = np.stack(frames).astype(np.float32) / 255.0   # (B, H, W, 3)
        return torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)  # (B, 3, H, W)

    Z_list = []
    with torch.no_grad():
        for start in tqdm(range(0, N, encode_batch), desc='encoding'):
            end   = min(start + encode_batch, N)
            c_t   = to_batch(curr_buf[start:end])
            p_t   = to_batch(prev_buf[start:end])
            z_b   = model.encode_obs(c_t, p_t)   # (B, d)
            Z_list.append(z_b.cpu().numpy())

    Z = np.concatenate(Z_list, axis=0).astype(np.float32)   # (N, d)
    O = np.stack(obs_buf).astype(np.float32) / 255.0         # (N, H, W, 3)
    O = np.transpose(O, (0, 3, 1, 2))                        # (N, 3, H, W)
    print(f'[data] collected {N:,} (z, obs) pairs')
    return Z, O


# ── Decoder training ───────────────────────────────────────────────────────────
def train_decoder(Z: np.ndarray, O: np.ndarray, device,
                  epochs: int = 200, batch_size: int = 512, lr: float = 3e-4):
    d = Z.shape[1]
    img_size = O.shape[-1]
    decoder = PixelDecoder(latent_dim=d, image_size=img_size).to(device)
    opt = torch.optim.Adam(decoder.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.1)

    Z_t = torch.tensor(Z, dtype=torch.float32)
    O_t = torch.tensor(O, dtype=torch.float32)
    ds  = TensorDataset(Z_t, O_t)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True,
                     num_workers=2, pin_memory=True, drop_last=True)

    print(f'[decoder] training for {epochs} epochs  '
          f'({len(ds):,} pairs, batch={batch_size})')
    for ep in range(epochs):
        decoder.train()
        losses = []
        for zb, ob in dl:
            zb, ob = zb.to(device), ob.to(device)
            pred = decoder(zb)
            loss = F.mse_loss(pred, ob)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        if (ep + 1) % 50 == 0 or ep == 0:
            mse   = float(np.mean(losses))
            psnr_ = 20 * np.log10(1.0 / (np.sqrt(mse) + 1e-8))
            print(f'  ep {ep+1:3d}/{epochs}  MSE={mse:.5f}  PSNR≈{psnr_:.1f} dB')
    decoder.eval()
    return decoder


# ── Rollout generation ─────────────────────────────────────────────────────────
def generate_rollout(cfg: dict, n_steps: int = 30, seed: int = 0, target_size: int = 64):
    """Generate a cartpole rollout starting near equilibrium. Returns list of uint8 obs."""
    from envs.cartpole_visual import ContinuousCartpoleVisual
    env_cfg = cfg['environment']
    env = ContinuousCartpoleVisual(
        frame_skip=int(env_cfg.get('frame_skip', 1)),
        image_size=int(env_cfg['image_size']),
        mass_cart=float(env_cfg['mass_cart']),
        mass_pole=float(env_cfg['mass_pole']),
        pole_length=float(env_cfg['pole_length']),
        gravity=float(env_cfg['gravity']),
        dt=float(env_cfg['dt']),
        seed=seed,
    )
    rng = np.random.RandomState(seed)
    x0  = rng.uniform(-0.05, 0.05, 4).astype(np.float32)
    obs0, _, _ = env.reset_to_state(x0)
    frames = [_resize_np(obs0, target_size)]
    for _ in range(n_steps - 1):
        obs, _, _, _, _ = env.step(0.0)   # passive — no control
        frames.append(_resize_np(obs, target_size))
    return frames   # list of (H, W, 3) uint8


# ── Visualisation ──────────────────────────────────────────────────────────────
def make_figure(originals, reconstructed, psnrs, out_path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(originals)
    # Show up to 15 frames evenly spaced
    idx = np.linspace(0, n - 1, min(n, 15), dtype=int)

    fig, axes = plt.subplots(3, len(idx), figsize=(len(idx) * 1.5, 5))
    fig.suptitle('Encoder reconstruction test\n'
                 'top: original  |  middle: decoded  |  bottom: |diff|×3',
                 fontsize=10)

    for col, i in enumerate(idx):
        orig = originals[i]
        recon = reconstructed[i]
        diff  = np.abs(orig.astype(np.float32) - recon.astype(np.float32)) * 3
        diff  = np.clip(diff, 0, 255).astype(np.uint8)

        axes[0, col].imshow(orig);    axes[0, col].axis('off')
        axes[1, col].imshow(recon);   axes[1, col].axis('off')
        axes[2, col].imshow(diff);    axes[2, col].axis('off')
        axes[0, col].set_title(f't={i}', fontsize=7)
        axes[1, col].set_title(f'{psnrs[i]:.1f}dB', fontsize=7)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[plot] saved → {out_path}')


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',      required=True)
    p.add_argument('--config',          required=True)
    p.add_argument('--data',            required=True)
    p.add_argument('--out',             default='results/encoder_reconstruction.png')
    p.add_argument('--decoder-epochs',  type=int,   default=200)
    p.add_argument('--decoder-batch',   type=int,   default=512)
    p.add_argument('--max-pairs',       type=int,   default=50_000)
    p.add_argument('--rollout-steps',   type=int,   default=30)
    p.add_argument('--rollout-seed',    type=int,   default=7)
    p.add_argument('--device',          default=None)
    args = p.parse_args()

    device = torch.device(args.device if args.device
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Device: {device}')

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    use_fd   = bool(cfg['model'].get('use_frame_diff', False))
    img_size = int(cfg['model'].get('image_size', 64))

    # 1. Load model (frozen)
    print('[model] loading encoder ...')
    model = load_model(args.checkpoint, cfg, device)
    d = model.config.latent_dim
    print(f'[model] latent_dim={d}  use_frame_diff={use_fd}  image_size={img_size}')

    # 2. Collect (z, obs) pairs from dataset and train decoder
    Z, O = collect_z_obs_pairs(model, args.data, device, use_fd,
                               max_pairs=args.max_pairs, target_size=img_size)
    decoder = train_decoder(Z, O, device,
                            epochs=args.decoder_epochs,
                            batch_size=args.decoder_batch)

    # 3. Generate a fresh rollout
    print(f'[rollout] generating {args.rollout_steps} steps (seed={args.rollout_seed}) ...')
    frames = generate_rollout(cfg, n_steps=args.rollout_steps,
                              seed=args.rollout_seed, target_size=img_size)

    # 4. Encode + decode each frame
    originals, reconstructed, psnrs = [], [], []
    with torch.no_grad():
        for t, curr in enumerate(frames):
            prev = frames[t-1] if t > 0 else curr
            z    = encode_frame(model, curr, prev, device)
            z_t  = torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0)
            rec  = decoder(z_t).squeeze(0).permute(1, 2, 0).cpu().numpy()
            rec  = (rec * 255).clip(0, 255).astype(np.uint8)
            originals.append(curr)
            reconstructed.append(rec)
            psnrs.append(psnr(curr, rec))

    mean_psnr = float(np.mean(psnrs))
    print(f'\n[results] mean PSNR = {mean_psnr:.2f} dB  '
          f'(range {min(psnrs):.1f}–{max(psnrs):.1f} dB)')
    print(f'[results] per-frame PSNR: '
          + '  '.join(f't{i}:{v:.1f}' for i, v in enumerate(psnrs)))

    # 5. Plot
    make_figure(originals, reconstructed, psnrs, args.out)


if __name__ == '__main__':
    main()
