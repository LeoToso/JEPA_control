"""Killer predictor test: decode the predicted latent and compare to real pixels.

Procedure:
  1. Load checkpoint (encoder + predictor, both frozen).
  2. Train a small pixel decoder  z ∈ R^d → 64×64×3  on the training dataset.
  3. Generate a cartpole rollout with noisy actions (stored).
  4. Encoder path:   encode obs_t → z_t^enc → decode → PSNR_enc(t)
  5. Predictor path: encode obs_0 → z_0, then roll forward
                       z_{t+1} = predict(z_{t-W+1:t+1}, u_{t-W+1:t+1})
                     decode z_t^pred → PSNR_pred(t)
  6. 4-row figure:
       row 0 — original obs
       row 1 — encoder-decoded  (upper-bound: encoder-only error)
       row 2 — predictor-decoded (the thing under test)
       row 3 — |diff_pred| × 3

The gap between PSNR_enc and PSNR_pred directly measures predictor error,
independent of decoder quality.

Usage:
    python experiments/test_predictor_rollout.py \\
        --checkpoint results/jepa_v11_difenc/checkpoints/checkpoint_epoch0050.pt \\
        --config     configs/cartpole_jepa_v11_difenc.yaml \\
        --data       data/cartpole_visual_fs5_v4 \\
        --out        results/predictor_rollout_ep050.png \\
        --decoder-epochs 200 \\
        --rollout-steps 30 \\
        --action-std 3.0
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
        self.proj   = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 4  → 8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128,  64, 4, stride=2, padding=1),  # 8  → 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d( 64,  32, 4, stride=2, padding=1),  # 16 → 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d( 32,   3, 4, stride=2, padding=1),  # 32 → 64
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.proj(z).view(z.shape[0], 256, 4, 4)
        return self.decode(x)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return float('inf') if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, cfg: dict, device):
    from models.jepa import make_jepa
    raw      = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state    = raw['model_state'] if 'model_state' in raw else raw
    ckpt_cfg = raw.get('config', {})
    mc = cfg['model']
    for k in ('latent_dim', 'action_dim', 'action_latent_dim', 'action_encoder',
              'encoder_type', 'patch_size', 'frame_stack', 'use_frame_diff',
              'vit_embed_dim', 'vit_depth', 'vit_num_heads',
              'predictor_type', 'predictor_window',
              'predictor_embed_dim', 'predictor_depth',
              'predictor_num_heads', 'predictor_mlp_ratio'):
        if k in ckpt_cfg:
            mc[k] = ckpt_cfg[k]
    model = make_jepa(
        variant='E-full',
        latent_dim=int(mc.get('latent_dim', 8)),
        action_dim=int(mc.get('action_dim', 1)),
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


# ── Dataset helpers ─────────────────────────────────────────────────────────────
def _resize_np(img: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return np.array(Image.fromarray(img).resize((size, size), Image.BILINEAR))


def collect_z_obs_pairs(model, data_dir: str, device, use_frame_diff: bool,
                        max_pairs: int = 50_000, target_size: int = 64,
                        encode_batch: int = 512):
    """Return (Z, O) arrays for decoder training."""
    train_h5 = Path(data_dir) / 'train.hdf5'
    curr_buf, prev_buf, obs_buf = [], [], []
    with h5py.File(train_h5, 'r') as f:
        for ep_key in f['episodes']:
            ep  = f['episodes'][ep_key]
            obs = ep['observations'][:]
            T   = len(obs)
            for t in range(T):
                curr = _resize_np(obs[t], target_size)
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

    def to_batch(frames):
        arr = np.stack(frames).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)

    Z_list = []
    with torch.no_grad():
        for start in tqdm(range(0, N, encode_batch), desc='encoding'):
            end = min(start + encode_batch, N)
            z_b = model.encode_obs(to_batch(curr_buf[start:end]),
                                   to_batch(prev_buf[start:end]))
            Z_list.append(z_b.cpu().numpy())

    Z = np.concatenate(Z_list, axis=0).astype(np.float32)
    O = np.stack(obs_buf).astype(np.float32) / 255.0
    O = np.transpose(O, (0, 3, 1, 2))
    print(f'[data] collected {N:,} (z, obs) pairs')
    return Z, O


# ── Decoder training ───────────────────────────────────────────────────────────
def train_decoder(Z: np.ndarray, O: np.ndarray, device,
                  epochs: int = 200, batch_size: int = 512, lr: float = 3e-4):
    d        = Z.shape[1]
    img_size = O.shape[-1]
    decoder  = PixelDecoder(latent_dim=d, image_size=img_size).to(device)
    opt   = torch.optim.Adam(decoder.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.1)

    Z_t = torch.tensor(Z, dtype=torch.float32)
    O_t = torch.tensor(O, dtype=torch.float32)
    dl  = DataLoader(TensorDataset(Z_t, O_t), batch_size=batch_size, shuffle=True,
                     num_workers=2, pin_memory=True, drop_last=True)

    print(f'[decoder] training for {epochs} epochs  ({len(Z_t):,} pairs, batch={batch_size})')
    for ep in range(epochs):
        decoder.train()
        losses = []
        for zb, ob in dl:
            zb, ob = zb.to(device), ob.to(device)
            loss = F.mse_loss(decoder(zb), ob)
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
def generate_rollout(cfg: dict, n_steps: int, seed: int,
                     action_std: float, target_size: int):
    """Return (frames, actions_normalized).

    frames            : list of (H,W,3) uint8 observations, length = n_steps
    actions_normalized: np.ndarray (n_steps-1, action_dim) — actions u_t that
                        produced obs_{t+1} from obs_t, already divided by action_scale.
    """
    from envs.cartpole_visual import ContinuousCartpoleVisual
    from data.dataset import load_discrete_dataset_meta

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
    action_range = env_cfg.get('action_range', [-10, 10])
    action_scale = max(abs(float(action_range[0])), abs(float(action_range[1])))
    frame_skip   = int(env_cfg.get('frame_skip', 1))
    action_dim   = int(cfg['model'].get('action_dim', 1))  # 1 = ZOH, fs = multi-action

    rng  = np.random.RandomState(seed)
    x0   = rng.uniform(-0.05, 0.05, 4).astype(np.float32)
    obs0, _, _ = env.reset_to_state(x0)
    frames  = [_resize_np(obs0, target_size)]
    actions = []   # raw physical actions

    for _ in range(n_steps - 1):
        if action_std > 0:
            if action_dim > 1:
                # multi-action: one independent sub-action per frame_skip step
                u_raw = rng.uniform(-action_std, action_std, frame_skip).astype(np.float32)
            else:
                u_raw = float(rng.uniform(-action_std, action_std))
        else:
            u_raw = np.zeros(frame_skip, dtype=np.float32) if action_dim > 1 else 0.0

        obs, _, _, done, _ = env.step(u_raw)
        frames.append(_resize_np(obs, target_size))

        # Normalize and store
        u_norm = np.atleast_1d(np.asarray(u_raw, dtype=np.float32)) / action_scale
        if action_dim == 1:
            u_norm = u_norm[:1]  # scalar → (1,)
        actions.append(u_norm)

        if done:
            # Episode ended early; pad remaining steps at rest
            for _ in range(n_steps - 1 - len(actions)):
                obs0_eq, _, _ = env.reset_to_state(np.zeros(4, dtype=np.float32))
                frames.append(_resize_np(obs0_eq, target_size))
                actions.append(np.zeros(action_dim, dtype=np.float32))
            break

    actions = np.stack(actions, axis=0)  # (n_steps-1, action_dim)
    return frames, actions, action_scale


# ── Predictor rollout ──────────────────────────────────────────────────────────
@torch.no_grad()
def predictor_rollout(model, frames, actions_norm, device):
    """Return list of predicted latents z_t^pred for t=0..T-1.

    Follows the same window convention as Trainer.fit():
      z_win_buf starts as W copies of z_0
      u_win_buf starts as (W-1) zero vectors
      At each step k: append u_k, call predict(z_win[-W:], u_win[-W:]), append z_{k+1}
    """
    W        = model.config.predictor_window
    d        = model.config.latent_dim
    act_dim  = model.config.action_dim
    use_fd   = model.config.use_frame_diff

    def to_t(img_np):
        return (torch.from_numpy(img_np).float().permute(2, 0, 1)
                .unsqueeze(0).to(device) / 255.0)

    T = len(frames)

    # Encode z_0 from obs_0
    z0 = model.encode_obs(to_t(frames[0]), to_t(frames[0])).squeeze(0)  # (d,)

    z_preds = [z0]   # z_t^pred at each step

    z_win_buf = [z0] * W                                               # (d,) each
    u_win_buf = [torch.zeros(act_dim, device=device)] * (W - 1)       # (act_dim,) each

    actions_t = torch.tensor(actions_norm, dtype=torch.float32, device=device)  # (T-1, act_dim)

    for k in range(T - 1):
        u_k = actions_t[k]                                             # (act_dim,)
        u_win_buf.append(u_k)

        z_stack = torch.stack(z_win_buf[-W:], dim=0).unsqueeze(0)     # (1, W, d)
        u_stack = torch.stack(u_win_buf[-W:], dim=0).unsqueeze(0)     # (1, W, act_dim)

        z_next = model.predict(z_stack, u_stack).squeeze(0)            # (d,)

        z_win_buf.append(z_next)
        z_preds.append(z_next)

    return z_preds   # list of T tensors, each (d,)


# ── Encoder baseline ───────────────────────────────────────────────────────────
@torch.no_grad()
def encoder_rollout(model, frames, device):
    """Encode each obs_t independently. Returns list of (d,) tensors."""
    use_fd = model.config.use_frame_diff

    def to_t(img_np):
        return (torch.from_numpy(img_np).float().permute(2, 0, 1)
                .unsqueeze(0).to(device) / 255.0)

    z_encs = []
    for t, curr in enumerate(frames):
        prev = frames[t - 1] if t > 0 else curr
        z    = model.encode_obs(to_t(curr), to_t(prev)).squeeze(0)
        z_encs.append(z)
    return z_encs


# ── Decode and score ───────────────────────────────────────────────────────────
@torch.no_grad()
def decode_latents(decoder, z_list, device):
    """Decode a list of (d,) tensors. Returns list of (H,W,3) uint8 arrays."""
    imgs = []
    for z in z_list:
        rec = decoder(z.unsqueeze(0)).squeeze(0).permute(1, 2, 0).cpu().numpy()
        imgs.append((rec * 255).clip(0, 255).astype(np.uint8))
    return imgs


# ── Visualisation ──────────────────────────────────────────────────────────────
def make_figure(frames, enc_decoded, pred_decoded, psnrs_enc, psnrs_pred, out_path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n   = len(frames)
    idx = np.linspace(0, n - 1, min(n, 15), dtype=int)
    C   = len(idx)

    fig, axes = plt.subplots(4, C, figsize=(C * 1.5, 6.5))
    fig.suptitle(
        'Predictor rollout test\n'
        'row 1: original  |  row 2: encoder-decoded  |  '
        'row 3: predictor-decoded  |  row 4: |diff_pred|×3',
        fontsize=9,
    )

    row_labels = ['orig', 'enc', 'pred', 'diff×3']
    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(label, fontsize=8, rotation=0, labelpad=28, va='center')

    for col, i in enumerate(idx):
        orig = frames[i]
        enc  = enc_decoded[i]
        pred = pred_decoded[i]
        diff = np.abs(orig.astype(np.float32) - pred.astype(np.float32)) * 3
        diff = np.clip(diff, 0, 255).astype(np.uint8)

        axes[0, col].imshow(orig);  axes[0, col].axis('off')
        axes[1, col].imshow(enc);   axes[1, col].axis('off')
        axes[2, col].imshow(pred);  axes[2, col].axis('off')
        axes[3, col].imshow(diff);  axes[3, col].axis('off')

        axes[0, col].set_title(f't={i}', fontsize=7)
        axes[1, col].set_title(f'{psnrs_enc[i]:.1f}dB', fontsize=6, color='steelblue')
        axes[2, col].set_title(f'{psnrs_pred[i]:.1f}dB', fontsize=6, color='darkorange')

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
    p.add_argument('--out',             default='results/predictor_rollout.png')
    p.add_argument('--decoder-epochs',  type=int,   default=200)
    p.add_argument('--decoder-batch',   type=int,   default=512)
    p.add_argument('--max-pairs',       type=int,   default=50_000)
    p.add_argument('--rollout-steps',   type=int,   default=30)
    p.add_argument('--rollout-seed',    type=int,   default=7)
    p.add_argument('--action-std',      type=float, default=3.0,
                   help='Std of random actions in raw physical units (default 3 N). '
                        'Set 0 for passive rollout.')
    p.add_argument('--device',          default=None)
    args = p.parse_args()

    device = torch.device(args.device if args.device
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Device: {device}')

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    img_size = int(cfg['model'].get('image_size', 64))
    act_dim  = int(cfg['model'].get('action_dim', 1))

    # 1. Load model (frozen)
    print('[model] loading checkpoint ...')
    model   = load_model(args.checkpoint, cfg, device)
    d       = model.config.latent_dim
    W       = model.config.predictor_window
    use_fd  = model.config.use_frame_diff
    print(f'[model] latent_dim={d}  action_dim={act_dim}  '
          f'predictor_window={W}  use_frame_diff={use_fd}')

    # 2. Collect (z, obs) pairs and train pixel decoder
    Z, O = collect_z_obs_pairs(model, args.data, device, use_fd,
                               max_pairs=args.max_pairs, target_size=img_size)
    decoder = train_decoder(Z, O, device,
                            epochs=args.decoder_epochs, batch_size=args.decoder_batch)

    # 3. Generate rollout with random actions
    print(f'[rollout] generating {args.rollout_steps} steps '
          f'(seed={args.rollout_seed}, action_std={args.action_std} N) ...')
    frames, actions_norm, action_scale = generate_rollout(
        cfg, n_steps=args.rollout_steps,
        seed=args.rollout_seed, action_std=args.action_std, target_size=img_size)
    print(f'[rollout] {len(frames)} frames  '
          f'actions shape={actions_norm.shape}  '
          f'action_scale={action_scale}  '
          f'||u_norm||_∞={np.abs(actions_norm).max():.3f}')

    # 4. Encoder path: encode each obs_t independently
    print('[encoder] encoding ground-truth frames ...')
    z_encs  = encoder_rollout(model, frames, device)
    enc_dec = decode_latents(decoder, z_encs, device)

    # 5. Predictor path: roll forward from z_0 using stored actions
    print(f'[predictor] rolling out {len(frames)} steps from z_0 ...')
    z_preds  = predictor_rollout(model, frames, actions_norm, device)
    pred_dec = decode_latents(decoder, z_preds, device)

    # 6. Score
    psnrs_enc  = [psnr(frames[t], enc_dec[t])  for t in range(len(frames))]
    psnrs_pred = [psnr(frames[t], pred_dec[t]) for t in range(len(frames))]

    print(f'\n[results] encoder  mean PSNR = {np.mean(psnrs_enc):.2f} dB  '
          f'(range {min(psnrs_enc):.1f}–{max(psnrs_enc):.1f} dB)')
    print(f'[results] predictor mean PSNR = {np.mean(psnrs_pred):.2f} dB  '
          f'(range {min(psnrs_pred):.1f}–{max(psnrs_pred):.1f} dB)')
    print(f'[results] PSNR gap  enc–pred  = {np.mean(psnrs_enc) - np.mean(psnrs_pred):.2f} dB')
    print(f'\n[results] per-frame enc  PSNR: '
          + '  '.join(f't{i}:{v:.1f}' for i, v in enumerate(psnrs_enc)))
    print(f'[results] per-frame pred PSNR: '
          + '  '.join(f't{i}:{v:.1f}' for i, v in enumerate(psnrs_pred)))

    # 7. Plot
    make_figure(frames, enc_dec, pred_dec, psnrs_enc, psnrs_pred, args.out)


if __name__ == '__main__':
    main()
