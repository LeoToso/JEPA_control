"""Visualize AEWorldModel reconstruction quality.

For a sequence of cartpole frames, show:
  Row 1: original frames
  Row 2: reconstructed frames (decoder(encoder(obs)))
  Row 3: absolute pixel error (scaled)

Usage:
    python experiments/visualize_ae_recon.py \
        --checkpoint results/cartpole_ae_noLQR_seed43/model_final.pt \
        --config configs/cartpole_ae_noLQR.yaml \
        --output viz_ae_recon.png
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml


def _load_model(ckpt_path: str, cfg: dict, device):
    from models.jepa import JEPAConfig
    from models.autoencoder import AEWorldModel

    model_cfg = cfg['model']
    env_cfg   = cfg['environment']
    frame_stack = int(model_cfg.get('frame_stack', 1))

    # Auto-detect frame_stack from checkpoint weights
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model_state', ckpt) if isinstance(ckpt, dict) else ckpt
    if 'encoder.net.0.weight' in state:
        in_chans_ckpt = state['encoder.net.0.weight'].shape[1]
        frame_stack = in_chans_ckpt // 3

    jepa_cfg = JEPAConfig(
        latent_dim=int(model_cfg.get('latent_dim', 8)),
        action_latent_dim=int(model_cfg.get('action_latent_dim', 1)),
        action_encoder=model_cfg.get('action_encoder', 'linear'),
        encoder_type=model_cfg.get('encoder_type', 'cnn'),
        image_size=int(env_cfg.get('image_size', 64)),
        patch_size=int(model_cfg.get('patch_size', 8)),
        frame_stack=frame_stack,
        vit_embed_dim=int(model_cfg.get('vit_embed_dim', 128)),
        vit_depth=int(model_cfg.get('vit_depth', 4)),
        vit_num_heads=int(model_cfg.get('vit_num_heads', 4)),
        predictor_hidden_dim=int(model_cfg.get('predictor_hidden_dim', 64)),
        predictor_n_layers=int(model_cfg.get('predictor_n_layers', 2)),
        predictor_window=int(model_cfg.get('predictor_window', 1)),
    )
    model = AEWorldModel(jepa_cfg)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, frame_stack


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config',     default='configs/cartpole_ae_noLQR.yaml')
    p.add_argument('--output',     default='viz_ae_recon.png')
    p.add_argument('--n-frames',   type=int, default=10,
                   help='Number of frames in the sequence')
    p.add_argument('--init-angle', type=float, default=0.3,
                   help='Initial pole angle in radians')
    p.add_argument('--device',     default=None)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model, frame_stack = _load_model(args.checkpoint, cfg, device)
    env_cfg = cfg['environment']

    from envs.cartpole_visual import ContinuousCartpoleVisual
    env = ContinuousCartpoleVisual(
        frame_skip=env_cfg.get('frame_skip', 1),
        image_size=env_cfg.get('image_size', 64),
        mass_cart=env_cfg.get('mass_cart', 1.0),
        mass_pole=env_cfg.get('mass_pole', 0.1),
        pole_length=env_cfg.get('pole_length', 0.5),
        gravity=env_cfg.get('gravity', 9.8),
        action_range=tuple(env_cfg.get('action_range', (-10, 10))),
    )

    # Collect a sequence of frames: start at init_angle, let it fall freely (u=0)
    x0 = np.array([0., 0., args.init_angle, 0.], dtype=np.float32)
    obs, _, _ = env.reset_to_state(x0)
    frames_orig = [obs.copy()]
    for _ in range(args.n_frames - 1):
        obs, _, _, _, _ = env.step(0.0)
        frames_orig.append(obs.copy())

    # Also collect equilibrium obs for frame_stack=2 (prev frame)
    obs_eq, _, _ = env.reset_to_state(np.zeros(4, dtype=np.float32))
    env.close()

    # Reconstruct each frame
    def _make_input(obs_curr, obs_prev=None):
        """(H,W,3) uint8 → (1, 3*fs, H, W) float32."""
        t = torch.from_numpy(obs_curr).float().permute(2,0,1) / 255.0
        if frame_stack > 1:
            if obs_prev is None:
                t_prev = torch.from_numpy(obs_eq).float().permute(2,0,1) / 255.0
            else:
                t_prev = torch.from_numpy(obs_prev).float().permute(2,0,1) / 255.0
            t = torch.cat([t_prev, t], dim=0)
        return t.unsqueeze(0).to(device)

    frames_recon = []
    with torch.no_grad():
        prev = None
        for obs in frames_orig:
            inp = _make_input(obs, prev)
            z   = model.encoder(inp)          # (1, d)
            rec = model.decode(z)             # (1, 3, H, W)  — current frame only
            rec_np = rec[0].cpu().clamp(0, 1).permute(1,2,0).numpy()
            frames_recon.append(rec_np)
            prev = obs

    # ── Plot ─────────────────────────────────────────────────────────────────
    n = args.n_frames
    fig, axes = plt.subplots(3, n, figsize=(n * 1.6, 5))
    fig.suptitle(
        f'AE reconstruction  |  {Path(args.checkpoint).parent.name}\n'
        f'θ₀={np.degrees(args.init_angle):.1f}°, free-fall (u=0)',
        fontsize=11, fontweight='bold')

    for col in range(n):
        orig  = frames_orig[col].astype(np.float32) / 255.0
        recon = frames_recon[col]
        err   = np.abs(orig - recon)

        for row, (img, title) in enumerate([
            (orig,                    f't={col}'),
            (recon,                   ''),
            (err / (err.max() + 1e-6), ''),
        ]):
            ax = axes[row, col]
            ax.imshow(np.clip(img, 0, 1))
            ax.axis('off')
            if col == 0:
                row_labels = ['Original', 'Recon', 'Error (scaled)']
                ax.set_ylabel(row_labels[row], fontsize=9, labelpad=4)
            if row == 0:
                ax.set_title(title, fontsize=8)

    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f'[viz] Saved {args.output}')

    # Print per-frame MSE
    print('\nPer-frame reconstruction MSE:')
    for i, (o, r) in enumerate(zip(frames_orig, frames_recon)):
        mse = float(np.mean((o.astype(np.float32)/255.0 - r)**2))
        print(f'  t={i}: MSE={mse:.5f}')


if __name__ == '__main__':
    main()
