"""Visualize AEWorldModel predictor quality.

Two diagnostics:
  1. One-step prediction: z_pred = f(z_t, u_t)  vs  z_actual = encoder(frame_{t+1})
     — shows whether the predictor can model a single dynamics step.
  2. Open-loop rollout: z_0 from encoder, then z_{t+1} = f(z_t, u_t) without
     re-encoding — shows how fast prediction errors accumulate.

Figure layout:
  Top grid  — Row 1: original frames
             — Row 2: one-step predicted frames (decode z_pred)
             — Row 3: open-loop rollout frames  (decode z_hat)
             — Row 4: open-loop pixel error (scaled)
  Bottom    — line plot: one-step latent error  vs  open-loop latent error

Usage:
    python experiments/visualize_ae_predictor.py \
        --checkpoint results/cartpole_ae_noLQR_seed43/model_final.pt \
        --config configs/cartpole_ae_noLQR.yaml \
        --output viz_ae_predictor.png
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

    ckpt  = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model_state', ckpt) if isinstance(ckpt, dict) else ckpt
    if 'encoder.net.0.weight' in state:
        frame_stack = state['encoder.net.0.weight'].shape[1] // 3

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
    p.add_argument('--output',     default='viz_ae_predictor.png')
    p.add_argument('--n-frames',   type=int, default=12)
    p.add_argument('--init-angle', type=float, default=0.3,
                   help='Initial pole angle in radians')
    p.add_argument('--action',     type=float, default=0.0,
                   help='Constant action applied at each step')
    p.add_argument('--device',     default=None)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model, frame_stack = _load_model(args.checkpoint, cfg, device)
    W = getattr(model.config, 'predictor_window', 1)
    d = model.config.latent_dim
    print(f'frame_stack={frame_stack}  predictor_window={W}  latent_dim={d}')

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

    # Collect frames with a constant action
    x0 = np.array([0., 0., args.init_angle, 0.], dtype=np.float32)
    obs, _, _ = env.reset_to_state(x0)
    frames = [obs.copy()]
    actions = []
    for _ in range(args.n_frames):
        obs, _, _, _, _ = env.step(args.action)
        frames.append(obs.copy())
        actions.append(args.action)

    # Equilibrium frame for frame_stack padding
    obs_eq, _, _ = env.reset_to_state(np.zeros(4, dtype=np.float32))
    env.close()

    # ── Helper: obs (H,W,3) uint8 → (1, 3*fs, H, W) tensor ─────────────────
    def _to_tensor(curr, prev=None):
        t = torch.from_numpy(curr).float().permute(2, 0, 1) / 255.0
        if frame_stack > 1:
            p = obs_eq if prev is None else prev
            tp = torch.from_numpy(p).float().permute(2, 0, 1) / 255.0
            t = torch.cat([tp, t], dim=0)
        return t.unsqueeze(0).to(device)

    # ── Encode all frames to get ground-truth latents ────────────────────────
    z_actual = []
    with torch.no_grad():
        prev = None
        for frm in frames:
            z = model.encoder(_to_tensor(frm, prev))  # (1, d)
            z_actual.append(z[0])
            prev = frm
    # z_actual[i] = encoder(frames[i])

    # ── One-step prediction ──────────────────────────────────────────────────
    # z_pred[t] = predictor(z_actual[t-W+1..t], u[t])  ≈  z_actual[t+1]
    # We compare z_pred[t] with z_actual[t+1].
    # z_eq used to pad the window at the start.
    z_eq_enc = z_actual[0].detach().clone()  # approximate equilibrium in latent

    def _make_window(z_list, t_last):
        """Return (1, W, d) latent window ending at t_last."""
        buf = []
        for k in range(W):
            idx = t_last - (W - 1 - k)
            buf.append(z_list[idx] if idx >= 0 else z_eq_enc)
        return torch.stack(buf, dim=0).unsqueeze(0)  # (1, W, d)

    one_step_z = []   # predicted z for t+1
    one_step_err = [] # ||z_pred - z_actual[t+1]||
    with torch.no_grad():
        for t in range(len(actions)):                         # t = 0..n_frames-1
            z_win = _make_window(z_actual, t)                 # window ending at z[t]
            u_win = torch.full((1, W, 1), args.action, device=device)
            z_pred = model.predict(z_win, u_win)              # (1, d)
            one_step_z.append(z_pred[0])
            err = float((z_pred[0] - z_actual[t + 1]).norm().cpu())
            one_step_err.append(err)

    # ── Open-loop rollout ────────────────────────────────────────────────────
    # Start from z_actual[0]; never re-encode.
    z_hat_list = [z_actual[0].clone()]   # index 0..n_frames-1 (same len as frames)
    open_loop_err = [0.0]                # error vs z_actual[i]
    with torch.no_grad():
        for t in range(len(actions)):
            z_win = _make_window(z_hat_list, t)
            u_win = torch.full((1, W, 1), args.action, device=device)
            z_next = model.predict(z_win, u_win)
            z_hat_list.append(z_next[0])
            err = float((z_next[0] - z_actual[t + 1]).norm().cpu())
            open_loop_err.append(err)

    # ── Decode predictions back to pixel space ───────────────────────────────
    with torch.no_grad():
        def _decode(z):
            return model.decode(z.unsqueeze(0))[0].cpu().clamp(0, 1).permute(1, 2, 0).numpy()

        frames_orig    = [f.astype(np.float32) / 255.0 for f in frames[1:]]  # t+1 targets
        frames_onestep = [_decode(z) for z in one_step_z]
        frames_openloop= [_decode(z) for z in z_hat_list[1:]]                # t=1..n_frames

    n = len(actions)  # number of prediction steps

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(n * 1.6, 9))
    gs  = fig.add_gridspec(5, n, height_ratios=[1.2, 1.2, 1.2, 1.2, 1.5], hspace=0.05, wspace=0.02)

    row_labels = ['Original (t+1)', '1-step pred', 'Open-loop', 'OL pixel error']
    for col in range(n):
        orig  = frames_orig[col]
        ost   = frames_onestep[col]
        ol    = frames_openloop[col]
        err   = np.abs(orig - ol)

        imgs = [orig, ost, ol, err / (err.max() + 1e-6)]
        for row, img in enumerate(imgs):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(np.clip(img, 0, 1))
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=8, labelpad=4)
            if row == 0:
                ax.set_title(f't+{col+1}', fontsize=7)

    # Error line plot
    ax_err = fig.add_subplot(gs[4, :])
    ts = np.arange(1, n + 1)
    ax_err.plot(ts, one_step_err,  marker='o', ms=4, label='1-step latent error')
    ax_err.plot(ts, open_loop_err[1:], marker='s', ms=4, label='Open-loop latent error')
    ax_err.set_xlabel('Prediction step')
    ax_err.set_ylabel('||z_pred − z_actual||')
    ax_err.legend(fontsize=8)
    ax_err.grid(True, alpha=0.3)
    ax_err.set_title('Latent prediction error vs time', fontsize=9)

    ckpt_name = Path(args.checkpoint).parent.name
    fig.suptitle(
        f'AE predictor quality  |  {ckpt_name}\n'
        f'θ₀={np.degrees(args.init_angle):.1f}°, u={args.action:.1f}',
        fontsize=11, fontweight='bold', y=0.99)

    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f'[viz] Saved {args.output}')

    # ── Summary stats ────────────────────────────────────────────────────────
    print(f'\n{"Step":>4}  {"1-step err":>12}  {"Open-loop err":>14}')
    print('-' * 36)
    for t in range(n):
        print(f'{t+1:>4}  {one_step_err[t]:>12.5f}  {open_loop_err[t+1]:>14.5f}')


if __name__ == '__main__':
    main()
