"""Diagnose whether the learned predictor captures the instability.

Runs a passive (u=0) rollout from a near-equilibrium state, records the
true physical theta, the latent z from the online encoder, and what the
predictor says z should be (open-loop prediction with u=0).

Prints per-step comparison and plots if --out is given.

Usage:
    python experiments/diag_predictor_divergence.py \
        --checkpoint results/jepa_v2_mixed2/checkpoints/checkpoint_epoch0450.pt \
        --config     configs/cartpole_jepa_pred_state_random.yaml \
        --theta0 0.10 --steps 30 \
        --out results/jepa_v2_mixed2/diag_divergence.png
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config',     required=True)
    p.add_argument('--theta0',     type=float, default=0.10)
    p.add_argument('--steps',      type=int,   default=30)
    p.add_argument('--out',        default=None)
    p.add_argument('--device',     default=None)
    args = p.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_cfg   = cfg['environment']
    model_cfg = cfg['model']
    frame_skip  = int(env_cfg.get('frame_skip', 1))

    raw = torch.load(args.checkpoint, map_location=device)
    state = raw.get('model_state', raw.get('model_state_dict', raw))
    if isinstance(raw, dict) and 'config' in raw:
        for k, v in raw['config'].items():
            if hasattr(type('_', (), {}), k):
                pass
            model_cfg[k] = model_cfg.get(k, v)
        _ckpt_cfg = raw['config']
        for k in ('latent_dim', 'action_latent_dim', 'action_encoder',
                  'encoder_type', 'patch_size', 'frame_stack',
                  'vit_embed_dim', 'vit_depth', 'vit_num_heads',
                  'predictor_type', 'predictor_hidden_dim', 'predictor_n_layers',
                  'predictor_window', 'predictor_embed_dim', 'predictor_depth',
                  'predictor_num_heads', 'predictor_mlp_ratio'):
            if k in _ckpt_cfg:
                model_cfg[k] = _ckpt_cfg[k]

    from models.jepa import make_jepa
    model = make_jepa(
        variant='E-full',
        latent_dim=int(model_cfg['latent_dim']),
        action_latent_dim=int(model_cfg.get('action_latent_dim', 1)),
        action_encoder=model_cfg.get('action_encoder', 'linear'),
        encoder_type=model_cfg.get('encoder_type', 'vit'),
        image_size=int(env_cfg['image_size']),
        patch_size=int(model_cfg.get('patch_size', 8)),
        frame_stack=int(model_cfg.get('frame_stack', 1)),
        vit_embed_dim=int(model_cfg.get('vit_embed_dim', 128)),
        vit_depth=int(model_cfg.get('vit_depth', 4)),
        vit_num_heads=int(model_cfg.get('vit_num_heads', 4)),
        predictor_type=model_cfg.get('predictor_type', 'mlp'),
        predictor_hidden_dim=int(model_cfg.get('predictor_hidden_dim', 64)),
        predictor_n_layers=int(model_cfg.get('predictor_n_layers', 3)),
        predictor_window=int(model_cfg.get('predictor_window', 5)),
        predictor_embed_dim=int(model_cfg.get('predictor_embed_dim', 128)),
        predictor_depth=int(model_cfg.get('predictor_depth', 4)),
        predictor_num_heads=int(model_cfg.get('predictor_num_heads', 4)),
        predictor_mlp_ratio=float(model_cfg.get('predictor_mlp_ratio', 4.0)),
    )
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    frame_stack = int(model_cfg.get('frame_stack', 1))
    W = getattr(model.config, 'predictor_window', 1)

    from envs.cartpole_visual import ContinuousCartpoleVisual
    env = ContinuousCartpoleVisual(
        frame_skip=frame_skip, image_size=int(env_cfg['image_size']),
        mass_cart=env_cfg['mass_cart'], mass_pole=env_cfg['mass_pole'],
        pole_length=env_cfg['pole_length'], gravity=env_cfg['gravity'],
        dt=env_cfg['dt'], seed=0,
    )

    def encode(obs_np, prev_obs_np=None):
        curr = torch.from_numpy(obs_np).float().permute(2, 0, 1)[None].to(device) / 255.0
        if frame_stack > 1:
            prev = (curr if prev_obs_np is None else
                    torch.from_numpy(prev_obs_np).float().permute(2, 0, 1)[None].to(device) / 255.0)
            inp = torch.cat([prev, curr], dim=1)
        else:
            inp = curr
        with torch.no_grad():
            return model.encoder(inp).cpu().numpy()[0]

    # Equilibrium z_star
    obs_eq, _, _ = env.reset_to_state(np.zeros(4, dtype=np.float32))
    z_star = encode(obs_eq)
    print(f'z_star: {np.round(z_star, 3)}  ||z_star||={np.linalg.norm(z_star):.3f}')

    # Start at perturbed state
    x0 = np.array([0., 0., args.theta0, 0.], dtype=np.float32)
    obs, state_phys, _ = env.reset_to_state(x0)
    prev_obs = obs.copy()

    # Real rollout (u=0) with encoder at each step
    real_z   = []  # (steps, d)
    real_phys = []  # (steps, 4)
    pred_z   = []  # open-loop model predictions

    # CEM-style window buffer for predictor
    z0 = encode(obs, None)
    z_win_buf = [torch.tensor(z0, device=device).unsqueeze(0)] * W  # W copies of z0
    u_win_buf = [torch.zeros(1, 1, device=device)] * (W - 1)  # W-1 zero actions

    real_z.append(z0.copy())
    real_phys.append(state_phys.copy())

    # First pred: from z_star repeated (cold start reflects equilibrium)
    z_win_eq = torch.tensor(z_star, device=device).float().unsqueeze(0).unsqueeze(1).expand(1, W, -1)
    u_win_eq = torch.zeros(1, W, 1, device=device)
    with torch.no_grad():
        z_pred_init = model.predict(z_win_eq, u_win_eq).cpu().numpy()[0]
    pred_z.append(z0.copy())  # at t=0 prediction = initial state

    print(f'\n{"step":>4}  {"theta_real":>12}  {"||z-z*||":>10}  {"||zpred-z*||":>12}  '
          f'{"||zpred-zreal||":>16}')
    print('-' * 60)
    print(f'{0:>4}  {state_phys[2]:>12.4f}  {np.linalg.norm(z0-z_star):>10.4f}  '
          f'{"-":>12}  {"-":>16}')

    z_prev_real = z0.copy()
    for step in range(1, args.steps + 1):
        # Apply u=0 to real environment
        obs_next, state_next, _, done, _ = env.step(0.0)
        z_real = encode(obs_next, obs)

        # Predictor: use the PREVIOUS W real latents (oracle window, best case).
        # Pad with z0 copies if fewer than W real latents are available.
        past = real_z[:]  # all latents recorded so far (not including current z_real)
        while len(past) < W:
            past = [past[0]] + past  # prepend oldest
        win_entries = past[-W:]  # W most recent past latents
        z_win_real = torch.stack(
            [torch.tensor(zz, device=device).float().unsqueeze(0)
             for zz in win_entries], dim=1)  # (1, W, d)
        u_win_zero = torch.zeros(1, W, 1, device=device)
        with torch.no_grad():
            z_pred_next = model.predict(z_win_real, u_win_zero).cpu().numpy()[0]

        real_z.append(z_real.copy())
        real_phys.append(state_next.copy())
        pred_z.append(z_pred_next.copy())

        print(f'{step:>4}  {state_next[2]:>12.4f}  {np.linalg.norm(z_real-z_star):>10.4f}  '
              f'{np.linalg.norm(z_pred_next-z_star):>12.4f}  '
              f'{np.linalg.norm(z_pred_next-z_real):>16.4f}')

        z_prev_real = z_real.copy()
        obs = obs_next
        state_phys = state_next
        if done:
            print(f'[done at step {step}]')
            break

    env.close()

    if args.out:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        Z  = np.array(real_z)    # (T, d)
        Zp = np.array(pred_z)    # (T, d)
        Xp = np.array(real_phys) # (T, 4)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        T = len(Xp)

        axes[0].plot(Xp[:, 2], 'r-', label='theta (real)')
        axes[0].axhline(0, color='k', lw=0.5)
        axes[0].set_title('Physical theta (u=0)')
        axes[0].set_xlabel('step'); axes[0].set_ylabel('theta (rad)')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(np.linalg.norm(Z - z_star, axis=1),  'b-',  label='||z_real - z*||')
        axes[1].plot(np.linalg.norm(Zp - z_star, axis=1), 'g--', label='||z_pred - z*||')
        axes[1].set_title('Latent distance from equilibrium')
        axes[1].set_xlabel('step'); axes[1].set_ylabel('||z - z*||')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        axes[2].plot(np.linalg.norm(Zp - Z, axis=1), 'm-')
        axes[2].set_title('One-step prediction error ||z_pred - z_real||')
        axes[2].set_xlabel('step')
        axes[2].grid(alpha=0.3)

        fig.suptitle(f'Predictor divergence diagnostic  theta0={args.theta0}', fontsize=11)
        fig.tight_layout()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150, bbox_inches='tight')
        print(f'\nSaved → {out}')


if __name__ == '__main__':
    main()
