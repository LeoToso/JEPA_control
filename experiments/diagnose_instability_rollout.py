"""Apples-to-apples instability comparison: GT vs learned model decoded to physical state.

Starting from x₀ = (0, 0, ε, 0) with u=0:
  GT linear    : x_{t+1} = A_star^frame_skip @ x_t
  GT nonlinear : actual ContinuousCartpoleVisual env with u=0
  Learned      : encoder → predictor (u=0) → state_head decode → physical θ

Both θ(t) curves are in radians. Matching slopes on the semilog plot means
the model correctly captures the instability growth rate.

Usage:
    python experiments/diagnose_instability_rollout.py \\
        --checkpoint results/jepa_sf_w3_fs5_finetune_v3/checkpoints/checkpoint_epoch0050.pt \\
        --config     configs/cartpole_jepa_sf_w3_fs5_finetune.yaml \\
        --data       data/cartpole_visual_fs5_passive_long \\
        --out        results/jepa_sf_w3_fs5_finetune_v3/instability_rollout_ep50.png \\
        --device     cuda
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _load_norm_stats(data_dir):
    try:
        from data.dataset import load_discrete_dataset_meta
        meta = load_discrete_dataset_meta(data_dir)
        return meta.get('state_mean'), meta.get('state_std')
    except Exception as e:
        print(f'[warn] Could not load norm stats: {e}')
        return None, None


def _load_model(ckpt_path, cfg, device):
    import torch.nn as nn
    from models.jepa import make_jepa
    env_cfg   = cfg['environment']
    model_cfg = dict(cfg['model'])

    raw   = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = raw.get('model_state', raw.get('model_state_dict', raw))
    if isinstance(raw, dict) and 'config' in raw:
        for k in ('latent_dim', 'action_latent_dim', 'action_encoder',
                  'encoder_type', 'patch_size', 'frame_stack',
                  'vit_embed_dim', 'vit_depth', 'vit_num_heads',
                  'predictor_type', 'predictor_hidden_dim', 'predictor_n_layers',
                  'predictor_window', 'predictor_embed_dim', 'predictor_depth',
                  'predictor_num_heads', 'predictor_mlp_ratio'):
            if k in raw['config']:
                model_cfg[k] = raw['config'][k]

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
        predictor_window=int(model_cfg.get('predictor_window', 1)),
        predictor_embed_dim=int(model_cfg.get('predictor_embed_dim', 128)),
        predictor_depth=int(model_cfg.get('predictor_depth', 4)),
        predictor_num_heads=int(model_cfg.get('predictor_num_heads', 4)),
        predictor_mlp_ratio=float(model_cfg.get('predictor_mlp_ratio', 4.0)),
    )
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    # state_head is saved separately in the checkpoint (not part of model)
    # It operates on L2-normalized z and predicts normalized physical state.
    state_head = None
    if isinstance(raw, dict) and 'state_head_state' in raw:
        d_lat = int(model_cfg['latent_dim'])
        state_head = nn.Linear(d_lat, 4).to(device)
        state_head.load_state_dict(raw['state_head_state'])
        state_head.eval()
        print('[model] state_head loaded from checkpoint')
    else:
        print('[warn] No state_head_state in checkpoint — decode unavailable')

    W = int(model_cfg.get('predictor_window', 1))
    fs = int(model_cfg.get('frame_stack', 1))
    return model, state_head, fs, W


def _encode(model, obs, device, frame_stack):
    t = torch.from_numpy(obs).float().permute(2, 0, 1)[None].to(device) / 255.0
    if frame_stack > 1:
        t = torch.cat([t, t], dim=1)
    with torch.no_grad():
        return model.encoder(t)   # (1, d)


def _predict(model, z_history, device):
    """z_history: list of (1,d) tensors, oldest first, len=W. Returns (1,d).

    Predictor signature: forward(z, a)
        z: (B, W*latent_dim)  — flattened window
        a: (B, W*action_dim)  — flattened action embeddings
    """
    with torch.no_grad():
        W      = len(z_history)
        z_flat = torch.cat(z_history, dim=1)            # (1, W*d)
        a_zero = torch.zeros(1, 1, device=device)
        a_emb  = model.action_encoder(a_zero)           # (1, d_a) for u=0
        a_flat = a_emb.repeat(1, W)                     # (1, W*d_a)
        return model.predictor(z_flat, a_flat)          # (1, d)


def _decode(state_head, z, state_mean, state_std):
    """state_head(normalize(z)) → denormalized physical state (4,)."""
    import torch.nn.functional as F
    with torch.no_grad():
        z_n   = F.normalize(z, dim=-1)                  # unit sphere, matches training
        s_norm = state_head(z_n).cpu().numpy()[0]        # (4,) normalized
    if state_mean is not None and state_std is not None:
        return s_norm * state_std + state_mean
    return s_norm


def _fit_growth_rate(thetas, n_fit):
    """Fit exponential |θ(t)| ~ A·λ^t, return λ (per latent step)."""
    thetas = np.abs(thetas[:n_fit + 1])
    valid  = thetas > 1e-8
    if valid.sum() < 3:
        return float('nan')
    t      = np.where(valid)[0].astype(float)
    log_th = np.log(thetas[valid])
    coeffs = np.polyfit(t, log_th, 1)
    return float(np.exp(coeffs[0]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config',     required=True)
    p.add_argument('--data',       required=True,
                   help='Dataset dir for normalization stats')
    p.add_argument('--eps',        type=float, default=0.05,
                   help='Initial pole angle (rad) — default 0.05 (~3 deg)')
    p.add_argument('--n-steps',    type=int,   default=20,
                   help='Latent steps to roll out (each = frame_skip physics steps)')
    p.add_argument('--n-eps',      type=int,   default=5,
                   help='Number of eps values to sweep (shows family of curves)')
    p.add_argument('--fit-steps',  type=int,   default=5,
                   help='How many model-prediction steps to use for the growth-rate fit '
                        '(default 5 = training horizon; should match --horizon used during training)')
    p.add_argument('--plot-steps', type=int,   default=None,
                   help='Clip the x-axis of the plot to this many total latent steps '
                        '(default: show all n-steps). Set to warmup_end+fit_steps to show '
                        'only the trained horizon.')
    p.add_argument('--out',        default=None)
    p.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg['environment']

    state_mean, state_std = _load_norm_stats(args.data)
    if state_mean is not None:
        print(f'[norm] mean={np.round(state_mean, 3)}  std={np.round(state_std, 3)}')

    model, state_head, frame_stack, W = _load_model(args.checkpoint, cfg, device)
    print(f'[model] frame_stack={frame_stack}  predictor_window={W}')
    if state_head is None:
        raise RuntimeError('No state_head_state in checkpoint — train with lambda_state > 0')

    from envs.cartpole_visual import ContinuousCartpoleVisual
    frame_skip = int(env_cfg.get('frame_skip', 1))
    env = ContinuousCartpoleVisual(
        frame_skip=frame_skip,
        image_size=int(env_cfg['image_size']),
        mass_cart=env_cfg['mass_cart'], mass_pole=env_cfg['mass_pole'],
        pole_length=env_cfg['pole_length'], gravity=env_cfg['gravity'],
        dt=env_cfg['dt'], seed=0,
    )

    from ground_truth.cartpole_gt import CartpoleGroundTruth
    gt = CartpoleGroundTruth(
        mass_cart=env_cfg['mass_cart'], mass_pole=env_cfg['mass_pole'],
        pole_length=env_cfg['pole_length'], gravity=env_cfg['gravity'],
        dt=env_cfg['dt'],
    )
    A_step  = np.linalg.matrix_power(gt.A_star, frame_skip)
    rho_gt  = float(np.max(np.abs(np.linalg.eigvals(A_step))))
    print(f'[GT] rho(A_star^{frame_skip}) = {rho_gt:.4f}')

    # Encode equilibrium → z★
    obs_eq, _, _ = env.reset_to_state(np.zeros(4, dtype=np.float32))
    z_star = _encode(model, obs_eq, device, frame_stack)

    # ── single ε: detailed trajectory table ──────────────────────────────────
    eps = args.eps
    x0  = np.array([0.0, 0.0, eps, 0.0], dtype=np.float32)

    # GT linear
    gt_lin = [x0.copy()]
    x = x0.copy().astype(float)
    for _ in range(args.n_steps):
        x = A_step @ x
        gt_lin.append(x.copy())
    gt_lin = np.array(gt_lin)

    # GT nonlinear
    obs0, s0, _ = env.reset_to_state(x0)
    gt_nlin = [s0.copy()]
    for _ in range(args.n_steps):
        obs_next, s_next, _, done, _ = env.step(0.0)
        gt_nlin.append(s_next.copy())
        if done:
            break
    gt_nlin = np.array(gt_nlin)

    # Learned model — warm-start: run real env for W steps to build
    # in-distribution context, then hand off to model for prediction.
    # This avoids the out-of-distribution [z★,z★,z₀] initialization.
    obs_cur, s_cur, _ = env.reset_to_state(x0)
    real_history = []   # real encoded latents for W warm-start steps
    warmup_states = [s_cur.copy()]
    for _ in range(W):
        z_cur = _encode(model, obs_cur, device, frame_stack)
        real_history.append(z_cur)
        obs_next, s_next, _, done, _ = env.step(0.0)
        warmup_states.append(s_next.copy())
        obs_cur = obs_next
        if done:
            break

    # Decode the warm-start latents to show the initial trajectory
    lrn_warmup = [_decode(state_head, z, state_mean, state_std) for z in real_history]

    # From step W onward: pure model prediction (u=0)
    history = list(real_history[-W:])   # in-distribution context
    lrn = lrn_warmup[:]
    for _ in range(args.n_steps):
        z_next = _predict(model, history, device)
        lrn.append(_decode(state_head, z_next, state_mean, state_std))
        history = history[1:] + [z_next]
    lrn       = np.array(lrn)
    warmup_end = len(lrn_warmup)  # index where model takes over

    # Growth-rate fits.
    # GT: fit from step 0 over the same number of steps as the trained horizon.
    # Learned: fit only over the model-prediction phase (post warm-start),
    #          limited to fit_steps so we don't penalise the model for steps
    #          it was never trained to predict.
    n_fit     = args.fit_steps
    rate_gl   = _fit_growth_rate(gt_lin[:, 2],  n_fit)
    rate_gn   = _fit_growth_rate(gt_nlin[:, 2], n_fit)
    # Learned slice: [warmup_end .. warmup_end + n_fit]
    lrn_fit_slice = lrn[warmup_end:warmup_end + n_fit + 1, 2]
    rate_lrn  = _fit_growth_rate(lrn_fit_slice, n_fit)

    print(f'\n{"step":>5}  {"GT_linear θ (°)":>16}  {"GT_nonlin θ (°)":>16}  {"Learned θ (°)":>14}  note')
    print('-' * 70)
    n_rep = min(len(gt_lin), len(gt_nlin), len(lrn))
    for t in range(n_rep):
        note = '← warm-start (real enc)' if t < warmup_end else '← model pred'
        print(f'{t:5d}  {np.degrees(gt_lin[t, 2]):16.3f}  '
              f'{np.degrees(gt_nlin[t, 2]):16.3f}  '
              f'{np.degrees(lrn[t, 2]):14.3f}  {note}')

    fit_start = warmup_end
    fit_end   = warmup_end + n_fit
    print(f'\n[growth rate per latent step — GT: steps 0–{n_fit}, Learned: model steps {fit_start}–{fit_end}]')
    print(f'  GT linear    : {rate_gl:.4f}  (expected {rho_gt:.4f})')
    print(f'  GT nonlinear : {rate_gn:.4f}')
    print(f'  Learned      : {rate_lrn:.4f}')
    ratio = rate_lrn / rho_gt if rho_gt > 0 else float('nan')
    print(f'  Learned/GT   : {ratio:.3f}  ({ratio*100:.1f}% of GT growth rate)')

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_end = args.plot_steps if args.plot_steps is not None else args.n_steps + warmup_end

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    t_gl  = np.arange(len(gt_lin))
    t_gn  = np.arange(len(gt_nlin))
    t_lrn = np.arange(len(lrn))

    # Panel 1: semilog
    ax = axes[0]
    ax.semilogy(t_gl,  np.abs(gt_lin[:, 2]),  'g-',   lw=2,    label=f'GT linear (λ={rate_gl:.3f})')
    ax.semilogy(t_gn,  np.abs(gt_nlin[:, 2]), 'b--',  lw=2,    label=f'GT nonlinear (λ={rate_gn:.3f})')
    ax.semilogy(t_lrn[:warmup_end], np.abs(lrn[:warmup_end, 2]), 'r--', lw=1.2, alpha=0.5, label=f'Learned warm-start (real enc)')
    ax.semilogy(t_lrn[warmup_end-1:], np.abs(lrn[warmup_end-1:, 2]), 'r-o', lw=1.5, ms=4, label=f'Learned model pred (λ={rate_lrn:.3f}, fit steps {fit_start}–{fit_end})')
    ax.axvline(warmup_end - 1, color='orange', lw=0.8, linestyle=':', label=f'model takes over (step {warmup_end-1})')
    ax.axhline(np.pi / 2, color='gray', lw=0.8, linestyle=':', label='90°')
    ax.set_xlabel(f'Latent step  (×{frame_skip} physics steps = ×{frame_skip * env_cfg["dt"]:.3f}s)')
    ax.set_ylabel('|θ| (rad)')
    ax.set_title(f'Passive divergence — semilog  (θ₀={np.degrees(eps):.1f}°, u=0)')
    ax.set_xlim(0, plot_end)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')

    # Panel 2: linear
    ax = axes[1]
    ax.plot(t_gl,  np.degrees(gt_lin[:, 2]),  'g-',   lw=2,   label='GT linear')
    ax.plot(t_gn,  np.degrees(gt_nlin[:, 2]), 'b--',  lw=2,   label='GT nonlinear')
    ax.axvline(warmup_end - 1, color='orange', lw=0.8, linestyle=':', label=f'model takes over')
    ax.plot(t_lrn, np.degrees(lrn[:, 2]),     'r-o',  lw=1.5, ms=4, label='Learned decoded')
    ax.set_xlabel(f'Latent step  (×{frame_skip} physics steps)')
    ax.set_ylabel('θ (degrees)')
    ax.set_title('Pole angle — linear scale')
    ax.set_xlim(0, plot_end)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ckpt_name = Path(args.checkpoint).stem
    fig.suptitle(f'Instability Rollout — {ckpt_name}', fontsize=11)
    fig.tight_layout()

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150, bbox_inches='tight')
        print(f'\nPlot → {out}')
    else:
        plt.show()

    env.close()


if __name__ == '__main__':
    main()
