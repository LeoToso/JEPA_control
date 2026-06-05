"""Train an Autoencoder World Model with DMD-based pixel prediction.

Follows Bounou et al. (NeurIPS 2021) — key design decisions:

1. 2-frame encoder input (frame_stack=2): encoder receives [obs_{t-1}, obs_t]
   → latent z encodes both position AND velocity.  Single-frame input cannot
   encode ẋ or θ̇, making prediction under-determined.

2. Dynamics estimated per-batch via least-squares DMD (not a learned parameter):
   (A, B) = argmin ||Z2 - A Z1 - B U||²  solved via ridge-regularised normal eqs.
   This gives a trajectory-specific linear model each batch rather than a global
   average — consistent with Bounou et al. §2.

3. Prediction loss in PIXEL space: decoder(A^k z_m) ≈ obs_{m+k}.
   Pixel targets are fixed images → no moving-target problem.
   The encoder is shaped by "what representation makes linear pixel prediction best?"

4. Small latent predictor loss: trains the MLP predictor network for CEM rollouts.
   Secondary to the pixel pred loss; encoder is mostly driven by pixel targets.

5. No EMA target encoder, no VICReg: the reconstruction loss + pixel prediction
   loss naturally prevent collapse by tying the encoder to pixel space.

Usage:
    python experiments/train_ae.py --config configs/cartpole_ae_baseline.yaml
    python experiments/train_ae.py --config configs/cartpole_ae_baseline.yaml \\
        --seed 44 --epochs 80 --force
"""
from __future__ import annotations
import argparse, copy, os, sys, json, time, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import yaml


# ── argument parsing ──────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',      default='configs/cartpole_ae_baseline.yaml')
    p.add_argument('--data-dir',    default='data')
    p.add_argument('--results-dir', default='results')
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--epochs',      type=int, default=None)
    p.add_argument('--device',      default=None)
    p.add_argument('--force',       action='store_true')
    p.add_argument('--eval-only',   action='store_true')
    p.add_argument('--checkpoint',  default=None)
    return p.parse_args()


def _set_seeds(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _make_obs_tensor(obs_np, device):
    """(H, W, 3) uint8 → (1, 3, H, W) float32."""
    return torch.from_numpy(obs_np).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0


def _make_obs_2frame(obs_np, device):
    """Single obs → (1, 6, H, W): duplicate the frame (zero-velocity context)."""
    t = _make_obs_tensor(obs_np, device)
    return torch.cat([t, t], dim=1)


# ── DMD helpers ───────────────────────────────────────────────────────────────

def _dmd_estimate(z_context: torch.Tensor, u_context: torch.Tensor,
                  ridge: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch least-squares DMD: z_{t+1} ≈ A z_t + B u_t.

    Args:
        z_context: (B, m+1, d)  — encoded context latents
        u_context: (B, m,   1)  — scalar actions in context window
        ridge:                  — regularisation strength (scaled by N)
    Returns:
        A: (d, d), B: (d, 1)
    """
    B_batch, mp1, d = z_context.shape
    m = mp1 - 1

    Z1 = z_context[:, :-1].reshape(-1, d)    # (B*m, d)
    Z2 = z_context[:, 1: ].reshape(-1, d)    # (B*m, d)
    U  = u_context.reshape(-1, 1)            # (B*m, 1)

    X  = torch.cat([Z1, U], dim=1)           # (B*m, d+1)
    N  = X.shape[0]

    # Ridge normal equations: K = (X^T X + λN I)^{-1} X^T Z2, shape (d+1, d)
    lam  = ridge * N
    gram = X.T @ X + lam * torch.eye(d + 1, device=X.device, dtype=X.dtype)
    rhs  = X.T @ Z2
    K    = torch.linalg.solve(gram, rhs)     # (d+1, d)

    # K = [A^T ; B^T]  →  A = K[:d].T, B = K[d:].T
    return K[:d].T, K[d:].T                  # A (d,d), B (d,1)


def _dmd_rollout(z_init: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
                 u_seq: torch.Tensor) -> torch.Tensor:
    """Linear rollout: z_{t+1} = A z_t + B u_t.

    Args:
        z_init: (B, d)
        A: (d, d), B: (d, 1)
        u_seq: (B, n_steps, 1)
    Returns:
        (B, n_steps, d)
    """
    preds, z = [], z_init
    for k in range(u_seq.shape[1]):
        z = z @ A.T + u_seq[:, k] @ B.T
        preds.append(z)
    return torch.stack(preds, dim=1)


# ── training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device,
                    lambda_dmd_pixel, lambda_recon, lambda_pred, lambda_fp,
                    dmd_context_len, dmd_ridge, predictor_window,
                    z_star_ema, frame_stack,
                    use_vicreg=False, vicreg_lambda=25.0, vicreg_nu=1.0):
    model.train()
    total_acc = dmd_acc = recon_acc = pred_acc = fp_acc = vic_acc = 0.0
    n_batches = 0

    for batch in loader:
        obs_seq = batch['obs_seq'].to(device)   # (B, H+1, C_fs, h, w)
        actions = batch['actions'].to(device)   # (B, H, 1)
        B, H1, C_fs, h, w = obs_seq.shape
        H = H1 - 1
        C = 3  # single-frame channels

        # Current (second) frame per timestep — pixel prediction target
        if frame_stack > 1:
            obs_curr = obs_seq[:, :, C:, :, :]   # (B, H+1, 3, h, w)
        else:
            obs_curr = obs_seq

        # ── Encode context frames (one batched forward pass) ─────────────
        m = min(dmd_context_len, H - 1)
        ctx_flat  = obs_seq[:, :m+1].reshape(-1, C_fs, h, w)   # (B*(m+1), C_fs, h, w)
        z_ctx_flat = model.encoder(ctx_flat)                     # (B*(m+1), d)
        d         = z_ctx_flat.shape[-1]
        z_ctx     = z_ctx_flat.reshape(B, m+1, d)               # (B, m+1, d)

        # ── DMD estimation ───────────────────────────────────────────────
        A, B_dmd = _dmd_estimate(z_ctx, actions[:, :m], ridge=dmd_ridge)  # (d,d), (d,1)

        # ── Pixel prediction loss (future frames decoded from DMD rollout) ─
        n_pred    = H - m
        z_pred    = _dmd_rollout(z_ctx[:, -1], A, B_dmd, actions[:, m:])  # (B, n_pred, d)
        z_pred_fl = z_pred.reshape(-1, d)                                   # (B*n_pred, d)
        obs_hat   = model.decode(z_pred_fl)                                 # (B*n_pred, 3, h, w)
        obs_tgt   = obs_curr[:, m+1:].reshape(-1, C, h, w)                 # (B*n_pred, 3, h, w)
        dmd_pixel_loss = F.mse_loss(obs_hat, obs_tgt)

        # ── Reconstruction loss (all context frames, Bounou eq. 12) ─────
        obs_hat_ctx = model.decode(z_ctx_flat)                          # (B*(m+1), 3, h, w)
        obs_ctx_tgt = obs_curr[:, :m+1].reshape(-1, C, h, w)           # (B*(m+1), 3, h, w)
        recon_loss  = F.mse_loss(obs_hat_ctx, obs_ctx_tgt)

        # ── Latent predictor loss (for CEM) — trained on context window ──
        # Encode H+1 frames for predictor targets (detached: predictor trains
        # to predict the encoder's output, encoder shaped by pixel losses above)
        with torch.no_grad():
            all_flat = obs_seq.reshape(-1, C_fs, h, w)
            z_all_fl = model.encoder(all_flat)
            z_all    = z_all_fl.reshape(B, H+1, d)
        z_targets = z_all[:, 1:].detach()

        # Teacher forcing: predict z_{k+1} from TRUE window z_{k-W+1..k}, u_{k-W+1..k}.
        # Open-loop rollout compounds errors across H steps and causes pred divergence.
        W = predictor_window
        pred_loss = torch.zeros(1, device=device)
        for k in range(H):
            s = max(0, k + 1 - W)
            z_win = z_all[:, s:k+1].detach()             # (B, ≤W, d)
            if z_win.shape[1] < W:
                pad = z_all[:, :1].expand(B, W - z_win.shape[1], d).detach()
                z_win = torch.cat([pad, z_win], dim=1)   # (B, W, d)
            u_win = actions[:, s:k+1]                    # (B, ≤W, 1)
            if u_win.shape[1] < W:
                pad_u = torch.zeros(B, W - u_win.shape[1], 1, device=device)
                u_win = torch.cat([pad_u, u_win], dim=1) # (B, W, 1)
            z_hat = model.predict(z_win, u_win)
            pred_loss = pred_loss + F.mse_loss(z_hat, z_targets[:, k])
        pred_loss = pred_loss / H

        # ── Fixed-point loss ─────────────────────────────────────────────
        fp_loss = torch.zeros(1, device=device)
        if z_star_ema is not None:
            z_sw = z_star_ema.unsqueeze(0).unsqueeze(0).expand(1, W, -1)
            z_sp = model.predict(z_sw, torch.zeros(1, W, 1, device=device))
            fp_loss = F.mse_loss(z_sp, z_star_ema.unsqueeze(0).detach())

        # ── VICReg collapse prevention on encoder outputs ─────────────────
        vic_loss = torch.zeros(1, device=device)
        if use_vicreg:
            from losses.prediction import vicreg_collapse_loss
            vic_loss, _ = vicreg_collapse_loss(
                z_ctx_flat, lambda_var=vicreg_lambda, nu_cov=vicreg_nu)

        loss = (lambda_dmd_pixel * dmd_pixel_loss
                + lambda_recon   * recon_loss
                + lambda_pred    * pred_loss
                + lambda_fp      * fp_loss
                + vic_loss)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Update z* EMA from near-equilibrium samples in context window
        if 'states' in batch:
            states_b = batch['states'][:, 0].to(device).float()
            eq_mask  = states_b.abs().max(dim=1).values < 0.05
            if eq_mask.sum() > 0:
                z0_eq = z_ctx[:, 0][eq_mask].mean(0).detach()
                z_star_ema = (z0_eq.clone() if z_star_ema is None
                              else 0.99 * z_star_ema + 0.01 * z0_eq)

        total_acc += loss.item()
        dmd_acc   += dmd_pixel_loss.item()
        recon_acc += recon_loss.item()
        pred_acc  += pred_loss.item()
        fp_acc    += fp_loss.item()
        vic_acc   += vic_loss.item()
        n_batches += 1

    return {
        'total': total_acc / n_batches,
        'dmd':   dmd_acc   / n_batches,
        'recon': recon_acc / n_batches,
        'pred':  pred_acc  / n_batches,
        'fp':    fp_acc    / n_batches,
        'vic':   vic_acc   / n_batches,
    }, z_star_ema


@torch.no_grad()
def val_one_epoch(model, loader, device,
                  lambda_dmd_pixel, lambda_recon, lambda_pred, lambda_fp,
                  dmd_context_len, dmd_ridge, predictor_window,
                  z_star_ema, frame_stack):
    model.eval()
    total_acc = n_batches = 0

    for batch in loader:
        obs_seq = batch['obs_seq'].to(device)
        actions = batch['actions'].to(device)
        B, H1, C_fs, h, w = obs_seq.shape
        H = H1 - 1
        C = 3

        obs_curr = obs_seq[:, :, C:, :, :] if frame_stack > 1 else obs_seq

        m         = min(dmd_context_len, H - 1)
        ctx_flat  = obs_seq[:, :m+1].reshape(-1, C_fs, h, w)
        z_ctx     = model.encoder(ctx_flat).reshape(B, m+1, -1)
        d         = z_ctx.shape[-1]

        A, B_dmd  = _dmd_estimate(z_ctx, actions[:, :m], ridge=dmd_ridge)
        n_pred    = H - m
        z_pred    = _dmd_rollout(z_ctx[:, -1], A, B_dmd, actions[:, m:])
        dmd_pixel_loss = F.mse_loss(
            model.decode(z_pred.reshape(-1, d)),
            obs_curr[:, m+1:].reshape(-1, C, h, w))

        obs_hat_ctx_v = model.decode(z_ctx.reshape(-1, d))
        recon_loss    = F.mse_loss(obs_hat_ctx_v, obs_curr[:, :m+1].reshape(-1, C, h, w))

        all_flat = obs_seq.reshape(-1, C_fs, h, w)
        z_all    = model.encoder(all_flat).reshape(B, H+1, d)
        z_targets = z_all[:, 1:].detach()
        W = predictor_window
        pred_loss = torch.zeros(1, device=device)
        for k in range(H):
            s = max(0, k + 1 - W)
            z_win = z_all[:, s:k+1]
            if z_win.shape[1] < W:
                pad = z_all[:, :1].expand(B, W - z_win.shape[1], d)
                z_win = torch.cat([pad, z_win], dim=1)
            u_win = actions[:, s:k+1]
            if u_win.shape[1] < W:
                pad_u = torch.zeros(B, W - u_win.shape[1], 1, device=device)
                u_win = torch.cat([pad_u, u_win], dim=1)
            z_hat = model.predict(z_win, u_win)
            pred_loss = pred_loss + F.mse_loss(z_hat, z_targets[:, k])
        pred_loss = pred_loss / H

        fp_loss = torch.zeros(1, device=device)
        if z_star_ema is not None:
            z_sw = z_star_ema.unsqueeze(0).unsqueeze(0).expand(1, W, -1)
            z_sp = model.predict(z_sw, torch.zeros(1, W, 1, device=device))
            fp_loss = F.mse_loss(z_sp, z_star_ema.unsqueeze(0))

        loss = (lambda_dmd_pixel * dmd_pixel_loss + lambda_recon * recon_loss
                + lambda_pred * pred_loss + lambda_fp * fp_loss)
        total_acc += loss.item()
        n_batches += 1

    return total_acc / n_batches


# ── CEM evaluation ────────────────────────────────────────────────────────────

def run_cem_eval(model, cfg, env_cfg, ctrl_cfg, device, seed, z_star_np, frame_stack):
    from envs.cartpole_visual import ContinuousCartpoleVisual
    from control.cem import CEMLatentPlanner
    from control.rollout import evaluate_stabilization_mpc

    cem_cfg    = cfg.get('cem', {})
    model_cfg  = cfg.get('model', {})
    n_trials   = int(cem_cfg.get('n_trials', 30))
    horizon    = int(cem_cfg.get('horizon', 25))
    n_samples  = int(cem_cfg.get('n_samples', 500))
    n_elites   = int(cem_cfg.get('n_elites', 50))
    n_iter     = int(cem_cfg.get('n_iter', 5))
    init_std   = float(cem_cfg.get('init_std', 3.0))
    pred_win   = int(model_cfg.get('predictor_window', 3))

    d = len(z_star_np)
    Q = np.eye(d)

    env = ContinuousCartpoleVisual(
        frame_skip=env_cfg.get('frame_skip', 1),
        image_size=env_cfg.get('image_size', 64),
        mass_cart=env_cfg.get('mass_cart', 1.0),
        mass_pole=env_cfg.get('mass_pole', 0.1),
        pole_length=env_cfg.get('pole_length', 0.5),
        gravity=env_cfg.get('gravity', 9.8),
        action_range=tuple(env_cfg.get('action_range', (-10, 10))),
        seed=seed,
    )

    planner = CEMLatentPlanner(
        predictor=model,
        action_encoder=model.action_encoder,
        predictor_window=pred_win,
        Q=Q, R=0.01,
        horizon=horizon,
        n_samples=n_samples, n_elites=n_elites, n_iter=n_iter,
        init_std=init_std,
        action_lb=float(env_cfg.get('action_range', [-10, 10])[0]),
        action_ub=float(env_cfg.get('action_range', [-10, 10])[1]),
        device=device,
    )

    results = evaluate_stabilization_mpc(
        encoder=model.encoder, mpc=planner, env=env,
        n_trials=n_trials,
        T=int(cfg.get('probes', {}).get('T_rollout', 200)),
        init_scale=float(ctrl_cfg.get('init_scale', 0.05)),
        stabilization_threshold=float(ctrl_cfg.get('stabilization_threshold', 0.1)),
        settling_threshold=float(ctrl_cfg.get('settling_threshold', 0.05)),
        seed=seed, device=device, z_star=z_star_np,
        frame_stack=frame_stack,
    )
    env.close()
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()
    _set_seeds(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_cfg   = cfg['environment']
    model_cfg = cfg['model']
    train_cfg = cfg['training']
    ctrl_cfg  = cfg.get('control', {})

    device = torch.device(args.device) if args.device else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    epochs          = args.epochs or int(train_cfg.get('epochs', 80))
    batch_size      = int(train_cfg.get('batch_size', 256))
    horizon         = int(train_cfg.get('horizon', 10))
    W               = int(model_cfg.get('predictor_window', 3))
    frame_stack     = int(model_cfg.get('frame_stack', 2))
    lr              = float(train_cfg.get('lr', 1e-4))
    wd              = float(train_cfg.get('weight_decay', 1e-4))
    lambda_dmd_pixel= float(train_cfg.get('lambda_dmd_pixel', 5.0))
    lambda_recon    = float(train_cfg.get('lambda_recon', 1.0))
    lambda_pred     = float(train_cfg.get('lambda_pred', 0.5))
    lambda_fp       = float(train_cfg.get('lambda_fp', 5.0))
    dmd_context_len = int(train_cfg.get('dmd_context_len', 5))
    dmd_ridge       = float(train_cfg.get('dmd_ridge', 1e-4))
    use_vicreg      = bool(train_cfg.get('use_vicreg', False))
    vicreg_lambda   = float(train_cfg.get('vicreg_lambda', 25.0))
    vicreg_nu       = float(train_cfg.get('vicreg_nu', 1.0))
    ckpt_every      = int(train_cfg.get('checkpoint_every', 10))
    _cfg_stem       = Path(args.config).stem

    out_dir    = Path(args.results_dir) / f'{_cfg_stem}_seed{args.seed}'
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_final = out_dir / 'model_final.pt'

    if ckpt_final.exists() and not args.force and not args.eval_only:
        print(f'[skip] {ckpt_final} exists. Use --force to retrain.')

    # ── equilibrium observation ───────────────────────────────────────────
    from envs.cartpole_visual import ContinuousCartpoleVisual
    _eq_env = ContinuousCartpoleVisual(
        frame_skip=env_cfg.get('frame_skip', 1),
        image_size=env_cfg.get('image_size', 64),
        mass_cart=env_cfg.get('mass_cart', 1.0),
        mass_pole=env_cfg.get('mass_pole', 0.1),
        pole_length=env_cfg.get('pole_length', 0.5),
        gravity=env_cfg.get('gravity', 9.8),
        action_range=tuple(env_cfg.get('action_range', (-10, 10))),
    )
    obs_eq, _, _ = _eq_env.reset_to_state(np.zeros(4, dtype=np.float32))
    _eq_env.close()
    obs_eq_t = _make_obs_tensor(obs_eq, device)
    # 2-frame equilibrium: duplicate (zero velocity at rest)
    obs_eq_2f = torch.cat([obs_eq_t, obs_eq_t], dim=1) if frame_stack > 1 else obs_eq_t

    # ── dataset ───────────────────────────────────────────────────────────
    from data.dataset import load_dataset, make_dataloaders, generate_dataset
    data_cfg = cfg['data']
    h5_path  = Path(args.data_dir) / f'{_cfg_stem}_ep_seed{args.seed}.h5'
    if h5_path.exists():
        print(f'[data] Loading {h5_path}')
        data = load_dataset(str(h5_path))
    else:
        print(f'[data] Generating dataset (episode-centric)...')
        data = generate_dataset(
            n_random_episodes=int(data_cfg.get('n_random_episodes', 50)),
            random_ep_len=int(data_cfg.get('random_ep_len', 200)),
            random_init_range=float(data_cfg.get('random_init_range', 1.0)),
            n_lqr_episodes=int(data_cfg.get('n_lqr_episodes', 50)),
            lqr_ep_len=int(data_cfg.get('lqr_ep_len', 200)),
            lqr_init_range=float(data_cfg.get('lqr_init_range', 0.30)),
            lqr_noise_std=float(data_cfg.get('lqr_noise_std', 0.25)),
            n_equilibrium=int(data_cfg.get('n_equilibrium', 500)),
            eq_ep_len=int(data_cfg.get('eq_ep_len', 50)),
            n_eq_selfloop=int(data_cfg.get('n_eq_selfloop', 200)),
            eq_init_range=float(data_cfg.get('eq_init_range', 0.002)),
            eq_noise_std=float(data_cfg.get('eq_noise_std', 0.001)),
            n_pe_episodes=int(data_cfg.get('n_pe_episodes', 0)),
            pe_ep_len=int(data_cfg.get('pe_ep_len', 40)),
            pe_init_range=float(data_cfg.get('pe_init_range', 0.05)),
            pe_action_amplitude=float(data_cfg.get('pe_action_amplitude', 3.0)),
            pe_flip_prob=float(data_cfg.get('pe_flip_prob', 0.15)),
            n_passive_episodes=int(data_cfg.get('n_passive_episodes', 0)),
            passive_ep_len=int(data_cfg.get('passive_ep_len', 50)),
            passive_init_range=float(data_cfg.get('passive_init_range', 0.05)),
            train_frac=float(data_cfg.get('train_frac', 0.8)),
            val_frac=float(data_cfg.get('val_frac', 0.1)),
            frame_skip=int(env_cfg.get('frame_skip', 1)),
            image_size=int(env_cfg['image_size']),
            action_range=tuple(env_cfg['action_range']),
            save_path=str(h5_path),
            seed=args.seed,
        )

    loaders = make_dataloaders(
        data, batch_size=batch_size,
        horizon=horizon, frame_stack=frame_stack,
        obs_eq=obs_eq,
        n_eq_selfloop=int(data_cfg.get('n_eq_selfloop', 200)),
    )
    print(f'[data] Train batches: {len(loaders["train"])}  '
          f'Val batches: {len(loaders["val"])}  '
          f'frame_stack={frame_stack}')

    # ── model ─────────────────────────────────────────────────────────────
    from models.jepa import JEPAConfig
    from models.autoencoder import AEWorldModel

    encoder_type = model_cfg.get('encoder_type', 'vit')
    jepa_cfg = JEPAConfig(
        latent_dim=int(model_cfg.get('latent_dim', 32)),
        action_latent_dim=int(model_cfg.get('action_latent_dim', 4)),
        action_encoder=model_cfg.get('action_encoder', 'linear'),
        encoder_type=encoder_type,
        image_size=int(env_cfg.get('image_size', 64)),
        patch_size=int(model_cfg.get('patch_size', 8)),
        frame_stack=frame_stack,
        vit_embed_dim=int(model_cfg.get('vit_embed_dim', 128)),
        vit_depth=int(model_cfg.get('vit_depth', 4)),
        vit_num_heads=int(model_cfg.get('vit_num_heads', 4)),
        predictor_hidden_dim=int(model_cfg.get('predictor_hidden_dim', 256)),
        predictor_n_layers=int(model_cfg.get('predictor_n_layers', 2)),
        predictor_window=W,
    )
    model = AEWorldModel(jepa_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[model] AEWorldModel  params: {n_params:,}  '
          f'(encoder={encoder_type}, in_chans={jepa_cfg.in_chans})')

    if args.eval_only or (ckpt_final.exists() and not args.force):
        ckpt_path = args.checkpoint or str(ckpt_final)
        print(f'[eval] Loading {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get('model_state', ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=False)
    else:
        optimizer = torch.optim.Adam([
            {'params': model.encoder.parameters()},
            {'params': model.decoder.parameters()},
            {'params': model.action_encoder.parameters(), 'weight_decay': 0.0},
            {'params': model.predictor.parameters()},
        ], lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.1)

        with torch.no_grad():
            z_star_ema = model.encoder(obs_eq_2f).squeeze(0)

        print(f'\n[train] Starting AE-DMD training for {epochs} epochs...')
        print(f'[train] DMD context={dmd_context_len}/{horizon} steps, '
              f'predict={horizon - dmd_context_len} steps in pixel space')
        vic_str = f'  vicreg×{vicreg_lambda}(nu={vicreg_nu})' if use_vicreg else '  vicreg=off'
        print(f'[train] Losses: dmd_pixel×{lambda_dmd_pixel}  '
              f'recon×{lambda_recon}  pred×{lambda_pred}  fp×{lambda_fp}{vic_str}')

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr, z_star_ema = train_one_epoch(
                model, loaders['train'], optimizer, device,
                lambda_dmd_pixel, lambda_recon, lambda_pred, lambda_fp,
                dmd_context_len, dmd_ridge, W, z_star_ema, frame_stack,
                use_vicreg=use_vicreg, vicreg_lambda=vicreg_lambda, vicreg_nu=vicreg_nu)
            val_loss = val_one_epoch(
                model, loaders['val'], device,
                lambda_dmd_pixel, lambda_recon, lambda_pred, lambda_fp,
                dmd_context_len, dmd_ridge, W, z_star_ema, frame_stack)
            scheduler.step()

            vic_log = f'  vic={tr["vic"]:.4f}' if use_vicreg else ''
            print(f'[Epoch {epoch:3d}/{epochs}]  '
                  f'train={tr["total"]:.4f}  val={val_loss:.4f}  '
                  f'dmd={tr["dmd"]:.4f}  recon={tr["recon"]:.4f}  '
                  f'pred={tr["pred"]:.4f}  fp={tr["fp"]:.4f}{vic_log}  '
                  f'({time.time()-t0:.1f}s)')

            if epoch % ckpt_every == 0:
                torch.save({'epoch': epoch, 'model_state': model.state_dict()},
                           str(out_dir / f'model_ep{epoch:04d}.pt'))

        torch.save({'epoch': epochs, 'model_state': model.state_dict()},
                   str(ckpt_final))
        print(f'[train] Saved {ckpt_final}')

    # ── evaluation ────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        z_star_np = model.encoder(obs_eq_2f).squeeze(0).cpu().numpy()

    print('\n[eval] Encoder sensitivity (θ varies, θ̇=0, 2-frame: [obs_t, obs_t]):')
    from envs.cartpole_visual import ContinuousCartpoleVisual
    _diag_env = ContinuousCartpoleVisual(
        frame_skip=env_cfg.get('frame_skip', 1),
        image_size=env_cfg.get('image_size', 64),
        action_range=tuple(env_cfg.get('action_range', (-10, 10))),
    )
    for _th in [0.02, 0.05, 0.10, 0.20, 0.40]:
        _x  = np.array([0.0, 0.0, _th, 0.0], dtype=np.float32)
        _obs, _, _ = _diag_env.reset_to_state(_x)
        _obs_t  = _make_obs_tensor(_obs, device)
        # 2-frame: duplicate same frame → zero velocity context
        _obs_2f = torch.cat([_obs_t, _obs_t], dim=1) if frame_stack > 1 else _obs_t
        with torch.no_grad():
            _z = model.encoder(_obs_2f).cpu().numpy()[0]
        print(f'  θ={_th:+.2f} rad: ||z-z*||={np.linalg.norm(_z - z_star_np):.4f}')
    _diag_env.close()

    print('\n[eval] Running CEM nonlinear (H=25, Q=I)...')
    cem_results = run_cem_eval(model, cfg, env_cfg, ctrl_cfg, device,
                               args.seed, z_star_np, frame_stack)
    print(f'  success_rate:        {cem_results["success_rate"]:.3f}')
    print(f'  mean_episode_length: {cem_results["mean_episode_length"]:.1f}')
    print(f'  mean_fraction_stable:{cem_results["mean_fraction_stable"]:.3f}')
    print(f'  mean_cost:           {cem_results["mean_cost"]:.1f}')

    cem_scalars = {k: v for k, v in cem_results.items() if k != 'vis_result'}
    results_out = {'cem_eval': cem_scalars, 'z_star': z_star_np.tolist()}
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results_out, f, indent=2)
    print(f'[done] Results saved to {out_dir}/results.json')


if __name__ == '__main__':
    main()
