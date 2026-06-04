"""Train an Autoencoder World Model baseline.

Training objective:
    L = lambda_pred * L_pred  +  lambda_recon * L_recon  +  lambda_fp * L_fp

    L_pred  = L2(z_hat, z_target)   (same as JEPA prediction loss)
    L_recon = MSE(decode(z_t), obs_t)  (pixel reconstruction — forces sensitivity)
    L_fp    = MSE(f(z*, 0), z*)     (fixed-point stability)

No dynSIG / PBH / spectral losses — this is an intentionally minimal baseline.
The reconstruction loss alone should prevent encoder collapse and make CEM work.

Usage:
    python experiments/train_ae.py --config configs/cartpole_ae_baseline.yaml
    python experiments/train_ae.py --config configs/cartpole_ae_baseline.yaml \\
        --data-dir data --results-dir results/ae --epochs 200
"""
from __future__ import annotations
import argparse, copy, os, sys, json, time, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm


# ── argument parsing ─────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',      default='configs/cartpole_ae_baseline.yaml')
    p.add_argument('--data-dir',    default='data')
    p.add_argument('--results-dir', default='results')
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--epochs',      type=int, default=None)
    p.add_argument('--device',      default=None)
    p.add_argument('--force',       action='store_true',
                   help='Retrain even if checkpoint exists')
    p.add_argument('--eval-only',   action='store_true')
    p.add_argument('--checkpoint',  default=None,
                   help='Path to existing checkpoint for eval-only')
    return p.parse_args()


# ── helpers ──────────────────────────────────────────────────────────────────

def _set_seeds(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _make_obs_tensor(obs_np, device):
    """(H, W, 3) uint8 → (1, 3, H, W) float32 on device."""
    return torch.from_numpy(obs_np).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0


# ── training loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, target_encoder, loader, optimizer, device,
                    lambda_pred, lambda_recon, lambda_fp,
                    predictor_window, z_star_ema, obs_eq_t,
                    target_momentum=0.9995, aug_noise_std=0.05,
                    lambda_vicreg=0.0, vicreg_nu=1.0):
    model.train()
    total, n_batches = 0.0, 0
    pred_acc, recon_acc, fp_acc, vicreg_acc = 0.0, 0.0, 0.0, 0.0

    for batch in loader:
        obs_seq = batch['obs_seq'].to(device)   # (B, H+1, C, h, w)
        actions = batch['actions'].to(device)   # (B, H, 1)
        B, H1, C, h, w = obs_seq.shape
        H = H1 - 1

        # ── encode ────────────────────────────────────────────────────────
        # Online encoder: augmented view (denoising AE).
        # Noise prevents trivial collapse: even if encoder maps clean obs → z*,
        # encoder(obs + ε) ≠ z* for ε ~ N(0, σ²), keeping the prediction and
        # reconstruction tasks non-trivial.  Decoder target stays CLEAN (denoising).
        obs_0_clean = obs_seq[:, 0]
        if aug_noise_std > 0:
            obs_0_aug = (obs_0_clean
                         + aug_noise_std * torch.randn_like(obs_0_clean)).clamp(0., 1.)
        else:
            obs_0_aug = obs_0_clean
        z_0 = model.encoder(obs_0_aug)          # (B, d) — gradient flows
        d   = z_0.shape[-1]
        with torch.no_grad():
            obs_rest = obs_seq[:, 1:].contiguous().view(B * H, C, h, w)
            # EMA target encoder: slower-moving copy of online encoder provides
            # diverse targets even when the online encoder begins to collapse,
            # keeping pred_loss non-trivial (same mechanism as BYOL/JEPA).
            z_rest   = target_encoder(obs_rest).view(B, H, d)
        z_all = torch.cat([z_0.unsqueeze(1), z_rest], dim=1)  # (B, H+1, d)

        # ── prediction loss (multi-step unrolled) ─────────────────────────
        z_targets = z_all[:, 1:].detach()
        W = predictor_window
        z_win_buf = [z_all[:, 0]] * W
        u_win_buf = [torch.zeros(B, 1, device=device)] * (W - 1)  # W-1 padding zeros

        pred_loss = torch.zeros(1, device=device)
        for k in range(H):
            u_k = actions[:, k]
            u_win_buf.append(u_k)                              # append BEFORE predict
            z_stack = torch.stack(z_win_buf[-W:], dim=1)
            u_stack = torch.stack(u_win_buf[-W:], dim=1)
            z_hat   = model.predict(z_stack, u_stack)
            pred_loss = pred_loss + F.mse_loss(z_hat, z_targets[:, k])
            z_win_buf.append(z_hat)
        pred_loss = pred_loss / H

        # ── reconstruction loss (denoising: predict clean from noisy encoding) ──
        obs_hat    = model.decode(z_0)          # (B, C, h, w) in [0, 1]
        recon_loss = F.mse_loss(obs_hat, obs_0_clean)  # target is always CLEAN

        # ── fixed-point loss ───────────────────────────────────────────────
        if z_star_ema is not None:
            z_star_w  = z_star_ema.unsqueeze(0).unsqueeze(0).expand(1, W, -1)
            u_zero_w  = torch.zeros(1, W, 1, device=device)
            z_star_pr = model.predict(z_star_w, u_zero_w)
            fp_loss   = F.mse_loss(z_star_pr, z_star_ema.unsqueeze(0).detach())
        else:
            fp_loss = torch.zeros(1, device=device)

        # ── VICReg collapse prevention ────────────────────────────────────
        if lambda_vicreg > 0:
            from losses.prediction import vicreg_collapse_loss
            vic_loss, _ = vicreg_collapse_loss(z_0, lambda_var=lambda_vicreg,
                                               nu_cov=vicreg_nu)
        else:
            vic_loss = torch.zeros(1, device=device)

        loss = (lambda_pred  * pred_loss
                + lambda_recon * recon_loss
                + lambda_fp    * fp_loss
                + vic_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA update of target encoder
        with torch.no_grad():
            for p_online, p_target in zip(model.encoder.parameters(),
                                          target_encoder.parameters()):
                p_target.data.mul_(target_momentum).add_(
                    p_online.data, alpha=1.0 - target_momentum)

        total      += loss.item()
        pred_acc   += pred_loss.item()
        vicreg_acc += vic_loss.item()
        recon_acc  += recon_loss.item()
        fp_acc     += fp_loss.item()
        n_batches  += 1

        # Update z* EMA from near-equilibrium samples
        if 'states' in batch:
            states_b = batch['states'][:, 0].to(device).float()
            eq_mask  = states_b.abs().max(dim=1).values < 0.05
            if eq_mask.sum() > 0:
                z0_eq = z_all[:, 0][eq_mask].mean(dim=0).detach()
                if z_star_ema is None:
                    z_star_ema = z0_eq.clone()
                else:
                    z_star_ema = 0.99 * z_star_ema + 0.01 * z0_eq

    return {
        'total':   total      / n_batches,
        'pred':    pred_acc   / n_batches,
        'recon':   recon_acc  / n_batches,
        'fp':      fp_acc     / n_batches,
        'vicreg':  vicreg_acc / n_batches,
    }, z_star_ema


@torch.no_grad()
def val_one_epoch(model, loader, device, lambda_pred, lambda_recon, lambda_fp,
                  predictor_window, z_star_ema):
    model.eval()
    total, n_batches = 0.0, 0

    for batch in loader:
        obs_seq = batch['obs_seq'].to(device)
        actions = batch['actions'].to(device)
        B, H1, C, h, w = obs_seq.shape
        H = H1 - 1

        z_0  = model.encoder(obs_seq[:, 0])
        d    = z_0.shape[-1]
        obs_r = obs_seq[:, 1:].contiguous().view(B * H, C, h, w)
        z_r   = model.encoder(obs_r).view(B, H, d)
        z_all = torch.cat([z_0.unsqueeze(1), z_r], dim=1)

        z_targets  = z_all[:, 1:].detach()
        W          = predictor_window
        z_win_buf  = [z_all[:, 0]] * W
        u_win_buf  = [torch.zeros(B, 1, device=device)] * (W - 1)

        pred_loss = torch.zeros(1, device=device)
        for k in range(H):
            u_k = actions[:, k]
            u_win_buf.append(u_k)
            zs     = torch.stack(z_win_buf[-W:], dim=1)
            us     = torch.stack(u_win_buf[-W:], dim=1)
            z_hat  = model.predict(zs, us)
            pred_loss = pred_loss + F.mse_loss(z_hat, z_targets[:, k])
            z_win_buf.append(z_hat)
        pred_loss = pred_loss / H

        obs_t_float = obs_seq[:, 0]
        obs_hat     = model.decode(z_0)
        recon_loss  = F.mse_loss(obs_hat, obs_t_float)

        fp_loss = torch.zeros(1, device=device)
        if z_star_ema is not None:
            z_star_w  = z_star_ema.unsqueeze(0).unsqueeze(0).expand(1, W, -1)
            u_zero_w  = torch.zeros(1, W, 1, device=device)
            z_star_pr = model.predict(z_star_w, u_zero_w)
            fp_loss   = F.mse_loss(z_star_pr, z_star_ema.unsqueeze(0).detach())

        loss = (lambda_pred * pred_loss
                + lambda_recon * recon_loss
                + lambda_fp    * fp_loss)
        total    += loss.item()
        n_batches+= 1

    return total / n_batches


# ── CEM evaluation ────────────────────────────────────────────────────────────

def run_cem_eval(model, cfg, env_cfg, ctrl_cfg, device, seed, z_star_np):
    from envs.cartpole_visual import ContinuousCartpoleVisual
    from control.cem import CEMLatentPlanner
    from control.rollout import evaluate_stabilization_mpc
    from control.jacobian import compute_jacobian_torch

    cem_cfg = cfg.get('cem', {})
    n_trials   = int(cem_cfg.get('n_trials', 30))
    horizon    = int(cem_cfg.get('horizon', 25))
    n_samples  = int(cem_cfg.get('n_samples', 500))
    n_elites   = int(cem_cfg.get('n_elites', 50))
    n_iter     = int(cem_cfg.get('n_iter', 5))
    init_std   = float(cem_cfg.get('init_std', 3.0))

    model.eval()
    z_star_t = torch.tensor(z_star_np, dtype=torch.float32, device=device)
    A_jac, _ = compute_jacobian_torch(model, z_star_t, device)
    d = A_jac.shape[0]

    # Q = I (basic cost in latent space)
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

    predictor_window = int(cfg.get('model', {}).get('predictor_window', 1))
    planner = CEMLatentPlanner(
        predictor=model,
        action_encoder=model.action_encoder,
        predictor_window=predictor_window,
        Q=Q,
        R=0.01,
        horizon=horizon,
        n_samples=n_samples,
        n_elites=n_elites,
        n_iter=n_iter,
        init_std=init_std,
        action_lb=float(env_cfg.get('action_range', [-10, 10])[0]),
        action_ub=float(env_cfg.get('action_range', [-10, 10])[1]),
        device=device,
    )

    results = evaluate_stabilization_mpc(
        encoder=model.encoder,
        mpc=planner,
        env=env,
        n_trials=n_trials,
        T=int(cfg.get('probes', {}).get('T_rollout', 200)),
        init_scale=float(ctrl_cfg.get('init_scale', 0.05)),
        stabilization_threshold=float(ctrl_cfg.get('stabilization_threshold', 0.1)),
        settling_threshold=float(ctrl_cfg.get('settling_threshold', 0.05)),
        seed=seed,
        device=device,
        z_star=z_star_np,
    )
    env.close()
    return results


# ── main ─────────────────────────────────────────────────────────────────────

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

    epochs      = args.epochs or int(train_cfg.get('epochs', 200))
    batch_size  = int(train_cfg.get('batch_size', 256))
    horizon     = int(train_cfg.get('horizon', 10))
    W           = int(train_cfg.get('predictor_window', 1))
    lr          = float(train_cfg.get('lr', 1e-4))
    wd          = float(train_cfg.get('weight_decay', 1e-4))
    lp          = float(train_cfg.get('lambda_pred',  1.0))
    lr_recon    = float(train_cfg.get('lambda_recon', 1.0))
    lf          = float(train_cfg.get('lambda_fp',    50.0))
    ckpt_every  = int(train_cfg.get('checkpoint_every', 10))

    # ── output dirs ──────────────────────────────────────────────────────
    out_dir = Path(args.results_dir) / f'ae_seed{args.seed}'
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

    # ── dataset ───────────────────────────────────────────────────────────
    from data.dataset import load_dataset, make_dataloaders, generate_dataset
    data_cfg = cfg['data']
    h5_path  = Path(args.data_dir) / f'cartpole_ae_seed{args.seed}.h5'
    if h5_path.exists():
        print(f'[data] Loading {h5_path}')
        data = load_dataset(str(h5_path))
    else:
        print(f'[data] Generating mixed dataset...')
        data = generate_dataset(
            dataset_type='mixed',
            n_transitions=int(data_cfg['n_random']),
            frame_skip=int(env_cfg.get('frame_skip', 1)),
            save_path=str(h5_path),
            seed=args.seed,
            image_size=int(env_cfg['image_size']),
            action_range=tuple(env_cfg['action_range']),
            init_range=float(data_cfg.get('random_init_range', 0.40)),
            lqr_init_range=float(data_cfg.get('lqr_init_range', 0.10)),
            lqr_noise_std=float(data_cfg.get('lqr_noise_std', 0.25)),
            n_equilibrium=int(data_cfg.get('n_equilibrium', 0)),
            eq_init_range=float(data_cfg.get('eq_init_range', 0.002)),
            eq_noise_std=float(data_cfg.get('eq_noise_std', 0.001)),
            n_pe=int(data_cfg.get('n_pe', 0)),
            pe_init_range=float(data_cfg.get('pe_init_range', 0.05)),
            pe_action_amplitude=float(data_cfg.get('pe_action_amplitude', 3.0)),
            pe_flip_prob=float(data_cfg.get('pe_flip_prob', 0.15)),
            pe_max_ep_len=int(data_cfg.get('pe_max_ep_len', 40)),
        )

    n_eq_selfloop = int(data_cfg.get('n_eq_selfloop', 0))
    loaders = make_dataloaders(
        data, batch_size=batch_size,
        horizon=horizon, frame_stack=int(model_cfg.get('frame_stack', 1)),
        obs_eq=obs_eq, n_eq_selfloop=n_eq_selfloop,
    )
    print(f"[data] Train batches: {len(loaders['train'])}  "
          f"Val batches: {len(loaders['val'])}")

    # ── model ─────────────────────────────────────────────────────────────
    from models.jepa import JEPAConfig
    from models.autoencoder import AEWorldModel

    jepa_cfg = JEPAConfig(
        latent_dim=int(model_cfg.get('latent_dim', 32)),
        action_latent_dim=int(model_cfg.get('action_latent_dim', 4)),
        action_encoder=model_cfg.get('action_encoder', 'linear'),
        image_size=int(env_cfg.get('image_size', 64)),
        patch_size=int(model_cfg.get('patch_size', 8)),
        vit_embed_dim=int(model_cfg.get('vit_embed_dim', 128)),
        vit_depth=int(model_cfg.get('vit_depth', 4)),
        vit_num_heads=int(model_cfg.get('vit_num_heads', 4)),
        predictor_hidden_dim=int(model_cfg.get('predictor_hidden_dim', 256)),
        predictor_n_layers=int(model_cfg.get('predictor_n_layers', 2)),
        predictor_window=W,
    )
    model = AEWorldModel(jepa_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[model] AEWorldModel  params: {n_params:,}')

    if args.eval_only or (ckpt_final.exists() and not args.force):
        ckpt_path = args.checkpoint or str(ckpt_final)
        print(f'[eval] Loading {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get('model_state', ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=False)
    else:
        # ── optimizer + scheduler ─────────────────────────────────────────
        optimizer = torch.optim.Adam([
            {'params': model.encoder.parameters()},
            {'params': model.decoder.parameters()},
            {'params': model.action_encoder.parameters(), 'weight_decay': 0.0},
            {'params': model.predictor.parameters()},
        ], lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.1)

        # Set z* from equilibrium obs
        obs_eq_t = _make_obs_tensor(obs_eq, device)
        with torch.no_grad():
            z_star_ema = model.encoder(obs_eq_t).squeeze(0)

        # EMA target encoder: separate slow-moving copy for prediction targets.
        target_encoder = copy.deepcopy(model.encoder)
        for p in target_encoder.parameters():
            p.requires_grad_(False)
        target_encoder.eval()
        target_momentum = float(train_cfg.get('target_encoder_momentum', 0.9995))
        aug_noise_std   = float(train_cfg.get('aug_noise_std', 0.05))
        lambda_vicreg   = float(train_cfg.get('vicreg_lambda', 0.0)) if train_cfg.get('use_vicreg', False) else 0.0
        vicreg_nu       = float(train_cfg.get('vicreg_nu', 1.0))

        # With N batches/epoch, EMA lag in epochs ≈ 1 / ((1-m) * N).
        # At m=0.99, N=180: lag = 0.56 epochs — target tracks online too fast.
        # At m=0.9995, N=180: lag = 11 epochs — diverse targets for first ~30 epochs.
        n_batches_per_epoch = len(loaders['train'])
        lag_epochs = 1.0 / ((1 - target_momentum) * n_batches_per_epoch)
        print(f'\n[train] Starting AE training for {epochs} epochs...')
        print(f'[train] EMA target encoder: momentum={target_momentum}  '
              f'effective_lag≈{lag_epochs:.1f} epochs')
        print(f'[train] Augmentation noise std={aug_noise_std} '
              f'(denoising AE: online sees noisy obs, decoder reconstructs clean)')
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_info, z_star_ema = train_one_epoch(
                model, target_encoder, loaders['train'], optimizer, device,
                lp, lr_recon, lf, W, z_star_ema, obs_eq_t,
                target_momentum=target_momentum, aug_noise_std=aug_noise_std,
                lambda_vicreg=lambda_vicreg, vicreg_nu=vicreg_nu)
            val_loss = val_one_epoch(
                model, loaders['val'], device, lp, lr_recon, lf, W, z_star_ema)
            scheduler.step()

            vic_str = f'  vic={train_info["vicreg"]:.4f}' if lambda_vicreg > 0 else ''
            print(f'[Epoch {epoch:3d}/{epochs}]  '
                  f'train={train_info["total"]:.4f}  val={val_loss:.4f}  '
                  f'pred={train_info["pred"]:.4f}  '
                  f'recon={train_info["recon"]:.4f}  '
                  f'fp={train_info["fp"]:.4f}'
                  f'{vic_str}  '
                  f'({time.time()-t0:.1f}s)')

            if epoch % ckpt_every == 0:
                ckpt_ep = out_dir / f'model_ep{epoch:04d}.pt'
                torch.save({'epoch': epoch, 'model_state': model.state_dict()},
                           str(ckpt_ep))

        torch.save({'epoch': epochs, 'model_state': model.state_dict()},
                   str(ckpt_final))
        print(f'[train] Saved {ckpt_final}')

    # ── CEM evaluation ───────────────────────────────────────────────────
    model.eval()
    obs_eq_t = _make_obs_tensor(obs_eq, device)
    with torch.no_grad():
        z_star_np = model.encoder(obs_eq_t).squeeze(0).cpu().numpy()

    print('\n[eval] Encoder sensitivity diagnostic:')
    from envs.cartpole_visual import ContinuousCartpoleVisual
    _diag_env = ContinuousCartpoleVisual(
        frame_skip=env_cfg.get('frame_skip', 1),
        image_size=env_cfg.get('image_size', 64),
        action_range=tuple(env_cfg.get('action_range', (-10, 10))),
    )
    for _th in [0.02, 0.05, 0.10, 0.20, 0.40]:
        _x = np.array([0.0, 0.0, _th, 0.0], dtype=np.float32)
        _obs, _, _ = _diag_env.reset_to_state(_x)
        _obs_t = _make_obs_tensor(_obs, device)
        with torch.no_grad():
            _z = model.encoder(_obs_t).cpu().numpy()[0]
        _dz_norm = float(np.linalg.norm(_z - z_star_np))
        print(f'  θ={_th:+.2f} rad: ||z-z*||={_dz_norm:.4f}')
    _diag_env.close()

    print('\n[eval] Running CEM nonlinear (H=25, Q=I)...')
    cem_results = run_cem_eval(model, cfg, env_cfg, ctrl_cfg, device, args.seed, z_star_np)
    print(f'  success_rate:       {cem_results["success_rate"]:.3f}')
    print(f'  mean_episode_length:{cem_results["mean_episode_length"]:.1f}')
    print(f'  mean_fraction_stable:{cem_results["mean_fraction_stable"]:.3f}')
    print(f'  mean_cost:          {cem_results["mean_cost"]:.1f}')

    # Exclude vis_result (contains numpy arrays) — keep only scalar metrics
    cem_scalars = {k: v for k, v in cem_results.items() if k != 'vis_result'}
    results_out = {'cem_eval': cem_scalars, 'z_star': z_star_np.tolist()}
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results_out, f, indent=2)
    print(f'[done] Results saved to {out_dir}/results.json')


if __name__ == '__main__':
    main()
