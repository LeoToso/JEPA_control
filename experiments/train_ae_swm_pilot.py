"""AE World Model pilot on stable-worldmodel's swm/CartPoleControl-v1 data.

Tests whether stable-worldmodel's rendering — where the pole occupies a much
larger fraction of the frame than in our custom horizontal-view env — fixes
the decoder/reconstruction bottleneck diagnosed earlier (per-pixel MSE
dominated by the static cart/track/background -> blurry "average pole").

Reuses the exact Bounou-style training loop from experiments/train_ae.py
(per-batch DMD + pixel-space prediction + reconstruction losses — the
*actual* mechanism behind lambda_dmd_pixel/lambda_recon; the generic
training.trainer.Trainer does not implement these losses at all), pointed at
the swm-collected dataset (data/cartpole_swm_pilot.h5, action_range=[-1,1],
Discrete(2) -> {-1,+1}).

Bypasses run_experiment.py / train_ae.py's own main() because both are
tightly coupled to ContinuousCartpoleVisual (equilibrium-frame collection via
reset_to_state, DMD-LQR / CEM evaluation on the simulator) — none of which
applies to this swm-rendered, sim-less pilot. We keep only the train/val loop
and add a simple reconstruction-quality snapshot instead.

Usage:
    python experiments/train_ae_swm_pilot.py \\
        --data data/cartpole_swm_pilot.h5 --epochs 100 \\
        --save-dir results/ae_swm_pilot_seed42
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml


def _save_recon_snapshot(model, loader, device, frame_stack, out_path, n=8):
    """Dump a target/reconstruction grid (current frame) for visual inspection."""
    import matplotlib.pyplot as plt

    model.eval()
    batch = next(iter(loader))
    obs_seq = batch['obs_seq'][:n].to(device)         # (n, H+1, C_fs, h, w)
    B, H1, C_fs, h, w = obs_seq.shape
    obs_curr = obs_seq[:, :, 3:, :, :] if frame_stack > 1 else obs_seq
    with torch.no_grad():
        z = model.encoder(obs_seq[:, 0])
        obs_hat = model.decode(z)
    tgt = obs_curr[:, 0].clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    hat = obs_hat.clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))
    for i in range(n):
        axes[0, i].imshow(tgt[i]); axes[0, i].axis('off')
        axes[1, i].imshow(hat[i]); axes[1, i].axis('off')
    axes[0, 0].set_ylabel('target', fontsize=10)
    axes[1, 0].set_ylabel('recon', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'[viz] reconstruction snapshot -> {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',      default='data/cartpole_swm_pilot.h5')
    p.add_argument('--config',    default='configs/cartpole_ae_bounou.yaml',
                   help='Base config to borrow model/training hyperparameters from '
                        '(action_range is overridden to [-1, 1] for the Discrete(2) '
                        '-> {-1, +1} mapping used by data/collect_swm_cartpole.py)')
    p.add_argument('--epochs',    type=int, default=100)
    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--save-dir',  default='results/ae_swm_pilot_seed42')
    p.add_argument('--device',    default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device) if args.device else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg['model']
    train_cfg = dict(cfg['training'])
    train_cfg['epochs'] = args.epochs

    from data.dataset import load_dataset, make_dataloaders
    print(f'[data] loading {args.data}')
    data = load_dataset(args.data)
    horizon = int(train_cfg.get('horizon', 10))
    frame_stack = int(model_cfg.get('frame_stack', 2))
    loaders = make_dataloaders(data, batch_size=train_cfg['batch_size'],
                               horizon=horizon, frame_stack=frame_stack)
    print(f'[data] train={len(loaders["train"].dataset)}  '
          f'val={len(loaders["val"].dataset)}  horizon={horizon}  frame_stack={frame_stack}')

    from models.jepa import JEPAConfig
    from models.autoencoder import AEWorldModel
    jepa_cfg = JEPAConfig(
        latent_dim=int(model_cfg.get('latent_dim', 8)),
        action_latent_dim=int(model_cfg.get('action_latent_dim', 1)),
        action_encoder=model_cfg.get('action_encoder', 'linear'),
        encoder_type=model_cfg.get('encoder_type', 'cnn'),
        image_size=64,
        patch_size=int(model_cfg.get('patch_size', 8)),
        frame_stack=frame_stack,
        vit_embed_dim=int(model_cfg.get('vit_embed_dim', 128)),
        vit_depth=int(model_cfg.get('vit_depth', 4)),
        vit_num_heads=int(model_cfg.get('vit_num_heads', 4)),
        predictor_hidden_dim=int(model_cfg.get('predictor_hidden_dim', 64)),
        predictor_n_layers=int(model_cfg.get('predictor_n_layers', 2)),
        predictor_window=int(model_cfg.get('predictor_window', 3)),
    )
    model = AEWorldModel(jepa_cfg).to(device)
    n_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f'[model] AEWorldModel  {n_params:,} trainable params  '
          f'(encoder={jepa_cfg.encoder_type}, in_chans={jepa_cfg.in_chans})')

    # ── Bounou-style loss weights (the actual lambda_dmd_pixel / lambda_recon
    # mechanism lives in experiments.train_ae.{train_one_epoch,val_one_epoch} —
    # the generic training.trainer.Trainer does not implement these losses) ──
    lambda_dmd_pixel = float(train_cfg.get('lambda_dmd_pixel', 5.0))
    lambda_recon     = float(train_cfg.get('lambda_recon', 1.0))
    lambda_pred      = float(train_cfg.get('lambda_pred', 0.0))
    lambda_fp        = float(train_cfg.get('lambda_fp', 0.0))
    dmd_context_len  = int(train_cfg.get('dmd_context_len', 5))
    dmd_ridge        = float(train_cfg.get('dmd_ridge', 1e-4))
    W                = int(model_cfg.get('predictor_window', 3))
    use_vicreg       = bool(train_cfg.get('use_vicreg', False))
    vicreg_lambda    = float(train_cfg.get('vicreg_lambda', 25.0))
    vicreg_nu        = float(train_cfg.get('vicreg_nu', 1.0))
    lr               = float(train_cfg.get('lr', 1e-4))
    wd               = float(train_cfg.get('weight_decay', 1e-4))
    ckpt_every       = int(train_cfg.get('checkpoint_every', 10))

    print(f'[train] Losses: dmd_pixel×{lambda_dmd_pixel}  recon×{lambda_recon}  '
          f'pred×{lambda_pred}  fp×{lambda_fp}  '
          f'(W={W}, dmd_context={dmd_context_len}/{horizon})')

    from experiments.train_ae import train_one_epoch, val_one_epoch

    optimizer = torch.optim.Adam([
        {'params': model.encoder.parameters()},
        {'params': model.decoder.parameters()},
        {'params': model.action_encoder.parameters(), 'weight_decay': 0.0},
        {'params': model.predictor.parameters()},
    ], lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=lr * 0.1)

    save_dir = Path(args.save_dir)
    (save_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)

    # No equilibrium-frame anchor: swm/CartPoleControl-v1 has no reset_to_state.
    # lambda_fp=0 in the Bounou config -> z_star_ema only feeds an unused fp_loss.
    z_star_ema = None

    print(f'[train] training AEWorldModel for {args.epochs} epochs on swm pilot data ...')
    for epoch in range(1, args.epochs + 1):
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
        print(f'[Epoch {epoch:3d}/{args.epochs}]  '
              f'train={tr["total"]:.4f}  val={val_loss:.4f}  '
              f'dmd={tr["dmd"]:.4f}  recon={tr["recon"]:.4f}  '
              f'pred={tr["pred"]:.4f}  fp={tr["fp"]:.4f}{vic_log}  '
              f'({time.time()-t0:.1f}s)')

        if epoch % ckpt_every == 0:
            torch.save({'epoch': epoch, 'model_state': model.state_dict()},
                       str(save_dir / 'checkpoints' / f'model_ep{epoch:04d}.pt'))

    final_state = {'epoch': args.epochs, 'model_state': model.state_dict()}
    torch.save(final_state, save_dir / 'model_final.pt')
    print(f'[done] saved -> {save_dir / "model_final.pt"}')

    _save_recon_snapshot(model, loaders['val'], device, frame_stack,
                         save_dir / 'recon_snapshot.png')


if __name__ == '__main__':
    main()
