"""JEPA training: prediction + endpoint action reconstruction + SIGreg + fixed-point loss.

Losses active (controlled by config lambdas):
  lambda_pred                  — multi-step JEPA prediction loss
  lambda_action_reconstruction — fixed-stride endpoint decoder φ(z_t, z_{t+T}) → [u_t…u_{t+T-1}]
  lambda_sigreg                — SIGreg slice-and-project diversity regularizer
  lambda_fp                    — fixed-point loss at equilibrium z*

State supervision is OFF (lambda_state=0) — the encoder is shaped entirely by
the predictor and action-reconstruction objectives.

Usage:
    python experiments/train_jepa_actionrec.py \\
        --data data/cartpole_visual_fs5_v4 \\
        --config configs/cartpole_jepa_actionrec.yaml \\
        --save-dir results/jepa_actionrec \\
        --epochs 500 --seed 42
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import torch
import yaml


def _find_near_eq_obs_hdf5(data_dir: str) -> np.ndarray:
    """Return the (H,W,3) uint8 observation closest to equilibrium in train.hdf5."""
    train_path = Path(data_dir) / 'train.hdf5'
    best_norm = float('inf')
    best_obs  = None
    with h5py.File(train_path, 'r') as f:
        for k in f['episodes']:
            ep     = f['episodes'][k]
            states = ep['states'][:]          # (T, 4)
            norms  = np.linalg.norm(states, axis=1)
            j = int(np.argmin(norms))
            if norms[j] < best_norm:
                best_norm = norms[j]
                best_obs  = ep['observations'][j]   # (H, W, 3) uint8
    return best_obs, best_norm


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',          default='data/cartpole_visual_fs5_v4')
    p.add_argument('--config',        default='configs/cartpole_jepa_actionrec.yaml')
    p.add_argument('--epochs',        type=int,   default=None)
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--save-dir',      default='results/jepa_actionrec')
    p.add_argument('--device',        default=None)
    p.add_argument('--batch-size',    type=int,   default=None)
    p.add_argument('--data-fraction', type=float, default=1.0)
    p.add_argument('--resume',        default=None,
                   help='Resume full training state from a .pt checkpoint')
    p.add_argument('--no-preload',    action='store_true',
                   help='Disable RAM preloading of observations (slow, use only for quick tests)')
    p.add_argument('--extra-data',    nargs='+',  default=None,
                   help='Additional HDF5 dataset dirs to concatenate with primary dataset')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = (torch.device(args.device) if args.device else
              torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Device: {device}')

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg['model']
    train_cfg = dict(cfg['training'])

    if args.epochs is not None:
        train_cfg['epochs'] = args.epochs
    if args.batch_size is not None:
        train_cfg['batch_size'] = args.batch_size
    epochs      = int(train_cfg.get('epochs', 500))
    horizon     = int(train_cfg.get('horizon', 10))
    frame_stack = int(model_cfg.get('frame_stack', 1))
    num_workers = int(train_cfg.get('num_workers', 2))

    # When resuming, honour the checkpoint's architecture so yaml changes don't break loading.
    if args.resume is not None:
        _peek = torch.load(args.resume, map_location='cpu')
        _ckpt_cfg = _peek.get('config', {})
        if _ckpt_cfg:
            _arch_keys = ('latent_dim', 'action_dim', 'action_latent_dim', 'action_encoder',
                          'encoder_type', 'patch_size', 'frame_stack', 'use_frame_diff',
                          'vit_embed_dim', 'vit_depth', 'vit_num_heads',
                          'predictor_type', 'predictor_window',
                          'predictor_hidden_dim', 'predictor_n_layers',
                          'predictor_embed_dim', 'predictor_depth',
                          'predictor_num_heads', 'predictor_mlp_ratio')
            for _k in _arch_keys:
                if _k in _ckpt_cfg and _ckpt_cfg[_k] != model_cfg.get(_k):
                    print(f'[resume] arch override: {_k}={_ckpt_cfg[_k]} '
                          f'(yaml had {model_cfg.get(_k)})')
                    model_cfg[_k] = _ckpt_cfg[_k]
        del _peek, _ckpt_cfg

    # ── Data ─────────────────────────────────────────────────────────────────
    from data.dataset import (load_discrete_dataset_meta, make_discrete_dataloaders,
                               DiscreteHDF5TrajectoryDataset)
    from torch.utils.data import ConcatDataset, DataLoader as TorchDataLoader

    normalize_actions = bool(train_cfg.get('normalize_actions', True))
    normalize_states  = bool(train_cfg.get('normalize_states',  True))

    print(f'[data] loading {args.data}')
    meta         = load_discrete_dataset_meta(args.data)
    action_scale = float(meta.get('action_scale', 1.0)) if normalize_actions else 1.0
    state_mean   = meta['state_mean'] if normalize_states else None
    state_std    = meta['state_std']  if normalize_states else None
    if normalize_states:
        print(f'[data] state_mean={np.round(state_mean, 4)}  '
              f'state_std={np.round(state_std, 4)}')

    img_size = int(model_cfg.get('image_size', 64))
    loaders  = make_discrete_dataloaders(
        args.data, batch_size=train_cfg['batch_size'],
        horizon=horizon, frame_stack=frame_stack,
        num_workers=num_workers,
        state_mean=state_mean, state_std=state_std,
        action_scale=action_scale,
        target_image_size=img_size,
        preload_obs=not args.no_preload,
        data_fraction=args.data_fraction)

    if args.extra_data:
        extra_ds = []
        for d in args.extra_data:
            print(f'[data] extra dataset: {d}')
            ds = DiscreteHDF5TrajectoryDataset(
                d, split='train', horizon=horizon, frame_stack=frame_stack,
                state_mean=state_mean, state_std=state_std, action_scale=action_scale,
                target_image_size=img_size, preload_obs=not args.no_preload,
                data_fraction=args.data_fraction)
            extra_ds.append(ds)
            print(f'[data]   -> {len(ds)} windows')
        combined = ConcatDataset([loaders['train'].dataset] + extra_ds)
        loaders['train'] = TorchDataLoader(
            combined, batch_size=train_cfg['batch_size'], shuffle=True,
            num_workers=num_workers, pin_memory=(num_workers > 0),
            drop_last=True, persistent_workers=(num_workers > 0),
            prefetch_factor=(4 if num_workers > 0 else None))

    _val_size = len(loaders['val'].dataset) if 'val' in loaders else 0
    print(f'[data] train={len(loaders["train"].dataset)}  val={_val_size}  '
          f'horizon={horizon}  frame_stack={frame_stack}')

    # Near-equilibrium observation — anchors z* exactly so fp_loss has a reliable
    # fixed point from epoch 1 (avoids the EMA threshold problem).
    print('[data] searching for near-equilibrium observation ...')
    obs_eq_np, eq_norm = _find_near_eq_obs_hdf5(args.data)
    print(f'[data] near-eq obs found: ||state||={eq_norm:.4f}')

    # ── Model ─────────────────────────────────────────────────────────────────
    from models.jepa import make_jepa
    model = make_jepa(
        variant='E-full',
        latent_dim=int(model_cfg.get('latent_dim', 8)),
        action_dim=int(model_cfg.get('action_dim', 1)),
        action_latent_dim=int(model_cfg.get('action_latent_dim', 8)),
        action_encoder=model_cfg.get('action_encoder', 'linear'),
        encoder_type=model_cfg.get('encoder_type', 'vit'),
        image_size=img_size,
        patch_size=int(model_cfg.get('patch_size', 8)),
        frame_stack=frame_stack,
        use_frame_diff=bool(model_cfg.get('use_frame_diff', False)),
        vit_embed_dim=int(model_cfg.get('vit_embed_dim', 128)),
        vit_depth=int(model_cfg.get('vit_depth', 3)),
        vit_num_heads=int(model_cfg.get('vit_num_heads', 4)),
        vit_mlp_ratio=float(model_cfg.get('vit_mlp_ratio', 2.0)),
        predictor_type=model_cfg.get('predictor_type', 'transformer'),
        predictor_window=int(model_cfg.get('predictor_window', 3)),
        predictor_embed_dim=int(model_cfg.get('predictor_embed_dim', 128)),
        predictor_depth=int(model_cfg.get('predictor_depth', 3)),
        predictor_num_heads=int(model_cfg.get('predictor_num_heads', 4)),
        predictor_mlp_ratio=float(model_cfg.get('predictor_mlp_ratio', 4.0)),
        predictor_hidden_dim=int(model_cfg.get('predictor_hidden_dim', 256)),
        predictor_n_layers=int(model_cfg.get('predictor_n_layers', 2)),
    )
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    use_fd   = model_cfg.get('use_frame_diff', False)
    ar_mode  = train_cfg.get('action_reconstruction_mode', 'fixed_stride')
    ar_T     = train_cfg.get('action_reconstruction_stride', 5)
    print(f'[model] JEPA  {n_params:,} params  '
          f'latent={model_cfg["latent_dim"]}  '
          f'encoder={model_cfg["encoder_type"]}(depth={model_cfg["vit_depth"]} '
          f'embed={model_cfg["vit_embed_dim"]} fd={use_fd})  '
          f'predictor={model_cfg["predictor_type"]}(W={model_cfg["predictor_window"]} '
          f'depth={model_cfg["predictor_depth"]})  '
          f'horizon={horizon}  '
          f'λ_pred={train_cfg.get("lambda_pred", 1.0)}  '
          f'λ_ar={train_cfg.get("lambda_action_reconstruction", 0.0)}({ar_mode}/T={ar_T})  '
          f'λ_ar_pred={train_cfg.get("lambda_action_reconstruction_pred", 0.0)}  '
          f'λ_sigreg={train_cfg.get("lambda_sigreg", 0.0)}  '
          f'λ_fp={train_cfg.get("lambda_fp", 0.0)}')

    # ── Trainer ───────────────────────────────────────────────────────────────
    from training.trainer import Trainer
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(model=model, config_dict=train_cfg, gt=None,
                      save_dir=str(save_dir / 'checkpoints'), device=device, seed=args.seed)

    # Provide exact equilibrium image so fp_loss has a reliable z* from epoch 1.
    trainer.set_obs_eq(obs_eq_np)

    # ── B_target updater for cos(B) diagnostics ───────────────────────────────
    # Recomputes the physical B direction in encoder space every checkpoint_every
    # epochs via finite differences on the real environment.  Keeps cos(B) in
    # [Diag] output meaningful as the encoder evolves during training.
    from envs.cartpole_visual import ContinuousCartpoleVisual
    env_cfg = cfg['environment']
    _b_env = ContinuousCartpoleVisual(
        frame_skip=int(env_cfg.get('frame_skip', 1)),
        image_size=int(env_cfg['image_size']),
        mass_cart=float(env_cfg['mass_cart']),
        mass_pole=float(env_cfg['mass_pole']),
        pole_length=float(env_cfg['pole_length']),
        gravity=float(env_cfg['gravity']),
        dt=float(env_cfg['dt']),
        seed=0)
    _action_scale = max(abs(float(env_cfg['action_range'][0])),
                        abs(float(env_cfg['action_range'][1])))
    _b_update_every = int(train_cfg.get('checkpoint_every', 10))
    _b_n_states     = int(train_cfg.get('b_target_n_states', 50))
    _b_init_scale   = 0.02
    _b_du_raw       = 2.0   # 2 N physical

    def _to_dev(obs_np):
        return (torch.from_numpy(obs_np).float().permute(2, 0, 1)
                .unsqueeze(0).to(device) / 255.0)

    def _update_B_target(epoch, _trainer):
        if epoch % _b_update_every != 0:
            return
        rng = np.random.RandomState(epoch)
        model.eval()
        B_list = []
        with torch.no_grad():
            for _ in range(_b_n_states):
                x0 = rng.uniform(-_b_init_scale, _b_init_scale, 4).astype(np.float32)
                obs_prev, state, _ = _b_env.reset_to_state(x0)
                obs_cur, state, _, _, _ = _b_env.step(0.0)
                _b_env.reset_to_state(state)
                obs_plus,  _, _, _, _ = _b_env.step( _b_du_raw)
                _b_env.reset_to_state(state)
                obs_minus, _, _, _, _ = _b_env.step(-_b_du_raw)
                obs_cur_t = _to_dev(obs_cur)
                zp = model.encode_obs(_to_dev(obs_plus),  obs_cur_t).squeeze(0).cpu().numpy()
                zm = model.encode_obs(_to_dev(obs_minus), obs_cur_t).squeeze(0).cpu().numpy()
                B_list.append((zp - zm) / (2.0 * _b_du_raw / _action_scale))
        model.train()
        B_target = np.mean(B_list, axis=0)
        _trainer.set_B_target(B_target)
        print(f'[B_target] epoch {epoch+1}: updated  ||B||={np.linalg.norm(B_target):.4f}')

    print(f'[train] training for {epochs} epochs ...')
    trainer.fit(loaders['train'], loaders['val'], epochs=epochs,
                checkpoint_every=int(train_cfg.get('checkpoint_every', 10)),
                resume_from=args.resume, on_epoch_start=_update_B_target)

    torch.save(trainer.final_state, save_dir / 'model_final.pt')
    print(f'[done] saved -> {save_dir / "model_final.pt"}')


if __name__ == '__main__':
    main()
