"""JEPA training: prediction + state supervision + SIGreg + fixed-point loss.

Losses active (controlled by config lambdas):
  lambda_pred    — multi-step JEPA prediction loss
  lambda_state   — linear state decoder supervision (theta wrapped to [-pi, pi])
  lambda_sigreg  — SIGreg slice-and-project diversity regularizer
  lambda_fp      — fixed-point loss at equilibrium z*

Usage:
    python experiments/train_jepa_state.py \\
        --data data/cartpole_visual_fs5_v4 \\
        --config configs/cartpole_jepa_v11_difenc.yaml \\
        --save-dir results/jepa_v11_difenc \\
        --epochs 500 --seed 42
"""
from __future__ import annotations
import argparse, sys, warnings
# Suppress FutureWarning from PyTorch's internal gradient-checkpoint code
# (torch/utils/checkpoint.py uses the deprecated torch.cpu.amp.autocast API).
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')
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
    p.add_argument('--data',         default='data/cartpole_visual_fs5_v4')
    p.add_argument('--config',       default='configs/cartpole_jepa_v11_difenc.yaml')
    p.add_argument('--epochs',       type=int,   default=None)
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--save-dir',     default='results/jepa_v11_difenc')
    p.add_argument('--device',       default=None)
    p.add_argument('--batch-size',   type=int,   default=None)
    p.add_argument('--data-fraction',type=float, default=1.0)
    p.add_argument('--resume',       default=None,
                   help='Resume full training state from a .pt checkpoint')
    p.add_argument('--no-preload',   action='store_true',
                   help='Disable RAM preloading of observations (slow, use only for quick tests)')
    p.add_argument('--extra-data',   nargs='+',  default=None,
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
    use_frame_diff = bool(model_cfg.get('use_frame_diff', False))
    if use_frame_diff and frame_stack != 1:
        raise ValueError(
            'use_frame_diff=true requires frame_stack=1: the encoder input is '
            'constructed explicitly as [o_{t-1}, o_t, o_t-o_{t-1}] (9 channels).')
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
                          'predictor_activation',
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
        train_cfg['equilibrium_state_target'] = (
            (-state_mean / state_std).astype(np.float32).tolist())
        train_cfg['state_normalization_mean'] = (
            state_mean.astype(np.float32).tolist())
        train_cfg['state_normalization_std'] = (
            state_std.astype(np.float32).tolist())
    else:
        train_cfg['equilibrium_state_target'] = [0.0, 0.0, 0.0, 0.0]
        train_cfg['state_normalization_mean'] = [0.0, 0.0, 0.0, 0.0]
        train_cfg['state_normalization_std'] = [1.0, 1.0, 1.0, 1.0]
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
        data_fraction=args.data_fraction,
        balanced_sampling=bool(train_cfg.get('balanced_sampling', False)),
        angle_bin_edges=tuple(
            train_cfg.get('angle_bin_edges', [0.05, 0.2, 0.6])))

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

    # Near-equilibrium observation — anchors z* exactly so fp_loss and checkpoint
    # diagnostics always have a reliable fixed point (avoids the EMA threshold problem).
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
        predictor_activation=model_cfg.get('predictor_activation', 'elu'),
    )
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    use_fd = model_cfg.get('use_frame_diff', False)
    lambda_B_align = float(train_cfg.get('lambda_B_align', 0.0))
    print(f'[model] JEPA  {n_params:,} params  '
          f'latent={model_cfg["latent_dim"]}  '
          f'encoder={model_cfg["encoder_type"]}(depth={model_cfg["vit_depth"]} '
          f'embed={model_cfg["vit_embed_dim"]} fd={use_fd})  '
          f'predictor={model_cfg["predictor_type"]}(W={model_cfg["predictor_window"]} '
          f'depth={model_cfg["predictor_depth"]})  '
          f'horizon={horizon}  '
          f'λ_pred={train_cfg.get("lambda_pred",1.0)}  '
          f'λ_state={train_cfg.get("lambda_state",0.0)}  '
          f'λ_sigreg={train_cfg.get("lambda_sigreg",0.0)}  '
          f'λ_fp={train_cfg.get("lambda_fp",0.0)}  '
          f'λ_B={lambda_B_align}')

    # ── Trainer ───────────────────────────────────────────────────────────────
    from training.trainer import Trainer
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(model=model, config_dict=train_cfg, gt=None,
                      save_dir=str(save_dir / 'checkpoints'), device=device, seed=args.seed)

    # Provide exact equilibrium image so fp_loss has a reliable z* from epoch 1.
    trainer.set_obs_eq(obs_eq_np)

    # ── B_target updater (for lambda_B_align) ────────────────────────────────
    # Recomputes the physical B direction in encoder space every few epochs using
    # finite differences on the real environment.  Correct prev-frame handling for
    # use_frame_diff: encode_obs(obs_after_force, obs_before_force).
    on_epoch_start = None
    # Building B_target requires many extra environment rollouts. Run it only
    # when it contributes to the loss or is explicitly requested as a diagnostic.
    compute_B_diagnostic = bool(train_cfg.get('compute_B_diagnostic', False))
    if ((lambda_B_align > 0 or compute_B_diagnostic)
            and 'environment' in cfg and 'image_size' in cfg['environment']):
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
        _b_update_every = int(train_cfg.get('b_target_update_every', 1))
        _b_n_states     = int(train_cfg.get('b_target_n_states', 50))
        _b_init_scale   = 0.02
        _b_force_levels = [float(v) for v in train_cfg.get(
            'b_target_force_levels', [1.0, 2.0, 3.0])]
        _b_impulse_horizon = int(train_cfg.get('b_target_impulse_horizon', 3))

        def _to_dev(obs_np):
            return (torch.from_numpy(obs_np).float().permute(2, 0, 1)
                    .unsqueeze(0).to(device) / 255.0)

        def _update_B_target(epoch, _trainer):
            if epoch % _b_update_every != 0:
                return
            rng = np.random.RandomState(epoch)
            model.eval()
            B_impulse_list = []
            with torch.no_grad():
                for _ in range(_b_n_states):
                    x0 = rng.uniform(-_b_init_scale, _b_init_scale, 4).astype(np.float32)
                    obs_prev, state, _ = _b_env.reset_to_state(x0)
                    # One passive step so there is actual motion in the diff channel
                    obs_cur, state, _, _, _ = _b_env.step(0.0)
                    # Symmetric impulse response: apply +/-du at the first
                    # macro-step, then zero force. Matching later responses avoids
                    # relying on a nearly subpixel one-step displacement.
                    for _b_du_raw in _b_force_levels:
                        branches = []
                        for _sign in (1.0, -1.0):
                            _b_env.reset_to_state(state)
                            _prev_obs = obs_cur
                            _zs = []
                            for _h in range(_b_impulse_horizon):
                                _u_raw = _sign * _b_du_raw if _h == 0 else 0.0
                                _next_obs, _, _, _done, _ = _b_env.step(_u_raw)
                                _z = model.encode_obs(
                                    _to_dev(_next_obs), _to_dev(_prev_obs)
                                ).squeeze(0).cpu().numpy()
                                _zs.append(_z)
                                _prev_obs = _next_obs
                                if _done:
                                    break
                            if len(_zs) != _b_impulse_horizon:
                                break
                            branches.append(np.stack(_zs))
                        if len(branches) == 2:
                            B_impulse_list.append(
                                (branches[0] - branches[1])
                                / (2.0 * _b_du_raw / _action_scale))
            model.train()
            if not B_impulse_list:
                raise RuntimeError('No valid B impulse-response targets were generated')
            B_target = np.mean(B_impulse_list, axis=0)  # (H_impulse, latent_dim)
            _trainer.set_B_target(B_target)
            _norms = np.linalg.norm(B_target, axis=1)
            print(f'[B_align] epoch {epoch+1}: updated impulse target  '
                  f'||[B,AB,A2B]||={np.round(_norms, 4).tolist()}')

        on_epoch_start = _update_B_target
        if lambda_B_align > 0:
            print(f'[B_align] enabled  λ={lambda_B_align}  '
                  f'update_every={_b_update_every}  n_states={_b_n_states}  '
                  f'forces={_b_force_levels}  H_imp={_b_impulse_horizon}')
        elif compute_B_diagnostic:
            print(f'[B_align] diagnostic only (λ=0)  '
                  f'update_every={_b_update_every}  n_states={_b_n_states}')

    print(f'[train] training for {epochs} epochs ...')
    trainer.fit(loaders['train'], loaders['val'], epochs=epochs,
                checkpoint_every=int(train_cfg.get('checkpoint_every', 10)),
                resume_from=args.resume, on_epoch_start=on_epoch_start)

    # Preserve the historical name while making checkpoint semantics explicit.
    torch.save(trainer.final_state, save_dir / 'model_final.pt')
    torch.save(trainer.final_state, save_dir / 'model_last.pt')
    torch.save(model.state_dict(), save_dir / 'model_selected.pt')
    if getattr(trainer, 'best_state', None) is not None:
        torch.save(trainer.best_state, save_dir / 'model_best_selection.pt')
    print(f'[done] saved last     -> {save_dir / "model_last.pt"}')
    print(f'[done] saved selected -> {save_dir / "model_selected.pt"}')
    print(f'[done] selection metric: {getattr(trainer, "selection_metric", "total_loss")}')


if __name__ == '__main__':
    main()
