"""Pred + VICReg + local inverse dynamics — no EMA, no stop-gradient.

Random-only dataset, normalized actions/states, 3-layer predictor MLP,
window=3, horizon=30.  The inverse dynamics head (inv_head) predicts u_t
from a local window of latents: ψ(z_{t-k}, …, z_{t+k}) → u_t.
For endpoint-only action sequence reconstruction (no intermediate states)
see experiments/train_jepa_pred_endpoint_act.py.

Usage:
    python experiments/train_jepa_pred_inv_random.py \\
        --data data/cartpole_random_seed42.h5 --epochs 200 \\
        --save-dir results/jepa_pred_inv_random_seed42
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml


def _find_near_eq_obs(data, split='train'):
    """Dataset observation with smallest ||state|| — stand-in for x* = 0."""
    idx    = data['splits'][split]
    states = data['states'][idx]
    obs    = data['obs'][idx]
    j = int(np.argmin(np.linalg.norm(states, axis=1)))
    return obs[j], states[j]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',      default='data/cartpole_visual')
    p.add_argument('--config',    default='configs/cartpole_jepa_pred_inv_random.yaml')
    p.add_argument('--epochs',    type=int, default=None)
    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--save-dir',  default='results/jepa_pred_inv_random_seed42')
    p.add_argument('--device',    default=None)
    p.add_argument('--batch-size', type=int, default=None,
                   help='Override batch_size from config')
    p.add_argument('--init-checkpoint', default=None,
                   help='Load encoder+predictor weights from this .pt file before training')
    p.add_argument('--resume', default=None,
                   help='Resume full training state (model+optimizer+scheduler) from this .pt checkpoint')
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
    if args.epochs is not None:
        train_cfg['epochs'] = args.epochs
    epochs = int(train_cfg.get('epochs', 200))
    if args.batch_size is not None:
        train_cfg['batch_size'] = args.batch_size

    # When resuming, override model architecture from the checkpoint's saved config
    # so that changes to the yaml (e.g. predictor_n_layers) don't break loading.
    if args.resume is not None:
        _ckpt_peek = torch.load(args.resume, map_location='cpu')
        _ckpt_cfg  = _ckpt_peek.get('config', {})
        if _ckpt_cfg:
            _arch_keys = ('latent_dim', 'action_latent_dim', 'action_encoder',
                          'encoder_type', 'patch_size', 'frame_stack',
                          'vit_embed_dim', 'vit_depth', 'vit_num_heads',
                          'predictor_hidden_dim', 'predictor_n_layers', 'predictor_window')
            for _k in _arch_keys:
                if _k in _ckpt_cfg and _ckpt_cfg[_k] != model_cfg.get(_k):
                    print(f'[resume] arch override: {_k}={_ckpt_cfg[_k]}  '
                          f'(yaml had {model_cfg.get(_k)})')
                    model_cfg[_k] = _ckpt_cfg[_k]
        del _ckpt_peek, _ckpt_cfg

    from data.dataset import (load_dataset, load_discrete_dataset_meta,
                              make_dataloaders, make_discrete_dataloaders)
    print(f'[data] loading {args.data}')
    horizon     = int(train_cfg.get('horizon', 30))
    frame_stack = int(model_cfg.get('frame_stack', 2))
    num_workers = int(train_cfg.get('num_workers', 0))

    normalize_actions = bool(train_cfg.get('normalize_actions', False))
    normalize_states  = bool(train_cfg.get('normalize_states', False))

    if Path(args.data).is_dir():
        # Discrete CartPole HDF5 dataset — lazy loading, no obs pre-load
        meta = load_discrete_dataset_meta(args.data)
        action_scale = float(meta.get('action_scale', 1.0)) if normalize_actions else 1.0
        state_mean   = meta['state_mean'] if normalize_states else None
        state_std    = meta['state_std']  if normalize_states else None
        if normalize_states:
            print(f'[data] state normalization: mean={np.round(state_mean, 4)}  '
                  f'std={np.round(state_std, 4)}')
        loaders = make_discrete_dataloaders(
            args.data, batch_size=train_cfg['batch_size'],
            horizon=horizon, frame_stack=frame_stack,
            num_workers=num_workers,
            state_mean=state_mean, state_std=state_std)
        obs_eq, state_eq = None, np.zeros(4, dtype=np.float32)
    else:
        # Legacy flat HDF5 format
        data = load_dataset(args.data)
        obs_eq, state_eq = _find_near_eq_obs(data, split='train')
        print(f'[data] near-eq anchor: ||state||={np.linalg.norm(state_eq):.4f}  '
              f'state={np.round(state_eq, 4)}')
        action_scale = float(data.get('action_scale', 1.0)) if normalize_actions else 1.0
        state_mean   = data.get('state_mean') if normalize_states else None
        state_std    = data.get('state_std')  if normalize_states else None
        if normalize_actions:
            print(f'[data] action normalization: scale={action_scale}')
        if normalize_states and state_mean is not None:
            print(f'[data] state normalization: mean={np.round(state_mean, 4)}  '
                  f'std={np.round(state_std, 4)}')
        loaders = make_dataloaders(data, batch_size=train_cfg['batch_size'],
                                   horizon=horizon, frame_stack=frame_stack,
                                   obs_eq=obs_eq,
                                   n_eq_selfloop=int(cfg.get('data', {}).get('n_eq_selfloop', 0)),
                                   action_scale=action_scale,
                                   state_mean=state_mean, state_std=state_std,
                                   num_workers=num_workers)

    print(f'[data] train={len(loaders["train"].dataset)}  '
          f'val={len(loaders["val"].dataset)}  horizon={horizon}  frame_stack={frame_stack}')

    from models.jepa import make_jepa
    model = make_jepa(
        variant='E-full',
        latent_dim=int(model_cfg.get('latent_dim', 8)),
        action_latent_dim=int(model_cfg.get('action_latent_dim', 1)),
        action_encoder=model_cfg.get('action_encoder', 'linear'),
        encoder_type=model_cfg.get('encoder_type', 'vit'),
        image_size=int(model_cfg.get('image_size', 64)),
        patch_size=int(model_cfg.get('patch_size', 8)),
        frame_stack=frame_stack,
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
    model.to(device)

    if args.init_checkpoint is not None:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        # Support both model_final.pt (flat state dict) and checkpoint_epochNNNN.pt
        # (which wraps the state dict under 'model_state')
        state = ckpt.get('model_state', ckpt.get('model_state_dict', ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f'[init] loaded encoder+predictor from {args.init_checkpoint}')
        if missing:
            print(f'[init]   missing keys  : {missing}')
        if unexpected:
            print(f'[init]   unexpected keys: {unexpected}')

    n_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    pred_type = model_cfg.get('predictor_type', 'mlp')
    pred_info  = (f'depth={model_cfg["predictor_depth"]}' if pred_type == 'transformer'
                  else f'n_layers={model_cfg.get("predictor_n_layers", "?")}')
    print(f'[model] JEPA (E-full)  {n_params:,} trainable params  '
          f'latent_dim={model_cfg["latent_dim"]}  frame_stack={frame_stack}  '
          f'predictor={pred_type}  window={model_cfg["predictor_window"]}  {pred_info}  '
          f'horizon={horizon}  lambda_inv={train_cfg.get("lambda_inv", 0.0)}  '
          f'detach_targets={train_cfg.get("detach_targets", True)}')

    from training.trainer import Trainer
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(model=model, config_dict=train_cfg, gt=None,
                      save_dir=str(save_dir / 'checkpoints'), device=device, seed=args.seed)
    if obs_eq is not None:
        trainer.set_obs_eq(obs_eq)

    print(f'[train] training for {epochs} epochs ...')
    trainer.fit(loaders['train'], loaders['val'], epochs=epochs,
                checkpoint_every=int(train_cfg.get('checkpoint_every', 10)),
                resume_from=args.resume)

    torch.save(trainer.final_state, save_dir / 'model_final.pt')
    print(f'[done] saved -> {save_dir / "model_final.pt"}')


if __name__ == '__main__':
    main()
