"""Baseline: JEPA pred + SIGreg — no EMA, no stop-gradient.

Random-only dataset, normalized actions/states, 3-layer predictor MLP, window=5.
Gradients flow through both predicted and target latents (no detach), so the
encoder receives gradient from every step in the multi-step unroll.

Usage:
    python experiments/generate_data.py \\
        --config configs/cartpole_jepa_baseline_random.yaml \\
        --output data/cartpole_random_seed42.h5

    python experiments/train_jepa_baseline_random.py \\
        --data data/cartpole_random_seed42.h5 --epochs 200 \\
        --save-dir results/jepa_baseline_random_seed42
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
    p.add_argument('--data',      default='data/cartpole_random_seed42.h5')
    p.add_argument('--config',    default='configs/cartpole_jepa_baseline_random.yaml')
    p.add_argument('--epochs',    type=int, default=None)
    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--save-dir',  default='results/jepa_baseline_random_seed42')
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
    if args.epochs is not None:
        train_cfg['epochs'] = args.epochs
    epochs = int(train_cfg.get('epochs', 200))

    from data.dataset import load_dataset, make_dataloaders
    print(f'[data] loading {args.data}')
    data = load_dataset(args.data)
    horizon     = int(train_cfg.get('horizon', 10))
    frame_stack = int(model_cfg.get('frame_stack', 2))

    obs_eq, state_eq = _find_near_eq_obs(data, split='train')
    print(f'[data] near-eq anchor: ||state||={np.linalg.norm(state_eq):.4f}  '
          f'state={np.round(state_eq, 4)}')

    normalize_actions = bool(train_cfg.get('normalize_actions', False))
    normalize_states  = bool(train_cfg.get('normalize_states', False))
    action_scale = float(data.get('action_scale', 1.0)) if normalize_actions else 1.0
    state_mean = data.get('state_mean') if normalize_states else None
    state_std  = data.get('state_std')  if normalize_states else None
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
                               state_mean=state_mean, state_std=state_std)
    print(f'[data] train={len(loaders["train"].dataset)}  '
          f'val={len(loaders["val"].dataset)}  horizon={horizon}  frame_stack={frame_stack}')

    from models.jepa import make_jepa
    model = make_jepa(
        variant='E-full',
        latent_dim=int(model_cfg.get('latent_dim', 8)),
        action_latent_dim=int(model_cfg.get('action_latent_dim', 1)),
        action_encoder=model_cfg.get('action_encoder', 'linear'),
        encoder_type=model_cfg.get('encoder_type', 'vit'),
        image_size=64,
        patch_size=int(model_cfg.get('patch_size', 8)),
        frame_stack=frame_stack,
        vit_embed_dim=int(model_cfg.get('vit_embed_dim', 128)),
        vit_depth=int(model_cfg.get('vit_depth', 4)),
        vit_num_heads=int(model_cfg.get('vit_num_heads', 4)),
        predictor_hidden_dim=int(model_cfg.get('predictor_hidden_dim', 64)),
        predictor_n_layers=int(model_cfg.get('predictor_n_layers', 3)),
        predictor_window=int(model_cfg.get('predictor_window', 5)),
        predictor_residual=bool(model_cfg.get('predictor_residual', False)),
    )
    model.to(device)
    n_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f'[model] JEPA (E-full)  {n_params:,} trainable params  '
          f'latent_dim={model_cfg["latent_dim"]}  frame_stack={frame_stack}  '
          f'window={model_cfg["predictor_window"]}  n_layers={model_cfg["predictor_n_layers"]}  '
          f'detach_targets={train_cfg.get("detach_targets", True)}  '
          f'use_target_encoder=False  predictor_lr_mult={train_cfg.get("predictor_lr_mult", 1.0)}')

    from training.trainer import Trainer
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(model=model, config_dict=train_cfg, gt=None,
                      save_dir=str(save_dir / 'checkpoints'), device=device, seed=args.seed)
    trainer.set_obs_eq(obs_eq)

    print(f'[train] training for {epochs} epochs ...')
    trainer.fit(loaders['train'], loaders['val'], epochs=epochs,
                checkpoint_every=int(train_cfg.get('checkpoint_every', 10)))

    torch.save(trainer.final_state, save_dir / 'model_final.pt')
    print(f'[done] saved -> {save_dir / "model_final.pt"}')
    print(f'\n[eval] run CEM evaluation with:')
    print(f'  python experiments/check_cem_planning.py \\')
    print(f'      --checkpoint {save_dir}/checkpoints/checkpoint_epoch{epochs:04d}.pt \\')
    print(f'      --config     {args.config} \\')
    print(f'      --data       {args.data} \\')
    print(f'      --q-pearson 50 --n-trials 50 --T 200 \\')
    print(f'      --out {save_dir}/cem_traj.png --gif {save_dir}/cem_episode.gif')


if __name__ == '__main__':
    main()
