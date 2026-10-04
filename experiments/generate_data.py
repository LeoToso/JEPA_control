#!/usr/bin/env python
"""Standalone dataset generator — run before training to pre-build the h5 file.

Saves to the same path the training scripts look for, so running this first
means train_ae.py / run_experiment.py will load rather than re-generate.

Usage (from repo root):
    python experiments/generate_data.py --config configs/cartpole_ae_baseline.yaml
    python experiments/generate_data.py --config configs/cartpole_v2_fullspec.yaml
    python experiments/generate_data.py --config configs/cartpole_ae_baseline.yaml --seed 44
    python experiments/generate_data.py --config configs/cartpole_ae_baseline.yaml --force
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',   default='configs/cartpole_ae_baseline.yaml')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--seed',     type=int, default=42)
    p.add_argument('--output',   default=None,
                   help='Custom h5 output path (overrides auto-naming)')
    p.add_argument('--force',    action='store_true',
                   help='Regenerate even if h5 already exists')
    p.add_argument('--n-random', type=int, default=None,
                   help='Override n_random_episodes from config')
    p.add_argument('--n-lqr',   type=int, default=None,
                   help='Override n_lqr_episodes from config')
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_cfg    = cfg['environment']
    data_cfg   = cfg['data']
    frame_skip = int(env_cfg.get('frame_skip', 1))

    if args.output:
        h5_path = Path(args.output)
    else:
        config_stem = Path(args.config).stem
        if 'ae' in config_stem:
            h5_name = f'cartpole_ae_ep_seed{args.seed}.h5'
        else:
            h5_name = f'cartpole_v2_ep_fs{frame_skip}_seed{args.seed}.h5'
        h5_path = Path(args.data_dir) / h5_name
    if h5_path.exists() and not args.force:
        print(f'[data] {h5_path} already exists. Use --force to regenerate.')
        return

    Path(args.data_dir).mkdir(parents=True, exist_ok=True)

    if args.n_random is not None:
        data_cfg['n_random_episodes'] = args.n_random
    if args.n_lqr is not None:
        data_cfg['n_lqr_episodes'] = args.n_lqr

    from data.dataset import generate_dataset
    generate_dataset(
        n_random_episodes=int(data_cfg.get('n_random_episodes', 50)),
        random_ep_len=int(data_cfg.get('random_ep_len', 200)),
        random_init_range=float(data_cfg.get('random_init_range', 1.0)),
        n_lqr_episodes=int(data_cfg.get('n_lqr_episodes', 50)),
        lqr_ep_len=int(data_cfg.get('lqr_ep_len', 200)),
        lqr_init_range=float(data_cfg.get('lqr_init_range', 0.30)),
        lqr_noise_std=float(data_cfg.get('lqr_noise_std', 0.25)),
        n_equilibrium=int(data_cfg.get('n_equilibrium', 500)),
        eq_ep_len=int(data_cfg.get('eq_ep_len', 50)),
        eq_init_range=float(data_cfg.get('eq_init_range', 0.002)),
        eq_noise_std=float(data_cfg.get('eq_noise_std', 0.001)),
        n_pe_episodes=int(data_cfg.get('n_pe_episodes', 0)),
        pe_ep_len=int(data_cfg.get('pe_ep_len', 40)),
        pe_init_range=float(data_cfg.get('pe_init_range', 0.05)),
        pe_action_amplitude=float(data_cfg.get('pe_action_amplitude', 3.0)),
        pe_action_amplitudes=data_cfg.get('pe_action_amplitudes'),
        pe_flip_prob=float(data_cfg.get('pe_flip_prob', 0.15)),
        n_passive_episodes=int(data_cfg.get('n_passive_episodes', 0)),
        passive_ep_len=int(data_cfg.get('passive_ep_len', 50)),
        passive_init_range=float(data_cfg.get('passive_init_range', 0.05)),
        n_angle_episodes=int(data_cfg.get('n_angle_episodes', 0)),
        angle_ep_len=int(data_cfg.get('angle_ep_len', 12)),
        angle_min=float(data_cfg.get('angle_min', 0.1)),
        angle_max=float(data_cfg.get('angle_max', 1.0)),
        angle_x_range=float(data_cfg.get('angle_x_range', 0.1)),
        angle_velocity_range=float(data_cfg.get('angle_velocity_range', 0.2)),
        angle_action_noise_std=float(data_cfg.get('angle_action_noise_std', 0.5)),
        train_frac=float(data_cfg.get('train_frac', 0.8)),
        val_frac=float(data_cfg.get('val_frac', 0.1)),
        frame_skip=frame_skip,
        image_size=int(env_cfg['image_size']),
        action_range=tuple(env_cfg['action_range']),
        max_abs_x=float(data_cfg.get('max_abs_x', 2.2)),
        max_abs_theta=float(data_cfg.get('max_abs_theta', 1.2)),
        save_path=str(h5_path),
        seed=args.seed,
    )
    print(f'[data] Saved → {h5_path}')


if __name__ == '__main__':
    main()

