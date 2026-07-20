#!/usr/bin/env python
"""CLI entry point for discrete CartPole-v1 visual dataset generation.

Usage
-----
    python experiments/generate_cartpole_dataset.py \\
        --output-dir data/cartpole_visual \\
        --num-transitions 200000 \\
        --image-size 64 \\
        --seed 0 \\
        --custom-reset

Ablation sizes: pass --num-transitions 25000 / 50000 / 100000 / 200000.

All parameters can also be supplied via --config pointing to a YAML file;
explicit CLI flags override config values.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.cartpole_discrete_dataset import DatasetConfig, generate_dataset

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Generate discrete CartPole-v1 visual dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--output-dir',       default='data/cartpole_visual',
                   help='Directory for train/val/test HDF5 files and metadata')
    p.add_argument('--num-transitions',  type=int, default=200_000,
                   help='Target transition count (25k / 50k / 100k / 200k)')
    p.add_argument('--image-size',       type=int, default=64,
                   help='Pixel height and width of stored observations')
    p.add_argument('--seed',             type=int, default=0,
                   help='Master random seed')
    p.add_argument('--custom-reset',     action='store_true', default=True,
                   help='Use wider initial-state distribution')
    p.add_argument('--no-custom-reset',  dest='custom_reset', action='store_false',
                   help='Use gymnasium default narrow reset (±0.05)')
    p.add_argument('--resume',           action='store_true',
                   help='Resume an interrupted generation run')
    p.add_argument('--config',           default=None,
                   help='Optional YAML config (CLI flags override it)')

    # Policy mixture overrides
    mix = p.add_argument_group('policy mixture fractions (must sum to 1.0)')
    mix.add_argument('--frac-expert',    type=float, default=None)
    mix.add_argument('--frac-noisy-005', type=float, default=None)
    mix.add_argument('--frac-noisy-010', type=float, default=None)
    mix.add_argument('--frac-noisy-020', type=float, default=None)
    mix.add_argument('--frac-burst',     type=float, default=None)
    mix.add_argument('--frac-random',    type=float, default=None)

    # Reset range overrides
    rst = p.add_argument_group('custom reset ranges')
    rst.add_argument('--cart-pos-range',   type=float, default=None)
    rst.add_argument('--cart-vel-range',   type=float, default=None)
    rst.add_argument('--pole-angle-range', type=float, default=None)
    rst.add_argument('--pole-vel-range',   type=float, default=None)

    return p.parse_args()


def main() -> None:
    args = _parse()

    # Start from YAML if given, then apply CLI overrides
    cfg = DatasetConfig.from_yaml(args.config) if args.config else DatasetConfig()

    cfg.output_dir      = args.output_dir
    cfg.num_transitions = args.num_transitions
    cfg.image_size      = args.image_size
    cfg.seed            = args.seed
    cfg.custom_reset    = args.custom_reset
    cfg.resume          = args.resume

    # Optional mixture overrides
    for attr, val in [
        ('frac_expert',    args.frac_expert),
        ('frac_noisy_005', args.frac_noisy_005),
        ('frac_noisy_010', args.frac_noisy_010),
        ('frac_noisy_020', args.frac_noisy_020),
        ('frac_burst',     args.frac_burst),
        ('frac_random',    args.frac_random),
    ]:
        if val is not None:
            setattr(cfg, attr, val)

    # Optional reset range overrides
    for attr, val in [
        ('cart_pos_range',   args.cart_pos_range),
        ('cart_vel_range',   args.cart_vel_range),
        ('pole_angle_range', args.pole_angle_range),
        ('pole_vel_range',   args.pole_vel_range),
    ]:
        if val is not None:
            setattr(cfg, attr, val)

    logger.info('Configuration:')
    logger.info('  output_dir      = %s', cfg.output_dir)
    logger.info('  num_transitions = %d', cfg.num_transitions)
    logger.info('  image_size      = %d', cfg.image_size)
    logger.info('  custom_reset    = %s', cfg.custom_reset)
    logger.info('  seed            = %d', cfg.seed)
    logger.info('  policy mixture: expert=%.0f%% noisy5=%.0f%% noisy10=%.0f%%'
                ' noisy20=%.0f%% burst=%.0f%% random=%.0f%%',
                cfg.frac_expert * 100, cfg.frac_noisy_005 * 100,
                cfg.frac_noisy_010 * 100, cfg.frac_noisy_020 * 100,
                cfg.frac_burst * 100, cfg.frac_random * 100)

    generate_dataset(cfg)


if __name__ == '__main__':
    main()
