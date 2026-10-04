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
    p.add_argument('--frame-skip',       type=int, default=1,
                   help='Physics steps per action (temporal resolution)')
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
    mix.add_argument('--frac-expert',       type=float, default=None)
    mix.add_argument('--frac-noisy-005',    type=float, default=None)
    mix.add_argument('--frac-noisy-010',    type=float, default=None)
    mix.add_argument('--frac-noisy-020',    type=float, default=None)
    mix.add_argument('--frac-burst',        type=float, default=None)
    mix.add_argument('--frac-random',       type=float, default=None)
    mix.add_argument('--frac-lqr-near-eq',  type=float, default=None,
                     help='Fraction of near-equilibrium LQR+noise episodes')
    mix.add_argument('--frac-passive',       type=float, default=None,
                     help='Fraction of passive (u=0) free-fall episodes from near-eq start')
    mix.add_argument('--lqr-near-eq-angle-range', type=float, default=None,
                     help='Initial pole angle range for near-eq and passive episodes (rad)')

    # Continuous environment / friction overrides
    cont = p.add_argument_group('continuous environment and friction')
    cont.add_argument('--use-continuous-env', action='store_true', default=False,
                      help='Use ContinuousCartpoleVisual instead of discrete CartPole-v1')
    cont.add_argument('--friction-cart', type=float, default=None,
                      help='Viscous cart friction coefficient [N·s/m]')
    cont.add_argument('--friction-pole', type=float, default=None,
                      help='Viscous pole friction coefficient [N·m·s/rad]')
    cont.add_argument('--theta-threshold', type=float, default=None,
                      help='Pole angle at which episode terminates (rad). '
                           'Default: 12 deg (0.2094). Raise to e.g. 1.5708 (90 deg) '
                           'for longer passive episodes.')
    cont.add_argument('--multi-action', action='store_true', default=False,
                      help='Store frame_skip independent sub-actions per macro-step '
                           '(DINO-WM style). Requires --use-continuous-env. '
                           'Actions shape: (T, frame_skip) instead of (T,).')
    cont.add_argument('--no-done', action='store_true', default=False,
                      help='Never terminate episodes early. When the physics triggers done '
                           '(cart hits wall or pole falls), soft-reset to a new initial state '
                           'and keep collecting. Use --max-episode-steps to set episode length.')
    cont.add_argument('--max-episode-steps', type=int, default=None,
                      help='Episode length when --no-done is active (default: 200).')

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
    cfg.frame_skip      = args.frame_skip
    cfg.seed            = args.seed
    cfg.custom_reset    = args.custom_reset
    cfg.resume          = args.resume

    # Optional mixture overrides.
    # If the user supplies ANY fraction explicitly, zero all fractions first so
    # only the stated ones are active (avoids the "hidden defaults sum over 1"
    # pitfall when specifying a partial mixture).
    _mix_attrs = ['frac_expert', 'frac_noisy_005', 'frac_noisy_010',
                  'frac_noisy_020', 'frac_burst', 'frac_random',
                  'frac_lqr_near_eq', 'frac_passive']
    _mix_vals  = [args.frac_expert, args.frac_noisy_005, args.frac_noisy_010,
                  args.frac_noisy_020, args.frac_burst, args.frac_random,
                  args.frac_lqr_near_eq, args.frac_passive]
    if any(v is not None for v in _mix_vals):
        for attr in _mix_attrs:
            setattr(cfg, attr, 0.0)

    for attr, val in zip(_mix_attrs, _mix_vals):
        if val is not None:
            setattr(cfg, attr, val)
    if args.lqr_near_eq_angle_range is not None:
        cfg.lqr_near_eq_angle_range = args.lqr_near_eq_angle_range

    # Optional continuous env / friction overrides
    if args.use_continuous_env:
        cfg.use_continuous_env = True
    for attr, val in [
        ('friction_cart',    args.friction_cart),
        ('friction_pole',    args.friction_pole),
        ('theta_threshold',  args.theta_threshold),
    ]:
        if val is not None:
            setattr(cfg, attr, val)
    if args.multi_action:
        cfg.multi_action = True
    if args.no_done:
        cfg.no_done = True
    if args.max_episode_steps is not None:
        cfg.max_episode_steps = args.max_episode_steps

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
    logger.info('  output_dir         = %s', cfg.output_dir)
    logger.info('  num_transitions    = %d', cfg.num_transitions)
    logger.info('  image_size         = %d', cfg.image_size)
    logger.info('  custom_reset       = %s', cfg.custom_reset)
    logger.info('  seed               = %d', cfg.seed)
    logger.info('  use_continuous_env = %s', cfg.use_continuous_env)
    if cfg.use_continuous_env:
        logger.info('  friction_cart      = %.4f', cfg.friction_cart)
        logger.info('  friction_pole      = %.4f', cfg.friction_pole)
    logger.info('  policy mixture: expert=%.0f%% noisy5=%.0f%% noisy10=%.0f%%'
                ' noisy20=%.0f%% burst=%.0f%% random=%.0f%% lqr_near_eq=%.0f%% passive=%.0f%%',
                cfg.frac_expert * 100, cfg.frac_noisy_005 * 100,
                cfg.frac_noisy_010 * 100, cfg.frac_noisy_020 * 100,
                cfg.frac_burst * 100, cfg.frac_random * 100,
                cfg.frac_lqr_near_eq * 100, cfg.frac_passive * 100)

    generate_dataset(cfg)


if __name__ == '__main__':
    main()

