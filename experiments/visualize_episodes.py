#!/usr/bin/env python
"""Visualize episode trajectories from the episode-centric dataset.

For each data type (random, lqr, prbs, passive):
  LEFT  — filmstrip: N equally-spaced frames from one sample episode
  RIGHT — theta (pole angle) over time for multiple episodes

The dashed red lines mark ±12° (0.21 rad), the gym done threshold that
previously caused mid-episode resets.  With the new episode-centric
generation, trajectories freely cross this boundary.

Usage (run from repo root):
    python experiments/visualize_episodes.py
    python experiments/visualize_episodes.py --config configs/cartpole_v2_fullspec.yaml
    python experiments/visualize_episodes.py --n-episodes 5 --n-strip 10
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',     default='configs/cartpole_ae_baseline.yaml')
    p.add_argument('--n-episodes', type=int, default=5,
                   help='Episodes per type shown in trajectory plot')
    p.add_argument('--n-strip',    type=int, default=10,
                   help='Frames shown in each filmstrip row')
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--out',        default='results/episode_viz.png')
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    env_cfg  = cfg['environment']
    data_cfg = cfg['data']
    dt       = float(env_cfg.get('dt', 0.02)) * int(env_cfg.get('frame_skip', 1))

    from envs.cartpole_visual import ContinuousCartpoleVisual
    from data.dataset import _compute_lqr_gain, _collect_episode

    env = ContinuousCartpoleVisual(
        frame_skip=int(env_cfg.get('frame_skip', 1)),
        image_size=int(env_cfg.get('image_size', 64)),
        mass_cart=float(env_cfg.get('mass_cart', 1.0)),
        mass_pole=float(env_cfg.get('mass_pole', 0.1)),
        pole_length=float(env_cfg.get('pole_length', 0.5)),
        gravity=float(env_cfg.get('gravity', 9.8)),
        action_range=tuple(env_cfg.get('action_range', (-10, 10))),
        seed=args.seed,
    )
    rng      = np.random.RandomState(args.seed)
    lqr_gain = _compute_lqr_gain()
    a_lo, a_hi = float(env_cfg['action_range'][0]), float(env_cfg['action_range'][1])

    # (label, mode, ep_len, init_range, hex-color)
    types = [
        ('Random  (u ~ Uniform[-10, 10])',
         'random',
         int(data_cfg.get('random_ep_len', 200)),
         float(data_cfg.get('random_init_range', 1.0)),
         '#e74c3c'),
        ('LQR  (pure, no noise)',
         'lqr',
         int(data_cfg.get('lqr_ep_len', 100)),
         float(data_cfg.get('lqr_init_range', 0.10)),
         '#2ecc71'),
        ('PRBS  (persistent excitation near eq)',
         'prbs',
         int(data_cfg.get('pe_ep_len', 40)),
         float(data_cfg.get('pe_init_range', 0.05)),
         '#9b59b6'),
        ('Passive  (u = 0, natural divergence)',
         'passive',
         int(data_cfg.get('passive_ep_len', 50)),
         float(data_cfg.get('passive_init_range', 0.05)),
         '#f39c12'),
    ]

    N_TYPES = len(types)
    N_EP    = args.n_episodes
    N_STRIP = args.n_strip

    # ── Collect episodes ──────────────────────────────────────────────────
    print('[viz] Collecting episodes...')
    collected: dict[str, list[dict]] = {}
    for label, mode, ep_len, init_range, color in types:
        print(f'  [{mode}]  {N_EP} ep × {ep_len} steps  '
              f'(init_range={init_range})')
        eps = []
        for _ in range(N_EP):
            ep = _collect_episode(
                env, ep_len, mode, lqr_gain,
                a_lo, a_hi, init_range,
                lqr_noise_std=float(data_cfg.get('lqr_noise_std', 0.25)),
                rng=rng,
                pe_action_amplitude=float(data_cfg.get('pe_action_amplitude', 3.0)),
                pe_flip_prob=float(data_cfg.get('pe_flip_prob', 0.15)),
            )
            eps.append(ep)
        collected[mode] = eps
    env.close()

    # Print summary statistics
    print('\n[viz] Trajectory statistics:')
    for label, mode, ep_len, init_range, color in types:
        thetas = np.concatenate([ep['states'][:, 2] for ep in collected[mode]])
        print(f'  {mode:<10}  ep_len={ep_len}  '
              f'max|θ|={np.abs(thetas).max():.3f} rad  '
              f'(gym limit: 0.21 rad)')

    # ── Figure layout ─────────────────────────────────────────────────────
    # Each episode type gets one row split into filmstrip (left) + trajectory (right)
    FIG_W = 2.5 + N_STRIP * 1.15 + 4.5   # filmstrip + traj col
    FIG_H = 3.8 * N_TYPES

    BG      = '#0f0f1e'
    PANEL   = '#1a1a30'
    GRID_C  = '#2a2a4a'
    TICK_C  = '#8888aa'
    DONE_C  = '#ff4444'

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(BG)

    outer = gridspec.GridSpec(N_TYPES, 2, figure=fig,
                              hspace=0.55, wspace=0.18,
                              width_ratios=[N_STRIP, 3.5],
                              left=0.02, right=0.98,
                              top=0.95, bottom=0.04)

    for row, (label, mode, ep_len, init_range, color) in enumerate(types):
        eps = collected[mode]
        ep0 = eps[0]
        obs = ep0['obs']         # (T, H, W, 3) uint8
        T   = len(obs)

        # ── Filmstrip ─────────────────────────────────────────────────────
        inner = gridspec.GridSpecFromSubplotSpec(
            1, N_STRIP, subplot_spec=outer[row, 0], wspace=0.03)

        frame_indices = np.linspace(0, T - 1, N_STRIP, dtype=int)
        for col, fi in enumerate(frame_indices):
            ax_f = fig.add_subplot(inner[0, col])
            ax_f.set_facecolor(PANEL)
            ax_f.imshow(obs[fi], interpolation='nearest')
            ax_f.axis('off')
            t_s = fi * dt
            ax_f.set_title(f'{t_s:.1f}s', fontsize=7.5,
                           color=TICK_C, pad=2)
            # Theta annotation on frame
            th_val = ep0['states'][fi, 2]
            ax_f.text(0.5, 0.02, f'θ={th_val:+.2f}',
                      transform=ax_f.transAxes,
                      fontsize=6.5, color='white', ha='center', va='bottom',
                      bbox=dict(facecolor='black', alpha=0.55, pad=1.5,
                                edgecolor='none'))

        # Episode-type label above the filmstrip row
        ax_label = fig.add_subplot(outer[row, 0])
        ax_label.axis('off')
        ax_label.set_facecolor('none')
        ax_label.text(0.0, 1.07, label,
                      transform=ax_label.transAxes,
                      fontsize=11, fontweight='bold',
                      color=color, va='bottom', ha='left')
        ax_label.text(0.0, 1.02,
                      f'{N_EP} episodes  ·  {ep_len} steps each  ·  '
                      f'init_range = ±{init_range:.2f} rad',
                      transform=ax_label.transAxes,
                      fontsize=8, color=TICK_C, va='bottom', ha='left')

        # ── Trajectory plot ───────────────────────────────────────────────
        ax_t = fig.add_subplot(outer[row, 1])
        ax_t.set_facecolor(PANEL)

        for ep_i, ep in enumerate(eps):
            th = ep['states'][:, 2]
            t  = np.arange(len(th)) * dt
            alpha = max(0.35, 1.0 - ep_i * 0.12)
            ax_t.plot(t, th, color=color, alpha=alpha,
                      linewidth=1.6, label=f'ep {ep_i}' if ep_i == 0 else None)
            # Mark start
            ax_t.scatter([0], [th[0]], color=color, s=25, zorder=5,
                         alpha=alpha)

        # Gym done threshold
        ax_t.axhline( 0.21, color=DONE_C, lw=1.3, ls='--', alpha=0.75,
                      label='±12° gym limit')
        ax_t.axhline(-0.21, color=DONE_C, lw=1.3, ls='--', alpha=0.75)
        ax_t.axhline(0, color='white', lw=0.6, alpha=0.25)

        max_th = max(np.abs(ep['states'][:, 2]).max() for ep in eps)
        ylim   = max(np.pi * 0.7, max_th * 1.2)
        ax_t.set_ylim(-ylim, ylim)
        ax_t.set_xlim(0, (T - 1) * dt)

        ax_t.set_xlabel('time  (s)', fontsize=8.5, color=TICK_C)
        ax_t.set_ylabel('θ  (rad)',  fontsize=8.5, color=TICK_C)
        ax_t.tick_params(colors=TICK_C, labelsize=7.5)
        for sp in ax_t.spines.values():
            sp.set_color(GRID_C)
        ax_t.grid(True, alpha=0.25, color=GRID_C, linewidth=0.8)
        ax_t.legend(fontsize=7.5, loc='upper right',
                    facecolor='#0d0d1e', labelcolor='#cccccc',
                    edgecolor=GRID_C, framealpha=0.85)

        # Stats in corner
        max_th_all = max(np.abs(ep['states'][:, 2]).max() for ep in eps)
        ax_t.text(0.02, 0.97,
                  f'max |θ| = {max_th_all:.2f} rad  '
                  f'({np.degrees(max_th_all):.0f}°)',
                  transform=ax_t.transAxes,
                  fontsize=7.5, color=color, va='top', ha='left',
                  bbox=dict(facecolor='black', alpha=0.4, pad=2,
                            edgecolor='none'))

    fig.suptitle('Episode-Centric Dataset  —  Trajectories by Data Type',
                 fontsize=14, fontweight='bold', color='white', y=0.98)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'\n[viz] → {out}')


if __name__ == '__main__':
    main()
