#!/usr/bin/env python
"""Visualize episodes from a saved swm-collected HDF5 dataset
(data/cartpole_swm_pilot.h5, produced by data/collect_swm_cartpole.py).

Unlike experiments/visualize_episodes.py (which re-simulates episodes live
from ContinuousCartpoleVisual), this reads directly from the saved .h5 —
useful for inspecting what the swm renderer actually produced (pole size,
background composition, action effects) before/after AE training.

For each of a few sample episodes:
  TOP    — filmstrip: N equally-spaced frames
  BOTTOM — state trajectory (cart position x, pole angle theta) over time

Usage (run from repo root):
    python experiments/visualize_swm_dataset.py
    python experiments/visualize_swm_dataset.py --data data/cartpole_swm_pilot.h5 \\
        --n-episodes 4 --n-strip 8
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',       default='data/cartpole_swm_pilot.h5')
    p.add_argument('--n-episodes', type=int, default=4, help='Number of sample episodes to show')
    p.add_argument('--n-strip',    type=int, default=8, help='Frames per filmstrip row')
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--out',        default='results/swm_dataset_viz.png')
    args = p.parse_args()

    from data.dataset import load_dataset
    print(f'[viz] loading {args.data}')
    data = load_dataset(args.data)
    obs        = data['obs']           # (N, H, W, 3) uint8
    states     = data['states']        # (N, 4) float32  [x, xdot, theta, thetadot]
    actions    = data['actions']       # (N, 1) float32  in {-1, +1}
    episode_ids = data['episode_ids']  # (N,) int32

    unique_eps = np.unique(episode_ids)
    rng = np.random.RandomState(args.seed)
    chosen = rng.choice(unique_eps, size=min(args.n_episodes, len(unique_eps)), replace=False)
    chosen.sort()

    print(f'[viz] dataset: {len(obs):,} transitions, {len(unique_eps)} episodes, '
          f'{len(obs) / len(unique_eps):.1f} avg steps/episode')
    print(f'[viz] showing episodes: {chosen.tolist()}')

    N_EP    = len(chosen)
    N_STRIP = args.n_strip

    BG, PANEL, GRID_C, TICK_C = '#0f0f1e', '#1a1a30', '#2a2a4a', '#8888aa'
    fig = plt.figure(figsize=(2.5 + N_STRIP * 1.6, 4.2 * N_EP))
    fig.patch.set_facecolor(BG)
    outer = gridspec.GridSpec(N_EP, 1, figure=fig, hspace=0.6,
                              left=0.06, right=0.98, top=0.94, bottom=0.05)

    for row, ep_id in enumerate(chosen):
        mask = episode_ids == ep_id
        ep_obs    = obs[mask]
        ep_states = states[mask]
        ep_act    = actions[mask, 0]
        T = len(ep_obs)

        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[row], height_ratios=[1.3, 1.0], hspace=0.35)

        # ── Filmstrip ─────────────────────────────────────────────────────
        strip_gs = gridspec.GridSpecFromSubplotSpec(
            1, N_STRIP, subplot_spec=inner[0], wspace=0.03)
        frame_idx = np.linspace(0, T - 1, N_STRIP, dtype=int)
        for col, fi in enumerate(frame_idx):
            ax_f = fig.add_subplot(strip_gs[0, col])
            ax_f.set_facecolor(PANEL)
            ax_f.imshow(ep_obs[fi], interpolation='bilinear')
            ax_f.axis('off')
            ax_f.set_title(f't={fi}', fontsize=7.5, color=TICK_C, pad=2)
            th = ep_states[fi, 2] if ep_states.shape[1] > 2 else float('nan')
            ax_f.text(0.5, 0.02, f'θ={th:+.2f}  u={ep_act[fi]:+.0f}',
                      transform=ax_f.transAxes, fontsize=6.5, color='white',
                      ha='center', va='bottom',
                      bbox=dict(facecolor='black', alpha=0.55, pad=1.5, edgecolor='none'))

        # ── State trajectory ──────────────────────────────────────────────
        ax_t = fig.add_subplot(inner[1])
        ax_t.set_facecolor(PANEL)
        t = np.arange(T)
        if ep_states.shape[1] >= 4:
            ax_t.plot(t, ep_states[:, 0], color='#3498db', lw=1.4, label='x (cart pos)')
            ax_t.plot(t, ep_states[:, 2], color='#e74c3c', lw=1.4, label='θ (pole angle)')
        ax_t.step(t, ep_act * 0.05, color='#95a5a6', lw=0.8, alpha=0.6,
                  where='mid', label='action u (scaled ×0.05)')
        ax_t.axhline(0, color='white', lw=0.6, alpha=0.25)
        ax_t.set_xlabel('step', fontsize=8.5, color=TICK_C)
        ax_t.set_ylabel('value', fontsize=8.5, color=TICK_C)
        ax_t.tick_params(colors=TICK_C, labelsize=7.5)
        for sp in ax_t.spines.values():
            sp.set_color(GRID_C)
        ax_t.grid(True, alpha=0.25, color=GRID_C, linewidth=0.8)
        ax_t.legend(fontsize=7, loc='upper right', facecolor='#0d0d1e',
                    labelcolor='#cccccc', edgecolor=GRID_C, framealpha=0.85)

        ax_label = fig.add_subplot(inner[0])
        ax_label.axis('off')
        ax_label.text(0.0, 1.12, f'Episode {ep_id}  ·  {T} steps',
                      transform=ax_label.transAxes, fontsize=11,
                      fontweight='bold', color='#2ecc71', va='bottom', ha='left')

    fig.suptitle(f'swm/CartPoleControl-v1 — Collected Dataset Sample Episodes\n'
                 f'({args.data})',
                 fontsize=12, fontweight='bold', color='white', y=0.985)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'[viz] -> {out}')


if __name__ == '__main__':
    main()
