"""Visualize a discrete CartPole HDF5 dataset.

Usage:
    python experiments/visualize_dataset.py \
        --data data/cartpole_visual_fs5_v4 \
        --out  results/dataset_viz.png
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_episodes(hdf5_path, max_eps=None):
    eps = []
    with h5py.File(hdf5_path, 'r') as f:
        keys = sorted(f['episodes'].keys(), key=int)
        if max_eps is not None:
            keys = keys[:max_eps]
        for k in keys:
            ep = f['episodes'][k]
            eps.append({
                'obs':    ep['observations'][:],   # (T, H, W, 3) uint8
                'states': ep['states'][:],          # (T, 4) float32
                'actions':ep['actions'][:],         # (T,) float32
                'policy': ep.attrs.get('policy_type', 'unknown'),
                'length': int(ep.attrs.get('length', len(ep['actions']))),
            })
    return eps


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/cartpole_visual_fs5_v4')
    p.add_argument('--out',  default='results/dataset_viz.png')
    p.add_argument('--n-eps', type=int, default=300,
                   help='Max episodes to load for statistics')
    args = p.parse_args()

    train_path = Path(args.data) / 'train.hdf5'
    print(f'Loading {train_path} …')
    eps = load_episodes(train_path, max_eps=args.n_eps)

    passive = [e for e in eps if 'passive' in e['policy']]
    random  = [e for e in eps if 'random'  in e['policy']]
    print(f'  passive={len(passive)}  random={len(random)}  total={len(eps)}')

    ep_lens = [e['length'] for e in eps]
    print(f'  episode lengths: min={min(ep_lens)}  mean={np.mean(ep_lens):.0f}'
          f'  max={max(ep_lens)}')

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 0: sample frames from one passive and one random episode ──────────
    ax_title_p = fig.add_subplot(gs[0, :2])
    ax_title_p.axis('off')
    ax_title_p.set_title('Passive episode — sample frames (every 5th step)',
                          fontsize=11, loc='left', pad=2)
    ax_title_r = fig.add_subplot(gs[0, 2:])
    ax_title_r.axis('off')
    ax_title_r.set_title('Random episode — sample frames (every 5th step)',
                          fontsize=11, loc='left', pad=2)

    # Manually place small image axes inside the two title cells
    def show_frames(ep, fig, left, bottom, n_frames=8):
        obs = ep['obs']
        idxs = np.linspace(0, len(obs) - 1, n_frames, dtype=int)
        w = 0.235 / n_frames
        for j, i in enumerate(idxs):
            ax = fig.add_axes([left + j * w, bottom, w * 0.92, 0.09])
            ax.imshow(obs[i])
            ax.set_title(f't={i}', fontsize=6, pad=1)
            ax.axis('off')

    if passive:
        show_frames(passive[0], fig, left=0.05,  bottom=0.75)
    if random:
        show_frames(random[0],  fig, left=0.55, bottom=0.75)

    # ── Row 1: theta trajectories ─────────────────────────────────────────────
    ax_th_p = fig.add_subplot(gs[1, :2])
    ax_th_r = fig.add_subplot(gs[1, 2:])

    n_show = min(10, len(passive))
    for e in passive[:n_show]:
        ax_th_p.plot(e['states'][:, 2], alpha=0.6, lw=1)
    ax_th_p.axhline(0, color='k', lw=0.7, ls='--')
    ax_th_p.set_title(f'θ over time — passive (first {n_show} eps)', fontsize=10)
    ax_th_p.set_xlabel('step'); ax_th_p.set_ylabel('θ (rad)')
    ax_th_p.grid(alpha=0.3)

    n_show_r = min(10, len(random))
    for e in random[:n_show_r]:
        ax_th_r.plot(e['states'][:, 2], alpha=0.4, lw=1)
    ax_th_r.axhline(0, color='k', lw=0.7, ls='--')
    ax_th_r.set_title(f'θ over time — random (first {n_show_r} eps)', fontsize=10)
    ax_th_r.set_xlabel('step'); ax_th_r.set_ylabel('θ (rad)')
    ax_th_r.grid(alpha=0.3)

    # ── Row 2: state & action distributions ───────────────────────────────────
    all_states  = np.concatenate([e['states']  for e in eps], axis=0)  # (N, 4)
    all_actions = np.concatenate([e['actions'] for e in eps], axis=0)  # (N,)
    labels = ['x (m)', 'ẋ (m/s)', 'θ (rad)', 'θ̇ (rad/s)']

    for col, (lbl, data) in enumerate(zip(labels, all_states.T)):
        ax = fig.add_subplot(gs[2, col])
        ax.hist(data, bins=60, color='steelblue', alpha=0.8, edgecolor='none')
        ax.axvline(0, color='k', lw=0.8, ls='--')
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel('value'); ax.set_ylabel('count')
        ax.grid(alpha=0.3)

    # Replace last panel with action distribution
    ax_act = fig.add_subplot(gs[2, 3])
    ax_act.clear()
    ax_act.hist(all_actions, bins=60, color='darkorange', alpha=0.8, edgecolor='none')
    ax_act.axvline(0, color='k', lw=0.8, ls='--')
    ax_act.set_title('action u (N)', fontsize=10)
    ax_act.set_xlabel('value'); ax_act.set_ylabel('count')
    ax_act.grid(alpha=0.3)

    fig.suptitle(
        f'Dataset: {Path(args.data).name}   '
        f'({len(eps)} eps / {sum(ep_lens)} transitions shown   '
        f'passive={len(passive)}  random={len(random)})',
        fontsize=12, y=0.99)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches='tight')
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()
