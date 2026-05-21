"""Dataset visualisation: observations, state/action distributions, episode trajectories."""
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


def load(h5_path: str):
    with h5py.File(h5_path, 'r') as f:
        obs      = f['obs'][:]
        next_obs = f['next_obs'][:]
        states   = f['states'][:]
        actions  = f['actions'][:, 0]
        ep_ids   = f['episode_ids'][:]
    return obs, next_obs, states, actions, ep_ids


def fig_sample_frames(obs, states, n=32, seed=0, out=None):
    """Grid of random frames annotated with (θ, x)."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(obs), n, replace=False)
    cols = 8; rows = n // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.9))
    for ax, i in zip(axes.flat, idx):
        ax.imshow(obs[i])
        th = states[i, 2]; x = states[i, 0]
        ax.set_title(f'θ={th:+.2f}\nx={x:+.2f}', fontsize=6)
        ax.axis('off')
    fig.suptitle('Random dataset frames  (θ=pole angle, x=cart pos)', fontsize=9)
    plt.tight_layout()
    _save(fig, out or 'dataset_frames.png')


def fig_distributions(states, actions, out=None):
    """Histograms of all state components and actions."""
    labels = ['x (cart pos, m)', 'ẋ (cart vel, m/s)',
              'θ (pole angle, rad)', 'θ̇ (pole vel, rad/s)', 'u (action, N)']
    data   = [states[:, i] for i in range(4)] + [actions]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3))
    for ax, d, lbl in zip(axes, data, labels):
        ax.hist(d, bins=60, color='steelblue', edgecolor='none', alpha=0.8)
        ax.axvline(0, color='red', lw=1, ls='--')
        ax.set_title(lbl, fontsize=8)
        ax.set_xlabel(f'μ={d.mean():.3f}  σ={d.std():.3f}', fontsize=7)
        ax.tick_params(labelsize=7)
    fig.suptitle('Dataset distributions', fontsize=10)
    plt.tight_layout()
    _save(fig, out or 'dataset_distributions.png')


def fig_episode_trajectories(states, actions, ep_ids, n_eps=8, seed=0, out=None):
    """State + action trajectories for a few random episodes."""
    rng   = np.random.RandomState(seed)
    uids  = np.unique(ep_ids)
    picks = rng.choice(uids, min(n_eps, len(uids)), replace=False)

    fig = plt.figure(figsize=(14, n_eps * 1.4 + 1))
    gs  = gridspec.GridSpec(n_eps, 1, hspace=0.55)

    for row, ep in enumerate(picks):
        mask = ep_ids == ep
        s = states[mask]; a = actions[mask]
        T = np.arange(len(s))
        ax = fig.add_subplot(gs[row])
        ax.plot(T, s[:, 2], label='θ', color='tab:red',   lw=1.2)
        ax.plot(T, s[:, 0], label='x', color='tab:blue',  lw=1.2, ls='--')
        ax.fill_between(T, a * 0.02, alpha=0.25, color='orange', label='u×0.02')
        ax.axhline(0, color='k', lw=0.5, ls=':')
        ax.set_xlim(0, len(T) - 1)
        ax.set_ylabel(f'ep {ep}', fontsize=7, rotation=0, labelpad=28)
        ax.tick_params(labelsize=6)
        if row == 0:
            ax.legend(fontsize=6, loc='upper right', ncol=3)

    fig.suptitle('Episode trajectories  (θ red, x blue, action orange shaded)', fontsize=9)
    _save(fig, out or 'dataset_episodes.png')


def fig_transition_pairs(obs, next_obs, states, actions, n=8, seed=0, out=None):
    """Show (obs_t, obs_{t+1}) pairs for a few transitions."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(obs), n, replace=False)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.6, 3.5))
    for col, i in enumerate(idx):
        u = actions[i]; th = states[i, 2]
        axes[0, col].imshow(obs[i]);      axes[0, col].axis('off')
        axes[1, col].imshow(next_obs[i]); axes[1, col].axis('off')
        axes[0, col].set_title(f'θ={th:+.2f}\nu={u:+.1f}', fontsize=6)
    axes[0, 0].set_ylabel('obs_t',      fontsize=8)
    axes[1, 0].set_ylabel('obs_{t+1}',  fontsize=8)
    fig.suptitle('Transition pairs  (top: obs_t,  bottom: obs_{t+1})', fontsize=9)
    plt.tight_layout()
    _save(fig, out or 'dataset_transitions.png')


def _save(fig, path):
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('h5', nargs='?',
                   default='data/cartpole_v2_mixed_fs1_seed42.h5')
    p.add_argument('--out_dir', default='.')
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f'Loading {args.h5} ...')
    obs, next_obs, states, actions, ep_ids = load(args.h5)
    print(f'  {len(obs):,} transitions  |  {len(np.unique(ep_ids)):,} episodes')

    print('Generating figures ...')
    fig_sample_frames(obs, states,
                      out=str(out / 'dataset_frames.png'))
    fig_distributions(states, actions,
                      out=str(out / 'dataset_distributions.png'))
    fig_episode_trajectories(states, actions, ep_ids,
                              out=str(out / 'dataset_episodes.png'))
    fig_transition_pairs(obs, next_obs, states, actions,
                         out=str(out / 'dataset_transitions.png'))
    print('Done.')
