#!/usr/bin/env python
"""Validation script for the discrete CartPole-v1 visual dataset.

Checks performed
----------------
1.  len(observations) == len(actions) + 1 for every episode.
2.  Transition alignment: re-simulate a sample of stored (state, action) pairs
    and verify that the resulting next_state matches the stored one.
3.  Episode-length distributions by noise level (histogram per policy type).
4.  Distributions of all four state variables (x, ẋ, θ, θ̇).
5.  Left / right action balance.
6.  Trajectory GIFs for sample episodes (clean expert, noisy, burst, random,
    failed, and partially recovered episodes).
7.  Duplicate / constant-frame detection.
8.  Train / val / test episode-seed overlap check.

Usage
-----
    python experiments/validate_cartpole_dataset.py \\
        --dataset-dir data/cartpole_visual \\
        --out-dir results/dataset_validation
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.cartpole_discrete_dataset import load_split, SRC_NAMES

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

POLICY_COLORS = {
    'expert_eps0.00':  '#2ecc71',
    'noisy_eps0.05':   '#3498db',
    'noisy_eps0.10':   '#9b59b6',
    'noisy_eps0.20':   '#e67e22',
    'burst_eps0.00':   '#e74c3c',
    'random_eps0.00':  '#95a5a6',
}


# ── Check 1 & 2: alignment ────────────────────────────────────────────────────

def check_shape_alignment(episodes: List[Dict], split: str) -> bool:
    """Verify len(observations) == len(actions) + 1 for every episode."""
    failures = 0
    for i, ep in enumerate(episodes):
        obs_len = ep['observations'].shape[0]
        act_len = ep['actions'].shape[0]
        if obs_len != act_len + 1:
            logger.error('[%s] ep %d: obs_len=%d  act_len=%d  (expected obs=act+1)',
                         split, i, obs_len, act_len)
            failures += 1
    if failures == 0:
        logger.info('[%s] Shape alignment OK (%d episodes)', split, len(episodes))
    else:
        logger.error('[%s] %d episodes failed shape alignment', split, failures)
    return failures == 0


def check_transition_alignment(episodes: List[Dict], split: str,
                               n_episodes: int = 5,
                               n_steps_per_ep: int = 10) -> bool:
    """Re-simulate stored (state_t, action_t) and compare to stored state_{t+1}.

    Uses gymnasium CartPole-v1 directly. Checks absolute error < 1e-4.
    """
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    failures = 0
    idxs = np.random.choice(len(episodes), size=min(n_episodes, len(episodes)),
                            replace=False)
    for ep_idx in idxs:
        ep       = episodes[ep_idx]
        T        = ep['length'] if 'length' in ep else len(ep['actions'])
        step_sel = np.random.choice(T, size=min(n_steps_per_ep, T), replace=False)

        for t in sorted(step_sel):
            s_t     = ep['states'][t].astype(np.float64)
            action  = int(ep['actions'][t])
            s_next  = ep['states'][t + 1].astype(np.float64)

            # Force env into state_t
            env.reset(seed=0)
            env.unwrapped.state = s_t.copy()
            _, _, _, _, _ = env.step(action)
            s_sim = env.unwrapped.state.copy()

            err = np.abs(s_sim - s_next).max()
            if err > 1e-4:
                logger.error('[%s] ep=%d t=%d  max_err=%.6f  '
                             'sim=%s  stored=%s',
                             split, ep_idx, t, err,
                             np.round(s_sim, 5), np.round(s_next, 5))
                failures += 1

    env.close()
    if failures == 0:
        logger.info('[%s] Transition alignment OK '
                    '(%d eps × %d steps sampled)', split, n_episodes, n_steps_per_ep)
    else:
        logger.error('[%s] %d transition-alignment failures', split, failures)
    return failures == 0


# ── Check 3: episode-length distributions ────────────────────────────────────

def plot_episode_lengths(episodes: List[Dict], out_dir: Path) -> None:
    """Histogram of episode lengths, one series per policy type."""
    by_policy: Dict[str, List[int]] = {}
    for ep in episodes:
        key = f"{ep['policy_type']}_eps{ep['epsilon']:.2f}"
        by_policy.setdefault(key, []).append(
            int(ep['length']) if 'length' in ep else len(ep['actions']))

    fig, ax = plt.subplots(figsize=(10, 4))
    bins = np.arange(0, 520, 20)
    for key, lengths in sorted(by_policy.items()):
        color = POLICY_COLORS.get(key, '#888888')
        ax.hist(lengths, bins=bins, alpha=0.5, label=key,
                color=color, edgecolor='none')

    ax.axvline(500, color='red', ls='--', lw=1.2, label='500-step limit')
    ax.set_xlabel('Episode length (steps)')
    ax.set_ylabel('Count')
    ax.set_title('Episode-length distribution by policy type')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / 'ep_length_distribution.png', dpi=130)
    plt.close(fig)
    logger.info('Saved ep_length_distribution.png')


# ── Check 4: state-variable distributions ────────────────────────────────────

STATE_LABELS = ['x  (cart pos)', 'ẋ  (cart vel)', 'θ  (pole angle)', 'θ̇  (pole vel)']

def plot_state_distributions(episodes: List[Dict], out_dir: Path) -> None:
    """Histogram of all four state dimensions."""
    states = np.concatenate([ep['states'] for ep in episodes], axis=0)  # (N+ep, 4)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3))
    for i, ax in enumerate(axes):
        ax.hist(states[:, i], bins=80, color='#3498db', edgecolor='none', alpha=0.8)
        ax.set_xlabel(STATE_LABELS[i])
        ax.set_ylabel('Count' if i == 0 else '')
        ax.set_title(f'dim {i}')
    fig.suptitle('State variable distributions (all transitions)', fontsize=11)
    plt.tight_layout()
    fig.savefig(out_dir / 'state_distributions.png', dpi=130)
    plt.close(fig)
    logger.info('Saved state_distributions.png')


# ── Check 5: action balance ───────────────────────────────────────────────────

def report_action_balance(episodes: List[Dict], split: str) -> None:
    actions = np.concatenate([ep['actions'] for ep in episodes])
    n_left  = int((actions == 0).sum())
    n_right = int((actions == 1).sum())
    total   = n_left + n_right
    logger.info('[%s] Action balance: left=%d (%.1f%%)  right=%d (%.1f%%)',
                split, n_left, 100 * n_left / total,
                n_right, 100 * n_right / total)

    if 'action_source' in episodes[0]:
        sources = np.concatenate([ep['action_source'] for ep in episodes])
        for k, name in SRC_NAMES.items():
            n = int((sources == k).sum())
            logger.info('  src=%s: %d (%.1f%%)', name, n, 100 * n / total)


# ── Check 6: GIF export ───────────────────────────────────────────────────────

def _save_gif(frames: np.ndarray, path: Path, duration_ms: int = 50) -> None:
    """Save an (T, H, W, 3) array as an animated GIF."""
    try:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(
            path, save_all=True, append_images=imgs[1:],
            loop=0, duration=duration_ms,
        )
    except ImportError:
        # Fallback: save first frame as PNG
        path = path.with_suffix('.png')
        try:
            import cv2
            cv2.imwrite(str(path), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
        except ImportError:
            import matplotlib.pyplot as plt
            plt.imsave(str(path), frames[0])


def save_episode_gifs(episodes: List[Dict], out_dir: Path) -> None:
    """Save one GIF per category: expert, noisy, burst, random, failed."""
    categories = {
        'expert':   lambda ep: ep['policy_type'] == 'expert',
        'noisy':    lambda ep: ep['policy_type'] == 'noisy' and ep['epsilon'] == 0.10,
        'burst':    lambda ep: ep['policy_type'] == 'burst',
        'random':   lambda ep: ep['policy_type'] == 'random',
        'failed':   lambda ep: bool(ep.get('terminated', [False])[-1]),
        'success':  lambda ep: ep.get('success', False),
    }

    gif_dir = out_dir / 'gifs'
    gif_dir.mkdir(exist_ok=True)
    saved = 0

    for cat, predicate in categories.items():
        candidates = [ep for ep in episodes if predicate(ep)]
        if not candidates:
            logger.warning('No episodes found for category "%s"', cat)
            continue
        # Pick the longest candidate (more interesting)
        ep = max(candidates, key=lambda e: len(e['actions']))
        frames = ep['observations']             # (T+1, H, W, 3)
        path   = gif_dir / f'{cat}.gif'
        _save_gif(frames, path)
        logger.info('Saved %s  (%d frames, len=%d)',
                    path.name, len(frames), len(ep['actions']))
        saved += 1

    logger.info('GIFs saved to %s  (%d total)', gif_dir, saved)


# ── Check 7: duplicate / constant frames ─────────────────────────────────────

def check_duplicate_frames(episodes: List[Dict], split: str,
                           n_sample: int = 200) -> None:
    """Spot-check for constant or duplicate frames within episodes."""
    constant_count   = 0
    near_dup_count   = 0
    ep_sample = np.random.choice(len(episodes),
                                 size=min(n_sample, len(episodes)),
                                 replace=False)
    for ep_idx in ep_sample:
        obs = episodes[ep_idx]['observations'].astype(np.float32)  # (T+1, H, W, 3)
        if len(obs) < 2:
            continue
        # Constant frame: all frames identical
        if obs.std(axis=0).mean() < 0.5:
            constant_count += 1
        # Near-duplicate consecutive frames in the middle
        mid   = len(obs) // 2
        diff  = np.abs(obs[mid + 1] - obs[mid]).mean()
        if diff < 1.0:
            near_dup_count += 1

    logger.info('[%s] Frame check on %d episodes: '
                'constant=%d  near-dup-consecutive=%d',
                split, len(ep_sample), constant_count, near_dup_count)
    if constant_count > 0:
        logger.warning('Constant-frame episodes detected — check rendering.')


# ── Check 8: seed overlap ─────────────────────────────────────────────────────

def check_seed_overlap(all_splits: Dict[str, List[Dict]]) -> None:
    """Verify that no episode seed appears in more than one split."""
    split_seeds = {
        name: set(int(ep['ep_seed']) for ep in eps)
        for name, eps in all_splits.items()
    }
    names = list(split_seeds.keys())
    any_overlap = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b   = names[i], names[j]
            shared = split_seeds[a] & split_seeds[b]
            if shared:
                logger.warning('Seed overlap between %s and %s: %d seeds',
                               a, b, len(shared))
                any_overlap = True
    if not any_overlap:
        logger.info('Seed overlap check: no overlaps across splits ✓')


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(meta: Dict, all_splits: Dict[str, List[Dict]]) -> None:
    logger.info('─' * 60)
    logger.info('Dataset summary')
    logger.info('  Total episodes:    %d', meta.get('total_episodes', '?'))
    logger.info('  Total transitions: %d', meta.get('total_transitions', '?'))
    logger.info('  Avg ep length:     %.1f  (min=%d  max=%d)',
                meta.get('ep_len_mean', 0),
                meta.get('ep_len_min', 0),
                meta.get('ep_len_max', 0))
    logger.info('  Success rate:      %.1f%%',
                meta.get('frac_success', 0) * 100)
    logger.info('  Terminated:        %.1f%%',
                meta.get('frac_terminated', 0) * 100)
    for split_name, eps in all_splits.items():
        n_trans = sum(len(ep['actions']) for ep in eps)
        logger.info('  %-6s: %d ep / %d trans', split_name, len(eps), n_trans)
    logger.info('─' * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description='Validate a CartPole-v1 visual dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--dataset-dir', default='data/cartpole_visual',
                   help='Directory containing train/val/test .hdf5 files')
    p.add_argument('--out-dir',     default='results/dataset_validation',
                   help='Where to save plots and GIFs')
    p.add_argument('--splits',      nargs='+', default=['train', 'val', 'test'])
    p.add_argument('--n-align-eps', type=int, default=5,
                   help='Episodes to re-simulate for transition alignment check')
    p.add_argument('--n-align-steps', type=int, default=10,
                   help='Steps per episode in transition alignment check')
    p.add_argument('--seed',        type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        logger.error('Dataset directory not found: %s', dataset_dir)
        sys.exit(1)

    # Load metadata
    meta_path = dataset_dir / 'metadata.json'
    meta: Dict = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        logger.warning('metadata.json not found — skipping summary table.')

    # Load splits
    all_splits: Dict[str, List[Dict]] = {}
    for split in args.splits:
        hdf5_path = dataset_dir / f'{split}.hdf5'
        if not hdf5_path.exists():
            logger.warning('Split file not found: %s', hdf5_path)
            continue
        eps, _ = load_split(str(dataset_dir), split)
        all_splits[split] = eps
        logger.info('Loaded %s: %d episodes', split, len(eps))

    if not all_splits:
        logger.error('No split files found in %s', dataset_dir)
        sys.exit(1)

    # Combine all for global plots
    all_episodes = [ep for eps in all_splits.values() for ep in eps]

    # Print summary
    print_summary(meta, all_splits)

    # ── Per-split checks ──────────────────────────────────────────────────────
    all_ok = True
    for split, episodes in all_splits.items():
        logger.info('\n── Split: %s  (%d episodes) ──', split, len(episodes))

        # Check 1: shape alignment
        ok = check_shape_alignment(episodes, split)
        all_ok = all_ok and ok

        # Check 2: transition alignment (re-simulate)
        ok = check_transition_alignment(
            episodes, split,
            n_episodes=args.n_align_eps,
            n_steps_per_ep=args.n_align_steps,
        )
        all_ok = all_ok and ok

        # Check 5: action balance
        report_action_balance(episodes, split)

        # Check 7: duplicate frames
        check_duplicate_frames(episodes, split)

    # Check 8: seed overlap
    check_seed_overlap(all_splits)

    # ── Global plots ──────────────────────────────────────────────────────────
    logger.info('\n── Generating plots ──')

    # Check 3: episode lengths (use train split for plots)
    primary_split = 'train' if 'train' in all_splits else list(all_splits.keys())[0]
    plot_episode_lengths(all_splits[primary_split], out_dir)

    # Check 4: state distributions
    plot_state_distributions(all_episodes, out_dir)

    # Check 6: GIFs
    save_episode_gifs(all_splits[primary_split], out_dir)

    # ── Final verdict ─────────────────────────────────────────────────────────
    if all_ok:
        logger.info('\n✓ All structural checks passed.')
    else:
        logger.error('\n✗ Some checks FAILED — see errors above.')

    logger.info('Results saved to %s', out_dir)


if __name__ == '__main__':
    main()
