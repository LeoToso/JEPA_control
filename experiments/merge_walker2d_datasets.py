#!/usr/bin/env python
"""Merge two Walker2d HDF5 datasets into one.

Takes --frac-a fraction of episodes from dataset A and --frac-b fraction
from dataset B (both expressed as fractions of their own total), shuffles,
and writes merged train/val/test splits to --output-dir.

Usage
-----
  # 50% random + 50% SAC  (same total episode count)
  python experiments/merge_walker2d_datasets.py \\
      --src-a  data/walker2d_fs5_64 \\
      --src-b  data/walker2d_sac_64 \\
      --output-dir data/walker2d_mixed_64 \\
      --frac-a 0.5 --frac-b 0.5

  # Keep all of A, add all of B (union)
  python experiments/merge_walker2d_datasets.py \\
      --src-a  data/walker2d_fs5_64 \\
      --src-b  data/walker2d_sac_64 \\
      --output-dir data/walker2d_union_64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


# ── per-split helpers ─────────────────────────────────────────────────────────

def read_split(hdf5_path: Path) -> list[dict]:
    """Load all episodes from one split file."""
    episodes = []
    with h5py.File(hdf5_path, 'r') as f:
        ep_grp = f['episodes']
        for k in sorted(ep_grp.keys(), key=int):
            ep = {}
            for key in ep_grp[k].keys():
                ep[key] = ep_grp[k][key][:]
            episodes.append(ep)
    return episodes


def write_split(episodes: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as f:
        grp = f.create_group('episodes')
        for i, ep in enumerate(episodes):
            g = grp.create_group(str(i))
            for key, val in ep.items():
                g.create_dataset(key, data=val,
                                 compression='gzip', compression_opts=4)
            g.attrs['length'] = len(ep['actions'])
        f.attrs['n_episodes']    = len(episodes)
        f.attrs['n_transitions'] = sum(len(ep['actions']) for ep in episodes)
    n_tr = sum(len(ep['actions']) for ep in episodes)
    print(f'  {path.name}: {len(episodes)} episodes, {n_tr} transitions')


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Merge two Walker2d HDF5 datasets.')
    p.add_argument('--src-a',      required=True,
                   help='Directory with existing train/val/test.hdf5 (dataset A).')
    p.add_argument('--src-b',      required=True,
                   help='Directory with new train/val/test.hdf5 (dataset B).')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--frac-a',     type=float, default=0.5,
                   help='Fraction of A episodes to keep (0–1). '
                        '1.0 = use all episodes in A.')
    p.add_argument('--frac-b',     type=float, default=0.5,
                   help='Fraction of B episodes to keep (0–1). '
                        '1.0 = use all episodes in B.')
    p.add_argument('--seed',       type=int,   default=0)
    return p.parse_args()


def main():
    args   = parse_args()
    rng    = np.random.default_rng(args.seed)
    src_a  = Path(args.src_a)
    src_b  = Path(args.src_b)
    out    = Path(args.output_dir)

    print(f'Merging datasets:')
    print(f'  A ({args.frac_a*100:.0f}%): {src_a}')
    print(f'  B ({args.frac_b*100:.0f}%): {src_b}')
    print(f'  → {out}')

    for split in ('train', 'val', 'test'):
        path_a = src_a / f'{split}.hdf5'
        path_b = src_b / f'{split}.hdf5'

        eps_a, eps_b = [], []
        if path_a.exists():
            eps_a = read_split(path_a)
        if path_b.exists():
            eps_b = read_split(path_b)

        if not eps_a and not eps_b:
            continue

        # Sample fractions
        n_a = max(1, round(len(eps_a) * args.frac_a)) if eps_a else 0
        n_b = max(1, round(len(eps_b) * args.frac_b)) if eps_b else 0

        idx_a = rng.choice(len(eps_a), size=n_a, replace=False) if eps_a else []
        idx_b = rng.choice(len(eps_b), size=n_b, replace=False) if eps_b else []

        merged = ([eps_a[i] for i in idx_a] +
                  [eps_b[i] for i in idx_b])

        # Shuffle the combined list
        order  = rng.permutation(len(merged))
        merged = [merged[i] for i in order]

        print(f'\n[{split}]  A: {len(eps_a)} → {n_a}  '
              f'B: {len(eps_b)} → {n_b}  '
              f'merged: {len(merged)}')
        write_split(merged, out / f'{split}.hdf5')

    # Write a metadata file summarising the merge
    meta = {
        'source_a':    str(src_a),
        'source_b':    str(src_b),
        'frac_a':      args.frac_a,
        'frac_b':      args.frac_b,
        'seed':        args.seed,
    }
    # Copy image_size / frame_skip from source A's metadata if it exists
    meta_a = src_a / 'metadata.json'
    if meta_a.exists():
        with open(meta_a) as f:
            meta['source_a_meta'] = json.load(f)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\n[done] → {out}')


if __name__ == '__main__':
    main()
