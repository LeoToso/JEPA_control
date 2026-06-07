#!/usr/bin/env python
"""Check an h5 dataset for rendering-aliasing artifacts.

Scans for near-equilibrium samples whose rendered observation is bit-identical
to another sample's observation despite the underlying physical states being
different. This pattern is the signature of the old nearest-neighbour
downsampling fallback in `ContinuousCartpoleVisual._render_obs`, which aliased
away small pole-angle differences (< ~0.03 rad) into identical pixels.

A correctly rendered (PIL BOX / area-average) dataset should report ~0
collisions between distinct states.

Usage:
    python experiments/check_render_aliasing.py --h5 data/cartpole_ae_noLQR_ep_seed43.h5
    python experiments/check_render_aliasing.py --h5 data/cartpole_ae_noLQR_ep_seed43.h5 --theta-thresh 0.05
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import h5py


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h5', required=True)
    p.add_argument('--theta-thresh', type=float, default=0.10,
                   help='|theta| below this is considered "near-equilibrium"')
    p.add_argument('--state-atol', type=float, default=1e-4,
                   help='states closer than this are considered "the same" physical state')
    p.add_argument('--theta-collision-thresh', type=float, default=0.02,
                   help='|Δtheta| above this for a bit-identical pair is considered '
                        'too large to be a sub-pixel resolution artifact (real aliasing)')
    p.add_argument('--dump-outliers', action='store_true',
                   help='Print full state/episode info for pairs with |Δtheta| above '
                        '--theta-collision-thresh, to inspect whether they are a '
                        'genuine renderer bug or e.g. episode-boundary padding')
    args = p.parse_args()

    with h5py.File(args.h5, 'r') as f:
        obs    = f['obs'][:]            # (N, H, W, 3) uint8
        states = f['states'][:]         # (N, 4)
        episode_ids = f['episode_ids'][:] if 'episode_ids' in f else None

    theta = states[:, 2]
    near_eq = np.abs(theta) < args.theta_thresh
    idxs = np.nonzero(near_eq)[0]
    print(f'[check] {args.h5}')
    print(f'[check] total samples={len(states)}  near-eq (|theta|<{args.theta_thresh}): {len(idxs)}')

    seen = {}            # obs bytes -> first index with that obs
    collisions = 0
    distinct_state_collisions = 0
    theta_diffs = []     # |Δtheta| for colliding pairs whose states differ
    outlier_pairs = []   # (i, j, |Δtheta|) for pairs above --theta-collision-thresh
    for i in idxs:
        key = obs[i].tobytes()
        if key in seen:
            j = seen[key]
            collisions += 1
            if not np.allclose(states[i], states[j], atol=args.state_atol):
                distinct_state_collisions += 1
                dtheta = abs(float(states[i][2] - states[j][2]))
                theta_diffs.append(dtheta)
                if dtheta > args.theta_collision_thresh:
                    outlier_pairs.append((int(j), int(i), dtheta))
        else:
            seen[key] = i

    print(f'[check] bit-identical obs collisions among near-eq samples: {collisions}')
    print(f'[check] of those, collisions between DISTINCT physical states: {distinct_state_collisions}')

    if theta_diffs:
        td = np.array(theta_diffs)
        pct = np.percentile(td, [50, 90, 95, 99, 100])
        print(f'[check] |Δtheta| among colliding "distinct" pairs '
              f'(median/p90/p95/p99/max): '
              f'{pct[0]:.5f} / {pct[1]:.5f} / {pct[2]:.5f} / {pct[3]:.5f} / {pct[4]:.5f} rad')
        large = int(np.sum(td > args.theta_collision_thresh))
        print(f'[check] of those, pairs with |Δtheta| > {args.theta_collision_thresh} rad '
              f'(too large to be a sub-pixel resolution floor): {large}')
    else:
        large = 0

    if args.dump_outliers and outlier_pairs:
        print(f'\n[outliers] {len(outlier_pairs)} pair(s) with |Δtheta| > '
              f'{args.theta_collision_thresh} rad:')
        for j, i, dtheta in outlier_pairs:
            ep_j = episode_ids[j] if episode_ids is not None else '?'
            ep_i = episode_ids[i] if episode_ids is not None else '?'
            print(f'  idx {j} (ep {ep_j}) state={states[j]}  <-->  '
                  f'idx {i} (ep {ep_i}) state={states[i]}   |Δtheta|={dtheta:.5f}')

    if large > 0:
        frac = large / max(len(idxs), 1)
        print(f'\n[RESULT] ALIASING DETECTED — {large} '
              f'({frac:.1%} of near-eq samples) frames are bit-identical despite '
              f'differing physical states by MORE than {args.theta_collision_thresh} rad '
              f'— too large to be explained by image resolution. This dataset is '
              f'consistent with the OLD (nearest-neighbour) renderer.')
        sys.exit(1)
    else:
        print('\n[RESULT] No significant aliasing detected — any bit-identical '
              'collisions between "distinct" states differ only by sub-pixel amounts '
              f'(|Δtheta| <= {args.theta_collision_thresh} rad), consistent with the '
              'inherent resolution floor of a 64x64 render rather than the old '
              'nearest-neighbour renderer bug. This dataset is consistent with the '
              'FIXED (area-average / PIL BOX) renderer.')
        sys.exit(0)


if __name__ == '__main__':
    main()
