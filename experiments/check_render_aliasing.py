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
    args = p.parse_args()

    with h5py.File(args.h5, 'r') as f:
        obs    = f['obs'][:]            # (N, H, W, 3) uint8
        states = f['states'][:]         # (N, 4)

    theta = states[:, 2]
    near_eq = np.abs(theta) < args.theta_thresh
    idxs = np.nonzero(near_eq)[0]
    print(f'[check] {args.h5}')
    print(f'[check] total samples={len(states)}  near-eq (|theta|<{args.theta_thresh}): {len(idxs)}')

    seen = {}            # obs bytes -> first index with that obs
    collisions = 0
    distinct_state_collisions = 0
    for i in idxs:
        key = obs[i].tobytes()
        if key in seen:
            j = seen[key]
            collisions += 1
            if not np.allclose(states[i], states[j], atol=args.state_atol):
                distinct_state_collisions += 1
        else:
            seen[key] = i

    print(f'[check] bit-identical obs collisions among near-eq samples: {collisions}')
    print(f'[check] of those, collisions between DISTINCT physical states: {distinct_state_collisions}')

    if distinct_state_collisions > 0:
        frac = distinct_state_collisions / max(len(idxs), 1)
        print(f'\n[RESULT] ALIASING DETECTED — {distinct_state_collisions} '
              f'({frac:.1%} of near-eq samples) frames are bit-identical despite '
              f'differing physical states. This dataset was generated with the '
              f'OLD (nearest-neighbour) renderer.')
        sys.exit(1)
    else:
        print('\n[RESULT] No aliasing detected — distinct near-eq states render to '
              'distinct pixels. This dataset is consistent with the FIXED '
              '(area-average / PIL BOX) renderer.')
        sys.exit(0)


if __name__ == '__main__':
    main()
