#!/usr/bin/env python
"""Subsample a Walker2d HDF5 dataset collected at frame_skip=1 to frame_skip=K.

For each episode with T actions at fs=1:
  - observations: take every K-th frame → (T//K + 1, H, W, C)
  - states:       take every K-th frame → same shape
  - actions:      take the action at each macro-step boundary → (T//K, 6)
                  (equivalent to "which action was issued at t=0,K,2K,...")
  - rewards:      sum K consecutive rewards per macro-step → (T//K,)

Usage
-----
  python experiments/subsample_walker2d_dataset.py \\
      --src  data/walker2d_sac_fs1_64 \\
      --dst  data/walker2d_sac_fs5_64 \\
      --stride 5
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np


def subsample_episode(ep: dict, stride: int) -> dict | None:
    """Return a subsampled episode dict, or None if too short."""
    obs  = ep['observations']   # (T+1, H, W, C)
    st   = ep['states']         # (T+1, 17)
    acts = ep['actions']        # (T, 6)
    rews = ep['rewards']        # (T,)

    T = len(acts)
    n_macro = T // stride
    if n_macro < 1:
        return None

    # Observations/states at macro-step boundaries (including final)
    obs_idx = np.arange(n_macro + 1) * stride
    obs_idx[-1] = min(obs_idx[-1], T)   # clip final index
    new_obs    = obs[obs_idx]            # (n_macro+1, H, W, C)
    new_states = st[obs_idx]

    # Action at the START of each macro-step
    act_idx  = np.arange(n_macro) * stride
    new_acts = acts[act_idx]             # (n_macro, 6)

    # Sum rewards over each macro-step window
    new_rews = np.array(
        [rews[i * stride: i * stride + stride].sum() for i in range(n_macro)],
        dtype=np.float32,
    )

    return {
        'observations': new_obs,
        'states':       new_states,
        'actions':      new_acts,
        'rewards':      new_rews,
    }


def process_split(src_path: Path, dst_path: Path, stride: int) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
        eps_grp = dst.create_group('episodes')
        n_written = 0
        n_transitions = 0

        src_eps = src['episodes']
        for key in sorted(src_eps.keys(), key=int):
            g = src_eps[key]
            ep = {k: g[k][:] for k in ('observations', 'states', 'actions', 'rewards')}
            sub = subsample_episode(ep, stride)
            if sub is None:
                continue
            out = eps_grp.create_group(str(n_written))
            for k, v in sub.items():
                out.create_dataset(k, data=v, compression='gzip', compression_opts=4)
            out.attrs['length'] = len(sub['actions'])
            n_written     += 1
            n_transitions += len(sub['actions'])

        dst.attrs['n_episodes']    = n_written
        dst.attrs['n_transitions'] = n_transitions
        print(f'  {dst_path.name}: {n_written} episodes, {n_transitions} transitions')


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Subsample a fs=1 Walker2d HDF5 dataset to fs=stride.')
    p.add_argument('--src',    required=True,
                   help='Source dataset directory (contains train/val/test.hdf5)')
    p.add_argument('--dst',    required=True,
                   help='Destination dataset directory')
    p.add_argument('--stride', type=int, default=5,
                   help='Subsampling stride (target frame_skip)')
    p.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    print(f'Subsampling {src} → {dst}  (stride={args.stride})')
    for split in args.splits:
        sp = src / f'{split}.hdf5'
        if not sp.exists():
            print(f'  [skip] {sp} not found')
            continue
        print(f'  {split} ...', end=' ', flush=True)
        process_split(sp, dst / f'{split}.hdf5', args.stride)

    # Copy metadata if present
    for fname in ('metadata.json', 'action_stats.json'):
        s = src / fname
        if s.exists():
            shutil.copy(s, dst / fname)
            print(f'  copied {fname}')

    # Write a provenance note
    meta = {
        'source':      str(src),
        'stride':      args.stride,
        'frame_skip':  args.stride,
        'note':        f'Subsampled from fs=1 dataset at stride={args.stride}',
    }
    (dst / 'subsample_meta.json').write_text(json.dumps(meta, indent=2))
    print('Done.')


if __name__ == '__main__':
    main()

