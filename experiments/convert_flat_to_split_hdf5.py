"""Convert flat single-file HDF5 (from generate_data.py) to per-split episode files.

The flat format stores all transitions contiguously:
  obs, next_obs, actions, states, next_states  — shape (N, ...)
  episode_ids                                   — shape (N,) episode membership
  splits/train, splits/val, splits/test         — episode indices per split

This script groups transitions back into episodes and writes:
  <out-dir>/train.hdf5
  <out-dir>/val.hdf5
  <out-dir>/test.hdf5

compatible with DiscreteHDF5TrajectoryDataset / make_discrete_dataloaders.

Usage:
    python experiments/convert_flat_to_split_hdf5.py \\
        --src  data/cartpole_visual_runs_BE/cartpole_v2_ep_fs1_seed42.h5 \\
        --out  data/cartpole_visual_runs_BE
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np


def convert(src_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(src_path, 'r') as f:
        obs         = f['obs'][:]           # (N, H, W, C) uint8
        next_obs    = f['next_obs'][:]      # (N, H, W, C) uint8
        actions     = f['actions'][:]       # (N,) or (N, action_dim)
        states      = f['states'][:]        # (N, 4)
        next_states = f['next_states'][:]   # (N, 4)
        episode_ids = f['episode_ids'][:]   # (N,)

    print(f'[convert] {src_path}')
    print(f'[convert] {len(obs):,} transitions  '
          f'episodes={len(set(episode_ids.tolist()))}  '
          f'obs_shape={obs.shape[1:]}')

    # Always re-split by episode ID (80/10/10).
    # Stored splits may contain transition indices rather than episode IDs,
    # which would incorrectly assign every episode to train.
    all_ep_ids = sorted(set(episode_ids.tolist()))
    rng = np.random.RandomState(42)
    shuffled = np.array(all_ep_ids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_tr  = int(0.8 * n)
    n_val = int(0.1 * n)
    splits = {
        'train': set(shuffled[:n_tr].tolist()),
        'val':   set(shuffled[n_tr:n_tr + n_val].tolist()),
        'test':  set(shuffled[n_tr + n_val:].tolist()),
    }
    print(f'[convert] episode split: train={len(splits["train"])}  '
          f'val={len(splits["val"])}  test={len(splits["test"])} episodes')

    for split_name, ep_id_set in splits.items():
        mask = np.array([eid in ep_id_set for eid in episode_ids])
        if not mask.any():
            print(f'[convert] {split_name}: 0 transitions — skipping')
            continue

        # Group transitions into episodes preserving order
        ep_dict: dict[int, dict] = {}
        for i in np.where(mask)[0]:
            eid = int(episode_ids[i])
            if eid not in ep_dict:
                ep_dict[eid] = {
                    'obs':     [obs[i]],
                    'actions': [],
                    'states':  [states[i]],
                }
            ep_dict[eid]['obs'].append(next_obs[i])
            ep_dict[eid]['actions'].append(actions[i])
            ep_dict[eid]['states'].append(next_states[i])

        dst = out / f'{split_name}.hdf5'
        opts = dict(compression='gzip', compression_opts=4)
        n_trans = 0
        with h5py.File(str(dst), 'w') as fout:
            grp = fout.require_group('episodes')
            for new_idx, (_, ep) in enumerate(sorted(ep_dict.items())):
                g = grp.require_group(str(new_idx))
                obs_arr = np.stack(ep['obs']).astype(np.uint8)     # (T+1, H, W, C)
                act_arr = np.stack(ep['actions'])                   # (T,) or (T, d)
                st_arr  = np.stack(ep['states']).astype(np.float32) # (T+1, 4)
                g.create_dataset('observations', data=obs_arr, **opts)
                g.create_dataset('actions',      data=act_arr, **opts)
                g.create_dataset('states',       data=st_arr,  **opts)
                n_trans += len(ep['actions'])
            fout.attrs['n_episodes']    = len(ep_dict)
            fout.attrs['n_transitions'] = n_trans
        print(f'[convert] {split_name}: {len(ep_dict)} episodes  '
              f'{n_trans:,} transitions  → {dst}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='Source flat .h5 file')
    p.add_argument('--out', required=True, help='Output directory for split HDF5 files')
    args = p.parse_args()
    convert(args.src, args.out)


if __name__ == '__main__':
    main()
