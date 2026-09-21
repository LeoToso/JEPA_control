#!/usr/bin/env python
"""Convert our cartpole split-HDF5 dataset to dino-wm .pth format.

dino-wm data layout (one directory per env):
  states.pth             torch.Tensor  [N, T_max, 4]   float32
  actions.pth            torch.Tensor  [N, T_max, 1]   float32  (coarse timestep)
  seq_lengths.pth        torch.Tensor  [N]             int64
  obses/
    episode_000.pth      torch.Tensor  [T, H, W, C]   uint8

Actions are stored at COARSE timestep (one per stored frame).
dino-wm's TrajSlicerDataset then:
  act = act[start : start + num_frames * frameskip]       # coarse slice
  act = rearrange(act, "(n f) d -> n (f d)", n=num_frames) # concat frameskip actions

Use frameskip=1 at training time (our data already has frame_skip=5 baked in).
If you want to further temporally subsample (e.g. frameskip=2), increase it
in the train.py command — no data reconversion needed.

Usage
-----
  python experiments/convert_cartpole_to_dinowm.py \\
      --src  data/cartpole_excitation_depth4_fs5_128 \\
      --out  /mnt/t7shield/dinowm_cartpole
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch


def read_split(hdf5_path: Path):
    """Yield (observations, actions, states) per episode."""
    episodes = []
    with h5py.File(hdf5_path, 'r') as f:
        ep_grp = f['episodes']
        for k in sorted(ep_grp.keys(), key=int):
            ep = ep_grp[k]
            obs = ep['observations'][:]  # (T+1, H, W, C) uint8
            act = ep['actions'][:]       # (T,) or (T, 1) float32
            st  = ep['states'][:]        # (T+1, 4) float32
            episodes.append((obs, act, st))
    return episodes


def convert(src_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_dir = out_dir / 'obses'
    obs_dir.mkdir(exist_ok=True)

    all_episodes = []
    for split in ('train', 'val', 'test'):
        p = src_dir / f'{split}.hdf5'
        if p.exists():
            eps = read_split(p)
            all_episodes.extend(eps)
            print(f'  {split}: {len(eps)} episodes')

    N         = len(all_episodes)
    T_coarses = [len(ep[1]) for ep in all_episodes]  # len(actions) = T_coarse
    T_max     = max(T_coarses)

    # Determine dims from first episode
    sample_act = all_episodes[0][1]
    action_dim = 1 if sample_act.ndim == 1 else sample_act.shape[1]
    state_dim  = all_episodes[0][2].shape[1]
    H, W, C    = all_episodes[0][0].shape[1:]

    print(f'\n[convert] N={N} episodes  T_max={T_max}  '
          f'action_dim={action_dim}  state_dim={state_dim}  img={H}x{W}x{C}')
    print(f'  actions.pth shape will be [{N}, {T_max}, {action_dim}]')
    print(f'  → use frameskip=1 in dino-wm train.py (data is already coarse-step)')

    # Pre-allocate padded tensors
    states_all  = torch.zeros(N, T_max, state_dim, dtype=torch.float32)
    actions_all = torch.zeros(N, T_max, action_dim, dtype=torch.float32)
    seq_lengths = torch.zeros(N, dtype=torch.int64)

    for i, (obs_arr, act_arr, st_arr) in enumerate(all_episodes):
        T = len(act_arr)
        seq_lengths[i] = T

        # States at steps 1…T (exclude initial frame)
        states_all[i, :T] = torch.from_numpy(st_arr[1:T + 1].astype(np.float32))

        # Actions: ensure shape [T, 1]
        if act_arr.ndim == 1:
            act_arr = act_arr[:, None]
        actions_all[i, :T] = torch.from_numpy(act_arr.astype(np.float32))

        # Images: [T, H, W, C] uint8 (exclude terminal obs)
        torch.save(torch.from_numpy(obs_arr[:T].astype(np.uint8)),
                   obs_dir / f'episode_{i:03d}.pth')

        if (i + 1) % 200 == 0:
            print(f'  {i + 1}/{N} episodes processed')

    torch.save(states_all,  out_dir / 'states.pth')
    torch.save(actions_all, out_dir / 'actions.pth')
    torch.save(seq_lengths, out_dir / 'seq_lengths.pth')

    print(f'\n[saved]  states       {tuple(states_all.shape)}')
    print(f'[saved]  actions      {tuple(actions_all.shape)}')
    print(f'[saved]  seq_lengths  {tuple(seq_lengths.shape)}')
    print(f'[saved]  {N} image files → {obs_dir}/')
    print(f'\n[done] → {out_dir}')


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Convert cartpole split-HDF5 to dino-wm .pth format.')
    p.add_argument('--src', required=True,
                   help='Source dir with train/val/test.hdf5')
    p.add_argument('--out', required=True,
                   help='Output dir for dino-wm dataset')
    args = p.parse_args()
    convert(Path(args.src), Path(args.out))


if __name__ == '__main__':
    main()
