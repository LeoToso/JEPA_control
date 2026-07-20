"""Pre-resize a discrete CartPole HDF5 dataset from its stored image size to a smaller one.

Reads train/val/test.hdf5 from --src, resizes all observation frames to
--size x --size using bilinear interpolation (torch), writes identical HDF5
structure to --dst.  Run once; training then loads the smaller images directly.

Usage:
    python experiments/resize_dataset.py \
        --src data/cartpole_visual \
        --dst data/cartpole_visual_64 \
        --size 64
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import h5py
from tqdm import tqdm


def resize_split(src_path: Path, dst_path: Path, size: int) -> None:
    if not src_path.exists():
        print(f'[skip] {src_path} not found')
        return

    print(f'  {src_path} → {dst_path}')
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
        # Copy top-level metadata attributes / datasets (action_scale, etc.)
        for key in src.keys():
            if key == 'episodes':
                continue
            src.copy(key, dst)
        for attr_name, attr_val in src.attrs.items():
            dst.attrs[attr_name] = attr_val

        ep_grp_src = src['episodes']
        ep_grp_dst = dst.require_group('episodes')
        ep_keys = sorted(ep_grp_src.keys(), key=int)

        for ep_key in tqdm(ep_keys, desc=dst_path.name, leave=False):
            ep_src = ep_grp_src[ep_key]
            ep_dst = ep_grp_dst.create_group(ep_key)

            # Copy non-observation datasets verbatim
            for ds_name in ep_src.keys():
                if ds_name == 'observations':
                    continue
                ep_src.copy(ds_name, ep_dst)

            # Resize observations: (T+1, H, W, C) uint8 → (T+1, size, size, C) uint8
            obs = ep_src['observations'][:]                   # (T+1, H, W, C) uint8
            T1, H, W, C = obs.shape
            if H == size and W == size:
                ep_dst.create_dataset('observations', data=obs, compression='lzf')
                continue

            t = torch.from_numpy(obs).permute(0, 3, 1, 2).float()  # (T+1, C, H, W)
            t = torch.nn.functional.interpolate(
                t, size=(size, size), mode='bilinear', align_corners=False)
            obs_small = t.permute(0, 2, 3, 1).to(torch.uint8).numpy()  # (T+1, s, s, C)
            ep_dst.create_dataset('observations', data=obs_small, compression='lzf')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='data/cartpole_visual',
                   help='Source HDF5 dataset directory')
    p.add_argument('--dst', default='data/cartpole_visual_64',
                   help='Destination directory for resized dataset')
    p.add_argument('--size', type=int, default=64, help='Target image size (square)')
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    print(f'Resizing {src} → {dst} at {args.size}×{args.size}')

    for split in ('train', 'val', 'test'):
        resize_split(src / f'{split}.hdf5', dst / f'{split}.hdf5', args.size)

    print('Done.')


if __name__ == '__main__':
    main()
