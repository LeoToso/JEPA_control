"""Quick validation of a generated CartPole HDF5 dataset.

Prints:
  - Layout (flat / nested), num episodes, obs shape, states shape
  - Per-dimension statistics (min, max, mean, std) for all 4 state channels
  - Histogram buckets for theta (|theta| < 0.15 vs larger)
  - Action range check
  - Visual sanity: first and last frame of episode 0 as ASCII art proxy (pixel stats)

Usage:
    python experiments/validate_dataset.py data/cartpole_visual_v3
    python experiments/validate_dataset.py data/cartpole_visual_v2  # for comparison
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import h5py


def _load_all_states_and_actions(hdf5_path: Path, max_eps: int = 9999):
    """Return (states, actions) as numpy arrays from an HDF5 dataset."""
    all_states  = []
    all_actions = []
    n_eps       = 0
    with h5py.File(hdf5_path, 'r') as f:
        root    = f['episodes'] if 'episodes' in f else f
        ep_keys = sorted(root.keys(),
                         key=lambda x: int(x) if x.isdigit() else x)[:max_eps]
        for ek in ep_keys:
            grp = root[ek]
            if 'states' in grp:
                all_states.append(grp['states'][:])
            if 'actions' in grp:
                all_actions.append(grp['actions'][:])
            n_eps += 1
    states  = np.concatenate(all_states,  axis=0) if all_states  else np.empty((0, 4))
    actions = np.concatenate(all_actions, axis=0) if all_actions else np.empty((0, 1))
    return states, actions, n_eps


def _layout_info(hdf5_path: Path) -> dict:
    with h5py.File(hdf5_path, 'r') as f:
        top = list(f.keys())
        nested = 'episodes' in f
        root = f['episodes'] if nested else f
        ep_keys = sorted(root.keys(),
                         key=lambda x: int(x) if x.isdigit() else x)
        n_eps = len(ep_keys)
        grp = root[ep_keys[0]]
        obs_shape    = grp['observations'].shape if 'observations' in grp else None
        states_shape = grp['states'].shape       if 'states'       in grp else None
        acts_shape   = grp['actions'].shape      if 'actions'      in grp else None
        ep_len = obs_shape[0] if obs_shape else '?'
    return {'layout': 'nested(episodes/<n>)' if nested else 'flat(<key>)',
            'top_keys': top[:5], 'n_eps': n_eps,
            'obs_shape': obs_shape, 'states_shape': states_shape,
            'acts_shape': acts_shape, 'ep_len': ep_len}


def _theta_histogram(theta: np.ndarray) -> str:
    """Return a string summary of the theta distribution."""
    edges = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, np.inf]
    counts, _ = np.histogram(np.abs(theta), bins=edges)
    total = len(theta)
    lines = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        label = f'[{lo:.2f}, {hi:.2f})' if hi != np.inf else f'[{lo:.2f}, ∞   )'
        bar   = '█' * int(30 * counts[i] / max(total, 1))
        pct   = 100.0 * counts[i] / max(total, 1)
        lines.append(f'  |θ| {label}  {counts[i]:6d} ({pct:5.1f}%)  {bar}')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dataset_dir', help='Path to dataset directory with train.hdf5')
    p.add_argument('--split', default='train', help='train / val / test')
    p.add_argument('--max-eps', type=int, default=9999)
    args = p.parse_args()

    ds_dir  = Path(args.dataset_dir)
    hdf5    = next((ds_dir / f'{args.split}{ext}'
                    for ext in ('.hdf5', '.h5')
                    if (ds_dir / f'{args.split}{ext}').exists()), None)
    if hdf5 is None:
        print(f'[error] No {args.split}.hdf5 / .h5 found in {ds_dir}')
        sys.exit(1)

    print(f'\n{"="*60}')
    print(f'Dataset: {hdf5}')
    print(f'{"="*60}')

    info = _layout_info(hdf5)
    print(f'Layout          : {info["layout"]}')
    print(f'Num episodes    : {info["n_eps"]}')
    print(f'Obs shape       : {info["obs_shape"]}')
    print(f'States shape    : {info["states_shape"]}')
    print(f'Actions shape   : {info["acts_shape"]}')

    states, actions, n_eps = _load_all_states_and_actions(hdf5, args.max_eps)
    N = len(states)
    print(f'\nTotal transitions (loaded): {N}  ({n_eps} episodes)')

    if N == 0:
        print('[warning] No state data found.')
        return

    names = ['x (pos)', 'xdot (vel)', 'theta (rad)', 'thetadot (rad/s)']
    print(f'\nState statistics (all {N} transitions):')
    print(f'  {"Channel":<20} {"min":>8} {"max":>8} {"mean":>8} {"std":>8}')
    print(f'  {"-"*56}')
    for i, name in enumerate(names):
        col = states[:, i]
        print(f'  {name:<20} {col.min():8.3f} {col.max():8.3f} '
              f'{col.mean():8.3f} {col.std():8.3f}')

    theta = states[:, 2]
    near_eq = np.sum(np.abs(theta) < 0.15)
    print(f'\nTheta coverage:')
    print(f'  near-eq |θ|<0.15 rad : {near_eq}/{N} ({100.*near_eq/N:.1f}%)')
    print(f'  large   |θ|≥0.15 rad : {N-near_eq}/{N} ({100.*(N-near_eq)/N:.1f}%)')
    print(f'\nTheta histogram (|θ| bucketed):')
    print(_theta_histogram(theta))

    if len(actions) > 0:
        a = actions.flatten()
        print(f'\nAction statistics:')
        print(f'  min={a.min():.3f}  max={a.max():.3f}  mean={a.mean():.3f}  std={a.std():.3f}')

    print(f'\n{"="*60}')
    print('Diagnosis:')
    pct_near = 100. * near_eq / N
    if pct_near > 95:
        print(f'  ⚠  {pct_near:.1f}% near-equilibrium — very low theta diversity.')
        print('     Encoder trained on this data may not distinguish small angles.')
        print('     Consider regenerating with --pole-angle-range 0.5.')
    elif pct_near > 70:
        print(f'  ~  {pct_near:.1f}% near-equilibrium — moderate diversity.')
        print('     State supervision should help, but more diverse data is better.')
    else:
        print(f'  ✓  {pct_near:.1f}% near-equilibrium — good theta diversity.')
        print('     This dataset should enable reliable theta encoding.')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
