#!/usr/bin/env python
"""Sweep over multiple probe random seeds; score each by SAC vs random cost ranking.

For each probe seed:
  1. Fit encoder MLP probe with that seed
  2. Load a few SAC episodes and random episodes from HDF5
  3. From each episode's z_0, run batched_rollout_cost on SAC actions vs random actions
  4. Score = fraction of episodes where SAC actions score lower cost than random

A probe with high score is more likely to work with latent iCEM.
Saves the best probe to --best-probe-path.

Usage
-----
MUJOCO_GL=egl python experiments/probe_sweep.py \\
    --ckpt  /path/to/model_final.pt \\
    --cfg   configs/walker2d_smwm_fwd_endpoint_inverse_act1.yaml \\
    --hdf5-dir data/walker2d_mixed_sac_fs5_64 \\
    --seeds 0 1 2 3 4 5 6 7 8 9 \\
    --best-probe-path results/probes/best_sweep_probe.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch
import h5py


def score_probe(bundle, probe, hdf5_dir: str, n_eval: int = 40,
                horizon: int = 30) -> float:
    """Return fraction of episodes where SAC-ish actions beat random-action cost."""
    from experiments.walker2d_smwm_utils import latent_step_batch
    from experiments.sensorimotor_probe_utils import encode_obs
    from experiments.latent_icem_walker import batched_rollout_cost

    device = bundle['device']
    hdf5_path = Path(hdf5_dir) / 'train.hdf5'
    probe_net = probe._net.to(device).eval()

    sac_beat_random = 0
    total = 0

    with h5py.File(hdf5_path, 'r') as f:
        ep_grp  = f['episodes']
        ep_keys = sorted(ep_grp.keys(), key=lambda k: int(k))
        # Use SAC episodes (positive x_vel) when possible
        sac_keys = []
        for k in ep_keys:
            ep_acts = np.array(ep_grp[k]['actions'])
            ep_states = np.array(ep_grp[k]['states'])
            # SAC episodes have x_vel (obs[8]) > 1.0 on average
            xvel = ep_states[:, 8].mean() if ep_states.ndim == 2 else 0.0
            if xvel > 1.0:
                sac_keys.append(k)
        eval_keys = (sac_keys if len(sac_keys) >= n_eval // 2 else ep_keys)[:n_eval]

        for ep_key in eval_keys:
            ep    = ep_grp[ep_key]
            obs_t = ep['observations'][:]  # (T+1, H, W, C)
            st_t  = ep['states'][:]        # (T+1, 17)
            acts  = ep['actions'][:]       # (T, 6)
            T     = obs_t.shape[0] - 1
            if T < horizon:
                continue

            # Encode z_0 from obs[1] (not obs[0] which is initial; obs[1] is after first action)
            with torch.no_grad():
                z0 = encode_obs(bundle, obs_t[1], obs_t[0], st_t[1])  # (1, D)

            # Build SAC action sequence from dataset
            sac_acts = acts[:horizon].astype(np.float32)  # (H, 6) — these are stored actions

            # Build random action sequence
            rng = np.random.default_rng(int(ep_key) if ep_key.isdigit() else 0)
            rand_acts = rng.uniform(-1.0, 1.0, (horizon, 6)).astype(np.float32)

            # Stack both into (2, H, 6) for batched cost evaluation
            actions_batch = np.stack([sac_acts, rand_acts], axis=0)

            costs = batched_rollout_cost(
                bundle, probe_net, z0, actions_batch,
                wx=0.2, wh=4.0, wu=1e-3, cf=50.0,
                wz=0.0, wang=1.5, height_target=1.35,
                wjoint=0.02, wsmooth=0.05, wback=2.0,
                healthy_ang_max=1.0,
            )
            if costs[0] < costs[1]:
                sac_beat_random += 1
            total += 1

    return sac_beat_random / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',            required=True)
    p.add_argument('--cfg',             default=None)
    p.add_argument('--hdf5-dir',        required=True)
    p.add_argument('--seeds',           type=int, nargs='+', default=list(range(10)))
    p.add_argument('--n-eval',          type=int, default=40,
                   help='Number of episodes to use for probe scoring')
    p.add_argument('--probe-horizon',   type=int, default=30,
                   help='Rollout horizon used in probe scoring')
    p.add_argument('--probe-episodes',  type=int, default=200)
    p.add_argument('--probe-epochs',    type=int, default=30)
    p.add_argument('--best-probe-path', required=True,
                   help='Where to save the best probe found')
    p.add_argument('--device',          default='cuda')
    args = p.parse_args()

    from experiments.walker2d_smwm_utils import load_walker_bundle, fit_walker_mlp_probe

    def _cfg(ckpt, override):
        if override:
            return override
        for name in ['env_config.yaml', 'config.yaml']:
            c = Path(ckpt).parent / name
            if c.exists():
                return str(c)
        raise FileNotFoundError(f'No config found near {ckpt}')

    print('[probe-sweep] Loading model bundle …')
    bundle = load_walker_bundle(args.ckpt, _cfg(args.ckpt, args.cfg), args.device)

    best_score = -1.0
    best_probe = None
    results = []

    for seed in args.seeds:
        print(f'\n[probe-sweep] === Seed {seed} ===')
        torch.manual_seed(seed)
        np.random.seed(seed)

        probe = fit_walker_mlp_probe(
            bundle, args.hdf5_dir,
            max_episodes=args.probe_episodes,
            n_epochs=args.probe_epochs,
        )

        score = score_probe(bundle, probe, args.hdf5_dir,
                            n_eval=args.n_eval, horizon=args.probe_horizon)
        print(f'[probe-sweep] seed={seed}  SAC-beats-random score: {score:.3f}')
        results.append((seed, score))

        if score > best_score:
            best_score = score
            best_probe = probe
            # Save incrementally in case later seeds are worse
            Path(args.best_probe_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_probe._net.state_dict(), args.best_probe_path)
            print(f'[probe-sweep]   ★ New best! Saved to {args.best_probe_path}')

    print('\n[probe-sweep] Summary:')
    for seed, score in sorted(results, key=lambda x: -x[1]):
        print(f'  seed={seed:3d}  score={score:.3f}')
    print(f'\nBest: seed={max(results, key=lambda x: x[1])[0]}  '
          f'score={best_score:.3f}  → {args.best_probe_path}')


if __name__ == '__main__':
    main()
