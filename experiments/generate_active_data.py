"""Generate an active (action-rich) dataset for JEPA predictor retraining.

The passive dataset (u=0 + random) never lets the predictor see that actions
have consequences: B_aug is computed from the Jacobian at u=0 on a network
that was never trained on u≠0 data.  This script fixes that by collecting:

  LQR episodes      — GT LQR stabilization from varied initial conditions.
                      Diverse, physically meaningful actions with consistent
                      state-action correlations.
  PRBS episodes     — Pseudo-Random Binary Sequence near equilibrium.
                      Persistent excitation: action changes sign frequently,
                      forcing the predictor to learn ∂z/∂u directly.
  Passive episodes  — u=0 from near-equilibrium to preserve rho(A_aug) > 1
                      learning (instability visible in latent space).

Output: per-episode HDF5 format compatible with DiscreteHDF5TrajectoryDataset.

Usage:
    python experiments/generate_active_data.py \\
        --config configs/cartpole_jepa_sf_w3_fs5_v9_phase2.yaml \\
        --out-dir data/cartpole_visual_fs5_active \\
        --n-lqr 600 --lqr-ep-len 100 \\
        --n-prbs 300 --prbs-ep-len 60 \\
        --n-passive 100 --passive-ep-len 80 \\
        --lqr-init-max 0.15 --seed 0
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import yaml


def _compute_lqr_gain(env_cfg, frame_skip):
    from ground_truth.cartpole_gt import CartpoleGroundTruth
    from control.lqr import solve_discrete_lqr
    gt = CartpoleGroundTruth(
        mass_cart=env_cfg['mass_cart'], mass_pole=env_cfg['mass_pole'],
        pole_length=env_cfg['pole_length'], gravity=env_cfg['gravity'],
        dt=env_cfg['dt'] * frame_skip,
    )
    Q = np.diag([1.0, 1.0, 100.0, 10.0])
    R = 0.01 * np.eye(1)
    K, _, cl_eigs = solve_discrete_lqr(gt.A_star, gt.B_star, Q, R)
    rho_cl = float(np.max(np.abs(cl_eigs)))
    print(f'[lqr] K={np.round(K, 4)}  rho_cl={rho_cl:.4f}')
    return K   # (1, 4)  u = -(K @ x)


def _collect_lqr_episode(env, K, ep_len, init_max, rng, action_lb, action_ub,
                          noise_std=0.0):
    """Collect one LQR episode.  u = clip(-K@x + noise, lb, ub)."""
    x0 = rng.uniform(-init_max, init_max, 4).astype(np.float32)
    obs, state, _ = env.reset_to_state(x0)
    obs_list, state_list, act_list = [obs.copy()], [state.copy()], []
    for _ in range(ep_len):
        u_lqr = float(-(K @ state)[0])
        if noise_std > 0:
            u_lqr += float(rng.normal(0, noise_std))
        u = float(np.clip(u_lqr, action_lb, action_ub))
        next_obs, next_state, _, _, _ = env.step(u)
        act_list.append(np.float32(u))
        obs_list.append(next_obs.copy())
        state_list.append(next_state.copy())
        obs, state = next_obs, next_state
    return (np.stack(obs_list).astype(np.uint8),   # (T+1, H, W, C)
            np.array(act_list, dtype=np.float32),   # (T,)
            np.stack(state_list).astype(np.float32))  # (T+1, 4)


def _collect_prbs_episode(env, ep_len, amplitude, flip_prob, init_max, rng,
                           action_lb, action_ub):
    """Pseudo-Random Binary Sequence: hold ±amplitude, flip sign w/ prob flip_prob."""
    x0 = rng.uniform(-init_max, init_max, 4).astype(np.float32)
    obs, state, _ = env.reset_to_state(x0)
    obs_list, state_list, act_list = [obs.copy()], [state.copy()], []
    sign = float(rng.choice([-1.0, 1.0]))
    for _ in range(ep_len):
        if rng.random() < flip_prob:
            sign = -sign
        u = float(np.clip(sign * amplitude, action_lb, action_ub))
        next_obs, next_state, _, _, _ = env.step(u)
        act_list.append(np.float32(u))
        obs_list.append(next_obs.copy())
        state_list.append(next_state.copy())
        obs, state = next_obs, next_state
    return (np.stack(obs_list).astype(np.uint8),
            np.array(act_list, dtype=np.float32),
            np.stack(state_list).astype(np.float32))


def _collect_passive_episode(env, ep_len, init_max, rng):
    """u=0: pole falls freely, showing open-loop instability."""
    x0 = rng.uniform(-init_max, init_max, 4).astype(np.float32)
    obs, state, _ = env.reset_to_state(x0)
    obs_list, state_list, act_list = [obs.copy()], [state.copy()], []
    for _ in range(ep_len):
        next_obs, next_state, _, _, _ = env.step(0.0)
        act_list.append(np.float32(0.0))
        obs_list.append(next_obs.copy())
        state_list.append(next_state.copy())
        obs, state = next_obs, next_state
    return (np.stack(obs_list).astype(np.uint8),
            np.array(act_list, dtype=np.float32),
            np.stack(state_list).astype(np.float32))


def _write_hdf5(path: Path, episodes_data: list) -> None:
    """Write episodes in DiscreteHDF5TrajectoryDataset-compatible format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    opts = dict(compression='gzip', compression_opts=4)
    with h5py.File(str(path), 'w') as f:
        grp = f.require_group('episodes')
        for i, (obs, acts, states) in enumerate(episodes_data):
            g = grp.require_group(str(i))
            g.create_dataset('observations', data=obs,    **opts)
            g.create_dataset('actions',      data=acts,   **opts)
            g.create_dataset('states',       data=states, **opts)
        f.attrs['n_episodes'] = len(episodes_data)
        f.attrs['n_transitions'] = sum(len(a) for _, a, _ in episodes_data)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config',         required=True)
    p.add_argument('--out-dir',        default='data/cartpole_visual_fs5_active')
    # LQR episodes
    p.add_argument('--n-lqr',          type=int,   default=600)
    p.add_argument('--lqr-ep-len',     type=int,   default=100)
    p.add_argument('--lqr-init-max',   type=float, default=0.15,
                   help='Max initial state magnitude for LQR episodes')
    p.add_argument('--lqr-noise-std',  type=float, default=0.5,
                   help='Gaussian noise on LQR action (N)')
    # PRBS episodes
    p.add_argument('--n-prbs',         type=int,   default=300)
    p.add_argument('--prbs-ep-len',    type=int,   default=60)
    p.add_argument('--prbs-amplitude', type=float, default=4.0,
                   help='PRBS action magnitude (N)')
    p.add_argument('--prbs-flip-prob', type=float, default=0.15)
    p.add_argument('--prbs-init-max',  type=float, default=0.05)
    # Passive episodes
    p.add_argument('--n-passive',      type=int,   default=100)
    p.add_argument('--passive-ep-len', type=int,   default=80)
    p.add_argument('--passive-init-max', type=float, default=0.05)
    # Split
    p.add_argument('--train-frac',     type=float, default=0.85)
    p.add_argument('--val-frac',       type=float, default=0.10)
    p.add_argument('--seed',           type=int,   default=0)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_cfg    = cfg['environment']
    model_cfg  = cfg.get('model', {})
    frame_skip = int(env_cfg.get('frame_skip', 1))
    action_lb  = float(env_cfg['action_range'][0])
    action_ub  = float(env_cfg['action_range'][1])

    rng = np.random.RandomState(args.seed)
    K   = _compute_lqr_gain(env_cfg, frame_skip)

    from envs.cartpole_visual import ContinuousCartpoleVisual
    env = ContinuousCartpoleVisual(
        frame_skip=frame_skip,
        image_size=int(env_cfg['image_size']),
        mass_cart=env_cfg['mass_cart'],
        mass_pole=env_cfg['mass_pole'],
        pole_length=env_cfg['pole_length'],
        gravity=env_cfg['gravity'],
        dt=env_cfg['dt'],
        seed=args.seed,
    )

    episodes = []

    # ── LQR episodes ──────────────────────────────────────────────────────────
    n_lqr = args.n_lqr
    print(f'[collect] LQR: {n_lqr} ep × {args.lqr_ep_len} steps  '
          f'init_max={args.lqr_init_max}  noise_std={args.lqr_noise_std}')
    t0 = time.time()
    for i in range(n_lqr):
        obs, acts, states = _collect_lqr_episode(
            env, K, args.lqr_ep_len, args.lqr_init_max, rng,
            action_lb, action_ub, noise_std=args.lqr_noise_std)
        episodes.append((obs, acts, states))
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{n_lqr}  ({time.time()-t0:.1f}s)')

    # ── PRBS episodes ─────────────────────────────────────────────────────────
    n_prbs = args.n_prbs
    print(f'[collect] PRBS: {n_prbs} ep × {args.prbs_ep_len} steps  '
          f'amp={args.prbs_amplitude}  flip_prob={args.prbs_flip_prob}')
    for i in range(n_prbs):
        obs, acts, states = _collect_prbs_episode(
            env, args.prbs_ep_len, args.prbs_amplitude, args.prbs_flip_prob,
            args.prbs_init_max, rng, action_lb, action_ub)
        episodes.append((obs, acts, states))
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{n_prbs}')

    # ── Passive episodes ──────────────────────────────────────────────────────
    n_passive = args.n_passive
    print(f'[collect] Passive: {n_passive} ep × {args.passive_ep_len} steps  '
          f'init_max={args.passive_init_max}')
    for i in range(n_passive):
        obs, acts, states = _collect_passive_episode(
            env, args.passive_ep_len, args.passive_init_max, rng)
        episodes.append((obs, acts, states))

    env.close()

    total_transitions = sum(len(a) for _, a, _ in episodes)
    print(f'\n[data] Total: {len(episodes)} episodes  {total_transitions:,} transitions')
    print(f'       Composition: {n_lqr} LQR + {n_prbs} PRBS + {n_passive} passive')

    # ── Shuffle and split ─────────────────────────────────────────────────────
    idx = np.arange(len(episodes))
    rng.shuffle(idx)
    n_tr  = int(len(idx) * args.train_frac)
    n_val = int(len(idx) * args.val_frac)
    splits = {
        'train': idx[:n_tr],
        'val':   idx[n_tr:n_tr + n_val],
        'test':  idx[n_tr + n_val:],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_idx in splits.items():
        split_eps = [episodes[i] for i in split_idx]
        path = out_dir / f'{split_name}.hdf5'
        n_tr_ep = len(split_eps)
        n_tr_transitions = sum(len(a) for _, a, _ in split_eps)
        print(f'[write] {split_name}: {n_tr_ep} ep  {n_tr_transitions:,} transitions  → {path}')
        _write_hdf5(path, split_eps)

    # ── Save generation config ────────────────────────────────────────────────
    import datetime
    gen_cfg = {
        'source_config':      args.config,
        'frame_skip':         frame_skip,
        'image_size':         int(env_cfg['image_size']),
        'n_lqr':              n_lqr,
        'lqr_ep_len':         args.lqr_ep_len,
        'lqr_init_max':       args.lqr_init_max,
        'lqr_noise_std':      args.lqr_noise_std,
        'n_prbs':             n_prbs,
        'prbs_ep_len':        args.prbs_ep_len,
        'prbs_amplitude':     args.prbs_amplitude,
        'prbs_flip_prob':     args.prbs_flip_prob,
        'prbs_init_max':      args.prbs_init_max,
        'n_passive':          n_passive,
        'passive_ep_len':     args.passive_ep_len,
        'passive_init_max':   args.passive_init_max,
        'seed':               args.seed,
        'total_transitions':  total_transitions,
        'generation_timestamp': datetime.datetime.now().isoformat(),
    }
    cfg_path = out_dir / 'generation_config.yaml'
    with open(cfg_path, 'w') as f:
        yaml.dump(gen_cfg, f, default_flow_style=False)
    print(f'[write] generation_config.yaml → {cfg_path}')
    print('\nDone. Use this dataset with:')
    print(f'  --data {out_dir}  (in training config or phase-2 trainer)')


if __name__ == '__main__':
    main()
