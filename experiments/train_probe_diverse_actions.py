#!/usr/bin/env python
"""Re-train the MLP probe using diverse (random/colored-noise) actions.

Root cause of latent iCEM backward-walking:
  The original probe was trained on z values encoded directly from SAC frames.
  During iCEM planning, z comes from the latent PREDICTOR under colored-noise
  CEM actions (far from SAC distribution).  The probe is badly calibrated
  for these OOD latent states, so cost estimates are unreliable.

Fix:
  Collect training pairs (z_{t+1}^pred, state_{t+1}^real) by applying the
  SAME diverse action to both the latent dynamics model and the real env.
  The new probe is then calibrated for predictor-output z under CEM-like
  actions.

Data mixture (configurable):
  --sac-episodes   N episodes from HDF5 with SAC actions (rollout probe style)
  --env-episodes   N episodes from real env with diverse actions
  --env-policy     random | colored_noise | mixed

Usage
-----
MUJOCO_GL=glfw python experiments/train_probe_diverse_actions.py \\
    --ckpt  "/path/to/walker2d_mixed_fwd_ep_ar_seed42.pt" \\
    --cfg   configs/walker2d_smwm_fwd_endpoint_inverse_act1.yaml \\
    --hdf5-dir "/path/to/walker2d_mixed_sac_fs5_64" \\
    --sac-episodes 200 --env-episodes 200 \\
    --env-policy mixed \\
    --n-epochs 50 --lr 5e-4 \\
    --out-probe /path/to/probes/fwd_ep_ar_diverse_probe.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent))
sys.path.insert(0, str(_here))

import numpy as np
import torch
import h5py

from walker2d_smwm_utils import (
    MLPStateProbe, load_walker_bundle, latent_step_batch,
    gym_obs_to_mj_state,
)
from sensorimotor_probe_utils import encode_obs

OBS_DIM  = 17
XVEL_IDX = 8


# ── colored-noise action sequence ─────────────────────────────────────────────

def _colored_noise_seq(H: int, beta: float = 0.5, std: float = 0.5,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    fft_len = H // 2 + 1
    freqs   = np.fft.rfftfreq(H)
    freqs[0] = 1.0
    power    = freqs ** (-beta / 2.0)
    power[0] = 0.0
    white  = (rng.standard_normal((6, fft_len))
              + 1j * rng.standard_normal((6, fft_len)))
    colored = white * power[None, :]
    noise   = np.fft.irfft(colored, n=H, axis=-1).T   # (H, 6)
    scale   = noise.std(axis=0, keepdims=True).clip(1e-8)
    return np.clip(noise / scale * std, -1.0, 1.0).astype(np.float32)


# ── SAC-action rollout pairs (from HDF5) ──────────────────────────────────────

def collect_sac_rollout_pairs(bundle, hdf5_dir, split, max_episodes,
                              rollout_steps=1, image_size=64):
    """Train on (z_{t+k}^pred with SAC action, state_{t+k}) from HDF5.

    Same as fit_walker_rollout_probe but returns arrays instead of a trained probe.
    """
    hdf5_path = Path(hdf5_dir) / f'{split}.hdf5'
    device    = bundle['device']
    scale     = bundle['action_scale']

    zs, ys = [], []
    with torch.no_grad():
        with h5py.File(hdf5_path, 'r') as f:
            ep_grp  = f['episodes']
            ep_keys = sorted(ep_grp.keys(), key=lambda k: int(k))[:max_episodes]
            print(f'[sac-pairs]  {split}: {len(ep_keys)} episodes, k={rollout_steps}')

            for ep_key in ep_keys:
                ep    = ep_grp[ep_key]
                obs_t = ep['observations'][:]
                st_t  = ep['states'][:]
                acts  = ep['actions'][:]
                T     = obs_t.shape[0] - 1
                if T < rollout_steps + 2:
                    continue

                if obs_t.shape[1] != image_size or obs_t.shape[2] != image_size:
                    import torch.nn.functional as F_
                    t_ = torch.from_numpy(obs_t).permute(0, 3, 1, 2).float()
                    t_ = F_.interpolate(t_, (image_size, image_size),
                                        mode='bilinear', align_corners=False)
                    obs_t = t_.permute(0, 2, 3, 1).byte().numpy()

                for t in range(T - rollout_steps + 1):
                    prev = obs_t[t - 1] if t > 0 else obs_t[t]
                    z    = encode_obs(bundle, obs_t[t], prev, st_t[t])
                    for k in range(rollout_steps):
                        a_k   = torch.as_tensor(acts[t + k], dtype=z.dtype,
                                                device=device).unsqueeze(0)
                        a_ctx = bundle['model'].expand_action(
                            a_k / scale).unsqueeze(1)
                        z = bundle['model'].predict(z.unsqueeze(1), a_ctx)[:, 0]
                    zs.append(z[0].cpu().numpy())
                    ys.append(st_t[t + rollout_steps])

    print(f'[sac-pairs]  N={len(zs)} samples')
    return np.stack(zs).astype(np.float32), np.stack(ys).astype(np.float32)


# ── Diverse-action pairs (real env + latent model) ────────────────────────────

def collect_diverse_pairs(bundle, hdf5_dir, split, max_episodes,
                          image_size=64, steps_per_ep=200,
                          policy='mixed', beta=0.5, std=0.5,
                          seed=42):
    """Collect (z_{t+1}^pred, state_{t+1}^real) under diverse actions.

    For each episode:
      1. Sample a starting state from the HDF5 dataset.
      2. Set the Walker2d env to that starting state (via set_state).
      3. Apply a diverse action sequence to BOTH the real env and the latent model.
      4. Record (z_{t+1}^pred, real_state_{t+1}) pairs.

    policy:
      'random'        — uniform [-1, 1]
      'colored_noise' — colored noise (β, σ) like iCEM
      'mixed'         — alternates between random and colored_noise
    """
    import gymnasium
    from envs.walker2d_visual import Walker2dVisual

    hdf5_path = Path(hdf5_dir) / f'{split}.hdf5'
    device    = bundle['device']
    rng       = np.random.default_rng(seed)

    env = Walker2dVisual(image_size=image_size, seed=seed)

    # Load starting states from HDF5
    starting_states: list[tuple] = []  # (obs_image, obs17)
    with h5py.File(hdf5_path, 'r') as f:
        ep_grp  = f['episodes']
        ep_keys = sorted(ep_grp.keys(), key=lambda k: int(k))[:max_episodes]
        for ep_key in ep_keys:
            ep    = ep_grp[ep_key]
            obs_t = ep['observations'][:]
            st_t  = ep['states'][:]
            T     = obs_t.shape[0] - 1
            # Sample a few starting points per episode
            idxs  = rng.choice(max(1, T), size=min(3, T), replace=False)
            for idx in idxs:
                img = obs_t[int(idx)]
                if (img.shape[0] != image_size or img.shape[1] != image_size):
                    import torch.nn.functional as F_
                    t_ = torch.from_numpy(img[None]).permute(0, 3, 1, 2).float()
                    t_ = F_.interpolate(t_, (image_size, image_size),
                                        mode='bilinear', align_corners=False)
                    img = t_.squeeze(0).permute(1, 2, 0).byte().numpy()
                starting_states.append((img, st_t[int(idx)]))

    print(f'[diverse-pairs]  {len(starting_states)} starting states, '
          f'policy={policy}, β={beta}, σ={std}')

    zs, ys = [], []
    n_done  = 0

    for i, (start_img, start_state) in enumerate(starting_states):
        # Set real env to starting state
        qpos, qvel = gym_obs_to_mj_state(start_state)
        try:
            env._env.unwrapped.set_state(qpos, qvel)
        except Exception:
            try:
                env._env.unwrapped.set_state(
                    np.clip(qpos, -5, 5), np.clip(qvel, -50, 50))
            except Exception:
                continue

        # Render to get the image at the starting state
        try:
            frame = env._render()
        except Exception:
            continue

        prev_frame = frame.copy()
        cur_state  = start_state.copy()
        cur_img    = frame

        # Decide action policy for this episode
        if policy == 'mixed':
            ep_policy = 'colored_noise' if i % 2 == 0 else 'random'
        else:
            ep_policy = policy

        # Generate action sequence for this episode
        if ep_policy == 'colored_noise':
            action_seq = _colored_noise_seq(steps_per_ep, beta=beta, std=std, rng=rng)
        else:
            action_seq = rng.uniform(-1.0, 1.0,
                                     size=(steps_per_ep, 6)).astype(np.float32)

        for step in range(steps_per_ep):
            a = action_seq[step]

            with torch.no_grad():
                # Encode current frame → z_t
                z_t = encode_obs(bundle, cur_img, prev_frame, cur_state)
                # Apply latent dynamics
                a_t   = torch.as_tensor(a, device=device).unsqueeze(0)
                z_pred = latent_step_batch(bundle, z_t, a_t)

            # Apply same action to real env
            try:
                next_img, next_state, _, done, _ = env.step(a)
            except Exception:
                done = True

            zs.append(z_pred[0].cpu().numpy())
            ys.append(next_state.copy())

            if done:
                break

            prev_frame = cur_img
            cur_img    = next_img
            cur_state  = next_state

        n_done += 1
        if n_done % 20 == 0:
            print(f'[diverse-pairs]  {n_done}/{len(starting_states)} starts  '
                  f'N={len(zs)} pairs so far')

    env.close()
    print(f'[diverse-pairs]  N={len(zs)} total samples')
    return np.stack(zs).astype(np.float32), np.stack(ys).astype(np.float32)


# ── Train probe ───────────────────────────────────────────────────────────────

def train_probe(Z: np.ndarray, Y: np.ndarray, z_dim: int, device,
                hidden: int = 256, n_epochs: int = 50,
                lr: float = 5e-4, batch_size: int = 512) -> MLPStateProbe:
    from torch.utils.data import DataLoader, TensorDataset

    Z_t = torch.from_numpy(Z).float()
    Y_t = torch.from_numpy(Y).float()
    print(f'[train-probe]  N={len(Z_t)} samples  z_dim={z_dim}  '
          f'hidden={hidden}  epochs={n_epochs}')

    probe = MLPStateProbe(z_dim, obs_dim=OBS_DIM, hidden=hidden)
    probe.to(device)
    net = probe._net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    dl  = DataLoader(TensorDataset(Z_t, Y_t), batch_size=batch_size, shuffle=True)

    for ep in range(n_epochs):
        total = 0.
        for zb, yb in dl:
            zb, yb = zb.to(device), yb.to(device)
            loss = torch.nn.functional.mse_loss(net(zb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f'  epoch {ep+1:3d}/{n_epochs}  loss={total/len(dl):.5f}',
                  flush=True)

    net.eval()
    with torch.no_grad():
        Y_hat = net(Z_t.to(device)).cpu().numpy()
    Y_np = Y

    _LABELS = ['z_h','ang','kL','aL','hR','kR','aR','fR',
               'xvel','zvel','ang_v','kL_v','aL_v','hR_v','kR_v','aR_v','fR_v']
    _CRIT   = {0, 1, 2, 3, 4, 5, 6, 7, 8}
    ss_res  = ((Y_hat - Y_np) ** 2).sum(axis=0)
    ss_tot  = ((Y_np - Y_np.mean(axis=0)) ** 2).sum(axis=0)
    r2_per  = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)

    xvel_r2 = r2_per[XVEL_IDX]
    print(f'[train-probe]  overall train R²={r2_per.mean():.4f}  '
          f'xvel R²={xvel_r2:.4f}')
    parts = [f'{l}={v:.3f}{"★" if i in _CRIT else ""}'
             for i, (l, v) in enumerate(zip(_LABELS, r2_per))]
    print('[train-probe] per-dim R²: ' + '  '.join(parts))
    return probe


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--ckpt',          required=True)
    p.add_argument('--cfg',           default=None)
    p.add_argument('--hdf5-dir',      required=True)
    p.add_argument('--out-probe',     required=True,
                   help='Where to save the new probe .pt file')

    # Data mixture
    p.add_argument('--sac-episodes',  type=int, default=200,
                   help='SAC episodes from HDF5 (rollout, k=1); 0 to skip')
    p.add_argument('--env-episodes',  type=int, default=200,
                   help='HDF5 starting states for diverse-action collection')
    p.add_argument('--env-steps',     type=int, default=150,
                   help='Max steps per diverse episode (walker falls quickly)')
    p.add_argument('--env-policy',    default='mixed',
                   choices=['random', 'colored_noise', 'mixed'],
                   help='Action policy for diverse episodes')
    p.add_argument('--cn-beta',       type=float, default=0.5,
                   help='Colored noise beta')
    p.add_argument('--cn-std',        type=float, default=0.5,
                   help='Colored noise sigma')

    # Probe architecture / training
    p.add_argument('--hidden',        type=int, default=256,
                   help='MLP hidden size (256 > 128 original for harder task)')
    p.add_argument('--n-epochs',      type=int, default=50)
    p.add_argument('--lr',            type=float, default=5e-4)
    p.add_argument('--batch-size',    type=int, default=512)

    p.add_argument('--device',        default='cpu',
                   help='cpu or cuda (cpu fine for probe training)')
    p.add_argument('--seed',          type=int, default=42)
    p.add_argument('--image-size',    type=int, default=64)
    args = p.parse_args()

    bundle = load_walker_bundle(args.ckpt, args.cfg, args.device)
    z_dim  = int(bundle['model_cfg'].get('latent_dim', 192))

    Z_list, Y_list = [], []

    # ── SAC rollout pairs ─────────────────────────────────────────────────────
    if args.sac_episodes > 0:
        Zs, Ys = collect_sac_rollout_pairs(
            bundle, args.hdf5_dir, 'train', args.sac_episodes,
            rollout_steps=1, image_size=args.image_size)
        Z_list.append(Zs); Y_list.append(Ys)
        print(f'  SAC pairs: {len(Zs)}')

    # ── Diverse-action pairs ──────────────────────────────────────────────────
    if args.env_episodes > 0:
        Zd, Yd = collect_diverse_pairs(
            bundle, args.hdf5_dir, 'train', args.env_episodes,
            image_size=args.image_size,
            steps_per_ep=args.env_steps,
            policy=args.env_policy,
            beta=args.cn_beta,
            std=args.cn_std,
            seed=args.seed,
        )
        Z_list.append(Zd); Y_list.append(Yd)
        print(f'  diverse pairs: {len(Zd)}')

    Z = np.concatenate(Z_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)

    # Shuffle
    idx = np.random.default_rng(args.seed).permutation(len(Z))
    Z, Y = Z[idx], Y[idx]
    print(f'[main]  total training pairs: {len(Z)}  '
          f'(SAC + diverse, shuffled)')

    # ── Train ─────────────────────────────────────────────────────────────────
    probe = train_probe(Z, Y, z_dim, bundle['device'],
                        hidden=args.hidden,
                        n_epochs=args.n_epochs,
                        lr=args.lr,
                        batch_size=args.batch_size)

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(args.out_probe).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': probe._net.state_dict(),
                'z_dim': z_dim,
                'hidden': args.hidden,
                'note': 'diverse-action probe: trained on z_pred from random/colored-noise actions'},
               args.out_probe)
    print(f'[done]  probe saved to {args.out_probe}')


if __name__ == '__main__':
    main()
