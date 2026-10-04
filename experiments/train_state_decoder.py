#!/usr/bin/env python
"""Train a state decoder  z -> x  for one or more SMWM models.

Collects (latent, physical-state) pairs by rolling out the environment
under random actions from diverse initial conditions, then fits a small
MLP  z ∈ R^d -> x ∈ R^4  ([cart_pos, cart_vel, pole_angle, pole_vel])
via MSE regression.

The decoder is saved alongside a summary of test-set R² per state
dimension so the K-step prediction experiment can use it.

Usage
-----
python experiments/train_state_decoder.py \\
    --model "1SP+EP-IDM:/path/ckpt.pt:cfg.yaml" \\
    --model "MSP+EP-IDM:/path/ckpt.pt:cfg.yaml" \\
    --n-episodes 300 \\
    --out-dir results/decoders/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.probe_utils import encode_obs, load_bundle, make_env

STATE_DIM  = 4
STATE_KEYS = ['cart_pos', 'cart_vel', 'pole_angle', 'pole_vel']


def _env_patch(bundle):
    if 'environment' not in bundle['env_cfg']:
        bundle['env_cfg']['environment'] = {
            'frame_skip': int(bundle['model_cfg'].get('frame_skip', 5)),
            'image_size': int(bundle['model_cfg'].get('image_size', 128)),
            'action_range': [-10, 10],
            'mass_cart': 1.0, 'mass_pole': 0.1,
            'pole_length': 0.5, 'gravity': 9.8, 'dt': 0.02,
        }


class StateDecoder(nn.Module):
    """Two-hidden-layer MLP: z -> x."""
    def __init__(self, z_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, STATE_DIM),
        )

    def forward(self, z):
        return self.net(z)


@torch.no_grad()
def collect_data(bundle, n_episodes, steps_per_ep, rng):
    """Roll out random policy; return (Z, X) numpy arrays."""
    action_scale = float(bundle['action_scale'])
    Zs, Xs = [], []

    for ep in range(n_episodes):
        # Diverse initial conditions
        x0 = rng.uniform(-0.15, 0.15, size=4).astype(np.float64)
        env = make_env(bundle['env_cfg'], seed=int(ep))
        obs, state, _ = env.reset_to_state(x0)
        prev_obs = obs.copy()

        for _ in range(steps_per_ep):
            z = encode_obs(bundle, obs, prev_obs, state)
            Zs.append(z.cpu().numpy().flatten())
            Xs.append(state.copy())

            u = float(rng.uniform(-action_scale, action_scale))
            prev_obs = obs.copy()
            obs, state, _, done, _ = env.step(u)
            if done:
                break

        env.close()

    return np.array(Zs, dtype=np.float32), np.array(Xs, dtype=np.float32)


def r2_score(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-10)


def train_decoder(Z_train, X_train, Z_test, X_test,
                  z_dim, device, epochs=200, batch=512, lr=1e-3, hidden=128):
    decoder = StateDecoder(z_dim).to(device)
    opt = torch.optim.Adam(decoder.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Zt = torch.from_numpy(Z_train).to(device)
    Xt = torch.from_numpy(X_train).to(device)
    ds = DataLoader(TensorDataset(Zt, Xt), batch_size=batch, shuffle=True)

    for epoch in range(epochs):
        decoder.train()
        for zb, xb in ds:
            opt.zero_grad()
            loss_fn(decoder(zb), xb).backward()
            opt.step()

        if (epoch + 1) % 50 == 0:
            decoder.eval()
            with torch.no_grad():
                X_pred = decoder(torch.from_numpy(Z_test).to(device)).cpu().numpy()
            r2 = r2_score(X_test, X_pred)
            print(f'    epoch {epoch+1:3d}  '
                  f'R²=[{", ".join(f"{v:.3f}" for v in r2)}]')

    return decoder


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model', action='append', dest='models',
                   metavar='LABEL:CKPT:CFG')
    p.add_argument('--n-episodes', type=int, default=300)
    p.add_argument('--steps-per-ep', type=int, default=60)
    p.add_argument('--test-frac', type=float, default=0.15)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out-dir', required=True)
    args = p.parse_args()

    if not args.models:
        p.error('Provide at least one --model "label:ckpt:cfg"')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    for spec in args.models:
        label, ckpt, cfg = spec.split(':', 2)
        print(f'\n[{label}] loading …')
        bundle = load_bundle(ckpt, cfg, args.device)
        _env_patch(bundle)
        device = bundle['device']

        print(f'  collecting {args.n_episodes} episodes × {args.steps_per_ep} steps …')
        Z, X = collect_data(bundle, args.n_episodes, args.steps_per_ep, rng)
        print(f'  dataset: {len(Z)} samples, z_dim={Z.shape[1]}')

        # Train / test split
        n_test = max(1, int(len(Z) * args.test_frac))
        idx = rng.permutation(len(Z))
        Z_tr, X_tr = Z[idx[n_test:]], X[idx[n_test:]]
        Z_te, X_te = Z[idx[:n_test]],  X[idx[:n_test]]

        print(f'  training decoder ({args.epochs} epochs) …')
        decoder = train_decoder(Z_tr, X_tr, Z_te, X_te,
                                z_dim=Z.shape[1], device=device,
                                epochs=args.epochs, hidden=args.hidden)

        # Final R² report
        decoder.eval()
        with torch.no_grad():
            X_pred = decoder(torch.from_numpy(Z_te).to(device)).cpu().numpy()
        r2 = r2_score(X_te, X_pred)
        print(f'  Test R²:')
        for name, v in zip(STATE_KEYS, r2):
            print(f'    {name:15s}: {v:.4f}')

        # Save
        safe = label.replace('/', '_').replace(' ', '_').replace('+', '_')
        out = out_dir / f'decoder_{safe}.pt'
        torch.save({
            'label': label,
            'z_dim': int(Z.shape[1]),
            'hidden': args.hidden,
            'state_dict': decoder.state_dict(),
            'r2_test': r2.tolist(),
            'state_keys': STATE_KEYS,
        }, out)
        print(f'  saved → {out}')

    print('\n[done]')


if __name__ == '__main__':
    main()
