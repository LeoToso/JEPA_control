#!/usr/bin/env python
"""Evaluate frozen DINOv2 state decoding with exact-dynamics CartPole CEM."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from control.lqr import solve_discrete_lqr
from experiments.evaluate_cartpole_control import make_env, physical_macro_jacobian
from experiments.evaluate_cartpole_encoder_gt_cem import GroundTruthCEM
from experiments.probe_cartpole_nonlinear_instability import probe_metrics


STATE_NAMES = ('x', 'x_dot', 'theta', 'theta_dot')
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sample_refs(path, n_samples, seed):
    refs = []
    with h5py.File(path, 'r') as f:
        for ep_key in f['episodes']:
            n = len(f['episodes'][ep_key]['states'])
            refs.extend((ep_key, t) for t in range(n))
    rng = np.random.RandomState(seed)
    if n_samples < len(refs):
        idx = rng.choice(len(refs), n_samples, replace=False)
        refs = [refs[int(i)] for i in idx]
    return refs


class DINOv2Features(nn.Module):
    """Frozen ViT-S/14 descriptor retaining low-order patch geometry."""
    def __init__(self, model_name, device):
        super().__init__()
        print(f'[dinov2] loading {model_name} from official torch hub ...')
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', model_name, pretrained=True,
            trust_repo=True).to(device).eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.register_buffer(
            'mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer(
            'std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def preprocess(self, images):
        x = images.float() / 255.
        x = F.interpolate(x, size=(224, 224), mode='bicubic',
                          align_corners=False, antialias=True)
        return (x - self.mean) / self.std

    @torch.no_grad()
    def frame_descriptor(self, images):
        out = self.backbone.forward_features(self.preprocess(images))
        cls = out['x_norm_clstoken']
        patches = out['x_norm_patchtokens']
        n = patches.shape[1]
        side = int(round(n ** .5))
        if side * side != n:
            raise RuntimeError(f'expected square patch grid, got {n} tokens')
        grid = patches.reshape(patches.shape[0], side, side, patches.shape[2])
        coord = torch.linspace(-1., 1., side, device=patches.device)
        yy, xx = torch.meshgrid(coord, coord, indexing='ij')
        mean = grid.mean((1, 2))
        x_moment = (grid * xx[None, :, :, None]).mean((1, 2))
        y_moment = (grid * yy[None, :, :, None]).mean((1, 2))
        return torch.cat((cls, mean, x_moment, y_moment), dim=1)

    @torch.no_grad()
    def pair_descriptor(self, current, previous):
        # A single backbone batch is faster than two sequential forwards.
        joined = torch.cat((current, previous), dim=0)
        feat = self.frame_descriptor(joined)
        cur, prev = feat.chunk(2, dim=0)
        return torch.cat((cur, prev, cur - prev), dim=1)


def encode_split(extractor, dataset_dir, split, n_samples, batch_size,
                 seed, device):
    path = Path(dataset_dir) / f'{split}.hdf5'
    refs = sample_refs(path, n_samples, seed)
    features, states = [], []
    print(f'[data:{split}] encoding {len(refs):,} two-frame observations ...')
    with h5py.File(path, 'r') as f:
        for start in range(0, len(refs), batch_size):
            rows = refs[start:start + batch_size]
            current, previous, target = [], [], []
            for ep_key, t in rows:
                ep = f['episodes'][ep_key]
                current.append(ep['observations'][t])
                previous.append(ep['observations'][max(t - 1, 0)])
                target.append(ep['states'][t])
            cur = torch.from_numpy(np.stack(current)).permute(0, 3, 1, 2).to(device)
            prev = torch.from_numpy(np.stack(previous)).permute(0, 3, 1, 2).to(device)
            features.append(extractor.pair_descriptor(cur, prev).cpu().numpy())
            states.append(np.asarray(target, dtype=np.float32))
            if (start // batch_size + 1) % 10 == 0:
                print(f'  {min(start + batch_size, len(refs)):,}/{len(refs):,}')
    return np.concatenate(features), np.concatenate(states)


class StateMLP(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 4))

    def forward(self, x):
        return self.net(x)


class FittedDecoder:
    def __init__(self, model, feature_mean, feature_std,
                 state_mean, state_std, device):
        self.model = model
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.state_mean = state_mean
        self.state_std = state_std
        self.device = device

    def __call__(self, features):
        x = ((np.asarray(features) - self.feature_mean) /
             self.feature_std).astype(np.float32)
        with torch.no_grad():
            y = self.model(torch.from_numpy(x).to(self.device)).cpu().numpy()
        state = y * self.state_std + self.state_mean
        state[..., 2] = np.arctan2(np.sin(state[..., 2]), np.cos(state[..., 2]))
        return state


def train_decoder(features, states, state_mean, state_std, local_mask,
                  local_weight, hidden, epochs, batch_size, lr, seed, device):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    feature_mean = features.mean(0).astype(np.float32)
    feature_std = np.maximum(features.std(0), 1e-4).astype(np.float32)
    x = ((features - feature_mean) / feature_std).astype(np.float32)
    y = ((states - state_mean) / state_std).astype(np.float32)
    weights = np.where(local_mask, local_weight, 1.).astype(np.float32)
    model = StateMLP(x.shape[1], hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    xt, yt, wt = map(torch.from_numpy, (x, y, weights))
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        total = 0.
        model.train()
        for start in range(0, len(x), batch_size):
            idx = order[start:start + batch_size]
            xb, yb = xt[idx].to(device), yt[idx].to(device)
            wb = wt[idx].to(device)
            per_row = torch.mean((model(xb) - yb) ** 2, dim=1)
            loss = torch.sum(wb * per_row) / torch.sum(wb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss) * len(idx)
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 50 == 0:
            print(f'  epoch {epoch + 1:03d}/{epochs} loss={total / len(x):.6f}')
    model.eval()
    return FittedDecoder(model, feature_mean, feature_std,
                         state_mean, state_std, device)


def local_metrics(truth, prediction, region, K):
    result = {}
    masks = {
        'global': np.ones(len(truth), dtype=bool),
        'local': (np.abs(truth) <= region[None]).all(1),
        'very_local': (np.abs(truth) <= (.5 * region)[None]).all(1),
    }
    for name, mask in masks.items():
        if not np.any(mask):
            result[name] = {'n': 0}
            continue
        row = probe_metrics(truth[mask], prediction[mask])
        u_true = -(truth[mask] @ K.ravel())
        u_pred = -(prediction[mask] @ K.ravel())
        row['action_mae'] = float(np.mean(np.abs(u_pred - u_true)))
        result[name] = row
    return result


@torch.no_grad()
def encode_online(extractor, obs, prev_obs, device):
    def tensor(x):
        return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
    return extractor.pair_descriptor(tensor(obs), tensor(prev_obs)).cpu().numpy()


def evaluate_cem(cfg, args, extractor, decoder, equilibrium,
                 terminal_q, initial_states, device):
    bounds = cfg['environment'].get('action_range', [-10., 10.])
    ctrl = cfg.get('control', {})
    rows = []
    for trial, x0 in enumerate(initial_states):
        env = make_env(cfg, args.seed + trial)
        planner = GroundTruthCEM(
            cfg, args.cem_horizon, args.cem_samples, args.cem_elites,
            args.cem_iters, args.cem_init_std, args.cem_warm_start_std,
            1., terminal_q, bounds[0], bounds[1], device)
        obs, state, _ = env.reset_to_state(x0)
        prev_obs = obs
        norms, terminated = [], False
        for step in range(args.steps):
            feature = encode_online(extractor, obs, prev_obs, device)
            estimate = decoder(feature)[0] - equilibrium
            torch.manual_seed(args.seed * 10000 + trial * 1000 + step)
            action = planner.plan_state(estimate)
            norms.append(float(np.linalg.norm(state)))
            old_obs = obs
            obs, state, _, done, _ = env.step(action)
            prev_obs = old_obs
            if done:
                terminated = True
                break
        env.close()
        norms.append(float(np.linalg.norm(state)))
        hold = min(int(ctrl.get('success_hold_steps', 10)), len(norms))
        success = bool(not terminated and np.all(
            np.asarray(norms[-hold:]) <
            float(ctrl.get('stabilization_threshold', .1))))
        rows.append({'success': success, 'terminated': terminated,
                     'steps': len(norms) - 1, 'final_error': norms[-1],
                     'max_error': max(norms)})
    return {
        'success_rate': float(np.mean([r['success'] for r in rows])),
        'termination_rate': float(np.mean([r['terminated'] for r in rows])),
        'mean_final_error': float(np.mean([r['final_error'] for r in rows])),
        'trials': rows,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--dino-model', default='dinov2_vits14')
    p.add_argument('--train-samples', type=int, default=6000)
    p.add_argument('--test-samples', type=int, default=1500)
    p.add_argument('--encode-batch-size', type=int, default=64)
    p.add_argument('--decoder-hidden', type=int, default=256)
    p.add_argument('--decoder-epochs', type=int, default=200)
    p.add_argument('--decoder-batch-size', type=int, default=256)
    p.add_argument('--decoder-lr', type=float, default=1e-3)
    p.add_argument('--local-weight', type=float, default=25.)
    p.add_argument('--control-trials', type=int, default=3)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--cem-horizon', type=int, default=2)
    p.add_argument('--cem-samples', type=int, default=2048)
    p.add_argument('--cem-elites', type=int, default=128)
    p.add_argument('--cem-iters', type=int, default=8)
    p.add_argument('--cem-init-std', type=float, default=3.)
    p.add_argument('--cem-warm-start-std', type=float, default=1.)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', required=True)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    extractor = DINOv2Features(args.dino_model, device)
    z_train, s_train = encode_split(
        extractor, args.data, 'train', args.train_samples,
        args.encode_batch_size, args.seed, device)
    z_test, s_test = encode_split(
        extractor, args.data, 'test', args.test_samples,
        args.encode_batch_size, args.seed + 1, device)
    state_mean, state_std = s_train.mean(0), np.maximum(s_train.std(0), 1e-5)
    region = np.asarray(cfg.get('training', {}).get(
        'local_sampling_region', [.1, .25, .05, .5]), dtype=np.float32)
    local_train = (np.abs(s_train) <= region[None]).all(1)
    print(f'[data] feature_dim={z_train.shape[1]}  '
          f'local_train={local_train.sum()}/{len(local_train)}')

    env = make_env(cfg, args.seed + 999)
    eq_obs, _, _ = env.reset_to_state(np.zeros(4, dtype=np.float32))
    eq_feature = encode_online(extractor, eq_obs, eq_obs, device)
    A, B = physical_macro_jacobian(env)
    K, terminal_q, poles = solve_discrete_lqr(
        A, B, np.diag([1., 1., 10., 1.]), np.array([[.01]]))
    env.close()
    rng = np.random.RandomState(args.seed)
    scale = float(cfg.get('control', {}).get('init_scale', .05))
    initial_states = [rng.uniform(-scale, scale, 4).astype(np.float32)
                      for _ in range(args.control_trials)]

    result = {'encoder': args.dino_model, 'state_names': STATE_NAMES,
              'feature_dim': int(z_train.shape[1]),
              'local_region': region.tolist(),
              'rho_lqr': float(np.max(np.abs(poles))), 'decoders': {}}
    for name, weight in (('global_mlp', 1.),
                         ('local_weighted_mlp', args.local_weight)):
        print(f'[decoder:{name}] training, local_weight={weight:g} ...')
        decoder = train_decoder(
            z_train, s_train, state_mean, state_std, local_train,
            weight, args.decoder_hidden, args.decoder_epochs,
            args.decoder_batch_size, args.decoder_lr,
            args.seed + (0 if weight == 1. else 1), device)
        prediction = decoder(z_test)
        equilibrium = decoder(eq_feature)[0]
        metrics = local_metrics(s_test, prediction, region, K)
        cem = evaluate_cem(
            cfg, args, extractor, decoder, equilibrium,
            terminal_q, initial_states, device)
        result['decoders'][name] = {
            'equilibrium_decode': equilibrium.tolist(),
            'metrics': metrics, 'cem': cem,
        }
        local = metrics['local']
        print(f'[{name}] global_R2={np.round(metrics["global"]["r2"], 3).tolist()}')
        print(f'  local n={local["n"]} RMSE={np.round(local["rmse"], 4).tolist()} '
              f'action_MAE={local["action_mae"]:.3f}')
        print(f'  CEM success={cem["success_rate"]:.1%} '
              f'term={cem["termination_rate"]:.1%} '
              f'final={cem["mean_final_error"]:.3f}')

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f'[done] {out}')


if __name__ == '__main__':
    main()
