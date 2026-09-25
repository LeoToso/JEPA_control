"""Shared utilities for SensorimotorWorldModel CartPole diagnostics."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml

from data.dataset import make_discrete_dataloaders
from envs.cartpole_visual import ContinuousCartpoleVisual
from models.sensorimotor_world_model import SensorimotorWorldModel


def make_env(cfg, seed):
    e, d = cfg['environment'], cfg.get('data', {})
    return ContinuousCartpoleVisual(
        frame_skip=int(e.get('frame_skip', 1)),
        image_size=int(e.get('image_size', 64)),
        action_range=tuple(e.get('action_range', [-10, 10])),
        mass_cart=float(e.get('mass_cart', 1.0)),
        mass_pole=float(e.get('mass_pole', 0.1)),
        pole_length=float(e.get('pole_length', 0.5)),
        gravity=float(e.get('gravity', 9.8)),
        dt=float(e.get('dt', 0.02)),
        friction_cart=float(e.get('friction_cart', 0.0)),
        friction_pole=float(e.get('friction_pole', 0.0)),
        theta_threshold=float(d.get('max_abs_theta', 1.2)),
        seed=seed)


def obs_tensor(obs, device):
    return (torch.from_numpy(np.asarray(obs).copy()).float().permute(2, 0, 1)
            .unsqueeze(0).to(device) / 255.)


def _build_obs_input(bundle, obs, prev_obs):
    """Return the observation tensor the encoder expects.

    frame_stack > 1: `obs` may be a list/tuple of frame_stack HxWxC arrays
                     (oldest first).  A bare array is repeated frame_stack times
                     (correct for initialisation / equilibrium).
    use_frame_diff:  single obs + prev_obs; handled downstream in model.encode.
    default (stack=1, no diff): single frame.
    """
    model = bundle['model']
    device = bundle['device']
    fs = int(model.frame_stack)
    if fs > 1:
        frames = list(obs) if isinstance(obs, (list, tuple)) else [obs] * fs
        return torch.cat([obs_tensor(f, device) for f in frames], dim=1)
    # fs == 1: callers may pass make_frame_buffer output ([obs]) — unwrap it
    if isinstance(obs, (list, tuple)):
        obs = obs[0]
    return obs_tensor(obs, device)


def make_frame_buffer(bundle, initial_obs):
    """Return a list of frame_stack copies of initial_obs (oldest first)."""
    fs = int(bundle['model'].frame_stack)
    return [initial_obs] * fs


def push_frame(frame_buffer, new_obs):
    """Append new_obs to frame_buffer and drop the oldest frame in-place."""
    frame_buffer.pop(0)
    frame_buffer.append(new_obs)


def load_bundle(checkpoint_path, config_path, device_name='cuda'):
    device = torch.device(
        device_name if torch.cuda.is_available() else 'cpu')
    with open(config_path) as f:
        env_cfg = yaml.safe_load(f)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = dict(checkpoint['model_config'])
    model = SensorimotorWorldModel(model_cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    state_mean = np.asarray(
        checkpoint.get('state_mean', np.zeros(4)), dtype=np.float32)
    state_std = np.asarray(
        checkpoint.get('state_std', np.ones(4)), dtype=np.float32)
    action_scale = float(checkpoint.get('action_scale', 10.))
    return {
        'device': device, 'env_cfg': env_cfg, 'checkpoint': checkpoint,
        'model': model, 'model_cfg': model_cfg,
        'state_mean': state_mean, 'state_std': state_std,
        'action_scale': action_scale,
    }


def normalize_state(state, bundle):
    return ((np.asarray(state, dtype=np.float32) - bundle['state_mean']) /
            bundle['state_std'])


@torch.no_grad()
def encode_obs(bundle, obs, prev_obs, state):
    """Encode an observation into a latent vector.

    obs      : HxWxC ndarray, OR a list of frame_stack such arrays (oldest first)
               when model.frame_stack > 1.  A bare array is repeated frame_stack
               times (suitable for initialisation).
    prev_obs : used only when model.use_frame_diff is True (frame_stack == 1).
    """
    model = bundle['model']
    device = bundle['device']
    proprio = None
    if model.use_proprio:
        norm = normalize_state(state, bundle)
        idx = model.proprio_indices
        if idx is not None:
            norm = norm[idx]
        proprio = torch.as_tensor(norm, device=device).float().reshape(1, -1)
    obs_in = _build_obs_input(bundle, obs, prev_obs)
    prev_in = (obs_tensor(prev_obs, device)
               if model.use_frame_diff and model.frame_stack == 1 else None)
    return model.encode(obs_in, prev_in, proprio)


def equilibrium_latent(bundle):
    env = make_env(bundle['env_cfg'], 987)
    state = np.zeros(4, dtype=np.float32)
    obs, _, _ = env.reset_to_state(state)
    # Pass obs directly so _build_obs_input handles frame_stack via
    # its bare-array path ([obs]*fs), avoiding a 4-D tensor when frame_stack==1.
    z = encode_obs(bundle, obs, obs, state)
    env.close()
    return z.detach()


def predict_one(bundle, z, action):
    model = bundle['model']
    if not torch.is_tensor(action):
        action = torch.as_tensor(action, dtype=z.dtype, device=z.device)
    action = action.reshape(-1)  # (n,)
    n = action.shape[0]
    # expand_action handles both schemes (repeat or zero-pad) via model flag.
    a_raw = (action / bundle['action_scale']).unsqueeze(-1)  # (n, 1)
    a_ctx = model.expand_action(a_raw).unsqueeze(1)          # (n, 1, ctx)
    if z.ndim == 1:
        z = z[None]
    bz = z.shape[0]
    if bz == 1 and n > 1:
        z = z.expand(n, -1)
    elif n == 1 and bz > 1:
        a_ctx = a_ctx.expand(bz, -1, -1)
    return model.predict(z[:, None], a_ctx)[:, 0]


def local_jacobians(bundle, z=None):
    if z is None:
        z = equilibrium_latent(bundle)
    z0 = z.detach().reshape(-1).requires_grad_(True)
    u0 = torch.zeros(1, device=z0.device, dtype=z0.dtype,
                     requires_grad=True)

    def fz(zz):
        return predict_one(bundle, zz[None], u0)[0]

    def fu(uu):
        return predict_one(bundle, z0[None], uu)[0]

    a = torch.autograd.functional.jacobian(fz, z0, vectorize=True)
    b = torch.autograd.functional.jacobian(fu, u0, vectorize=True)
    with torch.no_grad():
        z_next = predict_one(bundle, z.detach(), 0.)
    return (a.detach().cpu().numpy(), b.detach().cpu().numpy(),
            float(torch.linalg.vector_norm(z_next - z).cpu()))


class RidgeStateProbe:
    def fit(self, z, y, ridge=1e-3):
        self.z_mean = z.mean(0)
        self.z_std = np.maximum(z.std(0), 1e-6)
        self.y_mean = y.mean(0)
        x = (z - self.z_mean) / self.z_std
        x = np.c_[x, np.ones(len(x))]
        reg = ridge * np.eye(x.shape[1]); reg[-1, -1] = 0.
        self.weight = np.linalg.solve(
            x.T @ x + reg, x.T @ (y - self.y_mean))
        return self

    def __call__(self, z):
        z = np.asarray(z)
        one = z.ndim == 1
        z = np.atleast_2d(z)
        x = (z - self.z_mean) / self.z_std
        pred = np.c_[x, np.ones(len(x))] @ self.weight + self.y_mean
        return pred[0] if one else pred


@torch.no_grad()
def fit_state_probe(bundle, data_path, n_samples=4000, batch_size=128,
                    ridge=1e-3, split='train'):
    loaders = make_discrete_dataloaders(
        data_path, batch_size=batch_size, num_workers=0, horizon=1,
        frame_stack=1, state_mean=bundle['state_mean'],
        state_std=bundle['state_std'],
        action_scale=bundle['action_scale'],
        target_image_size=int(bundle['model_cfg'].get('image_size', 128)),
        preload_obs=True)
    zs, ys = [], []
    for batch in loaders[split]:
        current = batch['obs_seq'][:, 0].to(bundle['device']).float() / 255.
        previous = batch['prev_obs'].to(bundle['device']).float() / 255.
        proprio = None
        if bundle['model'].use_proprio:
            proprio = batch['states'][:, 0].to(bundle['device']).float()
        z = bundle['model'].encode(current, previous, proprio)
        y_norm = batch['states'][:, 0].numpy()
        y = y_norm * bundle['state_std'][None] + bundle['state_mean'][None]
        zs.append(z.cpu().numpy()); ys.append(y)
        if sum(len(x) for x in zs) >= n_samples:
            break
    z = np.concatenate(zs)[:n_samples]
    y = np.concatenate(ys)[:n_samples]
    return RidgeStateProbe().fit(z, y, ridge), z, y


def encode_rendered_state(bundle, state):
    env = make_env(bundle['env_cfg'], 321)
    obs, _, _ = env.reset_to_state(np.asarray(state, dtype=np.float32))
    buf = make_frame_buffer(bundle, obs)
    z = encode_obs(bundle, buf, obs, state)
    env.close()
    return z


def gt_rollout(bundle, initial_state, steps, action=0.):
    env = make_env(bundle['env_cfg'], 654)
    obs, state, _ = env.reset_to_state(
        np.asarray(initial_state, dtype=np.float32))
    frame_buf = make_frame_buffer(bundle, obs)
    states = [state.copy()]
    latents = [encode_obs(bundle, frame_buf, obs, state).cpu().numpy()[0]]
    for _ in range(steps):
        prev_obs = obs
        obs, state, _, done, _ = env.step(float(action))
        push_frame(frame_buf, obs)
        states.append(state.copy())
        latents.append(
            encode_obs(bundle, frame_buf, prev_obs, state).cpu().numpy()[0])
        if done:
            break
    env.close()
    return np.asarray(states), np.asarray(latents)


@torch.no_grad()
def learned_rollout(bundle, z0, steps, action=0.):
    model = bundle['model']
    ctx = model.action_context_dim
    scale = bundle['action_scale']
    z = z0.detach().reshape(1, -1)
    values = [z.cpu().numpy()[0]]
    if model.use_action_history:
        # Maintain a proper action-history ring buffer across steps.
        history = z.new_zeros(ctx)  # oldest … newest
        for _ in range(steps):
            a_scaled = float(action) / scale
            history = torch.cat([history[1:],
                                  history.new_tensor([a_scaled])])
            a_ctx = history.reshape(1, 1, ctx)
            z = model.predict(z[:, None], a_ctx)[:, 0]
            values.append(z.cpu().numpy()[0])
    else:
        for _ in range(steps):
            z = predict_one(bundle, z, action)
            values.append(z.cpu().numpy()[0])
    return np.asarray(values)


def finite_horizon_gramian(a, b, horizon=20):
    w = np.zeros((a.shape[0], a.shape[0]), dtype=np.float64)
    akb = np.asarray(b, dtype=np.float64)
    for _ in range(horizon):
        w += akb @ akb.T
        akb = a @ akb
    return w

