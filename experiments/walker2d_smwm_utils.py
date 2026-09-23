#!/usr/bin/env python
"""Walker2d-adapted SMWM bundle loader, ridge-probe fitter, and latent helpers.

Re-exports encode_obs, predict_one, RidgeStateProbe, local_jacobians from
sensorimotor_probe_utils — just with walker2d defaults (17-D state, action_scale=1.0).

HDF5 format expected in data/walker2d_fs5_64/{train,val,test}.hdf5:
  episodes/{i}/observations  (T+1, H, W, 3) uint8
  episodes/{i}/states        (T+1, 17)       float32
  episodes/{i}/actions       (T, 6)          float32
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import torch
import torch.nn as nn
import yaml

from models.sensorimotor_world_model import SensorimotorWorldModel
from experiments.sensorimotor_probe_utils import (
    encode_obs, predict_one, RidgeStateProbe, local_jacobians,
)

# Walker2d state layout
#   obs17 = qpos[1:9]  (8-D: z, torso_ang, joints)
#         + qvel[0:9]  (9-D)
# Full MuJoCo state: qpos (9-D), qvel (9-D)
OBS_DIM    = 17
ACTION_DIM = 6
QPOS_DIM   = 9
QVEL_DIM   = 9

HEALTHY_Z_MIN   = 0.8    # used by gt_icem (real physics); ignored in latent planner
HEALTHY_Z_MAX   = 2.0
HEALTHY_ANG_MAX = 1.0    # primary fall criterion for latent planner


# ── bundle loader ─────────────────────────────────────────────────────────────

def load_walker_bundle(ckpt_path: str, cfg_path: str,
                       device_name: str = 'cuda') -> dict:
    """Load a Walker2d SMWM checkpoint.

    Overrides sensorimotor_probe_utils defaults with walker2d-correct values:
      state_mean / state_std: 17-D (read from checkpoint)
      action_scale: 1.0 (read from checkpoint, fallback 1.0)
    """
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')

    with open(cfg_path) as f:
        env_cfg = yaml.safe_load(f)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = dict(ckpt['model_config'])

    model = SensorimotorWorldModel(model_cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    state_mean   = np.asarray(
        ckpt.get('state_mean', np.zeros(OBS_DIM)), dtype=np.float32)
    state_std    = np.asarray(
        ckpt.get('state_std',  np.ones(OBS_DIM)),  dtype=np.float32)
    action_scale = float(ckpt.get('action_scale', 1.0))

    print(f'[walker bundle] model={Path(ckpt_path).parent.name}  '
          f'latent_dim={model_cfg.get("latent_dim", "?")}  '
          f'action_scale={action_scale}  '
          f'state_mean[0]={state_mean[0]:.3f}')

    return {
        'device':       device,
        'env_cfg':      env_cfg,
        'checkpoint':   ckpt,
        'model':        model,
        'model_cfg':    model_cfg,
        'state_mean':   state_mean,
        'state_std':    state_std,
        'action_scale': action_scale,
    }


# ── state conversion ──────────────────────────────────────────────────────────

def gym_obs_to_mj_state(obs17: np.ndarray,
                        x_pos: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Walker2d gym obs (17-D) → (qpos 9-D, qvel 9-D).

    qpos[0] = x_pos (global x, not in obs; set to 0 for contact detection).
    """
    obs = np.asarray(obs17, dtype=np.float64)
    qpos = np.empty(QPOS_DIM, dtype=np.float64)
    qpos[0]   = x_pos        # global x (unobserved)
    qpos[1:9] = obs[0:8]     # z, torso_ang, 6 joint angles
    qvel      = obs[8:17].astype(np.float64)   # 9 velocities
    return qpos, qvel


def mj_state_to_gym_obs(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    """(qpos 9-D, qvel 9-D) → Walker2d gym obs (17-D)."""
    return np.concatenate([qpos[1:9], qvel[:9]]).astype(np.float32)


def is_healthy_obs(obs17: np.ndarray) -> bool:
    """Health check from decoded obs (latent planner).

    z_height (obs[0]) is not used: probe R² for that dimension is unreliable.
    Fall is detected via torso tilt angle alone (R² ≈ 0.94).
    """
    ang = float(obs17[1])   # qpos[2]  — torso tilt angle
    return abs(ang) < HEALTHY_ANG_MAX


# ── MLP state probe ───────────────────────────────────────────────────────────

class _MLPNet(nn.Module):
    def __init__(self, z_dim: int, obs_dim: int = 17, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, obs_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPStateProbe:
    """Trained MLP z (latent_dim,) → obs17 (17,).  Drop-in for RidgeStateProbe."""

    def __init__(self, z_dim: int, obs_dim: int = 17, hidden: int = 128):
        self._net    = _MLPNet(z_dim, obs_dim, hidden)
        self._device = torch.device('cpu')

    def to(self, device) -> 'MLPStateProbe':
        self._device = torch.device(device) if not isinstance(device, torch.device) else device
        self._net    = self._net.to(self._device)
        return self

    def __call__(self, z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(np.asarray(z, dtype=np.float32))
            if t.ndim == 1:
                t = t.unsqueeze(0)
            return self._net(t.to(self._device))[0].cpu().numpy().astype(np.float32)


def fit_walker_mlp_probe(
    bundle:       dict,
    hdf5_dir:     str,
    split:        str   = 'train',
    max_episodes: int   = 300,
    image_size:   int   = 64,
    hidden:       int   = 128,
    n_epochs:     int   = 30,
    lr:           float = 1e-3,
    batch_size:   int   = 512,
) -> MLPStateProbe:
    """Fit MLPStateProbe (z → gym_obs 17-D) on walker2d HDF5 data."""
    from torch.utils.data import DataLoader, TensorDataset

    hdf5_path = Path(hdf5_dir) / f'{split}.hdf5'
    model     = bundle['model']
    device    = bundle['device']
    z_dim     = int(bundle['model_cfg'].get('latent_dim', 192))

    zs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    with torch.no_grad():
        with h5py.File(hdf5_path, 'r') as f:
            ep_grp  = f['episodes']
            ep_keys = sorted(ep_grp.keys(), key=lambda k: int(k))[:max_episodes]
            print(f'[mlp-probe] encoding {len(ep_keys)} episodes from {hdf5_path.name}')

            for ep_key in ep_keys:
                ep    = ep_grp[ep_key]
                obs_t = ep['observations'][:]   # (T+1, H, W, 3)
                st_t  = ep['states'][:]         # (T+1, 17)
                T     = obs_t.shape[0] - 1
                if T < 2:
                    continue

                if obs_t.shape[1] != image_size or obs_t.shape[2] != image_size:
                    import torch.nn.functional as F_
                    t = torch.from_numpy(obs_t).permute(0, 3, 1, 2).float()
                    t = F_.interpolate(t, (image_size, image_size),
                                       mode='bilinear', align_corners=False)
                    obs_t = t.permute(0, 2, 3, 1).byte().numpy()

                for t in range(1, T + 1):
                    z = encode_obs(bundle, obs_t[t], obs_t[t - 1], st_t[t])
                    zs.append(z[0].cpu().numpy())
                    ys.append(st_t[t])

    Z = torch.from_numpy(np.stack(zs)).float()   # (N, z_dim)
    Y = torch.from_numpy(np.stack(ys)).float()   # (N, 17)
    print(f'[mlp-probe] N={len(Z)} samples  z_dim={z_dim}  training {n_epochs} epochs …')

    probe = MLPStateProbe(z_dim, obs_dim=Y.shape[1], hidden=hidden)
    probe.to(device)
    net = probe._net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    dl  = DataLoader(TensorDataset(Z, Y), batch_size=batch_size, shuffle=True)

    for ep in range(n_epochs):
        total = 0.
        for zb, yb in dl:
            zb, yb = zb.to(device), yb.to(device)
            loss = torch.nn.functional.mse_loss(net(zb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f'  epoch {ep+1:3d}/{n_epochs}  loss={total/len(dl):.5f}', flush=True)

    net.eval()
    with torch.no_grad():
        Y_hat = net(Z.to(device)).cpu().numpy()
        Y_np  = Y.numpy()
        r2 = float(1 - np.mean((Y_hat - Y_np) ** 2) / np.var(Y_np))
    print(f'[mlp-probe] train R²={r2:.4f}')

    # Per-dimension R² (iCEM-critical dims starred)
    _LABELS = ['z_h','ang','kL','aL','hR','kR','aR','fR',
               'xvel','zvel','ang_v','kL_v','aL_v','hR_v','kR_v','aR_v','fR_v']
    _CRIT   = {0,1,2,3,4,5,6,7,8}
    ss_res  = ((Y_hat - Y_np) ** 2).sum(axis=0)
    ss_tot  = ((Y_np - Y_np.mean(axis=0)) ** 2).sum(axis=0)
    r2_per  = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    parts   = [f'{l}={v:.3f}{"★" if i in _CRIT else ""}' for i,(l,v) in enumerate(zip(_LABELS, r2_per))]
    print('[mlp-probe] per-dim R²: ' + '  '.join(parts))
    return probe


def fit_walker_rollout_probe(
    bundle:        dict,
    hdf5_dir:      str,
    split:         str   = 'train',
    max_episodes:  int   = 300,
    rollout_steps: int   = 3,
    image_size:    int   = 64,
    hidden:        int   = 128,
    n_epochs:      int   = 30,
    lr:            float = 1e-3,
    batch_size:    int   = 512,
) -> MLPStateProbe:
    """Fit MLPStateProbe calibrated for latent PREDICTOR outputs.

    Unlike fit_walker_mlp_probe (which trains on z values from the visual
    encoder), this function:
      1. Encodes z_t from real observations.
      2. Rolls the predictor forward `rollout_steps` steps using the real
         actions from the dataset.
      3. Trains the probe to predict state_{t+k} from z_{t+k}_pred.

    The resulting probe maps predictor-output z → obs17 rather than
    encoder-output z → obs17, which gives a meaningful directional signal
    (x_vel sign, magnitude) during multi-step latent planning.

    Use `rollout_steps` equal to the model's training horizon (e.g. 3 for
    ms_sr) for best in-distribution calibration.
    """
    from torch.utils.data import DataLoader, TensorDataset

    hdf5_path = Path(hdf5_dir) / f'{split}.hdf5'
    device    = bundle['device']
    z_dim     = int(bundle['model_cfg'].get('latent_dim', 192))
    scale     = bundle['action_scale']

    zs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    with torch.no_grad():
        with h5py.File(hdf5_path, 'r') as f:
            ep_grp  = f['episodes']
            ep_keys = sorted(ep_grp.keys(), key=lambda k: int(k))[:max_episodes]
            print(f'[rollout-probe] encoding {len(ep_keys)} episodes '
                  f'(k={rollout_steps}) from {hdf5_path.name}')

            for ep_key in ep_keys:
                ep    = ep_grp[ep_key]
                obs_t = ep['observations'][:]   # (T+1, H, W, 3)
                st_t  = ep['states'][:]         # (T+1, 17)
                acts  = ep['actions'][:]        # (T, 6)
                T     = obs_t.shape[0] - 1
                if T < rollout_steps + 2:
                    continue

                if obs_t.shape[1] != image_size or obs_t.shape[2] != image_size:
                    import torch.nn.functional as F_
                    t_ = torch.from_numpy(obs_t).permute(0, 3, 1, 2).float()
                    t_ = F_.interpolate(t_, (image_size, image_size),
                                        mode='bilinear', align_corners=False)
                    obs_t = t_.permute(0, 2, 3, 1).byte().numpy()

                for t in range(1, T - rollout_steps + 1):
                    # Encode from real observation at t
                    z = encode_obs(bundle, obs_t[t], obs_t[t - 1], st_t[t])  # (1, D)

                    # Roll forward k steps with real actions
                    for k in range(rollout_steps):
                        a_k = torch.as_tensor(
                            acts[t + k], dtype=z.dtype, device=device
                        ).unsqueeze(0)                                         # (1, 6)
                        a_ctx = bundle['model'].expand_action(
                            a_k / scale).unsqueeze(1)                          # (1, 1, ctx)
                        z = bundle['model'].predict(z.unsqueeze(1), a_ctx)[:, 0]  # (1, D)

                    zs.append(z[0].cpu().numpy())
                    ys.append(st_t[t + rollout_steps])

    Z = torch.from_numpy(np.stack(zs)).float()   # (N, z_dim)
    Y = torch.from_numpy(np.stack(ys)).float()   # (N, 17)
    print(f'[rollout-probe] N={len(Z)} samples  z_dim={z_dim}  '
          f'training {n_epochs} epochs …')

    probe = MLPStateProbe(z_dim, obs_dim=Y.shape[1], hidden=hidden)
    probe.to(device)
    net = probe._net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    dl  = DataLoader(TensorDataset(Z, Y), batch_size=batch_size, shuffle=True)

    for ep in range(n_epochs):
        total = 0.
        for zb, yb in dl:
            zb, yb = zb.to(device), yb.to(device)
            loss = torch.nn.functional.mse_loss(net(zb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f'  epoch {ep+1:3d}/{n_epochs}  loss={total/len(dl):.5f}', flush=True)

    net.eval()
    with torch.no_grad():
        Y_hat = net(Z.to(device)).cpu().numpy()
        Y_np  = Y.numpy()
        r2 = float(1 - np.mean((Y_hat - Y_np) ** 2) / np.var(Y_np))
    print(f'[rollout-probe] train R²={r2:.4f}')

    _LABELS = ['z_h','ang','kL','aL','hR','kR','aR','fR',
               'xvel','zvel','ang_v','kL_v','aL_v','hR_v','kR_v','aR_v','fR_v']
    _CRIT   = {0,1,2,3,4,5,6,7,8}
    ss_res  = ((Y_hat - Y_np) ** 2).sum(axis=0)
    ss_tot  = ((Y_np - Y_np.mean(axis=0)) ** 2).sum(axis=0)
    r2_per  = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    parts   = [f'{l}={v:.3f}{"★" if i in _CRIT else ""}' for i,(l,v) in enumerate(zip(_LABELS, r2_per))]
    print('[rollout-probe] per-dim R²: ' + '  '.join(parts))
    return probe


# ── ridge probe fitter ────────────────────────────────────────────────────────

@torch.no_grad()
def fit_walker_ridge_probe(
    bundle:       dict,
    hdf5_dir:     str,
    split:        str = 'train',
    max_episodes: int = 300,
    ridge:        float = 1e-3,
    image_size:   int = 64,
) -> RidgeStateProbe:
    """Fit RidgeStateProbe (z → gym_obs 17-D) on walker2d HDF5 data.

    Reads episodes/{i}/{observations, states} from {split}.hdf5.
    """
    hdf5_path = Path(hdf5_dir) / f'{split}.hdf5'
    model  = bundle['model']
    device = bundle['device']

    zs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    with h5py.File(hdf5_path, 'r') as f:
        ep_grp   = f['episodes']
        ep_keys  = sorted(ep_grp.keys(), key=lambda k: int(k))[:max_episodes]
        print(f'[ridge] fitting on {len(ep_keys)} episodes from {hdf5_path.name}')

        for ep_key in ep_keys:
            ep    = ep_grp[ep_key]
            obs_t = ep['observations'][:]   # (T+1, H, W, 3) uint8
            st_t  = ep['states'][:]         # (T+1, 17) float32
            T     = obs_t.shape[0] - 1

            if T < 2:
                continue

            # Resize if needed
            if obs_t.shape[1] != image_size or obs_t.shape[2] != image_size:
                import torch.nn.functional as F_
                t = torch.from_numpy(obs_t).permute(0, 3, 1, 2).float()
                t = F_.interpolate(t, (image_size, image_size), mode='bilinear',
                                   align_corners=False)
                obs_t = t.permute(0, 2, 3, 1).byte().numpy()

            for t in range(1, T + 1):
                z = encode_obs(bundle, obs_t[t], obs_t[t - 1], st_t[t])
                zs.append(z[0].cpu().numpy())   # (latent_dim,)
                ys.append(st_t[t])              # (17,)

    Z = np.stack(zs)   # (N, latent_dim)
    Y = np.stack(ys)   # (N, 17)
    probe = RidgeStateProbe().fit(Z, Y, ridge=ridge)
    print(f'[ridge] fit on N={len(Z)} samples  '
          f'train R²={1 - np.mean((probe(Z) - Y)**2) / np.var(Y):.4f}')
    return probe


# ── latent helpers ────────────────────────────────────────────────────────────

@torch.no_grad()
def latent_step(bundle: dict, z: torch.Tensor,
                action: np.ndarray) -> torch.Tensor:
    """One latent prediction step for Walker2D (6-D action).

    z      : (1, latent_dim) tensor on bundle device
    action : (6,) float32 in env units (will be normalised by action_scale)
    Returns: (1, latent_dim) tensor
    """
    model = bundle['model']
    scale = bundle['action_scale']
    if z.ndim == 1:
        z = z.unsqueeze(0)                                           # (1, D)
    a     = torch.as_tensor(action, dtype=z.dtype,
                             device=z.device).unsqueeze(0)           # (1, 6)
    a_ctx = model.expand_action(a / scale).unsqueeze(1)             # (1, 1, ctx)
    return model.predict(z.unsqueeze(1), a_ctx)[:, 0]               # (1, D)


@torch.no_grad()
def latent_step_batch(bundle: dict, z_batch: torch.Tensor,
                      a_batch: torch.Tensor) -> torch.Tensor:
    """Batched latent prediction step for Walker2D.

    z_batch : (N, latent_dim) tensor on bundle device
    a_batch : (N, 6) tensor on bundle device, in env units
    Returns : (N, latent_dim) tensor
    """
    model = bundle['model']
    scale = bundle['action_scale']
    a_ctx = model.expand_action(a_batch / scale).unsqueeze(1)       # (N, 1, ctx)
    return model.predict(z_batch.unsqueeze(1), a_ctx)[:, 0]         # (N, D)


def decode_z(z: torch.Tensor, probe: RidgeStateProbe) -> np.ndarray:
    """Decode latent z (1, latent_dim) → gym obs (17,) via linear probe."""
    z_np = z[0].cpu().numpy() if torch.is_tensor(z) else np.asarray(z).ravel()
    return probe(z_np).astype(np.float32)
