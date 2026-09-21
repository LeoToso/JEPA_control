#!/usr/bin/env python
"""PPO and SAC policy utilities for Walker2d-v4 (CleanRL/HuggingFace checkpoints).

PPO architecture (CleanRL ppo_continuous_action.py):
  Linear(17,64) → Tanh → Linear(64,64) → Tanh → Linear(64,6)
  actor_logstd: Parameter(1, 6)
  HuggingFace: sdpkjc/Walker2d-v4-ppo_fix_continuous_action-seed3

SAC architecture (CleanRL sac_continuous_action.py):
  Linear(17,256) → ReLU → Linear(256,256) → ReLU → fc_mean(256,6)
  deterministic action = tanh(mean) * action_scale + action_bias
  checkpoint is a 3-tuple: (actor_sd, qf1_sd, qf2_sd); no obs normalisation
  HuggingFace: sdpkjc/Walker2d-v4-sac_continuous_action-seed4
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

OBS_DIM    = 17   # Walker2d-v4 gym obs (qpos[1:9] + qvel[:9])
ACTION_DIM = 6    # 6 joint torques, clipped to [-1, 1]


# ── network ───────────────────────────────────────────────────────────────────

class PPOActor(nn.Module):
    """CleanRL ppo_continuous_action actor (64-64 MLP, Tanh activations)."""

    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),     nn.Tanh(),
            nn.Linear(64, action_dim),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic mean action."""
        return self.actor_mean(x)


# ── policy wrapper ────────────────────────────────────────────────────────────

class PPOPolicy:
    """Deterministic PPO policy: obs → action (clipped to [-1, 1])."""

    def __init__(self, actor: PPOActor,
                 obs_mean: np.ndarray | None = None,
                 obs_var:  np.ndarray | None = None,
                 device: str = 'cpu'):
        self.actor    = actor.to(device).eval()
        self.obs_mean = obs_mean   # (OBS_DIM,) or None
        self.obs_var  = obs_var    # (OBS_DIM,) or None
        self.device   = device

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_mean is not None:
            norm = (obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)
            return np.clip(norm, -10.0, 10.0)   # matches CleanRL TransformObservation
        return obs

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        """obs: (17,) float32 → action: (6,) float32 ∈ [-1, 1]."""
        x = self.normalize(obs.astype(np.float32))
        t = torch.from_numpy(x).unsqueeze(0).to(self.device)
        a = self.actor(t)[0].cpu().numpy()
        return np.clip(a, -1.0, 1.0)


# ── checkpoint loading ────────────────────────────────────────────────────────

def _extract_obs_rms_from_sd(sd: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract obs_rms.mean / obs_rms.var if they are embedded in the state dict.

    CleanRL ppo_fix_continuous_action saves obs normalization as flat keys:
      'obs_rms.mean', 'obs_rms.var', 'obs_rms.count'
    """
    if 'obs_rms.mean' in sd and 'obs_rms.var' in sd:
        mean = np.asarray(sd['obs_rms.mean'], dtype=np.float32)
        var  = np.asarray(sd['obs_rms.var'],  dtype=np.float32)
        print(f'[PPO] obs normalization found in state dict '
              f'(mean[0]={mean[0]:.3f}, var[0]={var[0]:.3f})')
        return mean, var
    return None


def _load_actor_from_state_dict(sd: dict) -> PPOActor:
    """Parse a CleanRL agent or actor state dict into PPOActor."""
    actor = PPOActor()
    # Full agent dict has keys like "actor_mean.0.weight"
    if any(k.startswith('actor_mean.') for k in sd):
        am_sd = {k[len('actor_mean.'):]: v
                 for k, v in sd.items() if k.startswith('actor_mean.')}
        actor.actor_mean.load_state_dict(am_sd)
        if 'actor_logstd' in sd:
            actor.actor_logstd.data.copy_(sd['actor_logstd'])
    # Direct actor_mean dict (keys "0.weight" etc.)
    elif '0.weight' in sd:
        actor.actor_mean.load_state_dict(sd)
    else:
        actor.load_state_dict(sd, strict=False)
    return actor


def _try_load_obs_norm(save_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Try to load obs normalization stats (RunningMeanStd from vecnormalize)."""
    for fname in ['vecnormalize.pkl', 'obs_rms.pkl', 'normalize.pkl']:
        p = save_dir / fname
        if p.exists():
            try:
                with open(p, 'rb') as f:
                    obj = pickle.load(f)
                # gymnasium / stable-baselines3 VecNormalize
                if hasattr(obj, 'obs_rms'):
                    rms = obj.obs_rms
                    print(f'[PPO] obs normalization loaded from {fname}')
                    return rms.mean.astype(np.float32), rms.var.astype(np.float32)
                # dict format
                if isinstance(obj, dict) and 'mean' in obj:
                    return (np.asarray(obj['mean'], np.float32),
                            np.asarray(obj['var'],  np.float32))
            except Exception as e:
                print(f'[PPO] could not parse {fname}: {e}')
    return None


def download_and_load_ppo(
    repo_id:   str = 'sdpkjc/Walker2d-v4-ppo_fix_continuous_action-seed3',
    local_dir: str | None = None,
    device:    str = 'cpu',
) -> PPOPolicy:
    """Download CleanRL Walker2d PPO from HuggingFace and return PPOPolicy.

    CleanRL repos save checkpoints as .cleanrl_model files (torch state dicts).
    We prefer the final checkpoint (no step suffix) over intermediate ones.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    save_dir = Path(local_dir) if local_dir else Path.home() / '.cache' / 'walker2d_ppo'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Discover checkpoint — CleanRL uses .cleanrl_model extension
    ckpt_path = None
    try:
        repo_files = list(list_repo_files(repo_id))
        model_files = [f for f in repo_files
                       if f.endswith('.cleanrl_model') or f.endswith('.pt')
                       or f.endswith('.pth')]
        # Prefer the final checkpoint (no step number in stem)
        def _sort_key(fname):
            stem = Path(fname).stem          # e.g. "ppo_fix_continuous_action"
            try:
                int(stem.rsplit('-', 1)[-1])  # has step number → deprioritise
                return (1, fname)
            except ValueError:
                return (0, fname)            # no step number → final ckpt
        model_files.sort(key=_sort_key)
        for fname in model_files:
            try:
                ckpt_path = hf_hub_download(
                    repo_id=repo_id, filename=fname, local_dir=str(save_dir))
                print(f'[PPO] downloaded {fname} from {repo_id}')
                break
            except Exception:
                pass
    except Exception as e:
        print(f'[PPO] could not list repo files: {e}')

    # Fallback: try common names directly
    if ckpt_path is None:
        for fname in ['ppo_fix_continuous_action.cleanrl_model',
                      'ppo_continuous_action.cleanrl_model',
                      'agent.pt', 'actor.pt', 'model.pt']:
            try:
                ckpt_path = hf_hub_download(
                    repo_id=repo_id, filename=fname, local_dir=str(save_dir))
                print(f'[PPO] downloaded {fname} from {repo_id}')
                break
            except Exception:
                pass

    if ckpt_path is None:
        raise FileNotFoundError(
            f'No checkpoint found in {repo_id}. '
            'Pass --ppo-ckpt with a local checkpoint path instead.')

    # CleanRL .cleanrl_model: either a plain state dict (OrderedDict)
    # with embedded obs_rms.mean/var keys, or a tuple (state_dict, obs_rms).
    obj = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    obs_mean = obs_var = None
    if isinstance(obj, tuple) and len(obj) == 2:
        sd, obs_rms = obj
        if hasattr(obs_rms, 'mean') and hasattr(obs_rms, 'var'):
            obs_mean = np.asarray(obs_rms.mean, dtype=np.float32)
            obs_var  = np.asarray(obs_rms.var,  dtype=np.float32)
    elif isinstance(obj, dict):
        sd = obj
        norm = _extract_obs_rms_from_sd(sd)   # embedded obs_rms.* keys
        if norm:
            obs_mean, obs_var = norm
    else:
        sd = obj.state_dict()

    if not isinstance(sd, dict):
        sd = sd.state_dict()

    actor = _load_actor_from_state_dict(sd)

    # Fallback: try loading normalization from a separate pkl file
    if obs_mean is None:
        norm = _try_load_obs_norm(save_dir)
        if norm:
            obs_mean, obs_var = norm
        else:
            print('[PPO] no obs normalization found — using raw obs')

    return PPOPolicy(actor, obs_mean=obs_mean, obs_var=obs_var, device=device)


def load_ppo_from_local(
    ckpt_path:     str,
    obs_norm_path: str | None = None,
    device:        str = 'cpu',
) -> PPOPolicy:
    """Load PPO from a local checkpoint file."""
    obj = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    obs_mean = obs_var = None
    if isinstance(obj, tuple) and len(obj) == 2:
        sd, obs_rms = obj
        if hasattr(obs_rms, 'mean') and hasattr(obs_rms, 'var'):
            obs_mean = np.asarray(obs_rms.mean, dtype=np.float32)
            obs_var  = np.asarray(obs_rms.var,  dtype=np.float32)
    elif isinstance(obj, dict):
        sd = obj
        norm = _extract_obs_rms_from_sd(sd)
        if norm:
            obs_mean, obs_var = norm
    else:
        sd = obj.state_dict()
    if not isinstance(sd, dict):
        sd = sd.state_dict()
    actor = _load_actor_from_state_dict(sd)

    if obs_norm_path:
        norm = _try_load_obs_norm(Path(obs_norm_path).parent)
        if norm:
            obs_mean, obs_var = norm

    return PPOPolicy(actor, obs_mean=obs_mean, obs_var=obs_var, device=device)


# ── SAC actor ─────────────────────────────────────────────────────────────────

import torch.nn.functional as F


class SACActor(nn.Module):
    """CleanRL sac_continuous_action actor (256-256 MLP, ReLU activations).

    Deterministic evaluation: action = tanh(fc_mean(h)) * action_scale + action_bias.
    For Walker2d-v4 action_scale=1 and action_bias=0, so just tanh(mean).
    """

    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.fc1      = nn.Linear(obs_dim, 256)
        self.fc2      = nn.Linear(256, 256)
        self.fc_mean  = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        self.register_buffer('action_scale', torch.ones(1, action_dim))
        self.register_buffer('action_bias',  torch.zeros(1, action_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic mean action (no sampling)."""
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return torch.tanh(self.fc_mean(h)) * self.action_scale + self.action_bias


class SACPolicy:
    """Deterministic SAC policy: obs → action (deterministic mean, no stochastic sampling)."""

    def __init__(self, actor: SACActor, device: str = 'cpu'):
        self.actor  = actor.to(device).eval()
        self.device = device

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        """obs: (17,) float32 → action: (6,) float32."""
        t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        return self.actor(t)[0].cpu().numpy()


def _load_sac_actor(actor_sd: dict) -> SACActor:
    """Load SACActor from actor state dict (obj[0] in the 3-tuple checkpoint)."""
    actor = SACActor()
    actor.load_state_dict(actor_sd)
    return actor


def download_and_load_sac(
    repo_id:   str = 'sdpkjc/Walker2d-v4-sac_continuous_action-seed4',
    local_dir: str | None = None,
    device:    str = 'cpu',
) -> SACPolicy:
    """Download CleanRL Walker2d SAC from HuggingFace and return SACPolicy."""
    from huggingface_hub import hf_hub_download

    save_dir = Path(local_dir) if local_dir else Path.home() / '.cache' / 'walker2d_sac'
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = hf_hub_download(
        repo_id=repo_id,
        filename='sac_continuous_action.cleanrl_model',
        local_dir=str(save_dir),
    )
    print(f'[SAC] downloaded from {repo_id}')

    obj = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    # checkpoint is (actor_sd, qf1_sd, qf2_sd)
    actor_sd = obj[0]
    actor = _load_sac_actor(actor_sd)
    print(f'[SAC] loaded  action_scale={actor.action_scale[0, 0].item():.3f}  '
          f'action_bias={actor.action_bias[0, 0].item():.3f}')
    return SACPolicy(actor, device=device)


def load_sac_from_local(ckpt_path: str, device: str = 'cpu') -> SACPolicy:
    """Load SAC from a local checkpoint (3-tuple or actor-only dict)."""
    obj = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(obj, tuple):
        actor_sd = obj[0]
    elif isinstance(obj, dict) and 'fc1.weight' in obj:
        actor_sd = obj
    else:
        raise ValueError(f'Unrecognised SAC checkpoint format: {type(obj)}')
    return SACPolicy(_load_sac_actor(actor_sd), device=device)


# ── quick test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--repo-id', default='sdpkjc/Walker2d-v4-ppo_fix_continuous_action-seed3')
    p.add_argument('--ckpt',    default=None, help='Local checkpoint path (skip HF download)')
    p.add_argument('--n-episodes', type=int, default=3)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    if args.ckpt:
        policy = load_ppo_from_local(args.ckpt, device=args.device)
    else:
        policy = download_and_load_ppo(args.repo_id, device=args.device)

    import gymnasium as gym
    env = gym.make('Walker2d-v4')
    total_returns = []
    for ep in range(args.n_episodes):
        obs, _ = env.reset(seed=ep)
        ret, t = 0., 0
        while True:
            a = policy.act(obs.astype(np.float32))
            obs, r, term, trunc, _ = env.step(a)
            ret += r; t += 1
            if term or trunc:
                break
        total_returns.append(ret)
        print(f'  episode {ep}  return={ret:.1f}  length={t}')
    print(f'[PPO] mean return = {np.mean(total_returns):.1f}')
    env.close()
