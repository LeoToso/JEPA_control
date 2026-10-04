#!/usr/bin/env python
"""Walker2d Poincaré-section utilities shared by GT and learned-model pipelines.

Poincaré section: right hip angle (obs17[2] = qpos[3]) rising through 0.
  - Fires once per gait cycle regardless of shuffling / contact style.
  - Directly computable from gym obs — no MuJoCo contact API needed.
  - Works for both GT rollouts (from env) and latent rollouts (from probe).

State spaces
  - gait state  (16-D): qpos[1:9] + qvel[0:8]  (no global x)
  - gym obs     (17-D): qpos[1:9] + qvel[0:9]
  - latent z    (D,)  : SMWM latent vector
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

# ── gait state helpers ────────────────────────────────────────────────────────

GAIT_DIM = 16   # qpos[1:9] (8) + qvel[0:8] (8)

# Poincaré section: obs17[2] = qpos[3] = right hip angle rising through 0.
# Fires ~once per stride; no contact API needed.
HIP_IDX = 2


def to_gait_state(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    """Full MuJoCo state → 16-D gait state (drops global x, last qvel)."""
    return np.concatenate([qpos[1:9], qvel[0:8]])


def from_gait_state(g16: np.ndarray,
                    x_pos: float = 0.0,
                    last_qvel: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """16-D gait state → (qpos 9-D, qvel 9-D)."""
    g = np.asarray(g16, dtype=np.float64)
    qpos = np.empty(9, dtype=np.float64)
    qpos[0]   = x_pos
    qpos[1:9] = g[:8]
    qvel = np.empty(9, dtype=np.float64)
    qvel[0:8] = g[8:16]
    qvel[8]   = last_qvel
    return qpos, qvel


# ── MuJoCo helper wrapper ─────────────────────────────────────────────────────

class WalkerMuJoCoHelper:
    """Thin wrapper around gymnasium Walker2d-v4 for Poincaré analysis.

    Provides set_state / get_state / step / render_frame.
    Contact-detection methods are kept for optional use but the main
    Poincaré section now uses the hip-angle crossing (no contact API needed).
    """

    FLOOR_GEOMS      = {'floor'}
    RIGHT_FOOT_GEOMS = {'foot', 'foot_geom', 'right_foot'}
    LEFT_FOOT_GEOMS  = {'foot_left', 'foot_left_geom', 'left_foot'}

    def __init__(self, image_size: int = 64, render: bool = False,
                 render_mode: str = 'rgb_array'):
        import gymnasium as gym
        mode = render_mode if render else None
        self.env = gym.make('Walker2d-v4', render_mode=mode)
        self.env.reset(seed=0)
        self._image_size = image_size
        self._render     = render
        self._geom_name_to_id: dict[str, int] | None = None

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        data = self.env.unwrapped.data
        return data.qpos.copy(), data.qvel.copy()

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        self.env.unwrapped.set_state(
            np.asarray(qpos, dtype=np.float64),
            np.asarray(qvel, dtype=np.float64))

    def reset(self, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        self.env.reset(seed=seed)
        return self.get_state()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        obs, rew, term, trunc, _ = self.env.step(
            np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0))
        return obs.astype(np.float32), float(rew), bool(term or trunc)

    def render_frame(self) -> np.ndarray | None:
        if not self._render:
            return None
        frame = self.env.render()
        return np.ascontiguousarray(frame) if frame is not None else None

    def close(self) -> None:
        self.env.close()

    # Optional: contact detection (not used for Poincaré section)
    def _build_geom_map(self) -> None:
        try:
            import mujoco
            model = self.env.unwrapped.model
            self._geom_name_to_id = {
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i): i
                for i in range(model.ngeom)
                if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) is not None
            }
        except Exception:
            self._geom_name_to_id = {}

    def _geom_id_set(self, names: set[str]) -> set[int]:
        if self._geom_name_to_id is None:
            self._build_geom_map()
        return {self._geom_name_to_id[n]
                for n in names if n in self._geom_name_to_id}


# ── GT Poincaré crossing collector ───────────────────────────────────────────

def collect_gt_poincare(
    helper:       WalkerMuJoCoHelper,
    policy,                              # PPOPolicy / SACPolicy
    n_steps:      int = 2000,
    warmup_steps: int = 50,
    seed:         int = 42,
) -> tuple[list[dict], np.ndarray]:
    """Run policy in GT env; record gait state at crossings and at every step.

    Continues across episode boundaries (env truncates at 1 000 steps) so that
    the full n_steps budget is used regardless of episode length.  On a healthy
    truncation the physical state is saved, the step counter is reset, and the
    state is immediately restored — the trajectory is continuous.

    Returns
    -------
    crossings : list[dict]
        {'step', 'gait'(16,), 'qpos'(9,), 'qvel'(9,)} at each right-hip
        zero-crossing (rising through 0).
    all_gaits : np.ndarray  shape (n_steps, 16)
        Gait state at *every* post-warmup step (for distribution comparison).
    """
    from experiments.walker2d_utils import is_healthy_obs

    obs, _ = helper.env.reset(seed=seed)
    obs = obs.astype(np.float32)
    prev_hip = float(obs[HIP_IDX])
    crossings: list[dict] = []
    all_gaits: list[np.ndarray] = []
    n_episodes = 0
    n_unhealthy = 0

    for t in range(n_steps + warmup_steps):
        action = policy.act(obs)
        obs, rew, done = helper.step(action)
        cur_hip = float(obs[HIP_IDX])

        if not is_healthy_obs(obs):
            n_unhealthy += 1
            n_episodes += 1
            obs, _ = helper.env.reset(seed=seed + n_episodes)
            obs = obs.astype(np.float32)
            prev_hip = float(obs[HIP_IDX])
            continue

        if t >= warmup_steps:
            # obs[0:16] = [qpos[1:9], qvel[0:8]] = 16-D gait state
            all_gaits.append(obs[:GAIT_DIM].copy())

        # Rising zero-crossing of right hip angle (after warmup)
        if prev_hip < 0.0 and cur_hip >= 0.0 and t >= warmup_steps:
            qpos, qvel = helper.get_state()
            crossings.append({
                'step': t - warmup_steps,
                'gait': to_gait_state(qpos, qvel),
                'qpos': qpos.copy(),
                'qvel': qvel.copy(),
            })

        prev_hip = cur_hip

        if done:
            n_episodes += 1
            if is_healthy_obs(obs):
                # Truncation — save state, reset step counter, restore.
                qpos_saved, qvel_saved = helper.get_state()
                helper.env.reset(seed=0)
                helper.set_state(qpos_saved, qvel_saved)
            else:
                # Termination (fell) — fresh episode.
                n_unhealthy += 1
                obs, _ = helper.env.reset(seed=seed + n_episodes)
                obs = obs.astype(np.float32)
                prev_hip = float(obs[HIP_IDX])

    all_gaits_arr = (np.stack(all_gaits) if all_gaits
                     else np.empty((0, GAIT_DIM), dtype=np.float32))
    print(f'[GT] {n_episodes} episode resets  {n_unhealthy} terminations  '
          f'hip-crossings={len(crossings)}  dense-samples={len(all_gaits)}')
    return crossings, all_gaits_arr


# ── latent Poincaré rolling ───────────────────────────────────────────────────

def collect_latent_poincare(
    bundle:       dict,
    ridge_probe,                         # RidgeStateProbe  z → gym_obs (17,)
    policy,                              # PPOPolicy / SACPolicy
    z0:           torch.Tensor,          # (1, latent_dim) initial latent
    max_steps:    int = 2000,
    warmup_steps: int = 50,
) -> tuple[list[dict], np.ndarray]:
    """Roll out latent dynamics; detect crossings and collect dense gait states.

    Hip DC offset is estimated from a 100-step calibration pass so that models
    whose decoded hip is biased (never crosses 0) still produce crossings.

    Returns
    -------
    crossings : list[dict]
        {'step', 'z', 'obs', 'gait'} at each rising hip zero-crossing.
    all_gaits : np.ndarray  shape (n_rollout_steps, 16)
        Decoded gait state at *every* post-warmup step (for HALO-style plots).
    """
    from experiments.walker2d_utils import (
        gym_obs_to_mj_state, latent_step, decode_z,
    )

    CALIB_STEPS = 100

    z   = z0.clone()
    obs = decode_z(z, ridge_probe)
    print(f'[latent] init decode: height={obs[0]:.3f}  hip={obs[HIP_IDX]:.3f}')

    # ── calibration: estimate hip DC offset ───────────────────────────────────
    hip_trace: list[float] = [float(obs[HIP_IDX])]
    z_cal = z.clone()
    for _ in range(CALIB_STEPS - 1):
        obs_cal = decode_z(z_cal, ridge_probe)
        hip_trace.append(float(obs_cal[HIP_IDX]))
        z_cal = latent_step(bundle, z_cal, policy.act(obs_cal))
    hip_offset = float(np.mean(hip_trace))
    print(f'[latent] calib hip: min={min(hip_trace):.3f}  '
          f'max={max(hip_trace):.3f}  mean={hip_offset:.3f}  '
          f'(crossing threshold={hip_offset:.3f})')

    # ── main rollout ──────────────────────────────────────────────────────────
    z    = z0.clone()
    obs  = decode_z(z, ridge_probe)
    prev_hip = float(obs[HIP_IDX]) - hip_offset
    crossings: list[dict] = []
    all_gaits: list[np.ndarray] = []
    end_reason = 'max_steps'

    for t in range(max_steps + warmup_steps):
        obs = decode_z(z, ridge_probe)
        cur_hip = float(obs[HIP_IDX]) - hip_offset
        height  = float(obs[0])

        if not (0.3 < height < 4.0):
            end_reason = f'diverged at t={t} height={height:.3f}'
            break

        if t >= warmup_steps:
            all_gaits.append(obs[:GAIT_DIM].copy())

        if prev_hip < 0.0 and cur_hip >= 0.0 and t >= warmup_steps:
            qpos, qvel = gym_obs_to_mj_state(obs, x_pos=0.0)
            crossings.append({
                'step': t - warmup_steps,
                'z':    z.clone(),
                'obs':  obs.copy(),
                'gait': to_gait_state(qpos, qvel),
            })

        prev_hip = cur_hip
        action   = policy.act(obs)
        z        = latent_step(bundle, z, action)

    all_gaits_arr = (np.stack(all_gaits) if all_gaits
                     else np.empty((0, GAIT_DIM), dtype=np.float32))
    print(f'[latent] {end_reason}  hip-crossings={len(crossings)}  '
          f'dense-samples={len(all_gaits)}')
    return crossings, all_gaits_arr


# ── Poincaré map (single-step: given crossing → next crossing) ────────────────

def build_poincare_map_gt(
    helper:    WalkerMuJoCoHelper,
    policy,
    qpos0:     np.ndarray,
    qvel0:     np.ndarray,
    max_steps: int = 300,
) -> np.ndarray | None:
    """From (qpos0, qvel0) run until the next right-hip zero-crossing.

    Returns gait state (16,) at the next crossing, or None if episode ends.
    """
    from experiments.walker2d_utils import is_healthy_obs

    # Reset episode timer so truncation doesn't fire immediately, then
    # override the state to the desired Poincaré point.
    helper.env.reset(seed=0)
    helper.set_state(qpos0, qvel0)
    obs = np.concatenate([qpos0[1:9], qvel0[:9]]).astype(np.float32)
    prev_hip = float(obs[HIP_IDX])

    for _ in range(max_steps):
        action = policy.act(obs)
        obs, _, done = helper.step(action)
        if not is_healthy_obs(obs) or done:
            return None
        cur_hip = float(obs[HIP_IDX])
        if prev_hip < 0.0 and cur_hip >= 0.0:
            qpos, qvel = helper.get_state()
            return to_gait_state(qpos, qvel)
        prev_hip = cur_hip

    return None


def build_poincare_map_latent(
    bundle:      dict,
    ridge_probe,
    policy,
    z0:          torch.Tensor,
    max_steps:   int = 300,
) -> np.ndarray | None:
    """From z0 run until the next right-hip zero-crossing (decoded).

    Returns gait state (16,) at next crossing, or None if episode ends.
    """
    from experiments.walker2d_utils import (
        gym_obs_to_mj_state, is_healthy_obs, latent_step, decode_z,
    )

    z    = z0.clone()
    obs  = decode_z(z, ridge_probe)
    prev_hip = float(obs[HIP_IDX])

    for _ in range(max_steps):
        obs = decode_z(z, ridge_probe)
        if not is_healthy_obs(obs):
            return None
        cur_hip = float(obs[HIP_IDX])
        if prev_hip < 0.0 and cur_hip >= 0.0:
            qpos, qvel = gym_obs_to_mj_state(obs, x_pos=0.0)
            return to_gait_state(qpos, qvel)
        prev_hip = cur_hip
        z = latent_step(bundle, z, policy.act(obs))

    return None


# ── local Jacobian via finite differences ─────────────────────────────────────

def poincare_jacobian_fd(
    map_fn,
    x0:  np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray | None:
    """Central-difference Jacobian of Poincaré map around x0.

    map_fn: x (16,) → x_next (16,) or None.
    Returns (16, 16) Jacobian, or None if any perturbation ended episode.
    """
    n  = len(x0)
    Px = map_fn(x0)
    if Px is None:
        return None

    J = np.zeros((n, n))
    for i in range(n):
        xp = x0.copy(); xp[i] += eps
        xm = x0.copy(); xm[i] -= eps
        Pp = map_fn(xp)
        Pm = map_fn(xm)
        if Pp is None or Pm is None:
            return None
        J[:, i] = (Pp - Pm) / (2 * eps)

    return J


def spectral_radius(J: np.ndarray) -> float:
    """Largest absolute eigenvalue of J."""
    return float(np.max(np.abs(np.linalg.eigvals(J))))
