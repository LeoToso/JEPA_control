"""Walker2d-v4 visual wrapper — interface mirrors PointMazeVisual."""
from __future__ import annotations

import os
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np


class Walker2dVisual:
    """Walker2d-v4 with pixel observations.

    state  = 17-D proprioceptive obs (qpos[1:9] + qvel[0:9])
    action = 6-D joint torques clipped to [-1, 1]
    obs    = uint8 RGB image (image_size, image_size, 3)
    """

    ACTION_DIM = 6
    STATE_DIM  = 17

    def __init__(self, image_size: int = 64, seed: int = 0):
        import gymnasium
        self._image_size = image_size
        self._env = gymnasium.make('Walker2d-v4', render_mode='rgb_array')
        self._rng = np.random.default_rng(seed)

    # ── private ───────────────────────────────────────────────────────────────

    def _render(self) -> np.ndarray:
        img = self._env.render()  # (H, W, 3) uint8
        s = self._image_size
        if img.shape[0] != s or img.shape[1] != s:
            try:
                from PIL import Image as PilImage
                img = np.asarray(
                    PilImage.fromarray(img).resize((s, s), PilImage.BILINEAR),
                    dtype=np.uint8)
            except ImportError:
                import torch
                t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
                t = torch.nn.functional.interpolate(t, (s, s), mode='bilinear',
                                                    align_corners=False)
                img = t.squeeze(0).permute(1, 2, 0).byte().numpy()
        return np.asarray(img, dtype=np.uint8)

    # ── public ────────────────────────────────────────────────────────────────

    def reset(self, seed=None):
        s = int(self._rng.integers(0, 2**31)) if seed is None else int(seed)
        obs_arr, info = self._env.reset(seed=s)
        obs = self._render()
        return obs, obs_arr.astype(np.float32), info

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        obs_arr, reward, terminated, truncated, info = self._env.step(a)
        done = terminated or truncated
        obs = self._render()
        return obs, obs_arr.astype(np.float32), float(reward), done, info

    def step_no_render(self, action):
        """Physics step without rendering; returns (state, reward, done, info)."""
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        obs_arr, reward, terminated, truncated, info = self._env.step(a)
        done = terminated or truncated
        return obs_arr.astype(np.float32), float(reward), done, info

    def close(self):
        self._env.close()
