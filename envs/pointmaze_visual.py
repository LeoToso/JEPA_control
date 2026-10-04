"""Visual PointMaze wrapper — interface mirrors ContinuousCartpoleVisual."""
from __future__ import annotations

import numpy as np


class PointMazeVisual:
    """Wrap gymnasium_robotics PointMaze with pixel observations.

    state  = [x, y, vx, vy]  (4-D, same dimensionality as CartPole)
    action = array([ax, ay]) in [-action_scale, action_scale]  (clipped to [-1,1] before stepping)
    obs    = uint8 RGB image (image_size, image_size, 3)
    """

    def __init__(self, env_cfg: dict, seed: int = 0):
        try:
            import gymnasium
            import gymnasium_robotics  # noqa: F401 — registers envs
        except ImportError as exc:
            raise ImportError(
                'gymnasium-robotics is required: pip install gymnasium-robotics') from exc

        env_section      = env_cfg.get('environment', env_cfg)
        maze_map         = str(env_section.get('maze_map', 'U'))
        self._image_size = int(env_section.get('image_size', 64))
        self.action_scale = float(env_section.get('action_scale', 1.0))

        self._env = gymnasium.make(
            f'PointMaze_{maze_map}Maze-v3',
            render_mode='rgb_array',
            max_episode_steps=None,
        )
        self._rng  = np.random.default_rng(seed)
        self._seed = seed

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

    @staticmethod
    def _parse_state(obs_dict: dict) -> np.ndarray:
        return obs_dict['observation'].astype(np.float32)  # [x, y, vx, vy]

    def _scale_action(self, action) -> np.ndarray:
        a = np.asarray(action, dtype=np.float64).reshape(2)
        return np.clip(a / self.action_scale, -1.0, 1.0)

    # ── public ────────────────────────────────────────────────────────────────

    def reset(self, seed=None, **kwargs):
        s = int(self._rng.integers(0, 2**31)) if seed is None else int(seed)
        obs_dict, info = self._env.reset(seed=s)
        state = self._parse_state(obs_dict)
        obs   = self._render()
        return obs, state.copy(), info

    def reset_to_state(self, state, goal_xy=None):
        """Best-effort reset to [x, y, vx, vy] via direct MuJoCo state set.

        If goal_xy is given, the visual goal marker is moved to that position
        so rendered frames consistently show the correct target location.

        Returns the actual settled position after one physics step so that
        callers can detect wall collisions (agent will be pushed out of walls).
        """
        obs_dict, info = self._env.reset(seed=int(self._rng.integers(0, 2**31)))
        state_out = np.asarray(state[:4], dtype=np.float32)
        try:
            import mujoco
            inner = self._env.unwrapped
            model = inner.model
            data  = inner.data
            data.qpos[:2] = state[:2]
            data.qvel[:2] = [0., 0.]   # zero velocity so step doesn't drift
            if goal_xy is not None:
                inner.goal = np.asarray(goal_xy[:2], dtype=np.float64)
                inner.update_target_site_pos()
            mujoco.mj_forward(model, data)
            # Multiple physics steps to fully resolve wall penetrations
            data.ctrl[:] = 0.
            for _ in range(5):
                mujoco.mj_step(model, data)
            state_out = np.array([data.qpos[0], data.qpos[1],
                                   data.qvel[0], data.qvel[1]], dtype=np.float32)
        except Exception:
            pass
        obs = self._render()
        return obs, state_out, info

    def step(self, action):
        a = self._scale_action(action)
        obs_dict, reward, terminated, truncated, info = self._env.step(a)
        done  = terminated or truncated
        state = self._parse_state(obs_dict)
        obs   = self._render()
        return obs, state.copy(), float(reward), done, info

    def step_no_render(self, action):
        """Physics-only step: same as step() but skips the render.

        Returns (state, reward, done, info) — no obs image.
        Use for intermediate frameskip sub-steps where the image is not needed.
        """
        a = self._scale_action(action)
        obs_dict, reward, terminated, truncated, info = self._env.step(a)
        done  = terminated or truncated
        state = self._parse_state(obs_dict)
        return state.copy(), float(reward), done, info

    def close(self):
        self._env.close()

