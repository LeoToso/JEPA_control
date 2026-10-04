#!/usr/bin/env python
"""CEM planning in DINO-WM latent space for PointMaze.

Same seed/protocol as compare_cem_dinowm_pointmaze.py (JEPA models) but loads
the original DINO-WM checkpoint (encoder + predictor + proprio + action encoder).

Usage
-----
  MUJOCO_GL=egl python experiments/cem_dinowm_pointmaze.py \\
      --dino-wm-dir ~/dino_wm \\
      --ckpt /path/to/model_latest.pth \\
      --trials 10 --n-steps 200 \\
      --planning-horizon 25 --executed-steps 25 \\
      --cem-population 300 --cem-elites 30 --cem-iters 10 \\
      --output results/cem_dinowm_wm_pointmaze.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

# ── DINO-WM constants ─────────────────────────────────────────────────────────
FRAMESKIP   = 5
D_VIS       = 384
D_PROP_OUT  = 10
D_ACT_OUT   = 10
D_TOTAL     = D_VIS + D_PROP_OUT + D_ACT_OUT   # 404
ACT_IN_DIM  = FRAMESKIP * 2                     # 10
PROP_IN_DIM = 4


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup_dinowm(dino_wm_dir: str):
    p = str(Path(dino_wm_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Environment ───────────────────────────────────────────────────────────────

def make_env(seed: int = 0, image_size: int = 196):
    from envs.pointmaze_visual import PointMazeVisual
    env_cfg = {'environment': {'maze_map': 'U', 'image_size': image_size,
                               'action_scale': 1.0}}
    return PointMazeVisual(env_cfg, seed=seed)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, dino_wm_dir: str, device: torch.device) -> dict:
    setup_dinowm(dino_wm_dir)
    from models.dino import DinoV2Encoder
    print('[DINO-WM CEM] loading checkpoint …')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    parts = {}
    for k in ['predictor', 'proprio_encoder', 'action_encoder']:
        if k not in ckpt:
            raise KeyError(f"'{k}' missing — available: {list(ckpt.keys())}")
        parts[k] = ckpt[k].to(device).eval()
    encoder = DinoV2Encoder(name='dinov2_vits14', feature_key='x_norm_patchtokens')
    parts['encoder'] = encoder.to(device).eval()
    n_pred = sum(p.numel() for p in parts['predictor'].parameters())
    print(f'[DINO-WM CEM] predictor params={n_pred/1e6:.1f}M')
    return parts


# ── Observation encoding ──────────────────────────────────────────────────────

_NORM_MEAN = torch.tensor([0.5, 0.5, 0.5])
_NORM_STD  = torch.tensor([0.5, 0.5, 0.5])


def preprocess_obs(obs_hwc: np.ndarray, device: torch.device,
                   target_size: int = 196) -> torch.Tensor:
    x = torch.from_numpy(obs_hwc.copy()).float().permute(2, 0, 1) / 255.0
    if x.shape[-1] != target_size or x.shape[-2] != target_size:
        x = F.interpolate(x.unsqueeze(0), size=(target_size, target_size),
                          mode='bilinear', align_corners=False).squeeze(0)
    x = (x - _NORM_MEAN[:, None, None]) / _NORM_STD[:, None, None]
    return x.unsqueeze(0).to(device)


def encode_frame(parts: dict, obs_hwc: np.ndarray, state: np.ndarray,
                 action_10d: np.ndarray | None,
                 device: torch.device) -> torch.Tensor:
    """(obs, state, action) → (1, N_patches, D_TOTAL)."""
    if action_10d is None:
        action_10d = np.zeros(ACT_IN_DIM, dtype=np.float32)
    with torch.no_grad():
        vis_emb = parts['encoder'](preprocess_obs(obs_hwc, device))  # (1, N, 384)
    N = vis_emb.shape[1]
    _pe_dtype = next(parts['proprio_encoder'].parameters()).dtype
    prop_in = torch.as_tensor(state[:PROP_IN_DIM].astype(np.float32),
                               device=device).to(_pe_dtype).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        prop_emb = parts['proprio_encoder'](prop_in)
    prop_tiled = prop_emb[:, 0:1, :].expand(-1, N, -1)
    _ae_dtype = next(parts['action_encoder'].parameters()).dtype
    act_in = torch.as_tensor(action_10d[:ACT_IN_DIM].astype(np.float32),
                               device=device).to(_ae_dtype).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        act_emb = parts['action_encoder'](act_in)
    act_tiled = act_emb[:, 0:1, :].expand(-1, N, -1)
    return torch.cat([vis_emb, prop_tiled, act_tiled], dim=2)  # (1, N, D_TOTAL)


def build_goal_context(parts: dict, env, goal_xy: np.ndarray,
                       num_hist: int, device: torch.device):
    """Return (z_goal_ctx, z_goal_mean) for the given goal position."""
    goal_state = np.array([goal_xy[0], goal_xy[1], 0., 0.], dtype=np.float32)
    goal_obs, _, _ = env.reset_to_state(goal_state)
    zero_act = np.zeros(ACT_IN_DIM, dtype=np.float32)
    goal_frame  = encode_frame(parts, goal_obs, goal_state, zero_act, device)
    z_goal_ctx  = goal_frame.unsqueeze(1).expand(-1, num_hist, -1, -1).clone()
    z_goal_mean = z_goal_ctx[0, -1, :, :].mean(dim=0)   # (D_TOTAL,)
    return z_goal_ctx, z_goal_mean


# ── CEM planner ───────────────────────────────────────────────────────────────

class DinoWMPointMazeCEM:
    """Cross-entropy method planner in DINO-WM mean-pooled latent space.

    For each CEM iteration:
      1. Sample `population` action sequences of length `horizon` (10-D per step).
      2. Rollout all sequences in parallel through the predictor.
      3. Score by L2 distance of the final mean-pooled latent to z_goal_mean.
      4. Fit a new Gaussian to the top-`elites` sequences.
    Returns the first `executed_steps` actions of the best sequence.
    """

    def __init__(self, parts: dict, num_hist: int,
                 horizon: int, executed_steps: int,
                 population: int, elites: int, iterations: int,
                 device: torch.device, use_half: bool = False,
                 rollout_chunk: int = 50):
        self.parts          = parts
        self.num_hist       = num_hist
        self.horizon        = horizon
        self.executed_steps = executed_steps
        self.population     = population
        self.elites         = elites
        self.iterations     = iterations
        self.device         = device
        self.use_half       = use_half
        self.rollout_chunk  = rollout_chunk  # max samples per GPU call

    @torch.no_grad()
    def _rollout_chunk(self, z_ctx: torch.Tensor,
                       actions_batch: torch.Tensor) -> torch.Tensor:
        """Rollout a single chunk of trajectories.

        z_ctx         : (1, T, N, D)
        actions_batch : (B, H, ACT_IN_DIM)  B <= rollout_chunk
        Returns       : (B, D) mean-pooled final latents
        """
        B, H, _ = actions_batch.shape
        _, T, N, D = z_ctx.shape

        predictor      = self.parts['predictor']
        action_encoder = self.parts['action_encoder']

        z_batch = z_ctx.expand(B, -1, -1, -1).clone()   # (B, T, N, D)

        for h in range(H):
            acts      = actions_batch[:, h, :]              # (B, ACT_IN_DIM)
            act_emb   = action_encoder(acts.unsqueeze(1))   # (B, 1, 10)
            act_tiled = act_emb[:, 0:1, :].expand(-1, N, -1)  # (B, N, 10)

            z_in = z_batch.clone()
            z_in[:, -1, :, D_VIS + D_PROP_OUT:] = act_tiled.to(z_in.dtype)

            z_flat = z_in.reshape(B, T * N, D)
            z_pred = predictor(z_flat).reshape(B, T, N, D)

            z_batch = torch.cat([z_batch[:, 1:], z_pred[:, -1:]], dim=1)

        return z_batch[:, -1, :, :].mean(dim=1)   # (B, D)

    @torch.no_grad()
    def _rollout_batch(self, z_ctx: torch.Tensor,
                       actions_batch: torch.Tensor) -> torch.Tensor:
        """Rollout P trajectories in mini-batches to avoid OOM.

        z_ctx         : (1, T, N, D)
        actions_batch : (P, H, ACT_IN_DIM)
        Returns       : (P, D) mean-pooled final latents
        """
        P = actions_batch.shape[0]
        chunks = []
        for start in range(0, P, self.rollout_chunk):
            end = min(start + self.rollout_chunk, P)
            chunks.append(self._rollout_chunk(z_ctx, actions_batch[start:end]))
        return torch.cat(chunks, dim=0)   # (P, D)

    def plan(self, z_ctx: torch.Tensor,
             z_goal_mean: torch.Tensor) -> np.ndarray:
        """Return (executed_steps, ACT_IN_DIM) actions to execute next."""
        P, H, K = self.population, self.horizon, self.elites
        dtype    = z_ctx.dtype

        mu    = torch.zeros(H, ACT_IN_DIM, device=self.device, dtype=dtype)
        sigma = torch.ones (H, ACT_IN_DIM, device=self.device, dtype=dtype)
        z_goal = z_goal_mean.to(dtype)

        for _ in range(self.iterations):
            eps     = torch.randn(P, H, ACT_IN_DIM, device=self.device, dtype=dtype)
            actions = (mu.unsqueeze(0) + sigma.unsqueeze(0) * eps).clamp(-1., 1.)

            with torch.autocast(device_type=self.device.type,
                                enabled=self.use_half and self.device.type == 'cuda'):
                z_finals = self._rollout_batch(z_ctx, actions)   # (P, D)

            # visual-only objective: matches DINO-WM paper's alpha=0 setting
            dists     = torch.linalg.vector_norm(
                z_finals[:, :D_VIS] - z_goal[:D_VIS].unsqueeze(0), dim=1)  # (P,)
            elite_idx = torch.argsort(dists)[:K]
            elites    = actions[elite_idx]                        # (K, H, ACT_IN_DIM)

            mu    = elites.mean(dim=0)
            sigma = elites.std(dim=0).clamp(min=1e-6)

        return mu[:self.executed_steps].cpu().numpy()   # (executed_steps, ACT_IN_DIM)


# ── Trial loop ────────────────────────────────────────────────────────────────

def cem_trial(parts: dict, planner: DinoWMPointMazeCEM,
              env, z_goal_mean: torch.Tensor,
              goal_xy: np.ndarray, num_hist: int,
              n_steps: int, success_threshold: float, success_hold_steps: int,
              device: torch.device,
              init_state: np.ndarray | None = None) -> dict:
    if init_state is not None:
        obs, state, _ = env.reset_to_state(init_state.copy())
    else:
        obs, state, _ = env.reset()
    zero_act = np.zeros(ACT_IN_DIM, dtype=np.float32)

    # Bootstrap context with num_hist copies of the initial frame
    init_frame = encode_frame(parts, obs, state, zero_act, device)
    z_ctx = init_frame.unsqueeze(1).expand(-1, num_hist, -1, -1).clone()

    z_goal_gpu    = z_goal_mean.to(device)
    states        = [state.copy()]
    consec_stable = 0
    model_step    = 0
    done          = False

    while model_step < n_steps and not done:
        plan_acts = planner.plan(z_ctx, z_goal_gpu)   # (executed_steps, ACT_IN_DIM)

        for e in range(len(plan_acts)):
            if model_step >= n_steps or done:
                break
            u_10d = np.clip(plan_acts[e], -1., 1.)

            # Execute FRAMESKIP env steps; record every sub-step state
            obs_after, state_after = obs, state
            for sub in range(FRAMESKIP):
                a_2d = u_10d[sub * 2:(sub + 1) * 2]
                if sub == FRAMESKIP - 1:
                    obs_after, state_after, _, _done, _ = env.step(a_2d)
                    done = _done
                else:
                    state_after, _, _done, _ = env.step_no_render(a_2d)
                    done = _done
                states.append(state_after.copy())

                xy_dist = float(np.linalg.norm(state_after[:2] - goal_xy))
                if xy_dist < success_threshold:
                    consec_stable += 1
                    if consec_stable >= success_hold_steps:
                        done = True
                        break
                else:
                    consec_stable = 0
                if done:
                    break

            # Update rolling context
            new_frame = encode_frame(parts, obs_after, state_after, u_10d, device)
            z_ctx = torch.cat([z_ctx[:, 1:], new_frame.unsqueeze(1)], dim=1)
            obs, state = obs_after, state_after
            model_step += 1

    states  = np.array(states)
    xy_err  = np.linalg.norm(states[:, :2] - goal_xy[None], axis=1)
    stable  = xy_err < success_threshold
    tail    = stable[-success_hold_steps:]
    held    = bool(len(tail) == success_hold_steps and tail.all())
    success = bool(stable[-1])

    return {
        'success':         success,
        'held':            held,
        'final_error':     float(xy_err[-1]),
        'max_error':       float(np.max(xy_err)),
        'fraction_stable': float(np.mean(stable[1:])) if len(stable) > 1 else 0.,
        'goal_xy':         goal_xy.tolist(),
        'start_xy':        states[0, :2].tolist(),
        'states':          states.tolist(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='DINO-WM CEM for PointMaze — same seed scheme as JEPA CEM scripts.')
    p.add_argument('--dino-wm-dir',        required=True,
                   help='Path to cloned gaoyuezhou/dino_wm repo')
    p.add_argument('--ckpt',               required=True,
                   help='DINO-WM checkpoint (model_latest.pth)')
    p.add_argument('--trials',             type=int,   default=10)
    p.add_argument('--n-steps',            type=int,   default=200,
                   help='Max model steps per trial')
    p.add_argument('--num-hist',           type=int,   default=3)
    p.add_argument('--planning-horizon',   type=int,   default=25)
    p.add_argument('--executed-steps',     type=int,   default=25)
    p.add_argument('--cem-population',     type=int,   default=300)
    p.add_argument('--cem-elites',         type=int,   default=30)
    p.add_argument('--cem-iters',          type=int,   default=10)
    p.add_argument('--success-threshold',  type=float, default=0.5)
    p.add_argument('--success-hold-steps', type=int,   default=1)
    p.add_argument('--seed',               type=int,   default=0,
                   help='Base seed: trial i uses start=seed+99*i+1, goal=seed+99*i+2')
    p.add_argument('--device',             default='cuda')
    p.add_argument('--half',               action='store_true',
                   help='FP16 autocast for model inference (~1.5-2x faster)')
    p.add_argument('--rollout-chunk',      type=int,   default=50,
                   help='Max population samples per GPU call (reduce if OOM)')
    p.add_argument('--output',             required=True)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'[DINO-WM CEM] device={device}')
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device.index or 0)
        print(f'[DINO-WM CEM] GPU: {props.name}  VRAM: {props.total_memory/1e9:.1f} GB')

    parts = load_model(args.ckpt, args.dino_wm_dir, device)
    env   = make_env(seed=args.seed, image_size=196)

    planner = DinoWMPointMazeCEM(
        parts=parts,
        num_hist=args.num_hist,
        horizon=args.planning_horizon,
        executed_steps=args.executed_steps,
        population=args.cem_population,
        elites=args.cem_elites,
        iterations=args.cem_iters,
        device=device,
        use_half=args.half,
        rollout_chunk=args.rollout_chunk,
    )
    print(f'[DINO-WM CEM] H={planner.horizon} E={planner.executed_steps} '
          f'pop={planner.population} elites={planner.elites} '
          f'iters={planner.iterations}  half={args.half}')
    print(f'[DINO-WM CEM] seed scheme: start=seed+99*i+1, goal=seed+99*i+2')

    trials, n_success = [], 0

    for i in range(args.trials):
        start_seed = args.seed + 99 * i + 1
        goal_seed  = args.seed + 99 * i + 2

        obs_dict, _ = env._env.reset(seed=int(start_seed))
        start_state = obs_dict['observation'][:4].astype(np.float32)

        obs_dict, _ = env._env.reset(seed=int(goal_seed))
        goal_xy     = obs_dict['observation'][:2].astype(np.float32)

        print(f'[DINO-WM CEM] trial {i:03d}  '
              f'start=[{start_state[0]:.2f},{start_state[1]:.2f}]  '
              f'goal=[{goal_xy[0]:.2f},{goal_xy[1]:.2f}]  '
              f'building goal context …', end=' ', flush=True)

        _, z_goal_mean = build_goal_context(parts, env, goal_xy,
                                            args.num_hist, device)

        t0  = time.time()
        row = cem_trial(
            parts=parts, planner=planner, env=env,
            z_goal_mean=z_goal_mean, goal_xy=goal_xy,
            num_hist=args.num_hist,
            n_steps=args.n_steps,
            success_threshold=args.success_threshold,
            success_hold_steps=args.success_hold_steps,
            device=device, init_state=start_state,
        )
        t1 = time.time()
        n_success += int(row['success'])
        print(f'success={row["success"]}  err={row["final_error"]:.4f}  '
              f't={t1-t0:.1f}s')
        trials.append(row)

    sr = n_success / max(len(trials), 1)
    print(f'\n[DINO-WM CEM] success rate: {sr:.1%}  ({n_success}/{len(trials)})')
    env.close()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump({
            'protocol': vars(args),
            'models': [{
                'label':        'DINO-WM CEM',
                'success_rate': sr,
                'n_success':    n_success,
                'n_trials':     len(trials),
                'trials':       trials,
            }],
        }, f, indent=2)
    print(f'[done] {out}')


if __name__ == '__main__':
    main()

