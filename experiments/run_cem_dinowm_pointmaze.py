#!/usr/bin/env python
"""CEM evaluation for PointMaze using the original DINO-WM checkpoint.

Outputs the same {"models": [...]} JSON schema as compare_cem_dinowm_pointmaze.py
so the result can be passed directly to plot_lqr_trajectories_pointmaze.py.

DINO-WM evaluation protocol (arXiv 2411.04983):
  - Per-trial independent random start AND goal
  - CEM horizon = executed steps = 5 (replans every 5 steps)
  - Population=300, elites=30, iterations=10, initial variance=1.0
  - Success threshold = 0.5 maze units
  - 50 trials, seeds: start=99*n+1, goal=99*n+2

Usage
-----
  MUJOCO_GL=egl python experiments/run_cem_dinowm_pointmaze.py \\
      --dino-wm-dir ~/dino_wm \\
      --ckpt /mnt/t7shield/jepa_results/dinowm_checkpoints/model_latest.pth \\
      --trials 50 --n-steps 200 \\
      --output results/cem_dinowm_pointmaze.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')

from envs.pointmaze_visual import PointMazeVisual


# ── DINO-WM constants (must match training) ───────────────────────────────────
FRAMESKIP   = 5
D_VIS       = 384
D_PROP_OUT  = 10
D_ACT_OUT   = 10
D_TOTAL     = D_VIS + D_PROP_OUT + D_ACT_OUT   # 404
ACT_IN_DIM  = FRAMESKIP * 2                     # 10
PROP_IN_DIM = 4

_NORM_MEAN = torch.tensor([0.5, 0.5, 0.5])
_NORM_STD  = torch.tensor([0.5, 0.5, 0.5])


# ── model loading ─────────────────────────────────────────────────────────────

def setup_dinowm(dino_wm_dir: str):
    p = str(Path(dino_wm_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def load_model(ckpt_path: str, dino_wm_dir: str, device: torch.device) -> dict:
    setup_dinowm(dino_wm_dir)
    from models.dino import DinoV2Encoder
    print('[DINO-WM] loading checkpoint …')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    parts = {}
    for k in ['predictor', 'proprio_encoder', 'action_encoder']:
        if k not in ckpt:
            raise KeyError(f"'{k}' missing — keys: {list(ckpt.keys())}")
        parts[k] = ckpt[k].to(device).eval()
    encoder = DinoV2Encoder(name='dinov2_vits14', feature_key='x_norm_patchtokens')
    parts['encoder'] = encoder.to(device).eval()
    n = sum(p.numel() for p in parts['predictor'].parameters())
    print(f'[DINO-WM] predictor params={n/1e6:.1f}M')
    # Probe predictor output dim once on dummy input
    try:
        with torch.no_grad():
            dummy = torch.zeros(1, 196, D_TOTAL, device=device)
            out   = parts['predictor'](dummy)
            parts['_pred_out_dim'] = out.shape[-1]
            print(f'[DINO-WM] predictor output dim={out.shape[-1]}')
    except Exception as e:
        print(f'[DINO-WM] predictor probe failed ({e}); assuming output dim={D_TOTAL}')
        parts['_pred_out_dim'] = D_TOTAL
    return parts


# ── encoding helpers ──────────────────────────────────────────────────────────

def preprocess_obs(obs_hwc: np.ndarray, device: torch.device,
                   target_size: int = 196) -> torch.Tensor:
    x = torch.from_numpy(obs_hwc.copy()).float().permute(2, 0, 1) / 255.0
    if x.shape[-2] != target_size or x.shape[-1] != target_size:
        x = F.interpolate(x.unsqueeze(0), size=(target_size, target_size),
                          mode='bilinear', align_corners=False).squeeze(0)
    x = (x - _NORM_MEAN[:, None, None]) / _NORM_STD[:, None, None]
    return x.unsqueeze(0).to(device)


@torch.no_grad()
def encode_tokens(parts: dict, obs_hwc: np.ndarray, state: np.ndarray,
                  action_vec: np.ndarray, device: torch.device) -> torch.Tensor:
    """Encode (obs, state, action) → token tensor (1, N, D_TOTAL)."""
    vis = parts['encoder'](preprocess_obs(obs_hwc, device))   # (1, N, 384)
    N   = vis.shape[1]

    dtype_pe = next(parts['proprio_encoder'].parameters()).dtype
    prop_in  = torch.as_tensor(
        state[:PROP_IN_DIM].astype(np.float32), device=device
    ).to(dtype_pe).unsqueeze(0).unsqueeze(0)
    prop_emb = parts['proprio_encoder'](prop_in)               # (1, 1, D_PROP_OUT)
    prop_t   = prop_emb[:, 0:1, :].expand(-1, N, -1)

    dtype_ae = next(parts['action_encoder'].parameters()).dtype
    # Repeat action across FRAMESKIP sub-steps → (1, 1, ACT_IN_DIM)
    act_flat = np.tile(action_vec.astype(np.float32), FRAMESKIP)[None, None, :]
    act_in   = torch.as_tensor(act_flat, device=device, dtype=dtype_ae)
    act_emb  = parts['action_encoder'](act_in)                # (1, 1, D_ACT_OUT)
    act_t    = act_emb[:, 0:1, :].expand(-1, N, -1)

    return torch.cat([vis, prop_t, act_t], dim=2)             # (1, N, D_TOTAL)


@torch.no_grad()
def encode_mean(parts: dict, obs_hwc: np.ndarray,
                state: np.ndarray, device: torch.device) -> torch.Tensor:
    """Encode (obs, state) with zero action → mean-pooled latent (D_TOTAL,)."""
    zero_act = np.zeros(2, dtype=np.float32)
    tokens   = encode_tokens(parts, obs_hwc, state, zero_act, device)
    return tokens[0].mean(dim=0)                              # (D_TOTAL,)


@torch.no_grad()
def predict_next(parts: dict, tokens: torch.Tensor,
                 action_vec: np.ndarray, device: torch.device) -> torch.Tensor:
    """Run one prediction step.

    tokens : (B, N, D_TOTAL) — current state tokens with any action embedded
    action_vec : (2,) — next action in raw units

    Returns mean-pooled predicted latent: (B, D_TOTAL).

    The predictor is called with current tokens.  Depending on the DINO-WM
    variant, it may output (B, N, D_TOTAL) or (B, N, D_VIS).  We handle both
    by zero-padding to D_TOTAL when needed.
    """
    B, N, _ = tokens.shape

    # Embed the *next* action into a fresh action token for the predictor input
    dtype_ae = next(parts['action_encoder'].parameters()).dtype
    act_flat = np.tile(action_vec.astype(np.float32), FRAMESKIP)[None, None, :]
    act_in   = torch.as_tensor(act_flat, device=device, dtype=dtype_ae)
    act_emb  = parts['action_encoder'](act_in)              # (1, 1, D_ACT_OUT)
    act_t    = act_emb.expand(B, N, -1)

    # Replace the action slice in the current tokens with the new action
    tokens_in = tokens.clone().float()
    tokens_in[:, :, D_VIS + D_PROP_OUT:] = act_t.float()

    pred = parts['predictor'](tokens_in)                     # (B, N, D_out)

    out_dim = pred.shape[-1]
    if out_dim < D_TOTAL:
        # Predictor output is visual-only (D_VIS); pad with zeros for the rest
        pad = torch.zeros(B, N, D_TOTAL - out_dim, device=device, dtype=pred.dtype)
        pred = torch.cat([pred, pad], dim=-1)

    return pred.mean(dim=1)                                  # (B, D_TOTAL)


# ── CEM planner ───────────────────────────────────────────────────────────────

class DinoWMCEM:
    def __init__(self, parts, horizon, executed_steps,
                 population, elites, iterations, initial_variance,
                 action_lb, action_ub, visual_only, device):
        self.parts           = parts
        self.horizon         = int(horizon)
        self.executed_steps  = min(int(executed_steps), self.horizon)
        self.population      = int(population)
        self.elites          = min(int(elites), self.population)
        self.iterations      = int(iterations)
        self.initial_var     = float(initial_variance)
        self.action_lb       = float(action_lb)
        self.action_ub       = float(action_ub)
        self.visual_only     = visual_only
        self.device          = device

    @torch.no_grad()
    def _rollout_cost(self, z0_tokens: torch.Tensor,
                      actions: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """
        z0_tokens : (1, N, D_TOTAL)
        actions   : (pop, H, 2)
        z_goal    : (D_TOTAL,) or (D_VIS,)
        Returns   : (pop,) cost tensor
        """
        pop = actions.shape[0]
        N   = z0_tokens.shape[1]

        # Expand initial tokens to population size
        tokens = z0_tokens.expand(pop, -1, -1).clone()       # (pop, N, D_TOTAL)

        for t in range(self.horizon):
            # Embed action for this step
            dtype_ae = next(self.parts['action_encoder'].parameters()).dtype
            act_np   = actions[:, t, :].cpu().numpy()         # (pop, 2)
            act_flat = np.tile(act_np[:, :, None], (1, 1, FRAMESKIP)
                               ).reshape(pop, 1, ACT_IN_DIM)  # (pop, 1, 10)
            act_in   = torch.as_tensor(act_flat, device=self.device, dtype=dtype_ae)
            act_emb  = self.parts['action_encoder'](act_in)   # (pop, 1, D_ACT_OUT)
            act_t    = act_emb.expand(-1, N, -1)

            tokens_in = tokens.clone().float()
            tokens_in[:, :, D_VIS + D_PROP_OUT:] = act_t.float()

            pred = self.parts['predictor'](tokens_in)         # (pop, N, D_out)
            out_dim = pred.shape[-1]
            if out_dim < D_TOTAL:
                pad    = torch.zeros(pop, N, D_TOTAL - out_dim,
                                     device=self.device, dtype=pred.dtype)
                pred   = torch.cat([pred, pad], dim=-1)

            tokens = pred                                      # update for next step

        z_final = tokens.mean(dim=1)                          # (pop, D_TOTAL)

        if self.visual_only:
            diff = z_final[:, :D_VIS] - z_goal[:D_VIS]
        else:
            diff = z_final - z_goal[:D_TOTAL]
        return diff.square().sum(-1)                          # (pop,)

    @torch.no_grad()
    def plan(self, z0_tokens: torch.Tensor,
             z_goal: torch.Tensor) -> np.ndarray:
        """Return (executed_steps, 2) action array."""
        mean = torch.zeros(self.horizon, 2, device=self.device)
        var  = torch.full((self.horizon, 2), self.initial_var, device=self.device)

        for _ in range(self.iterations):
            noise   = torch.randn(self.population, self.horizon, 2, device=self.device)
            actions = (mean.unsqueeze(0) + var.sqrt().unsqueeze(0) * noise
                       ).clamp(self.action_lb, self.action_ub)
            costs   = self._rollout_cost(z0_tokens, actions, z_goal)
            elite   = actions[torch.argsort(costs)[:self.elites]]
            mean    = elite.mean(0)
            var     = elite.var(0, unbiased=False).clamp(min=1e-6)

        return mean[:self.executed_steps].clamp(
            self.action_lb, self.action_ub).cpu().numpy()


# ── trial evaluation ──────────────────────────────────────────────────────────

def sample_random_state(env, seed):
    obs_dict, _ = env._env.reset(seed=int(seed))
    return obs_dict['observation'].astype(np.float32)


def cem_trial(parts, env, planner, z_goal, z_goal_tokens, goal_xy,
              start_state, n_steps, success_threshold, success_hold_steps,
              device, frame_skip: int = 1):
    obs, state, _ = env.reset_to_state(start_state)
    states         = [state.copy()]

    step          = 0
    consec_stable = 0
    prev_obs      = obs.copy()

    while step < n_steps:
        with torch.no_grad():
            zero_act = np.zeros(2, dtype=np.float32)
            z0_tokens = encode_tokens(parts, obs, state, zero_act, device)

        sequence = planner.plan(z0_tokens, z_goal)

        for a in sequence:
            if step >= n_steps:
                break
            a_clipped = np.clip(a, planner.action_lb, planner.action_ub)
            prev_obs = obs.copy()
            done = False
            for sub in range(frame_skip):
                if sub < frame_skip - 1:
                    state, _, done, _ = env.step_no_render(a_clipped)
                else:
                    obs, state, _, done, _ = env.step(a_clipped)
                if done:
                    break
            states.append(state.copy())
            step += 1

            xy_dist = float(np.linalg.norm(state[:2] - goal_xy))
            if xy_dist < success_threshold:
                consec_stable += 1
                if consec_stable >= success_hold_steps:
                    break
            else:
                consec_stable = 0

            if done:
                break

        if consec_stable >= success_hold_steps:
            break

    states    = np.array(states)
    xy_err    = np.linalg.norm(states[:, :2] - goal_xy[None], axis=1)
    stable    = xy_err < success_threshold
    success   = bool(stable[-1])
    tail      = stable[-success_hold_steps:]
    held      = bool(len(tail) == success_hold_steps and tail.all())

    return {
        'success':         success,
        'held':            held,
        'final_error':     float(xy_err[-1]),
        'max_error':       float(np.max(xy_err)),
        'fraction_stable': float(np.mean(stable[1:])) if len(stable) > 1 else 0.,
        'goal_xy':         goal_xy.tolist(),
        'start_xy':        start_state[:2].tolist(),
        'states':          states.tolist(),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='CEM evaluation for PointMaze using the original DINO-WM checkpoint.')
    p.add_argument('--dino-wm-dir',  required=True,
                   help='Path to cloned gaoyuezhou/dino_wm repo')
    p.add_argument('--ckpt',         required=True,
                   help='DINO-WM checkpoint path (model_latest.pth)')
    p.add_argument('--trials',       type=int,   default=50)
    p.add_argument('--n-steps',      type=int,   default=200,
                   help='Max env steps per trial')
    p.add_argument('--planning-horizon',   type=int,   default=5)
    p.add_argument('--executed-steps',     type=int,   default=5)
    p.add_argument('--cem-population',     type=int,   default=300)
    p.add_argument('--cem-elites',         type=int,   default=30)
    p.add_argument('--cem-iters',          type=int,   default=10)
    p.add_argument('--cem-initial-variance', type=float, default=1.0)
    p.add_argument('--success-threshold',  type=float, default=0.5)
    p.add_argument('--success-hold-steps', type=int,   default=1)
    p.add_argument('--seed',               type=int,   default=0)
    p.add_argument('--frame-skip',          type=int,   default=1,
                   help='Execute each planned action this many times in the env. '
                        'Set 5 to match DINO-WM dynamics depth (each action = 5 mj_steps).')
    p.add_argument('--action-scale',       type=float, default=1.0,
                   help='Action magnitude limit (clip ±action_scale).')
    p.add_argument('--visual-only',        action='store_true', default=True,
                   help='Use only first D_VIS=384 dims for cost (default True).')
    p.add_argument('--image-size',         type=int,   default=196)
    p.add_argument('--maze-map',           default='U')
    p.add_argument('--device',             default='cuda')
    p.add_argument('--label',              default='DINO-WM',
                   help='Model label in output JSON.')
    p.add_argument('--output',             required=True)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    parts  = load_model(args.ckpt, args.dino_wm_dir, device)

    env_cfg = {'environment': {'maze_map': args.maze_map,
                               'image_size': args.image_size,
                               'action_scale': args.action_scale}}
    env = PointMazeVisual(env_cfg, seed=0)

    planner = DinoWMCEM(
        parts           = parts,
        horizon         = args.planning_horizon,
        executed_steps  = args.executed_steps,
        population      = args.cem_population,
        elites          = args.cem_elites,
        iterations      = args.cem_iters,
        initial_variance= args.cem_initial_variance,
        action_lb       = -args.action_scale,
        action_ub       =  args.action_scale,
        visual_only     = args.visual_only,
        device          = device,
    )
    print(f'[{args.label}] H={planner.horizon} K={planner.executed_steps} '
          f'pop={planner.population} elites={planner.elites} '
          f'iters={planner.iterations} frame_skip={args.frame_skip} '
          f'visual_only={planner.visual_only}')

    trials, n_success = [], 0
    for i in range(args.trials):
        start_seed = args.seed + 99 * i + 1
        goal_seed  = args.seed + 99 * i + 2

        start_state = sample_random_state(env, start_seed)
        goal_raw    = sample_random_state(env, goal_seed)
        goal_xy     = goal_raw[:2]

        # Encode goal at zero velocity
        goal_state = np.array([goal_xy[0], goal_xy[1], 0., 0.], dtype=np.float32)
        obs_g, st_g, _ = env.reset_to_state(goal_state, goal_xy=goal_xy)
        z_goal        = encode_mean(parts, obs_g, st_g, device)
        zero_act      = np.zeros(2, dtype=np.float32)
        z_goal_tokens = encode_tokens(parts, obs_g, st_g, zero_act, device)

        row = cem_trial(parts, env, planner, z_goal, z_goal_tokens, goal_xy,
                        start_state, args.n_steps,
                        args.success_threshold, args.success_hold_steps,
                        device, frame_skip=args.frame_skip)
        n_success += int(row['success'])
        print(f'[{args.label}] trial {i:03d}  success={row["success"]}  '
              f'err={row["final_error"]:.4f}  '
              f'start={np.round(start_state[:2], 2)}  '
              f'goal={np.round(goal_xy, 2)}')
        trials.append(row)

    sr = n_success / max(len(trials), 1)
    print(f'[{args.label}] success rate: {sr:.1%}')
    env.close()

    results = {
        'protocol': vars(args),
        'models': [{
            'label':        args.label,
            'success_rate': sr,
            'trials':       trials,
        }],
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[done] → {out}')


if __name__ == '__main__':
    main()

