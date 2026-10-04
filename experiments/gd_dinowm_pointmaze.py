#!/usr/bin/env python
"""Gradient-based planning with DINO-WM on PointMaze.

Sanity-check / baseline: if the paper shows GD planning works with DINO-WM
on PointMaze, this script should reproduce that.  It mirrors exactly the
differentiable rollout structure of the GD planner in

    github.com/qw3rtman/robust-world-model-planning  (planning/gd.py)

adapted to the DINO-WM checkpoint format used by our CEM script
(experiments/cem_dinowm_pointmaze.py).

Usage
-----
  MUJOCO_GL=egl python experiments/gd_dinowm_pointmaze.py \\
      --dino-wm-dir ~/dino_wm \\
      --ckpt /mnt/t7shield/jepa_results/dinowm_checkpoints/outputs/point_maze/checkpoints/model_latest.pth \\
      --trials 10 --n-steps 200 \\
      --planning-horizon 25 --executed-steps 1 \\
      --gd-steps 50 --lr 0.05 --action-noise 0.01 \\
      --objective last \\
      --output results/gd_dinowm_pointmaze.json
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

# ── DINO-WM constants (same as cem_dinowm_pointmaze.py) ─────────────────────
FRAMESKIP   = 5
D_VIS       = 384
D_PROP_OUT  = 10
D_ACT_OUT   = 10
D_TOTAL     = D_VIS + D_PROP_OUT + D_ACT_OUT   # 404
ACT_IN_DIM  = FRAMESKIP * 2                     # 10  (5 × 2D force)
PROP_IN_DIM = 4


# ── model loading (identical to cem_dinowm_pointmaze.py) ─────────────────────

def setup_dinowm(dino_wm_dir: str):
    p = str(Path(dino_wm_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def make_env(seed: int = 0, image_size: int = 196):
    from envs.pointmaze_visual import PointMazeVisual
    env_cfg = {'environment': {'maze_map': 'U', 'image_size': image_size,
                               'action_scale': 1.0}}
    return PointMazeVisual(env_cfg, seed=seed)


def load_model(ckpt_path: str, dino_wm_dir: str, device: torch.device) -> dict:
    setup_dinowm(dino_wm_dir)
    from models.dino import DinoV2Encoder
    print('[DINO-WM GD] loading checkpoint …')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    parts = {}
    for k in ['predictor', 'proprio_encoder', 'action_encoder']:
        if k not in ckpt:
            raise KeyError(f"'{k}' missing — available: {list(ckpt.keys())}")
        parts[k] = ckpt[k].to(device).eval()
    encoder = DinoV2Encoder(name='dinov2_vits14', feature_key='x_norm_patchtokens')
    parts['encoder'] = encoder.to(device).eval()
    n_pred = sum(p.numel() for p in parts['predictor'].parameters())
    print(f'[DINO-WM GD] predictor params={n_pred/1e6:.1f}M')
    return parts


# ── observation encoding (same as cem_dinowm_pointmaze.py) ───────────────────

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


@torch.no_grad()
def encode_frame(parts: dict, obs_hwc: np.ndarray, state: np.ndarray,
                 action_10d: np.ndarray | None,
                 device: torch.device) -> torch.Tensor:
    """(obs, state, action) → (1, N_patches, D_TOTAL)."""
    if action_10d is None:
        action_10d = np.zeros(ACT_IN_DIM, dtype=np.float32)
    vis_emb = parts['encoder'](preprocess_obs(obs_hwc, device))  # (1, N, 384)
    N = vis_emb.shape[1]
    _pe_dtype = next(parts['proprio_encoder'].parameters()).dtype
    prop_in = torch.as_tensor(state[:PROP_IN_DIM].astype(np.float32),
                               device=device).to(_pe_dtype).unsqueeze(0).unsqueeze(0)
    prop_emb = parts['proprio_encoder'](prop_in)
    prop_tiled = prop_emb[:, 0:1, :].expand(-1, N, -1)
    _ae_dtype = next(parts['action_encoder'].parameters()).dtype
    act_in = torch.as_tensor(action_10d[:ACT_IN_DIM].astype(np.float32),
                               device=device).to(_ae_dtype).unsqueeze(0).unsqueeze(0)
    act_emb = parts['action_encoder'](act_in)
    act_tiled = act_emb[:, 0:1, :].expand(-1, N, -1)
    return torch.cat([vis_emb, prop_tiled, act_tiled], dim=2)  # (1, N, D_TOTAL)


def build_goal_context(parts: dict, env, goal_xy: np.ndarray,
                       num_hist: int, device: torch.device):
    goal_state = np.array([goal_xy[0], goal_xy[1], 0., 0.], dtype=np.float32)
    goal_obs, _, _ = env.reset_to_state(goal_state)
    zero_act   = np.zeros(ACT_IN_DIM, dtype=np.float32)
    goal_frame = encode_frame(parts, goal_obs, goal_state, zero_act, device)
    z_goal_ctx  = goal_frame.unsqueeze(1).expand(-1, num_hist, -1, -1).clone()
    z_goal_mean = z_goal_ctx[0, -1, :, :].mean(dim=0)   # (D_TOTAL,)
    return z_goal_ctx, z_goal_mean


# ── differentiable rollout ────────────────────────────────────────────────────

def differentiable_rollout(parts: dict, z_ctx_init: torch.Tensor,
                           actions_seq: torch.Tensor,
                           objective: str = 'last') -> torch.Tensor:
    """Roll out DINO-WM under `actions_seq` with gradient tracking.

    Parameters
    ----------
    z_ctx_init : (1, T, N, D)  initial context, detached from computation graph
    actions_seq : (H, ACT_IN_DIM)  action sequence to optimise (requires_grad)
    objective : 'last' — return only final latent mean;
                'all'  — return mean of latents over all steps

    Returns
    -------
    Tensor of shape (1, D_VIS) for 'last', or (H, D_VIS) for 'all'.
    """
    predictor      = parts['predictor']
    action_encoder = parts['action_encoder']

    B, T, N, D = z_ctx_init.shape
    H = actions_seq.shape[0]

    z_batch = z_ctx_init.detach().clone()   # (1, T, N, D) — detach so we only optimize actions
    pred_dtype = next(predictor.parameters()).dtype

    z_vis_list = []
    for h in range(H):
        a = actions_seq[h:h+1].clamp(-1., 1.)               # (1, ACT_IN_DIM)
        act_emb   = action_encoder(a.to(pred_dtype).unsqueeze(1))    # (1, 1, D_ACT_OUT)
        act_tiled = act_emb[:, 0:1, :].expand(-1, N, -1)   # (1, N, D_ACT_OUT)

        # Build z_in: replace action slot of last context frame — no in-place ops
        z_last_vp  = z_batch[:, -1, :, :D_VIS + D_PROP_OUT]         # (1, N, D_VIS+D_PROP_OUT)
        z_last_new = torch.cat([z_last_vp.to(pred_dtype),
                                act_tiled.to(pred_dtype)], dim=-1)   # (1, N, D)
        if T > 1:
            z_in = torch.cat([z_batch[:, :-1, :, :].to(pred_dtype),
                               z_last_new.unsqueeze(1)], dim=1)       # (1, T, N, D)
        else:
            z_in = z_last_new.unsqueeze(1)                            # (1, 1, N, D)

        z_flat = z_in.reshape(B, T * N, D)
        z_pred = predictor(z_flat).reshape(B, T, N, D)               # (1, T, N, D)

        # Shift context window
        z_batch = torch.cat([z_batch[:, 1:].to(pred_dtype),
                              z_pred[:, -1:]], dim=1)                 # (1, T, N, D)

        # Collect visual mean of predicted frame
        z_vis_list.append(z_batch[:, -1, :, :D_VIS].mean(dim=1))    # (1, D_VIS)

    if objective == 'last':
        return z_vis_list[-1]   # (1, D_VIS)
    else:
        return torch.stack(z_vis_list, dim=0)   # (H, D_VIS) — but still (1, D_VIS) per step


# ── gradient-based planner ────────────────────────────────────────────────────

class DinoWMGDPlanner:
    """Gradient-based action-sequence optimiser for DINO-WM on PointMaze.

    Mirrors planning/gd.py from qw3rtman/robust-world-model-planning.
    Adds multi-restart support: run GD from N random initialisations per plan
    call and keep the trajectory with the lowest predicted cost.
    """

    def __init__(self, parts: dict, num_hist: int,
                 horizon: int, executed_steps: int,
                 gd_steps: int, lr: float, action_noise: float,
                 device: torch.device,
                 objective: str = 'last',
                 warm_start: bool = True,
                 optimizer_type: str = 'adam',
                 n_restarts: int = 1):
        self.parts          = parts
        self.num_hist       = num_hist
        self.horizon        = horizon
        self.executed_steps = min(executed_steps, horizon)
        self.gd_steps       = gd_steps
        self.lr             = lr
        self.action_noise   = action_noise
        self.device         = device
        self.objective      = objective
        self.warm_start     = warm_start
        self.optimizer_type = optimizer_type
        self.n_restarts     = max(1, n_restarts)
        self._prev_u        = None

    def _make_optimizer(self, u):
        if self.optimizer_type == 'adam':
            return torch.optim.Adam([u], lr=self.lr)
        elif self.optimizer_type == 'adamw':
            return torch.optim.AdamW([u], lr=self.lr)
        elif self.optimizer_type == 'sgd':
            return torch.optim.SGD([u], lr=self.lr)
        elif self.optimizer_type == 'momentum':
            return torch.optim.SGD([u], lr=self.lr, momentum=0.9)
        raise ValueError(f'Unknown optimizer_type: {self.optimizer_type}')

    def _run_one(self, z_ctx: torch.Tensor, z_goal_vis: torch.Tensor,
                 init_u: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Run GD from a single initialisation; return (u_clamped, final_cost)."""
        u = init_u.clone().requires_grad_(True)
        optimizer = self._make_optimizer(u)
        for _ in range(self.gd_steps):
            optimizer.zero_grad()
            if self.objective == 'last':
                z_pred = differentiable_rollout(
                    self.parts, z_ctx, u, objective='last')   # (1, D_VIS)
                loss = (z_pred[0] - z_goal_vis).pow(2).sum()
            else:
                z_preds = differentiable_rollout(
                    self.parts, z_ctx, u, objective='all')    # (H, D_VIS)
                loss = (z_preds - z_goal_vis.unsqueeze(0)).pow(2).sum(-1).mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                u += torch.randn_like(u) * self.action_noise
        with torch.no_grad():
            u_clamped = u.clamp(-1., 1.)
            # evaluate terminal cost for restart selection
            z_final = differentiable_rollout(
                self.parts, z_ctx, u_clamped, objective='last')
            cost = float((z_final[0] - z_goal_vis).pow(2).sum().cpu())
        return u_clamped.detach(), cost

    def plan(self, z_ctx: torch.Tensor,
             z_goal_mean: torch.Tensor) -> np.ndarray:
        """Optimise over n_restarts initialisations; return best plan."""
        z_goal_vis = z_goal_mean[:D_VIS].detach()  # (D_VIS,)

        # Build restart initialisations
        inits = []
        if self.warm_start and self._prev_u is not None:
            prev = self._prev_u.detach()
            shifted = torch.cat([
                prev[self.executed_steps:],
                torch.zeros(self.executed_steps, ACT_IN_DIM,
                            device=self.device, dtype=prev.dtype)
            ], dim=0)
            inits.append(shifted)   # first restart = warm-start
        for _ in range(self.n_restarts - len(inits)):
            inits.append(torch.randn(self.horizon, ACT_IN_DIM,
                                     device=self.device))

        best_u, best_cost = None, float('inf')
        for init_u in inits:
            u_clamped, cost = self._run_one(z_ctx, z_goal_vis, init_u)
            if cost < best_cost:
                best_cost = cost
                best_u    = u_clamped

        self._prev_u = best_u
        return best_u[:self.executed_steps].cpu().numpy()


# ── trial loop ────────────────────────────────────────────────────────────────

def gd_trial(parts: dict, planner: DinoWMGDPlanner,
             env, z_goal_mean: torch.Tensor,
             goal_xy: np.ndarray, num_hist: int,
             n_steps: int, success_threshold: float,
             success_hold_steps: int, device: torch.device,
             init_state: np.ndarray | None = None) -> dict:

    if init_state is not None:
        obs, state, _ = env.reset_to_state(init_state.copy())
    else:
        obs, state, _ = env.reset()
    zero_act = np.zeros(ACT_IN_DIM, dtype=np.float32)

    init_frame = encode_frame(parts, obs, state, zero_act, device)
    z_ctx = init_frame.unsqueeze(1).expand(-1, num_hist, -1, -1).clone()

    z_goal_gpu    = z_goal_mean.to(device)
    states        = [state.copy()]
    consec_stable = 0
    model_step    = 0
    done          = False

    planner._prev_u = None   # reset warm-start each episode

    while model_step < n_steps and not done:
        plan_acts = planner.plan(z_ctx, z_goal_gpu)   # (executed_steps, ACT_IN_DIM)

        for e in range(len(plan_acts)):
            if model_step >= n_steps or done:
                break
            u_10d = np.clip(plan_acts[e], -1., 1.)

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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='GD planning with DINO-WM on PointMaze — sanity check.')
    p.add_argument('--dino-wm-dir',        required=True)
    p.add_argument('--ckpt',               required=True)
    p.add_argument('--trials',             type=int,   default=10)
    p.add_argument('--n-steps',            type=int,   default=200)
    p.add_argument('--num-hist',           type=int,   default=3)
    p.add_argument('--planning-horizon',   type=int,   default=25)
    p.add_argument('--executed-steps',     type=int,   default=1)
    p.add_argument('--gd-steps',           type=int,   default=50)
    p.add_argument('--lr',                 type=float, default=0.05)
    p.add_argument('--action-noise',       type=float, default=0.01)
    p.add_argument('--objective',          choices=['last', 'all'], default='last')
    p.add_argument('--optimizer',          choices=['adam', 'adamw', 'sgd', 'momentum'],
                                           default='adam')
    p.add_argument('--n-restarts',          type=int,   default=1,
                   help='Number of GD restarts per plan call; best cost wins. '
                        'First restart is warm-start (if enabled), rest are randn.')
    p.add_argument('--no-warm-start',      action='store_true')
    p.add_argument('--success-threshold',  type=float, default=0.5)
    p.add_argument('--success-hold-steps', type=int,   default=1)
    p.add_argument('--seed',               type=int,   default=0)
    p.add_argument('--device',             default='cuda')
    p.add_argument('--output',             required=True)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'[DINO-WM GD] device={device}')

    parts = load_model(args.ckpt, args.dino_wm_dir, device)
    env   = make_env(seed=args.seed, image_size=196)

    planner = DinoWMGDPlanner(
        parts=parts,
        num_hist=args.num_hist,
        horizon=args.planning_horizon,
        executed_steps=args.executed_steps,
        gd_steps=args.gd_steps,
        lr=args.lr,
        action_noise=args.action_noise,
        device=device,
        objective=args.objective,
        warm_start=not args.no_warm_start,
        optimizer_type=args.optimizer,
        n_restarts=args.n_restarts,
    )
    print(f'[DINO-WM GD] H={planner.horizon} K={planner.executed_steps} '
          f'gd_steps={planner.gd_steps} lr={planner.lr} '
          f'noise={planner.action_noise} objective={planner.objective} '
          f'restarts={planner.n_restarts}')

    trials, n_success = [], 0

    for i in range(args.trials):
        start_seed = args.seed + 99 * i + 1
        goal_seed  = args.seed + 99 * i + 2

        obs_dict, _ = env._env.reset(seed=int(start_seed))
        start_state = obs_dict['observation'][:4].astype(np.float32)

        obs_dict, _ = env._env.reset(seed=int(goal_seed))
        goal_xy     = obs_dict['observation'][:2].astype(np.float32)

        print(f'[trial {i:03d}] start=[{start_state[0]:.2f},{start_state[1]:.2f}]  '
              f'goal=[{goal_xy[0]:.2f},{goal_xy[1]:.2f}]  '
              f'encoding goal …', end=' ', flush=True)

        _, z_goal_mean = build_goal_context(parts, env, goal_xy,
                                            args.num_hist, device)

        t0 = time.time()
        row = gd_trial(
            parts, planner, env, z_goal_mean, goal_xy,
            args.num_hist, args.n_steps,
            args.success_threshold, args.success_hold_steps, device,
            init_state=start_state)
        elapsed = time.time() - t0

        n_success += int(row['success'])
        trials.append(row)
        print(f'success={row["success"]}  final={row["final_error"]:.3f}  '
              f't={elapsed:.1f}s')

    sr = n_success / max(len(trials), 1)
    print(f'\n[DINO-WM GD] success rate: {sr:.1%}  ({n_success}/{len(trials)})')

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'protocol': {
            'planner':          'gradient_descent',
            'n_steps':          args.n_steps,
            'num_hist':         args.num_hist,
            'planning_horizon': args.planning_horizon,
            'executed_steps':   args.executed_steps,
            'gd_steps':         args.gd_steps,
            'lr':               args.lr,
            'action_noise':     args.action_noise,
            'objective':        args.objective,
            'optimizer':        args.optimizer,
            'warm_start':       not args.no_warm_start,
            'n_restarts':       args.n_restarts,
            'success_threshold': args.success_threshold,
            'seed':             args.seed,
            'n_trials':         args.trials,
        },
        'success_rate': sr,
        'trials': trials,
    }, indent=2))
    print(f'[done] {out}')


if __name__ == '__main__':
    main()

