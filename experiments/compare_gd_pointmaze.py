#!/usr/bin/env python
"""Gradient-based planning for PointMaze JEPA checkpoints.

Uses the same seed/trial protocol as compare_cem_dinowm_pointmaze.py and the
same GD optimisation loop as compare_gd_smwm.py (Adam, noise injection,
warm-start, multi-restart).

Usage
-----
  MUJOCO_GL=egl python experiments/compare_gd_pointmaze.py \\
      --ckpts /path/to/ckpt1.pt /path/to/ckpt2.pt \\
      --cfgs  configs/cfg1.yaml  configs/cfg2.yaml  \\
      --trials 10 --n-steps 200 \\
      --planning-horizon 25 --executed-steps 25 \\
      --gd-steps 50 --lr 0.1 --action-noise 0.05 \\
      --objective last \\
      --output results/gd_pointmaze.json
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

from experiments.probe_utils import encode_obs, load_bundle
from envs.pointmaze_visual import PointMazeVisual


# ── helpers (identical to compare_cem_dinowm_pointmaze.py) ───────────────────

def checkpoint_label(checkpoint):
    path = Path(checkpoint)
    run_name = (path.parent.parent.name
                if path.parent.name == 'checkpoints' else path.parent.name)
    r = run_name.lower()
    if 'endpoint' in r and 'rollout' in r: return 'Rollout+EndpointAR'
    if 'endpoint' in r and 'fwd' in r:     return 'Fwd+EndpointAR'
    if 'endpoint' in r:                     return 'EndpointAR'
    if 'sigreg' in r and 'rollout' in r:   return 'SIGreg+Rollout'
    if 'ar_1step_ms' in r or 'ar-1step-ms' in r: return 'AR 1step+MS'
    if 'sigreg' in r:                       return 'SIGreg'
    return run_name


def make_env(bundle, seed):
    env_cfg = bundle['env_cfg']
    if 'environment' not in env_cfg:
        env_cfg['environment'] = {
            'maze_map':    bundle['model_cfg'].get('maze_map', 'U'),
            'image_size':  int(bundle['model_cfg'].get('image_size', 64)),
            'action_scale': 1.0,
        }
    return PointMazeVisual(env_cfg, seed=seed)


def sample_random_state(env, seed):
    obs_dict, _ = env._env.reset(seed=int(seed))
    return obs_dict['observation'].astype(np.float32)


def encode_state_as_goal(bundle, env, state):
    goal_state = np.array([state[0], state[1], 0., 0.], dtype=np.float32)
    obs, actual_state, _ = env.reset_to_state(goal_state)
    z = encode_obs(bundle, obs, obs, actual_state)
    return z, goal_state


# ── differentiable rollout ────────────────────────────────────────────────────

def differentiable_rollout(model, z0, u, action_scale, objective='last'):
    """Roll out model from z0 under action sequence u with gradient tracking.

    u : (H, 2)   2-D raw actions to optimise
    Returns (1, D) for 'last', (H, D) for 'all'.
    """
    z   = z0          # (1, D)
    zs  = []
    for t in range(u.shape[0]):
        a_raw = (u[t:t+1] / action_scale).reshape(1, -1)   # (1, 2)
        a_ctx = model.expand_action(a_raw).unsqueeze(1)     # (1, 1, ctx)
        z     = model.predict(z[:, None], a_ctx)[:, 0]      # (1, D)
        zs.append(z)
    if objective == 'last':
        return zs[-1]                          # (1, D)
    return torch.stack(zs, dim=0).squeeze(1)  # (H, D)


# ── GD planner ────────────────────────────────────────────────────────────────

class PointMazeGDPlanner:
    """GD action-sequence optimiser for PointMaze JEPA models.

    Matches the structure of CartPoleGDPlanner in compare_gd_smwm.py.
    Multi-restart: runs n_restarts GD optimisations per plan call and keeps
    the trajectory with the lowest predicted terminal cost.
    """

    def __init__(self, model, action_scale, horizon, executed_steps,
                 gd_steps, lr, action_noise, device,
                 objective='last', warm_start=True,
                 optimizer_type='adam', n_restarts=1):
        self.model          = model
        self.action_scale   = float(action_scale)
        self.horizon        = int(horizon)
        self.executed_steps = min(int(executed_steps), self.horizon)
        self.gd_steps       = int(gd_steps)
        self.lr             = float(lr)
        self.action_noise   = float(action_noise)
        self.device         = device
        self.objective      = objective
        self.warm_start     = warm_start
        self.optimizer_type = optimizer_type
        self.n_restarts     = max(1, int(n_restarts))
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

    def _run_one(self, z0, z_goal, init_u):
        """Run GD from one initialisation; return (u_clamped, terminal_cost)."""
        u         = init_u.clone().requires_grad_(True)
        optimizer = self._make_optimizer(u)
        for _ in range(self.gd_steps):
            optimizer.zero_grad()
            if self.objective == 'last':
                z_pred = differentiable_rollout(
                    self.model, z0, u, self.action_scale, 'last')   # (1, D)
                loss = (z_pred[0] - z_goal[0]).pow(2).sum()
            else:
                z_preds = differentiable_rollout(
                    self.model, z0, u, self.action_scale, 'all')    # (H, D)
                loss = (z_preds - z_goal).pow(2).sum(-1).mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                u += torch.randn_like(u) * self.action_noise
        with torch.no_grad():
            u_clamped = u.clamp(-self.action_scale, self.action_scale)
            z_final   = differentiable_rollout(
                self.model, z0, u_clamped, self.action_scale, 'last')
            cost = float((z_final[0] - z_goal[0]).pow(2).sum().cpu())
        return u_clamped.detach(), cost

    def plan(self, z0, z_goal):
        """Return (executed_steps, 2) action array in raw action units."""
        inits = []
        if self.warm_start and self._prev_u is not None:
            prev    = self._prev_u.detach()
            shifted = torch.cat([
                prev[self.executed_steps:],
                torch.zeros(self.executed_steps, 2,
                            device=self.device, dtype=prev.dtype)
            ], dim=0)
            inits.append(shifted)
        for _ in range(self.n_restarts - len(inits)):
            inits.append(torch.randn(self.horizon, 2, device=self.device))

        best_u, best_cost = None, float('inf')
        for init_u in inits:
            u_clamped, cost = self._run_one(z0, z_goal, init_u)
            if cost < best_cost:
                best_cost = cost
                best_u    = u_clamped

        self._prev_u = best_u
        return best_u[:self.executed_steps].cpu().numpy()


# ── trial evaluation ──────────────────────────────────────────────────────────

def gd_trial(bundle, env, planner, z_goal, goal_xy, start_state,
             n_steps, success_threshold, success_hold_steps,
             frame_skip=1):
    obs, state, _ = env.reset_to_state(start_state)
    prev_obs  = obs.copy()
    states    = [state.copy()]
    step      = 0
    consec    = 0

    planner._prev_u = None   # reset warm-start per episode

    while step < n_steps:
        with torch.no_grad():
            z = encode_obs(bundle, obs, prev_obs, state)

        sequence = planner.plan(z, z_goal)   # (executed_steps, 2)

        for a in sequence:
            if step >= n_steps:
                break
            a_clipped = np.clip(a, -bundle['action_scale'], bundle['action_scale'])
            prev_obs  = obs.copy()
            done      = False
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
                consec += 1
                if consec >= success_hold_steps:
                    break
            else:
                consec = 0
            if done:
                break

        if consec >= success_hold_steps:
            break

    with torch.no_grad():
        z_final = encode_obs(bundle, obs, prev_obs, state)

    states  = np.array(states)
    xy_err  = np.linalg.norm(states[:, :2] - goal_xy[None], axis=1)
    stable  = xy_err < success_threshold
    success = bool(stable[-1])
    tail    = stable[-success_hold_steps:]
    held    = bool(len(tail) == success_hold_steps and tail.all())

    return {
        'success':            success,
        'held':               held,
        'final_error':        float(xy_err[-1]),
        'max_error':          float(np.max(xy_err)),
        'fraction_stable':    float(np.mean(stable[1:])) if len(stable) > 1 else 0.,
        'final_latent_error': float(torch.linalg.vector_norm(z_final - z_goal)),
        'goal_xy':            goal_xy.tolist(),
        'start_xy':           start_state[:2].tolist(),
        'states':             states.tolist(),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='GD planning for PointMaze JEPA models (DINO-WM protocol).')
    p.add_argument('--ckpts',             nargs='+', required=True)
    p.add_argument('--cfgs',              nargs='+', required=True)
    p.add_argument('--trials',            type=int,   default=10)
    p.add_argument('--n-steps',           type=int,   default=200)
    p.add_argument('--planning-horizon',  type=int,   default=25)
    p.add_argument('--executed-steps',    type=int,   default=25)
    p.add_argument('--gd-steps',          type=int,   default=50)
    p.add_argument('--lr',                type=float, default=0.1)
    p.add_argument('--action-noise',      type=float, default=0.05)
    p.add_argument('--objective',         choices=['last', 'all'], default='last')
    p.add_argument('--optimizer',         choices=['adam', 'adamw', 'sgd', 'momentum'],
                                          default='adam')
    p.add_argument('--n-restarts',        type=int,   default=1,
                   help='GD restarts per plan call; first=warm-start, rest=randn')
    p.add_argument('--no-warm-start',     action='store_true')
    p.add_argument('--success-threshold', type=float, default=0.5)
    p.add_argument('--success-hold-steps',type=int,   default=1)
    p.add_argument('--frame-skip',        type=int,   default=1,
                   help='Execute each planned action this many times (set 5 to match DINO-WM)')
    p.add_argument('--seed',              type=int,   default=0)
    p.add_argument('--device',            default='cuda')
    p.add_argument('--output',            required=True)
    args = p.parse_args()

    if len(args.cfgs) == 1:
        args.cfgs = args.cfgs * len(args.ckpts)

    device  = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    results = {'protocol': vars(args), 'models': []}

    for ckpt, cfg in zip(args.ckpts, args.cfgs):
        label = checkpoint_label(ckpt)
        print(f'\n[{label}] loading {ckpt}')
        bundle = load_bundle(ckpt, cfg, args.device)

        env = make_env(bundle, seed=0)

        planner = PointMazeGDPlanner(
            model        = bundle['model'],
            action_scale = bundle['action_scale'],
            horizon      = args.planning_horizon,
            executed_steps = args.executed_steps,
            gd_steps     = args.gd_steps,
            lr           = args.lr,
            action_noise = args.action_noise,
            device       = bundle['device'],
            objective    = args.objective,
            warm_start   = not args.no_warm_start,
            optimizer_type = args.optimizer,
            n_restarts   = args.n_restarts,
        )
        print(f'[{label}] H={planner.horizon} K={planner.executed_steps} '
              f'gd_steps={planner.gd_steps} lr={planner.lr} '
              f'noise={planner.action_noise} objective={planner.objective} '
              f'restarts={planner.n_restarts} frame_skip={args.frame_skip}')

        trials, n_success = [], 0
        for i in range(args.trials):
            start_seed = args.seed + 99 * i + 1
            goal_seed  = args.seed + 99 * i + 2

            start_state = sample_random_state(env, start_seed)
            goal_raw    = sample_random_state(env, goal_seed)
            goal_xy     = goal_raw[:2]

            z_goal, _ = encode_state_as_goal(bundle, env, goal_raw)

            torch.manual_seed(args.seed + i)
            row = gd_trial(bundle, env, planner, z_goal, goal_xy,
                           start_state, args.n_steps,
                           args.success_threshold, args.success_hold_steps,
                           frame_skip=args.frame_skip)

            n_success += int(row['success'])
            trials.append(row)
            print(f'[{label}] trial {i:03d}  success={row["success"]}  '
                  f'err={row["final_error"]:.4f}  '
                  f'start={np.round(start_state[:2], 2)}  '
                  f'goal={np.round(goal_xy, 2)}')

        sr = n_success / max(len(trials), 1)
        print(f'[{label}] success rate: {sr:.1%}  ({n_success}/{len(trials)})')
        env.close()

        results['models'].append({
            'label':        label,
            'success_rate': sr,
            'trials':       trials,
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f'\n[done] {out}')


if __name__ == '__main__':
    main()
