#!/usr/bin/env python
"""Gradient-descent planning for CartPole using a DINO-WM checkpoint.

Optimises an H-step action sequence by backpropagating through the latent
predictor to minimise the squared distance to a goal latent:

    L(u) = ||mean_patch(z_H(u)) - mean_patch(z*)||²       (objective='last')
or
    L(u) = mean_t ||mean_patch(z_t(u)) - mean_patch(z*)||² (objective='all')

Replanning is MPC-style: plan H steps, execute K steps, replan.

Usage
-----
  python experiments/gd_dinowm_cartpole.py \\
      --dino-wm-dir ~/dino_wm \\
      --ckpt /home/dz2478/dino_wm/outputs/2026-09-08/17-46-00/checkpoints/model_179.pth \\
      --data /mnt/t7shield/dinowm_cartpole \\
      --trials 10 --primitive-budget 300 \\
      --planning-horizon 10 --executed-steps 1 \\
      --gd-steps 50 --lr 0.1 --action-noise 0.05 \\
      --objective last \\
      --success-threshold 0.7 \\
      --output results/gd_dinowm_cartpole.json
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

# ── dino-wm CartPole constants ────────────────────────────────────────────────
FRAMESKIP   = 5
D_VIS       = 384
D_PROP_OUT  = 10
D_ACT_OUT   = 10
D_TOTAL     = D_VIS + D_PROP_OUT + D_ACT_OUT   # 404
ACT_IN_DIM  = 1
PROP_IN_DIM = 4
IMG_SIZE    = 112   # 8×14; matches training transform rounding

_NORM_MEAN = torch.tensor([0.5, 0.5, 0.5])
_NORM_STD  = torch.tensor([0.5, 0.5, 0.5])


# ── Package / env setup ───────────────────────────────────────────────────────

def setup_dinowm(dino_wm_dir: str):
    p = str(Path(dino_wm_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def make_env(seed: int = 0):
    from envs.cartpole_visual import ContinuousCartpoleVisual
    return ContinuousCartpoleVisual(
        frame_skip=FRAMESKIP, image_size=IMG_SIZE, seed=seed)


# ── Action stats ──────────────────────────────────────────────────────────────

def load_action_stats(data_path: str, device: torch.device):
    p = Path(data_path)
    actions     = torch.load(p / 'actions.pth').float()
    seq_lengths = torch.load(p / 'seq_lengths.pth')
    all_acts = [actions[i, :int(seq_lengths[i])] for i in range(len(seq_lengths))]
    all_acts = torch.vstack(all_acts)
    mean = all_acts.mean(dim=0).to(device)
    std  = all_acts.std(dim=0).to(device).clamp(min=1e-6)
    print(f'[action stats]  mean={mean.item():.4f}  std={std.item():.4f}')
    return mean, std


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, dino_wm_dir: str, device: torch.device) -> dict:
    setup_dinowm(dino_wm_dir)
    from models.dino import DinoV2Encoder
    print('[GD] loading checkpoint …')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    parts = {}
    for k in ['predictor', 'proprio_encoder', 'action_encoder']:
        if k not in ckpt:
            raise KeyError(f"'{k}' not found; available: {list(ckpt.keys())}")
        parts[k] = ckpt[k].to(device).eval()
    parts['encoder'] = DinoV2Encoder(
        name='dinov2_vits14', feature_key='x_norm_patchtokens').to(device).eval()
    print(f'[GD] predictor params={sum(p.numel() for p in parts["predictor"].parameters())/1e6:.1f}M')
    return parts


# ── Image / frame encoding ────────────────────────────────────────────────────

def preprocess_obs(obs_hwc: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(obs_hwc.copy()).float().permute(2, 0, 1) / 255.0
    if x.shape[1] != IMG_SIZE or x.shape[2] != IMG_SIZE:
        x = F.interpolate(x.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
                          mode='bilinear', align_corners=False).squeeze(0)
    x = (x - _NORM_MEAN[:, None, None]) / _NORM_STD[:, None, None]
    return x.unsqueeze(0).to(device)


@torch.no_grad()
def encode_frame(parts: dict, obs_hwc: np.ndarray, state: np.ndarray,
                 action_norm: np.ndarray, device: torch.device) -> torch.Tensor:
    """Returns (1, N_patches, D_TOTAL)."""
    vis_emb = parts['encoder'](preprocess_obs(obs_hwc, device))
    N = vis_emb.shape[1]

    _pe_dtype = next(parts['proprio_encoder'].parameters()).dtype
    prop_in  = torch.as_tensor(state[:PROP_IN_DIM].astype(np.float32),
                               device=device).to(_pe_dtype).unsqueeze(0).unsqueeze(0)
    prop_emb = parts['proprio_encoder'](prop_in)
    prop_t   = prop_emb[:, 0:1, :].expand(-1, N, -1)

    _ae_dtype = next(parts['action_encoder'].parameters()).dtype
    act_in  = torch.as_tensor(action_norm[:ACT_IN_DIM].astype(np.float32),
                              device=device).to(_ae_dtype).unsqueeze(0).unsqueeze(0)
    act_emb = parts['action_encoder'](act_in)
    act_t   = act_emb[:, 0:1, :].expand(-1, N, -1)

    return torch.cat([vis_emb, prop_t, act_t], dim=2)   # (1, N, 404)


def build_goal_ctx(parts, goal_state, num_hist, device, act_mean, act_std):
    env = make_env(seed=0)
    goal_obs, _, _ = env.reset_to_state(goal_state.copy())
    env.close()
    zero_norm  = np.zeros(ACT_IN_DIM, dtype=np.float32)
    goal_frame = encode_frame(parts, goal_obs, goal_state, zero_norm, device)
    z_goal_ctx  = goal_frame.unsqueeze(1).expand(-1, num_hist, -1, -1).clone()
    z_goal_mean = z_goal_ctx[0, -1, :, :].mean(dim=0)
    return z_goal_ctx, z_goal_mean.cpu().numpy()


# ── GD planner ────────────────────────────────────────────────────────────────

class DINOWMGDPlanner:
    """Adam-based action-sequence optimiser for DINO-WM CartPole.

    Works in raw action space (clipped to action bounds). Normalisation is
    applied inside the rollout before calling the action encoder.
    """

    def __init__(self, parts, act_mean, act_std, num_hist,
                 horizon, executed_steps, gd_steps, lr, action_noise,
                 action_lb, action_ub, device,
                 objective='last', warm_start=True, optimizer_type='adam'):
        self.parts          = parts
        self.act_mean       = act_mean
        self.act_std        = act_std
        self.num_hist       = num_hist
        self.horizon        = horizon
        self.executed_steps = min(executed_steps, horizon)
        self.gd_steps       = gd_steps
        self.lr             = lr
        self.action_noise   = action_noise
        self.lb             = action_lb
        self.ub             = action_ub
        self.device         = device
        self.objective      = objective
        self.warm_start     = warm_start
        self.optimizer_type = optimizer_type
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
        raise ValueError(f'Unknown optimizer: {self.optimizer_type}')

    def _rollout(self, z_ctx: torch.Tensor, u: torch.Tensor):
        """Forward pass with gradients.

        z_ctx : (1, T, N, D) – detached context
        u     : (H,) – raw actions, requires_grad=True
        Returns list of (1, D_TOTAL) mean-patch latents for each step.
        """
        ae       = self.parts['action_encoder']
        pred     = self.parts['predictor']
        ae_dtype = next(ae.parameters()).dtype
        _, T, N, D = z_ctx.shape

        z  = z_ctx.clone()   # (1, T, N, D) – no grad; grads enter through u
        zs = []
        for t in range(self.horizon):
            a_norm = ((u[t] - self.act_mean[0]) / self.act_std[0]).to(ae_dtype)
            a_emb  = ae(a_norm.reshape(1, 1, 1))               # (1,1,D_ACT)
            a_tile = a_emb.expand(1, N, -1)                    # (1,N,D_ACT)

            # Replace action part of last context frame (avoid in-place ops)
            z_last_vp  = z[:, -1:, :, :D_VIS + D_PROP_OUT]   # (1,1,N,D_VP)
            z_last_new = torch.cat([z_last_vp, a_tile.unsqueeze(1)], dim=-1)  # (1,1,N,D)
            z_in       = torch.cat([z[:, :-1], z_last_new], dim=1)            # (1,T,N,D)

            z_next_flat = pred(z_in.reshape(1, T * N, D))                     # (1,T*N,D)
            z_next      = z_next_flat.reshape(1, T, N, D)[:, -1]              # (1,N,D)

            zs.append(z_next.mean(dim=1))                      # (1,D)
            z = torch.cat([z[:, 1:], z_next.unsqueeze(1)], dim=1)

        return zs

    def plan_sequence(self, z_ctx: torch.Tensor,
                      z_goal_mean: np.ndarray) -> np.ndarray:
        """Returns (executed_steps,) raw action array."""
        z_goal = torch.tensor(z_goal_mean, device=self.device)

        # Initialise u (warm-start: shift previous plan)
        if self.warm_start and self._prev_u is not None:
            prev = self._prev_u.detach()
            u_init = torch.cat([prev[self.executed_steps:],
                                 torch.zeros(self.executed_steps, device=self.device)])
            u = u_init.clone().requires_grad_(True)
        else:
            u = torch.zeros(self.horizon, device=self.device, requires_grad=True)

        optimizer = self._make_optimizer(u)

        z_ctx_fixed = z_ctx.detach()
        for _ in range(self.gd_steps):
            optimizer.zero_grad()
            zs   = self._rollout(z_ctx_fixed, u)
            if self.objective == 'last':
                loss = (zs[-1] - z_goal).pow(2).sum()
            else:
                loss = torch.stack(zs, dim=1).sub(z_goal).pow(2).sum(-1).mean()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                u.add_(torch.randn_like(u) * self.action_noise)

        with torch.no_grad():
            u_clamped = u.clamp(self.lb, self.ub)

        self._prev_u = u_clamped.detach()
        return u_clamped[:self.executed_steps].cpu().numpy()


# ── Trial evaluation ──────────────────────────────────────────────────────────

def evaluate_trial(env, planner, parts, initial_state, goal_state,
                   z_goal_mean_np, act_mean, act_std,
                   primitive_budget, success_threshold, success_hold_steps,
                   num_hist, device):
    obs, state, _ = env.reset_to_state(initial_state.copy())
    frame_skip    = int(env.frame_skip)
    macro_budget  = int(np.ceil(primitive_budget / frame_skip))

    zero_norm  = np.zeros(ACT_IN_DIM, dtype=np.float32)
    init_frame = encode_frame(parts, obs, state, zero_norm, device)
    z_ctx      = init_frame.unsqueeze(1).expand(-1, num_hist, -1, -1).clone()

    planner._prev_u = None   # reset warm-start at episode start

    states  = [state.copy()]
    actions = []
    terminated = False

    while len(actions) < macro_budget and not terminated:
        seq       = planner.plan_sequence(z_ctx, z_goal_mean_np)
        remaining = macro_budget - len(actions)
        for a_raw in seq[:remaining]:
            obs, state, _, done, _ = env.step(float(a_raw))
            a_norm = np.array([(a_raw - float(act_mean[0].cpu()))
                               / float(act_std[0].cpu())], dtype=np.float32)
            new_frame = encode_frame(parts, obs, state, a_norm, device)
            z_ctx     = torch.cat([z_ctx[:, 1:], new_frame.unsqueeze(1)], dim=1)
            actions.append(float(a_raw))
            states.append(state.copy())
            if done:
                terminated = True
                break

    states = np.array(states)
    errors = np.linalg.norm(states - np.array(goal_state, dtype=np.float64)[None], axis=1)
    stable = errors < success_threshold
    tail   = stable[-success_hold_steps:]
    held   = bool(len(tail) == success_hold_steps and tail.all())
    success = bool(stable[-1]) if len(stable) else False

    return {
        'success':          success,
        'held':             held,
        'terminated':       terminated,
        'final_error':      float(errors[-1]),
        'max_error':        float(errors.max()),
        'fraction_stable':  float(np.mean(stable)),
        'macro_steps':      len(actions),
        'primitive_steps':  len(actions) * frame_skip,
        'actions':          actions,
        'states':           states.tolist(),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Gradient-descent planning for CartPole with DINO-WM.')
    p.add_argument('--dino-wm-dir', default=str(Path.home() / 'dino_wm'))
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data', default='/mnt/t7shield/dinowm_cartpole',
                   help='Dataset path for action stats')
    p.add_argument('--num-hist', type=int, default=3)
    p.add_argument('--trials', type=int, default=10)
    p.add_argument('--primitive-budget', type=int, default=300)
    p.add_argument('--planning-horizon', type=int, default=10)
    p.add_argument('--executed-steps', type=int, default=1)
    p.add_argument('--gd-steps', type=int, default=50)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--action-noise', type=float, default=0.05)
    p.add_argument('--objective', choices=['last', 'all'], default='last')
    p.add_argument('--optimizer', choices=['adam', 'adamw', 'sgd', 'momentum'],
                   default='adam')
    p.add_argument('--warm-start', action='store_true', default=True)
    p.add_argument('--no-warm-start', dest='warm_start', action='store_false')
    p.add_argument('--success-threshold', type=float, default=0.7)
    p.add_argument('--success-hold-steps', type=int, default=10)
    p.add_argument('--eps-range', type=float, nargs=2, default=[0.01, 0.15])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', required=True)
    args = p.parse_args()

    device = torch.device(args.device)
    rng    = np.random.default_rng(args.seed)

    parts             = load_model(args.ckpt, args.dino_wm_dir, device)
    act_mean, act_std = load_action_stats(args.data, device)

    planner = DINOWMGDPlanner(
        parts=parts, act_mean=act_mean, act_std=act_std, num_hist=args.num_hist,
        horizon=args.planning_horizon, executed_steps=args.executed_steps,
        gd_steps=args.gd_steps, lr=args.lr, action_noise=args.action_noise,
        action_lb=-10., action_ub=10., device=device,
        objective=args.objective, warm_start=args.warm_start,
        optimizer_type=args.optimizer)

    goal_state = np.zeros(4, dtype=np.float32)
    _, z_goal_mean_np = build_goal_ctx(parts, goal_state, args.num_hist,
                                       device, act_mean, act_std)

    print(f'[GD] H={args.planning_horizon} K={args.executed_steps} '
          f'gd_steps={args.gd_steps} lr={args.lr} noise={args.action_noise} '
          f'obj={args.objective} warm_start={args.warm_start}')

    lo, hi = args.eps_range
    initial_states = [rng.uniform(-hi, hi, size=4).astype(np.float32)
                      for _ in range(args.trials)]

    trial_results = []
    t0 = time.time()
    for i, x0 in enumerate(initial_states):
        print(f'  trial {i+1}/{args.trials} …', end=' ', flush=True)
        env = make_env(seed=args.seed + i)
        r   = evaluate_trial(env, planner, parts, x0, goal_state,
                             z_goal_mean_np, act_mean, act_std,
                             args.primitive_budget, args.success_threshold,
                             args.success_hold_steps, args.num_hist, device)
        env.close()
        trial_results.append(r)
        print(f"success={r['success']}  final_err={r['final_error']:.3f}  "
              f"steps={r['macro_steps']}")

    n_ok        = sum(r['success'] for r in trial_results)
    n_held      = sum(r['held']    for r in trial_results)
    frac_stable = float(np.mean([r['fraction_stable'] for r in trial_results]))
    print(f'\n[Results]  success={n_ok}/{args.trials} ({n_ok/args.trials:.1%})  '
          f'held={n_held}/{args.trials}  '
          f'mean_fraction_stable={frac_stable:.3f}  '
          f'elapsed={time.time()-t0:.0f}s')

    out = {
        'protocol': vars(args),
        'success_rate':         n_ok / args.trials,
        'held_rate':            n_held / args.trials,
        'mean_fraction_stable': frac_stable,
        'action_mean':          float(act_mean[0].cpu()),
        'action_std':           float(act_std[0].cpu()),
        'trials':               trial_results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'[saved] → {args.output}')


if __name__ == '__main__':
    main()
