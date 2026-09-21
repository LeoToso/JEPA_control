#!/usr/bin/env python
"""CEM planning for CartPole using a DINO-WM checkpoint.

Cross-Entropy Method plans an H-step action sequence in latent space and
executes K steps at a time (MPC). Cost = ||mean_patch(z_H) - mean_patch(z*)||².

Usage
-----
  python experiments/cem_dinowm_cartpole.py \\
      --dino-wm-dir ~/dino_wm \\
      --ckpt /home/dz2478/dino_wm/outputs/2026-09-08/17-46-00/checkpoints/model_179.pth \\
      --data /mnt/t7shield/dinowm_cartpole \\
      --trials 10 \\
      --primitive-budget 180 \\
      --planning-horizon 10 \\
      --executed-steps 3 \\
      --cem-population 300 \\
      --cem-elites 30 \\
      --cem-iters 30 \\
      --cem-initial-variance 1.0 \\
      --success-threshold 0.7 \\
      --seed 42 \\
      --device cuda \\
      --output results/cem_dinowm_cartpole.json
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
    print('[CEM] loading checkpoint …')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    parts = {}
    for k in ['predictor', 'proprio_encoder', 'action_encoder']:
        if k not in ckpt:
            raise KeyError(f"'{k}' not found; available: {list(ckpt.keys())}")
        parts[k] = ckpt[k].to(device).eval()
    parts['encoder'] = DinoV2Encoder(
        name='dinov2_vits14', feature_key='x_norm_patchtokens').to(device).eval()
    print(f'[CEM] predictor params={sum(p.numel() for p in parts["predictor"].parameters())/1e6:.1f}M')
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
    vis_emb = parts['encoder'](preprocess_obs(obs_hwc, device))   # (1, N, 384)
    N = vis_emb.shape[1]

    _pe_dtype = next(parts['proprio_encoder'].parameters()).dtype
    prop_in  = torch.as_tensor(state[:PROP_IN_DIM].astype(np.float32),
                               device=device).to(_pe_dtype).unsqueeze(0).unsqueeze(0)
    prop_emb = parts['proprio_encoder'](prop_in)                   # (1, 1, 10)
    prop_t   = prop_emb[:, 0:1, :].expand(-1, N, -1)

    _ae_dtype = next(parts['action_encoder'].parameters()).dtype
    act_in  = torch.as_tensor(action_norm[:ACT_IN_DIM].astype(np.float32),
                              device=device).to(_ae_dtype).unsqueeze(0).unsqueeze(0)
    act_emb = parts['action_encoder'](act_in)                      # (1, 1, 10)
    act_t   = act_emb[:, 0:1, :].expand(-1, N, -1)

    return torch.cat([vis_emb, prop_t, act_t], dim=2)              # (1, N, 404)


def build_goal_ctx(parts, goal_state, num_hist, device, act_mean, act_std):
    """Encode upright equilibrium → (z_goal_ctx, z_goal_mean_np)."""
    env = make_env(seed=0)
    goal_obs, _, _ = env.reset_to_state(goal_state.copy())
    env.close()
    zero_norm = np.zeros(ACT_IN_DIM, dtype=np.float32)
    goal_frame = encode_frame(parts, goal_obs, goal_state, zero_norm, device)
    z_goal_ctx  = goal_frame.unsqueeze(1).expand(-1, num_hist, -1, -1).clone()
    z_goal_mean = z_goal_ctx[0, -1, :, :].mean(dim=0)
    return z_goal_ctx, z_goal_mean.cpu().numpy()


# ── Batched latent rollout (no grad) ──────────────────────────────────────────

@torch.no_grad()
def rollout_batch(parts: dict, z_ctx: torch.Tensor,
                  u_batch: torch.Tensor,
                  act_mean: torch.Tensor, act_std: torch.Tensor) -> torch.Tensor:
    """Roll out pop action sequences from the current context.

    z_ctx   : (1, T, N, D) – single context, broadcast to population
    u_batch : (pop, H)     – raw action values
    Returns : (pop, H, D_TOTAL) – mean-over-patches latent at each step
    """
    B, H   = u_batch.shape
    _, T, N, D = z_ctx.shape
    predictor = parts['predictor']
    ae        = parts['action_encoder']
    ae_dtype  = next(ae.parameters()).dtype

    z = z_ctx.expand(B, -1, -1, -1).clone()   # (B, T, N, D)
    zs = []
    for t in range(H):
        a_t    = u_batch[:, t].to(ae_dtype)                                 # (B,)
        a_norm = ((a_t - act_mean[0]) / act_std[0]).reshape(B, 1, 1)       # (B,1,1)
        a_emb  = ae(a_norm)                                                 # (B,1,10)
        a_tile = a_emb.expand(B, N, -1)                                     # (B,N,10)

        # Splice action into last frame of context
        z_in   = torch.cat([z[:, -1:, :, :D_VIS + D_PROP_OUT],
                             a_tile.unsqueeze(1)], dim=-1)                  # (B,1,N,D)
        z_flat = torch.cat([z[:, :-1], z_in], dim=1).reshape(B, T * N, D) # (B,T*N,D)

        z_next_flat = predictor(z_flat)                                     # (B,T*N,D)
        z_next      = z_next_flat.reshape(B, T, N, D)[:, -1]               # (B,N,D)

        zs.append(z_next.mean(dim=1))                                       # (B,D)
        z = torch.cat([z[:, 1:], z_next.unsqueeze(1)], dim=1)

    return torch.stack(zs, dim=1)   # (B, H, D)


# ── CEM planner ───────────────────────────────────────────────────────────────

class DINOWMCEMPlanner:
    def __init__(self, parts, act_mean, act_std, num_hist,
                 horizon, executed_steps, population, n_elites, n_iters,
                 initial_variance, action_lb, action_ub, device):
        self.parts     = parts
        self.act_mean  = act_mean
        self.act_std   = act_std
        self.num_hist  = num_hist
        self.horizon   = horizon
        self.executed_steps = min(executed_steps, horizon)
        self.population     = population
        self.n_elites       = min(n_elites, population)
        self.n_iters        = n_iters
        self.init_var       = initial_variance
        self.lb             = action_lb
        self.ub             = action_ub
        self.device         = device

    def plan_sequence(self, z_ctx: torch.Tensor,
                      z_goal_mean: np.ndarray) -> np.ndarray:
        """Returns (executed_steps,) raw action array."""
        H, B = self.horizon, self.population
        z_goal = torch.tensor(z_goal_mean, device=self.device)

        mu  = torch.zeros(H, device=self.device)
        var = torch.full((H,), self.init_var, device=self.device)

        for _ in range(self.n_iters):
            u = (mu + var.sqrt() * torch.randn(B, H, device=self.device)).clamp(self.lb, self.ub)
            z_preds = rollout_batch(self.parts, z_ctx, u, self.act_mean, self.act_std)
            costs   = (z_preds[:, -1] - z_goal).pow(2).sum(dim=-1)   # (B,)
            elite_idx = costs.argsort()[:self.n_elites]
            elites    = u[elite_idx]
            mu  = elites.mean(dim=0)
            var = elites.var(dim=0).clamp(min=1e-6)

        return mu[:self.executed_steps].cpu().numpy()


# ── Trial evaluation ──────────────────────────────────────────────────────────

def evaluate_trial(env, planner, parts, initial_state, goal_state,
                   z_goal_mean_np, act_mean, act_std,
                   primitive_budget, success_threshold, success_hold_steps,
                   num_hist, device):
    obs, state, _ = env.reset_to_state(initial_state.copy())
    frame_skip   = int(env.frame_skip)
    macro_budget = int(np.ceil(primitive_budget / frame_skip))

    zero_norm  = np.zeros(ACT_IN_DIM, dtype=np.float32)
    init_frame = encode_frame(parts, obs, state, zero_norm, device)
    z_ctx      = init_frame.unsqueeze(1).expand(-1, num_hist, -1, -1).clone()

    states  = [state.copy()]
    actions = []
    terminated = False

    while len(actions) < macro_budget and not terminated:
        seq      = planner.plan_sequence(z_ctx, z_goal_mean_np)
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
        description='CEM planning for CartPole with DINO-WM.')
    p.add_argument('--dino-wm-dir', default=str(Path.home() / 'dino_wm'))
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data', default='/mnt/t7shield/dinowm_cartpole',
                   help='Dataset path for action stats')
    p.add_argument('--num-hist', type=int, default=3)
    p.add_argument('--trials', type=int, default=10)
    p.add_argument('--primitive-budget', type=int, default=180)
    p.add_argument('--planning-horizon', type=int, default=10)
    p.add_argument('--executed-steps', type=int, default=3)
    p.add_argument('--cem-population', type=int, default=300)
    p.add_argument('--cem-elites', type=int, default=30)
    p.add_argument('--cem-iters', type=int, default=30)
    p.add_argument('--cem-initial-variance', type=float, default=1.0)
    p.add_argument('--success-threshold', type=float, default=0.7)
    p.add_argument('--success-hold-steps', type=int, default=10)
    p.add_argument('--eps-range', type=float, nargs=2, default=[0.01, 0.15])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', required=True)
    args = p.parse_args()

    device = torch.device(args.device)
    rng    = np.random.default_rng(args.seed)

    parts                = load_model(args.ckpt, args.dino_wm_dir, device)
    act_mean, act_std    = load_action_stats(args.data, device)

    action_lb, action_ub = -10., 10.

    planner = DINOWMCEMPlanner(
        parts=parts, act_mean=act_mean, act_std=act_std, num_hist=args.num_hist,
        horizon=args.planning_horizon, executed_steps=args.executed_steps,
        population=args.cem_population, n_elites=args.cem_elites,
        n_iters=args.cem_iters, initial_variance=args.cem_initial_variance,
        action_lb=action_lb, action_ub=action_ub, device=device)

    goal_state = np.zeros(4, dtype=np.float32)
    _, z_goal_mean_np = build_goal_ctx(parts, goal_state, args.num_hist,
                                       device, act_mean, act_std)

    print(f'[CEM] H={args.planning_horizon} K={args.executed_steps} '
          f'pop={args.cem_population} elites={args.cem_elites} '
          f'iters={args.cem_iters} var0={args.cem_initial_variance}')

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

    n_ok         = sum(r['success'] for r in trial_results)
    n_held       = sum(r['held']    for r in trial_results)
    frac_stable  = float(np.mean([r['fraction_stable'] for r in trial_results]))
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
