#!/usr/bin/env python
"""Latent LQR for CartPole using DINO-WM.

Linearises the dino-wm predictor at the upright equilibrium (all-zeros state),
solves DARE, and runs closed-loop trials: u_t = -K @ (z_t - z*).

Usage
-----
  python experiments/lqr_dinowm_cartpole.py \\
      --dino-wm-dir ~/dino_wm \\
      --ckpt /home/dz2478/dino_wm/outputs/2026-09-08/17-46-00/checkpoints/model_179.pth \\
      --data /mnt/t7shield/dinowm_cartpole \\
      --trials 10 --n-steps 300 --seed 42 \\
      --success-threshold 0.7 --success-hold-steps 10 \\
      --q-scale 1.0 --r-scale 1.0 \\
      --output results/lqr_dinowm_cartpole.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F

# ── dino-wm CartPole constants ────────────────────────────────────────────────
FRAMESKIP   = 5       # env frame skip baked into training data
D_VIS       = 384     # DINOv2 ViT-S/14 patch token dim
D_PROP_OUT  = 10      # proprio encoder output dim
D_ACT_OUT   = 10      # action encoder output dim
D_TOTAL     = D_VIS + D_PROP_OUT + D_ACT_OUT   # 404
ACT_IN_DIM  = 1       # CartPole scalar action (dino-wm frameskip=1)
PROP_IN_DIM = 4       # [x, x_dot, theta, theta_dot]
IMG_SIZE    = 112     # 8×14=112; training transform rounded to this (gives N=64 patches, T*N=192)

_NORM_MEAN = torch.tensor([0.5, 0.5, 0.5])
_NORM_STD  = torch.tensor([0.5, 0.5, 0.5])


# ── Package setup ─────────────────────────────────────────────────────────────

def setup_dinowm(dino_wm_dir: str):
    p = str(Path(dino_wm_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Environment ───────────────────────────────────────────────────────────────

def make_env(image_size: int = IMG_SIZE, seed: int = 0):
    from envs.cartpole_visual import ContinuousCartpoleVisual
    return ContinuousCartpoleVisual(
        frame_skip=FRAMESKIP, image_size=image_size, seed=seed)


# ── Action normalisation stats ────────────────────────────────────────────────

def load_action_stats(data_path: str, device: torch.device):
    """Compute action mean/std from the saved .pth dataset (matches training)."""
    p = Path(data_path)
    actions     = torch.load(p / 'actions.pth').float()     # (N, T_max, 1)
    seq_lengths = torch.load(p / 'seq_lengths.pth')         # (N,)
    all_acts = []
    for i in range(len(seq_lengths)):
        T = int(seq_lengths[i])
        all_acts.append(actions[i, :T])
    all_acts = torch.vstack(all_acts)                        # (total_steps, 1)
    mean = all_acts.mean(dim=0).to(device)                   # (1,)
    std  = all_acts.std(dim=0).to(device)                    # (1,)
    std  = std.clamp(min=1e-6)
    print(f'[action stats]  mean={mean.item():.4f}  std={std.item():.4f}')
    return mean, std


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, dino_wm_dir: str, device: torch.device) -> dict:
    setup_dinowm(dino_wm_dir)
    from models.dino import DinoV2Encoder
    print('[DINO-WM LQR] loading checkpoint …')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    parts = {}
    for k in ['predictor', 'proprio_encoder', 'action_encoder']:
        if k not in ckpt:
            raise KeyError(f"'{k}' missing — available: {list(ckpt.keys())}")
        parts[k] = ckpt[k].to(device).eval()
    encoder = DinoV2Encoder(name='dinov2_vits14', feature_key='x_norm_patchtokens')
    parts['encoder'] = encoder.to(device).eval()
    n_pred = sum(p.numel() for p in parts['predictor'].parameters())
    print(f'[DINO-WM LQR] predictor params={n_pred/1e6:.1f}M')
    # Infer expected (T, N) from positional embedding to catch img_size mismatches early
    if hasattr(parts['predictor'], 'pos_embedding'):
        pos_seq = parts['predictor'].pos_embedding.shape[1]
        print(f'[DINO-WM LQR] predictor pos_embedding sequence length = {pos_seq}  '
              f'(expect T×N_patches; e.g. T=3 → N={pos_seq//3} → img_size={int((pos_seq//3)**0.5)*14}px)')
    return parts


# ── Image preprocessing ───────────────────────────────────────────────────────

def preprocess_obs(obs_hwc: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(obs_hwc.copy()).float().permute(2, 0, 1) / 255.0   # (C,H,W)
    H, W = x.shape[1], x.shape[2]
    if H != IMG_SIZE or W != IMG_SIZE:
        x = F.interpolate(x.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
                          mode='bilinear', align_corners=False).squeeze(0)
    x = (x - _NORM_MEAN[:, None, None]) / _NORM_STD[:, None, None]
    return x.unsqueeze(0).to(device)


# ── Frame encoding ────────────────────────────────────────────────────────────

def encode_frame(parts: dict, obs_hwc: np.ndarray, state: np.ndarray,
                 action_norm: np.ndarray, device: torch.device) -> torch.Tensor:
    """Encode one (obs, state, normalised_action) → (1, N_patches, D_TOTAL)."""
    with torch.no_grad():
        vis_emb = parts['encoder'](preprocess_obs(obs_hwc, device))  # (1, N, 384)
    N = vis_emb.shape[1]

    _pe_dtype = next(parts['proprio_encoder'].parameters()).dtype
    prop_in = torch.as_tensor(
        state[:PROP_IN_DIM].astype(np.float32), device=device
    ).to(_pe_dtype).unsqueeze(0).unsqueeze(0)                         # (1, 1, 4)
    with torch.no_grad():
        prop_emb = parts['proprio_encoder'](prop_in)                  # (1, 1, 10)
    prop_tiled = prop_emb[:, 0:1, :].expand(-1, N, -1)

    _ae_dtype = next(parts['action_encoder'].parameters()).dtype
    act_in = torch.as_tensor(
        action_norm[:ACT_IN_DIM].astype(np.float32), device=device
    ).to(_ae_dtype).unsqueeze(0).unsqueeze(0)                         # (1, 1, 1)
    with torch.no_grad():
        act_emb = parts['action_encoder'](act_in)                     # (1, 1, 10)
    act_tiled = act_emb[:, 0:1, :].expand(-1, N, -1)

    return torch.cat([vis_emb, prop_tiled, act_tiled], dim=2)         # (1, N, 404)


# ── Goal context ──────────────────────────────────────────────────────────────

def build_goal_context(parts: dict, env, num_hist: int, device: torch.device,
                       act_mean: torch.Tensor, act_std: torch.Tensor):
    """Return (z_goal_ctx, z_goal_mean) for the upright equilibrium."""
    goal_state = np.zeros(4, dtype=np.float32)
    goal_obs, _, _ = env.reset_to_state(goal_state)
    zero_act_norm = np.zeros(ACT_IN_DIM, dtype=np.float32)  # normalised zero action
    goal_frame = encode_frame(parts, goal_obs, goal_state, zero_act_norm, device)
    z_goal_ctx  = goal_frame.unsqueeze(1).expand(-1, num_hist, -1, -1).clone()
    z_goal_mean = z_goal_ctx[0, -1, :, :].mean(dim=0)                 # (D_TOTAL,)
    return z_goal_ctx, z_goal_mean


# ── Batched FD Jacobians ──────────────────────────────────────────────────────

@torch.no_grad()
def jacobians_fd(parts: dict, z_ctx: torch.Tensor, num_hist: int,
                 eps_z: float = 1e-3, eps_u: float = 1e-2,
                 chunk_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """A (D_TOTAL × D_TOTAL) and B (D_TOTAL × ACT_IN_DIM) via central FD."""
    predictor = parts['predictor']
    _ae       = parts['action_encoder']
    _ae_dtype = next(_ae.parameters()).dtype
    device    = z_ctx.device

    z_base = z_ctx[0]        # (T, N, D)
    T, N, D = z_base.shape

    def _predict_mean(zb: torch.Tensor) -> torch.Tensor:
        B_ = zb.shape[0]
        z_flat = zb.reshape(B_, T * N, D)
        z_pred = predictor(z_flat).reshape(B_, T, N, D)
        return z_pred[:, -1, :, :].mean(dim=1)    # (B, D)

    # ── A matrix ──────────────────────────────────────────────────────────────
    z_next_parts: list[torch.Tensor] = []
    for start in range(0, 2 * D, chunk_size):
        end = min(start + chunk_size, 2 * D)
        B_chunk = end - start
        chunk = z_base.unsqueeze(0).expand(B_chunk, -1, -1, -1).clone()
        for local_i in range(B_chunk):
            global_i = start + local_i
            feat = global_i // 2
            sign = 1.0 if (global_i % 2 == 0) else -1.0
            chunk[local_i, -1, :, feat] += sign * eps_z
        z_next_parts.append(_predict_mean(chunk))
    z_next_A = torch.cat(z_next_parts, dim=0)    # (2D, D)
    z_pos = z_next_A[0::2]
    z_neg = z_next_A[1::2]
    A = (z_pos - z_neg).T / (2.0 * eps_z)        # (D, D)

    # ── B matrix ──────────────────────────────────────────────────────────────
    u_batch = torch.zeros(2 * ACT_IN_DIM, ACT_IN_DIM, device=device, dtype=_ae_dtype)
    for k in range(ACT_IN_DIM):
        u_batch[2 * k,     k] += eps_u
        u_batch[2 * k + 1, k] -= eps_u

    act_embs = _ae(u_batch.unsqueeze(1))                          # (2, 1, 10)
    act_tiled = act_embs[:, 0:1, :].expand(-1, N, -1)            # (2, N, 10)

    z_batch_B = z_base.unsqueeze(0).expand(2 * ACT_IN_DIM, -1, -1, -1).clone()
    z_batch_B[:, -1, :, D_VIS + D_PROP_OUT:] = act_tiled
    z_next_B = _predict_mean(z_batch_B)                           # (2, D)

    u_pos = z_next_B[0::2]
    u_neg = z_next_B[1::2]
    B = (u_pos - u_neg).T / (2.0 * eps_u)                        # (D, 1)

    return A.cpu().numpy(), B.cpu().numpy()


# ── DARE ─────────────────────────────────────────────────────────────────────

def solve_dare(A, B, Q, R):
    try:
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        return K
    except Exception as e:
        print(f'[DARE] failed: {e}')
        return None


# ── Trial loop ────────────────────────────────────────────────────────────────

def lqr_trial(parts: dict, K: np.ndarray,
              z_goal_mean_np: np.ndarray,
              env, num_hist: int,
              n_steps: int, success_threshold: float, success_hold_steps: int,
              act_mean: torch.Tensor, act_std: torch.Tensor,
              action_range: tuple[float, float],
              device: torch.device,
              initial_state: np.ndarray | None = None) -> dict:
    """One closed-loop LQR trial."""
    if initial_state is not None:
        obs, state, _ = env.reset_to_state(initial_state.copy())
    else:
        obs, state, _ = env.reset()

    zero_act_norm = np.zeros(ACT_IN_DIM, dtype=np.float32)
    init_frame = encode_frame(parts, obs, state, zero_act_norm, device)
    z_ctx = init_frame.unsqueeze(1).expand(-1, num_hist, -1, -1).clone()

    goal_state = np.zeros(4, dtype=np.float32)
    states  = [state.copy()]
    actions = []
    terminated = False

    for step in range(n_steps):
        with torch.no_grad():
            z_mean = z_ctx[0, -1, :, :].mean(dim=0).cpu().numpy()   # (D_TOTAL,)

        # u_norm = -K @ (z - z*), then denormalise to real action
        u_norm = float(-(K @ (z_mean - z_goal_mean_np)).item())
        u_real = u_norm * float(act_std[0].cpu()) + float(act_mean[0].cpu())
        u_real = float(np.clip(u_real, action_range[0], action_range[1]))

        obs, state, _, done, _ = env.step(u_real)
        states.append(state.copy())
        actions.append(u_real)

        if done:
            terminated = True

        # Update context: roll left, append new frame
        u_norm_arr = np.array([(u_real - float(act_mean[0].cpu()))
                                / float(act_std[0].cpu())], dtype=np.float32)
        new_frame = encode_frame(parts, obs, state, u_norm_arr, device)
        z_ctx = torch.cat([z_ctx[:, 1:], new_frame.unsqueeze(1)], dim=1)

    states = np.array(states)
    errors = np.linalg.norm(states - goal_state[None], axis=1)
    stable_mask = errors < success_threshold
    tail = stable_mask[-success_hold_steps:]
    held = bool(len(tail) == success_hold_steps and tail.all())
    success = bool(stable_mask[-1])

    settling = None
    run = 0
    for i in range(len(stable_mask) - 1, -1, -1):
        if stable_mask[i]:
            run += 1
        else:
            break
    if run >= success_hold_steps:
        settling = len(stable_mask) - run

    return {
        'success': success,
        'held': held,
        'terminated': terminated,
        'final_error': float(errors[-1]),
        'max_error': float(np.max(errors)),
        'fraction_stable': float(np.mean(stable_mask)),
        'settling_step': settling,
        'states': states.tolist(),
        'actions': actions,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Latent LQR for CartPole using DINO-WM.')
    p.add_argument('--dino-wm-dir', default=str(Path.home() / 'dino_wm'),
                   help='Path to dino-wm repo root')
    p.add_argument('--ckpt', required=True,
                   help='Path to dino-wm checkpoint (.pth)')
    p.add_argument('--data', default='/mnt/t7shield/dinowm_cartpole',
                   help='Path to dinowm_cartpole dataset (for action stats)')
    p.add_argument('--num-hist', type=int, default=3)
    p.add_argument('--trials', type=int, default=10)
    p.add_argument('--n-steps', type=int, default=300,
                   help='Max model steps per trial (each = FRAMESKIP=5 env steps)')
    p.add_argument('--success-threshold', type=float, default=0.7,
                   help='State norm threshold for success')
    p.add_argument('--success-hold-steps', type=int, default=10)
    p.add_argument('--eps-range', type=float, nargs=2, default=[0.01, 0.15],
                   help='Initial state perturbation range')
    p.add_argument('--q-scale', type=float, default=1.0)
    p.add_argument('--r-scale', type=float, default=1.0)
    p.add_argument('--eps-z', type=float, default=1e-3)
    p.add_argument('--eps-u', type=float, default=1e-2)
    p.add_argument('--chunk-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', required=True)
    args = p.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    # Load model and action stats
    parts    = load_model(args.ckpt, args.dino_wm_dir, device)
    act_mean, act_std = load_action_stats(args.data, device)

    # Build environment and goal context
    env = make_env(seed=args.seed)
    action_range = (env.action_low, env.action_high)
    z_goal_ctx, z_goal_mean = build_goal_context(
        parts, env, args.num_hist, device, act_mean, act_std)
    z_goal_np = z_goal_mean.cpu().numpy()

    # Compute Jacobians and LQR gain at equilibrium
    print('[LQR] computing Jacobians at equilibrium …')
    t0 = time.time()
    A_np, B_np = jacobians_fd(
        parts, z_goal_ctx, args.num_hist,
        eps_z=args.eps_z, eps_u=args.eps_u, chunk_size=args.chunk_size)
    print(f'[LQR] Jacobians done in {time.time()-t0:.1f}s  '
          f'A={A_np.shape}  B={B_np.shape}')

    d = D_TOTAL
    Q = args.q_scale * np.eye(d)
    R = args.r_scale * np.eye(ACT_IN_DIM)
    K = solve_dare(A_np, B_np, Q, R)
    if K is None:
        print('[LQR] DARE failed — cannot run trials.')
        return

    # Spectral radius check
    A_cl = A_np - B_np @ K
    eigs = np.abs(np.linalg.eigvals(A_cl))
    print(f'[LQR] closed-loop spectral radius: {eigs.max():.4f}  '
          f'(stable if < 1.0)')

    # Generate initial states
    lo, hi = args.eps_range
    initial_states = []
    for _ in range(args.trials):
        s = rng.uniform(-hi, hi, size=4).astype(np.float64)
        s[2] = rng.uniform(-hi, hi)
        s[3] = rng.uniform(-hi, hi)
        initial_states.append(s)

    # Run trials
    trial_results = []
    for i, init_s in enumerate(initial_states):
        print(f'  trial {i+1}/{args.trials} …', end=' ', flush=True)
        r = lqr_trial(parts, K, z_goal_np, env, args.num_hist,
                      args.n_steps, args.success_threshold, args.success_hold_steps,
                      act_mean, act_std, action_range, device, initial_state=init_s)
        trial_results.append(r)
        print(f"success={r['success']}  final_err={r['final_error']:.3f}")

    env.close()

    n_ok = sum(r['success'] for r in trial_results)
    n_held = sum(r['held'] for r in trial_results)
    frac_stable = float(np.mean([r['fraction_stable'] for r in trial_results]))
    settling = [r['settling_step'] for r in trial_results if r['settling_step'] is not None]

    msg = (f'\n[Results]  success={n_ok}/{args.trials} ({n_ok/args.trials:.1%})  '
           f'held={n_held}/{args.trials}  '
           f'mean_fraction_stable={frac_stable:.3f}')
    if settling:
        msg += f'  avg_settling={np.mean(settling):.1f} steps'
    print(msg)

    output = {
        'protocol': vars(args),
        'spectral_radius': float(eigs.max()),
        'success_rate': n_ok / args.trials,
        'held_rate': n_held / args.trials,
        'mean_fraction_stable': frac_stable,
        'mean_settling': float(np.mean(settling)) if settling else None,
        'action_mean': float(act_mean[0].cpu()),
        'action_std': float(act_std[0].cpu()),
        'trials': trial_results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'[saved] → {args.output}')


if __name__ == '__main__':
    main()
