#!/usr/bin/env python
"""Latent LQR in DINO-WM's mean-pooled patch-token space for PointMaze.

Uses the SAME seed scheme as gt_lqr_gymnasium_pointmaze.py and the CEM oracle:
  trial i → start_seed = seed+99*i+1 ,  goal_seed = seed+99*i+2

--n-steps is now MODEL steps (each model step = FRAMESKIP=5 env steps).
  --n-steps 200 matches  gt_lqr_gymnasium_pointmaze.py --n-steps 200 --frame-skip 5.

Two K modes:
  default        Per-trial K — Jacobians re-computed at each trial's goal.
  --static-k     Compute K once at trial-0 goal, reuse for all trials (faster).

Relinearisation during a trial (--relinearize-every k):
  0  Static K within each trial.
  k  Re-linearise at current z_ctx every k model steps.

Usage
-----
  MUJOCO_GL=egl python experiments/lqr_dinowm_pointmaze.py \\
      --dino-wm-dir ~/dino_wm \\
      --ckpt /path/to/model_latest.pth \\
      --trials 50 --n-steps 200 --seed 0 \\
      --output results/lqr_dinowm_pointmaze.json
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

# ── DINO-WM constants (from train.yaml) ──────────────────────────────────────
FRAMESKIP   = 5
D_VIS       = 384
D_PROP_OUT  = 10
D_ACT_OUT   = 10
D_TOTAL     = D_VIS + D_PROP_OUT + D_ACT_OUT   # 404
ACT_IN_DIM  = FRAMESKIP * 2                     # 10
PROP_IN_DIM = 4


# ── Package setup ─────────────────────────────────────────────────────────────

def setup_dinowm(dino_wm_dir: str):
    p = str(Path(dino_wm_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)

_JEPA_ROOT = str(Path(__file__).resolve().parent.parent)
if _JEPA_ROOT not in sys.path:
    sys.path.insert(0, _JEPA_ROOT)


# ── Environment ───────────────────────────────────────────────────────────────

def make_env(seed: int = 0, image_size: int = 196):
    from envs.pointmaze_visual import PointMazeVisual
    env_cfg = {
        'environment': {
            'maze_map':    'U',
            'image_size':  image_size,
            'action_scale': 1.0,
        }
    }
    return PointMazeVisual(env_cfg, seed=seed)


def get_fixed_goal_xy(env) -> np.ndarray:
    obs_dict, _ = env._env.reset(seed=0)
    return obs_dict['desired_goal'].astype(np.float32)


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
    return parts


# ── Image preprocessing ───────────────────────────────────────────────────────

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


# ── Frame encoding ────────────────────────────────────────────────────────────

def encode_frame(parts: dict, obs_hwc: np.ndarray, state: np.ndarray,
                 action_10d: np.ndarray | None,
                 device: torch.device) -> torch.Tensor:
    """Encode one (obs, state, action) triple → (1, N_patches, D_TOTAL)."""
    if action_10d is None:
        action_10d = np.zeros(ACT_IN_DIM, dtype=np.float32)
    with torch.no_grad():
        vis_emb = parts['encoder'](preprocess_obs(obs_hwc, device))   # (1, N, 384)
    N = vis_emb.shape[1]
    _pe_dtype = next(parts['proprio_encoder'].parameters()).dtype
    prop_in = torch.as_tensor(
        state[:PROP_IN_DIM].astype(np.float32), device=device
    ).to(_pe_dtype).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        prop_emb = parts['proprio_encoder'](prop_in)   # (1, 1, 10)
    prop_tiled = prop_emb[:, 0:1, :].expand(-1, N, -1)
    _ae_dtype = next(parts['action_encoder'].parameters()).dtype
    act_in = torch.as_tensor(
        action_10d[:ACT_IN_DIM].astype(np.float32), device=device
    ).to(_ae_dtype).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        act_emb = parts['action_encoder'](act_in)      # (1, 1, 10)
    act_tiled = act_emb[:, 0:1, :].expand(-1, N, -1)
    return torch.cat([vis_emb, prop_tiled, act_tiled], dim=2)  # (1, N, D_TOTAL)


# ── Goal context ──────────────────────────────────────────────────────────────

def build_goal_context(parts: dict, env, goal_xy: np.ndarray,
                       num_hist: int, device: torch.device):
    """Return (z_goal_ctx, z_goal_mean).

    z_goal_ctx  : (1, num_hist, N, D_TOTAL) — num_hist copies of goal frame
    z_goal_mean : (D_TOTAL,) — mean over patches of the last goal frame
    """
    goal_state = np.array([goal_xy[0], goal_xy[1], 0., 0.], dtype=np.float32)
    goal_obs, _, _ = env.reset_to_state(goal_state)
    zero_act = np.zeros(ACT_IN_DIM, dtype=np.float32)
    goal_frame = encode_frame(parts, goal_obs, goal_state, zero_act, device)   # (1, N, D)
    z_goal_ctx  = goal_frame.unsqueeze(1).expand(-1, num_hist, -1, -1).clone()  # (1,T,N,D)
    z_goal_mean = z_goal_ctx[0, -1, :, :].mean(dim=0)                           # (D,)
    return z_goal_ctx, z_goal_mean


# ── Batched FD Jacobians ──────────────────────────────────────────────────────

@torch.no_grad()
def jacobians_fd(parts: dict, z_ctx: torch.Tensor, num_hist: int,
                 eps_z: float = 1e-3, eps_u: float = 1e-2,
                 chunk_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """A (D_TOTAL × D_TOTAL) and B (D_TOTAL × ACT_IN_DIM) via central FD.

    Perturbs the last frame's features uniformly across all N patches, which
    is consistent with the mean-pooling state representation.

    Chunking prevents OOM: full 2*D_TOTAL=808 batch ≈ 768 MB; chunk_size=64
    uses ≈60 MB per chunk and ~13 predictor calls for A.
    """
    predictor = parts['predictor']
    _ae       = parts['action_encoder']
    _ae_dtype = next(_ae.parameters()).dtype
    device    = z_ctx.device

    z_base = z_ctx[0]       # (T, N, D)
    T, N, D = z_base.shape

    def _predict_mean(zb: torch.Tensor) -> torch.Tensor:
        """zb: (B, T, N, D) → mean of predicted last frame (B, D)."""
        B_ = zb.shape[0]
        z_flat = zb.reshape(B_, T * N, D)
        z_pred = predictor(z_flat).reshape(B_, T, N, D)
        return z_pred[:, -1, :, :].mean(dim=1)   # (B, D)

    # ── A matrix: perturb each of D features in the last frame ────────────────
    z_next_parts: list[torch.Tensor] = []
    for start in range(0, 2 * D, chunk_size):
        end = min(start + chunk_size, 2 * D)
        B_chunk = end - start
        chunk = z_base.unsqueeze(0).expand(B_chunk, -1, -1, -1).clone()   # (B_c,T,N,D)
        for local_i in range(B_chunk):
            global_i = start + local_i
            feat = global_i // 2
            sign = 1.0 if (global_i % 2 == 0) else -1.0
            chunk[local_i, -1, :, feat] += sign * eps_z          # uniform across patches
        z_next_parts.append(_predict_mean(chunk))
    z_next_A = torch.cat(z_next_parts, dim=0)   # (2D, D)

    # A[:, j] = (f(z + eps*e_j) - f(z - eps*e_j)) / (2*eps)
    # z_next_A[2j]   = f(z + eps*e_j), z_next_A[2j+1] = f(z - eps*e_j)
    z_pos = z_next_A[0::2]   # (D, D) — one row per input feature (positive)
    z_neg = z_next_A[1::2]   # (D, D) — one row per input feature (negative)
    A = (z_pos - z_neg).T / (2.0 * eps_z)   # (D, D)

    # ── B matrix: perturb each of ACT_IN_DIM action dims ─────────────────────
    # Build 2*ACT_IN_DIM perturbed actions
    u_batch = torch.zeros(2 * ACT_IN_DIM, ACT_IN_DIM, device=device, dtype=_ae_dtype)
    for k in range(ACT_IN_DIM):
        u_batch[2 * k,     k] += eps_u
        u_batch[2 * k + 1, k] -= eps_u

    act_embs = _ae(u_batch.unsqueeze(1))                          # (2*M, 1, 10)
    act_tiled = act_embs[:, 0:1, :].expand(-1, N, -1)            # (2*M, N, 10)

    z_batch_B = z_base.unsqueeze(0).expand(2 * ACT_IN_DIM, -1, -1, -1).clone()
    z_batch_B[:, -1, :, D_VIS + D_PROP_OUT:] = act_tiled         # update action features
    z_next_B = _predict_mean(z_batch_B)                           # (2*M, D)

    u_pos = z_next_B[0::2]   # (M, D) — positive action perturbation
    u_neg = z_next_B[1::2]   # (M, D) — negative action perturbation
    B = (u_pos - u_neg).T / (2.0 * eps_u)   # (D, M)

    return A.cpu().numpy(), B.cpu().numpy()


# ── DARE ─────────────────────────────────────────────────────────────────────

def solve_dare(A: np.ndarray, B: np.ndarray,
               Q: np.ndarray, R: np.ndarray) -> np.ndarray | None:
    try:
        P = scipy.linalg.solve_discrete_are(A, B, Q, R)
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        return K
    except Exception:
        return None


# ── Trial loop ────────────────────────────────────────────────────────────────

def lqr_trial(parts: dict, K_init: np.ndarray | None,
              z_goal_mean_np: np.ndarray,
              env, goal_xy: np.ndarray, num_hist: int,
              n_steps: int, success_threshold: float, success_hold_steps: int,
              device: torch.device,
              init_state: np.ndarray | None = None,
              relinearize_every: int = 0,
              Q: np.ndarray | None = None,
              R: np.ndarray | None = None,
              eps_z: float = 1e-3, eps_u: float = 1e-2,
              chunk_size: int = 64, verbose: bool = False) -> dict:
    """One closed-loop trial.

    n_steps         : MODEL steps (each model step executes FRAMESKIP env steps)
    init_state      : if given, reset env to this state; otherwise env.reset()
    relinearize_every=0: static K (K_init fixed throughout)
    relinearize_every=k: re-linearise at current z_ctx every k model steps
    """
    if init_state is not None:
        obs, state, _ = env.reset_to_state(init_state.copy())
    else:
        obs, state, _ = env.reset()
    zero_act = np.zeros(ACT_IN_DIM, dtype=np.float32)

    # Bootstrap context: num_hist copies of the initial observation
    init_frame = encode_frame(parts, obs, state, zero_act, device)        # (1, N, D)
    z_ctx = init_frame.unsqueeze(1).expand(-1, num_hist, -1, -1).clone()  # (1,T,N,D)

    K = K_init
    model_step = 0
    done = False
    states = [state.copy()]
    consec_stable = 0
    relinearize_failures = 0

    while model_step < n_steps and not done:
        # Possibly re-linearise at current z_ctx
        if relinearize_every > 0 and model_step % relinearize_every == 0:
            A_z, B_z = jacobians_fd(parts, z_ctx, num_hist,
                                     eps_z=eps_z, eps_u=eps_u,
                                     chunk_size=chunk_size)
            K_new = solve_dare(A_z, B_z, Q, R)
            if K_new is not None:
                K = K_new
            else:
                relinearize_failures += 1
                if verbose:
                    print(f'  model_step {model_step}: DARE failed, using last K')

        # LQR control action
        if K is None:
            u_10d = np.zeros(ACT_IN_DIM)
        else:
            z_mean = z_ctx[0, -1, :, :].mean(dim=0).cpu().numpy()   # (D,)
            u_10d  = -(K @ (z_mean - z_goal_mean_np))
            u_10d  = np.clip(u_10d, -1.0, 1.0)

        # Execute FRAMESKIP env steps; only render on the last sub-step
        obs_after, state_after = obs, state
        for sub in range(FRAMESKIP):
            a_2d = np.clip(u_10d[sub * 2:(sub + 1) * 2], -1.0, 1.0)
            is_last_sub = (sub == FRAMESKIP - 1)
            if is_last_sub:
                obs_after, state_after, _, _env_done, _ = env.step(a_2d)
                done = _env_done
            else:
                state_after, _, _env_done, _ = env.step_no_render(a_2d)
                obs_after = obs       # keep previous rendered frame
                done = _env_done
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

    states = np.array(states)
    xy_err = np.linalg.norm(states[:, :2] - goal_xy[None], axis=1)
    stable = xy_err < success_threshold
    tail   = stable[-success_hold_steps:]
    held   = bool(len(tail) == success_hold_steps and tail.all())
    success = bool(stable[-1])

    run = 0
    for i in range(len(stable) - 1, -1, -1):
        if stable[i]: run += 1
        else: break
    settling = len(stable) - run if run >= success_hold_steps else None

    return {
        'success':               success,
        'held':                  held,
        'final_error':           float(xy_err[-1]),
        'max_error':             float(np.max(xy_err)),
        'fraction_stable':       float(np.mean(stable)),
        'settling_step':         settling,
        'relinearize_failures':  relinearize_failures,
        'states':  states.tolist(),
        'goal_xy': goal_xy.tolist(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='DINO-WM latent LQR for PointMaze — same seed scheme as oracle.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dino-wm-dir',        required=True,
                   help='Path to cloned gaoyuezhou/dino_wm repo')
    p.add_argument('--ckpt',               required=True,
                   help='Path to DINO-WM PointMaze checkpoint (model_latest.pth)')
    p.add_argument('--trials',             type=int,   default=50)
    p.add_argument('--n-steps',            type=int,   default=200,
                   help='Model steps per trial (each = FRAMESKIP=5 env steps). '
                        '200 matches gt_lqr_gymnasium_pointmaze --n-steps 200 --frame-skip 5.')
    p.add_argument('--num-hist',           type=int,   default=3)
    p.add_argument('--success-threshold',  type=float, default=0.5)
    p.add_argument('--success-hold-steps', type=int,   default=1)
    p.add_argument('--q-scale',            type=float, default=1.0)
    p.add_argument('--r-scale',            type=float, default=1.0)
    p.add_argument('--relinearize-every',  type=int,   default=0,
                   help='Re-linearise within trial every k model steps. 0 = static.')
    p.add_argument('--static-k',           action='store_true', default=False,
                   help='Compute K once at trial-0 goal and reuse. '
                        'Default: recompute K per trial (goal differs each trial).')
    p.add_argument('--eps-z',              type=float, default=1e-3)
    p.add_argument('--eps-u',              type=float, default=1e-2)
    p.add_argument('--fd-chunk',           type=int,   default=64,
                   help='Predictor batch size per FD chunk (reduce if OOM).')
    p.add_argument('--seed',               type=int,   default=0,
                   help='Base seed: trial i uses start=seed+99*i+1, goal=seed+99*i+2')
    p.add_argument('--device',             default='cuda')
    p.add_argument('--output',             required=True)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'[DINO-WM LQR] device={device}')
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device.index or 0)
        print(f'[DINO-WM LQR] GPU: {props.name}  VRAM: {props.total_memory/1e9:.1f} GB')

    print(f'[DINO-WM LQR] seed scheme: start=seed+99*i+1, goal=seed+99*i+2')
    print(f'[DINO-WM LQR] n_steps={args.n_steps} model steps  '
          f'(={args.n_steps * FRAMESKIP} env steps)  '
          f'hold={args.success_hold_steps}  static_k={args.static_k}')

    parts = load_model(args.ckpt, args.dino_wm_dir, device)
    env   = make_env(seed=args.seed, image_size=196)

    d = D_TOTAL
    Q = args.q_scale * np.eye(d)
    R = args.r_scale * np.eye(ACT_IN_DIM)

    mode_str = ('static' if args.relinearize_every == 0
                else f'receding-horizon (every {args.relinearize_every} model steps)')

    K_cached = None
    trials, n_success, n_dare_fail = [], 0, 0

    for i in range(args.trials):
        start_seed = args.seed + 99 * i + 1
        goal_seed  = args.seed + 99 * i + 2

        # Sample start and goal positions via seeded gymnasium resets
        obs_dict, _ = env._env.reset(seed=int(start_seed))
        start_state  = obs_dict['observation'][:4].astype(np.float32)

        obs_dict, _ = env._env.reset(seed=int(goal_seed))
        goal_xy = obs_dict['observation'][:2].astype(np.float32)

        # Encode goal in DINO-WM latent space
        z_goal_ctx, z_goal_mean = build_goal_context(
            parts, env, goal_xy, args.num_hist, device)
        z_goal_mean_np = z_goal_mean.cpu().numpy()

        # Compute K (per-trial or cached)
        if K_cached is None or not args.static_k:
            print(f'[DINO-WM LQR] trial {i:03d}  Jacobians at '
                  f'goal=[{goal_xy[0]:.2f},{goal_xy[1]:.2f}] …', end=' ', flush=True)
            t_jac = time.time()
            A_z, B_z = jacobians_fd(parts, z_goal_ctx, args.num_hist,
                                     eps_z=args.eps_z, eps_u=args.eps_u,
                                     chunk_size=args.fd_chunk)
            t_jac = time.time() - t_jac
            K = solve_dare(A_z, B_z, Q, R)
            if K is None:
                print(f'DARE failed  ({t_jac:.1f}s)')
                n_dare_fail += 1
                trials.append({'success': False, 'held': False,
                               'final_error': float('nan'), 'max_error': float('nan'),
                               'fraction_stable': 0., 'settling_step': None,
                               'relinearize_failures': 0})
                continue
            rho_cl = float(np.max(np.abs(np.linalg.eigvals(A_z - B_z @ K))))
            print(f'ρ(A-BK)={rho_cl:.4f}  ({t_jac:.1f}s)')
            if args.static_k:
                K_cached = K
        else:
            K = K_cached

        t_trial = time.time()
        row = lqr_trial(
            parts=parts,
            K_init=K,
            z_goal_mean_np=z_goal_mean_np,
            env=env,
            goal_xy=goal_xy,
            num_hist=args.num_hist,
            n_steps=args.n_steps,
            success_threshold=args.success_threshold,
            success_hold_steps=args.success_hold_steps,
            device=device,
            init_state=start_state,
            relinearize_every=args.relinearize_every,
            Q=Q, R=R,
            eps_z=args.eps_z, eps_u=args.eps_u,
            chunk_size=args.fd_chunk,
        )
        t_trial = time.time() - t_trial
        n_success += int(row['success'])
        print(f'[DINO-WM LQR] trial {i:03d}  success={row["success"]}  '
              f'err={row["final_error"]:.4f}  '
              f'goal=[{goal_xy[0]:.3f},{goal_xy[1]:.3f}]  '
              f'start=[{start_state[0]:.3f},{start_state[1]:.3f}]  '
              f't={t_trial:.1f}s')
        trials.append(row)

    sr = n_success / max(len(trials), 1)
    print(f'\n[DINO-WM LQR] success rate: {sr:.1%}  ({n_success}/{len(trials)})  '
          f'DARE_fails={n_dare_fail}')
    env.close()

    results = {
        'protocol': vars(args),
        'mode': mode_str,
        'models': [{
            'label':        'DINO-WM LQR',
            'mode':         mode_str,
            'success_rate': sr,
            'n_success':    n_success,
            'n_trials':     len(trials),
            'n_dare_fail':  n_dare_fail,
            'trials':       trials,
        }],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[done] {out}')


if __name__ == '__main__':
    main()

