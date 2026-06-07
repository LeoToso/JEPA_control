"""Standalone diagnostics for the minimal JEPA-control trainer (spec §8).

These are pure functions — no trainer state — so they can be called from
training loops, notebooks, or standalone analysis scripts. Each returns plain
Python floats / numpy arrays (no autograd).

    1. one_step_prediction_error   — ||f(z_t,c_t) - z_{t+1}||
    2. multistep_prediction_error  — open-loop rollout error at horizon H
    3. fixed_point_drift           — ||f(z*,0) - z*||
    4. action_sensitivity          — ||f(z,u_1) - f(z,u_2)||
    5. jacobian_spectral_radius    — rho(A) = max |eig(A)|
    6. b_matrix_norm               — ||B||_F
    7. pbh_min_singular_value      — min_lambda sigma_min([lambda I - A, B])
    8. local_lqr_eval              — closed-loop stabilization success near upright

Failure-mode -> fix mapping (spec §8):
  - B ~= 0                         -> strengthen inverse dynamics / add PBH loss
  - rho(A) < 1 but env is unstable -> add an instability-margin loss
  - LQR from (A,B) fails despite good prediction -> add local-linearization
    consistency loss
  - latent covariance collapses / ignores controllable directions -> strengthen
    dynSIG
  - spectrum / growth information is poor -> add temporal covariance consistency
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import torch

from control.jacobian import compute_jacobian_torch


@torch.no_grad()
def one_step_prediction_error(model, z_t: torch.Tensor, c_t: torch.Tensor,
                              z_tp1: torch.Tensor) -> float:
    """Mean ||f(z_t, c_t) - z_{t+1}|| over the batch.

    z_t, z_tp1 : (B, d) encoded latents (z_tp1 = encoder(frame_{t+1}))
    c_t        : (B, d_a) encoded actions
    """
    z_pred = model.predictor(z_t, c_t)
    return float((z_pred - z_tp1).norm(dim=-1).mean().cpu())


@torch.no_grad()
def multistep_prediction_error(model, z_0: torch.Tensor, c_seq: torch.Tensor,
                               z_seq: torch.Tensor) -> float:
    """Open-loop rollout error at the final horizon step.

    z_0   : (B, d)    initial latent
    c_seq : (B, H, d_a) encoded actions c_0..c_{H-1}
    z_seq : (B, H, d) encoder targets z_1..z_H (no stop-grad needed; no_grad ctx)

    Returns mean ||hat z_H - z_H|| (final-step open-loop error).
    """
    B, H, _ = c_seq.shape
    z_hat = z_0
    for t in range(H):
        z_hat = model.predictor(z_hat, c_seq[:, t])
    return float((z_hat - z_seq[:, -1]).norm(dim=-1).mean().cpu())


@torch.no_grad()
def fixed_point_drift(model, z_star: torch.Tensor) -> float:
    """||f(z*, 0) - z*|| — how far the predictor's fixed point is from z*.

    z_star : (d,) anchor latent (e.g. encoder(obs_eq))
    """
    device = z_star.device
    d_a = model.action_encoder.latent_action_dim
    z_in = z_star.unsqueeze(0)
    c_zero = torch.zeros(1, d_a, device=device, dtype=z_star.dtype)
    z_next = model.predictor(z_in, c_zero)
    return float((z_next[0] - z_star).norm().cpu())


@torch.no_grad()
def action_sensitivity(model, z: torch.Tensor, u1: torch.Tensor,
                       u2: torch.Tensor) -> float:
    """Mean ||f(z, c(u_1)) - f(z, c(u_2))|| — does the predictor react to actions?

    z       : (B, d) latents
    u1, u2  : (B, action_dim) raw actions (e.g. extremes of the action range)
    """
    c1 = model.action_encoder(u1)
    c2 = model.action_encoder(u2)
    z1 = model.predictor(z, c1)
    z2 = model.predictor(z, c2)
    return float((z1 - z2).norm(dim=-1).mean().cpu())


def jacobian_diagnostics(model, z_star: torch.Tensor, device) -> dict:
    """Local-linearization diagnostics at z*: rho(A), ||B||, PBH sigma_min.

    Returns a dict with keys: 'rho_A', 'b_norm', 'pbh_sigma_min', 'A', 'B'
    (A, B as numpy arrays for reuse, e.g. by local_lqr_eval).
    """
    z_star_np = z_star.detach().cpu().numpy()
    A, B = compute_jacobian_torch(model, z_star.detach(), device)
    A_np = A.detach().cpu().numpy()
    B_np = B.detach().cpu().numpy()

    eigs = np.linalg.eigvals(A_np)
    rho_A = float(np.max(np.abs(eigs)))
    b_norm = float(np.linalg.norm(B_np, ord='fro'))
    pbh_min = pbh_min_singular_value(A_np, B_np)

    return {'rho_A': rho_A, 'b_norm': b_norm, 'pbh_sigma_min': pbh_min,
            'A': A_np, 'B': B_np, 'eigs': eigs, 'z_star': z_star_np}


def pbh_min_singular_value(A: np.ndarray, B: np.ndarray) -> float:
    """PBH controllability test: min_{lambda in eig(A)} sigma_min([lambda I - A, B]).

    A small value indicates the pair (A, B) is close to losing controllability
    along the corresponding eigen-direction (commonly because B is too small
    or misaligned with that mode).
    """
    d = A.shape[0]
    eigs = np.linalg.eigvals(A)
    sigmas = []
    for lam in eigs:
        M = np.concatenate([lam * np.eye(d) - A, B], axis=1)
        s = np.linalg.svd(M, compute_uv=False)
        sigmas.append(s[-1])
    return float(np.min(np.real(sigmas)))


def local_lqr_eval(encoder, env, z_star: np.ndarray, A: np.ndarray, B: np.ndarray,
                   device, n_trials: int = 20, T: int = 200,
                   init_scale: float = 0.05, frame_stack: int = 1,
                   action_lb: float = -10.0, action_ub: float = 10.0,
                   seed: int = 0) -> Optional[dict]:
    """Diagnostic #8: design an LQR from (A,B) at z* and test closed-loop
    stabilization near the upright in the real environment.

    Uses LatentMPC(A, B, Q, R, horizon=1), which (by starting its Riccati
    backward-pass from the steady-state DARE solution) is effectively the
    infinite-horizon LQR gain — i.e. "local LQR" as required by the spec.

    Returns the dict produced by evaluate_stabilization_mpc (success_rate,
    mean_episode_length, mean_fraction_stable, mean_cost, vis_result), or
    None if the LQR/DARE design fails (e.g. (A,B) not stabilizable) — itself
    a diagnostic signal worth logging.
    """
    from control.mpc import LatentMPC
    from control.rollout import evaluate_stabilization_mpc

    d = A.shape[0]
    Q = np.eye(d)
    R = 0.01 * np.eye(B.shape[1])
    try:
        mpc = LatentMPC(A, B, Q, R, horizon=1,
                        action_lb=action_lb, action_ub=action_ub)
    except Exception:
        return None

    return evaluate_stabilization_mpc(
        encoder=encoder, mpc=mpc, env=env, n_trials=n_trials, T=T,
        init_scale=init_scale, device=device, z_star=z_star,
        frame_stack=frame_stack, seed=seed,
    )


def run_all_diagnostics(model, z_t: torch.Tensor, c_t: torch.Tensor, z_tp1: torch.Tensor,
                        z_0: torch.Tensor, c_seq: torch.Tensor, z_seq: torch.Tensor,
                        z_star: torch.Tensor, u_lo: torch.Tensor, u_hi: torch.Tensor,
                        device, env=None, encoder=None, run_lqr_eval: bool = False,
                        frame_stack: int = 1) -> dict:
    """Compute diagnostics 1-7 (and optionally 8) in one call; returns a flat dict.

    Diagnostic 8 (local_lqr_eval) is comparatively expensive (rolls out closed-
    loop trials in the real env), so it is gated behind `run_lqr_eval`.
    """
    out = {}
    out['one_step_error']  = one_step_prediction_error(model, z_t, c_t, z_tp1)
    out['multistep_error'] = multistep_prediction_error(model, z_0, c_seq, z_seq)
    out['fp_drift']        = fixed_point_drift(model, z_star)
    out['action_sens']     = action_sensitivity(model, z_t, u_lo, u_hi)

    jac = jacobian_diagnostics(model, z_star, device)
    out['rho_A']        = jac['rho_A']
    out['b_norm']       = jac['b_norm']
    out['pbh_sigma_min'] = jac['pbh_sigma_min']

    if run_lqr_eval and env is not None and encoder is not None:
        lqr_result = local_lqr_eval(encoder, env, jac['z_star'], jac['A'], jac['B'],
                                    device, frame_stack=frame_stack)
        out['lqr_success_rate'] = lqr_result['success_rate'] if lqr_result else None
    return out
