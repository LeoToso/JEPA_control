"""Design a linear controller from the *learned* latent dynamics and test it
in closed loop on the *original* (ground-truth) system.

This directly mirrors the real cartpole-JEPA evaluation protocol described
in the notes: extract a latent linear model, run LQR on it, and report
success rate / mean fraction stable / final-state distance from a battery of
closed-loop trials on the true environment.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.linalg import solve_discrete_are

from .systems import LTISystem


def extract_latent_system(predictor):
    return predictor.matrices()


def pbh_uncontrollable_modes(A: np.ndarray, B: np.ndarray, tol: float = 1e-6):
    n = A.shape[0]
    bad = []
    for lam in np.linalg.eigvals(A):
        M = np.hstack([A - lam * np.eye(n), B])
        if np.linalg.matrix_rank(M, tol=tol) < n:
            bad.append(lam)
    return bad


def is_stabilizable(A: np.ndarray, B: np.ndarray, tol: float = 1e-6) -> bool:
    return all(abs(lam) < 1.0 - 1e-9 for lam in pbh_uncontrollable_modes(A, B, tol))


def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray | None = None, R: np.ndarray | None = None):
    n, m = A.shape[0], B.shape[1]
    Q = np.eye(n) if Q is None else Q
    R = np.eye(m) if R is None else R
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(B.T @ P @ B + R, B.T @ P @ A)
    return K, P


def design_latent_controller(predictor, q_scale: float = 1.0, r_scale: float = 1.0) -> dict:
    """Attempt to design a stabilizing LQR gain purely from the learned
    latent (A_z, B_z). Reports the PBH stabilizability diagnosis explicitly:
    if an unstable latent eigenvalue is uncontrollable, no linear state
    feedback can stabilize it and K_z is left as None."""
    A_z, B_z = extract_latent_system(predictor)
    d, m = A_z.shape[0], B_z.shape[1]
    result = {
        "A_z": A_z,
        "B_z": B_z,
        "latent_eigvals": np.linalg.eigvals(A_z),
        "latent_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(A_z)))),
    }
    uncontrollable = pbh_uncontrollable_modes(A_z, B_z)
    result["uncontrollable_unstable_modes"] = [lam for lam in uncontrollable if abs(lam) >= 1.0 - 1e-9]
    result["stabilizable"] = len(result["uncontrollable_unstable_modes"]) == 0
    if not result["stabilizable"]:
        result["K_z"] = None
        return result
    try:
        K_z, _P_z = dlqr(A_z, B_z, q_scale * np.eye(d), r_scale * np.eye(m))
        result["K_z"] = K_z
        result["latent_closed_loop_spectral_radius"] = float(
            np.max(np.abs(np.linalg.eigvals(A_z - B_z @ K_z)))
        )
    except Exception as e:  # DARE can fail to converge on a pathological A_z, B_z
        result["K_z"] = None
        result["error"] = str(e)
    return result


def closed_loop_rollout(
    system: LTISystem,
    obs_model,
    encoder,
    K_z: np.ndarray,
    n_steps: int,
    x0: np.ndarray,
    process_noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
    action_clip: float | None = None,
) -> np.ndarray:
    """Simulate u_t = -K_z * encoder(y_t) in closed loop on the TRUE system."""
    rng = rng or np.random.default_rng()
    xs = np.zeros((n_steps + 1, system.n))
    xs[0] = x0
    encoder.eval()
    with torch.no_grad():
        for t in range(n_steps):
            y = obs_model.observe(xs[t], rng)
            z = encoder(torch.tensor(y, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
            u = -K_z @ z
            if action_clip is not None:
                u = np.clip(u, -action_clip, action_clip)
            xs[t + 1] = system.step(xs[t], u, process_noise_std, rng)
    return xs


def evaluate_controller(
    system: LTISystem,
    obs_model,
    encoder,
    K_z: np.ndarray | None,
    n_trials: int = 20,
    n_steps: int = 60,
    x0_std: float = 0.05,
    success_threshold: float = 0.5,
    hold_steps: int = 10,
    seed: int = 0,
    process_noise_std: float = 0.0,
    action_clip: float | None = None,
) -> dict:
    """Metrics mirror the notes' reporting: success rate (state norm stays
    below threshold for the final `hold_steps` steps), mean fraction of the
    episode spent "stable", and the average final-state distance."""
    if hold_steps > n_steps + 1:
        raise ValueError(
            f"hold_steps={hold_steps} exceeds the trajectory length (n_steps+1={n_steps + 1}); "
            "the 'stable for the last hold_steps steps' check can never be satisfied, so "
            "success_rate would silently be 0 regardless of how stable the trajectory actually "
            "was. Either increase n_steps or decrease hold_steps."
        )
    if K_z is None:
        return {
            "success_rate": 0.0,
            "mean_fraction_stable": 0.0,
            "final_state_distance_avg": float("inf"),
            "trajectories": [],
            "note": "no stabilizing latent controller could be designed (unstable mode uncontrollable in latent space)",
        }
    rng = np.random.default_rng(seed)
    successes = 0
    frac_stable_list, final_dist_list, trajectories = [], [], []
    for _trial in range(n_trials):
        x0 = x0_std * rng.standard_normal(system.n)
        xs = closed_loop_rollout(system, obs_model, encoder, K_z, n_steps, x0, process_noise_std, rng, action_clip)
        trajectories.append(xs)
        norms = np.linalg.norm(xs, axis=1)
        finite = np.isfinite(norms)
        stable_mask = finite & (norms < success_threshold)
        frac_stable_list.append(stable_mask.mean())
        success = bool(np.all(stable_mask[-hold_steps:])) if len(stable_mask) >= hold_steps else False
        successes += int(success)
        final_dist_list.append(norms[-1] if finite[-1] else np.inf)
    finite_final = [d for d in final_dist_list if np.isfinite(d)]
    return {
        "success_rate": successes / n_trials,
        "mean_fraction_stable": float(np.mean(frac_stable_list)),
        "final_state_distance_avg": float(np.mean(finite_final)) if finite_final else float("inf"),
        "trajectories": trajectories,
    }
