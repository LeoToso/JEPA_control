"""Diagnostics that quantify *why* a trained representation does or does not
support a stabilizing controller: a ridge-probe measure of how much of the
true unstable modal coordinate survives in z, plus the norm-divergence and
cosine-alignment rollout panels described in the notes (Panels 4 and 5).
"""
from __future__ import annotations

import numpy as np
import torch

from .data import EpisodeBatch, ObservationModel
from .systems import LTISystem


def ridge_fit(Z: np.ndarray, y: np.ndarray, alpha: float = 1e-2) -> np.ndarray:
    n, d = Z.shape
    Zb = np.hstack([Z, np.ones((n, 1))])
    A = Zb.T @ Zb + alpha * np.eye(d + 1)
    b = Zb.T @ y
    return np.linalg.solve(A, b)


def ridge_predict(Z: np.ndarray, beta: np.ndarray) -> np.ndarray:
    n = Z.shape[0]
    Zb = np.hstack([Z, np.ones((n, 1))])
    return Zb @ beta


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _encode_flat(encoder, batch: EpisodeBatch):
    y = batch.y.reshape(-1, batch.y.shape[-1])
    with torch.no_grad():
        z = encoder(torch.tensor(y, dtype=torch.float32)).numpy()
    x = batch.x.reshape(-1, batch.x.shape[-1])
    return z, x


def unstable_mode_retention(
    system: LTISystem, encoder, train_batch: EpisodeBatch, val_batch: EpisodeBatch, alpha: float = 1e-2
) -> float:
    """Fit z -> true unstable modal coordinate with ridge regression on
    `train_batch`; report held-out R^2 on `val_batch`. This is the
    quantitative version of "did the unstable mode collapse in z"."""
    Z_tr, x_tr = _encode_flat(encoder, train_batch)
    Z_val, x_val = _encode_flat(encoder, val_batch)
    xi_tr = system.unstable_modal_coordinate(x_tr)
    xi_val = system.unstable_modal_coordinate(x_val)
    beta = ridge_fit(Z_tr, xi_tr, alpha=alpha)
    xi_pred = ridge_predict(Z_val, beta)
    return r2_score(xi_val, xi_pred)


def fit_state_probe(encoder, batch: EpisodeBatch, alpha: float = 1e-2) -> np.ndarray:
    """Ridge-regress the FULL ground-truth state x (not just the unstable
    modal coordinate) from z = encoder(y). Used to decode a state-space
    quantity (position in the phase portrait, etc.) out of the latent for
    plotting, the same way `unstable_mode_retention` probes a single
    coordinate."""
    Z, x = _encode_flat(encoder, batch)
    return ridge_fit(Z, x, alpha=alpha)


def decode_state(Z: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return ridge_predict(Z, beta)


def build_2d_grid(lo: tuple[float, float], hi: tuple[float, float], n_points: int = 21):
    """Meshgrid over [lo[0], hi[0]] x [lo[1], hi[1]]."""
    g0 = np.linspace(lo[0], hi[0], n_points)
    g1 = np.linspace(lo[1], hi[1], n_points)
    return np.meshgrid(g0, g1)


def grid_to_states(XX: np.ndarray, YY: np.ndarray, dims: tuple[int, int], n: int) -> np.ndarray:
    """Embed a 2D (dims[0], dims[1]) grid into full n-dim state vectors, with
    every other coordinate held at zero. Returns (n_points^2, n)."""
    states = np.zeros((XX.size, n))
    states[:, dims[0]] = XX.ravel()
    states[:, dims[1]] = YY.ravel()
    return states


def _encode_states(obs_model: ObservationModel, encoder, states: np.ndarray) -> np.ndarray:
    """Deterministic (noiseless, zeroed-distractor) encode of raw states --
    used for the grid-based panels, where we want a fixed vector field / cost
    surface rather than a noisy Monte-Carlo estimate."""
    y_signal = states @ obs_model.C.T
    if obs_model.n_distractor:
        y = np.hstack([y_signal, np.zeros((states.shape[0], obs_model.n_distractor))])
    else:
        y = y_signal
    with torch.no_grad():
        return encoder(torch.tensor(y, dtype=torch.float32)).numpy()


def phase_portrait_panel(
    system: LTISystem,
    obs_model: ObservationModel,
    encoder,
    predictor,
    beta: np.ndarray,
    dims: tuple[int, int] = (0, 1),
    lo: tuple[float, float] = (-1.0, -1.0),
    hi: tuple[float, float] = (1.0, 1.0),
    n_points: int = 17,
) -> dict:
    """Panel 1: at each grid state x, compare the true one-step drift
    (A - I) x against the learned drift obtained by encoding x, applying the
    latent predictor with zero action, and decoding back to state space with
    the ridge probe `beta`."""
    XX, YY = build_2d_grid(lo, hi, n_points)
    states = grid_to_states(XX, YY, dims, system.n)
    dx_true = states @ (system.A - np.eye(system.n)).T

    z = _encode_states(obs_model, encoder, states)
    with torch.no_grad():
        z_next = predictor(
            torch.tensor(z, dtype=torch.float32), torch.zeros(states.shape[0], system.m)
        ).numpy()
    x_hat_next = decode_state(z_next, beta)
    dx_learned = x_hat_next - states

    d0, d1 = dims
    return {
        "XX": XX,
        "YY": YY,
        "U_true": dx_true[:, d0].reshape(XX.shape),
        "V_true": dx_true[:, d1].reshape(XX.shape),
        "U_learned": dx_learned[:, d0].reshape(XX.shape),
        "V_learned": dx_learned[:, d1].reshape(XX.shape),
        "dims": dims,
    }


def _rollout_latent_zero_action(predictor, z0: np.ndarray, m: int, H: int) -> np.ndarray:
    with torch.no_grad():
        z = torch.tensor(z0, dtype=torch.float32)
        zero_a = torch.zeros(z0.shape[0], m)
        for _ in range(H):
            z = predictor(z, zero_a)
        return z.numpy()


def h_step_prediction_error_panel(
    system: LTISystem,
    obs_model: ObservationModel,
    encoder,
    predictor,
    beta: np.ndarray,
    dims: tuple[int, int] = (0, 1),
    lo: tuple[float, float] = (-1.0, -1.0),
    hi: tuple[float, float] = (1.0, 1.0),
    n_points: int = 17,
    H: int = 10,
) -> dict:
    """Panel 2: from each grid state x0, decode the H-step-ahead latent
    prediction D(f_H(z0, 0)) (recursive predictor rollout with zero actions,
    decoded back to the ORIGINAL PHYSICAL STATE SPACE via the ridge probe
    `beta`) and compare it against the true H-step-ahead state x_H = A^H x0
    -- ||D(f_H(z0,0)) - x_H||, exactly the quantity plotted in
    plot_checkpoint_summary_physical_smwm.py's panel 2, rather than an error
    measured in raw latent coordinates."""
    XX, YY = build_2d_grid(lo, hi, n_points)
    states = grid_to_states(XX, YY, dims, system.n)

    x_H_true = states @ np.linalg.matrix_power(system.A, H).T
    z0 = _encode_states(obs_model, encoder, states)
    z_H_pred = _rollout_latent_zero_action(predictor, z0, system.m, H)
    s_pred = decode_state(z_H_pred, beta)

    error = np.linalg.norm(s_pred - x_H_true, axis=1)
    return {"XX": XX, "YY": YY, "error": error.reshape(XX.shape), "dims": dims, "H": H}


def planning_cost_panel(
    system: LTISystem,
    obs_model: ObservationModel,
    encoder,
    predictor,
    beta: np.ndarray,
    dims: tuple[int, int] = (0, 1),
    lo: tuple[float, float] = (-1.0, -1.0),
    hi: tuple[float, float] = (1.0, 1.0),
    n_points: int = 17,
    H: int = 10,
) -> dict:
    """Panel 3: log10 ||D(f_H(z0, 0)) - s_goal||^2 -- the squared distance,
    in the ORIGINAL PHYSICAL STATE SPACE (decoded via the ridge probe
    `beta`), between the same H-step zero-action latent rollout used in
    panel 2 and the equilibrium s_goal=0. A simple proxy for "how far does
    the learned model's own open-loop prediction drift from the goal",
    matching plot_checkpoint_summary_physical_smwm.py's panel 3 (no LQR
    value function / Riccati solution needed)."""
    XX, YY = build_2d_grid(lo, hi, n_points)
    states = grid_to_states(XX, YY, dims, system.n)
    z0 = _encode_states(obs_model, encoder, states)
    z_H = _rollout_latent_zero_action(predictor, z0, system.m, H)
    s_pred = decode_state(z_H, beta)

    cost = np.sum(s_pred**2, axis=1)
    cost = np.clip(cost, 1e-12, None)
    return {"XX": XX, "YY": YY, "log_cost": np.log10(cost).reshape(XX.shape), "dims": dims, "H": H}


def eigenvalue_comparison(system: LTISystem, predictor) -> dict:
    A_z, _B_z = predictor.matrices()
    return {
        "true_eigvals": system.eigvals(),
        "latent_eigvals": np.linalg.eigvals(A_z),
        "true_spectral_radius": system.spectral_radius(),
        "latent_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(A_z)))),
    }


def latent_norm_divergence(
    system: LTISystem,
    obs_model: ObservationModel,
    encoder,
    predictor,
    x0: np.ndarray,
    n_steps: int,
    rng: np.random.Generator,
):
    """Passive (zero-action) rollout: compare ||z_t|| from re-encoding the
    true trajectory each step against ||z_t_hat|| from recursively applying
    the learned predictor starting only from z_0 (Panel 4 in the notes)."""
    x = x0.copy()
    y0 = obs_model.observe(x, rng)
    with torch.no_grad():
        z_true = encoder(torch.tensor(y0, dtype=torch.float32).unsqueeze(0))
    z_rec = z_true.clone()
    zero_a = torch.zeros(1, system.m)

    true_norms = [float(z_true.norm())]
    rec_norms = [float(z_rec.norm())]
    for _t in range(n_steps):
        x = system.step(x, np.zeros(system.m))
        y = obs_model.observe(x, rng)
        with torch.no_grad():
            z_true = encoder(torch.tensor(y, dtype=torch.float32).unsqueeze(0))
            z_rec = predictor(z_rec, zero_a)
        true_norms.append(float(z_true.norm()))
        rec_norms.append(float(z_rec.norm()))
    return np.array(true_norms), np.array(rec_norms)


def cosine_alignment_unstable_direction(
    system: LTISystem,
    obs_model: ObservationModel,
    encoder,
    predictor,
    n_steps: int,
    perturbation_scales: list[float],
    rng: np.random.Generator,
):
    """Perturb the origin along the true unstable *right*-eigenvector, run
    both a ground-truth and a learned-latent passive rollout, and measure
    cos(Delta z_true, Delta z_pred) over time (Panel 5 in the notes). High
    cosine = the learned model predicts the correct *direction* of
    divergence, not just a numerically small residual."""
    w, V, _Vinv = system.modal_decomposition()
    idx = system.unstable_mode_index()
    v_u = np.real(V[:, idx])
    v_u = v_u / np.linalg.norm(v_u)

    y0 = obs_model.observe(np.zeros(system.n), rng)
    with torch.no_grad():
        z0 = encoder(torch.tensor(y0, dtype=torch.float32).unsqueeze(0))
    zero_a = torch.zeros(1, system.m)

    cos_over_time = np.zeros((len(perturbation_scales), n_steps))
    for i, scale in enumerate(perturbation_scales):
        x = scale * v_u
        y = obs_model.observe(x, rng)
        with torch.no_grad():
            z_true0 = encoder(torch.tensor(y, dtype=torch.float32).unsqueeze(0))
        z_pred = z_true0.clone()
        for t in range(n_steps):
            x = system.step(x, np.zeros(system.m))
            y = obs_model.observe(x, rng)
            with torch.no_grad():
                z_true = encoder(torch.tensor(y, dtype=torch.float32).unsqueeze(0))
                z_pred = predictor(z_pred, zero_a)
            dz_true = (z_true - z0).squeeze(0)
            dz_pred = (z_pred - z0).squeeze(0)
            denom = dz_true.norm() * dz_pred.norm()
            cos = float((dz_true @ dz_pred) / denom) if denom > 1e-9 else 0.0
            cos_over_time[i, t] = cos
    return cos_over_time
