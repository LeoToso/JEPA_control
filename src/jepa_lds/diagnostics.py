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
