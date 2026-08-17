"""Diagnostics that quantify *why* a trained representation does or does not
support a stabilizing controller: a ridge-probe measure of how much of the
true unstable modal coordinate survives in z, an eigenvector-alignment
check, and the two panels used by `probe_local_stability.py`.
"""
from __future__ import annotations

import numpy as np
import torch

from .control import closed_loop_rollout, encoder_state_to_latent_map
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


def build_2d_grid(lo: tuple[float, float], hi: tuple[float, float], n_points: int | tuple[int, int] = 21):
    """Meshgrid over [lo[0], hi[0]] x [lo[1], hi[1]]. `n_points` is either a
    single resolution shared by both axes, or a (n0, n1) pair for
    independently-sized axes (e.g. a tighter grid on angle than on angular
    velocity)."""
    n0, n1 = (n_points, n_points) if isinstance(n_points, int) else n_points
    g0 = np.linspace(lo[0], hi[0], n0)
    g1 = np.linspace(lo[1], hi[1], n1)
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


def unstable_eigenvector_alignment(
    system: LTISystem, obs_model: ObservationModel, encoder, predictor, dims: tuple[int, int] = (0, 1)
) -> dict:
    """Compares the TRUE system's unstable eigenvector against the LEARNED
    predictor's own dominant (largest-|eigenvalue|) eigenvector, mapped
    back to state space -- a direct, grid-free check of whether the
    learned dynamics' fastest-growing direction actually points along the
    true unstable mode, rather than inferring it indirectly from a local
    vector field or a closed-loop rollout.

    The state-space direction is recovered via `encoder_state_to_latent_map`
    (the encoder's EXACT linear map M: x -> z, derived analytically from
    its weights and the fixed observation matrix -- no data or fitting
    involved) and its Moore-Penrose pseudoinverse: `pinv(M) @ z_dom` is the
    minimum-norm state-space direction whose image under M is z_dom. This
    matters when latent_dim < system.n (as in the "killer collapse"
    construction): a NOISY, data-fit ridge-regression decode of a
    lower-dimensional z back to the full state is ill-posed and can pick
    up spurious correlations from whatever dataset happens to be used to
    fit it -- the exact pseudoinverse of the encoder's own analytic map has
    no such degree of freedom.

    Both eigenvectors are projected onto the 2 chosen `dims` (the same
    slice-and-hold-other-dims-at-zero convention used by the grid panels)
    and normalized to unit length so their alignment (cosine similarity)
    is directly comparable regardless of scale.

    For a 2-state system, also returns the true STABLE eigenvector as a
    reference, so it's visually obvious if the learned direction has
    instead aligned with the wrong (stable) mode; omitted for n > 2, where
    "the" stable eigenvector isn't unique."""
    w, V, _Vinv = system.modal_decomposition()
    idx_u = system.unstable_mode_index()
    d0, d1 = dims

    def _project_unit(v: np.ndarray) -> np.ndarray:
        v2 = np.real(v)[[d0, d1]]
        norm = np.linalg.norm(v2)
        return v2 / norm if norm > 1e-12 else v2

    v_true_unstable = _project_unit(V[:, idx_u])
    v_true_stable = _project_unit(V[:, 1 - idx_u]) if system.n == 2 else None

    A_z, _B_z = predictor.matrices()
    w_z, V_z = np.linalg.eig(A_z)
    dom_idx = int(np.argmax(np.abs(w_z)))
    z_dom = np.real(V_z[:, dom_idx])
    M = encoder_state_to_latent_map(system, obs_model, encoder)  # (latent_dim, n), z = M @ x
    x_dom = np.linalg.pinv(M) @ z_dom  # minimum-norm state-space preimage of z_dom
    v_learned = _project_unit(x_dom)
    # Eigenvectors are only defined up to sign; canonicalize so a well-aligned "learned"
    # arrow visually points the SAME way as "ground truth (unstable)" rather than
    # correctly-but-confusingly appearing as an antiparallel arrow.
    if np.dot(v_learned, v_true_unstable) < 0:
        v_learned = -v_learned

    result = {
        "v_true_unstable": v_true_unstable,
        "v_true_stable": v_true_stable,
        "v_learned": v_learned,
        "dims": dims,
        "learned_dominant_eigval": complex(w_z[dom_idx]),
        "cos_sim_unstable": float(np.dot(v_true_unstable, v_learned)),
    }
    if v_true_stable is not None:
        result["cos_sim_stable"] = float(np.dot(v_true_stable, v_learned))
    return result


def closed_loop_trajectory_panel(
    system: LTISystem,
    obs_model: ObservationModel,
    encoder,
    K_z: np.ndarray,
    x0s: list[np.ndarray],
    dims: tuple[int, int] = (0, 1),
    n_steps: int = 300,
    seed: int = 0,
) -> dict:
    """Panel 2 (trajectory view): one closed-loop rollout per starting state
    in `x0s`, under the LEARNED controller u_t = -K_z * encoder(y_t) on the
    TRUE system, projected onto `dims` -- a direct, qualitative "does this
    particular starting point actually converge to the equilibrium" view.
    Each x0 is simulated with its own draw from a seeded RNG stream (so
    results are reproducible, but not identical) for the observation-model
    measurement noise."""
    d0, d1 = dims
    rng = np.random.default_rng(seed)
    trajectories = []
    for x0 in x0s:
        x0_arr = np.asarray(x0, dtype=np.float64)
        xs = closed_loop_rollout(system, obs_model, encoder, K_z, n_steps, x0_arr, process_noise_std=0.0, rng=rng)
        trajectories.append(xs[:, [d0, d1]])
    return {
        "dims": dims,
        "trajectories": trajectories,
        "x0s": [np.asarray(x0, dtype=np.float64)[[d0, d1]] for x0 in x0s],
    }


def lyapunov_decrease_panel(
    system: LTISystem,
    obs_model: ObservationModel,
    encoder,
    K_z: np.ndarray,
    P_gt: np.ndarray,
    dims: tuple[int, int] = (0, 1),
    lo: tuple[float, float] = (-1.0, -1.0),
    hi: tuple[float, float] = (1.0, 1.0),
    n_points: int | tuple[int, int] = 17,
) -> dict:
    """Panel 3 of the local-stability probe: pointwise check of whether ONE
    step under the LEARNED closed-loop controller decreases the GT quadratic
    Lyapunov function V(x) = x^T P_gt x, where `P_gt` is the Riccati solution
    of a full-state-feedback oracle LQR designed directly on the true
    (A, B) (e.g. from `system.dlqr`) -- an honest, model-free check of
    whether the learned controller's action is a valid descent direction for
    the TRUE system's own natural cost, independent of what the learned
    model itself believes."""
    XX, YY = build_2d_grid(lo, hi, n_points)
    states = grid_to_states(XX, YY, dims, system.n)
    z = _encode_states(obs_model, encoder, states)
    u = -(K_z @ z.T).T
    x_next = states @ system.A.T + u @ system.B.T

    V = np.einsum("bi,ij,bj->b", states, P_gt, states)
    V_next = np.einsum("bi,ij,bj->b", x_next, P_gt, x_next)
    delta_V = V_next - V
    return {
        "XX": XX, "YY": YY, "dims": dims,
        "delta_V": delta_V.reshape(XX.shape),
        "frac_decrease": float(np.mean(delta_V < 0)),
    }
