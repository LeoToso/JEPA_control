"""Dynamics-aware SIGreg (L_dynSIG) and temporal covariance consistency (L_temp).

L_dynSIG replaces isotropic SIGreg with a target covariance derived from the
finite-horizon controllability Gramian of the learned local linearization:

    W_T = sum_{k=0}^{T_g-1} A^k B Sigma_c B^T (A^T)^k

Sigma_target = (1-alpha)*W_bar + alpha*I   (W_bar = W_T normalised to unit trace)

This encourages the latent distribution to align variance with controllable /
excitable / unstable directions — without requiring GT eigenvalues.

L_temp enforces temporal covariance consistency near equilibrium:

    Sigma_1_res ≈ A Sigma_0

where Sigma_0 = E[dz dz^T], Sigma_1_res = E[dz_{t+1}_res dz_t^T],
dz_{t+1}_res = dz_{t+1} - B c_t  (action contribution removed).

This carries spectral information: the recoverable Koopman operator
A_cov = Sigma_1 Sigma_0^{-1} should match the Jacobian A.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def compute_controllability_gramian(
    A: torch.Tensor,
    B: torch.Tensor,
    T_g: int = 5,
    Sigma_c: torch.Tensor | None = None,
) -> torch.Tensor:
    """Finite-horizon controllability Gramian.

    W_T = sum_{k=0}^{T_g-1} A^k B Sigma_c B^T (A^T)^k

    Args:
        A: (d, d) local transition Jacobian (detached).
        B: (d, m) control input Jacobian (detached).
        T_g: horizon; finite to avoid divergence on unstable A.
        Sigma_c: (m, m) action embedding covariance; defaults to I.
    Returns:
        W_T: (d, d) positive semi-definite Gramian.
    """
    d, m = B.shape
    if Sigma_c is None:
        Sigma_c = torch.eye(m, device=A.device, dtype=A.dtype)
    W_T = torch.zeros(d, d, device=A.device, dtype=A.dtype)
    Ak  = torch.eye(d, device=A.device, dtype=A.dtype)
    BBT = B @ Sigma_c @ B.T
    for _ in range(T_g):
        W_T = W_T + Ak @ BBT @ Ak.T
        Ak  = Ak @ A
    return W_T


def build_sigma_target(
    W_T: torch.Tensor,
    alpha: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize W_T and add an isotropic floor.

    Sigma_target = (1-alpha) * W_bar + alpha * I,
    where W_bar = W_T / (tr(W_T)/d + eps).
    """
    d   = W_T.shape[0]
    tr  = W_T.diagonal().sum()
    W_bar = W_T / (tr / d + eps)
    return (1 - alpha) * W_bar + alpha * torch.eye(d, device=W_T.device, dtype=W_T.dtype)


def dynsigreg_loss(
    z: torch.Tensor,
    Sigma_target: torch.Tensor,
    num_slices: int = 128,
    num_points: int = 17,
) -> torch.Tensor:
    """Dynamics-aware SIGreg.

    Same Epps-Pulley structure as standard SIGreg, but the target CF for
    projection w is exp(-0.5 * omega^2 * w^T Sigma_target w) rather than
    exp(-omega^2 / 2).  When Sigma_target = I this reduces to standard SIGreg.

    Args:
        z: (B, d) batch of latent vectors.
        Sigma_target: (d, d) detached target covariance.
        num_slices: K random unit projection directions.
        num_points: P frequency evaluation points in [0.5, 3.0].
    Returns:
        Scalar loss >= 0.
    """
    B, D = z.shape
    z = z - z.mean(dim=0)  # center before test

    W = F.normalize(torch.randn(num_slices, D, device=z.device, dtype=z.dtype), dim=1)
    y = W @ z.T  # (K, B)

    omega = torch.linspace(0.5, 3.0, num_points, device=z.device, dtype=z.dtype)  # (P,)

    # Per-projection target variance: v_k = w_k^T Sigma_target w_k  →  (K,)
    v = (W @ Sigma_target @ W.T).diagonal().clamp(min=1e-6)

    wy        = omega[None, :, None] * y[:, None, :]       # (K, P, B)
    phi_real  = wy.cos().mean(dim=-1)                       # (K, P)
    phi_imag  = wy.sin().mean(dim=-1)                       # (K, P)
    target_cf = torch.exp(-0.5 * omega[None, :] ** 2 * v[:, None])  # (K, P)

    return ((phi_real - target_cf) ** 2 + phi_imag ** 2).mean()


def temporal_consistency_loss(
    dz_t: torch.Tensor,
    dz_tp1: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    c_t: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Temporal covariance consistency loss.

    Enforces Sigma_1_res ≈ A Sigma_0, where:
        Sigma_0     = (1/N) dz_t^T dz_t
        Sigma_1_res = (1/N) dz_tp1_res^T dz_t
        dz_tp1_res  = dz_tp1 - B c_t  (remove linear action contribution)

    Normalised by ||Sigma_1_res||_F^2 to be scale-invariant.

    Args:
        dz_t:   (N, d) z_t  - z*  for near-equilibrium samples.
        dz_tp1: (N, d) z_{t+1} - z* for the same samples.
        A:      (d, d) detached Jacobian.
        B:      (d, m) detached control Jacobian.
        c_t:    (N, m) encoded actions; if provided, subtract B c_t.
        eps:    floor for normalisation denominator.
    Returns:
        Scalar loss >= 0.
    """
    N = dz_t.shape[0]
    if N < 4:
        return torch.zeros(1, device=A.device, dtype=A.dtype).squeeze()

    dz_tp1_res = dz_tp1 - (c_t @ B.T) if c_t is not None else dz_tp1

    Sigma_0     = dz_t.T @ dz_t / N            # (d, d)
    Sigma_1_res = dz_tp1_res.T @ dz_t / N      # (d, d)

    diff  = Sigma_1_res - A @ Sigma_0
    return (diff ** 2).sum() / ((Sigma_1_res ** 2).sum() + eps)
