"""Numerical injectivity of an encoder on the nonstable invariant subspace."""
from __future__ import annotations

import numpy as np
from scipy.linalg import schur


def nonstable_subspace_report(
    A, F, *, spectral_tol=1e-7, rank_rtol=1e-6, rank_atol=0.0
):
    """Check rank(F U) = dim(U), including generalized marginal directions.

    U is an orthonormal basis from a sorted complex Schur decomposition.
    This avoids using an incomplete eigenvector basis for defective A.
    Modes with |lambda| >= 1 - spectral_tol are classified as nonstable.

    Rank uses rank_atol + rank_rtol * ||F||_2, so weak nonstable directions
    are compared to the full encoder's gain, rather than renormalized away.
    This is a numerical injectivity diagnostic, not a learned-controller
    stability certificate. Near the unit circle, classification depends
    on spectral_tol.
    """
    A = np.asarray(A, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError("A must be a nonempty square matrix")
    if F.ndim != 2 or F.shape[1] != A.shape[0] or F.shape[0] == 0:
        raise ValueError("F must have shape (latent_dim, state_dim)")
    if not np.isfinite(A).all() or not np.isfinite(F).all():
        raise ValueError("A and F must be finite")
    for value in (spectral_tol, rank_rtol, rank_atol):
        if not np.isfinite(value) or value < 0:
            raise ValueError("Tolerances must be finite and nonnegative")
    if spectral_tol >= 1:
        raise ValueError("spectral_tol must be smaller than one")

    S, Q, count = schur(
        A, output="complex",
        sort=lambda eigenvalue: abs(eigenvalue) >= 1.0 - spectral_tol,
    )
    U = Q[:, :count]
    restricted = F @ U
    singular_values = np.linalg.svd(restricted, compute_uv=False)
    # If latent_dim < count, injectivity is impossible. Include structural
    # zeros so minimum_gain does not incorrectly report a positive value.
    singular_values = np.pad(
        singular_values, (0, count - len(singular_values))
    )
    encoder_gain = float(np.linalg.norm(F, ord=2))
    threshold = rank_atol + rank_rtol * encoder_gain
    rank = int(np.count_nonzero(singular_values > threshold))
    residual = A @ U - U @ (U.conj().T @ A @ U)
    selected = np.diag(S)[:count]
    return {
        "state_dim": int(A.shape[0]),
        "latent_dim": int(F.shape[0]),
        "nonstable_dim": int(count),
        "nonstable_eigenvalues": [
            {"real": float(v.real), "imag": float(v.imag)} for v in selected
        ],
        "rank_F_U": rank,
        "injective_at_tolerance": bool(rank == count),
        "singular_values_F_U": singular_values.tolist(),
        "minimum_gain": float(singular_values[-1]) if count else None,
        "minimum_gain_relative_to_F": (
            float(singular_values[-1] / encoder_gain)
            if count and encoder_gain > 0 else (0.0 if count else None)
        ),
        "encoder_gain": encoder_gain,
        "spectral_tol": float(spectral_tol),
        "rank_rtol": float(rank_rtol),
        "rank_atol": float(rank_atol),
        "rank_threshold": float(threshold),
        "rank_sensitivity": {
            str(tol): int(np.count_nonzero(
                singular_values > rank_atol + tol * encoder_gain
            )) for tol in (1e-4, 1e-6, 1e-8)
        },
        "invariant_subspace_residual": float(np.linalg.norm(residual, ord="fro")),
    }
