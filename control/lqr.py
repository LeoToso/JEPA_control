"""Discrete-time LQR solver via DARE."""
from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import scipy.linalg


def pre_stabilize_A(A, true_unstable_eigs, tol=0.05, target=0.9):
    """Deflate phantom unstable eigenvalues of A for well-conditioned DARE.

    Any eigenvalue λ of A with |λ| > 1 that is NOT within `tol` of any
    true (physical) unstable eigenvalue is treated as a phantom mode and
    scaled to `target` (inside the unit disk).  The true physical unstable
    modes are left untouched so the DARE can place them optimally.

    Uses real Schur decomposition for numerical stability.

    Returns:
        A_pre  : pre-stabilized matrix (same shape as A)
        n_deflated : number of eigenvalues deflated
    """
    T, Z = scipy.linalg.schur(A, output='real')
    n = T.shape[0]
    n_deflated = 0
    i = 0
    while i < n:
        # Check for 2×2 block (complex-conjugate pair)
        if i + 1 < n and abs(T[i + 1, i]) > 1e-10:
            lam = np.linalg.eigvals(T[i:i + 2, i:i + 2])[0]
            if abs(lam) > 1.0:
                is_physical = any(abs(lam - e) < tol for e in true_unstable_eigs)
                if not is_physical:
                    T[i:i + 2, i:i + 2] *= (target / abs(lam))
                    n_deflated += 2
            i += 2
        else:
            lam = float(T[i, i])
            if abs(lam) > 1.0:
                is_physical = any(abs(lam - e) < tol for e in true_unstable_eigs)
                if not is_physical:
                    T[i, i] = lam * target / abs(lam)
                    n_deflated += 1
            i += 1
    A_pre = (Z @ T @ Z.T).real
    return A_pre, n_deflated


def solve_discrete_lqr(A, B, Q, R, true_unstable_eigs=None, pre_stabilize=False):
    """Solve the discrete-time LQR problem.

    If `pre_stabilize=True` (and `true_unstable_eigs` is provided), phantom
    unstable modes are deflated before solving DARE so the solution does not
    blow up due to nearly-uncontrollable spurious eigenvalues.  K is then
    evaluated on the *original* A so cl_eigs reflect true closed-loop poles.
    """
    A_dare = A
    if pre_stabilize and true_unstable_eigs is not None:
        A_dare, n_def = pre_stabilize_A(A, true_unstable_eigs)
        if n_def > 0:
            rho = float(np.max(np.abs(np.linalg.eigvals(A_dare))))
            print(f'[lqr] deflated {n_def} phantom mode(s); '
                  f'A_lqr spectral radius: {rho:.4f}')
    try:
        P = scipy.linalg.solve_discrete_are(A_dare, B, Q, R)
    except Exception:
        P = _dare_iteration(A_dare, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A_dare)
    # Closed-loop eigenvalues on the *original* A (true performance indicator)
    A_cl = A - B @ K
    eigs = scipy.linalg.eigvals(A_cl)
    return K, P, eigs


def _dare_iteration(A, B, Q, R, max_iter=5000, tol=1e-10):
    P = Q.copy()
    for _ in range(max_iter):
        P_new = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        if np.linalg.norm(P_new - P, 'fro') < tol:
            return P_new
        P = P_new
    return P

