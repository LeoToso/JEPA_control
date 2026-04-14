"""DMD with control (DMDc) for latent system identification."""

from __future__ import annotations
import warnings
from typing import Optional, Tuple
import numpy as np
import scipy.linalg


def fit_dmdc(Z, A_actions, Z_next, rcond=1e-10):
    N, d = Z.shape; d_u = A_actions.shape[1]
    X = np.hstack([Z, A_actions]); Y = Z_next
    Theta, res, rank, sv = np.linalg.lstsq(X, Y, rcond=rcond)
    return Theta[:d, :].T, Theta[d:, :].T


def fit_dmdc_proximal(Z, A_actions, Z_next, rho=1.0, max_iter=100, tol=1e-6,
                      rcond=1e-10, A_init=None, B_init=None):
    if A_init is None or B_init is None:
        A_hat, B_hat = fit_dmdc(Z, A_actions, Z_next, rcond=rcond)
    else:
        A_hat, B_hat = A_init.copy(), B_init.copy()
    N, d = Z.shape; d_u = A_actions.shape[1]
    ZtZ_reg = Z.T @ Z + rho * np.eye(d)
    AtA_reg = A_actions.T @ A_actions + rho * np.eye(d_u)
    converged = False; dA = dB = 0.0
    for it in range(max_iter):
        A_old = A_hat.copy(); B_old = B_hat.copy()
        Residual_B = Z_next - A_actions @ B_hat.T
        A_hat = scipy.linalg.solve(ZtZ_reg, Z.T @ Residual_B + rho * A_old.T).T
        Residual_A = Z_next - Z @ A_hat.T
        B_hat = scipy.linalg.solve(AtA_reg, A_actions.T @ Residual_A + rho * B_old.T).T
        dA = np.linalg.norm(A_hat - A_old, 'fro'); dB = np.linalg.norm(B_hat - B_old, 'fro')
        if dA + dB < tol: converged = True; break
    residual_num = np.linalg.norm(Z_next - Z @ A_hat.T - A_actions @ B_hat.T, 'fro')
    rel_residual = float(residual_num / (np.linalg.norm(Z_next, 'fro') + 1e-12))
    return A_hat, B_hat, {'n_iters': it+1, 'residual': rel_residual, 'converged': converged,
                          'dA_final': float(dA), 'dB_final': float(dB)}


def fit_output_map(Z, Y, rcond=1e-10):
    C_hat_T, _, _, _ = np.linalg.lstsq(Z, Y, rcond=rcond)
    return C_hat_T.T


class DMDcFitter:
    def __init__(self, use_proximal=True, proximal_rho=1.0, proximal_max_iter=100,
                 proximal_tol=1e-6, rcond=1e-10):
        self.use_proximal=use_proximal; self.proximal_rho=proximal_rho
        self.proximal_max_iter=proximal_max_iter; self.proximal_tol=proximal_tol
        self.rcond=rcond; self.A_hat=None; self.B_hat=None; self.C_hat=None; self.fit_info={}

    def fit(self, Z, A_latent, Z_next, Y=None, A_init=None, B_init=None):
        if A_latent.shape[1] > 1:
            eigvals = np.linalg.eigvalsh(np.cov(A_latent.T))
            eigvals_pos = np.maximum(eigvals, 1e-12)
            kappa = float(eigvals_pos.max() / eigvals_pos.min())
        else:
            kappa = 1.0
        self.fit_info['action_cov_kappa'] = kappa
        if kappa > 1000:
            warnings.warn(f'Action covariance condition number kappa={kappa:.1f} > 1000.', RuntimeWarning)
        if self.use_proximal:
            self.A_hat, self.B_hat, prox_info = fit_dmdc_proximal(
                Z, A_latent, Z_next, rho=self.proximal_rho, max_iter=self.proximal_max_iter,
                tol=self.proximal_tol, rcond=self.rcond, A_init=A_init, B_init=B_init)
            self.fit_info.update(prox_info)
        else:
            self.A_hat, self.B_hat = fit_dmdc(Z, A_latent, Z_next, rcond=self.rcond)
            res_num = np.linalg.norm(Z_next - Z @ self.A_hat.T - A_latent @ self.B_hat.T, 'fro')
            self.fit_info['residual'] = float(res_num / (np.linalg.norm(Z_next, 'fro') + 1e-12))
        if Y is not None: self.C_hat = fit_output_map(Z, Y, rcond=self.rcond)
        self.fit_info['A_hat_cond'] = float(np.linalg.cond(self.A_hat) if self.A_hat is not None else float('nan'))
        self.fit_info['B_hat_cond'] = float(np.linalg.cond(self.B_hat) if self.B_hat is not None else float('nan'))
        return self

    def summary(self):
        lines = ['DMDcFitter summary:']
        for k, v in self.fit_info.items(): lines.append(f'  {k}: {v}')
        if self.A_hat is not None:
            lines.append(f'  spectral_radius(A_hat): {np.max(np.abs(np.linalg.eigvals(self.A_hat))):.4f}')
        return '\n'.join(lines)

    def compute_residual(self, Z_test, A_test, Z_next_test):
        Z_next_pred = Z_test @ self.A_hat.T + A_test @ self.B_hat.T
        return float(np.linalg.norm(Z_next_test - Z_next_pred, 'fro') / (np.linalg.norm(Z_next_test, 'fro') + 1e-12))
