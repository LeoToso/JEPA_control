"""Simplified probe suite: spectral match + PBH stabilisability + linearization residual."""
from __future__ import annotations
import warnings
from typing import Dict, Any
import numpy as np


def run_all_probes(A_jac: np.ndarray, B_jac: np.ndarray, gt, model,
                   env, z_star: np.ndarray, device, config: dict = {},
                   frame_stack: int = 1) -> Dict:
    results = {}
    eps_lambda = float(config.get('epsilon_lambda', 0.05))
    delta_tol  = float(config.get('delta_tol',      0.05))

    # ── P1: Spectral match ────────────────────────────────────────────────────
    try:
        eigvals = np.linalg.eigvals(A_jac)
        rho_jac = float(np.max(np.abs(eigvals)))
        rho_err = abs(rho_jac - gt.spectral_radius)
        # For each GT unstable eigenvalue, find nearest Jacobian eigenvalue
        min_dists = []
        for lam_star in gt.unstable_eigenvalues:
            dists = np.abs(eigvals - lam_star)
            min_dists.append(float(dists.min()))
        umr = float(np.mean([d < eps_lambda for d in min_dists])) if min_dists else float('nan')
        results['spectral'] = {
            'rho_jac':              rho_jac,
            'rho_gt':               gt.spectral_radius,
            'spectral_radius_error':rho_err,
            'min_dist_to_unstable': float(np.mean(min_dists)) if min_dists else float('nan'),
            'unstable_mode_recall': umr,
            'recovered':            bool(umr == 1.0),
        }
    except Exception as exc:
        warnings.warn(f'Spectral probe failed: {exc}')
        results['spectral'] = {'error': str(exc)}

    # ── P2: PBH stabilisability ───────────────────────────────────────────────
    try:
        d, d_u = A_jac.shape[0], B_jac.shape[1]
        sigma_mins = []
        for lam_star in gt.unstable_eigenvalues:
            lam_r = float(np.real(lam_star))
            M = np.hstack([lam_r * np.eye(d) - A_jac, B_jac])
            sv = np.linalg.svd(M, compute_uv=False)
            sigma_mins.append(float(sv[-1]))
        mu_S = float(np.min(sigma_mins)) if sigma_mins else float('nan')
        results['pbh'] = {
            'mu_S':           mu_S,
            'is_stabilizable':bool(mu_S > delta_tol),
            'sigma_mins':     sigma_mins,
        }
    except Exception as exc:
        warnings.warn(f'PBH probe failed: {exc}')
        results['pbh'] = {'error': str(exc)}

    # ── P3: Linearization residual in neighbourhood of z* ────────────────────
    try:
        import torch
        model.eval()
        rng = np.random.RandomState(7)
        thetas = [0.0, 0.05, -0.05, 0.10, -0.10]

        def _encode(obs_np, prev_obs_np=None):
            """Encode a single obs, optionally stacking with prev for frame_stack > 1."""
            curr_t = (torch.from_numpy(obs_np).float()
                      .permute(2, 0, 1)[None].to(device) / 255.0)
            if frame_stack > 1:
                prev_t = (curr_t if prev_obs_np is None else
                          torch.from_numpy(prev_obs_np).float()
                          .permute(2, 0, 1)[None].to(device) / 255.0)
                return torch.cat([prev_t, curr_t], dim=1)
            return curr_t

        # Compute c_drift = f(z*, 0) - z* so the linear model matches what CEM uses:
        #   z1_lin = A(z0 - z*) + B·u + z* + c_drift
        with torch.no_grad():
            _zs_t  = torch.tensor(z_star, dtype=torch.float32, device=device).unsqueeze(0)
            _pW = getattr(model.config, 'predictor_window', 1)
            _zs_win = _zs_t.unsqueeze(1).expand(1, _pW, -1)
            _u_zero_win = torch.zeros(1, _pW, 1, device=device)
            c_drift = (model.predict(_zs_win, _u_zero_win) - _zs_t).cpu().numpy()[0]

        residuals = {}
        abs_residuals = {}
        for theta in thetas:
            obs, _, _ = env.reset_to_state(
                np.array([0., 0., theta, 0.], dtype=np.float32))
            with torch.no_grad():
                z0 = model.encoder(_encode(obs)).cpu().numpy()[0]

            errs, abs_errs = [], []
            for _ in range(10):
                u = float(rng.uniform(-1.0, 1.0))
                prev_obs = obs.copy()
                obs_next, _, _, _, _ = env.step(u)
                with torch.no_grad():
                    z1 = model.encoder(_encode(obs_next, prev_obs)).cpu().numpy()[0]
                    obs, _, _ = env.reset_to_state(  # reset for next sample
                        np.array([0., 0., theta, 0.], dtype=np.float32))
                # linear prediction including constant drift c = f(z*,0)-z*
                z1_lin = A_jac @ (z0 - z_star) + B_jac[:, 0] * u + z_star + c_drift
                abs_err = float(np.linalg.norm(z1_lin - z1))
                abs_errs.append(abs_err)
                # relative error: normalise by displacement from z* (floor avoids /0)
                denom = max(np.linalg.norm(z1 - z_star), 0.01 * np.sqrt(len(z_star)))
                errs.append(abs_err / denom)
            residuals[f'theta={theta:+.2f}']     = float(np.mean(errs))
            abs_residuals[f'theta={theta:+.2f}'] = float(np.mean(abs_errs))

        results['linearization'] = {
            'residuals':          residuals,
            'abs_residuals':      abs_residuals,
            'mean_residual':      float(np.mean(list(residuals.values()))),
            'mean_abs_residual':  float(np.mean(list(abs_residuals.values()))),
        }
    except Exception as exc:
        warnings.warn(f'Linearization probe failed: {exc}')
        results['linearization'] = {'error': str(exc)}

    _print_summary(results, gt)
    return results


def _print_summary(results: Dict, gt):
    print('\n' + '=' * 55)
    print('PROBE RESULTS')
    print('=' * 55)
    sp = results.get('spectral', {})
    if 'error' not in sp:
        print(f"  Spectral radius (Jacobian): {sp.get('rho_jac', float('nan')):.4f}"
              f"  (GT: {sp.get('rho_gt', float('nan')):.4f})")
        print(f"  Spectral radius error:      {sp.get('spectral_radius_error', float('nan')):.4f}")
        print(f"  Min dist to unstable mode:  {sp.get('min_dist_to_unstable', float('nan')):.4f}")
        print(f"  Unstable mode recall:       {sp.get('unstable_mode_recall', float('nan')):.4f}")
    else:
        print(f"  Spectral probe ERROR: {sp['error']}")

    pb = results.get('pbh', {})
    if 'error' not in pb:
        print(f"  PBH mu_S:                   {pb.get('mu_S', float('nan')):.4f}")
        print(f"  Is stabilisable:            {pb.get('is_stabilizable', '?')}")
    else:
        print(f"  PBH probe ERROR: {pb['error']}")

    lin = results.get('linearization', {})
    if 'error' not in lin:
        print(f"  Mean linearization residual (relative):{lin.get('mean_residual', float('nan')):.4f}")
        print(f"  Mean linearization residual (absolute):{lin.get('mean_abs_residual', float('nan')):.4f}")
        abs_res = lin.get('abs_residuals', {})
        for k, v in lin.get('residuals', {}).items():
            abs_v = abs_res.get(k, float('nan'))
            print(f"    {k}: rel={v:.4f}  abs={abs_v:.4f}")
    else:
        print(f"  Linearization probe ERROR: {lin['error']}")
    print('=' * 55 + '\n')
