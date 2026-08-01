"""Stability diagnostics for a JEPA checkpoint.

Computes:
  - Eigenvalue spectrum of A_aug (learned dynamics at z*)
  - rho(A_aug): learned spectral radius vs GT
  - z_ss: ||(I-A_cl)^{-1} c_aug|| — LQR steady-state drift due to fp_err
  - fp_err: ||f(z*,0) - z*||

Usage:
    python experiments/check_stability_diag.py \\
        --checkpoint results/jepa_v11_difenc/checkpoints/checkpoint_epoch0030.pt \\
        --data       data/cartpole_visual_fs5_v4 \\
        --config     configs/cartpole_jepa_v11_difenc.yaml \\
        --out        results/stability_ep030.png
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import torch
import yaml
import scipy.linalg as sla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Ground-truth rho via finite-difference Jacobian from actual environment ───
def _gt_rho_from_env(env_cfg: dict) -> float:
    """Spectral radius from numerical Jacobian of the real environment at x*=0.

    Uses finite differences on the actual env step so it matches whatever
    integration scheme (Euler, RK4, …) the environment uses — unlike the
    analytical matrix-exponential which ignores coupling and exact integration.
    """
    from envs.cartpole_visual import ContinuousCartpoleVisual
    env = ContinuousCartpoleVisual(
        frame_skip=int(env_cfg.get('frame_skip', 5)),
        image_size=int(env_cfg.get('image_size', 64)),
        mass_cart=float(env_cfg.get('mass_cart', 1.0)),
        mass_pole=float(env_cfg.get('mass_pole', 0.1)),
        pole_length=float(env_cfg.get('pole_length', 0.5)),
        gravity=float(env_cfg.get('gravity', 9.8)),
        dt=float(env_cfg.get('dt', 0.02)),
        seed=0)
    x_star = np.zeros(4, dtype=np.float64)
    eps = 1e-5
    n = 4
    A_fd = np.zeros((n, n))
    for j in range(n):
        xp = x_star.copy(); xp[j] += eps
        xm = x_star.copy(); xm[j] -= eps
        env.reset_to_state(xp); _, sp, _, _, _ = env.step(0.0)
        env.reset_to_state(xm); _, sm, _, _, _ = env.step(0.0)
        A_fd[:, j] = (np.array(sp) - np.array(sm)) / (2 * eps)
    env.close()
    return float(np.max(np.abs(np.linalg.eigvals(A_fd))))


def _find_near_eq_obs_hdf5(data_dir):
    train_path = Path(data_dir) / 'train.hdf5'
    best_norm, best_obs = float('inf'), None
    with h5py.File(train_path, 'r') as f:
        for k in f['episodes']:
            ep    = f['episodes'][k]
            norms = np.linalg.norm(ep['states'][:], axis=1)
            j     = int(np.argmin(norms))
            if norms[j] < best_norm:
                best_norm = norms[j]
                best_obs  = ep['observations'][j]
    return best_obs, best_norm


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data',       default='data/cartpole_visual_fs5_v4')
    p.add_argument('--config',     default='configs/cartpole_jepa_v11_difenc.yaml')
    p.add_argument('--out',        default='results/stability_diag.png')
    p.add_argument('--R-lqr',      type=float, default=0.01,
                   help='LQR action cost (matches control.R_lqr in config)')
    p.add_argument('--device',     default=None)
    args = p.parse_args()

    device = (torch.device(args.device) if args.device else
              torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Device: {device}')

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model_cfg_dict = ckpt.get('config', {})
    print(f'Checkpoint config: {model_cfg_dict}')

    # Fill in any missing keys from yaml
    with open(args.config) as f:
        yaml_cfg = yaml.safe_load(f)
    for k, v in yaml_cfg['model'].items():
        if k not in model_cfg_dict:
            model_cfg_dict[k] = v

    from models.jepa import make_jepa
    model = make_jepa(
        variant=model_cfg_dict.get('variant', 'E-full'),
        latent_dim=int(model_cfg_dict.get('latent_dim', 8)),
        action_dim=int(model_cfg_dict.get('action_dim', 1)),
        action_latent_dim=int(model_cfg_dict.get('action_latent_dim', 8)),
        action_encoder=model_cfg_dict.get('action_encoder', 'linear'),
        encoder_type=model_cfg_dict.get('encoder_type', 'vit'),
        image_size=int(model_cfg_dict.get('image_size', 64)),
        patch_size=int(model_cfg_dict.get('patch_size', 8)),
        frame_stack=int(model_cfg_dict.get('frame_stack', 1)),
        use_frame_diff=bool(model_cfg_dict.get('use_frame_diff', False)),
        vit_embed_dim=int(model_cfg_dict.get('vit_embed_dim', 128)),
        vit_depth=int(model_cfg_dict.get('vit_depth', 3)),
        vit_num_heads=int(model_cfg_dict.get('vit_num_heads', 4)),
        vit_mlp_ratio=float(model_cfg_dict.get('vit_mlp_ratio', 2.0)),
        predictor_type=model_cfg_dict.get('predictor_type', 'transformer'),
        predictor_window=int(model_cfg_dict.get('predictor_window', 3)),
        predictor_embed_dim=int(model_cfg_dict.get('predictor_embed_dim', 128)),
        predictor_depth=int(model_cfg_dict.get('predictor_depth', 3)),
        predictor_num_heads=int(model_cfg_dict.get('predictor_num_heads', 4)),
        predictor_mlp_ratio=float(model_cfg_dict.get('predictor_mlp_ratio', 4.0)),
    )
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    print(f'Model loaded from {args.checkpoint}')

    # ── z* from near-eq observation ───────────────────────────────────────────
    print(f'Finding near-equilibrium observation in {args.data} ...')
    obs_np, eq_norm = _find_near_eq_obs_hdf5(args.data)
    print(f'Near-eq ||state||={eq_norm:.4f}')

    obs_t = torch.from_numpy(obs_np).float().permute(2,0,1).unsqueeze(0).div(255.0).to(device)
    with torch.no_grad():
        z_star = model.encode_obs(obs_t, obs_t).squeeze(0)   # (d,)
    z_np = z_star.cpu().numpy()

    # ── fp_err ────────────────────────────────────────────────────────────────
    W = model.config.predictor_window
    d = model.config.latent_dim
    act_dim = model.action_encoder.action_dim
    with torch.no_grad():
        z_win = z_star.unsqueeze(0).unsqueeze(0).expand(1, W, -1)
        u_win = torch.zeros(1, W, act_dim, device=device)
        z_pred = model.predict(z_win, u_win).squeeze(0)
    fp_err = float(torch.norm(z_pred - z_star).item())
    c_drift = (z_pred - z_star).cpu().numpy()   # affine offset in latent space
    print(f'fp_err = {fp_err:.6f}')

    # ── Augmented Jacobian ────────────────────────────────────────────────────
    print('Computing augmented Jacobian ...')
    from control.jacobian import compute_augmented_jacobian_np
    A_aug, B_aug = compute_augmented_jacobian_np(model, z_np, device)

    # Project B through action encoder weight: (Wd, m) @ (m, act_dim) → (Wd, act_dim)
    if hasattr(model.action_encoder, 'W'):
        W_enc = model.action_encoder.W.weight.detach().cpu().numpy()   # (m, act_dim)
        B_aug = B_aug @ W_enc    # (Wd, act_dim)

    eigs      = np.linalg.eigvals(A_aug)
    rho_learn = float(np.max(np.abs(eigs)))
    print('Computing GT rho from actual environment (finite-difference Jacobian) ...')
    rho_gt    = _gt_rho_from_env(yaml_cfg['environment'])
    print(f'rho(A_aug) learned = {rho_learn:.4f}   GT = {rho_gt:.4f}')
    print(f'||B_aug|| = {float(np.linalg.norm(B_aug)):.4f}')

    # ── z_ss via DARE ─────────────────────────────────────────────────────────
    Wd   = W * d
    R_lqr = args.R_lqr * np.eye(act_dim)
    Q_lqr = np.eye(Wd)
    c_aug = np.concatenate([c_drift, np.zeros((W-1)*d)])   # (Wd,)

    z_ss_norm = float('nan')
    K = None
    try:
        P     = sla.solve_discrete_are(A_aug, B_aug, Q_lqr, R_lqr)
        K     = np.linalg.solve(R_lqr + B_aug.T @ P @ B_aug,
                                B_aug.T @ P @ A_aug)          # (act_dim, Wd)
        A_cl  = A_aug - B_aug @ K
        eigs_cl = np.linalg.eigvals(A_cl)
        rho_cl  = float(np.max(np.abs(eigs_cl)))
        z_ss  = np.linalg.solve(np.eye(Wd) - A_cl, c_aug)   # (Wd,)
        z_ss_norm = float(np.linalg.norm(z_ss))
        print(f'z_ss = {z_ss_norm:.4f}   rho(A_cl) = {rho_cl:.4f}')
    except Exception as e:
        print(f'DARE failed: {e}')

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f'{Path(args.checkpoint).name}   '
        f'ρ(A_aug)={rho_learn:.4f} (GT={rho_gt:.4f})   '
        f'fp_err={fp_err:.4f}   '
        f'z_ss={"N/A" if np.isnan(z_ss_norm) else f"{z_ss_norm:.4f}"}',
        fontsize=11)

    # Panel 1: eigenvalues in complex plane
    ax = axes[0]
    theta_c = np.linspace(0, 2*np.pi, 300)
    ax.plot(np.cos(theta_c), np.sin(theta_c), 'k--', lw=0.8, label='unit circle')
    unstable = np.abs(eigs) >= 1.0
    ax.scatter(eigs[~unstable].real, eigs[~unstable].imag,
               c='steelblue', s=60, zorder=3, label='stable')
    ax.scatter(eigs[unstable].real, eigs[unstable].imag,
               c='crimson', s=80, marker='*', zorder=4, label='unstable')
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_title('Eigenvalues of A_aug', fontsize=10)
    ax.set_xlabel('Re'); ax.set_ylabel('Im')
    ax.legend(fontsize=8)
    ax.axhline(0, color='k', lw=0.4); ax.axvline(0, color='k', lw=0.4)

    # Panel 2: sorted eigenvalue magnitudes
    ax2 = axes[1]
    mags = np.sort(np.abs(eigs))[::-1]
    colors = ['crimson' if m >= 1.0 else 'steelblue' for m in mags]
    ax2.bar(range(len(mags)), mags, color=colors, edgecolor='none')
    ax2.axhline(1.0, color='k', lw=1.0, ls='--', label='unit circle')
    ax2.axhline(rho_gt, color='orange', lw=1.2, ls='--',
                label=f'GT ρ={rho_gt:.3f}')
    ax2.axhline(rho_learn, color='crimson', lw=1.2, ls='-',
                label=f'learned ρ={rho_learn:.3f}')
    ax2.set_title('|eigenvalues| (sorted)', fontsize=10)
    ax2.set_xlabel('index'); ax2.set_ylabel('|λ|')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # Panel 3: fp_err and z_ss summary
    ax3 = axes[2]
    ax3.axis('off')
    metrics = [
        ('fp_err',         f'{fp_err:.6f}',    fp_err < 0.05),
        ('z_ss',           f'{"N/A" if np.isnan(z_ss_norm) else f"{z_ss_norm:.4f}"}',
                           (not np.isnan(z_ss_norm)) and z_ss_norm < 1.0),
        ('ρ(A_aug)',       f'{rho_learn:.4f}',  True),
        ('ρ(A_aug) GT',    f'{rho_gt:.4f}',     True),
        ('ρ gap',          f'{rho_gt - rho_learn:.4f}',
                           abs(rho_gt - rho_learn) < 0.1),
        ('||B_aug||',      f'{float(np.linalg.norm(B_aug)):.4f}', True),
        ('DARE solved',    str(K is not None),  K is not None),
    ]
    y = 0.92
    for name, val, ok in metrics:
        color = '#2ca02c' if ok else '#d62728'
        ax3.text(0.05, y, f'{name}:', fontsize=11, transform=ax3.transAxes,
                 va='top', fontweight='bold')
        ax3.text(0.55, y, val, fontsize=11, transform=ax3.transAxes,
                 va='top', color=color)
        y -= 0.12
    ax3.set_title('Summary', fontsize=10)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches='tight')
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()
