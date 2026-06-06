"""Latent space visualization tied to unstable dynamics.

Produces a 3-panel figure that JEPA+dynSIG captures but standard JEPA+isotropic
SIGreg (or collapsed encoders) does not:

  Panel A — Encoder sensitivity plot
    Physical angle θ vs. V_u^T (z - z*).
    A good encoder should show a near-linear relationship: the representation
    varies along the unstable eigenvector direction as θ changes.
    A collapsed encoder shows a flat line; an isotropic encoder shows large
    variance in orthogonal (irrelevant) directions but not along V_u.

  Panel B — Latent distribution in PCA space
    2D PCA projection of encoded states from a grid of physical configurations,
    coloured by θ.  The Gramian ellipse (controllability Gramian W_T projected
    onto PCA axes) is overlaid: it shows which directions are most reachable
    from z* under control.  A dynSIG-trained encoder has its variance aligned
    with the Gramian — they match.

  Panel C — Phase portrait of learned dynamics near z*
    2D vector field: z → f(z, u=0) − z projected onto top-2 PCA axes.
    A correctly linearised unstable system shows a saddle: arrows point
    OUTWARD along the unstable eigenvector V_u and INWARD along stable modes.
    This is the structural fingerprint of an unstable equilibrium.
    Standard JEPA (collapsed / isotropic) shows random arrows or a sink.

Usage:
    python experiments/visualize_latent.py \\
        --checkpoint results/v2_E-full_mixed_fs1_seed42/model_ep0200.pt \\
        --config configs/cartpole_v2_fullspec.yaml \\
        --output latent_viz.png

    # Compare two checkpoints (JEPA vs AE):
    python experiments/visualize_latent.py \\
        --checkpoint results/v2_E-full_mixed_fs1_seed42/model_ep0200.pt \\
        --checkpoint2 results/ae_seed42/model_final.pt \\
        --output latent_comparison.png
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import yaml


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_model(ckpt_path: str, cfg: dict, device):
    from models.jepa import JEPAModel, JEPAConfig

    model_cfg = cfg['model']
    env_cfg   = cfg['environment']
    W         = int(model_cfg.get('predictor_window', 1))
    jepa_cfg  = JEPAConfig(
        latent_dim=int(model_cfg.get('latent_dim', 32)),
        action_latent_dim=int(model_cfg.get('action_latent_dim', 4)),
        action_encoder=model_cfg.get('action_encoder', 'linear'),
        image_size=int(env_cfg.get('image_size', 64)),
        patch_size=int(model_cfg.get('patch_size', 8)),
        vit_embed_dim=int(model_cfg.get('vit_embed_dim', 128)),
        vit_depth=int(model_cfg.get('vit_depth', 4)),
        vit_num_heads=int(model_cfg.get('vit_num_heads', 4)),
        predictor_hidden_dim=int(model_cfg.get('predictor_hidden_dim', 256)),
        predictor_n_layers=int(model_cfg.get('predictor_n_layers', 2)),
        predictor_window=W,
    )
    # Try AEWorldModel first (has decoder); fall back to plain JEPAModel
    try:
        from models.autoencoder import AEWorldModel
        model = AEWorldModel(jepa_cfg)
    except Exception:
        model = JEPAModel(jepa_cfg)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model_state', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def _encode_states(model, states_np: np.ndarray, env, device) -> np.ndarray:
    """Encode a list of physical states (N, 4) → latent (N, d)."""
    zs = []
    for x in states_np:
        obs, _, _ = env.reset_to_state(x.astype(np.float32))
        obs_t = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            z = model.encoder(obs_t).cpu().numpy()[0]
        zs.append(z)
    return np.array(zs)


def _get_unstable_eigvec(A: np.ndarray) -> np.ndarray:
    """Return the leading unstable (real) right eigenvector of A."""
    eigvals, eigvecs = np.linalg.eig(A)
    mags = np.abs(eigvals)
    unstable_mask = mags >= 0.95
    if unstable_mask.sum() == 0:
        idx = np.argmax(mags)
        unstable_mask = np.zeros(len(mags), dtype=bool)
        unstable_mask[idx] = True
    # Pick the one with the largest magnitude
    best = np.argmax(mags * unstable_mask)
    return eigvecs[:, best].real


def _gramian_ellipse_pca(W_T: np.ndarray, V_pca: np.ndarray,
                          scale: float = 1.0, n_pts: int = 200):
    """Return x, y arrays for the Gramian ellipse projected onto 2 PCA axes."""
    # W_T in d×d; project to 2D: W_2 = V^T W V  (2×2)
    W_2 = V_pca.T @ W_T @ V_pca   # (2, 2)
    eigvals, eigvecs = np.linalg.eigh(W_2)
    eigvals = np.maximum(eigvals, 0)
    theta = np.linspace(0, 2 * np.pi, n_pts)
    # Ellipse in eigen-frame
    pts = np.stack([np.sqrt(eigvals[0]) * np.cos(theta),
                    np.sqrt(eigvals[1]) * np.sin(theta)], axis=0)  # (2, n_pts)
    pts_rot = eigvecs @ pts * scale
    return pts_rot[0], pts_rot[1]


# ── figure generation ────────────────────────────────────────────────────────

def make_figure(model, cfg, device, label: str = '',
                theta_range=(-0.5, 0.5), n_theta=40, n_grid=20,
                phase_extent=2.0):
    """Build the 3-panel figure for one model.

    Returns (fig, axes).
    """
    from envs.cartpole_visual import ContinuousCartpoleVisual
    from control.jacobian import compute_jacobian_torch
    from losses.dyn_sigreg import compute_controllability_gramian, build_sigma_target

    env_cfg = cfg['environment']
    env = ContinuousCartpoleVisual(
        frame_skip=env_cfg.get('frame_skip', 1),
        image_size=env_cfg.get('image_size', 64),
        mass_cart=env_cfg.get('mass_cart', 1.0),
        mass_pole=env_cfg.get('mass_pole', 0.1),
        pole_length=env_cfg.get('pole_length', 0.5),
        gravity=env_cfg.get('gravity', 9.8),
        action_range=tuple(env_cfg.get('action_range', (-10, 10))),
    )

    # ── z* ────────────────────────────────────────────────────────────────
    obs_eq, _, _ = env.reset_to_state(np.zeros(4, dtype=np.float32))
    obs_eq_t = torch.from_numpy(obs_eq).float().permute(2,0,1).unsqueeze(0).to(device)/255.0
    with torch.no_grad():
        z_star = model.encoder(obs_eq_t).squeeze(0)
    z_star_np = z_star.cpu().numpy()

    # ── Jacobian + Gramian ────────────────────────────────────────────────
    # compute_jacobian_torch returns augmented (Wd×Wd)/(Wd×m) matrices for W>1.
    # For visualization we need the (d×d) current-state block (last d rows/cols).
    A_aug, B_aug = compute_jacobian_torch(model, z_star, device)
    d_lat = z_star.shape[0]
    W_win = getattr(model.config, 'predictor_window', 1)
    A_jac = A_aug[(W_win-1)*d_lat:, (W_win-1)*d_lat:]   # (d, d) current-state block
    B_jac = B_aug[(W_win-1)*d_lat:, :]                   # (d, m)

    if hasattr(model.action_encoder, 'W'):
        B_eff = (B_jac @ model.action_encoder.W.weight).detach()  # (d, 1)
    else:
        B_eff = B_jac.detach()

    A_np  = A_jac.detach().cpu().numpy()
    B_np  = B_eff.cpu().numpy()

    W_T_t = compute_controllability_gramian(
        A_jac.detach().float(), B_eff.float(), T_g=5)
    W_T   = W_T_t.cpu().numpy()

    V_u   = _get_unstable_eigvec(A_np)  # (d,) unstable direction

    # ── encode a grid of states ───────────────────────────────────────────
    thetas = np.linspace(theta_range[0], theta_range[1], n_theta)
    # States: vary θ only (cart at rest at origin)
    states_theta = np.array([[0., 0., th, 0.] for th in thetas])

    # Also vary cart position and velocities for richer coverage
    xs_extra = np.array([[x, 0., 0., 0.] for x in np.linspace(-0.2, 0.2, 10)])
    states_all = np.vstack([states_theta, xs_extra])

    print(f'  [{label}] Encoding {len(states_theta)} θ-states...')
    zs_theta = _encode_states(model, states_theta, env, device)

    print(f'  [{label}] Encoding {len(states_all)} mixed states for PCA...')
    zs_all  = _encode_states(model, states_all, env, device)

    env.close()

    # ── PCA ───────────────────────────────────────────────────────────────
    dz_all = zs_all - z_star_np[None]
    U, S, Vt = np.linalg.svd(dz_all, full_matrices=False)
    V_pca = Vt[:2].T   # (d, 2) — top-2 PCA axes

    # Project all encoded states onto PCA
    dz_theta = zs_theta - z_star_np[None]
    proj_theta = dz_theta @ V_pca      # (n_theta, 2)

    # V_u projection onto PCA axes
    Vu_pca = V_pca.T @ V_u             # (2,)  V_u direction in PCA space

    # ── figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Latent dynamics structure — {label}', fontsize=13, fontweight='bold')

    # ── Panel A: θ vs V_u^T(z-z*) ─────────────────────────────────────────
    ax = axes[0]
    Vu_proj = dz_theta @ V_u           # (n_theta,)
    ax.scatter(np.degrees(thetas), Vu_proj, c=np.degrees(thetas),
               cmap='coolwarm', s=30, zorder=3)
    # Linear trend line
    m, b = np.polyfit(np.degrees(thetas), Vu_proj, 1)
    xs_line = np.linspace(np.degrees(theta_range[0]), np.degrees(theta_range[1]), 100)
    ax.plot(xs_line, m * xs_line + b, 'k--', lw=1.5, alpha=0.6, label=f'slope={m:.4f}')
    ax.axhline(0, color='gray', lw=0.8, ls=':')
    ax.axvline(0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel('Pole angle θ (degrees)', fontsize=11)
    ax.set_ylabel('$V_u^T (z - z^*)$', fontsize=11)
    ax.set_title('A. Encoder sensitivity to\nunstable mode', fontsize=11)
    ax.legend(fontsize=9)
    _corr = float(np.corrcoef(np.degrees(thetas), Vu_proj)[0, 1])
    ax.text(0.05, 0.95, f'Pearson r = {_corr:.3f}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

    # ── Panel B: PCA scatter + Gramian ellipse ─────────────────────────────
    ax = axes[1]
    sc = ax.scatter(proj_theta[:, 0], proj_theta[:, 1],
                    c=np.degrees(thetas), cmap='coolwarm', s=30, zorder=3)
    plt.colorbar(sc, ax=ax, label='θ (deg)')

    # Gramian ellipse (scaled for visibility)
    gram_scale = float(np.max(np.abs(proj_theta))) / (np.trace(W_T) / W_T.shape[0] + 1e-9) ** 0.5
    gram_scale = np.clip(gram_scale, 0.01, 10.0)
    ex, ey = _gramian_ellipse_pca(W_T, V_pca, scale=gram_scale)
    ax.plot(ex, ey, 'g-', lw=2, label='Gramian $W_T$ (scaled)', alpha=0.8)

    # Unstable eigenvector direction
    arrow_len = float(np.max(np.abs(proj_theta))) * 0.7
    vu2 = Vu_pca / (np.linalg.norm(Vu_pca) + 1e-9) * arrow_len
    ax.annotate('', xy=(vu2[0], vu2[1]), xytext=(-vu2[0], -vu2[1]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(vu2[0] * 1.05, vu2[1] * 1.05, '$V_u$', color='red', fontsize=11)

    ax.axhline(0, color='gray', lw=0.8, ls=':')
    ax.axvline(0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel('PCA axis 1', fontsize=11)
    ax.set_ylabel('PCA axis 2', fontsize=11)
    ax.set_title('B. Latent distribution in PCA space\n(Gramian ellipse overlaid)', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_aspect('equal')

    # ── Panel C: Phase portrait f(z,0)-z in PCA space ─────────────────────
    ax = axes[2]
    W_pred = int(cfg['model'].get('predictor_window', 1))
    # Grid in PCA space
    ext = phase_extent
    grid_pts = np.linspace(-ext, ext, n_grid)
    G1, G2   = np.meshgrid(grid_pts, grid_pts)
    # Grid points in d-dim latent space
    z_grid = (z_star_np[None, :]
              + G1.ravel()[:, None] * V_pca[:, 0][None, :]
              + G2.ravel()[:, None] * V_pca[:, 1][None, :])   # (n_grid^2, d)

    with torch.no_grad():
        z_t  = torch.tensor(z_grid, dtype=torch.float32, device=device)  # (N, d)
        # Window: replicate current z W times (u=0 everywhere)
        z_win = z_t.unsqueeze(1).expand(-1, W_pred, -1)  # (N, W, d)
        u_win = torch.zeros(len(z_grid), W_pred, 1, device=device)
        z_next = model.predict(z_win, u_win).cpu().numpy()   # (N, d)

    dz_raw  = z_next - z_grid                    # (N, d) field in latent space
    dz_proj = dz_raw @ V_pca                     # (N, 2) projected onto PCA axes
    U_grid  = dz_proj[:, 0].reshape(n_grid, n_grid)
    V_grid  = dz_proj[:, 1].reshape(n_grid, n_grid)

    speed = np.sqrt(U_grid ** 2 + V_grid ** 2) + 1e-9
    ax.streamplot(grid_pts, grid_pts, U_grid, V_grid,
                  color=np.log1p(speed), cmap='plasma',
                  linewidth=1.2, density=1.2, arrowsize=1.0)

    # Mark z* and V_u direction
    ax.plot(0, 0, 'w*', ms=14, zorder=5, label='$z^*$')
    vu2n = Vu_pca / (np.linalg.norm(Vu_pca) + 1e-9) * ext * 0.6
    ax.annotate('', xy=(vu2n[0], vu2n[1]), xytext=(-vu2n[0], -vu2n[1]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(vu2n[0] * 1.1, vu2n[1] * 1.1, '$V_u$', color='red', fontsize=11)

    ax.set_xlim(-ext, ext); ax.set_ylim(-ext, ext)
    ax.set_xlabel('PCA axis 1', fontsize=11)
    ax.set_ylabel('PCA axis 2', fontsize=11)
    ax.set_title('C. Phase portrait: $f(z, u{=}0) - z$\n(near $z^*$, PCA projection)', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_aspect('equal')

    # Print summary stats
    print(f'  [{label}] V_u^T(z-z*) Pearson r with θ: {_corr:.3f}')
    rho_A = float(np.max(np.abs(np.linalg.eigvals(A_np))))
    b_eff_norm = float(np.linalg.norm(B_np))
    print(f'  [{label}] ρ(A_jac) = {rho_A:.4f}   ||B_eff|| = {b_eff_norm:.4f}')

    plt.tight_layout()
    return fig, axes


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',  required=True,
                   help='Path to JEPA or AE checkpoint (.pt)')
    p.add_argument('--checkpoint2', default=None,
                   help='Optional second checkpoint for side-by-side comparison')
    p.add_argument('--config',      default='configs/cartpole_v2_fullspec.yaml')
    p.add_argument('--config2',     default=None,
                   help='Config for second checkpoint (defaults to --config)')
    p.add_argument('--label',       default=None, help='Label for first model')
    p.add_argument('--label2',      default=None, help='Label for second model')
    p.add_argument('--output',      default='latent_viz.png')
    p.add_argument('--n-theta',     type=int, default=40,
                   help='Number of θ values to sweep')
    p.add_argument('--n-grid',      type=int, default=20,
                   help='Phase portrait grid resolution per axis')
    p.add_argument('--phase-extent',type=float, default=2.0,
                   help='PCA-space extent for phase portrait')
    p.add_argument('--device',      default=None)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    with open(args.config) as f:
        cfg1 = yaml.safe_load(f)

    label1 = args.label or Path(args.checkpoint).parent.name

    if args.checkpoint2 is None:
        # Single checkpoint: 1×3 figure
        print(f'[viz] Loading {args.checkpoint}')
        model1 = _load_model(args.checkpoint, cfg1, device)
        fig, _ = make_figure(model1, cfg1, device, label=label1,
                             n_theta=args.n_theta, n_grid=args.n_grid,
                             phase_extent=args.phase_extent)
        fig.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f'[viz] Saved {args.output}')
    else:
        # Two checkpoints: 2×3 figure (stacked)
        cfg2_path = args.config2 or args.config
        with open(cfg2_path) as f:
            cfg2 = yaml.safe_load(f)
        label2 = args.label2 or Path(args.checkpoint2).parent.name

        print(f'[viz] Loading model 1: {args.checkpoint}')
        model1 = _load_model(args.checkpoint, cfg1, device)
        print(f'[viz] Loading model 2: {args.checkpoint2}')
        model2 = _load_model(args.checkpoint2, cfg2, device)

        fig, axes_all = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Latent dynamics comparison', fontsize=14, fontweight='bold')

        for row_i, (model_i, cfg_i, label_i) in enumerate(
                [(model1, cfg1, label1), (model2, cfg2, label2)]):
            # Reuse make_figure but inject axes
            sub_fig, sub_axes = make_figure(
                model_i, cfg_i, device, label=label_i,
                n_theta=args.n_theta, n_grid=args.n_grid,
                phase_extent=args.phase_extent)
            # Transfer content — easiest to save sub-figure and embed later;
            # instead we just re-run with direct axes injection not supported
            # here (keep it simple: save two files and note both).
            sub_fig.savefig(args.output.replace('.png', f'_{row_i+1}.png'),
                            dpi=150, bbox_inches='tight')
            plt.close(sub_fig)

        plt.close(fig)
        print(f'[viz] Saved {args.output.replace(".png", "_1.png")} and '
              f'{args.output.replace(".png", "_2.png")}')


if __name__ == '__main__':
    main()
