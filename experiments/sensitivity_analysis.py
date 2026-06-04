"""Standalone encoder sensitivity analysis for JEPA and AE models.

Sweeps pole angle θ over a fine grid and reports:
  ||z(θ) - z*||          — total latent distance from equilibrium
  V_u^T (z(θ) - z*)      — projection onto the unstable eigenvector
  cos_angle              — alignment with V_u direction

Works for any model saved by run_experiment.py or train_ae.py.

Usage:
    # Single model
    python experiments/sensitivity_analysis.py \
        --checkpoint results/v2_E-full_mixed_fs1_seed43/model_final.pt \
        --config configs/cartpole_v2_fullspec.yaml

    # Compare JEPA and AE side-by-side
    python experiments/sensitivity_analysis.py \
        --checkpoint  results/v2_E-full_mixed_fs1_seed43/model_final.pt \
        --checkpoint2 results/ae_seed43/model_final.pt \
        --config  configs/cartpole_v2_fullspec.yaml \
        --config2 configs/cartpole_v2_fullspec.yaml \
        --output  sensitivity_comparison.png
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_model(ckpt_path: str, cfg: dict, device):
    from models.jepa import JEPAModel, JEPAConfig

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model_state', ckpt) if isinstance(ckpt, dict) else ckpt

    # Prefer embedded config (always matches the saved weights)
    embedded = ckpt.get('config') if isinstance(ckpt, dict) else None
    if embedded:
        model_cfg = embedded
        env_image_size = int(embedded.get('image_size', cfg['environment'].get('image_size', 64)))
        print(f'  [load] using embedded config: image_size={env_image_size}  patch_size={embedded.get("patch_size")}')
    else:
        model_cfg = cfg['model']
        env_image_size = int(cfg['environment'].get('image_size', 64))

    W = int(model_cfg.get('predictor_window', 1))
    jepa_cfg = JEPAConfig(
        latent_dim        = int(model_cfg.get('latent_dim', 32)),
        action_latent_dim = int(model_cfg.get('action_latent_dim', 4)),
        action_encoder    = model_cfg.get('action_encoder', 'linear'),
        image_size        = env_image_size,
        patch_size        = int(model_cfg.get('patch_size', 8)),
        vit_embed_dim     = int(model_cfg.get('vit_embed_dim', 128)),
        vit_depth         = int(model_cfg.get('vit_depth', 4)),
        vit_num_heads     = int(model_cfg.get('vit_num_heads', 4)),
        predictor_hidden_dim = int(model_cfg.get('predictor_hidden_dim', 256)),
        predictor_n_layers   = int(model_cfg.get('predictor_n_layers', 2)),
        predictor_window  = W,
    )
    try:
        from models.autoencoder import AEWorldModel
        model = AEWorldModel(jepa_cfg)
    except Exception:
        model = JEPAModel(jepa_cfg)

    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, jepa_cfg


def make_env(cfg: dict, image_size: int | None = None):
    from envs.cartpole_visual import ContinuousCartpoleVisual
    env_cfg = cfg['environment']
    return ContinuousCartpoleVisual(
        frame_skip   = env_cfg.get('frame_skip', 1),
        image_size   = image_size or int(env_cfg.get('image_size', 64)),
        mass_cart    = env_cfg.get('mass_cart', 1.0),
        mass_pole    = env_cfg.get('mass_pole', 0.1),
        pole_length  = env_cfg.get('pole_length', 0.5),
        gravity      = env_cfg.get('gravity', 9.8),
        action_range = tuple(env_cfg.get('action_range', (-10, 10))),
    )


def encode_state(model, env, state_np, device):
    obs, _, _ = env.reset_to_state(state_np.astype(np.float32))
    obs_t = torch.from_numpy(obs).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        return model.encoder(obs_t).squeeze(0).cpu().numpy()


def run_sensitivity(model, cfg, device, label: str, n_theta: int = 60, image_size: int | None = None):
    """Return dict of sensitivity metrics over θ grid."""
    from control.jacobian import compute_jacobian_torch

    env = make_env(cfg, image_size=image_size)

    # z*
    obs_eq, _, _ = env.reset_to_state(np.zeros(4, dtype=np.float32))
    obs_eq_t = torch.from_numpy(obs_eq).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        z_star = model.encoder(obs_eq_t).squeeze(0)
    z_star_np = z_star.cpu().numpy()

    # Jacobian → unstable eigenvector V_u
    A_jac, B_jac = compute_jacobian_torch(model, z_star, device)
    A_np = A_jac.detach().cpu().numpy()
    eigvals, eigvecs = np.linalg.eig(A_np)
    mags = np.abs(eigvals)
    idx_u = np.argmax(mags)
    V_u = eigvecs[:, idx_u].real
    V_u = V_u / np.linalg.norm(V_u)

    rho = mags[idx_u]
    print(f'\n[{label}] z* norm={np.linalg.norm(z_star_np):.3f}  rho(A)={rho:.4f}')
    print(f'[{label}] V_u aligned with B: cos={float(np.abs(V_u @ A_jac.detach().cpu().numpy().T @ z_star_np)):.3f}')

    thetas = np.linspace(-0.5, 0.5, n_theta)
    dz_norms, vu_projs, cos_angles = [], [], []

    for th in thetas:
        z = encode_state(model, env, np.array([0., 0., th, 0.]), device)
        dz = z - z_star_np
        norm = float(np.linalg.norm(dz))
        proj = float(dz @ V_u)
        cos  = float(abs(proj) / (norm + 1e-8))
        dz_norms.append(norm)
        vu_projs.append(proj)
        cos_angles.append(cos)

    # Print table at key angles (env still open)
    print(f'\n[{label}] Encoder sensitivity sweep:')
    print(f'  {"θ (deg)":>8}  {"θ (rad)":>8}  {"||z-z*||":>10}  {"V_u^T dz":>10}  {"cos":>6}')
    print('  ' + '-'*52)
    key_thetas = [-0.40, -0.20, -0.10, -0.05, -0.02, 0.00, +0.02, +0.05, +0.10, +0.20, +0.40]
    for th in key_thetas:
        z = encode_state(model, env, np.array([0., 0., th, 0.]), device)
        dz = z - z_star_np
        norm = float(np.linalg.norm(dz))
        proj = float(dz @ V_u)
        cos  = float(abs(proj) / (norm + 1e-8))
        print(f'  {np.degrees(th):>8.1f}  {th:>8.3f}  {norm:>10.4f}  {proj:>10.4f}  {cos:>6.3f}')
    env.close()

    # Fixed-point error
    W = int(cfg['model'].get('predictor_window', 1))
    z_star_win = z_star.unsqueeze(0).unsqueeze(0).expand(1, W, -1)
    u_zero = torch.zeros(1, W, 1, device=device)
    with torch.no_grad():
        z_fp_pred = model.predict(z_star_win, u_zero)
    fp_err = float((z_fp_pred.squeeze(0) - z_star).norm().cpu())
    print(f'[{label}] Predictor fp_err ||f(z*,0)-z*|| = {fp_err:.4f}')

    return {
        'label':      label,
        'thetas':     thetas,
        'dz_norms':   np.array(dz_norms),
        'vu_projs':   np.array(vu_projs),
        'cos_angles': np.array(cos_angles),
        'z_star_np':  z_star_np,
        'V_u':        V_u,
        'rho':        rho,
        'fp_err':     fp_err,
    }


def plot_sensitivity(results: list[dict], output: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['tab:blue', 'tab:orange']

    for i, res in enumerate(results):
        c = colors[i % len(colors)]
        th_deg = np.degrees(res['thetas'])
        axes[0].plot(th_deg, res['dz_norms'],   color=c, lw=2, label=res['label'])
        axes[1].plot(th_deg, res['vu_projs'],   color=c, lw=2, label=res['label'])
        axes[2].plot(th_deg, res['cos_angles'], color=c, lw=2, label=res['label'])

    for ax, title, ylabel in zip(
        axes[:3],
        ['||z(θ) - z*||  (total latent distance)',
         'V_u^T (z - z*)  (unstable mode projection)',
         'cos(z-z*, V_u)  (alignment with unstable dir)'],
        ['||z - z*||', 'V_u^T (z - z*)', 'cos angle'],
    ):
        ax.set_xlabel('Pole angle θ (degrees)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.axvline(0, color='gray', lw=0.8, ls=':')
        ax.axhline(0, color='gray', lw=0.8, ls=':')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Annotate with fp_err
    for i, res in enumerate(results):
        fp_txt = f'{res["label"]}: fp_err={res["fp_err"]:.4f}, ρ={res["rho"]:.4f}'
        axes[0].text(0.02, 0.95 - i * 0.08, fp_txt,
                     transform=axes[0].transAxes, fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='wheat', alpha=0.8))

    fig.suptitle('Encoder Sensitivity Analysis — θ sweep', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches='tight')
    print(f'\n[viz] Saved → {output}')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',  required=True, help='JEPA or AE checkpoint (.pt)')
    p.add_argument('--checkpoint2', default=None,  help='Optional second checkpoint for comparison')
    p.add_argument('--config',      default='configs/cartpole_v2_fullspec.yaml')
    p.add_argument('--config2',     default=None,  help='Config for second checkpoint (defaults to --config)')
    p.add_argument('--label',       default=None)
    p.add_argument('--label2',      default=None)
    p.add_argument('--output',      default='sensitivity_analysis.png')
    p.add_argument('--n-theta',     type=int, default=60)
    p.add_argument('--device',      default=None)
    args = p.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'[device] {device}')

    with open(args.config) as f:
        cfg1 = yaml.safe_load(f)

    label1 = args.label or Path(args.checkpoint).parent.name
    model1, jcfg1 = load_model(args.checkpoint, cfg1, device)
    res1 = run_sensitivity(model1, cfg1, device, label1,
                           n_theta=args.n_theta, image_size=jcfg1.image_size)

    all_results = [res1]

    if args.checkpoint2:
        cfg2_path = args.config2 or args.config
        with open(cfg2_path) as f:
            cfg2 = yaml.safe_load(f)
        label2 = args.label2 or Path(args.checkpoint2).parent.name
        model2, jcfg2 = load_model(args.checkpoint2, cfg2, device)
        res2 = run_sensitivity(model2, cfg2, device, label2,
                               n_theta=args.n_theta, image_size=jcfg2.image_size)
        all_results.append(res2)

    plot_sensitivity(all_results, args.output)


if __name__ == '__main__':
    main()
