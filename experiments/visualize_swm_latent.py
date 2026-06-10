"""Latent space visualization for SWM pilot runs (dataset-based, no GT required).

Encodes observations directly from the h5 dataset and produces a 3-panel figure:

  Panel A — 2D PCA of latent space, coloured by pole angle θ
    Shows whether the encoder captures θ and whether representations vary
    smoothly with angle.

  Panel B — θ vs. first principal latent component
    Linear regression + Pearson r quantifies how much variance in PC-1 is
    explained by θ.  A good encoder shows |r| ≈ 1; a collapsed encoder
    shows |r| ≈ 0.

  Panel C — PCA scatter coloured by sign(θ)
    Red = θ > 0, blue = θ < 0.  Tests whether the mirror-antisymmetry
    constraint (Run J) resolves the |θ| vs. θ sign degeneracy.
    Good: clean separation.  Baseline (Run F/A): red/blue overlap.

Usage:
    python experiments/visualize_swm_latent.py \\
        --checkpoint results/jepa_sigreg_fp_seed42/checkpoints/checkpoint_epoch0080.pt \\
        --config configs/cartpole_jepa_sigreg_fp.yaml \\
        --data data/cartpole_swm_pilot_large.h5 \\
        --label "Run-F (pred+SIGreg)" \\
        --out results/jepa_sigreg_fp_seed42/latent_viz.png
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
import yaml


def _load_model(ckpt_path: str, cfg_yaml: dict, device):
    """Load JEPAModel from a trainer checkpoint, inferring arch from saved config."""
    from models.jepa import make_jepa

    ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError(f'Unexpected checkpoint format: {type(ckpt)}')

    # Prefer architecture config embedded in the checkpoint; fall back to yaml
    arch = ckpt.get('config', {})
    model_yaml = cfg_yaml.get('model', {})

    model = make_jepa(
        variant=arch.get('variant', 'E-full'),
        latent_dim=int(arch.get('latent_dim', model_yaml.get('latent_dim', 8))),
        action_latent_dim=int(arch.get('action_latent_dim',
                                       model_yaml.get('action_latent_dim', 1))),
        action_encoder=arch.get('action_encoder',
                                model_yaml.get('action_encoder', 'linear')),
        encoder_type=arch.get('encoder_type',
                               model_yaml.get('encoder_type', 'vit')),
        image_size=int(arch.get('image_size', 64)),
        patch_size=int(arch.get('patch_size', model_yaml.get('patch_size', 8))),
        frame_stack=int(arch.get('frame_stack', model_yaml.get('frame_stack', 1))),
        vit_embed_dim=int(arch.get('vit_embed_dim',
                                   model_yaml.get('vit_embed_dim', 128))),
        vit_depth=int(arch.get('vit_depth', model_yaml.get('vit_depth', 4))),
        vit_num_heads=int(arch.get('vit_num_heads',
                                   model_yaml.get('vit_num_heads', 4))),
        predictor_hidden_dim=int(arch.get('predictor_hidden_dim',
                                          model_yaml.get('predictor_hidden_dim', 64))),
        predictor_n_layers=int(arch.get('predictor_n_layers',
                                        model_yaml.get('predictor_n_layers', 2))),
        predictor_window=int(arch.get('predictor_window',
                                      model_yaml.get('predictor_window', 3))),
    )
    missing, unexpected = model.load_state_dict(ckpt['model_state'], strict=False)
    if missing:
        print(f'  WARNING: missing keys: {missing}')
    if unexpected:
        print(f'  NOTE: ignoring unexpected keys: {unexpected[:3]}{"..." if len(unexpected)>3 else ""}')
    model.to(device).eval()
    frame_stack = int(arch.get('frame_stack', model_yaml.get('frame_stack', 1)))
    print(f'  Loaded epoch={ckpt.get("epoch", "?")}  '
          f'latent_dim={arch.get("latent_dim", "?")}  frame_stack={frame_stack}')
    return model, frame_stack


def _encode_dataset(model, data, split: str, frame_stack: int,
                    device, max_samples: int = 2000):
    """Encode observations from the dataset.

    Returns:
        zs     (N, d)  latent vectors
        states (N, 4)  physical states [x, ẋ, θ, θ̇]
    """
    idx = data['splits'][split]
    obs_all    = data['obs'][idx]        # (N, H, W, 3) uint8
    states_all = data['states'][idx]     # (N, 4) float

    N = min(len(obs_all), max_samples)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(obs_all), size=N, replace=False)
    sel.sort()

    zs_out     = []
    states_out = []

    def _to_tensor(arr):
        return torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0  # (3, H, W)

    for i in sel:
        curr = _to_tensor(obs_all[i])
        if frame_stack > 1:
            # Use previous obs (same episode if possible, otherwise duplicate)
            if i > 0:
                prev = _to_tensor(obs_all[i - 1])
            else:
                prev = curr
            obs_t = torch.cat([prev, curr], dim=0).unsqueeze(0).to(device)  # (1, 6, H, W)
        else:
            obs_t = curr.unsqueeze(0).to(device)  # (1, 3, H, W)

        with torch.no_grad():
            z = model.encoder(obs_t).squeeze(0).cpu().numpy()
        zs_out.append(z)
        states_out.append(states_all[i])

    return np.array(zs_out), np.array(states_out)


def make_figure(model, frame_stack, data, label: str, device,
                split: str = 'val', max_samples: int = 2000):
    """Build the 3-panel figure.  Returns (fig, Pearson-r-for-theta)."""

    print(f'  [{label}] Encoding {split} observations (up to {max_samples})...')
    zs, states = _encode_dataset(model, data, split, frame_stack, device, max_samples)
    thetas = states[:, 2]          # pole angle θ (radians)
    N, d   = zs.shape
    print(f'  [{label}] Encoded {N} observations, latent_dim={d}')

    # ── z* at equilibrium (θ=0) ─────────────────────────────────────────────
    eq_mask = np.abs(states).max(axis=1) < 0.05
    if eq_mask.sum() > 0:
        z_star = zs[eq_mask].mean(axis=0)
    else:
        z_star = np.zeros(d)

    # ── PCA ─────────────────────────────────────────────────────────────────
    dz = zs - z_star[None]
    U, S, Vt = np.linalg.svd(dz, full_matrices=False)
    proj = U[:, :2] * S[:2]     # (N, 2) projection onto top-2 PCA axes
    var_exp = S[:2] ** 2 / (S ** 2).sum()

    # ── Pearson r: θ vs. PC-1 ────────────────────────────────────────────────
    r_pc1 = float(np.corrcoef(thetas, proj[:, 0])[0, 1])
    # Also check PC-2 in case θ is better captured there
    r_pc2 = float(np.corrcoef(thetas, proj[:, 1])[0, 1])
    # Best single-latent-dim correlation
    r_per_dim = [float(np.corrcoef(thetas, zs[:, k])[0, 1]) for k in range(d)]
    best_k = int(np.argmax(np.abs(r_per_dim)))
    r_best = r_per_dim[best_k]

    print(f'  [{label}] PC-1 var={var_exp[0]:.1%}  PC-2 var={var_exp[1]:.1%}')
    print(f'  [{label}] Pearson r(θ, PC-1)={r_pc1:.3f}  r(θ, PC-2)={r_pc2:.3f}')
    print(f'  [{label}] Best single dim z[{best_k}]: r(θ, z[{best_k}])={r_best:.3f}')

    # Sign separation: fraction of θ>0 / θ<0 pairs that are correctly separated
    # by PC-1 sign (after accounting for overall sign flip)
    pos_mask = thetas > 0.02
    neg_mask = thetas < -0.02
    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
        mean_pos_pc1 = proj[pos_mask, 0].mean()
        mean_neg_pc1 = proj[neg_mask, 0].mean()
        sign_sep = float(np.sign(mean_pos_pc1 - mean_neg_pc1))
    else:
        sign_sep = 0.0

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Latent space — {label}', fontsize=13, fontweight='bold')
    theta_deg = np.degrees(thetas)
    cmap = 'coolwarm'

    # Panel A: PCA scatter coloured by θ
    ax = axes[0]
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=theta_deg, cmap=cmap,
                    s=8, alpha=0.6, rasterized=True)
    plt.colorbar(sc, ax=ax, label='θ (deg)')
    ax.plot(0, 0, 'k*', ms=10, zorder=5, label='$z^*$')
    ax.set_xlabel(f'PC-1 ({var_exp[0]:.1%} var)', fontsize=11)
    ax.set_ylabel(f'PC-2 ({var_exp[1]:.1%} var)', fontsize=11)
    ax.set_title('A. Latent PCA — coloured by θ', fontsize=11)
    ax.legend(fontsize=9)

    # Panel B: θ vs. PC-1
    ax = axes[1]
    ax.scatter(theta_deg, proj[:, 0], c=theta_deg, cmap=cmap,
               s=8, alpha=0.5, rasterized=True)
    # Trend line
    m, b_coef = np.polyfit(theta_deg, proj[:, 0], 1)
    xs_line = np.linspace(theta_deg.min(), theta_deg.max(), 100)
    ax.plot(xs_line, m * xs_line + b_coef, 'k--', lw=1.5, label=f'slope={m:.4f}')
    ax.axhline(0, color='gray', lw=0.8, ls=':')
    ax.axvline(0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel('Pole angle θ (degrees)', fontsize=11)
    ax.set_ylabel(f'PC-1 ({var_exp[0]:.1%} var)', fontsize=11)
    ax.set_title('B. θ vs. PC-1  (sign encoding)', fontsize=11)
    ax.legend(fontsize=9)
    ax.text(0.05, 0.95, f'Pearson r = {r_pc1:.3f}',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

    # Panel C: PCA scatter coloured by sign(θ)
    ax = axes[2]
    colors_sign = np.where(thetas > 0.02, 'tab:red',
                  np.where(thetas < -0.02, 'tab:blue', 'gray'))
    for color, label_str, mask in [
        ('tab:red',  'θ > 0', thetas > 0.02),
        ('tab:blue', 'θ < 0', thetas < -0.02),
        ('gray',     '|θ| ≤ 0.02 rad', np.abs(thetas) <= 0.02),
    ]:
        if mask.sum() > 0:
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       c=color, s=8, alpha=0.6, label=label_str, rasterized=True)
    ax.plot(0, 0, 'k*', ms=10, zorder=5, label='$z^*$')
    ax.set_xlabel(f'PC-1 ({var_exp[0]:.1%} var)', fontsize=11)
    ax.set_ylabel(f'PC-2 ({var_exp[1]:.1%} var)', fontsize=11)
    ax.set_title('C. Sign degeneracy test\n(red=θ>0, blue=θ<0)', fontsize=11)
    ax.legend(fontsize=9, markerscale=2)

    n_pos = (thetas > 0.02).sum()
    n_neg = (thetas < -0.02).sum()
    ax.text(0.05, 0.95,
            f'n(θ>0)={n_pos}  n(θ<0)={n_neg}\nPC-1 sep: {sign_sep:+.0f}',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    return fig, r_pc1


def main():
    p = argparse.ArgumentParser(
        description='Latent visualization for SWM pilot JEPA checkpoints')
    p.add_argument('--checkpoint', required=True,
                   help='Trainer checkpoint (.pt)')
    p.add_argument('--config',     required=True,
                   help='YAML config used to train this model')
    p.add_argument('--data',       required=True,
                   help='Path to cartpole_swm_pilot_large.h5')
    p.add_argument('--label',      default=None,
                   help='Figure title label (defaults to checkpoint dir name)')
    p.add_argument('--out',        required=True,
                   help='Output PNG path')
    p.add_argument('--split',      default='val',
                   choices=['train', 'val', 'test'],
                   help='Dataset split to visualize')
    p.add_argument('--max-samples', type=int, default=2000,
                   help='Max observations to encode')
    p.add_argument('--device',     default=None)
    args = p.parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    label = args.label or Path(args.checkpoint).parent.name

    from data.dataset import load_dataset
    print(f'[data] Loading {args.data}')
    data = load_dataset(args.data)

    print(f'[viz] Loading {args.checkpoint}')
    model, frame_stack = _load_model(args.checkpoint, cfg, device)

    fig, r_theta = make_figure(model, frame_stack, data, label, device,
                               split=args.split,
                               max_samples=args.max_samples)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[viz] Saved → {out_path}')
    print(f'[viz] Summary: Pearson r(θ, PC-1) = {r_theta:.3f}')


if __name__ == '__main__':
    main()
