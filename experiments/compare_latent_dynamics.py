"""Compare latent dynamics structure across multiple JEPA/AE models.

Generates a 3-panel figure per model:
  Col 1  Pearson r(z_i, state_j)    — how well latent dims encode physical state
  Col 2  Controllability Gramian λ  — how many controllable directions exist
  Col 3  PCA scatter (PC1 vs PC2)   — latent geometry colored by pole angle θ

Usage:
  python experiments/compare_latent_dynamics.py \\
      --ckpts  results/.../checkpoint_epoch0100.pt \\
               results/ae_seed44/model_final.pt \\
               results/.../checkpoint_epoch0100.pt \\
      --cfgs   configs/cartpole_v2_fullspec.yaml \\
               configs/cartpole_ae_baseline.yaml \\
               configs/cartpole_jepa_sigreg_baseline.yaml \\
      --labels "JEPA-ctrl" "AE-DMD" "JEPA-SIGreg" \\
      --out    results/comparison/latent_dynamics.png
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
from pathlib import Path


def _pca2(z: np.ndarray):
    """Return (z2, var_ratio) — first 2 principal components via SVD."""
    z_c = z - z.mean(axis=0)
    _, S, Vt = np.linalg.svd(z_c, full_matrices=False)
    z2 = z_c @ Vt[:2].T
    var = S ** 2 / (len(z) - 1)
    var_ratio = var[:2] / (var.sum() + 1e-12)
    return z2, var_ratio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ── Model loading ─────────────────────────────────────────────────────────────

def _build_jepa_config(d: dict):
    from models.jepa import JEPAConfig
    cfg = JEPAConfig()
    for k in ('latent_dim', 'action_latent_dim', 'action_encoder', 'image_size',
              'patch_size', 'frame_stack', 'vit_embed_dim', 'vit_depth',
              'vit_num_heads', 'predictor_window', 'predictor_hidden_dim',
              'predictor_n_layers', 'variant'):
        if k in d:
            setattr(cfg, k, d[k])
    return cfg


def load_model(ckpt_path: str, cfg_yaml: str, device):
    """Load a JEPA or AE model. Returns (model, frame_stack, label)."""
    from models.jepa import JEPAModel
    from models.autoencoder import AEWorldModel

    ckpt = torch.load(ckpt_path, map_location=device)

    # ── Determine model architecture ──────────────────────────────────────
    full_yaml_cfg = {}
    if cfg_yaml and Path(cfg_yaml).exists():
        with open(cfg_yaml) as f:
            full_yaml_cfg = yaml.safe_load(f)

    model_cfg_yaml = full_yaml_cfg.get('model', {})
    train_cfg_yaml = full_yaml_cfg.get('training', {})

    # Architecture dict: prefer checkpoint's saved config, fall back to yaml
    if isinstance(ckpt, dict) and 'config' in ckpt:
        arch = ckpt['config']
    else:
        arch = {
            'latent_dim':           int(model_cfg_yaml.get('latent_dim', 32)),
            'action_latent_dim':    int(model_cfg_yaml.get('action_latent_dim', 4)),
            'action_encoder':       model_cfg_yaml.get('action_encoder', 'linear'),
            'image_size':           int(full_yaml_cfg.get('environment', {}).get('image_size', 64)),
            'patch_size':           int(model_cfg_yaml.get('patch_size', 8)),
            'frame_stack':          int(model_cfg_yaml.get('frame_stack', 1)),
            'vit_embed_dim':        int(model_cfg_yaml.get('vit_embed_dim', 128)),
            'vit_depth':            int(model_cfg_yaml.get('vit_depth', 4)),
            'vit_num_heads':        int(model_cfg_yaml.get('vit_num_heads', 4)),
            'predictor_window':     int(model_cfg_yaml.get('predictor_window', 3)),
            'predictor_hidden_dim': int(model_cfg_yaml.get('predictor_hidden_dim', 256)),
            'predictor_n_layers':   int(model_cfg_yaml.get('predictor_n_layers', 2)),
        }

    jcfg = _build_jepa_config(arch)
    frame_stack = jcfg.frame_stack

    is_ae = (float(train_cfg_yaml.get('lambda_dmd_pixel', 0.0)) > 0
             or float(train_cfg_yaml.get('lambda_recon', 0.0)) > 0)
    model = AEWorldModel(jcfg) if is_ae else JEPAModel(jcfg)

    state = ckpt.get('model_state', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, frame_stack


# ── Observation helpers ───────────────────────────────────────────────────────

def _to_tensor(obs_hwc: np.ndarray, device) -> torch.Tensor:
    """HWC uint8 → (1, C, H, W) float32 [0,1]."""
    return torch.from_numpy(obs_hwc).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0


def _stack_obs(curr_hwc, prev_hwc, frame_stack: int, device) -> torch.Tensor:
    curr = _to_tensor(curr_hwc, device)
    if frame_stack > 1:
        prev = _to_tensor(prev_hwc, device)
        return torch.cat([prev, curr], dim=1)
    return curr


# ── Data collection ───────────────────────────────────────────────────────────

def collect_rollouts(env, model, frame_stack: int, device,
                     n_rollouts: int = 60, rollout_len: int = 50,
                     theta_range: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """Return (z_array, state_array) from random rollouts with varied θ."""
    zs, states = [], []
    rng = np.random.RandomState(0)

    for _ in range(n_rollouts):
        theta0 = rng.uniform(-theta_range, theta_range)
        x0 = np.array([rng.uniform(-0.3, 0.3), 0.0, theta0, 0.0], dtype=np.float32)
        obs, state, _ = env.reset_to_state(x0)
        prev_obs = obs.copy()

        for _ in range(rollout_len):
            obs_t = _stack_obs(obs, prev_obs, frame_stack, device)
            with torch.no_grad():
                z = model.encoder(obs_t).cpu().numpy()[0]
            zs.append(z)
            states.append(state.copy())

            action = env.sample_action()
            obs_next, state_next, _, done, _ = env.step(action)
            prev_obs = obs.copy()
            obs, state = obs_next, state_next
            if done:
                break

    return np.array(zs), np.array(states)


# ── Panel 1: Pearson r ────────────────────────────────────────────────────────

def compute_pearson_matrix(z: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Return (4, d) matrix of Pearson r between each state var and latent dim."""
    n_state = states.shape[1]
    d = z.shape[1]
    R = np.zeros((n_state, d))
    for j in range(n_state):
        for i in range(d):
            # Pearson r via correlation coefficient
            xi, sj = z[:, i], states[:, j]
            xi_c, sj_c = xi - xi.mean(), sj - sj.mean()
            denom = (np.linalg.norm(xi_c) * np.linalg.norm(sj_c) + 1e-12)
            R[j, i] = float(np.dot(xi_c, sj_c) / denom)
    return R  # (4, d)


def plot_pearson(ax, R: np.ndarray, title: str):
    """Heatmap of Pearson r. Latent dims sorted by max |r| across state vars."""
    sort_idx = np.argsort(-np.max(np.abs(R), axis=0))
    R_sorted = R[:, sort_idx]
    im = ax.imshow(R_sorted, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_yticks(range(4))
    ax.set_yticklabels(['x', 'ẋ', 'θ', 'θ̇'], fontsize=9)
    ax.set_xlabel('Latent dim (sorted by |r|)', fontsize=8)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Mark the max |r| per state var
    for j in range(4):
        best = int(np.argmax(np.abs(R_sorted[j])))
        ax.add_patch(plt.Rectangle((best - 0.5, j - 0.5), 1, 1,
                                   fill=False, edgecolor='lime', lw=1.5))


# ── Panel 2: Controllability Gramian ─────────────────────────────────────────

def compute_gramian_eigenvalues(model, z_star: np.ndarray, device,
                                T: int = 20) -> np.ndarray:
    """Controllability Gramian W_T = Σ A^k B_eff B_eff^T (A^T)^k, return sorted eigvals."""
    from control.jacobian import compute_jacobian_np
    A, B_jac = compute_jacobian_np(model, z_star, device)

    # B_eff: project B_jac through W_enc for raw scalar action
    if hasattr(model.action_encoder, 'W'):
        W_enc = model.action_encoder.W.weight.detach().cpu().numpy()  # (m, 1)
        B_eff = B_jac @ W_enc  # (d, 1)
    else:
        B_eff = B_jac  # (d, 1) already

    # Gramian accumulation
    d = A.shape[0]
    W_gram = np.zeros((d, d))
    AB = B_eff.copy()
    for _ in range(T):
        W_gram += AB @ AB.T
        AB = A @ AB

    eigvals = np.linalg.eigvalsh(W_gram)
    return np.sort(np.abs(eigvals))[::-1], A, B_eff


def plot_gramian(ax, eigvals: np.ndarray, title: str):
    """Log-scale bar chart of Gramian eigenvalue spectrum."""
    d = len(eigvals)
    # Clip for log scale
    ev_plot = np.maximum(eigvals, 1e-12)
    ax.bar(range(d), ev_plot, color='steelblue', alpha=0.8, width=0.8)
    ax.set_yscale('log')
    ax.set_xlabel('Eigenvalue index', fontsize=8)
    ax.set_ylabel('λ', fontsize=8)
    ax.set_title(title, fontsize=9)
    # Mark 1% threshold
    thresh = ev_plot[0] * 0.01
    ax.axhline(thresh, color='red', linestyle='--', linewidth=1, alpha=0.7, label='1% max')
    n_ctrl = int(np.sum(eigvals >= thresh))
    ax.text(0.98, 0.95, f'{n_ctrl}/{d} controllable',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            color='darkred')
    ax.legend(fontsize=7)


# ── Panel 3: PCA scatter ──────────────────────────────────────────────────────

def plot_pca(ax, z: np.ndarray, states: np.ndarray, title: str):
    """PC1 vs PC2 scatter colored by pole angle θ."""
    z2, var_ratio = _pca2(z)
    theta = states[:, 2]  # pole angle
    sc = ax.scatter(z2[:, 0], z2[:, 1], c=theta, cmap='coolwarm',
                    s=3, alpha=0.4, vmin=-0.6, vmax=0.6)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='θ (rad)')
    ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}%)', fontsize=8)
    ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}%)', fontsize=8)
    ax.set_title(title, fontsize=9)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpts',  nargs='+', required=True,
                        help='Checkpoint paths (1 per model)')
    parser.add_argument('--cfgs',   nargs='+', required=True,
                        help='Config yaml paths (matched by position to --ckpts)')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Model labels for plot titles')
    parser.add_argument('--out',    default='results/comparison/latent_dynamics.png',
                        help='Output figure path')
    parser.add_argument('--n_rollouts',  type=int, default=60)
    parser.add_argument('--rollout_len', type=int, default=50)
    parser.add_argument('--seed',        type=int, default=0)
    parser.add_argument('--gramian_T',   type=int, default=20)
    parser.add_argument('--device',      default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    n_models = len(args.ckpts)
    if len(args.cfgs) != n_models:
        raise ValueError(f'--ckpts ({n_models}) and --cfgs ({len(args.cfgs)}) must have same length')
    labels = args.labels or [f'Model {i}' for i in range(n_models)]
    if len(labels) != n_models:
        raise ValueError(f'--labels must have same length as --ckpts')

    device = torch.device(args.device)
    np.random.seed(args.seed)

    # ── Load models ───────────────────────────────────────────────────────
    models = []
    for ckpt_p, cfg_p in zip(args.ckpts, args.cfgs):
        print(f'[load] {Path(ckpt_p).name}  ({cfg_p})')
        model, frame_stack = load_model(ckpt_p, cfg_p, device)
        models.append((model, frame_stack, cfg_p))
        print(f'       frame_stack={frame_stack}  params={sum(p.numel() for p in model.parameters()):,}')

    # ── Build shared environment ──────────────────────────────────────────
    from envs.cartpole_visual import ContinuousCartpoleVisual
    env = ContinuousCartpoleVisual(image_size=64, action_range=(-10, 10))
    obs_eq, _, _ = env.reset_to_state(np.zeros(4, dtype=np.float32))

    # ── Collect data and compute metrics per model ────────────────────────
    print(f'\n[data] Collecting {args.n_rollouts} rollouts × {args.rollout_len} steps ...')
    all_data = []
    for (model, frame_stack, cfg_p), label in zip(models, labels):
        print(f'  {label} ...')
        zs, states = collect_rollouts(
            env, model, frame_stack, device,
            n_rollouts=args.n_rollouts, rollout_len=args.rollout_len)
        print(f'    collected {len(zs)} points')

        R = compute_pearson_matrix(zs, states)
        max_r_per_state = np.max(np.abs(R), axis=1)
        print(f'    max |r|: x={max_r_per_state[0]:.3f}  ẋ={max_r_per_state[1]:.3f}  '
              f'θ={max_r_per_state[2]:.3f}  θ̇={max_r_per_state[3]:.3f}')

        # z* for Gramian
        obs_eq_t = _to_tensor(obs_eq, device)
        if frame_stack > 1:
            obs_eq_t = torch.cat([obs_eq_t, obs_eq_t], dim=1)
        with torch.no_grad():
            z_star = model.encoder(obs_eq_t).cpu().numpy()[0]

        print(f'    computing Gramian (T={args.gramian_T}) ...')
        try:
            gramian_eigvals, A_jac, B_eff = compute_gramian_eigenvalues(
                model, z_star, device, T=args.gramian_T)
            rho = float(np.max(np.abs(np.linalg.eigvals(A_jac))))
            print(f'    rho(A_jac)={rho:.4f}  ||B_eff||={np.linalg.norm(B_eff):.4f}')
        except Exception as exc:
            print(f'    Gramian failed: {exc}')
            gramian_eigvals = None

        all_data.append({
            'label': label, 'zs': zs, 'states': states,
            'R': R, 'gramian_eigvals': gramian_eigvals,
        })

    env.close()

    # ── Plot ──────────────────────────────────────────────────────────────
    print('\n[plot] Generating figure ...')
    fig, axes = plt.subplots(n_models, 3, figsize=(14, 4 * n_models))
    if n_models == 1:
        axes = axes[np.newaxis, :]

    for row, d in enumerate(all_data):
        label = d['label']

        # Col 0: Pearson heatmap
        plot_pearson(axes[row, 0], d['R'],
                     title=f'{label}\nPearson r(z, state)')

        # Col 1: Gramian spectrum
        if d['gramian_eigvals'] is not None:
            plot_gramian(axes[row, 1], d['gramian_eigvals'],
                         title=f'{label}\nControllability Gramian λ')
        else:
            axes[row, 1].text(0.5, 0.5, 'Gramian N/A', ha='center', va='center',
                              transform=axes[row, 1].transAxes)
            axes[row, 1].set_title(f'{label}\nControllability Gramian λ', fontsize=9)

        # Col 2: PCA scatter
        plot_pca(axes[row, 2], d['zs'], d['states'],
                 title=f'{label}\nLatent PCA (color=θ)')

    fig.suptitle('Latent Dynamics Structure Comparison', fontsize=13, y=1.01)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f'[plot] Saved → {out_path}')


if __name__ == '__main__':
    main()
