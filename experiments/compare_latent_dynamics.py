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

    state = ckpt.get('model_state', ckpt) if isinstance(ckpt, dict) else ckpt

    # Auto-detect in_chans from checkpoint weights so old (fs=1) checkpoints
    # load correctly even after the yaml was updated to frame_stack=2.
    if 'encoder.patch_embed.proj.weight' in state:
        in_chans_ckpt = state['encoder.patch_embed.proj.weight'].shape[1]
    elif 'encoder.net.0.weight' in state:
        in_chans_ckpt = state['encoder.net.0.weight'].shape[1]
    else:
        in_chans_ckpt = None

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
        yaml_fs = int(model_cfg_yaml.get('frame_stack', 1))
        # Override frame_stack with the value inferred from checkpoint weights
        # so visualisation works on old checkpoints after the yaml was updated.
        if in_chans_ckpt is not None:
            actual_fs = in_chans_ckpt // 3
            if actual_fs != yaml_fs:
                print(f'[load] frame_stack mismatch: yaml={yaml_fs}, '
                      f'checkpoint in_chans={in_chans_ckpt} → using fs={actual_fs}')
        else:
            actual_fs = yaml_fs
        arch = {
            'latent_dim':           int(model_cfg_yaml.get('latent_dim', 32)),
            'action_latent_dim':    int(model_cfg_yaml.get('action_latent_dim', 4)),
            'action_encoder':       model_cfg_yaml.get('action_encoder', 'linear'),
            'encoder_type':         model_cfg_yaml.get('encoder_type', 'vit'),
            'image_size':           int(full_yaml_cfg.get('environment', {}).get('image_size', 64)),
            'patch_size':           int(model_cfg_yaml.get('patch_size', 8)),
            'frame_stack':          actual_fs,
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
                     theta_range: float = 0.6):
    """Return (zs, zs_vel, states, states_vel).

    zs:         (N, d)   single-frame encodings  — used for phase portrait
    zs_vel:     (M, 2d)  [z_t | z_t - z_{t-1}]  — used for Pearson (captures velocity)
    states:     (N, 4)   physical states for zs
    states_vel: (M, 4)   physical states at time t for zs_vel
    """
    zs, zs_vel, states, states_vel = [], [], [], []
    rng = np.random.RandomState(0)

    for _ in range(n_rollouts):
        theta0 = rng.uniform(-theta_range, theta_range)
        x0 = np.array([rng.uniform(-0.3, 0.3), 0.0, theta0, 0.0], dtype=np.float32)
        obs, state, _ = env.reset_to_state(x0)
        prev_obs = obs.copy()
        prev_z = None

        for _ in range(rollout_len):
            obs_t = _stack_obs(obs, prev_obs, frame_stack, device)
            with torch.no_grad():
                z = model.encoder(obs_t).cpu().numpy()[0]

            zs.append(z)
            states.append(state.copy())

            if prev_z is not None:
                zs_vel.append(np.concatenate([z, z - prev_z]))
                states_vel.append(state.copy())

            prev_z = z
            action = env.sample_action()
            obs_next, state_next, _, done, _ = env.step(action)
            prev_obs = obs.copy()
            obs, state = obs_next, state_next
            if done:
                prev_z = None  # no valid Δz across episode boundary
                break

    zs_arr = np.array(zs)
    d = zs_arr.shape[1]
    zs_vel_arr = np.array(zs_vel) if zs_vel else np.empty((0, 2 * d))
    return zs_arr, zs_vel_arr, np.array(states), np.array(states_vel) if states_vel else np.empty((0, 4))


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
    """Heatmap of Pearson r. Feature dims sorted by max |r| across state vars.

    R is expected to be (4, 2d) where the feature is [z_t | z_t - z_{t-1}]:
    the first d columns probe position encoding; the next d probe velocity encoding.
    """
    sort_idx = np.argsort(-np.max(np.abs(R), axis=0))
    R_sorted = R[:, sort_idx]
    im = ax.imshow(R_sorted, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_yticks(range(4))
    ax.set_yticklabels(['x', 'ẋ', 'θ', 'θ̇'], fontsize=9)
    ax.set_xlabel('Feature dim (sorted by |r|)  [z_t | Δz_t]', fontsize=8)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for j in range(4):
        best = int(np.argmax(R_sorted[j]))   # max positive r
        if R_sorted[j, best] > 0.05:         # only mark if meaningfully positive
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


# ── Near-equilibrium encodings for phase portrait PCA ────────────────────────

def collect_near_eq_encodings(env, model, frame_stack: int, device,
                               theta_range: float = 0.5, n_theta: int = 40) -> np.ndarray:
    """Encode states sweeping θ ∈ [-theta_range, theta_range] at rest (x=ẋ=θ̇=0).

    Returns (n_theta, d).  Used as the PCA basis for the phase portrait so that
    z* (encoded at θ=0) lands near the centre of the projection rather than at
    the edge of a wide random-rollout distribution.
    """
    thetas = np.linspace(-theta_range, theta_range, n_theta)
    zs = []
    for th in thetas:
        x0 = np.array([0.0, 0.0, float(th), 0.0], dtype=np.float32)
        obs, _, _ = env.reset_to_state(x0)
        obs_t = _to_tensor(obs, device)
        if frame_stack > 1:
            obs_t = torch.cat([obs_t, obs_t], dim=1)
        with torch.no_grad():
            z = model.encoder(obs_t).cpu().numpy()[0]
        zs.append(z)
    return np.array(zs)


# ── Panel 3: Phase portrait ───────────────────────────────────────────────────

def plot_phase_portrait(ax, model, frame_stack: int, device,
                        z_star: np.ndarray, zs: np.ndarray, title: str,
                        grid_size: int = 14, zs_near_eq: np.ndarray = None):
    """2D phase portrait of f(z, u=0) - z in the PC1-PC2 subspace around z*.

    PCA basis is computed from near-equilibrium encodings (zs_near_eq) so that
    z* sits near the centre of the projection.  Falls back to rollout data if
    zs_near_eq is not provided.
    """
    # PCA from near-eq data so z* is centred; fall back to rollout data
    pca_src = zs_near_eq if zs_near_eq is not None else zs
    z_centered = pca_src - z_star[np.newaxis, :]
    _, _, Vt = np.linalg.svd(z_centered, full_matrices=False)
    v1, v2 = Vt[0], Vt[1]   # (d,) each

    # Grid range from near-eq spread (tight, centred on z*)
    c1_all = z_centered @ v1
    c2_all = z_centered @ v2
    r1 = float(np.percentile(np.abs(c1_all), 95)) * 1.2
    r2 = float(np.percentile(np.abs(c2_all), 95)) * 1.2

    c1_vals = np.linspace(-r1, r1, grid_size)
    c2_vals = np.linspace(-r2, r2, grid_size)
    C1, C2 = np.meshgrid(c1_vals, c2_vals)
    dC1 = np.zeros_like(C1)
    dC2 = np.zeros_like(C2)

    W = getattr(model.config, 'predictor_window', 1)
    model.eval()

    for i in range(grid_size):
        for j in range(grid_size):
            z = z_star + float(C1[i, j]) * v1 + float(C2[i, j]) * v2
            z_t = torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0)
            z_win = z_t.unsqueeze(1).expand(1, W, -1)
            u_win = torch.zeros(1, W, 1, device=device)
            with torch.no_grad():
                z_next = model.predict(z_win, u_win).cpu().numpy()[0]
            dz = z_next - z
            dC1[i, j] = float(dz @ v1)
            dC2[i, j] = float(dz @ v2)

    speed = np.sqrt(dC1 ** 2 + dC2 ** 2) + 1e-9
    ax.quiver(C1, C2, dC1 / speed, dC2 / speed, speed,
              cmap='plasma', alpha=0.85, scale=grid_size * 1.2)
    ax.plot(0, 0, 'r*', markersize=12, zorder=5, label='z*')
    ax.set_xlabel(f'PC1', fontsize=8)
    ax.set_ylabel(f'PC2', fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc='upper right')


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
        zs, zs_vel, states, states_vel = collect_rollouts(
            env, model, frame_stack, device,
            n_rollouts=args.n_rollouts, rollout_len=args.rollout_len)
        print(f'    collected {len(zs)} steps, {len(zs_vel)} paired steps')

        # Pearson on [z_t | Δz_t] feature — captures both position and velocity
        R = compute_pearson_matrix(zs_vel, states_vel)
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

        print(f'    collecting near-eq encodings for phase portrait PCA ...')
        zs_near_eq = collect_near_eq_encodings(env, model, frame_stack, device)

        all_data.append({
            'label': label, 'zs': zs, 'zs_vel': zs_vel,
            'states': states, 'states_vel': states_vel,
            'R': R, 'gramian_eigvals': gramian_eigvals,
            'model': model, 'frame_stack': frame_stack, 'z_star': z_star,
            'zs_near_eq': zs_near_eq,
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

        # Col 2: Phase portrait — PCA centred on z* via near-eq encodings
        print(f'  [{label}] computing phase portrait ...')
        plot_phase_portrait(axes[row, 2], d['model'], d['frame_stack'], device,
                            d['z_star'], d['zs'],
                            title=f'{label}\nPhase portrait (u=0)',
                            zs_near_eq=d['zs_near_eq'])

    fig.suptitle('Latent Dynamics Structure Comparison', fontsize=13, y=1.01)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f'[plot] Saved → {out_path}')


if __name__ == '__main__':
    main()
