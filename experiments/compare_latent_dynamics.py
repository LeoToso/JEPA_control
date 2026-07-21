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
              'vit_num_heads', 'predictor_type', 'predictor_window',
              'predictor_hidden_dim', 'predictor_n_layers',
              'predictor_embed_dim', 'predictor_depth', 'predictor_num_heads',
              'predictor_mlp_ratio', 'variant'):
        if k in d:
            setattr(cfg, k, d[k])
    return cfg


def load_model(ckpt_path: str, cfg_yaml: str, device):
    """Load a JEPA or AE model. Returns (model, frame_stack, label)."""
    from models.jepa import JEPAModel
    from models.autoencoder import AEWorldModel

    ckpt = torch.load(ckpt_path, map_location=device)
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
            'latent_dim':            int(model_cfg_yaml.get('latent_dim', 32)),
            'action_latent_dim':     int(model_cfg_yaml.get('action_latent_dim', 4)),
            'action_encoder':        model_cfg_yaml.get('action_encoder', 'linear'),
            'encoder_type':          model_cfg_yaml.get('encoder_type', 'vit'),
            'image_size':            int(full_yaml_cfg.get('environment', {}).get('image_size', 64)),
            'patch_size':            int(model_cfg_yaml.get('patch_size', 8)),
            'frame_stack':           actual_fs,
            'vit_embed_dim':         int(model_cfg_yaml.get('vit_embed_dim', 128)),
            'vit_depth':             int(model_cfg_yaml.get('vit_depth', 4)),
            'vit_num_heads':         int(model_cfg_yaml.get('vit_num_heads', 4)),
            'predictor_type':        model_cfg_yaml.get('predictor_type', 'mlp'),
            'predictor_window':      int(model_cfg_yaml.get('predictor_window', 3)),
            'predictor_hidden_dim':  int(model_cfg_yaml.get('predictor_hidden_dim', 256)),
            'predictor_n_layers':    int(model_cfg_yaml.get('predictor_n_layers', 2)),
            'predictor_embed_dim':   int(model_cfg_yaml.get('predictor_embed_dim', 128)),
            'predictor_depth':       int(model_cfg_yaml.get('predictor_depth', 4)),
            'predictor_num_heads':   int(model_cfg_yaml.get('predictor_num_heads', 4)),
            'predictor_mlp_ratio':   float(model_cfg_yaml.get('predictor_mlp_ratio', 4.0)),
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
    """Heatmap of Pearson r. Feature dims sorted by max |r| across state vars."""
    sort_idx = np.argsort(-np.max(np.abs(R), axis=0))
    R_sorted = R[:, sort_idx]
    im = ax.imshow(R_sorted, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_yticks(range(4))
    ax.set_yticklabels([r'$x$', r'$\dot{x}$', r'$\theta$', r'$\dot{\theta}$'], fontsize=9)
    ax.set_xlabel('Latent dim (sorted by max |r|)', fontsize=8)
    ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for j in range(4):
        best = int(np.argmax(np.abs(R_sorted[j])))
        if np.abs(R_sorted[j, best]) > 0.05:
            ax.add_patch(plt.Rectangle((best - 0.5, j - 0.5), 1, 1,
                                       fill=False, edgecolor='lime', lw=1.5))


# ── Panel 2: Controllability Gramian ─────────────────────────────────────────

def compute_gramian_eigenvalues(model, z_star: np.ndarray, device,
                                T: int = 20) -> np.ndarray:
    """Controllability Gramian in the augmented (W*d)-dimensional state space.

    Uses the full augmented Jacobian A_aug so that rho(A_aug) and the
    Gramian eigenspectrum are correct for windowed predictors (W > 1).
    For W=1 this is identical to the previous behaviour.
    """
    from control.jacobian import compute_augmented_jacobian_np
    A_aug, B_aug = compute_augmented_jacobian_np(model, z_star, device)

    # B_eff_aug: project encoded-action columns through W_enc → scalar action
    if hasattr(model.action_encoder, 'W'):
        W_enc = model.action_encoder.W.weight.detach().cpu().numpy()  # (m, 1)
        B_eff = B_aug @ W_enc   # (W*d, 1)
    else:
        B_eff = B_aug            # (W*d, 1) already

    # Gramian accumulation in augmented space
    Wd = A_aug.shape[0]
    W_gram = np.zeros((Wd, Wd))
    AB = B_eff.copy()
    for _ in range(T):
        W_gram += AB @ AB.T
        AB = A_aug @ AB

    eigvals = np.linalg.eigvalsh(W_gram)
    return np.sort(np.abs(eigvals))[::-1], A_aug, B_eff


def plot_gramian(ax, eigvals: np.ndarray, title: str,
                 rho: float = None, rho_gt: float = None,
                 b_eff_norm: float = None):
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
    # Spectral radius annotation (learned + GT reference)
    lines = []
    if rho is not None:
        color = 'green' if rho > 1.0 else 'darkorange'
        lines.append((f'ρ(A)={rho:.4f}', color))
    if rho_gt is not None:
        lines.append((f'ρ_GT={rho_gt:.4f}', 'steelblue'))
    if b_eff_norm is not None:
        lines.append((f'‖B‖={b_eff_norm:.4f}', 'gray'))
    if lines:
        y0 = 0.04
        for text, col in reversed(lines):
            ax.text(0.02, y0, text, transform=ax.transAxes,
                    ha='left', va='bottom', fontsize=8, color=col,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
            y0 += 0.11


# ── (collect_near_eq_encodings removed — phase portrait now uses physical coords)


# ── Panel 3: Phase portrait in physical (θ, θ̇) space ─────────────────────────

def plot_phase_portrait(ax, model, frame_stack: int, device, env,
                        zs: np.ndarray, states: np.ndarray, title: str,
                        theta_range: float = 0.5, thetadot_range: float = 2.0,
                        grid_size: int = 14):
    """Phase portrait of learned dynamics in (θ, θ̇) space with GT overlay.

    For each grid point: render obs → encode z → predict z_next → decode
    Δz to (Δθ, Δθ̇) via least-squares fit on rollout data.
    Colored arrows = learned, thin black arrows = ground truth (env.step).
    """
    W_pred = getattr(model.config, 'predictor_window', 1)
    model.eval()

    w_decode, _, _, _ = np.linalg.lstsq(zs, states[:, 2:4], rcond=None)

    theta_vals = np.linspace(-theta_range, theta_range, grid_size)
    thetadot_vals = np.linspace(-thetadot_range, thetadot_range, grid_size)
    TH, THD = np.meshgrid(theta_vals, thetadot_vals)
    DTH_l  = np.zeros_like(TH)
    DTHD_l = np.zeros_like(THD)
    DTH_gt  = np.zeros_like(TH)
    DTHD_gt = np.zeros_like(THD)

    for i in range(grid_size):
        for j in range(grid_size):
            th  = float(TH[i, j])
            thd = float(THD[i, j])
            x0 = np.array([0.0, 0.0, th, thd], dtype=np.float32)

            obs, _, _ = env.reset_to_state(x0)
            obs_t = _to_tensor(obs, device)
            if frame_stack > 1:
                obs_t = torch.cat([obs_t, obs_t], dim=1)

            with torch.no_grad():
                z = model.encoder(obs_t)
                z_win = z.unsqueeze(1).expand(1, W_pred, -1)
                u_win = torch.zeros(1, W_pred, 1, device=device)
                z_next = model.predict(z_win, u_win)
                dz = (z_next - z).cpu().numpy()[0]

            d_phys = dz @ w_decode
            DTH_l[i, j]  = d_phys[0]
            DTHD_l[i, j] = d_phys[1]

            env.reset_to_state(x0)
            _, state_next, _, _, _ = env.step(0.0)
            DTH_gt[i, j]  = state_next[2] - th
            DTHD_gt[i, j] = state_next[3] - thd

    TH_d  = np.degrees(TH)
    THD_d = np.degrees(THD)

    speed_gt = np.sqrt(np.degrees(DTH_gt)**2 + np.degrees(DTHD_gt)**2) + 1e-9
    ax.quiver(TH_d, THD_d,
              np.degrees(DTH_gt) / speed_gt, np.degrees(DTHD_gt) / speed_gt,
              color='black', alpha=0.25, scale=grid_size * 1.4,
              width=0.003, zorder=1, label='GT')

    speed_l = np.sqrt(np.degrees(DTH_l)**2 + np.degrees(DTHD_l)**2) + 1e-9
    ax.quiver(TH_d, THD_d,
              np.degrees(DTH_l) / speed_l, np.degrees(DTHD_l) / speed_l,
              speed_l, cmap='plasma', alpha=0.85, scale=grid_size * 1.2,
              zorder=2, label='Learned')

    ax.plot(0, 0, 'r*', markersize=12, zorder=5, label='eq')
    ax.set_xlabel(r'$\theta$ (deg)', fontsize=8)
    ax.set_ylabel(r'$\dot{\theta}$ (deg/s)', fontsize=8)
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

    # ── Ground-truth spectral radius ──────────────────────────────────────
    try:
        from ground_truth.cartpole_gt import CartpoleGroundTruth
        _gt = CartpoleGroundTruth()
        rho_gt = float(np.max(np.abs(np.linalg.eigvals(_gt.A_star))))
        print(f'[GT] rho(A_star)={rho_gt:.4f}  (target: >1.0 for instability)')
    except Exception as _e:
        print(f'[GT] could not compute ground-truth rho: {_e}')
        rho_gt = None

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
        print(f'    collected {len(zs)} steps')

        R = compute_pearson_matrix(zs, states)
        max_r_per_state = np.max(np.abs(R), axis=1)
        print(f'    max |r|: x={max_r_per_state[0]:.3f}  '
              f'xdot={max_r_per_state[1]:.3f}  '
              f'theta={max_r_per_state[2]:.3f}  '
              f'thetadot={max_r_per_state[3]:.3f}')

        obs_eq_t = _to_tensor(obs_eq, device)
        if frame_stack > 1:
            obs_eq_t = torch.cat([obs_eq_t, obs_eq_t], dim=1)
        with torch.no_grad():
            z_star = model.encoder(obs_eq_t).cpu().numpy()[0]

        print(f'    computing Gramian (T={args.gramian_T}) ...')
        rho = None
        b_eff_norm = None
        try:
            gramian_eigvals, A_aug, B_eff = compute_gramian_eigenvalues(
                model, z_star, device, T=args.gramian_T)
            rho = float(np.max(np.abs(np.linalg.eigvals(A_aug))))
            b_eff_norm = float(np.linalg.norm(B_eff[:len(z_star)]))
            print(f'    rho(A_aug)={rho:.4f}  ||B_eff||={b_eff_norm:.4f}'
                  + (f'  [GT rho={rho_gt:.4f}]' if rho_gt is not None else ''))
        except Exception as exc:
            print(f'    Gramian failed: {exc}')
            gramian_eigvals = None

        all_data.append({
            'label': label, 'zs': zs, 'states': states,
            'R': R, 'gramian_eigvals': gramian_eigvals,
            'rho': rho, 'b_eff_norm': b_eff_norm,
            'model': model, 'frame_stack': frame_stack, 'z_star': z_star,
        })

    # ── Plot ──────────────────────────────────────────────────────────────
    print('\n[plot] Generating figure ...')
    fig, axes = plt.subplots(n_models, 3, figsize=(14, 4 * n_models))
    if n_models == 1:
        axes = axes[np.newaxis, :]

    for row, d in enumerate(all_data):
        label = d['label']

        plot_pearson(axes[row, 0], d['R'],
                     title='Pearson Correlation')

        if d['gramian_eigvals'] is not None:
            plot_gramian(axes[row, 1], d['gramian_eigvals'],
                         title='Controllability Gramian',
                         rho=d.get('rho'), rho_gt=rho_gt,
                         b_eff_norm=d.get('b_eff_norm'))
        else:
            axes[row, 1].text(0.5, 0.5, 'Gramian N/A', ha='center', va='center',
                              transform=axes[row, 1].transAxes)
            axes[row, 1].set_title('Controllability Gramian', fontsize=9)

        print(f'  [{label}] computing phase portrait ...')
        plot_phase_portrait(axes[row, 2], d['model'], d['frame_stack'], device,
                            env, d['zs'], d['states'],
                            title='Phase Portrait')

    env.close()

    if n_models > 1:
        fig.suptitle('Latent Dynamics Comparison', fontsize=13, y=1.01)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f'[plot] Saved → {out_path}')


if __name__ == '__main__':
    main()
