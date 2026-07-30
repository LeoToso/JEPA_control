"""Action sensitivity test: does the predictor's B match the real physical B?

Hypothesis: the predictor was trained on passive data (u≈0), so its action
Jacobian B_eff = B_aug @ W_enc captures gradient noise, not real action effects.

Test: from the same physical state (upright equilibrium), apply a range of
actions to both the predictor and the real environment, then compare:

  Δz_pred(u) = predictor(z_win, u_norm) - predictor(z_win, 0)   [model response]
  Δz_real(u) = encoder(env.step(u))     - encoder(env.step(0))   [physical response]

Key metrics:
  cos(Δz_pred, Δz_real)        — direction agreement  (0=orthogonal, 1=aligned)
  ||Δz_pred|| / ||Δz_real||    — magnitude ratio       (1=correct scale)
  R²(Δz_pred vs Δz_real)       — overall model fidelity across all u values

  Also compares Jacobian B_eff vs empirical B (finite difference from real env).

Expected results under hypothesis:
  cosine ≈ 0       — predictor response to u is orthogonal to real response
  R²     ≈ 0       — model cannot predict how actions change z

Usage:
    python experiments/check_action_sensitivity.py \\
        --checkpoint results/jepa_sf_w3_fs5_v9_phase2/checkpoints/checkpoint_epoch0090.pt \\
        --config     configs/cartpole_jepa_sf_w3_fs5_v9_phase2.yaml \\
        --device cuda:2
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml


def _get_z(model, obs_np, device):
    t = torch.from_numpy(obs_np).float().permute(2, 0, 1)[None].to(device) / 255.0
    with torch.no_grad():
        return model.encode_obs(t, t).cpu().numpy()[0]


def _predict_step(model, z_win_np, u_raw, action_scale, device):
    """Run one predictor step from numpy arrays. Returns z_next as numpy."""
    W, d = z_win_np.shape
    z_t = torch.tensor(z_win_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1,W,d)
    u_norm = u_raw / action_scale
    # u_win shape must be (1, W, 1): zeros for history, u_norm at last (current) slot
    u_t = torch.zeros(1, W, 1, dtype=torch.float32, device=device)
    u_t[0, -1, 0] = u_norm
    with torch.no_grad():
        z_next = model.predict(z_t, u_t).cpu().numpy()[0]
    return z_next


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config',     required=True)
    p.add_argument('--n-states',   type=int, default=10,
                   help='Number of different initial states to average over')
    p.add_argument('--init-scale', type=float, default=0.02,
                   help='Initial state perturbation range (near upright)')
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--device',     default=None)
    p.add_argument('--out',        default=None,
                   help='Save figure to this path (PNG). If omitted no figure is saved.')
    args = p.parse_args()

    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Device: {device}')

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_cfg   = cfg['environment']
    model_cfg = cfg['model']

    frame_skip  = int(env_cfg.get('frame_skip', 1))
    action_lb   = float(env_cfg['action_range'][0])
    action_ub   = float(env_cfg['action_range'][1])
    action_scale = max(abs(action_lb), abs(action_ub))

    # ── Load model ─────────────────────────────────────────────────────────────
    raw = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = (raw['model_state'] if isinstance(raw, dict) and 'model_state' in raw
             else raw.get('model_state_dict', raw))
    if isinstance(raw, dict) and 'config' in raw:
        for k in ('latent_dim', 'action_latent_dim', 'action_encoder', 'encoder_type',
                  'patch_size', 'frame_stack', 'vit_embed_dim', 'vit_depth', 'vit_num_heads',
                  'predictor_type', 'predictor_hidden_dim', 'predictor_n_layers',
                  'predictor_window', 'predictor_embed_dim', 'predictor_depth',
                  'predictor_num_heads', 'predictor_mlp_ratio'):
            if k in raw['config'] and raw['config'][k] != model_cfg.get(k):
                model_cfg[k] = raw['config'][k]

    from models.jepa import make_jepa
    model = make_jepa(
        variant='E-full',
        latent_dim=int(model_cfg['latent_dim']),
        action_latent_dim=int(model_cfg.get('action_latent_dim', 1)),
        action_encoder=model_cfg.get('action_encoder', 'linear'),
        encoder_type=model_cfg.get('encoder_type', 'vit'),
        image_size=int(env_cfg['image_size']),
        patch_size=int(model_cfg.get('patch_size', 8)),
        frame_stack=int(model_cfg.get('frame_stack', 1)),
        use_frame_diff=bool(model_cfg.get('use_frame_diff', False)),
        vit_embed_dim=int(model_cfg.get('vit_embed_dim', 128)),
        vit_depth=int(model_cfg.get('vit_depth', 4)),
        vit_num_heads=int(model_cfg.get('vit_num_heads', 4)),
        predictor_type=model_cfg.get('predictor_type', 'mlp'),
        predictor_hidden_dim=int(model_cfg.get('predictor_hidden_dim', 64)),
        predictor_n_layers=int(model_cfg.get('predictor_n_layers', 3)),
        predictor_window=int(model_cfg.get('predictor_window', 5)),
        predictor_embed_dim=int(model_cfg.get('predictor_embed_dim', 128)),
        predictor_depth=int(model_cfg.get('predictor_depth', 4)),
        predictor_num_heads=int(model_cfg.get('predictor_num_heads', 4)),
        predictor_mlp_ratio=float(model_cfg.get('predictor_mlp_ratio', 4.0)),
    )
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    d = int(model_cfg['latent_dim'])
    W = int(model_cfg.get('predictor_window', 1))

    # ── Environment ─────────────────────────────────────────────────────────────
    from envs.cartpole_visual import ContinuousCartpoleVisual
    env = ContinuousCartpoleVisual(
        frame_skip=frame_skip, image_size=int(env_cfg['image_size']),
        mass_cart=env_cfg['mass_cart'], mass_pole=env_cfg['mass_pole'],
        pole_length=env_cfg['pole_length'], gravity=env_cfg['gravity'],
        dt=env_cfg['dt'], seed=args.seed,
    )

    rng = np.random.RandomState(args.seed)
    u_test_values = np.array([-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0])

    # ── Per-state measurements ──────────────────────────────────────────────────
    all_cos_pred_real = []
    all_mag_ratio     = []
    all_r2            = []
    all_B_empirical   = []

    print(f'\n[test] n_states={args.n_states}  W={W}  d={d}')
    print(f'       action_scale={action_scale}  u_test={u_test_values.tolist()}')
    print()

    for s_idx in range(args.n_states):
        # Sample a near-upright initial state
        x0 = rng.uniform(-args.init_scale, args.init_scale, 4).astype(np.float32)

        # Build window history: run W passive steps from x0
        obs, state_phys, _ = env.reset_to_state(x0)
        z_hist = [_get_z(model, obs, device)]
        for _ in range(W - 1):
            obs, state_phys, _, _, _ = env.step(0.0)
            z_hist.append(_get_z(model, obs, device))
        # z_hist[0] = oldest, z_hist[-1] = newest (current)
        z_win_np = np.stack(z_hist)   # (W, d)
        z_current = z_hist[-1]        # the current latent we'll compare from

        # ── Predictor response: z_pred(u) for each test action ─────────────────
        z_pred_list = []
        for u in u_test_values:
            z_next_pred = _predict_step(model, z_win_np, u, action_scale, device)
            z_pred_list.append(z_next_pred)
        z_pred_arr = np.stack(z_pred_list)   # (n_u, d)

        # Δz_pred(u) = z_pred(u) - z_pred(0)
        u0_idx = np.where(u_test_values == 0.0)[0][0]
        dz_pred = z_pred_arr - z_pred_arr[u0_idx:u0_idx+1]   # (n_u, d)

        # ── Real environment response: z_real(u) ──────────────────────────────
        # Must reset to the EXACT same physical state for each action
        z_real_list = []
        for u in u_test_values:
            obs_u, _, _ = env.reset_to_state(state_phys)
            obs_next, _, _, _, _ = env.step(float(u))
            z_real_list.append(_get_z(model, obs_next, device))
        z_real_arr = np.stack(z_real_list)   # (n_u, d)

        # Δz_real(u) = z_real(u) - z_real(0)
        dz_real = z_real_arr - z_real_arr[u0_idx:u0_idx+1]   # (n_u, d)

        # ── Metrics ────────────────────────────────────────────────────────────

        # 1. Cosine similarity between pred and real response vectors
        # Flatten the (n_u, d) response matrices and compute cosine
        dz_pred_flat = dz_pred.flatten()
        dz_real_flat = dz_real.flatten()
        cos_pr = float(np.dot(dz_pred_flat, dz_real_flat) /
                       (np.linalg.norm(dz_pred_flat) * np.linalg.norm(dz_real_flat) + 1e-12))
        all_cos_pred_real.append(cos_pr)

        # 2. Magnitude ratio per action (excluding u=0)
        non_zero = u_test_values != 0.0
        mag_pred = np.linalg.norm(dz_pred[non_zero], axis=1)
        mag_real = np.linalg.norm(dz_real[non_zero], axis=1)
        ratio = float(np.mean(mag_pred / (mag_real + 1e-8)))
        all_mag_ratio.append(ratio)

        # 3. R²: how well does dz_pred explain dz_real?
        ss_res = np.sum((dz_pred_flat - dz_real_flat) ** 2)
        ss_tot = np.sum((dz_real_flat - dz_real_flat.mean()) ** 2)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
        all_r2.append(r2)

        # 4. Empirical B: finite difference from real env (antisymmetric)
        # Use u=+2N and u=-2N for a clean central difference
        idx_p2 = np.where(u_test_values ==  2.0)[0][0]
        idx_m2 = np.where(u_test_values == -2.0)[0][0]
        u_fd   = 2.0
        B_emp  = (z_real_arr[idx_p2] - z_real_arr[idx_m2]) / (2 * u_fd / action_scale)
        all_B_empirical.append(B_emp)

    # ── Jacobian B_eff from model ─────────────────────────────────────────────
    from control.jacobian import compute_augmented_jacobian_np
    obs_eq, _, _ = env.reset_to_state(np.zeros(4, dtype=np.float32))
    z_star = _get_z(model, obs_eq, device)
    A_aug, B_aug = compute_augmented_jacobian_np(model, z_star, device)
    if hasattr(model.action_encoder, 'W'):
        W_enc = model.action_encoder.W.weight.detach().cpu().numpy()  # (m, 1)
        B_eff = (B_aug @ W_enc)[:d].flatten()   # only first d dims (current step)
    else:
        B_eff = B_aug[:d, 0].flatten()

    # Average empirical B across states
    B_emp_mean = np.mean(all_B_empirical, axis=0)   # (d,)

    # ── Report ────────────────────────────────────────────────────────────────
    print('=' * 60)
    print('ACTION SENSITIVITY TEST — KEY RESULTS')
    print('=' * 60)

    cos_mean = float(np.mean(all_cos_pred_real))
    cos_std  = float(np.std(all_cos_pred_real))
    mag_mean = float(np.mean(all_mag_ratio))
    r2_mean  = float(np.mean(all_r2))

    print(f'\n[1] Direction agreement: cos(Δz_pred, Δz_real)')
    print(f'    mean = {cos_mean:.4f}  std = {cos_std:.4f}')
    print(f'    (1.0 = perfect alignment, 0.0 = orthogonal, -1.0 = opposite)')

    print(f'\n[2] Magnitude ratio: ||Δz_pred|| / ||Δz_real||')
    print(f'    mean = {mag_mean:.4f}')
    print(f'    (1.0 = correct scale, >>1 = model oversensitive, <<1 = undersensitive)')

    print(f'\n[3] R² of predictor on real z-changes')
    print(f'    mean = {r2_mean:.4f}')
    print(f'    (1.0 = perfect prediction, 0.0 = no better than mean, <0 = worse)')

    print(f'\n[4] Jacobian B_eff vs empirical B (physical finite difference)')
    cos_B = float(np.dot(B_eff, B_emp_mean) /
                  (np.linalg.norm(B_eff) * np.linalg.norm(B_emp_mean) + 1e-12))
    print(f'    ||B_eff_model||    = {np.linalg.norm(B_eff):.4f}')
    print(f'    ||B_eff_physical|| = {np.linalg.norm(B_emp_mean):.4f}')
    print(f'    cos(B_model, B_physical) = {cos_B:.4f}')
    print(f'    (1.0 = model B points in physical direction)')

    print()
    print('─' * 60)
    print('VERDICT')
    print('─' * 60)
    if cos_mean < 0.1 and r2_mean < 0.1:
        print('  ✗ HYPOTHESIS CONFIRMED: predictor response to u is orthogonal')
        print('    to the real physical response. B_eff is gradient noise from')
        print('    passive training — not real action sensitivity.')
        print('    → Active data retraining is required.')
    elif cos_mean > 0.5 and r2_mean > 0.3:
        print('  ✓ Predictor DOES track real action effects (cos > 0.5, R² > 0.3).')
        print('    The issue is NOT passive training. Look at Q/cost calibration,')
        print('    fp_err, or action clipping as the failure cause.')
    else:
        print(f'  ~ Partial alignment (cos={cos_mean:.3f}, R²={r2_mean:.3f}).')
        print('    Predictor has some sensitivity but in the wrong direction/scale.')
        print('    Active data retraining should improve B alignment.')

    print()
    print('[per-action breakdown at one state (state 0)]')
    print(f'  {"u (N)":>8}  {"||Δz_pred||":>12}  {"||Δz_real||":>12}  ratio')
    dz_pred_per_u, dz_real_per_u = [], []
    obs_eq2, _, _ = env.reset_to_state(np.zeros(4, dtype=np.float32))
    z_h = [_get_z(model, obs_eq2, device)] * W
    z_w  = np.stack(z_h)
    obs_eq3, st3, _ = env.reset_to_state(np.zeros(4, dtype=np.float32))
    z_real0_ref = _get_z(model, obs_eq3, device)
    obs_ref, _, _, _, _ = env.step(0.0)
    z_real_ref = _get_z(model, obs_ref, device)
    for u in u_test_values:
        zp = _predict_step(model, z_w, u, action_scale, device)
        obs_u, _, _ = env.reset_to_state(st3)
        obs_n, _, _, _, _ = env.step(float(u))
        zr = _get_z(model, obs_n, device)
        dz_p = np.linalg.norm(zp - _predict_step(model, z_w, 0.0, action_scale, device))
        dz_r = np.linalg.norm(zr - z_real_ref)
        ratio_str = f'{dz_p/(dz_r+1e-8):.2f}' if abs(u) > 0 else '—'
        print(f'  {u:>8.1f}  {dz_p:>12.4f}  {dz_r:>12.4f}  {ratio_str}')
        dz_pred_per_u.append(dz_p)
        dz_real_per_u.append(dz_r)
    env.close()

    # ── Figure ────────────────────────────────────────────────────────────────
    if args.out:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ckpt_name = Path(args.checkpoint).name
        fig.suptitle(f'Action Sensitivity — {ckpt_name}\n'
                     f'cos={cos_mean:.3f}  mag_ratio={mag_mean:.3f}  '
                     f'R²={r2_mean:.3f}  cos(B)={cos_B:.3f}', fontsize=10)

        # Panel 1: per-state metrics
        ax = axes[0]
        xs = np.arange(args.n_states)
        ax.bar(xs - 0.25, all_cos_pred_real, 0.25, label='cos(Δpred,Δreal)', color='steelblue')
        ax.bar(xs,        all_r2,            0.25, label='R²',               color='darkorange')
        ax.bar(xs + 0.25, all_mag_ratio,     0.25, label='mag ratio',        color='seagreen', alpha=0.7)
        ax.axhline(0, color='k', lw=0.6); ax.axhline(1, color='k', lw=0.6, ls='--')
        ax.set_title('Per-state metrics', fontsize=10)
        ax.set_xlabel('state index'); ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # Panel 2: per-action response magnitudes at eq
        ax2 = axes[1]
        ax2.plot(u_test_values, dz_pred_per_u, 'o-', label='||Δz_pred||', color='steelblue')
        ax2.plot(u_test_values, dz_real_per_u, 's--', label='||Δz_real||', color='crimson')
        ax2.set_title('Response magnitude vs action (at eq)', fontsize=10)
        ax2.set_xlabel('u (N)'); ax2.set_ylabel('||Δz||')
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

        # Panel 3: B_model vs B_physical per latent dim
        ax3 = axes[2]
        dims = np.arange(len(B_eff))
        ax3.bar(dims - 0.2, B_eff,      0.4, label=f'B_model (||·||={np.linalg.norm(B_eff):.3f})',
                color='steelblue')
        ax3.bar(dims + 0.2, B_emp_mean, 0.4, label=f'B_phys  (||·||={np.linalg.norm(B_emp_mean):.3f})',
                color='darkorange', alpha=0.8)
        ax3.axhline(0, color='k', lw=0.5)
        ax3.set_title(f'B_model vs B_physical  cos={cos_B:.3f}', fontsize=10)
        ax3.set_xlabel('latent dim'); ax3.legend(fontsize=7); ax3.grid(alpha=0.3)

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
        print(f'Saved → {out_path}')


if __name__ == '__main__':
    main()
