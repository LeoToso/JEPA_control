"""PBH controllability test for the augmented latent dynamics.

For each unstable eigenvalue λ of A_aug, the system (A_aug, B_aug) is
controllable at λ iff the left eigenvector v satisfies |v^H B_aug| > 0.

Key metric:  PBH_score = |v^H B_aug| / (||v|| * ||B_aug||)
    = 0   → completely uncontrollable at λ (B orthogonal to unstable mode)
    = 1   → maximally controllable
    < 0.1 → practically uncontrollable

Also computes sigma_min([λI - A | B]) for each unstable λ — the standard
controllability Gramian singularity check.

Usage:
    python experiments/check_pbh_controllability.py \\
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
import scipy.linalg


def _get_z_star(model, device):
    """Encode a canonical upright cartpole image to get z*."""
    from envs.cartpole_visual import ContinuousCartpoleVisual
    env = ContinuousCartpoleVisual(seed=0)
    obs, _, _ = env.reset_to_state(np.zeros(4, dtype=np.float32))
    env.close()
    obs_t = torch.from_numpy(obs).float().permute(2, 0, 1)[None].to(device) / 255.0
    with torch.no_grad():
        z = model.encoder(obs_t).cpu().numpy()[0]
    return z


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config',     required=True)
    p.add_argument('--device',     default=None)
    args = p.parse_args()

    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Device: {device}')

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_cfg   = cfg['environment']
    model_cfg = cfg['model']

    ckpt_path = Path(args.checkpoint)
    print(f'[model] Loading {ckpt_path}')
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    Wd = W * d

    # Equilibrium latent
    z_star = _get_z_star(model, device)
    print(f'\n[latent] d={d}  W={W}  aug_dim={Wd}')
    print(f'[latent] ||z*||={np.linalg.norm(z_star):.4f}')

    # Augmented Jacobian
    from control.jacobian import compute_augmented_jacobian_np
    print('[jacobian] computing A_aug, B_aug ...')
    A_aug, B_aug = compute_augmented_jacobian_np(model, z_star, device)
    print(f'[jacobian] A_aug shape={A_aug.shape}  B_aug (pre-proj) shape={B_aug.shape}')

    eigs_aug = scipy.linalg.eigvals(A_aug)
    rho_aug  = float(np.max(np.abs(eigs_aug)))
    print(f'[jacobian] ρ(A_aug)={rho_aug:.4f}  ||B_aug (pre-proj)||={np.linalg.norm(B_aug):.4f}')

    # Project B through action encoder (latent action → scalar u), matching check_lqr_aug.py.
    # B_jac/B_aug are ∂f/∂c (c = encoded action, shape m).  The real control input is
    # scalar u_raw, and c = W_enc @ u_raw  →  B_eff = B_aug @ W_enc  (Wd, 1).
    if hasattr(model.action_encoder, 'W'):
        W_enc = model.action_encoder.W.weight.detach().cpu().numpy()  # (m, 1)
        B_aug = B_aug @ W_enc                                          # (Wd, 1)
        print(f'[jacobian] projected B through action encoder W_enc ({W_enc.shape})')
        print(f'[jacobian] B_eff shape={B_aug.shape}  ||B_eff||={np.linalg.norm(B_aug):.4f}')
    else:
        print('[jacobian] WARNING: action_encoder has no .W — using B_aug as-is '
              f'(shape {B_aug.shape}, may be multi-column)')

    # Fixed-point error
    c_aug = np.zeros(Wd)
    z_eq_t = torch.tensor(z_star, dtype=torch.float32, device=device).unsqueeze(0)
    z_win  = z_eq_t.unsqueeze(1).expand(1, W, d)
    u_win  = torch.zeros(1, W, 1, device=device)
    with torch.no_grad():
        z_next = model.predict(z_win, u_win).cpu().numpy()[0]
    c_aug[:d] = z_next - z_star
    print(f'[fp_err]  ||c||={np.linalg.norm(c_aug[:d]):.4f}  '
          f'(predictor drift at u=0, z=z*)')

    # ── PBH controllability test ──────────────────────────────────────────────
    print('\n' + '─'*60)
    print('PBH CONTROLLABILITY TEST')
    print('─'*60)

    unstable_mask = np.abs(eigs_aug) > 1.0
    unstable_eigs = eigs_aug[unstable_mask]
    print(f'Unstable eigenvalues of A_aug ({len(unstable_eigs)} total):')
    for lam in sorted(unstable_eigs, key=lambda x: -abs(x)):
        print(f'  λ = {lam:.4f}  |λ|={abs(lam):.4f}')

    print()
    # Left eigenvectors: v^H A = λ v^H  ↔  A^H v = λ* v
    # scipy.linalg.eig returns RIGHT eigenvectors; left = right eigenvectors of A^T
    _, vl_mat = scipy.linalg.eig(A_aug, left=True, right=False)

    for lam in sorted(unstable_eigs, key=lambda x: -abs(x)):
        # Find matching left eigenvector column
        residuals = [np.linalg.norm(A_aug.T @ vl_mat[:, i] - np.conj(lam) * vl_mat[:, i])
                     for i in range(Wd)]
        idx = int(np.argmin(residuals))
        v = vl_mat[:, idx]

        # PBH score: ||v^H B||_2 / (||v|| ||B||_F)
        # v^H B_aug is a (m,) row vector; use its 2-norm
        vHB      = v.conj() @ B_aug               # (m,) complex row vector
        vHB_norm = float(np.linalg.norm(vHB))
        score    = vHB_norm / (np.linalg.norm(v) * np.linalg.norm(B_aug, 'fro') + 1e-12)

        # σ_min([λI - A | B])
        M      = np.hstack([lam * np.eye(Wd) - A_aug, B_aug])
        svs    = np.linalg.svd(M, compute_uv=False)
        sig_min = float(svs[-1].real)

        print(f'λ = {lam:.4f}  |λ|={abs(lam):.4f}')
        print(f'  PBH score (cosine)    = {score:.6f}  '
              f'(0=uncontrollable, 1=max)')
        print(f'  σ_min([λI-A | B])     = {sig_min:.6f}  '
              f'(0=rank-deficient = uncontrollable)')
        print(f'  ||v^H B||_2           = {vHB_norm:.6f}')
        print()

    # ── Controllability Gramian (finite-time) ─────────────────────────────────
    print('─'*60)
    print('CONTROLLABILITY GRAMIAN  (W_c = sum_{k=0}^{N-1} A^k B B^T (A^T)^k)')
    print('─'*60)
    N_gram = 50
    Wc = np.zeros((Wd, Wd))
    Ak = np.eye(Wd)
    for _ in range(N_gram):
        Wc += Ak @ B_aug @ B_aug.T @ Ak.T
        Ak  = Ak @ A_aug
    eigs_wc = np.linalg.eigvalsh(Wc)
    print(f'Gramian eigenvalues (top 5): '
          f'{np.sort(np.abs(eigs_wc))[::-1][:5].round(4).tolist()}')
    print(f'min eigenvalue = {float(np.min(eigs_wc)):.4e}  '
          f'(> 0 → reachable from origin)')
    print(f'cond(Wc)       = {float(np.max(np.abs(eigs_wc)) / (np.min(np.abs(eigs_wc))+1e-30)):.2e}')

    # ── GT comparison ─────────────────────────────────────────────────────────
    print()
    print('─'*60)
    print('GT REFERENCE (physical cartpole)')
    print('─'*60)
    from ground_truth.cartpole_gt import CartpoleGroundTruth
    frame_skip = int(env_cfg.get('frame_skip', 1))
    gt = CartpoleGroundTruth(
        mass_cart=env_cfg['mass_cart'], mass_pole=env_cfg['mass_pole'],
        pole_length=env_cfg['pole_length'], gravity=env_cfg['gravity'],
        dt=env_cfg['dt'] * frame_skip,
    )
    eigs_gt = scipy.linalg.eigvals(gt.A_star)
    rho_gt  = float(np.max(np.abs(eigs_gt)))
    print(f'ρ(A_gt) = {rho_gt:.4f}  (physical system)')

    _, vl_gt = scipy.linalg.eig(gt.A_star, left=True, right=False)
    for lam_gt in sorted(eigs_gt, key=lambda x: -abs(x)):
        if abs(lam_gt) <= 1.0:
            continue
        residuals = [np.linalg.norm(gt.A_star.T @ vl_gt[:, i] - np.conj(lam_gt) * vl_gt[:, i])
                     for i in range(4)]
        idx = int(np.argmin(residuals))
        v_gt  = vl_gt[:, idx]
        vHB_gt   = v_gt.conj() @ gt.B_star          # (m_gt,) vector
        score_gt = float(np.linalg.norm(vHB_gt) /
                         (np.linalg.norm(v_gt) * np.linalg.norm(gt.B_star, 'fro') + 1e-12))
        print(f'GT λ={lam_gt:.4f}  PBH score={score_gt:.6f}  '
              f'(GT is fully controllable)')

    print()
    print('─'*60)
    print('SUMMARY')
    print('─'*60)
    any_uncontrollable = False
    for lam in sorted(unstable_eigs, key=lambda x: -abs(x)):
        residuals = [np.linalg.norm(A_aug.T @ vl_mat[:, i] - np.conj(lam) * vl_mat[:, i])
                     for i in range(Wd)]
        idx   = int(np.argmin(residuals))
        v     = vl_mat[:, idx]
        score = float(np.linalg.norm(v.conj() @ B_aug) /
                      (np.linalg.norm(v) * np.linalg.norm(B_aug, 'fro') + 1e-12))
        status = 'CONTROLLABLE' if score > 0.1 else ('MARGINAL' if score > 0.01 else 'UNCONTROLLABLE')
        print(f'  λ={lam:.4f}  PBH={score:.4f}  → {status}')
        if score <= 0.01:
            any_uncontrollable = True
    print()
    if any_uncontrollable:
        print('  ✗ System is NOT controllable at one or more unstable modes.')
        print('    B_aug is (nearly) orthogonal to the unstable subspace.')
        print('    Fix: PBH loss during training, or collect active data.')
    else:
        print('  ✓ System appears controllable (B_aug has projection onto unstable modes).')
        print('    If DARE LQR still fails, the problem is model fidelity (not controllability).')


if __name__ == '__main__':
    main()
