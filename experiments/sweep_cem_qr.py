"""Sweep Q=αI, R=βI for linear CEM on the current best model.

Usage:
    python experiments/sweep_cem_qr.py
    python experiments/sweep_cem_qr.py --model results/v2_E-full_mixed_fs1_fstack2_seed42/model_final.pt
    python experiments/sweep_cem_qr.py --alphas 0.1 1 10 100 --betas 0.001 0.01 0.1
    python experiments/sweep_cem_qr.py --n-trials 20  # faster, fewer trials
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',   default='configs/cartpole_v2_fullspec.yaml')
    parser.add_argument('--model',    default='results/v2_E-full_mixed_fs1_fstack2_seed42/model_final.pt')
    parser.add_argument('--seed',     type=int,   default=42)
    parser.add_argument('--n-trials', type=int,   default=50)
    parser.add_argument('--T',        type=int,   default=200)
    parser.add_argument('--horizons', type=int,   nargs='+', default=[10, 25])
    parser.add_argument('--alphas',   type=float, nargs='+',
                        default=[0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0])
    parser.add_argument('--betas',    type=float, nargs='+',
                        default=[0.001, 0.01, 0.1, 1.0])
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--n-elites',  type=int, default=50)
    parser.add_argument('--n-iter',    type=int, default=20)
    parser.add_argument('--init-std',  type=float, default=3.0)
    parser.add_argument('--device',    default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Device: {device}')

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_cfg   = cfg['environment']
    model_cfg = cfg['model']
    frame_stack = int(model_cfg.get('frame_stack', 1))
    frame_skip  = int(env_cfg.get('frame_skip', 1))
    action_lb   = float(env_cfg['action_range'][0])
    action_ub   = float(env_cfg['action_range'][1])

    # Ground truth (for pre-stabilisation of A_jac)
    from ground_truth.cartpole_gt import CartpoleGroundTruth
    gt = CartpoleGroundTruth(
        mass_cart=env_cfg['mass_cart'], mass_pole=env_cfg['mass_pole'],
        pole_length=env_cfg['pole_length'], gravity=env_cfg['gravity'],
        dt=env_cfg['dt'] * frame_skip,
    )
    print(f'[GT] unstable eigenvalues: {np.round(gt.unstable_eigenvalues, 4)}')

    # Build model
    from models.jepa import make_jepa
    model = make_jepa(
        variant='E-full',
        latent_dim=model_cfg['latent_dim'],
        action_latent_dim=model_cfg['action_latent_dim'],
        action_encoder=model_cfg.get('action_encoder', 'none'),
        image_size=env_cfg['image_size'],
        patch_size=model_cfg.get('patch_size', 8),
        frame_stack=frame_stack,
        vit_embed_dim=model_cfg.get('vit_embed_dim', 128),
        vit_depth=model_cfg.get('vit_depth', 4),
        vit_num_heads=model_cfg.get('vit_num_heads', 4),
        predictor_hidden_dim=model_cfg['predictor_hidden_dim'],
    )
    model_path = Path(args.model)
    print(f'[model] Loading {model_path}')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device).eval()

    # Environment and equilibrium observation
    from envs.cartpole_visual import ContinuousCartpoleVisual
    env = ContinuousCartpoleVisual(
        frame_skip=frame_skip, image_size=env_cfg['image_size'],
        mass_cart=env_cfg['mass_cart'], mass_pole=env_cfg['mass_pole'],
        pole_length=env_cfg['pole_length'], gravity=env_cfg['gravity'],
        dt=env_cfg['dt'], seed=args.seed,
    )

    def _make_obs_t(obs_np, prev_obs_np=None):
        curr = torch.from_numpy(obs_np).float().permute(2, 0, 1)[None].to(device) / 255.0
        if frame_stack > 1:
            prev = (curr if prev_obs_np is None else
                    torch.from_numpy(prev_obs_np).float().permute(2, 0, 1)[None].to(device) / 255.0)
            return torch.cat([prev, curr], dim=1)
        return curr

    obs_eq, _, _ = env.reset_to_state(np.zeros(4, dtype=np.float32))
    obs_eq_t = _make_obs_t(obs_eq)

    with torch.no_grad():
        z_star = model.encoder(obs_eq_t).cpu().numpy()[0]
    d = len(z_star)
    print(f'[control] d={d}  ||z*||={np.linalg.norm(z_star):.3f}')

    # Jacobian and diagnostics
    from control.jacobian import compute_jacobian_np
    A_jac, B_jac = compute_jacobian_np(model.predictor, model.action_encoder, z_star, device)
    rho_jac = float(np.max(np.abs(np.linalg.eigvals(A_jac))))
    print(f'[control] ρ(A_jac)={rho_jac:.4f}  ||B||={np.linalg.norm(B_jac):.4f}')

    with torch.no_grad():
        z_star_t = torch.tensor(z_star, dtype=torch.float32, device=device).unsqueeze(0)
        a_zero   = model.action_encoder(torch.zeros(1, 1, device=device))
        z_pred_eq = model.predictor(z_star_t, a_zero)
        fp_err   = float(torch.norm(z_pred_eq - z_star_t).item())
    c_drift = (z_pred_eq - z_star_t).cpu().numpy()[0]
    BtB = B_jac.T @ B_jac + 1e-4 * np.eye(B_jac.shape[1])
    u_ff_lin = float(-np.linalg.solve(BtB, B_jac.T @ c_drift)[0])
    print(f'[control] fp_err={fp_err:.4f}  u_ff={u_ff_lin:.4f}')

    # Pre-stabilise A for linear CEM
    from control.lqr import pre_stabilize_A
    A_stab, n_def = pre_stabilize_A(A_jac, gt.unstable_eigenvalues, tol=0.05, target=0.9)
    print(f'[control] A pre-stab: deflated={n_def}  '
          f'ρ={np.max(np.abs(np.linalg.eigvals(A_stab))):.4f}')

    from control.cem import CEMLatentPlanner
    from control.rollout import evaluate_stabilization_mpc

    ctrl_cfg   = cfg.get('control', {})
    init_scale = float(ctrl_cfg.get('init_scale', 0.05))
    stab_thr   = float(ctrl_cfg.get('stabilization_threshold', 0.1))
    sett_thr   = float(ctrl_cfg.get('settling_threshold', 0.05))

    # Collect (α, β, H) combinations
    combos = [(a, b, H)
              for H in args.horizons
              for a in args.alphas
              for b in args.betas]

    print(f'\nSweeping {len(combos)} combinations '
          f'({len(args.horizons)} horizons × '
          f'{len(args.alphas)} α × {len(args.betas)} β)  '
          f'n_trials={args.n_trials}\n')

    header = f"{'H':>4}  {'α':>8}  {'β':>8}  {'success':>8}  {'ep_len':>8}  {'frac_stb':>9}  {'mean_cost':>10}"
    print(header)
    print('-' * len(header))

    results = []
    t0 = time.time()
    for (alpha, beta, H) in combos:
        Q  = alpha * np.eye(d)
        Qf = alpha * np.eye(d)   # terminal = same scale as stage cost
        R  = beta  * np.eye(1)

        cem = CEMLatentPlanner(
            A=A_stab, B=B_jac, c_offset=c_drift,
            Q=Q, R=R, Q_f=Qf,
            horizon=H, chunk_size=1,
            n_samples=args.n_samples,
            n_elites=args.n_elites,
            n_iter=args.n_iter,
            init_std=args.init_std,
            warm_start_sigma=0.5,
            action_lb=action_lb,
            action_ub=action_ub,
            device=device,
        )

        cr = evaluate_stabilization_mpc(
            encoder=model.encoder, mpc=cem, env=env,
            n_trials=args.n_trials, T=args.T,
            init_scale=init_scale,
            stabilization_threshold=stab_thr,
            settling_threshold=sett_thr,
            seed=args.seed, device=device, z_star=z_star,
            frame_stack=frame_stack,
        )
        row = dict(H=H, alpha=alpha, beta=beta,
                   success=cr['success_rate'],
                   ep_len=cr['mean_episode_length'],
                   frac_stable=cr['mean_fraction_stable'],
                   mean_cost=cr.get('mean_cost', float('nan')))
        results.append(row)
        print(f"{H:>4}  {alpha:>8.3f}  {beta:>8.4f}  "
              f"{cr['success_rate']:>8.3f}  "
              f"{cr['mean_episode_length']:>8.1f}  "
              f"{cr['mean_fraction_stable']:>9.3f}  "
              f"{cr.get('mean_cost', float('nan')):>10.1f}")

    elapsed = time.time() - t0
    print(f'\nDone in {elapsed:.0f}s')

    # Summary: top-5 by success_rate then ep_len
    results.sort(key=lambda r: (-r['success'], -r['frac_stable'], r['ep_len']))
    print('\n── Top-10 configurations (by success → frac_stable → ep_len) ──')
    print(header)
    print('-' * len(header))
    for row in results[:10]:
        print(f"{row['H']:>4}  {row['alpha']:>8.3f}  {row['beta']:>8.4f}  "
              f"{row['success']:>8.3f}  "
              f"{row['ep_len']:>8.1f}  "
              f"{row['frac_stable']:>9.3f}  "
              f"{row['mean_cost']:>10.1f}")

    # Save results as numpy-friendly dict
    out_dir = Path('results') / 'cem_qr_sweep'
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = out_dir / f'sweep_{ts}.npy'
    np.save(out_path, results)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
