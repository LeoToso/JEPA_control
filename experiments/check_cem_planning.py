"""Quick CEM stabilization check for a single checkpoint.

Loads a model (supports both flat model_final.pt and epoch checkpoint_epochNNNN.pt),
runs CEM latent planning for n_trials episodes, and reports success rate.

Usage:
    python experiments/check_cem_planning.py \\
        --checkpoint results/jepa_baseline_random_seed42/checkpoints/checkpoint_epoch0120.pt \\
        --config     configs/cartpole_jepa_baseline_random.yaml \\
        --n-trials 20 --T 200 \\
        --out results/comparison/cem_baseline_epoch120.png
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True,
                   help='Path to checkpoint_epochNNNN.pt or model_final.pt')
    p.add_argument('--config',     required=True)
    p.add_argument('--n-trials',   type=int,   default=20)
    p.add_argument('--T',          type=int,   default=200)
    p.add_argument('--horizon',    type=int,   default=None,
                   help='CEM planning horizon (default: from config cem.horizon)')
    p.add_argument('--alpha',      type=float, default=1.0,
                   help='Q = alpha * I  (state cost scale)')
    p.add_argument('--beta',       type=float, default=0.01,
                   help='R = beta  * I  (action cost scale)')
    p.add_argument('--n-samples',  type=int,   default=500)
    p.add_argument('--n-elites',   type=int,   default=50)
    p.add_argument('--n-iter',     type=int,   default=10)
    p.add_argument('--init-std',   type=float, default=3.0)
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--out',        default=None,
                   help='Save frame strip PNG to this path')
    p.add_argument('--gif',        default=None,
                   help='Save episode GIF to this path')
    p.add_argument('--device',     default=None)
    args = p.parse_args()

    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Device: {device}')

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_cfg   = cfg['environment']
    model_cfg = cfg['model']
    cem_cfg   = cfg.get('cem', {})
    ctrl_cfg  = cfg.get('control', {})

    frame_stack = int(model_cfg.get('frame_stack', 1))
    frame_skip  = int(env_cfg.get('frame_skip', 1))
    action_lb   = float(env_cfg['action_range'][0])
    action_ub   = float(env_cfg['action_range'][1])
    horizon     = args.horizon or int(cem_cfg.get('horizon', 25))

    # Ground truth for pre-stabilisation
    from ground_truth.cartpole_gt import CartpoleGroundTruth
    gt = CartpoleGroundTruth(
        mass_cart=env_cfg['mass_cart'], mass_pole=env_cfg['mass_pole'],
        pole_length=env_cfg['pole_length'], gravity=env_cfg['gravity'],
        dt=env_cfg['dt'] * frame_skip,
    )
    print(f'[GT] unstable eigenvalues: {np.round(gt.unstable_eigenvalues, 4)}')

    # Build and load model
    from models.jepa import make_jepa
    model = make_jepa(
        variant='E-full',
        latent_dim=int(model_cfg['latent_dim']),
        action_latent_dim=int(model_cfg.get('action_latent_dim', 1)),
        action_encoder=model_cfg.get('action_encoder', 'linear'),
        encoder_type=model_cfg.get('encoder_type', 'vit'),
        image_size=int(env_cfg['image_size']),
        patch_size=int(model_cfg.get('patch_size', 8)),
        frame_stack=frame_stack,
        vit_embed_dim=int(model_cfg.get('vit_embed_dim', 128)),
        vit_depth=int(model_cfg.get('vit_depth', 4)),
        vit_num_heads=int(model_cfg.get('vit_num_heads', 4)),
        predictor_hidden_dim=int(model_cfg.get('predictor_hidden_dim', 64)),
        predictor_n_layers=int(model_cfg.get('predictor_n_layers', 3)),
        predictor_window=int(model_cfg.get('predictor_window', 5)),
        predictor_residual=bool(model_cfg.get('predictor_residual', False)),
    )
    ckpt_path = Path(args.checkpoint)
    print(f'[model] Loading {ckpt_path}')
    raw = torch.load(ckpt_path, map_location=device)
    # Handle both epoch checkpoints (model_state key) and flat model_final.pt
    if isinstance(raw, dict) and 'model_state' in raw:
        state = raw['model_state']
    elif isinstance(raw, dict) and 'model_state_dict' in raw:
        state = raw['model_state_dict']
    else:
        state = raw
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f'[model]  missing   : {missing}')
    if unexpected:
        print(f'[model]  unexpected: {unexpected}')
    model.to(device).eval()

    # Environment + equilibrium observation
    from envs.cartpole_visual import ContinuousCartpoleVisual
    env = ContinuousCartpoleVisual(
        frame_skip=frame_skip, image_size=int(env_cfg['image_size']),
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

    # Jacobian — partial (d×d) and augmented (Wd×Wd)
    from control.jacobian import compute_jacobian_np, compute_augmented_jacobian_np
    W = getattr(model.config, 'predictor_window', 1)
    A_jac, B_jac = compute_jacobian_np(model, z_star, device)
    rho_partial = float(np.max(np.abs(np.linalg.eigvals(A_jac))))
    A_aug, B_aug = compute_augmented_jacobian_np(model, z_star, device)
    rho_aug = float(np.max(np.abs(np.linalg.eigvals(A_aug))))
    print(f'[control] W={W}  d={d}  ||z*||={np.linalg.norm(z_star):.3f}')
    print(f'[control] ρ(A_partial,{d}×{d})={rho_partial:.4f}  '
          f'ρ(A_aug,{W*d}×{W*d})={rho_aug:.4f}  '
          f'||B_partial||={np.linalg.norm(B_jac):.4f}  '
          f'||B_aug||={np.linalg.norm(B_aug):.4f}')

    # Fixed-point drift using windowed predict
    with torch.no_grad():
        z_star_t  = torch.tensor(z_star, dtype=torch.float32, device=device).unsqueeze(0)
        z_win_eq  = z_star_t.unsqueeze(1).expand(1, W, -1)
        u_win_eq  = torch.zeros(1, W, 1, device=device)
        z_pred_eq = model.predict(z_win_eq, u_win_eq)
        fp_err    = float(torch.norm(z_pred_eq - z_star_t).item())
    # Drift in partial (d) and augmented (W*d) spaces
    c_drift_partial = (z_pred_eq - z_star_t).cpu().numpy()[0]          # (d,)
    c_drift_aug     = np.concatenate([c_drift_partial,
                                      np.zeros((W - 1) * d)])           # (W*d,)
    print(f'[control] fp_err={fp_err:.4f}')

    # Pre-stabilise: try augmented first; fall back to partial if no unstable modes found
    from control.lqr import pre_stabilize_A
    A_aug_stab, n_def_aug = pre_stabilize_A(A_aug, gt.unstable_eigenvalues,
                                             tol=0.05, target=0.9)
    A_stab,     n_def     = pre_stabilize_A(A_jac, gt.unstable_eigenvalues,
                                             tol=0.05, target=0.9)
    print(f'[control] partial pre-stab: deflated={n_def}  '
          f'ρ={np.max(np.abs(np.linalg.eigvals(A_stab))):.4f}')
    print(f'[control] augmented pre-stab: deflated={n_def_aug}  '
          f'ρ={np.max(np.abs(np.linalg.eigvals(A_aug_stab))):.4f}')

    # Use augmented Jacobian if it found unstable modes; else fall back to partial
    use_aug = (n_def_aug > 0)
    if use_aug:
        print(f'[control] → using augmented ({W*d}D) Jacobian for CEM')
        A_plan   = A_aug_stab
        B_plan   = B_aug
        c_plan   = c_drift_aug
        d_plan   = W * d
        z_star_plan = np.tile(z_star, W)
    else:
        print(f'[control] → using partial ({d}D) Jacobian for CEM '
              f'(augmented found no unstable modes either)')
        A_plan   = A_stab
        B_plan   = B_jac
        c_plan   = c_drift_partial
        d_plan   = d
        z_star_plan = z_star

    # CEM
    from control.cem import CEMLatentPlanner
    from control.rollout import evaluate_stabilization_mpc

    Q  = args.alpha * np.eye(d_plan)
    Qf = args.alpha * np.eye(d_plan)
    R  = args.beta  * np.eye(1)

    cem = CEMLatentPlanner(
        A=A_plan, B=B_plan, c_offset=c_plan,
        Q=Q, R=R, Q_f=Qf,
        horizon=horizon, chunk_size=1,
        n_samples=args.n_samples,
        n_elites=args.n_elites,
        n_iter=args.n_iter,
        init_std=args.init_std,
        warm_start_sigma=0.5,
        action_lb=action_lb,
        action_ub=action_ub,
        device=device,
    )

    init_scale = float(ctrl_cfg.get('init_scale', 0.05))
    stab_thr   = float(ctrl_cfg.get('stabilization_threshold', 0.1))
    sett_thr   = float(ctrl_cfg.get('settling_threshold', 0.05))

    print(f'\n[eval] CEM  H={horizon}  α={args.alpha}  β={args.beta}  '
          f'n_trials={args.n_trials}  T={args.T}')

    cr = evaluate_stabilization_mpc(
        encoder=model.encoder, mpc=cem, env=env,
        n_trials=args.n_trials, T=args.T,
        init_scale=init_scale,
        stabilization_threshold=stab_thr,
        settling_threshold=sett_thr,
        seed=args.seed, device=device, z_star=z_star_plan,
        vis_trial=0,
        frame_stack=frame_stack,
    )

    print(f'\n[result] success_rate      = {cr["success_rate"]:.3f}')
    print(f'[result] mean_ep_length    = {cr["mean_episode_length"]:.1f}')
    print(f'[result] mean_frac_stable  = {cr["mean_fraction_stable"]:.3f}')
    if 'mean_cost' in cr:
        print(f'[result] mean_cost         = {cr["mean_cost"]:.2f}')

    # Save visualizations
    vis = cr.get('vis_result')
    if vis is not None:
        title = (f'CEM H={horizon} α={args.alpha} β={args.beta}  '
                 f'success={cr["success_rate"]:.2f}  '
                 f'frac={cr["mean_fraction_stable"]:.2f}')
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            from control.visualize import save_rollout_frames
            save_rollout_frames(vis, out_path, n_frames=8, title=title)
            print(f'[saved] frames -> {out_path}')
        if args.gif:
            gif_path = Path(args.gif)
            gif_path.parent.mkdir(parents=True, exist_ok=True)
            from control.visualize import save_rollout_video
            save_rollout_video(vis, gif_path, fps=15, title=title)
            print(f'[saved] gif    -> {gif_path}')


if __name__ == '__main__':
    main()
