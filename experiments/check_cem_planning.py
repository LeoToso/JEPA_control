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
                   help='Q = alpha * Q_shape  (state cost scale)')
    p.add_argument('--beta',       type=float, default=0.01,
                   help='R = beta  * I  (action cost scale)')
    p.add_argument('--pearson-Q',  action='store_true',
                   help='Build Q from Pearson correlations: Q = alpha*(C@W_phys@Cᵀ + eps*I)')
    p.add_argument('--w-phys',     type=float, nargs=4,
                   default=[1.0, 0.5, 10.0, 5.0],
                   metavar=('W_X','W_XDOT','W_THETA','W_THETADOT'),
                   help='Physical state weights for Pearson Q (default: 1 0.5 10 5)')
    p.add_argument('--pearson-eps', type=float, default=0.05,
                   help='Regularization added to Pearson Q before scaling (default: 0.05)')
    p.add_argument('--pearson-rollouts', type=int, default=60,
                   help='Rollouts used to estimate Pearson correlation matrix (default: 60)')
    p.add_argument('--nonlinear',  action='store_true',
                   help='Use the actual nonlinear predictor (windowed MLP) for CEM '
                        'instead of the linearized Jacobian. Q stays d-dimensional.')
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

    # Load checkpoint first so we can read the saved architecture config
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
    # Override model_cfg arch keys from checkpoint's saved config so that
    # changes to the yaml after training don't cause size mismatches.
    if isinstance(raw, dict) and 'config' in raw:
        _ckpt_cfg = raw['config']
        _arch_keys = ('latent_dim', 'action_latent_dim', 'action_encoder',
                      'encoder_type', 'patch_size', 'frame_stack',
                      'vit_embed_dim', 'vit_depth', 'vit_num_heads',
                      'predictor_type', 'predictor_hidden_dim', 'predictor_n_layers',
                      'predictor_window', 'predictor_embed_dim', 'predictor_depth',
                      'predictor_num_heads', 'predictor_mlp_ratio')
        for _k in _arch_keys:
            if _k in _ckpt_cfg and _ckpt_cfg[_k] != model_cfg.get(_k):
                print(f'[model] arch override: {_k}={_ckpt_cfg[_k]}  (yaml had {model_cfg.get(_k)})')
                model_cfg[_k] = _ckpt_cfg[_k]

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
        predictor_type=model_cfg.get('predictor_type', 'mlp'),
        predictor_hidden_dim=int(model_cfg.get('predictor_hidden_dim', 64)),
        predictor_n_layers=int(model_cfg.get('predictor_n_layers', 3)),
        predictor_window=int(model_cfg.get('predictor_window', 5)),
        predictor_embed_dim=int(model_cfg.get('predictor_embed_dim', 128)),
        predictor_depth=int(model_cfg.get('predictor_depth', 4)),
        predictor_num_heads=int(model_cfg.get('predictor_num_heads', 4)),
        predictor_mlp_ratio=float(model_cfg.get('predictor_mlp_ratio', 4.0)),
    )
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

    # Project B through the action encoder so CEM receives scalar (1-D) actions.
    # B_jac / B_aug have shape (d, action_latent_dim) or (W*d, action_latent_dim).
    # W_enc maps scalar action → latent action: shape (action_latent_dim, 1).
    if hasattr(model.action_encoder, 'W'):
        W_enc = model.action_encoder.W.weight.detach().cpu().numpy()  # (m, 1)
        B_jac = B_jac @ W_enc   # (d, 1)
        B_aug = B_aug @ W_enc   # (W*d, 1)
        print(f'[control] projected B through action encoder: B_jac {B_jac.shape}  B_aug {B_aug.shape}')

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

    # Use augmented Jacobian when rho(A_aug) > 1 — the unstable mode lives in the
    # history coupling and the partial 8×8 Jacobian misses it entirely.
    # Note: n_def_aug=0 is CORRECT when the unstable mode is physical (within tol
    # of GT eigenvalue); pre_stabilize_A leaves physical modes for DARE to handle.
    use_aug = (rho_aug > 1.0)
    if use_aug:
        print(f'[control] → using augmented ({W*d}D) Jacobian for CEM  '
              f'ρ={rho_aug:.4f}>1  phantom_deflated={n_def_aug}')
        A_plan   = A_aug_stab
        B_plan   = B_aug
        c_plan   = c_drift_aug
        d_plan   = W * d
        z_star_plan = np.tile(z_star, W)
    else:
        print(f'[control] → using partial ({d}D) Jacobian for CEM  '
              f'ρ(aug)={rho_aug:.4f}≤1')
        A_plan   = A_stab
        B_plan   = B_jac
        c_plan   = c_drift_partial
        d_plan   = d
        z_star_plan = z_star

    # Override to nonlinear d-dim planning if requested
    if args.nonlinear:
        d_plan = d
        z_star_plan = z_star
        print(f'[control] → nonlinear CEM mode  predictor_window={W}  Q is {d}×{d}')

    # CEM
    from control.cem import CEMLatentPlanner
    from control.rollout import evaluate_stabilization_mpc

    if args.pearson_Q:
        # Collect rollouts to estimate Pearson correlation matrix C (d×4)
        print(f'[pearson-Q] collecting {args.pearson_rollouts} rollouts to estimate C ...')
        rng_p = np.random.RandomState(0)
        zs_p, states_p = [], []
        for _ in range(args.pearson_rollouts):
            theta0 = rng_p.uniform(-0.6, 0.6)
            x0 = np.array([rng_p.uniform(-0.3, 0.3), 0., theta0, 0.], dtype=np.float32)
            obs, state, _ = env.reset_to_state(x0)
            prev_obs = obs.copy()
            for _ in range(50):
                obs_t = _make_obs_t(obs, prev_obs)
                with torch.no_grad():
                    z = model.encoder(obs_t).cpu().numpy()[0]
                zs_p.append(z); states_p.append(state.copy())
                action = env.sample_action()
                obs_next, state_next, _, done, _ = env.step(action)
                prev_obs = obs.copy(); obs = obs_next; state = state_next
                if done: break
        Z = np.array(zs_p)      # (N, d)
        S = np.array(states_p)  # (N, 4)
        # Pearson correlation matrix C: (d, 4)
        C = np.zeros((d, 4))
        for i in range(d):
            for j in range(4):
                zi, sj = Z[:, i] - Z[:, i].mean(), S[:, j] - S[:, j].mean()
                C[i, j] = np.dot(zi, sj) / (np.linalg.norm(zi) * np.linalg.norm(sj) + 1e-12)
        W_phys = np.diag(args.w_phys)
        Q_base = C @ W_phys @ C.T + args.pearson_eps * np.eye(d)
        # Normalise so trace = d (same as I), then apply alpha for overall scale
        Q_base = Q_base / (np.trace(Q_base) / d)
        Q_base = args.alpha * Q_base
        # Tile block-diagonally for augmented (W*d) state if needed; skip for nonlinear mode
        tile = W if (use_aug and not args.nonlinear) else 1
        Q  = np.kron(np.eye(tile), Q_base)
        Qf = Q.copy()
        print(f'[pearson-Q] C max|r|: x={np.max(np.abs(C[:,0])):.3f}  '
              f'xdot={np.max(np.abs(C[:,1])):.3f}  '
              f'theta={np.max(np.abs(C[:,2])):.3f}  '
              f'thetadot={np.max(np.abs(C[:,3])):.3f}')
        eigvals = np.linalg.eigvalsh(Q_base)
        print(f'[pearson-Q] Q_base eigs: min={eigvals[0]:.4f}  max={eigvals[-1]:.4f}  '
              f'cond={eigvals[-1]/(eigvals[0]+1e-12):.1f}  '
              f'w_phys={args.w_phys}')
    else:
        Q  = args.alpha * np.eye(d_plan)
        Qf = args.alpha * np.eye(d_plan)
    R  = args.beta  * np.eye(1)

    if args.nonlinear:
        cem = CEMLatentPlanner(
            predictor=model, action_encoder=model.action_encoder, predictor_window=W,
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
    else:
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

    q_desc = f'PearsonQ(w={args.w_phys})' if args.pearson_Q else 'αI'
    dyn_desc = 'nonlinear-predictor' if args.nonlinear else 'linear-jacobian'
    print(f'\n[eval] CEM  H={horizon}  Q={q_desc}  α={args.alpha}  β={args.beta}  '
          f'dynamics={dyn_desc}  n_trials={args.n_trials}  T={args.T}')

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

    print(f'\n[result] success_rate          = {cr["success_rate"]:.3f}')
    print(f'[result] mean_ep_length        = {cr["mean_episode_length"]:.1f}')
    print(f'[result] mean(1/ep_length²)    = {cr["mean_inv_sq_ep_length"]:.6f}  (lower = better)')
    print(f'[result] mean_frac_stable      = {cr["mean_fraction_stable"]:.3f}')
    if 'mean_cost' in cr:
        print(f'[result] mean_cost             = {cr["mean_cost"]:.2f}')

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
