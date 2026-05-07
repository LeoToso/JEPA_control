"""JEPA v2 experiment runner."""
from __future__ import annotations
import os, sys, json, time, warnings, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import yaml


def run_experiment(encoder_variant='E-full', dataset_name='mixed', frame_skip=1,
                   seed=42, config_path='configs/cartpole_v2.yaml',
                   data_dir='data', results_dir='results',
                   device=None, skip_if_exists=True, eval_only=False):

    exp_name   = f'v2_{encoder_variant}_{dataset_name}_fs{frame_skip}_seed{seed}'
    out_dir    = Path(results_dir) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / 'results.json'

    if skip_if_exists and results_file.exists():
        print(f'[skip] {exp_name} already exists.')
        with open(results_file) as f:
            return json.load(f)

    print(f'\n{"="*60}\nEXPERIMENT: {exp_name}\n{"="*60}')
    t_start = time.time()

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    env_cfg   = cfg['environment']
    model_cfg = cfg['model']
    train_cfg = cfg['training']
    ctrl_cfg  = cfg['control']
    mpc_cfg   = cfg.get('mpc', {})
    probe_cfg = cfg.get('probes', {})
    env_cfg['frame_skip'] = frame_skip

    # Ground truth
    from ground_truth.cartpole_gt import CartpoleGroundTruth
    gt = CartpoleGroundTruth(
        mass_cart=env_cfg['mass_cart'], mass_pole=env_cfg['mass_pole'],
        pole_length=env_cfg['pole_length'], gravity=env_cfg['gravity'],
        dt=env_cfg['dt'] * frame_skip,
    )
    print(f'[GT] unstable eigenvalues: {np.round(gt.unstable_eigenvalues, 4)}')

    # Dataset
    from data.dataset import load_dataset, make_dataloaders, generate_dataset
    h5_path = Path(data_dir) / f'cartpole_v2_{dataset_name}_fs{frame_skip}_seed{seed}.h5'
    horizon = int(train_cfg.get('horizon', 20))
    if h5_path.exists():
        print(f'[data] Loading {h5_path}')
        data = load_dataset(str(h5_path))
    else:
        print(f'[data] Generating {dataset_name} dataset...')
        data = generate_dataset(
            dataset_type=dataset_name,
            n_transitions=cfg['data']['n_random'],
            frame_skip=frame_skip,
            save_path=str(h5_path),
            seed=seed,
            image_size=env_cfg['image_size'],
            init_range=float(cfg['data'].get('random_init_range', 0.15)),
            lqr_init_range=float(cfg['data'].get('lqr_init_range', 0.10)),
            lqr_noise_std=float(cfg['data'].get('lqr_noise_std', 0.1)),
        )
    loaders = make_dataloaders(data, batch_size=train_cfg['batch_size'],
                               horizon=horizon)
    print(f'[data] train={len(loaders["train"].dataset)}  '
          f'val={len(loaders["val"].dataset)}  horizon={horizon}')

    # Model
    from models.jepa import make_jepa, JEPAConfig
    model = make_jepa(
        variant=encoder_variant,
        latent_dim=model_cfg['latent_dim'],
        action_latent_dim=model_cfg['action_latent_dim'],
        image_size=env_cfg['image_size'],
        patch_size=model_cfg.get('patch_size', 8),
        vit_embed_dim=model_cfg.get('vit_embed_dim', 128),
        vit_depth=model_cfg.get('vit_depth', 4),
        vit_num_heads=model_cfg.get('vit_num_heads', 4),
        predictor_hidden_dim=model_cfg['predictor_hidden_dim'],
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[model] {encoder_variant}  {n_params:,} trainable params')

    # Train
    from training.trainer import Trainer
    train_cfg_exp = dict(train_cfg)
    trainer = Trainer(model=model, config_dict=train_cfg_exp, gt=gt,
                      save_dir=str(out_dir / 'checkpoints'), device=device, seed=seed)

    _state_head_trained = False
    saved_model = out_dir / 'model_final.pt'
    if eval_only and saved_model.exists():
        print(f'[train] --eval-only: loading {saved_model}')
        model.load_state_dict(torch.load(saved_model, map_location=device), strict=False)
        sh_path = out_dir / 'state_head.pt'
        if sh_path.exists() and trainer.state_head is not None:
            trainer.state_head.load_state_dict(
                torch.load(sh_path, map_location=device))
            _state_head_trained = True
            print(f'[train] loaded state_head from {sh_path}')
        history = {'train': [], 'val': []}
    else:
        print(f'[train] training for {train_cfg_exp["epochs"]} epochs ...')
        history = trainer.fit(
            loaders['train'], loaders['val'],
            epochs=train_cfg_exp['epochs'],
            checkpoint_every=train_cfg_exp.get('checkpoint_every', 10),
        )
        _state_head_trained = True

    # Compute z*
    from envs.cartpole_visual import ContinuousCartpoleVisual
    env = ContinuousCartpoleVisual(
        frame_skip=frame_skip, image_size=env_cfg['image_size'],
        mass_cart=env_cfg['mass_cart'], mass_pole=env_cfg['mass_pole'],
        pole_length=env_cfg['pole_length'], gravity=env_cfg['gravity'],
        dt=env_cfg['dt'], seed=seed,
    )
    model.eval()
    obs_eq, _, _ = env.reset_to_state(np.zeros(4))
    obs_eq_t = (torch.from_numpy(obs_eq).float()
                .permute(2, 0, 1)[None].to(device) / 255.0)
    with torch.no_grad():
        z_star = model.encoder(obs_eq_t).cpu().numpy()[0]
    print(f'[control] z_star norm: {np.linalg.norm(z_star):.3f}')

    # Jacobian A_jac, B_jac
    print('[control] Computing Jacobian at z* ...')
    from control.jacobian import compute_jacobian_np
    A_jac, B_jac = compute_jacobian_np(
        model.predictor, model.action_encoder, z_star, device)
    rho_jac = float(np.max(np.abs(np.linalg.eigvals(A_jac))))
    print(f'[control] rho(A_jac)={rho_jac:.4f}')

    # Probes
    print('\n[probes] Running probes ...')
    from probes.suite import run_all_probes
    probe_results = run_all_probes(
        A_jac=A_jac, B_jac=B_jac, gt=gt,
        model=model, env=env, z_star=z_star,
        device=device, config=probe_cfg,
    )

    # MPC setup
    action_lb = float(env_cfg.get('action_range', [-10, 10])[0])
    action_ub = float(env_cfg.get('action_range', [-10, 10])[1])
    mpc_horizon  = int(mpc_cfg.get('horizon', 20))
    mpc_chunk    = int(mpc_cfg.get('chunk_size', 1))
    mpc_Qf_mult  = float(mpc_cfg.get('Q_f_multiplier', 10.0))
    n_trials     = int(probe_cfg.get('n_trials_control', 100))
    T_rollout    = int(probe_cfg.get('T_rollout', 200))
    init_scale   = float(ctrl_cfg.get('init_scale', 0.05))
    R_lqr        = float(ctrl_cfg.get('R_lqr', 0.01)) * np.eye(B_jac.shape[1])
    d            = len(z_star)

    # Q from state_head if available, else identity
    if _state_head_trained and trainer.state_head is not None:
        W = trainer.state_head.weight.detach().cpu().numpy()  # (4, d)
        Q_phys_ctrl = np.diag([10.0, 0.1, 100.0, 0.1])
        Q_lqr = W.T @ Q_phys_ctrl @ W + 0.01 * np.eye(d)
        Q_lqr *= d / (np.trace(Q_lqr) + 1e-12)
    else:
        Q_lqr = np.eye(d)
    Q_f = mpc_Qf_mult * Q_lqr

    from control.rollout import evaluate_stabilization_mpc
    from control.visualize import save_rollout_frames, save_rollout_video
    ctrl_results = {}

    # ── GT-LQR sanity check ───────────────────────────────────────────────────
    try:
        from control.lqr import solve_discrete_lqr
        K_gt, _, _ = solve_discrete_lqr(gt.A_star, gt.B_star,
                                         np.diag([10., 0.1, 100., 0.1]), R_lqr)
        rng_gt = np.random.RandomState(seed + 999)
        gt_succs = []
        for _ in range(20):
            x0_gt = rng_gt.uniform(-init_scale, init_scale, 4).astype(np.float32)
            _, s_gt, _ = env.reset_to_state(x0_gt)
            done_gt = False
            for _ in range(T_rollout):
                u_gt = float(np.clip((-K_gt @ s_gt)[0], action_lb, action_ub))
                _, s_gt, _, done_gt, _ = env.step(u_gt)
                if done_gt: break
            gt_succs.append(int(not done_gt))
        print(f'[control] GT-LQR sanity: {np.mean(gt_succs):.2f}  ({sum(gt_succs)}/20)')
    except Exception as e:
        print(f'[control] GT-LQR failed: {e}')

    # ── Linear MPC (from Jacobian) ────────────────────────────────────────────
    print('\n[control] --- Linear MPC (Jacobian) ---')
    try:
        from control.mpc import LatentMPC
        # Pre-stabilise if needed
        from control.lqr import pre_stabilize_A
        A_stab, n_def = pre_stabilize_A(A_jac, gt.unstable_eigenvalues,
                                          tol=0.05, target=0.9)
        print(f'[control] A_jac pre-stab: deflated={n_def}  rho={np.max(np.abs(np.linalg.eigvals(A_stab))):.4f}')
        mpc_lin = LatentMPC(A=A_stab, B=B_jac, Q=Q_lqr, R=R_lqr,
                             horizon=mpc_horizon, chunk_size=mpc_chunk,
                             Q_f=Q_f, action_lb=action_lb, action_ub=action_ub)
        K_lin   = mpc_lin.K_list[0]
        A_cl    = A_jac - B_jac @ K_lin
        rho_cl  = float(np.max(np.abs(np.linalg.eigvals(A_cl))))
        print(f'[control] {mpc_lin.summary()}')
        print(f'[control] rho(A_cl)={rho_cl:.4f}  {"STABLE" if rho_cl < 1 else "UNSTABLE"}')

        cr_lin = evaluate_stabilization_mpc(
            encoder=model.encoder, mpc=mpc_lin, env=env,
            n_trials=n_trials, T=T_rollout, init_scale=init_scale,
            seed=seed, device=device, z_star=z_star, vis_trial=0,
        )
        print(f'[control] Linear MPC: success={cr_lin["success_rate"]:.3f}'
              f'  ep_len={cr_lin["mean_episode_length"]:.1f}'
              f'  frac_stable={cr_lin["mean_fraction_stable"]:.3f}')
        ctrl_results['linear_mpc'] = {k: v for k, v in cr_lin.items()
                                       if k != 'vis_result'}

        vis = cr_lin.get('vis_result')
        if vis:
            save_rollout_frames(vis, out_dir / 'linear_mpc_frames.png',
                                n_frames=8, title=f'{exp_name}  Linear MPC')
            save_rollout_video(vis, out_dir / 'linear_mpc.gif',
                               fps=15, title=f'{exp_name}  Linear MPC')
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f'[control] Linear MPC failed: {exc}')
        ctrl_results['linear_mpc'] = {'error': str(exc)}

    # ── Nonlinear gradient MPC ────────────────────────────────────────────────
    print('\n[control] --- Nonlinear Gradient MPC ---')
    try:
        from control.grad_mpc import GradientLatentMPC
        grad_mpc = GradientLatentMPC(
            predictor=model.predictor,
            action_encoder=model.action_encoder,
            Q=Q_lqr, R=R_lqr, Q_f=Q_f,
            horizon=mpc_horizon, chunk_size=mpc_chunk,
            action_lb=action_lb, action_ub=action_ub,
            lr=float(mpc_cfg.get('grad_lr', 0.05)),
            n_iter=int(mpc_cfg.get('grad_n_iter', 40)),
            device=device,
        )
        print(f'[control] {grad_mpc.summary()}')
        n_grad = min(int(probe_cfg.get('n_trials_grad_mpc', 50)), n_trials)
        cr_grad = evaluate_stabilization_mpc(
            encoder=model.encoder, mpc=grad_mpc, env=env,
            n_trials=n_grad, T=T_rollout, init_scale=init_scale,
            seed=seed, device=device, z_star=z_star, vis_trial=0,
        )
        print(f'[control] Grad MPC:   success={cr_grad["success_rate"]:.3f}'
              f'  ep_len={cr_grad["mean_episode_length"]:.1f}'
              f'  frac_stable={cr_grad["mean_fraction_stable"]:.3f}'
              f'  (n={n_grad})')
        ctrl_results['grad_mpc'] = {k: v for k, v in cr_grad.items()
                                     if k != 'vis_result'}

        vis_g = cr_grad.get('vis_result')
        if vis_g:
            save_rollout_frames(vis_g, out_dir / 'grad_mpc_frames.png',
                                n_frames=8, title=f'{exp_name}  Grad MPC')
            save_rollout_video(vis_g, out_dir / 'grad_mpc.gif',
                               fps=15, title=f'{exp_name}  Grad MPC')
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f'[control] Grad MPC failed: {exc}')
        ctrl_results['grad_mpc'] = {'error': str(exc)}

    env.close()

    # Save
    torch.save(model.state_dict(), out_dir / 'model_final.pt')
    np.save(out_dir / 'A_jac.npy', A_jac)
    np.save(out_dir / 'B_jac.npy', B_jac)
    np.save(out_dir / 'z_star.npy', z_star)
    if trainer.state_head is not None and _state_head_trained:
        torch.save(trainer.state_head.state_dict(), out_dir / 'state_head.pt')

    results = {
        'experiment': {'name': exp_name, 'variant': encoder_variant,
                       'dataset': dataset_name, 'seed': seed},
        'model':   {'n_params': n_params},
        'probes':  probe_results,
        'control': ctrl_results,
        'jacobian':{'rho': rho_jac},
        'elapsed_s': time.time() - t_start,
    }

    def _serial(obj):
        if isinstance(obj, dict):   return {k: _serial(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [_serial(v) for v in obj]
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)): return float(obj)
        if isinstance(obj, complex): return {'re': float(obj.real), 'im': float(obj.imag)}
        return obj

    with open(results_file, 'w') as f:
        json.dump(_serial(results), f, indent=2)
    print(f'\n[done] {exp_name} in {time.time()-t_start:.1f}s')
    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--variant',   default='E-full',
                   choices=['E-full', 'E-noact'])
    p.add_argument('--dataset',   default='mixed',
                   choices=['random', 'lqr', 'mixed'])
    p.add_argument('--frame_skip',type=int, default=1)
    p.add_argument('--seed',      type=int, default=42)
    p.add_argument('--config',    default='configs/cartpole_v2.yaml')
    p.add_argument('--data_dir',  default='data')
    p.add_argument('--results_dir', default='results')
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--force',     action='store_true')
    args = p.parse_args()
    run_experiment(
        encoder_variant=args.variant, dataset_name=args.dataset,
        frame_skip=args.frame_skip, seed=args.seed,
        config_path=args.config, data_dir=args.data_dir,
        results_dir=args.results_dir,
        skip_if_exists=not args.force,
        eval_only=args.eval_only,
    )
