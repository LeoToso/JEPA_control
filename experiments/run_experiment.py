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
                   device=None, skip_if_exists=True, eval_only=False,
                   epochs_override=None):

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
            n_equilibrium=int(cfg['data'].get('n_equilibrium', 0)),
            eq_init_range=float(cfg['data'].get('eq_init_range', 0.002)),
            eq_noise_std=float(cfg['data'].get('eq_noise_std', 0.001)),
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
    if epochs_override is not None:
        train_cfg_exp['epochs'] = int(epochs_override)
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

    # Fixed-point diagnostic: does f(z*, 0) ≈ z*?
    with torch.no_grad():
        z_star_t  = torch.tensor(z_star, dtype=torch.float32, device=device).unsqueeze(0)
        a_zero    = model.action_encoder(torch.zeros(1, 1, device=device))
        z_pred_eq = model.predictor(z_star_t, a_zero)
        fp_err = float(torch.norm(z_pred_eq - z_star_t).item())
    print(f'[control] Predictor fixed-point error ||f(z*,0)-z*|| = {fp_err:.4f}')

    # ── Diagnostics: equilibrium drift and state estimation ───────────────────
    c_drift = (z_pred_eq - z_star_t).cpu().numpy()[0]  # (d,) constant bias
    # Feedforward action that best cancels drift along B_jac direction:
    #   u_ff = -B^+ @ c  (least-squares; cancels the B-aligned component)
    BtB = B_jac.T @ B_jac + 1e-4 * np.eye(B_jac.shape[1])
    u_ff_lin = float(-np.linalg.solve(BtB, B_jac.T @ c_drift)[0])
    b_norm = float(np.linalg.norm(B_jac))
    cancelled = float(np.linalg.norm(B_jac * u_ff_lin))
    print(f'[control] B_jac norm: {b_norm:.4f}')
    print(f'[control] u_ff={u_ff_lin:.4f}  cancels {cancelled:.4f}/{fp_err:.4f} of drift')

    with torch.no_grad():
        # How does the predictor drift from z* over H steps with zero action?
        print('[control] Predictor drift from z* (u=0, 5 steps):')
        z_cur = z_star_t.clone()
        a_zero = model.action_encoder(torch.zeros(1, 1, device=device))
        for k in range(5):
            z_cur = model.predictor(z_cur, a_zero)
            dist = float(torch.norm(z_cur - z_star_t).item())
            state_str = ''
            if trainer.state_head is not None:
                x_k = trainer.state_head(z_cur).cpu().numpy()[0]
                state_str = f'  [x={x_k[0]:.3f} ẋ={x_k[1]:.3f} θ={x_k[2]:.3f} θ̇={x_k[3]:.3f}]'
            print(f'  step {k+1}: ||z-z*||={dist:.4f}{state_str}')
        # State_head estimate at z* (should be ~[0,0,0,0])
        if trainer.state_head is not None:
            x_eq = trainer.state_head(z_star_t).cpu().numpy()[0]
            print(f'[control] state_head(z*) = [{x_eq[0]:.3f}, {x_eq[1]:.3f},'
                  f' {x_eq[2]:.3f}, {x_eq[3]:.3f}]  (ideal: [0,0,0,0])')

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

    # Q from state_head if available, else identity.
    # Quality check: if the theta row of W is too weak, Q_lqr loses sensitivity
    # to the key control variable. Fall back to Q=I in that case.
    Q_lqr = np.eye(d)
    Q_lqr_source = 'identity'
    if _state_head_trained and trainer.state_head is not None:
        W = trainer.state_head.weight.detach().cpu().numpy()  # (4, d)
        theta_row_norm = float(np.linalg.norm(W[2]))
        Q_phys_ctrl = np.diag([10.0, 0.1, 100.0, 0.1])
        Q_cand = W.T @ Q_phys_ctrl @ W + 0.01 * np.eye(d)
        Q_cand *= d / (np.trace(Q_cand) + 1e-12)
        ev = np.linalg.eigvalsh(Q_cand)
        cond = float(ev.max() / (ev.min() + 1e-12))
        print(f'[control] Q_lqr candidate: theta_row_norm={theta_row_norm:.4f}'
              f'  cond={cond:.1f}')
        # Use state_head Q only if theta row is substantial (not collapsed)
        if theta_row_norm > 0.1 and cond < 1e6:
            Q_lqr = Q_cand
            Q_lqr_source = 'state_head'
    print(f'[control] Using Q_lqr from: {Q_lqr_source}')
    Q_f = mpc_Qf_mult * Q_lqr

    from control.rollout import evaluate_stabilization_mpc
    from control.visualize import save_rollout_frames, save_rollout_video
    ctrl_results = {}

    # Compute GT-LQR gain once — reused for sanity check and encoder-observer LQR
    from control.lqr import solve_discrete_lqr
    K_gt = None
    try:
        K_gt, _, _ = solve_discrete_lqr(gt.A_star, gt.B_star,
                                         np.diag([10., 0.1, 100., 0.1]), R_lqr)
    except Exception as e:
        print(f'[control] solve_discrete_lqr failed: {e}')

    # ── GT-LQR sanity check ───────────────────────────────────────────────────
    try:
        if K_gt is None:
            raise RuntimeError('K_gt not available')
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

    # ── Encoder-observer LQR ─────────────────────────────────────────────────
    # Use encoder+state_head as a visual state observer, apply GT-LQR gain.
    # No predictor needed — tests whether the encoder alone suffices for control.
    print('\n[control] --- Encoder-Observer LQR ---')
    try:
        if K_gt is None:
            raise RuntimeError('K_gt not available')
        if not (_state_head_trained and trainer.state_head is not None):
            raise RuntimeError('state_head not trained')
        state_head = trainer.state_head
        state_head.eval(); model.eval()
        rng_enc = np.random.RandomState(seed)
        succs_e, ep_lens_e, fracs_e = [], [], []
        vis_enc = None
        for trial in range(n_trials):
            x0 = rng_enc.uniform(-init_scale, init_scale, 4).astype(np.float32)
            obs, state, _ = env.reset_to_state(x0)
            done = False
            states_e, actions_e, all_obs_e = [state.copy()], [], []
            for _ in range(T_rollout):
                all_obs_e.append(obs.copy())
                obs_t = (torch.from_numpy(obs).float()
                         .permute(2, 0, 1)[None].to(device) / 255.0)
                with torch.no_grad():
                    x_hat = state_head(model.encoder(obs_t)).cpu().numpy()[0]
                u = float(np.clip((-K_gt @ x_hat)[0], action_lb, action_ub))
                actions_e.append([u])
                obs, state, _, done, _ = env.step(u)
                states_e.append(state.copy())
                if done:
                    break
            ep_len = len(states_e) - 1
            success = int(not done)
            frac = float(np.mean([abs(s[2]) < 0.1 for s in states_e]))
            succs_e.append(success); ep_lens_e.append(ep_len); fracs_e.append(frac)
            if trial == 0:
                vis_enc = {
                    'all_obs':           all_obs_e,
                    'states':            np.array(states_e),
                    'actions':           np.array(actions_e) if actions_e else np.zeros((1, 1)),
                    'done_at':           ep_len,
                    'stabilized':        bool(success),
                    'final_state_error': float(abs(states_e[-1][2])),
                }
        print(f'[control] Enc-Obs LQR: success={np.mean(succs_e):.3f}'
              f'  ep_len={np.mean(ep_lens_e):.1f}'
              f'  frac_stable={np.mean(fracs_e):.3f}')
        ctrl_results['enc_obs_lqr'] = {
            'success_rate':         float(np.mean(succs_e)),
            'mean_episode_length':  float(np.mean(ep_lens_e)),
            'mean_fraction_stable': float(np.mean(fracs_e)),
        }
        if vis_enc:
            save_rollout_frames(vis_enc, out_dir / 'enc_obs_lqr_frames.png',
                                n_frames=8, title=f'{exp_name}  Enc-Obs LQR')
            save_rollout_video(vis_enc, out_dir / 'enc_obs_lqr.gif',
                               fps=15, title=f'{exp_name}  Enc-Obs LQR')
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f'[control] Enc-Obs LQR failed: {exc}')
        ctrl_results['enc_obs_lqr'] = {'error': str(exc)}

    # ── Encoder-observer LQR (theta-only) ────────────────────────────────────
    # Zero out x and ẋ estimates — only trust visual theta/θ̇ from state_head.
    # Diagnoses whether inaccurate cart-position estimation causes the failure.
    print('\n[control] --- Encoder-Observer LQR (theta-only) ---')
    try:
        if K_gt is None:
            raise RuntimeError('K_gt not available')
        if not (_state_head_trained and trainer.state_head is not None):
            raise RuntimeError('state_head not trained')
        state_head = trainer.state_head
        state_head.eval(); model.eval()
        rng_enc_th = np.random.RandomState(seed)
        succs_th, ep_lens_th, fracs_th = [], [], []
        vis_enc_th = None
        for trial in range(n_trials):
            x0 = rng_enc_th.uniform(-init_scale, init_scale, 4).astype(np.float32)
            obs, state, _ = env.reset_to_state(x0)
            done = False
            states_th, actions_th, all_obs_th = [state.copy()], [], []
            for _ in range(T_rollout):
                all_obs_th.append(obs.copy())
                obs_t = (torch.from_numpy(obs).float()
                         .permute(2, 0, 1)[None].to(device) / 255.0)
                with torch.no_grad():
                    x_hat_full = state_head(model.encoder(obs_t)).cpu().numpy()[0]
                # Only use visual angle/angular-velocity estimates; zero cart x and ẋ
                x_hat = np.array([0.0, 0.0, x_hat_full[2], x_hat_full[3]], dtype=np.float32)
                u = float(np.clip((-K_gt @ x_hat)[0], action_lb, action_ub))
                actions_th.append([u])
                obs, state, _, done, _ = env.step(u)
                states_th.append(state.copy())
                if done:
                    break
            ep_len = len(states_th) - 1
            success = int(not done)
            frac = float(np.mean([abs(s[2]) < 0.1 for s in states_th]))
            succs_th.append(success); ep_lens_th.append(ep_len); fracs_th.append(frac)
            if trial == 0:
                vis_enc_th = {
                    'all_obs':           all_obs_th,
                    'states':            np.array(states_th),
                    'actions':           np.array(actions_th) if actions_th else np.zeros((1, 1)),
                    'done_at':           ep_len,
                    'stabilized':        bool(success),
                    'final_state_error': float(abs(states_th[-1][2])),
                }
        print(f'[control] Enc-Obs LQR (theta-only): success={np.mean(succs_th):.3f}'
              f'  ep_len={np.mean(ep_lens_th):.1f}'
              f'  frac_stable={np.mean(fracs_th):.3f}')
        ctrl_results['enc_obs_lqr_theta_only'] = {
            'success_rate':         float(np.mean(succs_th)),
            'mean_episode_length':  float(np.mean(ep_lens_th)),
            'mean_fraction_stable': float(np.mean(fracs_th)),
        }
        if vis_enc_th:
            save_rollout_frames(vis_enc_th, out_dir / 'enc_obs_lqr_theta_frames.png',
                                n_frames=8, title=f'{exp_name}  Enc-Obs LQR (θ-only)')
            save_rollout_video(vis_enc_th, out_dir / 'enc_obs_lqr_theta.gif',
                               fps=15, title=f'{exp_name}  Enc-Obs LQR (θ-only)')
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f'[control] Enc-Obs LQR (theta-only) failed: {exc}')
        ctrl_results['enc_obs_lqr_theta_only'] = {'error': str(exc)}

    # ── Pure-latent LQR (Q=I, bypasses state_head) ───────────────────────────
    # When state_head hasn't converged, Q_lqr=W^T@Q_phys@W is degenerate.
    # This controller drives z → z* directly with Q=I in latent space,
    # using only the Jacobian linearization — no state estimation needed.
    print('\n[control] --- Pure-Latent LQR (Q=I, no state_head) ---')
    try:
        from control.lqr import solve_discrete_lqr
        Q_lat_I = np.eye(d)
        K_lat, _, cl_eigs_lat = solve_discrete_lqr(A_jac, B_jac, Q_lat_I, R_lqr,
                                                    true_unstable_eigs=gt.unstable_eigenvalues,
                                                    pre_stabilize=True)
        rho_lat_cl = float(np.max(np.abs(cl_eigs_lat)))
        print(f'[control] Pure-Latent LQR: rho(A_cl)={rho_lat_cl:.4f}  '
              f'{"STABLE" if rho_lat_cl < 1 else "UNSTABLE"}')
        model.eval()
        rng_lat = np.random.RandomState(seed)
        succs_lat, ep_lens_lat, fracs_lat = [], [], []
        vis_lat = None
        for trial in range(n_trials):
            x0 = rng_lat.uniform(-init_scale, init_scale, 4).astype(np.float32)
            obs, state, _ = env.reset_to_state(x0)
            done = False
            states_lat, actions_lat, all_obs_lat = [state.copy()], [], []
            for _ in range(T_rollout):
                all_obs_lat.append(obs.copy())
                obs_t = (torch.from_numpy(obs).float()
                         .permute(2, 0, 1)[None].to(device) / 255.0)
                with torch.no_grad():
                    z_t_lat = model.encoder(obs_t).cpu().numpy()[0]
                u = float(np.clip(
                    (-K_lat @ (z_t_lat - z_star) + u_ff_lin)[0],
                    action_lb, action_ub))
                actions_lat.append([u])
                obs, state, _, done, _ = env.step(u)
                states_lat.append(state.copy())
                if done:
                    break
            ep_len = len(states_lat) - 1
            success = int(not done)
            frac = float(np.mean([abs(s[2]) < 0.1 for s in states_lat]))
            succs_lat.append(success); ep_lens_lat.append(ep_len); fracs_lat.append(frac)
            if trial == 0:
                vis_lat = {
                    'all_obs': all_obs_lat,
                    'states': np.array(states_lat),
                    'actions': np.array(actions_lat) if actions_lat else np.zeros((1,1)),
                    'done_at': ep_len,
                    'stabilized': bool(success),
                    'final_state_error': float(abs(states_lat[-1][2])),
                }
        print(f'[control] Pure-Latent LQR: success={np.mean(succs_lat):.3f}'
              f'  ep_len={np.mean(ep_lens_lat):.1f}'
              f'  frac_stable={np.mean(fracs_lat):.3f}')
        ctrl_results['pure_latent_lqr'] = {
            'success_rate':         float(np.mean(succs_lat)),
            'mean_episode_length':  float(np.mean(ep_lens_lat)),
            'mean_fraction_stable': float(np.mean(fracs_lat)),
        }
        if vis_lat:
            save_rollout_frames(vis_lat, out_dir / 'pure_latent_lqr_frames.png',
                                n_frames=8, title=f'{exp_name}  Pure-Latent LQR')
            save_rollout_video(vis_lat, out_dir / 'pure_latent_lqr.gif',
                               fps=15, title=f'{exp_name}  Pure-Latent LQR')
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f'[control] Pure-Latent LQR failed: {exc}')
        ctrl_results['pure_latent_lqr'] = {'error': str(exc)}

    # ── Linear MPC (from Jacobian) ────────────────────────────────────────────
    print('\n[control] --- Linear MPC (Jacobian) ---')
    try:
        from control.mpc import LatentMPC
        from control.lqr import pre_stabilize_A
        A_stab, n_def = pre_stabilize_A(A_jac, gt.unstable_eigenvalues,
                                          tol=0.05, target=0.9)
        print(f'[control] A_jac pre-stab: deflated={n_def}  rho={np.max(np.abs(np.linalg.eigvals(A_stab))):.4f}')

        for q_label, Q_use, Qf_use in [
            (Q_lqr_source, Q_lqr, mpc_Qf_mult * Q_lqr),
            ('identity',   np.eye(d), mpc_Qf_mult * np.eye(d)),
        ]:
            tag = 'linear_mpc' if q_label == Q_lqr_source else 'linear_mpc_qI'
            if tag == 'linear_mpc_qI' and Q_lqr_source == 'identity':
                continue  # already ran Q=I above; skip duplicate
            mpc_lin = LatentMPC(A=A_stab, B=B_jac, Q=Q_use, R=R_lqr,
                                 horizon=mpc_horizon, chunk_size=mpc_chunk,
                                 Q_f=Qf_use, action_lb=action_lb, action_ub=action_ub,
                                 u_offset=u_ff_lin, c_offset=c_drift)
            K_lin  = mpc_lin.K_list[0]
            rho_cl = float(np.max(np.abs(np.linalg.eigvals(A_jac - B_jac @ K_lin))))
            print(f'[control] {mpc_lin.summary()}  Q={q_label}')
            print(f'[control] rho(A_cl)={rho_cl:.4f}  {"STABLE" if rho_cl < 1 else "UNSTABLE"}')

            cr_lin = evaluate_stabilization_mpc(
                encoder=model.encoder, mpc=mpc_lin, env=env,
                n_trials=n_trials, T=T_rollout, init_scale=init_scale,
                seed=seed, device=device, z_star=z_star, vis_trial=0,
            )
            print(f'[control] Linear MPC (Q={q_label}): success={cr_lin["success_rate"]:.3f}'
                  f'  ep_len={cr_lin["mean_episode_length"]:.1f}'
                  f'  frac_stable={cr_lin["mean_fraction_stable"]:.3f}')
            ctrl_results[tag] = {k: v for k, v in cr_lin.items() if k != 'vis_result'}
            vis = cr_lin.get('vis_result')
            if vis:
                save_rollout_frames(vis, out_dir / f'{tag}_frames.png',
                                    n_frames=8, title=f'{exp_name}  Linear MPC (Q={q_label})')
                save_rollout_video(vis, out_dir / f'{tag}.gif',
                                   fps=15, title=f'{exp_name}  Linear MPC (Q={q_label})')
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

    # ── Gradient MPC with state_head physical-state cost ─────────────────────
    # Cost = Q_phys * ||state_head(z_t)||^2 instead of Q * ||z_t - z*||^2.
    # Optimizes physical state error directly; bypasses latent-space cost mis-alignment.
    print('\n[control] --- Gradient MPC (state_head cost) ---')
    try:
        from control.grad_mpc import GradientLatentMPC
        if not (_state_head_trained and trainer.state_head is not None):
            raise RuntimeError('state_head not trained')
        # Theta-focused physical cost: theta and theta_dot weighted heavily,
        # x/xdot included but lower weight since state_head may not estimate them well yet.
        Q_phys_sh = np.diag([1.0, 0.1, 100.0, 1.0])
        grad_mpc_sh = GradientLatentMPC(
            predictor=model.predictor,
            action_encoder=model.action_encoder,
            Q=Q_lqr, R=R_lqr, Q_f=Q_f,
            horizon=mpc_horizon, chunk_size=mpc_chunk,
            action_lb=action_lb, action_ub=action_ub,
            lr=float(mpc_cfg.get('grad_lr', 0.05)),
            n_iter=int(mpc_cfg.get('grad_n_iter', 50)),
            device=device,
            state_head=trainer.state_head,
            Q_phys=Q_phys_sh,
        )
        print(f'[control] {grad_mpc_sh.summary()} [state_head cost]')
        n_grad = min(int(probe_cfg.get('n_trials_grad_mpc', 50)), n_trials)
        cr_grad_sh = evaluate_stabilization_mpc(
            encoder=model.encoder, mpc=grad_mpc_sh, env=env,
            n_trials=n_grad, T=T_rollout, init_scale=init_scale,
            seed=seed, device=device, z_star=z_star, vis_trial=0,
        )
        print(f'[control] Grad MPC (SH): success={cr_grad_sh["success_rate"]:.3f}'
              f'  ep_len={cr_grad_sh["mean_episode_length"]:.1f}'
              f'  frac_stable={cr_grad_sh["mean_fraction_stable"]:.3f}'
              f'  (n={n_grad})')
        ctrl_results['grad_mpc_statehead'] = {k: v for k, v in cr_grad_sh.items()
                                               if k != 'vis_result'}
        vis_gsh = cr_grad_sh.get('vis_result')
        if vis_gsh:
            save_rollout_frames(vis_gsh, out_dir / 'grad_mpc_sh_frames.png',
                                n_frames=8, title=f'{exp_name}  Grad MPC (SH)')
            save_rollout_video(vis_gsh, out_dir / 'grad_mpc_sh.gif',
                               fps=15, title=f'{exp_name}  Grad MPC (SH)')
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f'[control] Grad MPC (state_head) failed: {exc}')
        ctrl_results['grad_mpc_statehead'] = {'error': str(exc)}

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
        'jacobian':{'rho': rho_jac, 'fixed_point_error': fp_err},
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
    p.add_argument('--epochs',    type=int, default=None,
                   help='Override training epochs from config')
    args = p.parse_args()
    run_experiment(
        encoder_variant=args.variant, dataset_name=args.dataset,
        frame_skip=args.frame_skip, seed=args.seed,
        config_path=args.config, data_dir=args.data_dir,
        results_dir=args.results_dir,
        skip_if_exists=not args.force,
        eval_only=args.eval_only,
        epochs_override=args.epochs,
    )
