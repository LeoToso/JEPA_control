"""JEPA v2 experiment runner."""
from __future__ import annotations
import os, sys, json, time, warnings, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


def run_experiment(encoder_variant='E-full', dataset_name='mixed', frame_skip=1,
                   seed=42, config_path='configs/cartpole_v2.yaml',
                   data_dir='data', results_dir='results',
                   device=None, skip_if_exists=True, eval_only=False,
                   epochs_override=None, cem_only=False):

    t_start = time.time()

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    frame_stack = int(model_cfg.get('frame_stack', 1))

    fstack_str = f'_fstack{frame_stack}' if frame_stack > 1 else ''
    exp_name   = f'v2_{encoder_variant}_{dataset_name}_fs{frame_skip}{fstack_str}_seed{seed}'
    # ckpt_dir: stable path for model weights — --eval-only always finds the latest model here.
    # out_dir:  timestamped path for all eval outputs (results, frames, videos, npy arrays).
    ckpt_dir   = Path(results_dir) / exp_name
    timestamp  = time.strftime('%Y%m%d_%H%M%S')
    out_dir    = ckpt_dir / timestamp
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / 'results.json'

    if skip_if_exists and (ckpt_dir / 'model_final.pt').exists() and not eval_only:
        print(f'[skip] {exp_name} already trained. Use --force to retrain.')
        latest = sorted(ckpt_dir.glob('*/results.json'))
        if latest:
            with open(latest[-1]) as f:
                return json.load(f)

    print(f'\n{"="*60}\nEXPERIMENT: {exp_name}\n{"="*60}')
    print(f'Device: {device}')

    # Ground truth
    from ground_truth.cartpole_gt import CartpoleGroundTruth
    gt = CartpoleGroundTruth(
        mass_cart=env_cfg['mass_cart'], mass_pole=env_cfg['mass_pole'],
        pole_length=env_cfg['pole_length'], gravity=env_cfg['gravity'],
        dt=env_cfg['dt'] * frame_skip,
    )
    print(f'[GT] unstable eigenvalues: {np.round(gt.unstable_eigenvalues, 4)}')

    # Equilibrium observation — needed both for self-loop injection and fp-loss anchor.
    # Create early so it can be passed to make_dataloaders.
    from envs.cartpole_visual import ContinuousCartpoleVisual as _CVEnv
    _eq_env = _CVEnv(frame_skip=frame_skip, image_size=env_cfg['image_size'],
                     mass_cart=env_cfg['mass_cart'], mass_pole=env_cfg['mass_pole'],
                     pole_length=env_cfg['pole_length'], gravity=env_cfg['gravity'],
                     action_range=tuple(env_cfg['action_range']))
    _obs_eq, _, _ = _eq_env.reset_to_state(np.zeros(4, dtype=np.float32))
    _eq_env.close()
    n_eq_selfloop = int(cfg['data'].get('n_eq_selfloop', 0))
    if n_eq_selfloop > 0:
        print(f'[data] Self-loop injection: n_eq_selfloop={n_eq_selfloop}')

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
            action_range=tuple(env_cfg['action_range']),
            init_range=float(cfg['data'].get('random_init_range', 0.15)),
            lqr_init_range=float(cfg['data'].get('lqr_init_range', 0.10)),
            lqr_noise_std=float(cfg['data'].get('lqr_noise_std', 0.1)),
            n_equilibrium=int(cfg['data'].get('n_equilibrium', 0)),
            eq_init_range=float(cfg['data'].get('eq_init_range', 0.002)),
            eq_noise_std=float(cfg['data'].get('eq_noise_std', 0.001)),
        )
    loaders = make_dataloaders(data, batch_size=train_cfg['batch_size'],
                               horizon=horizon, frame_stack=frame_stack,
                               obs_eq=_obs_eq, n_eq_selfloop=n_eq_selfloop)
    print(f'[data] train={len(loaders["train"].dataset)}  '
          f'val={len(loaders["val"].dataset)}  horizon={horizon}'
          + (f'  (+{n_eq_selfloop} eq-selfloops)' if n_eq_selfloop > 0 else ''))

    # Model
    from models.jepa import make_jepa, JEPAConfig
    model = make_jepa(
        variant=encoder_variant,
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
        predictor_window=int(model_cfg.get('predictor_window', 1)),
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
                      save_dir=str(ckpt_dir / 'checkpoints'), device=device, seed=seed)

    # Give trainer the exact equilibrium image so fp loss trains at encoder(obs_eq),
    # not at an EMA over near-eq batch samples. Eliminates train/eval z* mismatch.
    # (_obs_eq already created above before make_dataloaders)
    trainer.set_obs_eq(_obs_eq)
    print('[train] z* anchor: using exact equilibrium observation for fp loss')

    # Pre-compute stacked equilibrium tensor — used throughout for z* and encoder calls.
    # With frame_stack > 1, duplicate the same frame (prev=curr at episode start).
    def _make_obs_t(obs_np, prev_obs_np=None):
        """Convert HWC uint8 obs to (1, C, H, W) float tensor; stack with prev if needed."""
        curr = torch.from_numpy(obs_np).float().permute(2, 0, 1)[None].to(device) / 255.0
        if frame_stack > 1:
            prev = (curr if prev_obs_np is None else
                    torch.from_numpy(prev_obs_np).float().permute(2, 0, 1)[None].to(device) / 255.0)
            return torch.cat([prev, curr], dim=1)
        return curr

    obs_eq_t = _make_obs_t(_obs_eq)   # (1, 3*FS, h, w) — used for z* throughout

    saved_model = ckpt_dir / 'model_final.pt'
    if eval_only and saved_model.exists():
        print(f'[train] --eval-only: loading {saved_model}')
        model.load_state_dict(torch.load(saved_model, map_location=device), strict=False)
        history = {'train': [], 'val': []}
    else:
        print(f'[train] training for {train_cfg_exp["epochs"]} epochs ...')
        history = trainer.fit(
            loaders['train'], loaders['val'],
            epochs=train_cfg_exp['epochs'],
            checkpoint_every=train_cfg_exp.get('checkpoint_every', 10),
        )
        # Save immediately so --eval-only works even if probes are interrupted
        torch.save(model.state_dict(), ckpt_dir / 'model_final.pt')

    # ── Post-training linear state decoder (evaluation metric for encoder comparison) ──
    # Train a simple linear z→state decoder on training data, then evaluate:
    # State decoder disabled: action lifting + dynamics-aware losses make
    # the latent geometry harder to interpret via a linear probe, and the
    # downstream evaluation uses Q=I (identity) CEM which doesn't need Q_lat.
    print('[state_decoder] skipped (disabled)')
    state_decoder_results = {}
    Q_lat_diag = np.eye(d)   # unused placeholder; CEM uses Q=I

    # ── Post-hoc linear state probe (diagnostic only, not used for CEM cost) ──
    # Skipped in --cem-only mode since CEM doesn't use state_head.
    sh_path = ckpt_dir / 'state_head.pt'
    if cem_only:
        print('[probe] --cem-only: skipping state probe training')
        state_head = None
    else:
        state_head = torch.nn.Linear(model.config.latent_dim, 4).to(device)
        if eval_only and sh_path.exists():
            state_head.load_state_dict(torch.load(sh_path, map_location=device))
            print(f'[probe] loaded state probe from {sh_path}')
        else:
            print('[probe] Training post-hoc linear state probe (30 epochs, frozen encoder)...')
            model.encoder.eval()
            for p in model.encoder.parameters():
                p.requires_grad_(False)
            probe_opt = torch.optim.Adam(state_head.parameters(), lr=1e-3, weight_decay=1e-4)
            w_pr = torch.tensor([50., 0.1, 100., 1.], device=device)
            # Equilibrium anchor: force state_head(z*) -> 0 during probe training
            # so the post-hoc probe agrees with the in-training anchor constraint.
            with torch.no_grad():
                z_star_probe = model.encoder(obs_eq_t).squeeze(0)
            for _ in range(30):
                for batch in loaders['train']:
                    obs_seq  = batch['obs_seq'].to(device)
                    states_b = batch['states'].to(device).float()
                    Bp, H1p, Cp, hp, wp = obs_seq.shape
                    with torch.no_grad():
                        z_flat = model.encoder(obs_seq.view(Bp * H1p, Cp, hp, wp))
                    z_view = z_flat.view(Bp, H1p, -1)
                    loss_p = sum(
                        (w_pr * (state_head(z_view[:, k]) - states_b[:, k]).pow(2)).mean()
                        for k in range(H1p)
                    ) / H1p
                    # anchor: state_head(z*) must read as [0,0,0,0]
                    loss_p = loss_p + (w_pr * state_head(z_star_probe.unsqueeze(0)).pow(2)).mean()
                    probe_opt.zero_grad(); loss_p.backward(); probe_opt.step()
            torch.save(state_head.state_dict(), sh_path)
            print(f'[probe] state probe saved to {sh_path}')
        state_head.eval()

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
    # obs_eq_t already computed from _obs_eq above; recompute here from fresh env obs
    # to confirm they match (both are the equilibrium image — should be identical).
    obs_eq_t = _make_obs_t(obs_eq)   # (1, 3*FS, h, w)
    with torch.no_grad():
        z_star = model.encoder(obs_eq_t).cpu().numpy()[0]
    print(f'[control] z_star norm: {np.linalg.norm(z_star):.3f}')

    # Jacobian A_jac, B_jac
    print('[control] Computing Jacobian at z* ...')
    from control.jacobian import compute_jacobian_np
    A_jac, B_jac = compute_jacobian_np(model, z_star, device)
    rho_jac = float(np.max(np.abs(np.linalg.eigvals(A_jac))))
    print(f'[control] rho(A_jac)={rho_jac:.4f}')

    # Fixed-point diagnostic: does f(z*, 0) ≈ z*?
    _W = model.config.predictor_window
    with torch.no_grad():
        z_star_t = torch.tensor(z_star, dtype=torch.float32, device=device).unsqueeze(0)
        z_star_win = z_star_t.unsqueeze(1).expand(1, _W, -1)   # (1, W, d)
        u_zero_win = torch.zeros(1, _W, 1, device=device)
        z_pred_eq  = model.predict(z_star_win, u_zero_win)
        fp_err = float(torch.norm(z_pred_eq - z_star_t).item())
    print(f'[control] Predictor fixed-point error ||f(z*,0)-z*|| = {fp_err:.4f}')

    # ── Diagnostics: equilibrium drift and state estimation ───────────────────
    c_drift = (z_pred_eq - z_star_t).cpu().numpy()[0]  # (d,) constant bias — from windowed predict above
    # Feedforward action that best cancels drift along B_jac direction:
    #   u_ff = -B^+ @ c  (least-squares; cancels the B-aligned component)
    BtB = B_jac.T @ B_jac + 1e-4 * np.eye(B_jac.shape[1])
    u_ff_lin = float(-np.linalg.solve(BtB, B_jac.T @ c_drift)[0])
    b_norm = float(np.linalg.norm(B_jac))
    cancelled = float(np.linalg.norm(B_jac * u_ff_lin))
    print(f'[control] B_jac norm: {b_norm:.4f}')
    print(f'[control] u_ff={u_ff_lin:.4f}  cancels {cancelled:.4f}/{fp_err:.4f} of drift')

    # ── True predictor fixed point z_fp ──────────────────────────────────────
    # Solve (I - A_jac) @ dz = c_drift + B_jac * u_ff_lin in the linear model.
    # At z_fp the model predicts f(z_fp, u_ff) ≈ z_fp, giving CEM an achievable target.
    try:
        _rhs_fp  = c_drift + B_jac.flatten() * u_ff_lin
        _dz_fp   = np.linalg.solve(np.eye(A_jac.shape[0]) - A_jac, _rhs_fp)
        z_fp     = z_star + _dz_fp
        with torch.no_grad():
            _z_fp_t  = torch.tensor(z_fp, dtype=torch.float32, device=device).unsqueeze(0)
            _z_fp_win = _z_fp_t.unsqueeze(1).expand(1, _W, -1)
            _u_ff_win = torch.full((1, _W, 1), u_ff_lin, device=device)
            _z_fp_nl  = model.predict(_z_fp_win, _u_ff_win)
            fp_err_fp = float(torch.norm(_z_fp_nl - _z_fp_t).item())
        print(f'[control] z_fp: ||z_fp-z_star||={np.linalg.norm(_dz_fp):.4f}  '
              f'fp_err={fp_err_fp:.4f}  (was {fp_err:.4f} at z_star)')
        np.save(out_dir / 'z_fp.npy', z_fp)
    except Exception as _e:
        print(f'[control] z_fp computation failed: {_e}  (falling back to z_star)')
        z_fp = z_star

    with torch.no_grad():
        # How does the predictor drift from z* over H steps with zero action?
        print('[control] Predictor drift from z* (u=0, 5 steps):')
        z_drift_win = z_star_t.unsqueeze(1).expand(1, _W, -1).clone()  # (1, W, d)
        u_drift_win = torch.zeros(1, _W, 1, device=device)
        for k in range(5):
            z_next = model.predict(z_drift_win, u_drift_win)
            z_drift_win = torch.cat([z_drift_win[:, 1:], z_next.unsqueeze(1)], dim=1)
            dist = float(torch.norm(z_next - z_star_t).item())
            state_str = ''
            if state_head is not None:
                x_k = state_head(z_next).cpu().numpy()[0]
                state_str = f'  [x={x_k[0]:.3f} ẋ={x_k[1]:.3f} θ={x_k[2]:.3f} θ̇={x_k[3]:.3f}]'
            print(f'  step {k+1}: ||z-z*||={dist:.4f}{state_str}')
        # State_head estimate at z* (should be ~[0,0,0,0])
        if state_head is not None:
            x_eq = state_head(z_star_t).cpu().numpy()[0]
            print(f'[control] state_head(z*) = [{x_eq[0]:.3f}, {x_eq[1]:.3f},'
                  f' {x_eq[2]:.3f}, {x_eq[3]:.3f}]  (ideal: [0,0,0,0])')

    # Probes
    print('\n[probes] Running probes ...')
    from probes.suite import run_all_probes
    probe_results = run_all_probes(
        A_jac=A_jac, B_jac=B_jac, gt=gt,
        model=model, env=env, z_star=z_star,
        device=device, config=probe_cfg,
        frame_stack=frame_stack,
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

    # Pure JEPA: CEM cost is identity in latent space — ||z - z*||^2.
    # State probe is diagnostic only; encoder was never trained with state gradients.
    Q_lqr = np.eye(d)
    Q_lqr_source = 'identity'
    print('[control] Using Q_lqr: identity (pure latent cost)')
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

    if cem_only:
        print('[control] --cem-only: skipping LQR and Linear MPC controllers')

    # ── Encoder-observer LQR ─────────────────────────────────────────────────
    # Use encoder+state_head as a visual state observer, apply GT-LQR gain.
    # No predictor needed — tests whether the encoder alone suffices for control.
    if not cem_only:
        print('\n[control] --- Encoder-Observer LQR ---')
    if not cem_only:
        try:
            if K_gt is None:
                raise RuntimeError('K_gt not available')
            if not (state_head is not None):
                raise RuntimeError('state_head not trained')
            state_head = state_head
            state_head.eval(); model.eval()
            rng_enc = np.random.RandomState(seed)
            succs_e, ep_lens_e, fracs_e = [], [], []
            vis_enc = None
            for trial in range(n_trials):
                x0 = rng_enc.uniform(-init_scale, init_scale, 4).astype(np.float32)
                obs, state, _ = env.reset_to_state(x0)
                done = False
                prev_obs_e = None
                states_e, actions_e, all_obs_e = [state.copy()], [], []
                for _ in range(T_rollout):
                    all_obs_e.append(obs.copy())
                    obs_t = _make_obs_t(obs, prev_obs_e)
                    with torch.no_grad():
                        x_hat = state_head(model.encoder(obs_t)).cpu().numpy()[0]
                    u = float(np.clip((-K_gt @ x_hat)[0], action_lb, action_ub))
                    actions_e.append([u])
                    prev_obs_e = obs.copy()
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
    if not cem_only: print('\n[control] --- Encoder-Observer LQR (theta-only) ---')
    if not cem_only:
        try:
            if K_gt is None:
                raise RuntimeError('K_gt not available')
            if not (state_head is not None):
                raise RuntimeError('state_head not trained')
            state_head = state_head
            state_head.eval(); model.eval()
            rng_enc_th = np.random.RandomState(seed)
            succs_th, ep_lens_th, fracs_th = [], [], []
            vis_enc_th = None
            for trial in range(n_trials):
                x0 = rng_enc_th.uniform(-init_scale, init_scale, 4).astype(np.float32)
                obs, state, _ = env.reset_to_state(x0)
                done = False
                prev_obs_th = None
                states_th, actions_th, all_obs_th = [state.copy()], [], []
                for _ in range(T_rollout):
                    all_obs_th.append(obs.copy())
                    obs_t = _make_obs_t(obs, prev_obs_th)
                    with torch.no_grad():
                        x_hat_full = state_head(model.encoder(obs_t)).cpu().numpy()[0]
                    # Only use visual angle/angular-velocity estimates; zero cart x and ẋ
                    x_hat = np.array([0.0, 0.0, x_hat_full[2], x_hat_full[3]], dtype=np.float32)
                    u = float(np.clip((-K_gt @ x_hat)[0], action_lb, action_ub))
                    actions_th.append([u])
                    prev_obs_th = obs.copy()
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
    if not cem_only: print('\n[control] --- Pure-Latent LQR (Q=I, no state_head) ---')
    if not cem_only:
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
                prev_obs_lat = None
                states_lat, actions_lat, all_obs_lat = [state.copy()], [], []
                for _ in range(T_rollout):
                    all_obs_lat.append(obs.copy())
                    obs_t = _make_obs_t(obs, prev_obs_lat)
                    with torch.no_grad():
                        z_t_lat = model.encoder(obs_t).cpu().numpy()[0]
                    u = float(np.clip(
                        (-K_lat @ (z_t_lat - z_star) + u_ff_lin)[0],
                        action_lb, action_ub))
                    actions_lat.append([u])
                    prev_obs_lat = obs.copy()
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

    # Pre-stabilise A_jac — used by both Linear MPC and CEM-linear
    from control.lqr import pre_stabilize_A
    A_stab, _n_def = pre_stabilize_A(A_jac, gt.unstable_eigenvalues, tol=0.05, target=0.9)

    # ── Linear MPC (from Jacobian) ────────────────────────────────────────────
    if not cem_only: print('\n[control] --- Linear MPC (Jacobian) ---')
    if not cem_only:
        try:
            from control.mpc import LatentMPC
            print(f'[control] A_jac pre-stab: deflated={_n_def}  rho={np.max(np.abs(np.linalg.eigvals(A_stab))):.4f}')

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
                    frame_stack=frame_stack,
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

    # ── CEM setup ─────────────────────────────────────────────────────────────
    cem_cfg      = cfg.get('cem', {})
    cem_horizon  = int(cem_cfg.get('horizon',   mpc_horizon))
    cem_chunk    = int(cem_cfg.get('chunk_size', mpc_chunk))
    cem_n_samp   = int(cem_cfg.get('n_samples',  500))
    cem_n_elite  = int(cem_cfg.get('n_elites',    50))
    cem_n_iter   = int(cem_cfg.get('n_iter',       5))
    cem_std      = float(cem_cfg.get('init_std',  3.0))
    n_trials_cem = int(cem_cfg.get('n_trials',    50))

    # ── CEM sweep: nonlinear predictor dynamics, Q=I, R=0.1·I ─────────────────
    # Warm-starting (σ_warm=0.5) shifts the previous optimal action sequence to the
    # next step's initialisation — prevents the cold-start bang-bang bias where
    # random σ=3 samples produce saturated elites that lock in ±10 actions.
    _R_cem    = 0.1 * np.eye(1)   # scalar action; R=0.1·I matches user choice
    _cem_sweep = [
        dict(tag='H10_QI_nl', horizon=10, Q=np.eye(d), Qf=np.eye(d),
             zs=z_star, std=3.0, ws=0.5, R=_R_cem, ni=20,
             desc='H=10 Q=I nonlinear'),
        dict(tag='H25_QI_nl', horizon=25, Q=np.eye(d), Qf=np.eye(d),
             zs=z_star, std=3.0, ws=0.5, R=_R_cem, ni=20,
             desc='H=25 Q=I nonlinear'),
    ]

    from control.cem import CEMLatentPlanner
    for _sc in _cem_sweep:
        _tag  = _sc['tag'];   _H    = _sc['horizon']
        _Q    = _sc['Q'];     _Qf   = _sc['Qf']
        _desc = _sc['desc'];  _zs   = _sc.get('zs', z_star)
        _std  = _sc.get('std', cem_std)
        _ws   = _sc.get('ws', 0.5)
        _R_sc = _sc.get('R', _R_cem)
        _ni   = _sc.get('ni',  cem_n_iter)

        print(f'\n[control] --- CEM nonlinear  {_desc} ---')
        try:
            _cem_nl = CEMLatentPlanner(
                predictor=model.predictor, action_encoder=model.action_encoder,
                predictor_window=model.config.predictor_window,
                Q=_Q, R=_R_sc, Q_f=_Qf,
                warm_start_sigma=_ws,
                horizon=_H, chunk_size=cem_chunk,
                n_samples=cem_n_samp, n_elites=cem_n_elite,
                n_iter=_ni, init_std=_std,
                action_lb=action_lb, action_ub=action_ub,
                device=device,
            )
            print(f'[control] {_cem_nl.summary()}')
            _cr_cn = evaluate_stabilization_mpc(
                encoder=model.encoder, mpc=_cem_nl, env=env,
                n_trials=n_trials_cem, T=T_rollout, init_scale=init_scale,
                seed=seed, device=device, z_star=_zs, vis_trial=0,
                frame_stack=frame_stack,
            )
            _rk_cn = f'cem_nonlinear_{_tag}'
            ctrl_results[_rk_cn] = {k: v for k, v in _cr_cn.items() if k != 'vis_result'}
            print(f'[control] {_rk_cn}: success={_cr_cn["success_rate"]:.3f}'
                  f'  ep_len={_cr_cn["mean_episode_length"]:.1f}'
                  f'  frac_stable={_cr_cn["mean_fraction_stable"]:.3f}')
            _vis_cn = _cr_cn.get('vis_result')
            if _vis_cn:
                save_rollout_frames(_vis_cn, out_dir / f'{_rk_cn}_frames.png',
                                    n_frames=8, title=f'{exp_name}  CEM-nl {_desc}')
                save_rollout_video(_vis_cn, out_dir / f'{_rk_cn}.gif',
                                   fps=15, title=f'{exp_name}  CEM-nl {_desc}')
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f'[control] CEM-nonlinear {_tag} failed: {exc}')
            ctrl_results[f'cem_nonlinear_{_tag}'] = {'error': str(exc)}

    # ── CEM sweep summary table ───────────────────────────────────────────────
    print('\n' + '═' * 55)
    print('CEM SWEEP SUMMARY  (nonlinear predictor, Q=I)')
    print('═' * 55)
    print(f'  {"Config":<28} {"succ":>7} {"frac_stb":>9} {"ep_len":>8}')
    print('  ' + '-' * 51)
    for _sc in _cem_sweep:
        _t  = _sc['tag']
        _cn = ctrl_results.get(f'cem_nonlinear_{_t}', {})
        _f  = lambda d, k: f'{d[k]:.3f}' if k in d else '  err'
        _fl = lambda d, k: f'{d[k]:.1f}' if k in d else '   err'
        print(f'  {_sc["desc"]:<28} '
              f'{_f(_cn, "success_rate"):>7} '
              f'{_f(_cn, "mean_fraction_stable"):>9} '
              f'{_fl(_cn, "mean_episode_length"):>8}')
    print('═' * 55 + '\n')

    # ── Nonlinear gradient MPC ────────────────────────────────────────────────
    if not cem_only: print('\n[control] --- Nonlinear Gradient MPC ---')
    if not cem_only:
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
                frame_stack=frame_stack,
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
    if not cem_only: print('\n[control] --- Gradient MPC (state_head cost) ---')
    if not cem_only:
        try:
            from control.grad_mpc import GradientLatentMPC
            if not (state_head is not None):
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
                state_head=state_head,
                Q_phys=Q_phys_sh,
            )
            print(f'[control] {grad_mpc_sh.summary()} [state_head cost]')
            n_grad = min(int(probe_cfg.get('n_trials_grad_mpc', 50)), n_trials)
            cr_grad_sh = evaluate_stabilization_mpc(
                encoder=model.encoder, mpc=grad_mpc_sh, env=env,
                n_trials=n_grad, T=T_rollout, init_scale=init_scale,
                seed=seed, device=device, z_star=z_star, vis_trial=0,
                frame_stack=frame_stack,
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

    # Save model weights to stable ckpt_dir; eval artifacts to timestamped out_dir
    torch.save(model.state_dict(), ckpt_dir / 'model_final.pt')
    if state_head is not None:
        torch.save(state_head.state_dict(), ckpt_dir / 'state_head.pt')
    np.save(out_dir / 'A_jac.npy',  A_jac)
    np.save(out_dir / 'B_jac.npy',  B_jac)
    np.save(out_dir / 'z_star.npy', z_star)
    np.save(out_dir / 'z_fp.npy',   z_fp)
    np.save(out_dir / 'Q_lat.npy',  Q_lat_diag)
    print(f'[done] eval artifacts saved to {out_dir}')

    results = {
        'experiment': {'name': exp_name, 'variant': encoder_variant,
                       'dataset': dataset_name, 'seed': seed,
                       'frame_stack': frame_stack},
        'model':   {'n_params': n_params},
        'state_decoder': state_decoder_results,
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
    p.add_argument('--cem-only',  action='store_true',
                   help='Skip all non-CEM controllers during evaluation')
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
        cem_only=args.cem_only,
    )
