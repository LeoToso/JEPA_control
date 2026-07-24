"""Encoder + linear-decoder + ground-truth LQR for CartPole stabilization.

This bypasses the JEPA predictor entirely.  Instead:
  1. Encode each observation → z_t  (JEPA encoder, frozen weights)
  2. Decode z_t → ŝ_t ∈ R^4       (linear probe fit via least squares)
  3. Apply GT-LQR on ŝ_t           (gain K from true CartPole A_star, B_star)

This tests whether the ENCODER carries enough physical information for control,
independent of the predictor's fp_err / divergence underestimation problems.

The linear decoder is fitted on the fly: we run N random rollout episodes,
collect (z, physical_state) pairs, then solve a simple least-squares problem.

Usage:
    python experiments/check_decoder_lqr.py \\
        --checkpoint results/jepa_v5_sigreg_stage2/checkpoints/checkpoint_epoch0010.pt \\
        --config     configs/cartpole_jepa_pred_state_random_stage2.yaml \\
        --n-trials 20
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import scipy.linalg
import torch
import yaml


class DecoderLQRController:
    """Apply GT-LQR on z decoded to physical state.

    Control law: u = -K_gt @ D_weight @ (z - z_star)
    where (z - z_star) is the deviation from the encoded equilibrium.
    Using deviations rather than absolute z eliminates any decoder bias
    at the reference point.

    Parameters
    ----------
    D_weight : (4, d) decoder weight matrix
    z_star   : (d,) encoded equilibrium latent state
    K_gt     : (1, 4) GT-LQR gain
    action_lb/ub : raw action bounds
    """

    def __init__(self, D_weight, z_star, K_gt,
                 action_lb=-10.0, action_ub=10.0):
        self.D_weight  = D_weight         # (4, d)
        self._z_star   = z_star           # (d,)
        self.K_gt      = K_gt             # (1, 4)
        self.action_lb = action_lb
        self.action_ub = action_ub
        # Fake A so rollout_latent_mpc infers W_aug=1 (we don't use augmented state)
        self.A = np.eye(1)
        self._n_sat   = 0
        self._n_steps = 0

    def plan(self, z_t: np.ndarray, z_star: np.ndarray):
        delta_z = z_t - self._z_star               # (d,) deviation from equilibrium
        s_hat   = self.D_weight @ delta_z           # (4,) decoded physical deviation
        u_raw   = float(-(self.K_gt @ s_hat)[0])
        clipped = float(np.clip(u_raw, self.action_lb, self.action_ub))
        self._n_sat   += int(abs(u_raw) >= self.action_ub - 1e-6)
        self._n_steps += 1
        return [np.array([clipped])], []

    def reset(self):
        pass

    @property
    def saturation_fraction(self):
        return self._n_sat / max(1, self._n_steps)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config',     required=True)
    p.add_argument('--n-trials',   type=int,   default=20)
    p.add_argument('--T',          type=int,   default=200)
    p.add_argument('--n-probe-rollouts', type=int, default=200,
                   help='Number of random rollouts to collect (z, state) pairs for decoder')
    p.add_argument('--probe-ep-len', type=int, default=50,
                   help='Steps per probe rollout')
    p.add_argument('--lqr-Q-diag', type=float, nargs=4, default=[1.0, 1.0, 100.0, 10.0],
                   metavar=('Qx','Qxd','Qtheta','Qthetad'),
                   help='LQR state cost diagonal (physical units)')
    p.add_argument('--lqr-R',     type=float, default=0.01,
                   help='LQR action cost')
    p.add_argument('--dataset-dir', default=None,
                   help='Path to HDF5 dataset dir (train.hdf5 / val.hdf5). '
                        'If given, encoder is run on the stored images for a much '
                        'larger and more diverse (z, state) probe set.')
    p.add_argument('--max-probe-eps', type=int, default=50,
                   help='Max episodes to load from HDF5 (batched encoding per episode). '
                        '50 eps × 500 steps = 25k pairs, takes ~30 s on GPU.')
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--out',        default=None)
    p.add_argument('--device',     default=None)
    args = p.parse_args()

    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Device: {device}')

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_cfg   = cfg['environment']
    model_cfg = cfg['model']
    ctrl_cfg  = cfg.get('control', {})

    frame_stack = int(model_cfg.get('frame_stack', 1))
    frame_skip  = int(env_cfg.get('frame_skip', 1))
    action_lb   = float(env_cfg['action_range'][0])
    action_ub   = float(env_cfg['action_range'][1])

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    print(f'[model] Loading {ckpt_path}')
    raw   = torch.load(ckpt_path, map_location=device)
    state = (raw['model_state'] if isinstance(raw, dict) and 'model_state' in raw else
             raw['model_state_dict'] if isinstance(raw, dict) and 'model_state_dict' in raw
             else raw)
    if isinstance(raw, dict) and 'config' in raw:
        _ckpt_cfg = raw['config']
        for _k in ('latent_dim', 'action_latent_dim', 'action_encoder',
                   'encoder_type', 'patch_size', 'frame_stack',
                   'vit_embed_dim', 'vit_depth', 'vit_num_heads',
                   'predictor_type', 'predictor_hidden_dim', 'predictor_n_layers',
                   'predictor_window', 'predictor_embed_dim', 'predictor_depth',
                   'predictor_num_heads', 'predictor_mlp_ratio'):
            if _k in _ckpt_cfg and _ckpt_cfg[_k] != model_cfg.get(_k):
                print(f'[model] arch override: {_k}={_ckpt_cfg[_k]}')
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
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    encoder = model.encoder
    d = model.latent_dim
    print(f'[model] latent_dim={d}')

    # ── Environment ───────────────────────────────────────────────────────────
    from envs.cartpole_visual import ContinuousCartpoleVisual
    env = ContinuousCartpoleVisual(
        frame_skip=frame_skip, image_size=int(env_cfg['image_size']),
        mass_cart=env_cfg['mass_cart'], mass_pole=env_cfg['mass_pole'],
        pole_length=env_cfg['pole_length'], gravity=env_cfg['gravity'],
        dt=env_cfg['dt'], seed=args.seed,
    )

    def _encode(obs_np, prev_obs_np=None):
        curr = torch.from_numpy(obs_np).float().permute(2, 0, 1)[None].to(device) / 255.0
        if frame_stack > 1:
            prev = (curr if prev_obs_np is None else
                    torch.from_numpy(prev_obs_np).float().permute(2, 0, 1)[None].to(device) / 255.0)
            inp = torch.cat([prev, curr], dim=1)
        else:
            inp = curr
        with torch.no_grad():
            return encoder(inp).cpu().numpy()[0]

    # ── Collect (z, physical_state) pairs for decoder fitting ────────────────
    Zs, Ss = [], []

    if args.dataset_dir is not None:
        # Load from HDF5 training data — much larger and more diverse than rollouts.
        # Format: one HDF5 group per episode with datasets 'observations' (T, H, W, C)
        # and 'states' (T, 4).  We try train.hdf5 first, then val.hdf5.
        import h5py
        ds_dir = Path(args.dataset_dir)
        hdf5_paths = [ds_dir / 'train.hdf5', ds_dir / 'val.hdf5',
                      ds_dir / 'train.h5', ds_dir / 'val.h5']
        hdf5_path  = next((p for p in hdf5_paths if p.exists()), None)
        if hdf5_path is None:
            raise FileNotFoundError(f'No train.hdf5/val.hdf5 found in {ds_dir}')
        print(f'[probe] loading from {hdf5_path} (max {args.max_probe_eps} eps, batched) ...')
        with h5py.File(hdf5_path, 'r') as f:
            # Support two layouts:
            #   (A) flat: f['ep_000'] -> group with 'observations', 'states'
            #   (B) nested: f['episodes']['0'] -> group with 'observations', 'states'
            root    = f['episodes'] if 'episodes' in f else f
            ep_keys = sorted(root.keys(), key=lambda x: int(x) if x.isdigit() else x)
            ep_keys = ep_keys[:args.max_probe_eps]
            n_loaded = 0
            for ep_key in ep_keys:
                grp        = root[ep_key]
                obs_all    = grp['observations'][:]   # (T+1, H, W, C) uint8
                states_all = grp['states'][:]         # (T+1, 4)
                T = obs_all.shape[0]
                # Build batch for the whole episode: (T-1, C*frame_stack, H, W)
                obs_t  = torch.from_numpy(obs_all[1:]).float().permute(0, 3, 1, 2).to(device) / 255.0
                obs_tm1 = torch.from_numpy(obs_all[:-1]).float().permute(0, 3, 1, 2).to(device) / 255.0
                if frame_stack > 1:
                    inp_batch = torch.cat([obs_tm1, obs_t], dim=1)   # (T-1, 2C, H, W)
                else:
                    inp_batch = obs_t                                  # (T-1, C, H, W)
                with torch.no_grad():
                    z_batch = encoder(inp_batch).cpu().numpy()        # (T-1, d)
                for t_idx in range(T - 1):
                    Zs.append(z_batch[t_idx])
                    Ss.append(states_all[t_idx + 1].astype(np.float64))
                    n_loaded += 1
        print(f'[probe] loaded {n_loaded} (z, s) pairs from {len(ep_keys)} episodes')
    else:
        print(f'[probe] collecting {args.n_probe_rollouts} rollouts × {args.probe_ep_len} steps ...')
        rng_p = np.random.RandomState(args.seed + 1)
        for ep in range(args.n_probe_rollouts):
            theta0 = rng_p.uniform(-0.8, 0.8)
            x0     = np.array([rng_p.uniform(-0.5, 0.5), rng_p.uniform(-0.2, 0.2),
                                theta0, rng_p.uniform(-0.3, 0.3)], dtype=np.float32)
            obs, state, _ = env.reset_to_state(x0)
            prev_obs = obs.copy()
            for _ in range(args.probe_ep_len):
                z = _encode(obs, prev_obs)
                Zs.append(z.copy())
                Ss.append(state.copy())
                u_raw = rng_p.uniform(action_lb, action_ub)
                obs_next, state_next, _, done, _ = env.step(u_raw)
                prev_obs = obs.copy()
                obs = obs_next
                state = state_next
                if done:
                    break
        print(f'[probe] collected {len(Zs)} (z, s) pairs from {args.n_probe_rollouts} rollouts')

    Z = np.array(Zs, dtype=np.float64)   # (N, d)
    S = np.array(Ss, dtype=np.float64)   # (N, 4)
    N_pairs = len(Z)
    print(f'[probe] collected {N_pairs} (z, s) pairs')

    # Pearson correlations as diagnostic
    C = np.zeros((d, 4))
    for i in range(d):
        for j in range(4):
            zi = Z[:, i] - Z[:, i].mean()
            sj = S[:, j] - S[:, j].mean()
            denom = np.linalg.norm(zi) * np.linalg.norm(sj)
            C[i, j] = np.dot(zi, sj) / (denom + 1e-12)
    print(f'[probe] Pearson max|r|: x={np.max(np.abs(C[:,0])):.3f}  '
          f'xdot={np.max(np.abs(C[:,1])):.3f}  '
          f'theta={np.max(np.abs(C[:,2])):.3f}  '
          f'thetadot={np.max(np.abs(C[:,3])):.3f}')

    # Fit linear decoder via least squares: S ≈ Z @ W.T + b
    # Add bias column: [Z | 1] @ [W; b].T = S
    Z_aug = np.hstack([Z, np.ones((N_pairs, 1))])    # (N, d+1)
    params, residuals, rank, sv = np.linalg.lstsq(Z_aug, S, rcond=None)
    # params shape: (d+1, 4)
    D_weight = params[:d, :].T    # (4, d)
    D_bias   = params[d, :]       # (4,)

    # Evaluate decoder quality: R² per physical dimension
    S_pred = Z @ D_weight.T + D_bias   # (N, 4)
    ss_res = np.sum((S - S_pred) ** 2, axis=0)
    ss_tot = np.sum((S - S.mean(axis=0)) ** 2, axis=0)
    r2     = 1.0 - ss_res / (ss_tot + 1e-12)
    print(f'[probe] decoder R²: x={r2[0]:.3f}  xdot={r2[1]:.3f}  '
          f'theta={r2[2]:.3f}  thetadot={r2[3]:.3f}')

    # Decoder at equilibrium — check that decoded state is near (0,0,0,0)
    obs_eq, _, _ = env.reset_to_state(np.zeros(4, dtype=np.float32))
    z_star_arr   = _encode(obs_eq)
    # Deviation from z_star is 0 at equilibrium, so decoded deviation should be 0.
    # But verify the absolute decoded state to diagnose any calibration offset.
    s_eq_decoded = D_weight @ z_star_arr + D_bias
    print(f'[probe] decoded state at z* (should be ~0): '
          f'x={s_eq_decoded[0]:.3f}  xdot={s_eq_decoded[1]:.3f}  '
          f'theta={s_eq_decoded[2]:.3f} rad  thetadot={s_eq_decoded[3]:.3f}')

    # ── Ground-truth LQR gain ─────────────────────────────────────────────────
    from ground_truth.cartpole_gt import CartpoleGroundTruth
    gt = CartpoleGroundTruth(
        mass_cart=env_cfg['mass_cart'], mass_pole=env_cfg['mass_pole'],
        pole_length=env_cfg['pole_length'], gravity=env_cfg['gravity'],
        dt=env_cfg['dt'] * frame_skip,
    )
    Q_gt = np.diag(args.lqr_Q_diag)
    R_gt = args.lqr_R * np.eye(1)
    P_gt = scipy.linalg.solve_discrete_are(gt.A_star, gt.B_star, Q_gt, R_gt)
    K_gt = np.linalg.solve(R_gt + gt.B_star.T @ P_gt @ gt.B_star,
                            gt.B_star.T @ P_gt @ gt.A_star)   # (1, 4)
    A_cl_gt = gt.A_star - gt.B_star @ K_gt
    rho_cl_gt = float(np.max(np.abs(np.linalg.eigvals(A_cl_gt))))
    print(f'[GT-LQR] K={np.round(K_gt[0], 4)}  ρ(A_cl_GT)={rho_cl_gt:.4f}')

    # The decoder maps z - z* → physical state deviation.
    # Control: u = -K_gt @ D_weight @ (z - z*).
    # Effective gain in latent space: K_eff = K_gt @ D_weight  (1, d).
    K_eff_latent = K_gt @ D_weight   # (1, d)
    print(f'[GT-LQR] ||K_eff_latent||={np.linalg.norm(K_eff_latent):.4f}  '
          f'max_u at ||Δz||=0.1: {float(np.linalg.norm(K_eff_latent))*0.1:.3f} N')

    # ── Build controller ──────────────────────────────────────────────────────
    ctrl = DecoderLQRController(
        D_weight=D_weight, z_star=z_star_arr,
        K_gt=K_gt,
        action_lb=action_lb, action_ub=action_ub,
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    init_scale = float(ctrl_cfg.get('init_scale', 0.05))
    stab_thr   = float(ctrl_cfg.get('stabilization_threshold', 0.1))
    sett_thr   = float(ctrl_cfg.get('settling_threshold', 0.05))

    print(f'\n[eval] Decoder-LQR  Q_diag={args.lqr_Q_diag}  R={args.lqr_R}  '
          f'n_trials={args.n_trials}  T={args.T}')

    # rollout_latent_mpc uses mpc.A.shape[0] to infer d_mpc.
    # Our fake A=eye(1) signals d_mpc=1 → W_aug=1 → s_t = z_t (no augmentation).
    # plan() decodes z_t itself, so the z_star arg from rollout is irrelevant.
    from control.rollout import evaluate_stabilization_mpc
    cr = evaluate_stabilization_mpc(
        encoder=encoder, mpc=ctrl, env=env,
        n_trials=args.n_trials, T=args.T,
        init_scale=init_scale,
        stabilization_threshold=stab_thr,
        settling_threshold=sett_thr,
        seed=args.seed, device=device,
        z_star=np.zeros(1),   # dummy — not used by our plan()
        vis_trial=0,
        frame_stack=frame_stack,
    )

    print(f'\n[result] success_rate          = {cr["success_rate"]:.3f}')
    print(f'[result] mean_ep_length        = {cr["mean_episode_length"]:.1f}')
    print(f'[result] mean(1/ep_length²)    = {cr["mean_inv_sq_ep_length"]:.6f}')
    print(f'[result] mean_frac_stable      = {cr["mean_fraction_stable"]:.3f}')
    if 'mean_cost' in cr:
        print(f'[result] mean_cost             = {cr["mean_cost"]:.2f}')
    print(f'[result] action_saturation     = {ctrl.saturation_fraction:.3f}')

    vis = cr.get('vis_result')
    if vis is not None and args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        from control.visualize import save_rollout_frames
        title = (f'Decoder-LQR R²_θ={r2[2]:.2f}  '
                 f'success={cr["success_rate"]:.2f}  '
                 f'frac={cr["mean_fraction_stable"]:.2f}')
        from control.visualize import save_rollout_frames
        save_rollout_frames(vis, out_path, n_frames=8, title=title)
        print(f'[saved] frames -> {out_path}')


if __name__ == '__main__':
    main()
