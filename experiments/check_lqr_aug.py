"""LQR control in the augmented (W*d)-dimensional latent space.

Solves discrete-time LQR via DARE on the augmented Jacobian (A_aug, B_aug)
and evaluates pole stabilization on the real CartPole environment.

The augmented state s_t = [z_t, z_{t-1}, ..., z_{t-W+1}] ∈ R^{W*d} captures
the predictor's full memory window.  rho(A_aug) > 1 while rho(A_partial) < 1,
so CEM in 8D misses the unstable mode; this script operates in the full 40D
space where DARE can see (and stabilize) it.

Usage:
    python experiments/check_lqr_aug.py \\
        --checkpoint results/jepa_v5_sigreg_stage2/checkpoints/checkpoint_epoch0100.pt \\
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


class LQRLatentController:
    """LQR gain applied in the augmented (W*d) latent space.

    Parameters
    ----------
    A       : (Wd, Wd) augmented Jacobian — stored so rollout_latent_mpc
              can infer W_aug = Wd / d_lat automatically.
    K       : (1, Wd) LQR gain matrix (u_norm = -K @ (s - s*)).
    action_lb/ub : raw action bounds for clipping.
    action_scale : model action scale; u_raw = u_norm * action_scale.
    """

    def __init__(self, A, K, action_lb=-10.0, action_ub=10.0, action_scale=1.0):
        self.A            = A
        self.K            = K
        self.action_lb    = action_lb
        self.action_ub    = action_ub
        self._action_scale = float(action_scale)

    def plan(self, s_t: np.ndarray, z_star: np.ndarray):
        err    = s_t - z_star
        u_norm = float(-(self.K @ err)[0])
        u_raw  = float(np.clip(u_norm * self._action_scale,
                               self.action_lb, self.action_ub))
        return [np.array([u_raw])], []

    def reset(self):
        pass


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--config',     required=True)
    p.add_argument('--n-trials',   type=int,   default=20)
    p.add_argument('--T',          type=int,   default=200)
    p.add_argument('--alpha',      type=float, default=1.0,
                   help='Q = alpha * I  (state cost scale)')
    p.add_argument('--beta',       type=float, default=0.01,
                   help='R = beta * I  (action cost scale)')
    p.add_argument('--hist-alpha', type=float, default=0.1,
                   help='Cost weight on history blocks of Q (relative to alpha)')
    p.add_argument('--pre-stabilize', action='store_true',
                   help='Deflate phantom unstable modes before DARE')
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

    train_cfg        = cfg.get('training', {})
    normalize_actions = bool(train_cfg.get('normalize_actions', False))
    action_scale     = max(abs(action_lb), abs(action_ub)) if normalize_actions else 1.0
    print(f'[control] normalize_actions={normalize_actions}  action_scale={action_scale}')

    from ground_truth.cartpole_gt import CartpoleGroundTruth
    gt = CartpoleGroundTruth(
        mass_cart=env_cfg['mass_cart'], mass_pole=env_cfg['mass_pole'],
        pole_length=env_cfg['pole_length'], gravity=env_cfg['gravity'],
        dt=env_cfg['dt'] * frame_skip,
    )
    print(f'[GT] unstable eigenvalues: {np.round(gt.unstable_eigenvalues, 4)}')

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    print(f'[model] Loading {ckpt_path}')
    raw = torch.load(ckpt_path, map_location=device)
    if isinstance(raw, dict) and 'model_state' in raw:
        state = raw['model_state']
    elif isinstance(raw, dict) and 'model_state_dict' in raw:
        state = raw['model_state_dict']
    else:
        state = raw
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

    # ── Environment + equilibrium ─────────────────────────────────────────────
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
    obs_eq_t     = _make_obs_t(obs_eq)
    with torch.no_grad():
        z_star = model.encoder(obs_eq_t).cpu().numpy()[0]
    d = len(z_star)
    print(f'[control] d={d}  ||z*||={np.linalg.norm(z_star):.3f}')

    # ── Augmented Jacobian ────────────────────────────────────────────────────
    from control.jacobian import compute_augmented_jacobian_np
    W      = getattr(model.config, 'predictor_window', 1)
    A_aug, B_aug = compute_augmented_jacobian_np(model, z_star, device)
    rho_aug = float(np.max(np.abs(np.linalg.eigvals(A_aug))))
    print(f'[control] W={W}  augmented state dim={W*d}')
    print(f'[control] ρ(A_aug,{W*d}×{W*d})={rho_aug:.4f}  ||B_aug||={np.linalg.norm(B_aug):.4f}')

    # Fixed-point error
    with torch.no_grad():
        z_star_t  = torch.tensor(z_star, dtype=torch.float32, device=device).unsqueeze(0)
        z_win_eq  = z_star_t.unsqueeze(1).expand(1, W, -1)
        u_win_eq  = torch.zeros(1, W, 1, device=device)
        z_pred_eq = model.predict(z_win_eq, u_win_eq)
        fp_err    = float(torch.norm(z_pred_eq - z_star_t).item())
    print(f'[control] fp_err={fp_err:.4f}')

    # Project B through action encoder (scalar → latent action)
    if hasattr(model.action_encoder, 'W'):
        W_enc = model.action_encoder.W.weight.detach().cpu().numpy()  # (m, 1)
        B_aug = B_aug @ W_enc   # (W*d, 1)
        print(f'[control] projected B_aug through action encoder: {B_aug.shape}')
    print(f'[control] ||B_aug_projected||={np.linalg.norm(B_aug):.4f}')

    # ── Build Q: current z_t block weighted α, history blocks weighted α*hist_alpha ──
    Wd = W * d
    Q  = np.kron(np.diag([1.0] + [args.hist_alpha] * (W - 1)), np.eye(d)) * args.alpha
    R  = args.beta * np.eye(1)

    # ── Optional pre-stabilization ────────────────────────────────────────────
    A_dare = A_aug
    if args.pre_stabilize:
        from control.lqr import pre_stabilize_A
        # Use a wider tol so the learned unstable mode (~1.06) is kept
        A_dare, n_def = pre_stabilize_A(A_aug, gt.unstable_eigenvalues, tol=0.3, target=0.9)
        rho_stab = float(np.max(np.abs(np.linalg.eigvals(A_dare))))
        print(f'[lqr] pre-stab: deflated {n_def} phantom mode(s)  ρ(A_dare)={rho_stab:.4f}')
    else:
        print(f'[lqr] using raw A_aug for DARE  ρ={rho_aug:.4f}')

    # ── Solve DARE ────────────────────────────────────────────────────────────
    print('[lqr] solving DARE ...')
    try:
        P = scipy.linalg.solve_discrete_are(A_dare, B_aug, Q, R)
        print('[lqr] DARE converged (scipy)')
    except Exception as e:
        print(f'[lqr] scipy DARE failed ({e}), falling back to iteration ...')
        from control.lqr import _dare_iteration
        P = _dare_iteration(A_dare, B_aug, Q, R)
        print('[lqr] DARE iteration done')

    K = np.linalg.solve(R + B_aug.T @ P @ B_aug, B_aug.T @ P @ A_dare)  # (1, Wd)
    A_cl = A_aug - B_aug @ K   # closed-loop on ORIGINAL A_aug
    cl_eigs = np.abs(scipy.linalg.eigvals(A_cl))
    rho_cl  = float(np.max(cl_eigs))
    print(f'[lqr] K shape: {K.shape}  ||K||={np.linalg.norm(K):.4f}')
    print(f'[lqr] closed-loop ρ(A_cl)={rho_cl:.4f}  '
          f'(n_unstable_cl={int(np.sum(cl_eigs > 1.0))})')

    # ── Controller + rollout ──────────────────────────────────────────────────
    z_star_aug = np.tile(z_star, W)   # (W*d,) reference in augmented space

    lqr = LQRLatentController(
        A=A_aug, K=K,
        action_lb=action_lb, action_ub=action_ub,
        action_scale=action_scale,
    )

    from control.rollout import evaluate_stabilization_mpc
    init_scale = float(ctrl_cfg.get('init_scale', 0.05))
    stab_thr   = float(ctrl_cfg.get('stabilization_threshold', 0.1))
    sett_thr   = float(ctrl_cfg.get('settling_threshold', 0.05))

    print(f'\n[eval] LQR-aug  α={args.alpha}  β={args.beta}  hist_α={args.hist_alpha}  '
          f'n_trials={args.n_trials}  T={args.T}')

    cr = evaluate_stabilization_mpc(
        encoder=model.encoder, mpc=lqr, env=env,
        n_trials=args.n_trials, T=args.T,
        init_scale=init_scale,
        stabilization_threshold=stab_thr,
        settling_threshold=sett_thr,
        seed=args.seed, device=device, z_star=z_star_aug,
        vis_trial=0,
        frame_stack=frame_stack,
    )

    print(f'\n[result] success_rate          = {cr["success_rate"]:.3f}')
    print(f'[result] mean_ep_length        = {cr["mean_episode_length"]:.1f}')
    print(f'[result] mean(1/ep_length²)    = {cr["mean_inv_sq_ep_length"]:.6f}')
    print(f'[result] mean_frac_stable      = {cr["mean_fraction_stable"]:.3f}')
    if 'mean_cost' in cr:
        print(f'[result] mean_cost             = {cr["mean_cost"]:.2f}')

    vis = cr.get('vis_result')
    if vis is not None and args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        from control.visualize import save_rollout_frames
        title = (f'LQR-aug α={args.alpha} β={args.beta}  '
                 f'success={cr["success_rate"]:.2f}  '
                 f'frac={cr["mean_fraction_stable"]:.2f}')
        save_rollout_frames(vis, out_path, n_frames=8, title=title)
        print(f'[saved] frames -> {out_path}')


if __name__ == '__main__':
    main()
