"""LQR control in the augmented (W*d)-dimensional latent space.

Solves discrete-time LQR via DARE on the augmented Jacobian (A_aug, B_aug)
and evaluates pole stabilization on the real CartPole environment.

The augmented state s_t = [z_t, z_{t-1}, ..., z_{t-W+1}] ∈ R^{W*d} captures
the predictor's full memory window.  rho(A_aug) > 1 while rho(A_partial) < 1,
so CEM in 8D misses the unstable mode; this script operates in the full 40D
space where DARE can see (and stabilize) it.

Key diagnostics printed:
  - ρ(A_cl): closed-loop spectral radius (< 1 = LQR succeeded linearly)
  - z_ss (affine fixed point): the physical state the closed-loop actually
    converges to due to the fp_err offset.  If |theta_ss| is large the approach
    is fundamentally broken; use --feedforward to cancel the drift.
  - saturation fraction: fraction of steps where action is clipped.

Usage:
    python experiments/check_lqr_aug.py \\
        --checkpoint results/jepa_v5_sigreg_stage2/checkpoints/checkpoint_epoch0100.pt \\
        --config     configs/cartpole_jepa_pred_state_random_stage2.yaml \\
        --n-trials 20 [--beta 0.01] [--feedforward] [--use-affine-ref]
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
    A          : (Wd, Wd) augmented Jacobian (so rollout can infer W_aug).
    K          : (1, Wd) LQR gain; u_norm = -K @ (s - ref).
    ref        : (Wd,) reference state (z*_aug or affine fixed point).
    u_ff       : scalar feedforward action (cancels B-projected affine drift).
    action_lb/ub : raw action bounds for clipping.
    action_scale : u_raw = u_norm * action_scale.
    """

    def __init__(self, A, K, ref, u_ff=0.0,
                 action_lb=-10.0, action_ub=10.0, action_scale=1.0):
        self.A             = A
        self.K             = K
        self._ref          = ref
        self._u_ff         = float(u_ff)
        self.action_lb     = action_lb
        self.action_ub     = action_ub
        self._action_scale = float(action_scale)
        self._n_sat        = 0
        self._n_steps      = 0

    def plan(self, s_t: np.ndarray, z_star: np.ndarray):
        # z_star arg from rollout is the tiled z*; we use our own ref instead
        err    = s_t - self._ref
        u_norm = float(-(self.K @ err)[0]) + self._u_ff
        u_raw  = u_norm * self._action_scale
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
    p.add_argument('--alpha',      type=float, default=1.0,
                   help='Q = alpha * diag(1, hist_alpha, ...) ⊗ I_d')
    p.add_argument('--beta',       type=float, default=0.01,
                   help='R = beta * I  (action cost; larger → smaller K → less saturation)')
    p.add_argument('--hist-alpha', type=float, default=0.1,
                   help='Q weight on history blocks relative to current z_t')
    p.add_argument('--pre-stabilize', action='store_true',
                   help='Deflate phantom unstable modes before DARE (uses tol=0.3)')
    p.add_argument('--feedforward', action='store_true',
                   help='Add constant feedforward u_ff = -B_aug^+ c_aug to cancel affine drift')
    p.add_argument('--use-affine-ref', action='store_true',
                   help='Use z_ss=(I-A_cl)^{-1}c_aug as reference instead of z*')
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

    train_cfg         = cfg.get('training', {})
    normalize_actions = bool(train_cfg.get('normalize_actions', False))
    action_scale      = max(abs(action_lb), abs(action_ub)) if normalize_actions else 1.0
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

    # Try to load state probe from checkpoint (trained with lambda_state)
    state_probe_W = None
    if isinstance(raw, dict):
        # Common key names for the state probe weight matrix
        for probe_key in ('state_probe.weight', 'state_probe', 'probe_weight'):
            if probe_key in raw:
                w = raw[probe_key]
                state_probe_W = (w.cpu().numpy() if isinstance(w, torch.Tensor) else w)
                print(f'[probe] loaded state_probe from key "{probe_key}" '
                      f'shape={state_probe_W.shape}')
                break
        if state_probe_W is None:
            # Scan model_state for probe key
            for k, v in state.items():
                if 'state_probe' in k and 'weight' in k:
                    state_probe_W = v.cpu().numpy()
                    print(f'[probe] loaded state_probe from model_state["{k}"] '
                          f'shape={state_probe_W.shape}')
                    break

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
    Wd = W * d
    print(f'[control] W={W}  augmented state dim={Wd}')
    print(f'[control] ρ(A_aug,{Wd}×{Wd})={rho_aug:.4f}  ||B_aug||={np.linalg.norm(B_aug):.4f}')

    # Affine drift c_aug at equilibrium (predictor(z*,0) - z*)
    with torch.no_grad():
        z_star_t  = torch.tensor(z_star, dtype=torch.float32, device=device).unsqueeze(0)
        z_win_eq  = z_star_t.unsqueeze(1).expand(1, W, -1)
        u_win_eq  = torch.zeros(1, W, 1, device=device)
        z_pred_eq = model.predict(z_win_eq, u_win_eq)
        fp_err    = float(torch.norm(z_pred_eq - z_star_t).item())
        c_partial = (z_pred_eq - z_star_t).cpu().numpy()[0]   # (d,)
    c_aug = np.concatenate([c_partial, np.zeros((W - 1) * d)])   # (Wd,)
    print(f'[control] fp_err={fp_err:.4f}  ||c_aug[0:d]||={np.linalg.norm(c_partial):.4f}')

    # Project B through action encoder (scalar → latent action → 40D)
    if hasattr(model.action_encoder, 'W'):
        W_enc = model.action_encoder.W.weight.detach().cpu().numpy()   # (m, 1)
        B_aug = B_aug @ W_enc   # (Wd, 1)
    print(f'[control] ||B_aug_projected||={np.linalg.norm(B_aug):.4f}')

    # ── Build Q ───────────────────────────────────────────────────────────────
    Q = np.kron(np.diag([1.0] + [args.hist_alpha] * (W - 1)), np.eye(d)) * args.alpha
    R = args.beta * np.eye(1)

    # ── Optional pre-stabilization ────────────────────────────────────────────
    A_dare = A_aug
    if args.pre_stabilize:
        from control.lqr import pre_stabilize_A
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

    K   = np.linalg.solve(R + B_aug.T @ P @ B_aug, B_aug.T @ P @ A_dare)   # (1, Wd)
    A_cl = A_aug - B_aug @ K
    cl_eigs  = np.abs(scipy.linalg.eigvals(A_cl))
    rho_cl   = float(np.max(cl_eigs))
    n_unst   = int(np.sum(cl_eigs > 1.0))
    print(f'[lqr] K shape={K.shape}  ||K||={np.linalg.norm(K):.4f}')
    print(f'[lqr] closed-loop ρ(A_cl)={rho_cl:.4f}  n_unstable_cl={n_unst}')

    # ── Affine fixed-point diagnostic ─────────────────────────────────────────
    # The affine CL system has a steady state z_ss = (I - A_cl)^{-1} c_aug.
    # If z_ss corresponds to a large |theta|, the controller is driving to the wrong place.
    try:
        I_minus_Acl = np.eye(Wd) - A_cl
        z_ss_delta  = np.linalg.solve(I_minus_Acl, c_aug)   # offset from z*_aug
        z_ss        = np.tile(z_star, W) + z_ss_delta        # (Wd,) actual steady state
        print(f'[affine] ||z_ss_delta||={np.linalg.norm(z_ss_delta):.4f}  '
              f'(affine offset from z*)')
        if state_probe_W is not None:
            # state_probe_W: (4, d) or (d, 4) — figure out from shape
            Dw = state_probe_W
            if Dw.shape[0] == d:
                Dw = Dw.T   # now (4, d)
            phys_ss = Dw @ z_ss[:d]   # physical state at z_ss
            print(f'[affine] physical state at z_ss: '
                  f'x={phys_ss[0]:.3f}  xdot={phys_ss[1]:.3f}  '
                  f'theta={phys_ss[2]:.3f} rad  thetadot={phys_ss[3]:.3f}')
    except np.linalg.LinAlgError:
        print('[affine] (I - A_cl) singular — cannot compute z_ss')
        z_ss = np.tile(z_star, W)

    # ── Feedforward: cancel B-projected component of c_aug ────────────────────
    u_ff = 0.0
    if args.feedforward:
        # Minimum-norm u_ff such that B_aug @ u_ff ≈ -c_aug (least squares)
        # B_aug is (Wd, 1): u_ff = -B^T c / ||B||^2
        u_ff = float(-B_aug.T @ c_aug / (np.dot(B_aug.ravel(), B_aug.ravel()) + 1e-12))
        proj = float(np.dot(B_aug.ravel(), c_aug) / (np.linalg.norm(B_aug) + 1e-12))
        print(f'[ff] u_ff_norm={u_ff:.4f}  u_ff_raw={u_ff*action_scale:.4f}  '
              f'B·c projection={proj:.4f}')

    # ── Reference selection ───────────────────────────────────────────────────
    z_star_aug = np.tile(z_star, W)
    ref = z_ss if args.use_affine_ref else z_star_aug
    ref_label = 'z_ss (affine fp)' if args.use_affine_ref else 'z* (upright)'
    print(f'[lqr] reference = {ref_label}')

    # ── Controller + rollout ──────────────────────────────────────────────────
    lqr = LQRLatentController(
        A=A_aug, K=K, ref=ref, u_ff=u_ff,
        action_lb=action_lb, action_ub=action_ub,
        action_scale=action_scale,
    )

    from control.rollout import evaluate_stabilization_mpc
    init_scale = float(ctrl_cfg.get('init_scale', 0.05))
    stab_thr   = float(ctrl_cfg.get('stabilization_threshold', 0.1))
    sett_thr   = float(ctrl_cfg.get('settling_threshold', 0.05))

    ff_label = f'+ff({u_ff*action_scale:.2f}N)' if args.feedforward else ''
    print(f'\n[eval] LQR-aug  α={args.alpha}  β={args.beta}  hist_α={args.hist_alpha}  '
          f'{ff_label}  ref={ref_label}  n_trials={args.n_trials}  T={args.T}')

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
    print(f'[result] action_saturation     = {lqr.saturation_fraction:.3f}  '
          f'(fraction of steps at ±{action_ub})')

    vis = cr.get('vis_result')
    if vis is not None and args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        from control.visualize import save_rollout_frames
        title = (f'LQR-aug β={args.beta} {ff_label}  '
                 f'success={cr["success_rate"]:.2f}  '
                 f'frac={cr["mean_fraction_stable"]:.2f}')
        save_rollout_frames(vis, out_path, n_frames=8, title=title)
        print(f'[saved] frames -> {out_path}')


if __name__ == '__main__':
    main()
