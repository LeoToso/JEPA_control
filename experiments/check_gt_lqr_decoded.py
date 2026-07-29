"""GT LQR control using the encoder + state_head decoder as a state estimator.

The pipeline is:
  obs → encoder(obs) → z  → state_head(z) → decoded_state
  decoded_state → u = -K @ decoded_state   (GT LQR gain)

This tests whether the encoder representation is rich enough to support control,
bypassing the predictor entirely.  If this works, the encoder is fine but the
predictor doesn't couple action to the unstable mode (uncontrollable B).
If this also fails, the encoder itself is losing information needed for control.

Usage:
    python experiments/check_gt_lqr_decoded.py \\
        --checkpoint   results/jepa_sf_w3_fs5_v9_phase2/checkpoints/checkpoint_epoch0090.pt \\
        --config       configs/cartpole_jepa_sf_w3_fs5_v9_phase2.yaml \\
        --state-head-ckpt results/jepa_sf_w3_fs5_v9/checkpoints/checkpoint_epoch0100.pt \\
        --data         data/cartpole_visual_fs5_passive_long \\
        --n-trials 30 --T 200 --device cuda:2
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import yaml


def _load_state_head(ckpt_path, d_lat, device):
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and 'state_head_state' in raw:
        sh = nn.Linear(d_lat, 4).to(device)
        sh.load_state_dict(raw['state_head_state'])
        sh.eval()
        return sh
    raise RuntimeError(f'No state_head_state found in {ckpt_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',      required=True)
    p.add_argument('--config',          required=True)
    p.add_argument('--state-head-ckpt', default=None,
                   help='Checkpoint with state_head_state (if not in main ckpt)')
    p.add_argument('--data',            default=None,
                   help='Dataset dir for state normalisation stats')
    p.add_argument('--n-trials',        type=int,   default=30)
    p.add_argument('--T',               type=int,   default=200)
    p.add_argument('--beta',            type=float, default=0.01,
                   help='LQR action cost R = beta * I')
    p.add_argument('--q-phys',          type=float, nargs=4,
                   default=[1.0, 1.0, 100.0, 10.0],
                   metavar=('Q_X', 'Q_XDOT', 'Q_THETA', 'Q_THETADOT'))
    p.add_argument('--seed',            type=int,   default=42)
    p.add_argument('--device',          default=None)
    args = p.parse_args()

    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Device: {device}')

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_cfg   = cfg['environment']
    model_cfg = cfg['model']

    frame_stack = int(model_cfg.get('frame_stack', 1))
    frame_skip  = int(env_cfg.get('frame_skip', 1))
    action_lb   = float(env_cfg['action_range'][0])
    action_ub   = float(env_cfg['action_range'][1])

    # Load model (encoder only is used for control; predictor not needed)
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
    d = int(model_cfg['latent_dim'])

    # State head — try main checkpoint first, then --state-head-ckpt
    state_head = None
    if isinstance(raw, dict) and 'state_head_state' in raw:
        state_head = nn.Linear(d, 4).to(device)
        state_head.load_state_dict(raw['state_head_state'])
        state_head.eval()
        print(f'[state_head] loaded from main checkpoint')
    if state_head is None:
        if args.state_head_ckpt is None:
            raise RuntimeError('No state_head in checkpoint; provide --state-head-ckpt')
        state_head = _load_state_head(args.state_head_ckpt, d, device)
        print(f'[state_head] loaded from {args.state_head_ckpt}')

    # State normalisation stats (un-normalise state_head output → raw physical state)
    state_mean = state_std = None
    if args.data is not None:
        from experiments.diagnose_instability_rollout import _load_norm_stats
        state_mean, state_std = _load_norm_stats(args.data)
        print(f'[data] state_mean={np.round(state_mean, 4)}  state_std={np.round(state_std, 4)}')

    def decode_state(z_np):
        z_t = torch.tensor(z_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            s = state_head(z_t).cpu().numpy()[0]
        if state_mean is not None:
            s = s * state_std + state_mean
        return s

    # Ground truth LQR (discrete, frame_skip-step)
    from ground_truth.cartpole_gt import CartpoleGroundTruth
    from control.lqr import solve_discrete_lqr
    gt = CartpoleGroundTruth(
        mass_cart=env_cfg['mass_cart'], mass_pole=env_cfg['mass_pole'],
        pole_length=env_cfg['pole_length'], gravity=env_cfg['gravity'],
        dt=env_cfg['dt'] * frame_skip,
    )
    Q_phys = np.diag(args.q_phys)
    R_lqr  = args.beta * np.eye(1)
    K, _, cl_eigs = solve_discrete_lqr(gt.A_star, gt.B_star, Q_phys, R_lqr)
    rho_cl = float(np.max(np.abs(cl_eigs)))
    print(f'[GT LQR] Q_phys={args.q_phys}  R={args.beta}')
    print(f'[GT LQR] K={np.round(K, 4)}')
    print(f'[GT LQR] ρ_cl={rho_cl:.4f}  (< 1 = GT is stabilizable)')

    # Controllability diagnostic: how well does state_head recover theta?
    Ws = state_head.weight.detach().cpu().numpy()   # (4, d)
    print(f'[state_head] ||Ws_theta_row||={np.linalg.norm(Ws[2]):.4f}  '
          f'(larger = theta better encoded in z)')

    # Environment
    from envs.cartpole_visual import ContinuousCartpoleVisual
    env = ContinuousCartpoleVisual(
        frame_skip=frame_skip, image_size=int(env_cfg['image_size']),
        mass_cart=env_cfg['mass_cart'], mass_pole=env_cfg['mass_pole'],
        pole_length=env_cfg['pole_length'], gravity=env_cfg['gravity'],
        dt=env_cfg['dt'], seed=args.seed,
    )

    ctrl_cfg = cfg.get('control', {})
    init_scale = float(ctrl_cfg.get('init_scale', 0.05))
    stab_thr   = float(ctrl_cfg.get('stabilization_threshold', 0.1))
    sett_thr   = float(ctrl_cfg.get('settling_threshold', 0.05))

    rng = np.random.RandomState(args.seed)
    x_star = np.zeros(4)
    successes, ep_lengths, frac_stables, decoding_errors = [], [], [], []

    print(f'\n[eval] GT-LQR via decoded state  n_trials={args.n_trials}  T={args.T}')

    for trial in range(args.n_trials):
        x0 = rng.uniform(-init_scale, init_scale, 4).astype(np.float32)
        obs, state, _ = env.reset_to_state(x0)
        prev_obs = obs.copy()

        settling_time = args.T
        done_at = args.T
        states_hist, frac_stable_steps = [], 0
        decode_errs = []

        for t in range(args.T):
            states_hist.append(state.copy())

            # Encode → decode → LQR
            obs_t = torch.from_numpy(obs).float().permute(2, 0, 1)[None].to(device) / 255.0
            if frame_stack > 1:
                prev_t = torch.from_numpy(prev_obs).float().permute(2, 0, 1)[None].to(device) / 255.0
                enc_input = torch.cat([prev_t, obs_t], dim=1)
            else:
                enc_input = obs_t
            with torch.no_grad():
                z = model.encoder(enc_input).cpu().numpy()[0]

            s_decoded = decode_state(z)
            decode_errs.append(np.linalg.norm(s_decoded - state))

            u_norm = float(-(K @ s_decoded)[0])
            u_raw  = float(np.clip(u_norm, action_lb, action_ub))

            if np.linalg.norm(state - x_star) < sett_thr and settling_time == args.T:
                settling_time = t

            prev_obs = obs.copy()
            obs, state, _, done, _ = env.step(u_raw)
            if np.linalg.norm(state - x_star) < sett_thr:
                frac_stable_steps += 1

            if done:
                done_at = t + 1
                break

        stabilized = bool(np.linalg.norm(state - x_star) < stab_thr)
        successes.append(stabilized)
        ep_lengths.append(done_at)
        frac_stables.append(frac_stable_steps / done_at)
        decoding_errors.append(float(np.mean(decode_errs)))

    env.close()

    ep_arr = np.array(ep_lengths, dtype=float)
    print(f'\n[result] success_rate          = {np.mean(successes):.3f}')
    print(f'[result] mean_ep_length        = {np.mean(ep_arr):.1f}')
    print(f'[result] mean_frac_stable      = {np.mean(frac_stables):.3f}')
    print(f'[result] mean_decode_error     = {np.mean(decoding_errors):.4f}  '
          f'(mean ||s_decoded - s_real||)')
    print(f'[result] mean(1/ep_length²)    = {np.mean(1/np.maximum(ep_arr,1)**2):.6f}')

    # Also run GT LQR on REAL state as upper-bound reference
    print('\n[baseline] Running GT LQR on REAL physical state (upper bound)...')
    rng2 = np.random.RandomState(args.seed)
    succ_real, ep_real = [], []
    for trial in range(args.n_trials):
        x0 = rng2.uniform(-init_scale, init_scale, 4).astype(np.float32)
        obs, state, _ = env.reset_to_state(x0)
        done_at = args.T
        for t in range(args.T):
            u_raw = float(np.clip(float(-(K @ state)[0]), action_lb, action_ub))
            obs, state, _, done, _ = env.step(u_raw)
            if done:
                done_at = t + 1
                break
        succ_real.append(bool(np.linalg.norm(state - x_star) < stab_thr))
        ep_real.append(done_at)
    env.close()
    print(f'[baseline] success_rate={np.mean(succ_real):.3f}  '
          f'mean_ep_length={np.mean(ep_real):.1f}')


if __name__ == '__main__':
    main()
