#!/usr/bin/env python
"""Run SAC policy in the real Walker2d-v4 env via a latent probe bridge.

Instead of CEM planning, we close the loop purely through the visual encoder:
    real frame → encoder → z → probe → obs_hat → SAC.act → action → real env

This avoids the OOD predictor-z problem that plagues latent iCEM: the probe is
only ever applied to encoder-output z values (R²≈0.98), never to multi-step
predictor rollout z values (which are OOD for the encoder-trained probe).

Usage
-----
MUJOCO_GL=egl python experiments/eval_latent_sac_walker.py \\
    --ckpt  /mnt/t7shield/jepa_results/walker2d_mixed_sac_smwm_fwd_endpoint_inverse_act1_seed42/model_final.pt \\
    --cfg   configs/walker2d_smwm_fwd_endpoint_inverse_act1.yaml \\
    --hdf5-dir data/walker2d_mixed_sac_fs5_64 \\
    --probe-path results/probes/fwd_ep_ar_mlp_probe.pt \\
    --sac-repo sdpkjc/Walker2d-v4-sac_continuous_action-seed4 \\
    --trials 5 --n-steps 600
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch


def run_trial(bundle, probe_net, sac_policy, eval_env, visual_env,
              seed: int, n_steps: int,
              real_height: bool = False,
              real_angle: bool = False,
              real_obs: bool = False):
    """Close loop: real frame → encoder → probe → SAC → real env.

    real_height / real_angle: replace probe output dims 0/1 with the
    corresponding real gymnasium obs value for an ablation study.
    real_obs: bypass probe entirely and use the full real gymnasium obs.
    """
    from experiments.sensorimotor_probe_utils import encode_obs, make_frame_buffer, push_frame

    obs_gym, _ = eval_env.reset(seed=seed)
    frame, state, _ = visual_env.reset(seed=seed)
    prev_frame = frame.copy()

    _fs = int(bundle['model'].frame_stack)
    if _fs > 1:
        frame_buf = make_frame_buffer(bundle, frame)
    else:
        frame_buf = frame

    device = bundle['device']
    step = 0
    x_vels: list[float] = []
    terminated = False

    while step < n_steps and not terminated:
        if real_obs:
            obs_hat = obs_gym
        else:
            # Encode real observation
            z = encode_obs(bundle, frame_buf, prev_frame, state)   # (1, D)

            # Decode z → obs_hat via probe (encoder z, never predictor z)
            with torch.no_grad():
                obs_hat = probe_net(z)[0].cpu().numpy()            # (17,)

            # Ablation: patch specific dims with real obs values
            if real_height:
                obs_hat = obs_hat.copy()
                obs_hat[0] = obs_gym[0]
            if real_angle:
                obs_hat = obs_hat.copy()
                obs_hat[1] = obs_gym[1]

        # SAC acts on (possibly patched) obs
        action = sac_policy.act(obs_hat)                       # (6,)

        # Step real environment
        prev_frame = frame.copy()
        frame, state, _, done_vis, info_vis = visual_env.step(action)
        if _fs > 1:
            push_frame(frame_buf, frame)
        else:
            frame_buf = frame

        obs_gym, reward, term, trunc, info = eval_env.step(action)
        step += 1
        x_vels.append(float(info.get('x_velocity', 0.0)))
        terminated = term or trunc or done_vis

    data         = eval_env.unwrapped.data
    forward_disp = float(data.qpos[0])
    final_height = float(data.qpos[1])
    final_angle  = float(data.qpos[2])
    survived     = (step >= n_steps) and not terminated
    avg_vel      = float(np.mean(x_vels)) if x_vels else 0.0

    return {
        'survived': survived,
        'steps': step,
        'forward_disp': forward_disp,
        'avg_x_velocity': avg_vel,
        'final_height': final_height,
        'final_angle': final_angle,
    }


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Latent-SAC bridge: encoder → probe → SAC → real env')
    p.add_argument('--ckpt',        required=True)
    p.add_argument('--cfg',         default=None)
    p.add_argument('--hdf5-dir',    required=True)
    p.add_argument('--probe-path',  default=None)
    p.add_argument('--probe-episodes', type=int, default=200)
    p.add_argument('--probe-epochs',   type=int, default=30)
    p.add_argument('--probe-hidden',   type=int, default=128,
                   help='Hidden size of MLP probe (default 128; try 256 for better quality)')
    p.add_argument('--sac-repo',    default='sdpkjc/Walker2d-v4-sac_continuous_action-seed4')
    p.add_argument('--sac-ckpt',    default=None)
    p.add_argument('--trials',      type=int, default=5)
    p.add_argument('--n-steps',     type=int, default=600)
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--image-size',  type=int, default=64)
    p.add_argument('--device',      default='cuda')
    # obs_hat patch: clamp or substitute badly-predicted dims
    p.add_argument('--fix-height',  action='store_true',
                   help='Replace probe-decoded z_height with fixed constant --height-val '
                        '(was harmful when height R² > 0.4 — kept for reference)')
    p.add_argument('--height-val',  type=float, default=1.35)
    p.add_argument('--fix-angle',   action='store_true',
                   help='Clamp probe-decoded torso angle to [-0.5, 0.5]')
    # ablation flags
    p.add_argument('--real-height', action='store_true',
                   help='Replace probe obs[0] (height) with real gymnasium obs[0] '
                        '(ablation: does the probe\'s height error cause early falling?)')
    p.add_argument('--real-angle',  action='store_true',
                   help='Replace probe obs[1] (tilt) with real gymnasium obs[1]')
    p.add_argument('--real-obs',    action='store_true',
                   help='Bypass probe entirely; feed real gymnasium obs to SAC '
                        '(upper bound ablation; should match SAC real-env baseline ~3.9 m/s)')
    args = p.parse_args()

    # ── load bundle ──────────────────────────────────────────────────────────
    from experiments.walker2d_smwm_utils import (
        load_walker_bundle, fit_walker_mlp_probe, MLPStateProbe,
    )

    def _cfg(ckpt, override):
        if override:
            return override
        for name in ['env_config.yaml', 'config.yaml']:
            c = Path(ckpt).parent / name
            if c.exists():
                return str(c)
        raise FileNotFoundError(f'No config found near {ckpt}')

    print('[latent-SAC] Loading model bundle …')
    bundle = load_walker_bundle(args.ckpt, _cfg(args.ckpt, args.cfg), args.device)

    # ── fit or load probe ─────────────────────────────────────────────────────
    z_dim = int(bundle['model_cfg'].get('latent_dim', 192))
    if args.probe_path and Path(args.probe_path).exists():
        print(f'[latent-SAC] Loading probe from {args.probe_path} …')
        _ck     = torch.load(args.probe_path, map_location=bundle['device'],
                             weights_only=False)
        _hidden = _ck.get('hidden', 128) if isinstance(_ck, dict) else 128
        probe   = MLPStateProbe(z_dim, hidden=_hidden).to(bundle['device'])
        probe._net.load_state_dict(
            _ck['state_dict'] if isinstance(_ck, dict) and 'state_dict' in _ck
            else _ck)
        probe._net.eval()
        probe_net = probe._net
    else:
        print(f'[latent-SAC] Fitting encoder MLP probe (hidden={args.probe_hidden}) …')
        probe = fit_walker_mlp_probe(bundle, args.hdf5_dir,
                                     max_episodes=args.probe_episodes,
                                     n_epochs=args.probe_epochs,
                                     hidden=args.probe_hidden)
        probe_net = probe._net.eval()
        if args.probe_path:
            Path(args.probe_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({'state_dict': probe._net.state_dict(),
                        'hidden': args.probe_hidden}, args.probe_path)
            print(f'[latent-SAC] Probe saved → {args.probe_path}')

    # Optionally wrap probe_net to patch badly-predicted dimensions
    if args.fix_height or args.fix_angle:
        _raw_net = probe_net
        _hval    = args.height_val
        _fh      = args.fix_height
        _fa      = args.fix_angle

        class _PatchedProbe(torch.nn.Module):
            def forward(self, z):
                out = _raw_net(z).clone()
                if _fh:
                    out[:, 0] = _hval
                if _fa:
                    out[:, 1] = out[:, 1].clamp(-0.5, 0.5)
                return out

        probe_net = _PatchedProbe().to(bundle['device']).eval()

    # ── load SAC policy ───────────────────────────────────────────────────────
    from experiments.walker2d_ppo_utils import download_and_load_sac, load_sac_from_local
    print('[latent-SAC] Loading SAC policy …')
    sac_policy = (load_sac_from_local(args.sac_ckpt)
                  if args.sac_ckpt else
                  download_and_load_sac(args.sac_repo, device='cpu'))
    print('[latent-SAC] SAC policy loaded.')

    # ── environments ─────────────────────────────────────────────────────────
    import gymnasium as gym
    from envs.walker2d_visual import Walker2dVisual

    eval_env   = gym.make('Walker2d-v4')
    visual_env = Walker2dVisual(image_size=args.image_size)

    mode = ('real-obs' if args.real_obs else
            '+'.join(filter(None, [
                'real-h' if args.real_height else '',
                'real-ang' if args.real_angle else '',
                'fix-h' if args.fix_height else '',
                'fix-ang' if args.fix_angle else '',
            ])) or 'probe-only')
    print(f'\n[latent-SAC Walker2d]  obs-mode={mode}')
    print(f'  trials={args.trials}  n_steps={args.n_steps}  seed={args.seed}')
    print()

    all_vels = []
    for i in range(args.trials):
        t0  = time.time()
        row = run_trial(bundle, probe_net, sac_policy, eval_env, visual_env,
                        seed=args.seed + i, n_steps=args.n_steps,
                        real_height=args.real_height,
                        real_angle=args.real_angle,
                        real_obs=args.real_obs)
        elapsed = time.time() - t0
        all_vels.append(row['avg_x_velocity'])
        status = 'SUCCESS' if row['survived'] else 'FAIL'
        print(f'  trial {i:02d}  {status}  steps={row["steps"]}  '
              f'avg_vel={row["avg_x_velocity"]:+.3f}  '
              f'disp={row["forward_disp"]:+.2f}m  '
              f'h={row["final_height"]:.3f}  t={elapsed:.1f}s')

    print(f'\nMean avg_x_velocity: {np.mean(all_vels):.3f} m/s  '
          f'(range {min(all_vels):.3f} – {max(all_vels):.3f})')

    eval_env.close()
    visual_env.close()


if __name__ == '__main__':
    main()
