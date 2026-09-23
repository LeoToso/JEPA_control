#!/usr/bin/env python
"""Latent iCEM-MPC for Walker2d-v4 using a learned SMWM world model.

Mirrors gt_icem_walker.py exactly, with one substitution:
  GT:     rollout_cost uses real MuJoCo physics
  Latent: rollout_cost uses latent_step (world model) + MLP probe (z → obs17)

The cost is identical — all terms computable from the decoded 17-D obs:
  obs[0]   = torso z-height
  obs[1]   = torso tilt angle
  obs[2:8] = joint angles (qpos[3:9])
  obs[8]   = x-velocity (qvel[0])  ← direct, no Δx/dt needed

MPC loop
--------
  1. Encode real observation → z_t
  2. iCEM plans optimal action sequence in latent space
  3. Execute first executed_steps actions in REAL environment
  4. Re-encode from new real observation → z_{t+1}
  5. Repeat

This tests whether the world model is accurate enough to plan walking gaits.
A model with poor action sensitivity (MSP+SIG) will produce bad cost estimates
and fail to find walking sequences; a good model (1SP+EP-IDM) should succeed.

Usage
-----
MUJOCO_GL=egl python experiments/latent_icem_walker.py \\
    --ckpt  /mnt/t7shield/jepa_results/walker2d_mixed_sac_smwm_fwd_endpoint_inverse_act1_seed42/model_final.pt \\
    --cfg   configs/walker2d_smwm_fwd_endpoint_inverse_act1.yaml \\
    --hdf5-dir data/walker2d_mixed_sac_fs5_64 \\
    --probe-path results/probes/fwd_ep_ar_mlp_probe.pt \\
    --trials 10 --n-steps 600 \\
    --planning-horizon 60 --executed-steps 5 \\
    --cem-population 500 --cem-elites 50 --cem-iters 5 \\
    --beta 0.5 --initial-std 0.5 \\
    --keep-fraction 0.3 --shift-fraction 0.3 --sample-decay 1.25 \\
    --wx 0.2 --wh 4.0 --wu 1e-3 --cf 50.0 \\
    --wang 1.5 --wz 0.0 --height-target 1.35 \\
    --wjoint 0.02 --wsmooth 0.05 \\
    --healthy-angle-max 1.0 \\
    --render-dir results/latent_icem_fwd_ep_ar_frames \\
    --output results/latent_icem_fwd_ep_ar.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch

ACTION_LOW  = -1.0
ACTION_HIGH =  1.0
ACTION_DIM  =  6

HEALTHY_Z_MIN   = 0.8
HEALTHY_Z_MAX   = 2.0
HEALTHY_ANG_MAX = 1.0


# ── latent rollout cost ────────────────────────────────────────────────────────

@torch.no_grad()
def latent_rollout_cost(
    bundle, probe, start_z: torch.Tensor,
    actions: np.ndarray,
    wx=1.0, wh=1.0, wu=1e-3, cf=10.0,
    wz=0.0, wang=0.0, height_target=1.2,
    wjoint=0.0, wsmooth=0.0, wback=2.0,
    wstability=0.0,
) -> float:
    """Evaluate one action sequence in latent space.

    start_z : (1, latent_dim) tensor on bundle device
    actions : (H, 6) numpy array

    Cost = -sum(per-step reward) + cf * (H - t_alive)
    Per-step reward = wx*max(x_vel,0) - wback*max(-x_vel,0) + wh
                      - wu*||u||² - wz*(h-h*)² - wang*ang²
                      - wjoint*||joints||² - wsmooth*||Δa||²
                      - wstability*||posture_t - posture_{t-1}||²

    x_vel    = decoded obs[8]      (qvel[0], forward velocity)
    h        = decoded obs[0]      (torso z-height)
    ang      = decoded obs[1]      (torso tilt angle)
    joints   = decoded obs[2:8]    (qpos[3:9])
    posture  = decoded obs[0:9]    (height + angle + joints, for stability term)
    """
    from experiments.walker2d_smwm_utils import latent_step, is_healthy_obs

    z      = start_z.clone()
    H      = len(actions)
    t_alive      = 0
    dense_reward = 0.0
    a_prev        = np.zeros(ACTION_DIM, dtype=np.float32)
    prev_posture  = None

    for t in range(H):
        a     = np.clip(actions[t], ACTION_LOW, ACTION_HIGH).astype(np.float32)
        x_hat = probe(z[0].cpu().numpy())   # decoded 17-D obs

        if not is_healthy_obs(x_hat):
            break

        x_vel      = float(x_hat[8])
        h          = float(x_hat[0])
        ang        = float(x_hat[1])
        joints     = x_hat[2:8]
        posture    = x_hat[0:9]
        ctrl_cost  = wu * float(np.dot(a, a))
        posture_pen = wz * (h - height_target) ** 2 + wang * ang ** 2
        joint_pen  = wjoint  * float(np.dot(joints, joints))
        smooth_pen = wsmooth * float(np.dot(a - a_prev, a - a_prev))
        fwd_term   = wx * max(x_vel, 0.0) - wback * max(-x_vel, 0.0)
        stab_pen   = (wstability * float(np.dot(posture - prev_posture,
                                                posture - prev_posture))
                      if prev_posture is not None else 0.0)

        dense_reward += (fwd_term + wh - ctrl_cost
                         - posture_pen - joint_pen - smooth_pen - stab_pen)
        t_alive      += 1
        a_prev        = a
        prev_posture  = posture
        z             = latent_step(bundle, z, a)

    return -dense_reward + cf * (H - t_alive)


# ── colored noise (identical to gt_icem_walker.py) ────────────────────────────

def sample_colored_noise(n, horizon, action_dim, beta, std):
    fft_len = horizon // 2 + 1
    freqs   = np.fft.rfftfreq(horizon)
    freqs[0] = 1.0
    power    = freqs ** (-beta / 2.0)
    power[0] = 0.0
    white   = (np.random.randn(n, action_dim, fft_len)
               + 1j * np.random.randn(n, action_dim, fft_len))
    colored = white * power[None, None, :]
    noise   = np.fft.irfft(colored, n=horizon, axis=-1).transpose(0, 2, 1)
    scale   = noise.std(axis=1, keepdims=True).clip(1e-8)
    return noise / scale * std[None]


# ── iCEM planner ──────────────────────────────────────────────────────────────

class LatentWalkeriCEM:
    """iCEM-MPC that plans in the world model's latent space."""

    def __init__(self, horizon, executed_steps,
                 population, elites, iterations,
                 initial_std, beta,
                 keep_fraction, shift_fraction, sample_decay,
                 bundle, probe,
                 wx=1.0, wh=1.0, wu=1e-3, cf=10.0,
                 wz=0.0, wang=0.0, height_target=1.2,
                 wjoint=0.0, wsmooth=0.0, wback=2.0, wstability=0.0,
                 execute_best=True):
        self.horizon        = int(horizon)
        self.executed_steps = min(int(executed_steps), horizon)
        self.population     = int(population)
        self.elites         = min(int(elites), population)
        self.iterations     = int(iterations)
        self.initial_std    = float(initial_std)
        self.beta           = float(beta)
        self.keep_fraction  = float(keep_fraction)
        self.shift_fraction = float(shift_fraction)
        self.sample_decay   = float(sample_decay)
        self.bundle         = bundle
        self.probe          = probe
        self.wx             = float(wx)
        self.wh             = float(wh)
        self.wu             = float(wu)
        self.cf             = float(cf)
        self.wz             = float(wz)
        self.wang           = float(wang)
        self.height_target  = float(height_target)
        self.wjoint         = float(wjoint)
        self.wsmooth        = float(wsmooth)
        self.wback          = float(wback)
        self.wstability     = float(wstability)
        self.execute_best   = bool(execute_best)
        self._prev_mean    = None
        self._shift_elites = None

    def _init_std(self):
        return np.full((self.horizon, ACTION_DIM), self.initial_std)

    def _shift_seq(self, seq):
        e = self.executed_steps
        h = self.horizon
        d = seq.shape[-1]
        prefix = seq[..., e:, :]
        pad    = np.zeros(seq.shape[:-2] + (e, d))
        return np.concatenate([prefix, pad], axis=-2)

    def plan(self, start_z: torch.Tensor) -> np.ndarray:
        """Run iCEM from latent state start_z; return (executed_steps, 6) actions."""
        mean = self._shift_seq(self._prev_mean) if self._prev_mean is not None \
               else np.zeros((self.horizon, ACTION_DIM))
        std          = self._init_std()
        best_cost    = np.inf
        best_seq     = mean.copy()
        prev_elites  = None
        shift_pool   = self._shift_elites

        for i in range(self.iterations):
            n_new = max(int(math.ceil(self.population / (self.sample_decay ** i))),
                        2 * self.elites)
            noise   = sample_colored_noise(n_new, self.horizon, ACTION_DIM,
                                           self.beta, std)
            samples = np.clip(mean[None] + noise, ACTION_LOW, ACTION_HIGH)

            if i == 0 and shift_pool is not None:
                n_add   = max(1, int(self.shift_fraction * len(shift_pool)))
                samples = np.concatenate([samples, shift_pool[:n_add]], axis=0)
            elif i > 0 and prev_elites is not None:
                n_add   = max(1, int(self.keep_fraction * len(prev_elites)))
                samples = np.concatenate([samples, prev_elites[:n_add]], axis=0)

            if i == self.iterations - 1:
                samples = np.concatenate([samples, mean[None]], axis=0)

            costs = np.array([
                latent_rollout_cost(
                    self.bundle, self.probe, start_z, samples[k],
                    self.wx, self.wh, self.wu, self.cf,
                    self.wz, self.wang, self.height_target,
                    self.wjoint, self.wsmooth, self.wback, self.wstability)
                for k in range(len(samples))
            ])

            bi = int(np.argmin(costs))
            if costs[bi] < best_cost:
                best_cost = costs[bi]
                best_seq  = samples[bi].copy()

            elite_idx   = np.argsort(costs)[:self.elites]
            prev_elites = samples[elite_idx]
            mean        = prev_elites.mean(0)
            std         = prev_elites.std(0).clip(min=1e-4)

        if prev_elites is not None:
            n_shift            = max(1, int(self.shift_fraction * len(prev_elites)))
            self._shift_elites = np.clip(
                self._shift_seq(prev_elites[:n_shift]), ACTION_LOW, ACTION_HIGH)
        self._prev_mean = mean

        seq = best_seq if self.execute_best else mean
        return np.clip(seq[:self.executed_steps], ACTION_LOW, ACTION_HIGH)


# ── MPC trial ─────────────────────────────────────────────────────────────────

def run_trial(planner, eval_env, visual_env,
              bundle, initial_seed, n_steps, do_render):
    """MPC loop: plan in latent space, execute in real env, re-encode."""
    from experiments.walker2d_smwm_utils import is_healthy_obs
    from experiments.sensorimotor_probe_utils import encode_obs

    # Reset both envs to same initial state
    obs_gym, _ = eval_env.reset(seed=initial_seed)
    frame, state, _ = visual_env.reset()
    prev_frame = frame.copy()

    planner._prev_mean    = None
    planner._shift_elites = None

    # Initial encode
    z = encode_obs(bundle, frame, prev_frame, state)

    step        = 0
    x_vels      = []
    frames      = []
    terminated  = False

    while step < n_steps and not terminated:
        sequence = planner.plan(z)

        for a in sequence:
            if step >= n_steps or terminated:
                break
            prev_frame = frame.copy()
            frame, state, _, done, info = visual_env.step(a)
            obs_gym, _, term_gym, trunc_gym, info_gym = eval_env.step(a)
            step += 1
            x_vels.append(float(info_gym.get('x_velocity', 0.0)))
            if do_render:
                frames.append(eval_env.render())
            terminated = term_gym or trunc_gym or done
            if terminated:
                break

        if not terminated:
            # Re-encode from new real observation
            z = encode_obs(bundle, frame, prev_frame, state)

    data         = eval_env.unwrapped.data
    final_height = float(data.qpos[1])
    final_angle  = float(data.qpos[2])
    forward_disp = float(data.qpos[0])
    survived     = (step >= n_steps) and not terminated
    avg_vel      = float(np.mean(x_vels)) if x_vels else 0.0

    return {
        'survived_full':  survived,
        'steps':          step,
        'forward_disp':   forward_disp,
        'avg_x_velocity': avg_vel,
        'final_height':   final_height,
        'final_angle':    final_angle,
        'x_velocities':   x_vels,
    }, frames


# ── visualization helpers (copied from gt_icem_walker.py) ─────────────────────

def save_gif(frames, out_path, fps=30):
    if not frames:
        return
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio
        imageio.mimsave(str(out), frames, fps=fps)
    except ImportError:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(str(out), save_all=True, append_images=imgs[1:],
                     loop=0, duration=int(1000 / fps))
    print(f'[gif saved] {out}')


def save_frame_grid(frames, out_path, every=10, title='Latent iCEM Walker2d'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    subset = frames[::every]
    n = len(subset)
    if n == 0:
        return
    cols = min(8, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = np.array(axes).reshape(rows, cols)
    for idx, frame in enumerate(subset):
        r, c = divmod(idx, cols)
        axes[r, c].imshow(frame)
        axes[r, c].axis('off')
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis('off')
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'[grid saved] {out}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Latent iCEM-MPC for Walker2d-v4 using a SMWM world model.')
    # model
    p.add_argument('--ckpt',         required=True, help='Path to model_final.pt')
    p.add_argument('--cfg',          default=None,  help='Path to YAML config (auto-detected if omitted)')
    p.add_argument('--hdf5-dir',     required=True, help='Dataset dir for fitting MLP probe')
    p.add_argument('--probe-path',   default=None,
                   help='Save/load MLP probe (.pt).  Trained once, reused on subsequent runs.')
    p.add_argument('--probe-episodes', type=int, default=200)
    p.add_argument('--probe-epochs',   type=int, default=30)
    p.add_argument('--device',       default='cuda')
    p.add_argument('--image-size',   type=int, default=64)
    # trial
    p.add_argument('--trials',       type=int,   default=10)
    p.add_argument('--n-steps',      type=int,   default=600)
    # planner
    p.add_argument('--planning-horizon',  type=int,   default=60)
    p.add_argument('--executed-steps',    type=int,   default=5)
    # iCEM
    p.add_argument('--cem-population',    type=int,   default=500)
    p.add_argument('--cem-elites',        type=int,   default=50)
    p.add_argument('--cem-iters',         type=int,   default=5)
    p.add_argument('--initial-std',       type=float, default=0.5)
    p.add_argument('--beta',              type=float, default=0.5)
    p.add_argument('--keep-fraction',     type=float, default=0.3)
    p.add_argument('--shift-fraction',    type=float, default=0.3)
    p.add_argument('--sample-decay',      type=float, default=1.25)
    p.add_argument('--no-execute-best',   action='store_true')
    # cost
    p.add_argument('--wx',            type=float, default=0.2)
    p.add_argument('--wh',            type=float, default=4.0)
    p.add_argument('--wu',            type=float, default=1e-3)
    p.add_argument('--cf',            type=float, default=50.0)
    p.add_argument('--wz',            type=float, default=0.0,
                   help='Height-deviation penalty weight; set 0 (default) because '
                        'z_height probe R²≈−0.5 — use --wang instead')
    p.add_argument('--wang',          type=float, default=1.5)
    p.add_argument('--height-target', type=float, default=1.35)
    p.add_argument('--wjoint',        type=float, default=0.02)
    p.add_argument('--wsmooth',       type=float, default=0.05)
    p.add_argument('--wback',         type=float, default=2.0,
                   help='Penalty weight for backward motion (x_vel < 0)')
    p.add_argument('--wstability',    type=float, default=0.0,
                   help='Penalty for change in decoded posture between steps '
                        '(||posture_t - posture_{t-1}||²); use with --wx 0 to '
                        'drop velocity term and optimise for stable dynamics only')
    p.add_argument('--healthy-angle-max', type=float, default=1.0,
                   help='Latent-rollout fall threshold: |torso_angle| > this → fallen. '
                        'Replaces z_height check (unreliable from probe).')
    # warm-start
    p.add_argument('--sac-warmstart', action='store_true',
                   help='Seed first iCEM mean from a real SAC action sequence '
                        'from the HDF5 dataset (avoids cold-start in random space)')
    p.add_argument('--sac-warmstart-episode', type=int, default=0,
                   help='Which HDF5 episode to take SAC actions from')
    # success
    p.add_argument('--success-min-velocity', type=float, default=0.5)
    # output
    p.add_argument('--render-dir',    default='')
    p.add_argument('--render-every',  type=int,   default=10)
    p.add_argument('--gif-fps',       type=int,   default=30)
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--output',        required=True)
    args = p.parse_args()

    # ── override health bound ─────────────────────────────────────────────────
    import experiments.walker2d_smwm_utils as _wu
    _wu.HEALTHY_ANG_MAX = args.healthy_angle_max

    np.random.seed(args.seed)

    # ── load bundle ───────────────────────────────────────────────────────────
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

    print('[latent-iCEM] Loading model bundle …')
    bundle = load_walker_bundle(args.ckpt, _cfg(args.ckpt, args.cfg), args.device)

    # ── fit or load MLP probe ─────────────────────────────────────────────────
    z_dim = int(bundle['model_cfg'].get('latent_dim', 192))
    if args.probe_path and Path(args.probe_path).exists():
        print(f'[latent-iCEM] Loading MLP probe from {args.probe_path} …')
        probe = MLPStateProbe(z_dim).to(bundle['device'])
        probe._net.load_state_dict(
            torch.load(args.probe_path, map_location=bundle['device'],
                       weights_only=True))
        probe._net.eval()
    else:
        print('[latent-iCEM] Fitting MLP probe …')
        probe = fit_walker_mlp_probe(bundle, args.hdf5_dir,
                                     max_episodes=args.probe_episodes,
                                     n_epochs=args.probe_epochs)
        if args.probe_path:
            Path(args.probe_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(probe._net.state_dict(), args.probe_path)
            print(f'[latent-iCEM] Probe saved → {args.probe_path}')

    # ── SAC warm-start: load real action sequence from HDF5 ───────────────────
    sac_warmstart_seq = None
    if args.sac_warmstart:
        import h5py, glob as _glob
        hdf5_files = sorted(_glob.glob(str(Path(args.hdf5_dir) / '*.hdf5')))
        if not hdf5_files:
            print('[latent-iCEM] WARNING: no HDF5 files found, skipping warm-start')
        else:
            hf = h5py.File(hdf5_files[0], 'r')
            # structure: hdf5['episodes'][ep_key]['actions']
            ep_group = hf['episodes'] if 'episodes' in hf else hf
            eps  = list(ep_group.keys())
            ep   = eps[args.sac_warmstart_episode % len(eps)]
            acts = np.array(ep_group[ep]['actions'])     # (T, 6)
            hf.close()
            H = args.planning_horizon
            if len(acts) >= H:
                sac_warmstart_seq = acts[:H].astype(np.float32)
            else:
                pad = np.zeros((H - len(acts), ACTION_DIM), dtype=np.float32)
                sac_warmstart_seq = np.concatenate([acts, pad], axis=0)
            print(f'[latent-iCEM] SAC warm-start from {Path(hdf5_files[0]).name} '
                  f'ep={ep}  H={H}  act_mean={sac_warmstart_seq.mean():.3f}')

    # ── build planner ─────────────────────────────────────────────────────────
    planner = LatentWalkeriCEM(
        horizon        = args.planning_horizon,
        executed_steps = args.executed_steps,
        population     = args.cem_population,
        elites         = args.cem_elites,
        iterations     = args.cem_iters,
        initial_std    = args.initial_std,
        beta           = args.beta,
        keep_fraction  = args.keep_fraction,
        shift_fraction = args.shift_fraction,
        sample_decay   = args.sample_decay,
        bundle         = bundle,
        probe          = probe,
        wx             = args.wx,
        wh             = args.wh,
        wu             = args.wu,
        cf             = args.cf,
        wz             = args.wz,
        wang           = args.wang,
        height_target  = args.height_target,
        wjoint         = args.wjoint,
        wsmooth        = args.wsmooth,
        wback          = args.wback,
        wstability     = args.wstability,
        execute_best   = not args.no_execute_best,
    )

    # ── environments ──────────────────────────────────────────────────────────
    import gymnasium as gym
    from envs.walker2d_visual import Walker2dVisual

    eval_env   = gym.make('Walker2d-v4',
                          render_mode='rgb_array' if args.render_dir else None)
    visual_env = Walker2dVisual(image_size=args.image_size)

    model_name = Path(args.ckpt).parent.name
    print(f'\n[latent-iCEM Walker2d]  model={model_name}')
    print(f'  H={args.planning_horizon}  exec={args.executed_steps}  '
          f'pop={args.cem_population}  elites={args.cem_elites}  '
          f'iters={args.cem_iters}')
    print(f'  β={args.beta}  std₀={args.initial_std}  '
          f'keep={args.keep_fraction}  shift={args.shift_fraction}  '
          f'decay={args.sample_decay}')
    print(f'  cost: wx={args.wx}  wh={args.wh}  wu={args.wu}  cf={args.cf}  '
          f'wz={args.wz}  wang={args.wang}  h*={args.height_target}  '
          f'wback={args.wback}  wstability={args.wstability}')
    print(f'  fall criterion: |torso_angle| > {args.healthy_angle_max}')

    trials_data  = []
    frames_list  = []
    success_mask = []

    for i in range(args.trials):
        np.random.seed(args.seed + i)
        print(f'[trial {i:03d}] ', end='', flush=True)
        t0 = time.time()

        # seed iCEM mean from SAC actions if requested
        if sac_warmstart_seq is not None:
            planner._prev_mean = sac_warmstart_seq.copy()
        else:
            planner._prev_mean = None
        planner._shift_elites = None

        row, frames = run_trial(
            planner, eval_env, visual_env,
            bundle, args.seed + i, args.n_steps,
            bool(args.render_dir))

        elapsed = time.time() - t0
        success = row['survived_full'] and row['avg_x_velocity'] >= args.success_min_velocity
        row['success'] = success
        success_mask.append(success)
        trials_data.append(row)
        frames_list.append(frames)

        status = 'SUCCESS' if success else 'FAIL'
        print(f'{status}  survived={row["survived_full"]}  '
              f'steps={row["steps"]}  disp={row["forward_disp"]:.2f}m  '
              f'vel={row["avg_x_velocity"]:.3f}m/s  '
              f'h={row["final_height"]:.3f}  t={elapsed:.1f}s')

    n_success = sum(success_mask)
    sr        = n_success / max(args.trials, 1)
    mean_vel  = float(np.mean([r['avg_x_velocity'] for r in trials_data]))
    mean_disp = float(np.mean([r['forward_disp']   for r in trials_data]))

    print(f'\nSuccess: {n_success}/{args.trials} ({sr:.1%})')
    print(f'Mean avg_x_velocity: {mean_vel:.3f} m/s  |  '
          f'Mean forward disp: {mean_disp:.2f} m')

    if args.render_dir:
        successes = [i for i, s in enumerate(success_mask) if s]
        viz_idx   = (max(successes, key=lambda i: trials_data[i]['avg_x_velocity'])
                     if successes else
                     max(range(len(trials_data)),
                         key=lambda i: trials_data[i]['steps']))
        viz_frames = frames_list[viz_idx]
        row        = trials_data[viz_idx]
        tag        = 'success' if success_mask[viz_idx] else 'best_failed'
        stem       = f'{tag}_trial_{viz_idx:03d}'
        title      = (f'Latent iCEM [{model_name}] — trial {viz_idx} [{tag}] '
                      f'vel={row["avg_x_velocity"]:.3f} disp={row["forward_disp"]:.2f}m')
        if viz_frames:
            save_gif(viz_frames, Path(args.render_dir) / f'{stem}.gif',
                     fps=args.gif_fps)
            save_frame_grid(viz_frames, Path(args.render_dir) / f'{stem}.png',
                            every=args.render_every, title=title)

    for r in trials_data:
        r.pop('x_velocities', None)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'protocol': {
            'planner':          'latent_icem_mpc',
            'model':            model_name,
            'env':              'Walker2d-v4',
            'n_steps':          args.n_steps,
            'horizon':          args.planning_horizon,
            'executed_steps':   args.executed_steps,
            'population':       args.cem_population,
            'elites':           args.cem_elites,
            'iterations':       args.cem_iters,
            'initial_std':      args.initial_std,
            'beta':             args.beta,
            'keep_fraction':    args.keep_fraction,
            'shift_fraction':   args.shift_fraction,
            'sample_decay':     args.sample_decay,
            'execute_best':     not args.no_execute_best,
            'wx': args.wx, 'wh': args.wh, 'wu': args.wu, 'cf': args.cf,
            'wz': args.wz, 'wang': args.wang, 'height_target': args.height_target,
            'wjoint': args.wjoint, 'wsmooth': args.wsmooth,
            'wback': args.wback, 'wstability': args.wstability,
            'healthy_angle_max': args.healthy_angle_max,
            'success_min_vel':  args.success_min_velocity,
            'seed':             args.seed,
            'n_trials':         args.trials,
        },
        'success_rate':      sr,
        'n_success':         n_success,
        'mean_avg_velocity': mean_vel,
        'mean_forward_disp': mean_disp,
        'trials':            trials_data,
    }, indent=2))
    print(f'[done] {out}')

    eval_env.close()
    visual_env.close()


if __name__ == '__main__':
    main()
