#!/usr/bin/env python
"""GT iCEM-MPC oracle for Walker2d-v4 — Pinneri et al. 2020 improvements.

iCEM adds four improvements over vanilla CEM:
  1. Colored noise (β > 0): temporally-correlated action proposals instead of
     white Gaussian noise.  β=2.5 is recommended for Walker Walk (Table S6).
  2. Elite memory: keep a fraction of elites from each inner CEM iteration and
     add them to the next iteration's candidate pool.
  3. Shift elites: keep a fraction of elites from the last inner iteration and
     shift them forward by executed_steps to warm-start the next MPC call.
  4. Sample decay: shrink the population across inner iterations so compute is
     concentrated on refinement: N_i = max(ceil(N / γ^i), 2K).

Full (H, 6) action sequences are optimised directly — no action blocking.
Colored noise already provides temporal smoothness (analogous to blocking but
theoretically grounded).

Reference
---------
Pinneri et al. (2020) "Sample-Efficient Cross-Entropy Method for Real-time
Planning."  arXiv:2008.06389.

Recommended settings (Table S6, ground-truth dynamics):
  horizon 30 | β 2.5 | initial_std 0.5 | keep/shift fraction 0.3 | γ 1.25

Usage
-----
  MUJOCO_GL=egl python experiments/gt_icem_walker.py \\
      --trials 5 --n-steps 400 \\
      --planning-horizon 30 \\
      --cem-population 500 --cem-elites 50 --cem-iters 5 \\
      --executed-steps 10 \\
      --beta 2.5 --initial-std 0.5 \\
      --keep-fraction 0.3 --shift-fraction 0.3 --sample-decay 1.25 \\
      --wx 5.0 --cf 10.0 --wu 1e-3 \\
      --wz 0.2 --wang 0.3 --height-target 1.2 \\
      --success-min-velocity 0.5 \\
      --render-dir results/walker_gt_icem_frames \\
      --output results/gt_icem_walker.json
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
import gymnasium as gym

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

ACTION_LOW  = -1.0
ACTION_HIGH =  1.0
ACTION_DIM  =  6

HEALTHY_Z_MIN   = 0.8   # overridden at runtime by --healthy-z-min
HEALTHY_Z_MAX   = 2.0
HEALTHY_ANG_MAX = 1.0


# ── env state helpers ─────────────────────────────────────────────────────────

def get_state(env):
    d = env.unwrapped.data
    return d.qpos.copy(), d.qvel.copy()


def set_state(env, qpos, qvel):
    env.unwrapped.set_state(qpos, qvel)


def is_healthy(data):
    z   = float(data.qpos[1])
    ang = float(data.qpos[2])
    return HEALTHY_Z_MIN < z < HEALTHY_Z_MAX and abs(ang) < HEALTHY_ANG_MAX


def failure_reason(data):
    z   = float(data.qpos[1])
    ang = float(data.qpos[2])
    if z <= HEALTHY_Z_MIN:    return 'low_height'
    if z >= HEALTHY_Z_MAX:    return 'high_height'
    if abs(ang) >= HEALTHY_ANG_MAX: return 'excessive_angle'
    return 'other'


# ── rollout cost — dense per-step Gym reward ──────────────────────────────────

def rollout_cost(plan_env, start_qpos, start_qvel, actions, frame_skip,
                 wx=1.0, wh=1.0, wu=0.001, cf=10.0,
                 wz=0.0, wang=0.0, height_target=1.2,
                 wjoint=0.0, wsmooth=0.0):
    """Dense per-step objective matching MuJoCo Gym Walker2d-v4.

    Per alive step:
        r_t = wx * x_vel_t  +  wh  -  wu * ||u_t||²
              - wz*(h_t - h*)²  - wang*ang_t²
              - wjoint * ||qpos[3:9]||²          (leg joint angle regularisation)
              - wsmooth * ||a_t - a_{t-1}||²     (action smoothness)

    Cost = -sum(r_t)  +  cf*(H - t_alive)

    x_vel_t = Δx / (frame_skip * dt_model) — same formula as Gym's forward_reward.
    This rewards steady forward movement at every step and penalises
    'store-up-then-sprint' strategies that the endpoint objective exploits.
    """
    set_state(plan_env, start_qpos, start_qvel)
    model = plan_env.unwrapped.model
    data  = plan_env.unwrapped.data
    dt    = frame_skip * float(model.opt.timestep)

    H            = len(actions)
    t_alive      = 0
    dense_reward = 0.0
    a_prev       = np.zeros(ACTION_DIM)

    for t in range(H):
        a        = np.clip(actions[t], ACTION_LOW, ACTION_HIGH)
        x_before = float(data.qpos[0])

        if HAS_MUJOCO:
            data.ctrl[:] = a
            for _ in range(frame_skip):
                mujoco.mj_step(model, data)
            if not is_healthy(data):
                break
        else:
            plan_env.step(a)
            if not is_healthy(plan_env.unwrapped.data):
                break

        t_alive   += 1
        x_vel_t    = (float(data.qpos[0]) - x_before) / dt
        h          = float(data.qpos[1])
        ang        = float(data.qpos[2])
        ctrl_cost  = wu * float(np.dot(a, a))
        posture    = wz * (h - height_target) ** 2 + wang * ang ** 2
        joint_pen  = wjoint  * float(np.dot(data.qpos[3:9], data.qpos[3:9]))
        smooth_pen = wsmooth * float(np.dot(a - a_prev, a - a_prev))
        dense_reward += wx * x_vel_t + wh - ctrl_cost - posture - joint_pen - smooth_pen
        a_prev = a

    return -dense_reward + cf * (H - t_alive)


# ── colored noise ─────────────────────────────────────────────────────────────

def sample_colored_noise(n: int, horizon: int, action_dim: int,
                         beta: float, std: np.ndarray) -> np.ndarray:
    """Return (n, horizon, action_dim) colored-noise samples.

    Power spectrum ∝ f^{-β}: β=0 is white noise, β=2 is Brownian (red) noise,
    β=2.5 is what Pinneri et al. recommend for Walker Walk.

    std has shape (horizon, action_dim); noise is scaled per time-step.
    """
    fft_len = horizon // 2 + 1
    freqs   = np.fft.rfftfreq(horizon)
    freqs[0] = 1.0                    # avoid /0 at DC
    power    = freqs ** (-beta / 2.0)
    power[0] = 0.0                    # zero DC → zero mean

    # Complex white noise: (n, action_dim, fft_len)
    white = (np.random.randn(n, action_dim, fft_len)
             + 1j * np.random.randn(n, action_dim, fft_len))
    colored = white * power[None, None, :]

    # Inverse FFT → (n, action_dim, horizon) → (n, horizon, action_dim)
    noise = np.fft.irfft(colored, n=horizon, axis=-1).transpose(0, 2, 1)

    # Normalise to unit std per sample per dimension, then scale by std
    scale = noise.std(axis=1, keepdims=True).clip(1e-8)
    return noise / scale * std[None]   # (n, horizon, action_dim)


# ── iCEM planner ──────────────────────────────────────────────────────────────

class WalkerGTiCEM:
    """GT iCEM-MPC planner (Pinneri et al. 2020).

    Optimises full (H, ACTION_DIM) action sequences with colored noise,
    elite memory, shift-elites warm start, and sample decay.
    """

    def __init__(self, horizon: int, executed_steps: int,
                 population: int, elites: int, iterations: int,
                 initial_std: float, beta: float,
                 keep_fraction: float, shift_fraction: float,
                 sample_decay: float,
                 plan_env, frame_skip: int,
                 wx: float = 1.0, wh: float = 1.0, wu: float = 0.001,
                 cf: float = 10.0,
                 wz: float = 0.0, wang: float = 0.0,
                 height_target: float = 1.2,
                 wjoint: float = 0.0, wsmooth: float = 0.0,
                 execute_best: bool = True):
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
        self.plan_env       = plan_env
        self.frame_skip     = int(frame_skip)
        self.wx             = float(wx)
        self.wh             = float(wh)
        self.wu             = float(wu)
        self.cf             = float(cf)
        self.wz             = float(wz)
        self.wang           = float(wang)
        self.height_target  = float(height_target)
        self.wjoint         = float(wjoint)
        self.wsmooth        = float(wsmooth)
        self.execute_best   = bool(execute_best)

        self._prev_mean     = None   # (H, D): warm-start mean from last MPC call
        self._shift_elites  = None   # (K_shift, H, D): shifted elites from last call
        self._gt_init_mean  = None   # (H, D): one-shot mean override (SAC warm-start)

    def _init_std(self) -> np.ndarray:
        return np.full((self.horizon, ACTION_DIM), self.initial_std)

    def _shift_seq(self, seq: np.ndarray) -> np.ndarray:
        """Shift a (..., H, D) sequence forward by executed_steps (pad with zeros)."""
        e = self.executed_steps
        h = self.horizon
        d = seq.shape[-1]
        prefix = seq[..., e:, :]                            # (..., H-e, D)
        pad    = np.zeros(seq.shape[:-2] + (e, d))          # (..., e, D)
        return np.concatenate([prefix, pad], axis=-2)        # (..., H, D)

    def plan(self, start_qpos: np.ndarray, start_qvel: np.ndarray) -> np.ndarray:
        """Run iCEM; return (executed_steps, ACTION_DIM) actions to apply."""
        # ── initialise mean & std ────────────────────────────────────────────
        if self._prev_mean is not None:
            mean = self._shift_seq(self._prev_mean)
        elif self._gt_init_mean is not None:
            mean = self._gt_init_mean.copy()
        else:
            mean = np.zeros((self.horizon, ACTION_DIM))

        std          = self._init_std()
        best_cost    = np.inf
        best_seq     = mean.copy()
        prev_elites  = None          # elites from previous inner iteration
        shift_pool   = self._shift_elites   # shifted elites from previous call

        for i in range(self.iterations):
            # ── sample decay ─────────────────────────────────────────────────
            n_new = max(int(math.ceil(self.population / (self.sample_decay ** i))),
                        2 * self.elites)

            # ── colored-noise samples ─────────────────────────────────────────
            noise   = sample_colored_noise(n_new, self.horizon, ACTION_DIM,
                                           self.beta, std)
            samples = np.clip(mean[None] + noise, ACTION_LOW, ACTION_HIGH)

            # ── elite memory ─────────────────────────────────────────────────
            if i == 0 and shift_pool is not None:
                n_add   = max(1, int(self.shift_fraction * len(shift_pool)))
                samples = np.concatenate([samples, shift_pool[:n_add]], axis=0)
            elif i > 0 and prev_elites is not None:
                n_add   = max(1, int(self.keep_fraction * len(prev_elites)))
                samples = np.concatenate([samples, prev_elites[:n_add]], axis=0)

            # ── always include current mean ────────────────────────────────
            if i == self.iterations - 1:
                samples = np.concatenate([samples, mean[None]], axis=0)

            # ── evaluate ────────────────────────────────────────────────────
            costs = np.empty(len(samples))
            for k in range(len(samples)):
                costs[k] = rollout_cost(
                    self.plan_env, start_qpos, start_qvel,
                    samples[k], self.frame_skip,
                    self.wx, self.wh, self.wu, self.cf,
                    self.wz, self.wang, self.height_target,
                    self.wjoint, self.wsmooth)

            # track best seen across all iterations
            bi = int(np.argmin(costs))
            if costs[bi] < best_cost:
                best_cost = costs[bi]
                best_seq  = samples[bi].copy()

            # ── elite update ──────────────────────────────────────────────
            elite_idx   = np.argsort(costs)[:self.elites]
            prev_elites = samples[elite_idx]
            mean        = prev_elites.mean(0)
            std         = prev_elites.std(0).clip(min=1e-4)

        # ── store shifted elites for next MPC call ────────────────────────
        if prev_elites is not None:
            n_shift              = max(1, int(self.shift_fraction * len(prev_elites)))
            self._shift_elites   = np.clip(
                self._shift_seq(prev_elites[:n_shift]), ACTION_LOW, ACTION_HIGH)

        self._prev_mean = mean

        # ── action to execute ──────────────────────────────────────────────
        seq = best_seq if self.execute_best else mean
        return np.clip(seq[:self.executed_steps], ACTION_LOW, ACTION_HIGH)


# ── trial evaluation (identical structure to gt_cem_walker.py) ───────────────

def run_trial(planner, eval_env, initial_qpos, initial_qvel,
              n_steps, do_render, do_save_states=False, sac_policy=None):
    set_state(eval_env, initial_qpos, initial_qvel)
    planner._prev_mean    = None
    planner._shift_elites = None
    planner._gt_init_mean = None

    step        = 0
    x_vels      = []
    actions     = []
    frames      = []
    qpos_seq    = []
    qvel_seq    = []
    terminated  = False
    truncated   = False

    # capture initial obs for SAC warm-start (reconstruct from qpos/qvel)
    current_obs = eval_env.unwrapped._get_obs() if sac_policy is not None else None

    # capture initial state
    if do_save_states:
        d = eval_env.unwrapped.data
        qpos_seq.append(d.qpos.copy().tolist())
        qvel_seq.append(d.qvel.copy().tolist())

    while step < n_steps and not (terminated or truncated):
        qpos, qvel = get_state(eval_env)

        if sac_policy is not None:
            # Tile the current SAC action as a constant mean over the horizon
            a_sac    = sac_policy.act(current_obs)               # (6,)
            sac_mean = np.tile(
                np.clip(a_sac, ACTION_LOW, ACTION_HIGH)[None],
                (planner.horizon, 1),
            ).astype(np.float32)
            planner._gt_init_mean = sac_mean
            planner._prev_mean    = None   # _prev_mean has priority; must be None
            planner._shift_elites = None   # prevent zero-action contamination

        sequence   = planner.plan(qpos, qvel)

        for a in sequence:
            if step >= n_steps or terminated or truncated:
                break
            obs, reward, terminated, truncated, info = eval_env.step(a)
            current_obs = obs
            step += 1
            actions.append(a.tolist())
            x_vels.append(float(info.get('x_velocity', 0.0)))
            if do_render:
                frames.append(eval_env.render())
            if do_save_states:
                d = eval_env.unwrapped.data
                qpos_seq.append(d.qpos.copy().tolist())
                qvel_seq.append(d.qvel.copy().tolist())

    data          = eval_env.unwrapped.data
    final_height  = float(data.qpos[1])
    final_angle   = float(data.qpos[2])
    final_x       = float(data.qpos[0])
    forward_disp  = final_x - float(initial_qpos[0])
    survived_full = (step >= n_steps) and not terminated
    avg_vel       = float(np.mean(x_vels)) if x_vels else 0.0

    if terminated:
        reason = failure_reason(data)
    elif truncated:
        reason = 'truncated'
    else:
        reason = 'timeout'

    row = {
        'survived_full':  survived_full,
        'steps':          step,
        'forward_disp':   forward_disp,
        'avg_x_velocity': avg_vel,
        'final_height':   final_height,
        'final_angle':    final_angle,
        'fail_reason':    reason,
        'x_velocities':   x_vels,
        'actions':        actions,
    }
    if do_save_states:
        row['qpos_seq'] = qpos_seq
        row['qvel_seq'] = qvel_seq
    return row, frames


# ── visualization (same as gt_cem_walker.py) ──────────────────────────────────

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


def save_frame_grid(frames, out_path, every=10, title='GT iCEM Walker2d'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    subset = frames[::every]
    n      = len(subset)
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


def pick_viz_trial(trials_data, success_mask):
    successes = [i for i, s in enumerate(success_mask) if s]
    if successes:
        return max(successes, key=lambda i: trials_data[i]['avg_x_velocity'])
    return max(range(len(trials_data)),
               key=lambda i: (trials_data[i]['steps'], trials_data[i]['forward_disp']))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='GT iCEM-MPC for Walker2d-v4 (Pinneri et al. 2020).')
    p.add_argument('--trials',              type=int,   default=5)
    p.add_argument('--n-steps',             type=int,   default=400)
    # planner
    p.add_argument('--planning-horizon',    type=int,   default=30,
                   help='MPC look-ahead steps (paper recommends 30 for GT dynamics)')
    p.add_argument('--executed-steps',      type=int,   default=10,
                   help='Steps to execute before replanning')
    # iCEM hyperparameters
    p.add_argument('--cem-population',      type=int,   default=500)
    p.add_argument('--cem-elites',          type=int,   default=50)
    p.add_argument('--cem-iters',           type=int,   default=5)
    p.add_argument('--initial-std',         type=float, default=0.5,
                   help='Initial action std (paper default 0.5 for Walker)')
    p.add_argument('--beta',                type=float, default=2.5,
                   help='Colored-noise exponent β (0=white, 2.5=recommended for Walker)')
    p.add_argument('--keep-fraction',       type=float, default=0.3,
                   help='Fraction of elites kept from each inner CEM iteration')
    p.add_argument('--shift-fraction',      type=float, default=0.3,
                   help='Fraction of elites shifted forward for next MPC call')
    p.add_argument('--sample-decay',        type=float, default=1.25,
                   help='Population decay factor γ across inner iterations')
    p.add_argument('--no-execute-best',     action='store_true',
                   help='Execute mean instead of best-seen trajectory (default: best)')
    # objective — dense per-step reward matching Gym Walker2d-v4
    p.add_argument('--wx',   type=float, default=1.0,
                   help='Per-step x-velocity weight (Gym forward_reward scale)')
    p.add_argument('--wh',   type=float, default=1.0,
                   help='Per-step healthy-alive bonus (Gym healthy_reward)')
    p.add_argument('--wu',   type=float, default=1e-3,
                   help='Per-step ctrl cost weight (Gym ctrl_cost_weight)')
    p.add_argument('--cf',   type=float, default=10.0,
                   help='Extra penalty per wasted step after fall')
    p.add_argument('--wz',     type=float, default=0.0,
                   help='Per-step height-deviation penalty')
    p.add_argument('--wang',   type=float, default=0.0,
                   help='Per-step torso-angle penalty')
    p.add_argument('--wjoint', type=float, default=0.0,
                   help='Per-step leg joint angle regularisation (penalises qpos[3:9]²)')
    p.add_argument('--wsmooth', type=float, default=0.0,
                   help='Per-step action smoothness penalty (penalises ||a_t - a_{t-1}||²)')
    p.add_argument('--height-target',       type=float, default=1.2)
    p.add_argument('--healthy-z-min',       type=float, default=0.8,
                   help='Minimum torso height before episode terminates in the planner '
                        '(raise to 1.0 to force the planner out of crawling gaits)')
    # success criterion
    p.add_argument('--success-min-velocity', type=float, default=0.5)
    # misc
    p.add_argument('--frame-skip',          type=int,   default=4)
    # SAC policy warm-start
    p.add_argument('--sac-policy-warmstart', action='store_true',
                   help='Use a pretrained SAC policy to initialise the CEM mean each step')
    p.add_argument('--sac-repo',   default='sdpkjc/Walker2d-v4-sac_continuous_action-seed4',
                   help='HuggingFace repo for SAC policy (used with --sac-policy-warmstart)')
    p.add_argument('--sac-ckpt',   default=None,
                   help='Local checkpoint for SAC policy (alternative to --sac-repo)')
    # rendering / output
    p.add_argument('--render-dir',          default='')
    p.add_argument('--render-every',        type=int,   default=10)
    p.add_argument('--gif-fps',             type=int,   default=30)
    p.add_argument('--save-states',         action='store_true',
                   help='Store qpos_seq and qvel_seq per trial (needed for render_task_frames.py)')
    p.add_argument('--seed',                type=int,   default=42)
    p.add_argument('--output',              required=True)
    args = p.parse_args()

    # Override module-level health bound so both is_healthy() and failure_reason()
    # use the user-supplied floor.  Must happen before any env is created.
    global HEALTHY_Z_MIN
    HEALTHY_Z_MIN = args.healthy_z_min

    np.random.seed(args.seed)

    plan_env = gym.make('Walker2d-v4')
    eval_env = gym.make('Walker2d-v4',
                        render_mode='rgb_array' if args.render_dir else None)
    plan_env.reset(seed=args.seed)
    eval_env.reset(seed=args.seed)

    planner = WalkerGTiCEM(
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
        plan_env       = plan_env,
        frame_skip     = args.frame_skip,
        wx             = args.wx,
        wh             = args.wh,
        wu             = args.wu,
        cf             = args.cf,
        wz             = args.wz,
        wang           = args.wang,
        height_target  = args.height_target,
        wjoint         = args.wjoint,
        wsmooth        = args.wsmooth,
        execute_best   = not args.no_execute_best,
    )

    # ── SAC policy warm-start ─────────────────────────────────────────────────
    sac_policy = None
    if args.sac_policy_warmstart:
        from experiments.walker2d_ppo_utils import download_and_load_sac, load_sac_from_local
        print('[GT-iCEM] Loading SAC policy …')
        sac_policy = (load_sac_from_local(args.sac_ckpt)
                      if args.sac_ckpt else
                      download_and_load_sac(args.sac_repo, device='cpu'))
        print('[GT-iCEM] SAC policy loaded.')

    print(f'[GT-iCEM Walker2d] H={args.planning_horizon} '
          f'pop={args.cem_population} elites={args.cem_elites} '
          f'iters={args.cem_iters}  exec={args.executed_steps}')
    print(f'  iCEM: β={args.beta}  keep={args.keep_fraction}  '
          f'shift={args.shift_fraction}  decay={args.sample_decay}  '
          f'std₀={args.initial_std}  execute_best={not args.no_execute_best}')
    print(f'  obj:  wx={args.wx}  wh={args.wh}  wu={args.wu}  cf={args.cf}  '
          f'wz={args.wz}  wang={args.wang}  h*={args.height_target}')
    print(f'  cost = -sum(wx*xvel + wh - wu*||u||²) per step  +  cf*(H-T_alive)')
    print(f'  health bounds: z ∈ ({HEALTHY_Z_MIN:.2f}, {HEALTHY_Z_MAX:.2f})  |ang| < {HEALTHY_ANG_MAX:.2f}')
    print(f'  SAC warm-start: {args.sac_policy_warmstart}')
    print(f'  success: survived_full AND avg_vel > {args.success_min_velocity} m/s')

    trials_data  = []
    frames_list  = []
    success_mask = []

    for i in range(args.trials):
        np.random.seed(args.seed + i)
        obs, _ = eval_env.reset(seed=args.seed + i)
        initial_qpos, initial_qvel = get_state(eval_env)
        plan_env.reset(seed=args.seed + i)
        set_state(plan_env, initial_qpos, initial_qvel)

        print(f'[trial {i:03d}] ', end='', flush=True)
        t0 = time.time()

        row, frames = run_trial(
            planner, eval_env, initial_qpos, initial_qvel,
            args.n_steps, bool(args.render_dir),
            do_save_states=args.save_states,
            sac_policy=sac_policy)

        elapsed = time.time() - t0
        success = row['survived_full'] and row['avg_x_velocity'] >= args.success_min_velocity
        row['success'] = success
        success_mask.append(success)
        trials_data.append(row)
        frames_list.append(frames)

        status = 'SUCCESS' if success else f'FAIL({row["fail_reason"]})'
        print(f'{status}  survived={row["survived_full"]}  '
              f'steps={row["steps"]}  disp={row["forward_disp"]:.2f}m  '
              f'avg_vel={row["avg_x_velocity"]:.3f}m/s  '
              f'h={row["final_height"]:.3f}  ang={row["final_angle"]:.3f}  '
              f't={elapsed:.1f}s')

    n_success = sum(success_mask)
    sr        = n_success / max(args.trials, 1)
    mean_vel  = float(np.mean([r['avg_x_velocity'] for r in trials_data]))
    mean_disp = float(np.mean([r['forward_disp']   for r in trials_data]))

    print(f'\nSuccess: {n_success}/{args.trials} ({sr:.1%})')
    print(f'Mean avg_x_velocity: {mean_vel:.3f} m/s  |  '
          f'Mean forward disp: {mean_disp:.2f} m')

    reasons = [r['fail_reason'] for r in trials_data if not r['success']]
    if reasons:
        from collections import Counter
        print('Failure breakdown:', dict(Counter(reasons)))

    if args.render_dir:
        viz_idx    = pick_viz_trial(trials_data, success_mask)
        viz_frames = frames_list[viz_idx]
        row        = trials_data[viz_idx]
        tag        = 'success' if success_mask[viz_idx] else 'best_failed'
        stem       = f'{tag}_trial_{viz_idx:03d}'
        title      = (f'GT iCEM Walker2d — trial {viz_idx} [{tag}] '
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
            'planner':         'gt_icem_mpc',
            'env':             'Walker2d-v4',
            'n_steps':         args.n_steps,
            'horizon':         args.planning_horizon,
            'executed_steps':  args.executed_steps,
            'population':      args.cem_population,
            'elites':          args.cem_elites,
            'iterations':      args.cem_iters,
            'initial_std':     args.initial_std,
            'beta':            args.beta,
            'keep_fraction':   args.keep_fraction,
            'shift_fraction':  args.shift_fraction,
            'sample_decay':    args.sample_decay,
            'execute_best':    not args.no_execute_best,
            'wx':              args.wx,
            'wh':              args.wh,
            'wu':              args.wu,
            'cf':              args.cf,
            'wz':              args.wz,
            'wang':            args.wang,
            'height_target':   args.height_target,
            'frame_skip':      args.frame_skip,
            'success_min_vel': args.success_min_velocity,
            'seed':            args.seed,
            'n_trials':        args.trials,
        },
        'success_rate':      sr,
        'n_success':         n_success,
        'mean_avg_velocity': mean_vel,
        'mean_forward_disp': mean_disp,
        'trials':            trials_data,
    }, indent=2))
    print(f'[done] {out}')

    plan_env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
