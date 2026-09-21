#!/usr/bin/env python
"""GT CEM-MPC for Walker2d — faithful reimplementation of mbbl.

Reference: https://github.com/WilsonWangTHU/mbbl
  sampler/singletask_pets_sampler.py   (CEM loop)
  worker/cem_worker.py                 (parallel rollout, num_workers=10)
  network/dynamics/groundtruth_forward_dynamics.py
  network/reward/groundtruth_reward.py (Walker2d reward)
  config/cem_config.py + pets_gt.sh    (Walker2d defaults)

Algorithm (exact match to mbbl):
  - Reward per step: x_vel - 3*(height-1.3)^2 - 0.1*||u||^2 + 1.0
  - True MuJoCo dynamics: set state + step (no early termination, check_done=0)
  - CEM sampling: truncated normal (-2,+2), variance constrained to bounds
  - Distribution update: momentum blend (alpha=0.1) on both mean and variance
  - Stopping criterion: stop early if max(var) <= 0.001
  - Warm-start: shift mean forward by ACTION_DIM each env step
  - Parallel workers: each holds its own MuJoCo env (mbbl: num_workers=10)

Walker2d defaults match mbbl shell script (pets_gt.sh):
  population=500, horizon=50, iters=5, elite_frac=0.1, alpha=0.1, var0=0.25

Evaluation protocol (mbbl):
  - Run for max_timesteps=20000 total env steps
  - Auto-reset episode when terminated or truncated
  - Report average return per episode over all completed episodes

Usage
-----
  python experiments/gt_cem_walker_mbbl.py \\
      --max-timesteps 20000 --num-workers 10 \\
      --output results/gt_cem_walker_mbbl_seed42.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
try:
    import gym as _gym_module   # old gym + mujoco_py (mbbl environment)
    _OLD_GYM    = True
    _GYM_ENV_ID = 'Walker2d-v3'
except ImportError:
    import gymnasium as _gym_module
    _OLD_GYM    = False
    _GYM_ENV_ID = 'Walker2d-v4'
from scipy import stats

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


# ── thin compat wrappers so the rest of the code is API-agnostic ──────────────

def _make_env(render_rgb=False):
    """Create Walker2d env, handling old gym vs gymnasium API differences."""
    if _OLD_GYM:
        return _gym_module.make(_GYM_ENV_ID)   # render_mode not supported
    else:
        kw = {'render_mode': 'rgb_array'} if render_rgb else {}
        return _gym_module.make(_GYM_ENV_ID, **kw)

def _env_reset(env, seed=None):
    """Reset env; return obs only (normalises old gym vs gymnasium)."""
    result = env.reset(seed=seed) if seed is not None else env.reset()
    # gym 0.26+ returns (obs, info); older gym and gymnasium both handled here
    if isinstance(result, tuple) and len(result) == 2 and not isinstance(result[0], np.ndarray):
        return result[0]   # unlikely edge case
    if isinstance(result, tuple) and len(result) == 2:
        return result[0]   # (obs, info) from gym 0.26+ or gymnasium
    return result           # plain obs from old gym <0.26

def _env_step(env, action):
    """Step env; return (obs, rew, terminated, truncated, info) always."""
    result = env.step(action)
    if len(result) == 4:                       # old gym: (obs, rew, done, info)
        obs, rew, done, info = result
        return obs, rew, done, False, info
    return result                              # gymnasium: 5-tuple already

def _env_render(env):
    """Render frame; return RGB array."""
    if _OLD_GYM:
        return env.render(mode='rgb_array')
    return env.render()

ACTION_DIM  = 6
ACTION_LOW  = -1.0
ACTION_HIGH =  1.0


# ── per-process worker env (persistent across calls) ──────────────────────────

_worker_env = None

def _worker_init(seed):
    global _worker_env
    os.environ.setdefault('MUJOCO_GL', 'egl')
    _worker_env = _make_env()
    _env_reset(_worker_env, seed=int(seed))


# ── mbbl reward (groundtruth_reward.py + walker.py) ───────────────────────────

def _mbbl_reward(data, action):
    """x_vel - 3*(height-1.3)^2 - 0.1*||u||^2 + 1.0"""
    height = float(data.qpos[1])
    x_vel  = float(data.qvel[0])
    ctrl   = float(np.sum(np.asarray(action, dtype=float) ** 2))
    return x_vel - 3.0 * (height - 1.3) ** 2 - 0.1 * ctrl + 1.0


# ── single trajectory rollout (check_done=0, no early termination) ────────────

def _rollout(env, qpos, qvel, actions):
    env.unwrapped.set_state(qpos, qvel)
    data  = env.unwrapped.data
    total = 0.0
    for t in range(len(actions)):
        a = np.clip(actions[t], ACTION_LOW, ACTION_HIGH)
        total += _mbbl_reward(data, a)
        env.step(a)      # works with both mujoco_py and new mujoco backends
    return total


# ── worker function: evaluate a batch of trajectories ─────────────────────────

def _eval_batch(args):
    """Runs in a worker process; _worker_env is set by _worker_init."""
    qpos, qvel, samples_2d, horizon = args   # samples_2d: (batch, sol_dim)
    costs = []
    for row in samples_2d:
        actions = row.reshape(horizon, ACTION_DIM)
        costs.append(-_rollout(_worker_env, qpos, qvel, actions))
    return costs


# ── CEM planner (singletask_pets_sampler._act) ────────────────────────────────

class MbblGTCEM:
    """GT CEM matching mbbl exactly, with parallel trajectory evaluation."""

    def __init__(self, horizon, population, elites, iterations,
                 initial_variance, alpha, pool, n_workers):
        self.horizon  = int(horizon)
        self.sol_dim  = horizon * ACTION_DIM
        self.pop      = int(population)
        self.elites   = int(elites)
        self.iters    = int(iterations)
        self.var0     = float(initial_variance)
        self.alpha    = float(alpha)
        self.pool     = pool
        self.n_workers = int(n_workers)

        self._lb = np.full(self.sol_dim, ACTION_LOW)
        self._ub = np.full(self.sol_dim, ACTION_HIGH)

        # scalar standard TN(-2,2); mbbl initialises with zeros_like(mean)/ones_like(mean)
        # but those are all-zero/all-one arrays → equivalent to scalar TN(-2,2).
        self._X    = stats.truncnorm(-2, 2)
        self._mean = np.zeros(self.sol_dim)

    def reset(self):
        self._mean = np.zeros(self.sol_dim)

    def _shift(self, mean):
        out = np.empty_like(mean)
        out[:-ACTION_DIM] = mean[ACTION_DIM:]
        out[-ACTION_DIM:] = 0.0
        return out

    def plan(self, qpos, qvel):
        mean = self._mean.copy()
        var  = np.full(self.sol_dim, self.var0)

        t = 0
        while t < self.iters and np.max(var) > 0.001:
            # constrained variance (mbbl logic)
            lb_dist = mean - self._lb
            ub_dist = self._ub - mean
            cvar = np.minimum(
                np.minimum(np.square(lb_dist / 2.0),
                           np.square(ub_dist / 2.0)),
                var)

            # sample from TN(-2,2) then scale+shift (mbbl exact)
            noise   = self._X.rvs(size=[self.pop, self.sol_dim])
            samples = noise * np.sqrt(cvar) + mean   # (pop, sol_dim)

            # split work across workers
            batches = np.array_split(samples, self.n_workers)
            args = [(qpos, qvel, b, self.horizon) for b in batches]

            if self.pool is not None:
                results = self.pool.map(_eval_batch, args)
            else:
                results = [_eval_batch(a) for a in args]

            costs = np.array([c for batch in results for c in batch])

            # elite selection + momentum update (mbbl exact)
            elite_idx = np.argsort(costs)[:self.elites]
            elites    = samples[elite_idx]
            new_mean  = elites.mean(0)
            new_var   = elites.var(0)
            mean = self.alpha * mean + (1.0 - self.alpha) * new_mean
            var  = self.alpha * var  + (1.0 - self.alpha) * new_var
            t += 1

        self._mean = self._shift(mean)
        return np.clip(mean[:ACTION_DIM], ACTION_LOW, ACTION_HIGH)


# ── GIF rendering: replay saved actions with rgb_array mode ──────────────────

def render_episode_gif(actions, seed, ep_idx, gif_path, fps=30):
    """Replay a recorded action sequence and save as GIF."""
    if not HAS_IMAGEIO:
        print('[gif] imageio not installed — skipping GIF render')
        return
    render_env = _make_env(render_rgb=True)
    _env_reset(render_env, seed=seed)
    frames = [_env_render(render_env)]
    for a in actions:
        _, _, terminated, truncated, _ = _env_step(render_env, a)
        frames.append(_env_render(render_env))
        if terminated or truncated:
            break
    render_env.close()
    Path(gif_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(gif_path, frames, fps=fps)
    print(f'[gif] ep {ep_idx:04d} saved → {gif_path}  ({len(frames)} frames)')


# ── mbbl evaluation: run for max_timesteps, auto-reset, avg return/episode ───

def run_evaluation(planner, eval_env, seed, max_timesteps):
    """Run for max_timesteps total steps; auto-reset episodes. Return stats."""
    _env_reset(eval_env, seed=seed)
    planner.reset()

    total_steps     = 0
    ep_return       = 0.0
    ep_steps        = 0
    ep_actions      = []          # actions taken this episode (for GIF replay)
    episode_returns = []
    episode_lengths = []
    episode_actions = []          # one list of actions per episode

    # track episode seeds so we can replay deterministically
    ep_seeds        = [seed]
    ep_idx          = 0
    t_ep_start      = time.time()

    print(f'[ep {ep_idx:04d}] start', flush=True)

    while total_steps < max_timesteps:
        qpos = eval_env.unwrapped.data.qpos.copy()
        qvel = eval_env.unwrapped.data.qvel.copy()
        action = planner.plan(qpos, qvel)
        ep_actions.append(action.copy())

        _, rew, terminated, truncated, _ = _env_step(eval_env, action)
        ep_return  += float(rew)
        ep_steps   += 1
        total_steps += 1

        done = terminated or truncated

        if done or total_steps >= max_timesteps:
            episode_returns.append(ep_return)
            episode_lengths.append(ep_steps)
            episode_actions.append(ep_actions)
            elapsed = time.time() - t_ep_start
            print(f'[ep {ep_idx:04d}] '
                  f'steps={ep_steps}  return={ep_return:.1f}  '
                  f'total_steps={total_steps}  t={elapsed:.1f}s', flush=True)

            if total_steps < max_timesteps:
                ep_idx     += 1
                ep_return   = 0.0
                ep_steps    = 0
                ep_actions  = []
                _env_reset(eval_env)
                ep_seeds.append(None)    # subsequent episodes use env's own rng
                planner.reset()          # mbbl resets warm-start each episode
                t_ep_start = time.time()
                print(f'[ep {ep_idx:04d}] start', flush=True)

    return {
        'total_timesteps': total_steps,
        'n_episodes':      len(episode_returns),
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
        'episode_actions': episode_actions,
        'episode_seeds':   ep_seeds,
        'mean_return':     float(np.mean(episode_returns)) if episode_returns else 0.0,
        'std_return':      float(np.std(episode_returns))  if episode_returns else 0.0,
        'mean_ep_length':  float(np.mean(episode_lengths)) if episode_lengths else 0.0,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='GT CEM for Walker2d — mbbl faithful reimplementation.')
    p.add_argument('--max-timesteps',         type=int,   default=20000,
                   help='Total env steps (mbbl: 20000)')
    # CEM — mbbl Walker2d defaults
    p.add_argument('--planning-horizon',      type=int,   default=50)
    p.add_argument('--cem-population',        type=int,   default=500)
    p.add_argument('--cem-elite-fraction',    type=float, default=0.1)
    p.add_argument('--cem-iters',             type=int,   default=5)
    p.add_argument('--cem-initial-variance',  type=float, default=0.25)
    p.add_argument('--cem-alpha',             type=float, default=0.1,
                   help='Momentum coefficient (mbbl: 0.1)')
    # parallelism
    p.add_argument('--num-workers',           type=int,   default=10,
                   help='Parallel rollout workers (mbbl: 10)')
    p.add_argument('--seed',                  type=int,   default=42)
    p.add_argument('--output',                required=True)
    p.add_argument('--gif',                   default=None,
                   help='Path to save a GIF of the longest episode (e.g. results/best.gif)')
    args = p.parse_args()

    np.random.seed(args.seed)

    n_elites = max(1, int(args.cem_population * args.cem_elite_fraction))
    print(f'[mbbl GT-CEM {_GYM_ENV_ID}]')
    print(f'  H={args.planning_horizon}  pop={args.cem_population}  '
          f'elites={n_elites}  iters={args.cem_iters}  '
          f'var0={args.cem_initial_variance}  alpha={args.cem_alpha}  '
          f'workers={args.num_workers}')
    print(f'  reward: x_vel - 3*(h-1.3)^2 - 0.1*||u||^2 + 1  (mbbl Walker2d)')
    print(f'  no early termination during planning (check_done=0)')
    print(f'  evaluation: {args.max_timesteps} total timesteps, avg return/episode')

    # spawn worker pool with one env per worker
    ctx  = mp.get_context('spawn')
    pool = ctx.Pool(processes=args.num_workers,
                    initializer=_worker_init,
                    initargs=(args.seed,))

    eval_env = _make_env()

    planner = MbblGTCEM(
        horizon          = args.planning_horizon,
        population       = args.cem_population,
        elites           = n_elites,
        iterations       = args.cem_iters,
        initial_variance = args.cem_initial_variance,
        alpha            = args.cem_alpha,
        pool             = pool,
        n_workers        = args.num_workers,
    )

    t0    = time.time()
    stats = run_evaluation(planner, eval_env, args.seed, args.max_timesteps)
    wall  = time.time() - t0

    pool.close()
    pool.join()
    eval_env.close()

    # render GIF of the longest episode
    if args.gif and stats['episode_actions']:
        best_idx = int(np.argmax(stats['episode_lengths']))
        render_episode_gif(
            actions  = stats['episode_actions'][best_idx],
            seed     = stats['episode_seeds'][best_idx] if stats['episode_seeds'][best_idx] is not None else args.seed,
            ep_idx   = best_idx,
            gif_path = args.gif,
        )

    print(f'\n[done] {args.max_timesteps} steps in {wall:.1f}s')
    print(f'  episodes:       {stats["n_episodes"]}')
    print(f'  mean return:    {stats["mean_return"]:.1f} ± {stats["std_return"]:.1f}')
    print(f'  mean ep length: {stats["mean_ep_length"]:.1f}')

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'protocol': {
            'planner':              'gt_cem_mbbl',
            'env':                  _GYM_ENV_ID,
            'max_timesteps':        args.max_timesteps,
            'planning_horizon':     args.planning_horizon,
            'cem_population':       args.cem_population,
            'cem_elites':           n_elites,
            'cem_elite_fraction':   args.cem_elite_fraction,
            'cem_iters':            args.cem_iters,
            'cem_initial_variance': args.cem_initial_variance,
            'cem_alpha':            args.cem_alpha,
            'num_workers':          args.num_workers,
            'reward':               'x_vel - 3*(h-1.3)^2 - 0.1*||u||^2 + 1.0',
            'check_done':           0,
            'warm_start':           True,
            'warm_start_reset':     'per_episode',
            'seed':                 args.seed,
        },
        'mean_return':    stats['mean_return'],
        'std_return':     stats['std_return'],
        'n_episodes':     stats['n_episodes'],
        'mean_ep_length': stats['mean_ep_length'],
        'total_timesteps': stats['total_timesteps'],
        'wall_seconds':   wall,
        'episode_returns': stats['episode_returns'],
        'episode_lengths': stats['episode_lengths'],
        'best_episode_idx': int(np.argmax(stats['episode_lengths'])) if stats['episode_lengths'] else None,
    }, indent=2))
    print(f'[saved] {out}')


if __name__ == '__main__':
    main()
