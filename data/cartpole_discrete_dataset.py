"""Reproducible dataset-generation pipeline for discrete CartPole-v1.

Collects offline trajectories from a mixture of behavior policies:
  20%  clean expert  (LQR-thresholded, ε=0)
  20%  expert + action-flip  ε=0.05
  25%  expert + action-flip  ε=0.10
  20%  expert + action-flip  ε=0.20
  10%  expert + burst noise
   5%  fully random policy

Pixel observations come from env.render() (400×600 RGB, resized to image_size).
The native CartPole state (x, ẋ, θ, θ̇) is stored alongside for probing only.

Output
------
{output_dir}/
  train.hdf5            episode-centric HDF5
  val.hdf5
  test.hdf5
  metadata.json         aggregate statistics
  generation_config.yaml
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import h5py
import numpy as np
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Per-action source labels ──────────────────────────────────────────────────
SRC_EXPERT  = np.uint8(0)   # expert action, no noise
SRC_FLIPPED = np.uint8(1)   # action-flip noise
SRC_BURST   = np.uint8(2)   # burst-noise action
SRC_RANDOM  = np.uint8(3)   # fully random

SRC_NAMES = {0: 'expert', 1: 'flipped', 2: 'burst', 3: 'random'}

# ── Policy schedule (policy_type, epsilon, config_frac_attr) ─────────────────
_SCHEDULE = [
    ('expert',  0.00, 'frac_expert'),
    ('noisy',   0.05, 'frac_noisy_005'),
    ('noisy',   0.10, 'frac_noisy_010'),
    ('noisy',   0.20, 'frac_noisy_020'),
    ('burst',   0.00, 'frac_burst'),
    ('random',  0.00, 'frac_random'),
]


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    # Target
    num_transitions: int = 200_000

    # Image
    image_size: int = 64
    image_crop: bool = False

    # Behavior-policy mixture (fractions; must sum to 1.0)
    frac_expert:    float = 0.20
    frac_noisy_005: float = 0.20
    frac_noisy_010: float = 0.25
    frac_noisy_020: float = 0.20
    frac_burst:     float = 0.10
    frac_random:    float = 0.05

    # Burst-noise parameters
    burst_start_prob: float = 0.03
    burst_min_len:    int   = 1
    burst_max_len:    int   = 4

    # Initial-state distribution
    custom_reset:     bool  = True
    cart_pos_range:   float = 0.50
    cart_vel_range:   float = 0.50
    pole_angle_range: float = 0.10
    pole_vel_range:   float = 0.50

    # Split fractions (by episode count)
    train_frac: float = 0.80
    val_frac:   float = 0.10

    # Output
    output_dir: str  = 'data/cartpole_visual'
    seed:       int  = 0
    resume:     bool = False

    # Set at runtime
    gymnasium_version:    str = ''
    generation_timestamp: str = ''

    @classmethod
    def from_yaml(cls, path: str) -> 'DatasetConfig':
        with open(path) as f:
            d = yaml.safe_load(f)
        cfg = cls()
        for k, v in (d or {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, type(getattr(cfg, k))(v))
        return cfg

    def validate(self) -> None:
        total = (self.frac_expert + self.frac_noisy_005 + self.frac_noisy_010
                 + self.frac_noisy_020 + self.frac_burst + self.frac_random)
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f'Policy fractions must sum to 1.0, got {total:.5f}')
        if not (16 <= self.image_size <= 1024):
            raise ValueError(f'image_size must be in [16, 1024], got {self.image_size}')
        if not (0.0 < self.train_frac < 1.0):
            raise ValueError('train_frac must be in (0, 1)')
        if self.train_frac + self.val_frac >= 1.0:
            raise ValueError('train_frac + val_frac must be < 1.0')

    def to_dict(self) -> dict:
        return asdict(self)


# ── Expert policy ─────────────────────────────────────────────────────────────

def _compute_expert_gain() -> np.ndarray:
    """LQR gain for CartPole-v1, thresholded to binary action.

    Reuses _compute_lqr_gain() from data/dataset.py which matches CartPole-v1
    default physical parameters exactly (M=1, m=0.1, l=0.5, g=9.8, dt=0.02).
    Returns gain of shape (4,) where u = dot(gain, state):
      u >= 0 → action=1 (push right)
      u  < 0 → action=0 (push left)
    """
    try:
        here = Path(__file__).resolve().parent.parent
        if str(here) not in sys.path:
            sys.path.insert(0, str(here))
        from data.dataset import _compute_lqr_gain
        gain = _compute_lqr_gain().squeeze()
        logger.info('Expert LQR gain: %s', np.round(gain, 3))
        return gain.astype(np.float64)
    except Exception as exc:
        logger.warning('LQR gain failed (%s); using pole-placement fallback.', exc)
        return np.array([0.0, 0.0, 10.0, 0.0], dtype=np.float64)


def _expert_action(state: np.ndarray, gain: np.ndarray) -> int:
    """Binary action from LQR: action=1 if u>=0, else action=0."""
    return 1 if float(np.dot(gain, state)) >= 0.0 else 0


# ── Noisy policy wrappers ─────────────────────────────────────────────────────

def _noisy_action(state: np.ndarray, gain: np.ndarray,
                  epsilon: float, rng: np.random.Generator
                  ) -> Tuple[int, np.uint8]:
    """Expert action with i.i.d. action-flip probability epsilon."""
    expert = _expert_action(state, gain)
    if rng.random() < epsilon:
        return 1 - expert, SRC_FLIPPED
    return expert, SRC_EXPERT


class BurstNoisePolicy:
    """Expert policy with random-burst interruptions.

    At each step, starts a burst with probability burst_start_prob.
    During a burst of length Uniform[min, max]:
      50% chance: hold opposite-of-expert; 50% chance: random binary action.
    """

    def __init__(self, gain: np.ndarray, burst_start_prob: float,
                 burst_min: int, burst_max: int, rng: np.random.Generator):
        self.gain = gain
        self.burst_start_prob = burst_start_prob
        self.burst_min = burst_min
        self.burst_max = burst_max
        self.rng = rng
        self._remaining: int = 0
        self._burst_action: int = 0

    def act(self, state: np.ndarray) -> Tuple[int, np.uint8]:
        expert = _expert_action(state, self.gain)
        if self._remaining > 0:
            self._remaining -= 1
            return self._burst_action, SRC_BURST
        if self.rng.random() < self.burst_start_prob:
            length = int(self.rng.integers(self.burst_min, self.burst_max + 1))
            self._remaining = length - 1
            self._burst_action = (1 - expert if self.rng.random() < 0.5
                                  else int(self.rng.integers(0, 2)))
            return self._burst_action, SRC_BURST
        return expert, SRC_EXPERT

    def reset(self) -> None:
        self._remaining = 0


# ── Environment helpers ───────────────────────────────────────────────────────

def _make_env(seed: int) -> gym.Env:
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    env.reset(seed=seed)
    return env


def _preprocess_obs(frame: np.ndarray, cfg: DatasetConfig) -> np.ndarray:
    """Resize raw render frame (H, W, 3) → (image_size, image_size, 3) uint8."""
    S = cfg.image_size
    if frame.shape[0] == S and frame.shape[1] == S:
        return frame.astype(np.uint8)
    try:
        import cv2
        return cv2.resize(frame, (S, S), interpolation=cv2.INTER_AREA).astype(np.uint8)
    except ImportError:
        src_h, src_w = frame.shape[:2]
        r = (np.arange(S) * src_h // S).astype(int)
        c = (np.arange(S) * src_w // S).astype(int)
        return frame[np.ix_(r, c)].astype(np.uint8)


def _reset_env(env: gym.Env, cfg: DatasetConfig,
               rng: np.random.Generator, ep_seed: int
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Reset env, optionally apply custom state, return (obs_img, state)."""
    env.reset(seed=ep_seed)
    if cfg.custom_reset:
        state = np.array([
            rng.uniform(-cfg.cart_pos_range,   cfg.cart_pos_range),
            rng.uniform(-cfg.cart_vel_range,    cfg.cart_vel_range),
            rng.uniform(-cfg.pole_angle_range,  cfg.pole_angle_range),
            rng.uniform(-cfg.pole_vel_range,    cfg.pole_vel_range),
        ], dtype=np.float64)
        env.unwrapped.state = state.copy()
    else:
        state = env.unwrapped.state.copy()
    obs = _preprocess_obs(env.render(), cfg)
    return obs, state.astype(np.float32)


# ── Episode collection ────────────────────────────────────────────────────────

def _collect_episode(
    env: gym.Env,
    policy_type: str,
    epsilon: float,
    cfg: DatasetConfig,
    rng: np.random.Generator,
    ep_seed: int,
    gain: np.ndarray,
) -> Dict:
    """Collect one complete episode.

    Runs until termination or truncation (max 500 steps for CartPole-v1).
    Returns a dict with arrays and scalar metadata.
    """
    obs, state = _reset_env(env, cfg, rng, ep_seed)

    obs_list:    List[np.ndarray] = [obs]
    state_list:  List[np.ndarray] = [state]
    action_list: List[int]        = []
    reward_list: List[float]      = []
    term_list:   List[bool]       = []
    trunc_list:  List[bool]       = []
    source_list: List[np.uint8]   = []
    step_list:   List[int]        = []

    burst_pol: Optional[BurstNoisePolicy] = None
    if policy_type == 'burst':
        burst_pol = BurstNoisePolicy(
            gain, cfg.burst_start_prob,
            cfg.burst_min_len, cfg.burst_max_len, rng)

    t = 0
    done = False
    while not done:
        if policy_type == 'expert':
            action, src = _expert_action(state, gain), SRC_EXPERT
        elif policy_type == 'noisy':
            action, src = _noisy_action(state, gain, epsilon, rng)
        elif policy_type == 'burst':
            action, src = burst_pol.act(state)
        else:
            action, src = int(rng.integers(0, 2)), SRC_RANDOM

        _, reward, terminated, truncated, _ = env.step(action)
        next_state = env.unwrapped.state.astype(np.float32)
        next_obs   = _preprocess_obs(env.render(), cfg)

        action_list.append(action)
        reward_list.append(float(reward))
        term_list.append(bool(terminated))
        trunc_list.append(bool(truncated))
        source_list.append(src)
        step_list.append(t)
        obs_list.append(next_obs)
        state_list.append(next_state)

        state = next_state
        t    += 1
        done  = terminated or truncated

    T = len(action_list)
    return {
        'observations':  np.stack(obs_list).astype(np.uint8),       # (T+1, S, S, 3)
        'actions':       np.array(action_list,  dtype=np.uint8),     # (T,)
        'states':        np.stack(state_list).astype(np.float32),    # (T+1, 4)
        'rewards':       np.array(reward_list,  dtype=np.float32),   # (T,)
        'terminated':    np.array(term_list,    dtype=np.bool_),     # (T,)
        'truncated':     np.array(trunc_list,   dtype=np.bool_),     # (T,)
        'action_source': np.array(source_list,  dtype=np.uint8),     # (T,)
        'timesteps':     np.array(step_list,    dtype=np.int32),     # (T,)
        'policy_type':   policy_type,
        'epsilon':       float(epsilon),
        'ep_seed':       int(ep_seed),
        'length':        T,
        'success':       bool(trunc_list[-1]) if trunc_list else False,
    }


# ── Policy assignment ─────────────────────────────────────────────────────────

def _assign_policies(n: int, cfg: DatasetConfig,
                     rng: np.random.Generator) -> List[Tuple[str, float]]:
    """Return a shuffled list of (policy_type, epsilon) for n episodes."""
    assignments: List[Tuple[str, float]] = []
    for pt, eps, attr in _SCHEDULE:
        count = max(1, round(getattr(cfg, attr) * n))
        assignments.extend([(pt, eps)] * count)

    # Trim/pad to exactly n
    while len(assignments) < n:
        dominant = max(_SCHEDULE, key=lambda s: getattr(cfg, s[2]))
        assignments.append((dominant[0], dominant[1]))
    assignments = assignments[:n]
    rng.shuffle(assignments)
    return assignments


# ── Expert verification ───────────────────────────────────────────────────────

def _verify_expert(gain: np.ndarray, cfg: DatasetConfig,
                   rng: np.random.Generator, n_trials: int = 10) -> None:
    """Run n_trials expert episodes and report success rate."""
    logger.info('Verifying expert policy (%d trials)...', n_trials)
    env = _make_env(cfg.seed + 99999)
    successes, lengths = 0, []
    for i in range(n_trials):
        ep = _collect_episode(env, 'expert', 0.0, cfg, rng,
                              ep_seed=cfg.seed + 99999 + i, gain=gain)
        if ep['success']:
            successes += 1
        lengths.append(ep['length'])
    env.close()
    rate = successes / n_trials
    logger.info('Expert: %d/%d success  avg_len=%.0f  min=%d  max=%d',
                successes, n_trials, np.mean(lengths), min(lengths), max(lengths))
    if rate < 0.5:
        logger.warning(
            'Expert success rate %.0f%% is low — LQR thresholding may be misaligned.',
            rate * 100)


# ── HDF5 writer ───────────────────────────────────────────────────────────────

def _write_hdf5(path: Path, episodes: List[Dict], ep_id_offset: int = 0) -> None:
    """Write episodes to an HDF5 file organised by episode group."""
    with h5py.File(path, 'w') as f:
        grp = f.require_group('episodes')
        for local_i, ep in enumerate(episodes):
            g = grp.create_group(str(ep_id_offset + local_i))
            opts = dict(compression='gzip', compression_opts=4)
            g.create_dataset('observations',  data=ep['observations'],  **opts)
            g.create_dataset('actions',       data=ep['actions'],       **opts)
            g.create_dataset('states',        data=ep['states'],        **opts)
            g.create_dataset('rewards',       data=ep['rewards'],       **opts)
            g.create_dataset('terminated',    data=ep['terminated'],    **opts)
            g.create_dataset('truncated',     data=ep['truncated'],     **opts)
            g.create_dataset('action_source', data=ep['action_source'], **opts)
            g.create_dataset('timesteps',     data=ep['timesteps'],     **opts)
            g.attrs['policy_type'] = ep['policy_type']
            g.attrs['epsilon']     = ep['epsilon']
            g.attrs['ep_seed']     = ep['ep_seed']
            g.attrs['length']      = ep['length']
            g.attrs['success']     = ep['success']
            g.attrs['episode_id']  = ep_id_offset + local_i

        # File-level summary attrs
        lengths   = [ep['length']  for ep in episodes]
        successes = [ep['success'] for ep in episodes]
        f.attrs['n_episodes']    = len(episodes)
        f.attrs['n_transitions'] = int(sum(lengths))
        f.attrs['ep_len_mean']   = float(np.mean(lengths))   if lengths else 0.0
        f.attrs['ep_len_min']    = int(np.min(lengths))      if lengths else 0
        f.attrs['ep_len_max']    = int(np.max(lengths))      if lengths else 0
        f.attrs['success_rate']  = float(np.mean(successes)) if successes else 0.0


# ── Episode split ─────────────────────────────────────────────────────────────

def _split_episodes(
    episodes: List[Dict],
    cfg: DatasetConfig,
    rng: np.random.Generator,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Stratified split by (policy_type, epsilon) to preserve mixture.

    Episodes within each stratum are shuffled before splitting so that
    train/val/test seeds never overlap within a stratum.
    """
    by_stratum: Dict[str, List[int]] = {}
    for i, ep in enumerate(episodes):
        key = f"{ep['policy_type']}_{ep['epsilon']:.3f}"
        by_stratum.setdefault(key, []).append(i)

    train_idx: List[int] = []
    val_idx:   List[int] = []
    test_idx:  List[int] = []

    for idxs in by_stratum.values():
        arr = np.array(idxs)
        rng.shuffle(arr)
        n       = len(arr)
        n_train = max(1, int(n * cfg.train_frac))
        n_val   = max(1, int(n * cfg.val_frac))
        train_idx.extend(arr[:n_train].tolist())
        val_idx.extend(arr[n_train:n_train + n_val].tolist())
        test_idx.extend(arr[n_train + n_val:].tolist())

    return ([episodes[i] for i in train_idx],
            [episodes[i] for i in val_idx],
            [episodes[i] for i in test_idx])


# ── Metadata ──────────────────────────────────────────────────────────────────

def _build_metadata(
    all_eps:   List[Dict],
    train_eps: List[Dict],
    val_eps:   List[Dict],
    test_eps:  List[Dict],
    cfg: DatasetConfig,
) -> Dict:
    lengths  = [ep['length']  for ep in all_eps]
    terms    = [bool(ep['terminated'][-1]) for ep in all_eps if ep['length'] > 0]
    truncs   = [bool(ep['truncated'][-1])  for ep in all_eps if ep['length'] > 0]
    actions  = np.concatenate([ep['actions'] for ep in all_eps])
    sources  = np.concatenate([ep['action_source'] for ep in all_eps])
    policies = [f"{ep['policy_type']}_eps{ep['epsilon']:.2f}" for ep in all_eps]

    n_left  = int((actions == 0).sum())
    n_right = int((actions == 1).sum())

    source_counts = {SRC_NAMES[k]: int(v)
                     for k, v in Counter(sources.tolist()).items()}

    def _split_stats(eps: List[Dict]) -> Dict:
        ls = [ep['length'] for ep in eps]
        return {
            'n_episodes':   len(eps),
            'n_transitions': int(sum(ls)),
            'ep_len_mean':  float(np.mean(ls)) if ls else 0.0,
            'ep_len_min':   int(np.min(ls))    if ls else 0,
            'ep_len_max':   int(np.max(ls))    if ls else 0,
            'episode_seeds': [ep['ep_seed'] for ep in eps],
        }

    return {
        'total_episodes':    len(all_eps),
        'total_transitions': int(sum(lengths)),
        'ep_len_mean': float(np.mean(lengths)),
        'ep_len_min':  int(np.min(lengths)),
        'ep_len_max':  int(np.max(lengths)),
        'ep_len_std':  float(np.std(lengths)),
        'frac_terminated': float(np.mean(terms)) if terms else 0.0,
        'frac_truncated':  float(np.mean(truncs)) if truncs else 0.0,
        'frac_success': float(np.mean([ep['success'] for ep in all_eps])),
        'action_balance': {
            'n_left':     n_left,
            'n_right':    n_right,
            'frac_right': float(n_right / max(n_left + n_right, 1)),
        },
        'action_source_counts': source_counts,
        'policy_mixture_counts': dict(Counter(policies)),
        'splits': {
            'train': _split_stats(train_eps),
            'val':   _split_stats(val_eps),
            'test':  _split_stats(test_eps),
        },
        'config': {
            'image_size':         cfg.image_size,
            'custom_reset':       cfg.custom_reset,
            'cart_pos_range':     cfg.cart_pos_range   if cfg.custom_reset else None,
            'cart_vel_range':     cfg.cart_vel_range   if cfg.custom_reset else None,
            'pole_angle_range':   cfg.pole_angle_range if cfg.custom_reset else None,
            'pole_vel_range':     cfg.pole_vel_range   if cfg.custom_reset else None,
            'seed':               cfg.seed,
            'num_transitions':    cfg.num_transitions,
        },
        'environment': {
            'id':                'CartPole-v1',
            'gymnasium_version': cfg.gymnasium_version,
            'force_mag':         10.0,
            'gravity':           9.8,
            'masscart':          1.0,
            'masspole':          0.1,
            'length':            0.5,
            'tau':               0.02,
            'max_episode_steps': 500,
        },
        'generation_timestamp': cfg.generation_timestamp,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def generate_dataset(cfg: DatasetConfig) -> None:
    """Generate the full CartPole-v1 visual dataset.

    Saves train.hdf5, val.hdf5, test.hdf5, metadata.json,
    and generation_config.yaml inside cfg.output_dir.
    """
    import gymnasium as gymnasium_mod
    cfg.gymnasium_version    = gymnasium_mod.__version__
    cfg.generation_timestamp = time.strftime('%Y-%m-%dT%H:%M:%S')
    cfg.validate()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot immediately
    with open(output_dir / 'generation_config.yaml', 'w') as f:
        yaml.dump(cfg.to_dict(), f, default_flow_style=False, sort_keys=True)

    # Seed everything
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    gain = _compute_expert_gain()
    _verify_expert(gain, cfg, rng)

    # Pre-generate a large pool of (policy_type, epsilon) assignments and seeds
    pool_size   = max(500, cfg.num_transitions // 50)
    assignments = _assign_policies(pool_size, cfg, rng)
    ep_seeds    = rng.integers(0, 2**31, size=pool_size).tolist()

    # Resume support
    state_file = output_dir / '.generation_state.json'
    episodes:     List[Dict] = []
    n_collected   = 0
    start_ep      = 0

    if cfg.resume and state_file.exists():
        try:
            with open(state_file) as f:
                saved = json.load(f)
            start_ep    = saved.get('n_episodes', 0)
            n_collected = saved.get('n_transitions', 0)
            logger.info('Resuming from episode %d  (%d transitions done)',
                        start_ep, n_collected)
        except Exception as exc:
            logger.warning('Could not read resume state: %s', exc)
            start_ep = 0

    env = _make_env(cfg.seed)

    pbar = tqdm(
        total=cfg.num_transitions,
        initial=n_collected,
        unit='trans',
        desc='Collecting',
        dynamic_ncols=True,
    )

    for ep_idx in range(start_ep, pool_size):
        if n_collected >= cfg.num_transitions:
            break

        pt, eps   = assignments[ep_idx]
        ep_seed   = int(ep_seeds[ep_idx])
        ep        = _collect_episode(env, pt, eps, cfg, rng, ep_seed, gain)
        ep['global_ep_id'] = ep_idx
        episodes.append(ep)

        n_collected += ep['length']
        pbar.update(ep['length'])

        if (ep_idx + 1) % 50 == 0:
            with open(state_file, 'w') as f:
                json.dump({'n_episodes': ep_idx + 1,
                           'n_transitions': n_collected}, f)

    pbar.close()
    env.close()

    logger.info('Collected %d episodes / %d transitions', len(episodes), n_collected)

    # Split episodes preserving policy mixture
    train_eps, val_eps, test_eps = _split_episodes(episodes, cfg, rng)
    logger.info('Split: train=%d  val=%d  test=%d episodes',
                len(train_eps), len(val_eps), len(test_eps))

    # Write HDF5 files
    for name, eps, offset in [
        ('train', train_eps, 0),
        ('val',   val_eps,   len(train_eps)),
        ('test',  test_eps,  len(train_eps) + len(val_eps)),
    ]:
        path = output_dir / f'{name}.hdf5'
        logger.info('Writing %s (%d ep, %d trans)...',
                    path.name, len(eps), sum(e['length'] for e in eps))
        _write_hdf5(path, eps, ep_id_offset=offset)

    # Metadata
    meta = _build_metadata(episodes, train_eps, val_eps, test_eps, cfg)
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # Clean up resume state
    if state_file.exists():
        state_file.unlink()

    logger.info('Done. Dataset written to %s', output_dir)
    logger.info('  train: %d ep / %d transitions',
                len(train_eps), meta['splits']['train']['n_transitions'])
    logger.info('  val:   %d ep / %d transitions',
                len(val_eps),   meta['splits']['val']['n_transitions'])
    logger.info('  test:  %d ep / %d transitions',
                len(test_eps),  meta['splits']['test']['n_transitions'])


# ── Loader (for validation / training) ───────────────────────────────────────

def load_split(dataset_dir: str, split: str) -> Tuple[List[Dict], Dict]:
    """Load one split from {dataset_dir}/{split}.hdf5.

    Returns (episodes, file_attrs) where each episode is a dict of arrays.
    """
    path = Path(dataset_dir) / f'{split}.hdf5'
    episodes = []
    with h5py.File(path, 'r') as f:
        file_attrs = dict(f.attrs)
        ep_grp = f['episodes']
        for ep_id in sorted(ep_grp.keys(), key=int):
            g   = ep_grp[ep_id]
            ep  = {k: g[k][:] for k in g.keys()}
            ep.update({k: v for k, v in g.attrs.items()})
            episodes.append(ep)
    return episodes, file_attrs
