#!/usr/bin/env python
"""Poincaré-map and Region-of-Attraction probe for GT iCEM Walker2d.

Uses gt_icem_walker.py (Pinneri et al. 2020 colored-noise iCEM with dense
per-step reward) as both the reference controller and the closed-loop ROA
evaluator.

Two fast evaluation modes:

  --fast        ROA with sparse iCEM (pop=50, iters=1, H=30, eval_steps=50)
                → ~5-10 minutes for a 7×7 grid

  --open-loop   ROA by replaying the reference action sequence open-loop
                from each perturbed state.  No CEM at all — seconds total.

Typical workflow
----------------
  # Step 1 — run iCEM reference once (saves arrays, exits):
  MUJOCO_GL=egl python experiments/probe_poincare_roa_walker.py \\
      --ref-steps 400 --section-stride 10 --nominal-step 200 \\
      --planning-horizon 60 --executed-steps 10 \\
      --cem-population 500 --cem-elites 50 --cem-iters 5 \\
      --beta 2.5 --initial-std 0.5 \\
      --keep-fraction 0.3 --shift-fraction 0.3 --sample-decay 1.25 \\
      --wx 1.0 --wh 1.0 --wu 1e-3 --cf 10.0 \\
      --output-dir results/icem_poincare_roa_ref \\
      --skip-roa

  # Step 2a — open-loop ROA (seconds):
  MUJOCO_GL=egl python experiments/probe_poincare_roa_walker.py \\
      --open-loop --load-ref results/icem_poincare_roa_ref \\
      --grid-nz 7 --grid-nang 7 --nominal-step 200 \\
      --success-min-velocity 0.5 \\
      --output-dir results/icem_poincare_roa_openloop

  # Step 2b — fast iCEM ROA (~5-10 min):
  MUJOCO_GL=egl python experiments/probe_poincare_roa_walker.py \\
      --fast --load-ref results/icem_poincare_roa_ref \\
      --grid-nz 7 --grid-nang 7 --nominal-step 200 \\
      --wx 1.0 --wh 1.0 --wu 1e-3 --cf 10.0 \\
      --beta 2.5 --initial-std 0.5 \\
      --keep-fraction 0.3 --shift-fraction 0.3 --sample-decay 1.25 \\
      --output-dir results/icem_poincare_roa_fast
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

# ── reuse GT iCEM infrastructure ──────────────────────────────────────────────
from experiments.gt_icem_walker import (
    get_state, set_state, is_healthy, failure_reason,
    rollout_cost, WalkerGTiCEM,
    ACTION_DIM, ACTION_LOW, ACTION_HIGH,
    HEALTHY_Z_MIN, HEALTHY_Z_MAX, HEALTHY_ANG_MAX,
)

# Walker2d-v4 state layout:
#   qpos: [x(0), z(1), torso_ang(2), right_hip(3), right_knee(4), right_ankle(5),
#                                     left_hip(6),  left_knee(7),  left_ankle(8)]  → 9 dims
#   qvel: [xdot(0), zdot(1), torso_angdot(2), ... 6 joint vels]  → 9 dims
# Gait state (16-D): qpos[1:9] + qvel[0:8]  (excludes global x-position)
GAIT_DIM = 16


def to_gait_state(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    """17-D (qpos) + 9-D (qvel) → 16-D gait state (no global x)."""
    return np.concatenate([qpos[1:9], qvel[0:8]])


# ── GT iCEM builders ──────────────────────────────────────────────────────────

def build_planner(args, plan_env) -> WalkerGTiCEM:
    """Reference planner — full iCEM for the long reference run."""
    return WalkerGTiCEM(
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
    )


def build_roa_planner(args, plan_env) -> WalkerGTiCEM:
    """ROA planner — lighter iCEM for fast sweeps."""
    return WalkerGTiCEM(
        horizon        = args.roa_horizon,
        executed_steps = args.roa_executed_steps,
        population     = args.roa_population,
        elites         = args.roa_elites,
        iterations     = args.roa_iters,
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
    )


# ── reference persistence ─────────────────────────────────────────────────────

def save_reference(ref: dict, out_dir: Path):
    """Persist all reference arrays so they can be reloaded without re-running CEM."""
    np.save(out_dir / 'ref_section_states.npy', ref['states'])
    np.save(out_dir / 'ref_section_qpos.npy',   ref['qpos'])
    np.save(out_dir / 'ref_section_qvel.npy',   ref['qvel'])
    np.save(out_dir / 'ref_x_vels.npy',         ref['x_vels'])
    np.save(out_dir / 'ref_z_vels.npy',         ref['z_vels'])
    np.save(out_dir / 'ref_actions.npy',         ref['actions'])
    meta = {'survived': bool(ref['survived']), 'n_steps': int(ref['n_steps'])}
    with open(out_dir / 'ref_meta.json', 'w') as f:
        json.dump(meta, f)
    print(f'  Saved reference arrays to {out_dir}/')


def load_reference(ref_dir: Path) -> dict:
    """Load a previously saved reference run — no CEM needed."""
    ref_dir = Path(ref_dir)
    missing = [p for p in [
        'ref_section_states.npy', 'ref_section_qpos.npy', 'ref_section_qvel.npy',
        'ref_x_vels.npy', 'ref_z_vels.npy', 'ref_actions.npy',
    ] if not (ref_dir / p).exists()]
    if missing:
        # Also accept legacy names written by the old probe version
        if (ref_dir / 'section_states.npy').exists() and (ref_dir / 'ref_actions.npy').exists():
            states  = np.load(ref_dir / 'section_states.npy')
            qpos    = np.load(ref_dir / 'ref_section_qpos.npy') if (ref_dir / 'ref_section_qpos.npy').exists() else None
            qvel    = np.load(ref_dir / 'ref_section_qvel.npy') if (ref_dir / 'ref_section_qvel.npy').exists() else None
            actions = np.load(ref_dir / 'ref_actions.npy')
            if qpos is None or qvel is None:
                raise FileNotFoundError(
                    f'--load-ref: missing ref_section_qpos/qvel.npy in {ref_dir}. '
                    'Re-run the reference step with the updated probe to save them.')
            return {
                'states': states, 'qpos': qpos, 'qvel': qvel,
                'x_vels': np.zeros(0), 'z_vels': np.zeros(0),
                'actions': actions, 'survived': True, 'n_steps': -1,
            }
        raise FileNotFoundError(
            f'--load-ref: missing files in {ref_dir}: {missing}\n'
            'Run with --skip-roa first to save the reference, then use --load-ref.')
    with open(ref_dir / 'ref_meta.json') as f:
        meta = json.load(f)
    ref = {
        'states':   np.load(ref_dir / 'ref_section_states.npy'),
        'qpos':     np.load(ref_dir / 'ref_section_qpos.npy'),
        'qvel':     np.load(ref_dir / 'ref_section_qvel.npy'),
        'x_vels':   np.load(ref_dir / 'ref_x_vels.npy'),
        'z_vels':   np.load(ref_dir / 'ref_z_vels.npy'),
        'actions':  np.load(ref_dir / 'ref_actions.npy'),
        'survived': meta['survived'],
        'n_steps':  meta['n_steps'],
    }
    print(f'  Loaded reference from {ref_dir}:  '
          f'sections={len(ref["states"])}  actions={len(ref["actions"])}  '
          f'survived={ref["survived"]}')
    return ref


# ── reference run ─────────────────────────────────────────────────────────────

def run_reference(planner, eval_env, n_steps, section_stride, seed):
    """Run one long trial; return section states, full velocity trace, and all actions."""
    eval_env.reset(seed=seed)
    qpos0, qvel0 = get_state(eval_env)
    set_state(eval_env, qpos0, qvel0)
    planner._prev_blocks = None

    section_states = []          # (T_sec, GAIT_DIM)
    section_qpos   = []          # (T_sec, 9) full qpos for perturbation later
    section_qvel   = []          # (T_sec, 9) full qvel
    x_vels         = []
    z_vels         = []          # zdot for period estimation
    all_actions    = []          # every executed action (for open-loop replay)

    step = 0
    terminated = truncated = False
    step_in_stride = 0

    print(f'  Reference run: {n_steps} steps, section every {section_stride} steps')
    while step < n_steps and not (terminated or truncated):
        qpos, qvel = get_state(eval_env)

        # record section state at the start of each stride
        if step_in_stride == 0:
            section_states.append(to_gait_state(qpos, qvel))
            section_qpos.append(qpos.copy())
            section_qvel.append(qvel.copy())

        sequence = planner.plan(qpos, qvel)

        for a in sequence:
            if step >= n_steps or terminated or truncated:
                break
            obs, reward, terminated, truncated, info = eval_env.step(a)
            all_actions.append(a.copy())
            step += 1
            step_in_stride = (step_in_stride + 1) % section_stride
            x_vels.append(float(info.get('x_velocity', 0.0)))
            z_vels.append(float(eval_env.unwrapped.data.qvel[1]))

        if terminated or truncated:
            break

    survived = step >= n_steps and not terminated
    avg_vel  = float(np.mean(x_vels)) if x_vels else 0.0
    print(f'  Reference: survived={survived}  steps={step}  avg_vel={avg_vel:.3f} m/s  '
          f'sections={len(section_states)}  actions_recorded={len(all_actions)}')
    return {
        'states':    np.array(section_states),   # (T_sec, 16)
        'qpos':      np.array(section_qpos),     # (T_sec, 9)
        'qvel':      np.array(section_qvel),     # (T_sec, 9)
        'x_vels':    np.array(x_vels),
        'z_vels':    np.array(z_vels),
        'actions':   np.array(all_actions),      # (n_steps, ACTION_DIM)
        'survived':  survived,
        'n_steps':   step,
    }


# ── gait period estimation ────────────────────────────────────────────────────

def estimate_period(z_vels: np.ndarray, max_lag: int = 100) -> int:
    """Estimate gait period from auto-correlation of vertical velocity."""
    z = z_vels - z_vels.mean()
    n = len(z)
    if n < 10:
        return -1
    lags = range(1, min(max_lag, n // 2))
    acf  = [float(np.corrcoef(z[:n - k], z[k:])[0, 1]) for k in lags]
    # first local max after lag 1
    for i in range(1, len(acf) - 1):
        if acf[i] > acf[i - 1] and acf[i] > acf[i + 1] and acf[i] > 0.1:
            return int(lags[i])
    return -1


# ── Poincaré return map ───────────────────────────────────────────────────────

def poincare_analysis(section_states: np.ndarray, out_dir: Path,
                      period_steps: int) -> dict:
    """PCA projection and return-map figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    T = len(section_states)
    if T < 4:
        print('  [poincare] too few section states — skipping')
        return {}

    pca    = PCA(n_components=2)
    coords = pca.fit_transform(section_states)     # (T, 2)
    var    = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # ── panel 1: PC1, PC2 over time ───────────────────────────────────────────
    t = np.arange(T)
    axes[0].plot(t, coords[:, 0], label='PC1', alpha=0.8)
    axes[0].plot(t, coords[:, 1], label='PC2', alpha=0.8)
    axes[0].set_xlabel('Section index')
    axes[0].set_ylabel('PC coordinate')
    axes[0].set_title('Gait state PCs over time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── panel 2: Poincaré scatter (s_t vs s_{t+1} in PC1) ────────────────────
    axes[1].scatter(coords[:-1, 0], coords[1:, 0],
                    c=t[:-1], cmap='viridis', s=15, alpha=0.7)
    axes[1].plot([coords[:, 0].min(), coords[:, 0].max()],
                 [coords[:, 0].min(), coords[:, 0].max()],
                 'r--', lw=1, label='identity (fixed point)')
    axes[1].set_xlabel('PC1(t)')
    axes[1].set_ylabel('PC1(t+1)')
    axes[1].set_title(f'Poincaré return map (PC1)\nvar={var[0]:.1%}')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # ── panel 3: phase portrait PC1 vs PC2 ───────────────────────────────────
    sc = axes[2].scatter(coords[:, 0], coords[:, 1],
                         c=t, cmap='viridis', s=20, alpha=0.8)
    plt.colorbar(sc, ax=axes[2], label='Section index')
    axes[2].set_xlabel(f'PC1 ({var[0]:.1%} var)')
    axes[2].set_ylabel(f'PC2 ({var[1]:.1%} var)')
    axes[2].set_title('Poincaré section states (phase portrait)')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('Poincaré Map — GT iCEM Walker2d', fontsize=12)
    plt.tight_layout()
    path = out_dir / 'poincare.pdf'
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  [poincare] saved {path}')

    # compute spread (convergence indicator)
    post_transient = coords[T // 3:]   # last 2/3 of run
    spread_pc1 = float(post_transient[:, 0].std())
    spread_pc2 = float(post_transient[:, 1].std())
    return {
        'n_sections':     T,
        'var_pc1':        float(var[0]),
        'var_pc2':        float(var[1]),
        'spread_pc1_post_transient': spread_pc1,
        'spread_pc2_post_transient': spread_pc2,
        'pca_components': pca.components_.tolist(),
        'pca_mean':       pca.mean_.tolist(),
    }


# ── ROA perturbation evaluators ───────────────────────────────────────────────

def eval_perturbation(planner, eval_env, plan_env,
                      nom_qpos, nom_qvel,
                      dz, dang,
                      eval_steps, min_vel, seed):
    """Test whether (dz, dang) perturbation from nominal recovers using CEM."""
    qpos = nom_qpos.copy()
    qvel = nom_qvel.copy()
    qpos[1] += dz
    qpos[2] += dang

    z   = float(qpos[1])
    ang = float(qpos[2])
    if not (HEALTHY_Z_MIN < z < HEALTHY_Z_MAX and abs(ang) < HEALTHY_ANG_MAX):
        return False, 0.0, 'initial_unhealthy'

    eval_env.reset()
    set_state(eval_env, qpos, qvel)
    plan_env.reset(seed=seed)
    set_state(plan_env, qpos, qvel)
    planner._prev_blocks = None

    step = 0
    x_vels = []
    terminated = truncated = False

    while step < eval_steps and not (terminated or truncated):
        qp, qv = get_state(eval_env)
        sequence = planner.plan(qp, qv)
        for a in sequence:
            if step >= eval_steps or terminated or truncated:
                break
            obs, reward, terminated, truncated, info = eval_env.step(a)
            step += 1
            x_vels.append(float(info.get('x_velocity', 0.0)))

    survived = step >= eval_steps and not terminated
    avg_vel  = float(np.mean(x_vels)) if x_vels else 0.0
    success  = survived and avg_vel >= min_vel
    reason   = 'ok' if success else (failure_reason(eval_env.unwrapped.data)
                                     if terminated else 'low_vel' if survived else 'fall')
    return success, avg_vel, reason


def eval_perturbation_open_loop(eval_env,
                                nom_qpos, nom_qvel,
                                dz, dang,
                                ref_actions, eval_steps, min_vel):
    """Apply reference actions open-loop from perturbed state.  Milliseconds per call.

    No CEM — just replay the reference controller's action sequence verbatim.
    A point succeeds if the walker survives all eval_steps AND avg_vel >= min_vel.
    This tests the *open-loop stability* of the reference trajectory, i.e. whether
    the perturbed state lies inside the basin of attraction of the nominal trajectory
    under open-loop execution.
    """
    qpos = nom_qpos.copy()
    qvel = nom_qvel.copy()
    qpos[1] += dz
    qpos[2] += dang

    z   = float(qpos[1])
    ang = float(qpos[2])
    if not (HEALTHY_Z_MIN < z < HEALTHY_Z_MAX and abs(ang) < HEALTHY_ANG_MAX):
        return False, 0.0, 'initial_unhealthy'

    # reset resets _elapsed_steps in the TimeLimit wrapper; then override state
    eval_env.reset()
    set_state(eval_env, qpos, qvel)
    x_vels = []
    step   = 0
    terminated = truncated = False

    actions_to_use = ref_actions[:eval_steps]
    for a in actions_to_use:
        if terminated or truncated:
            break
        obs, reward, terminated, truncated, info = eval_env.step(a)
        step += 1
        x_vels.append(float(info.get('x_velocity', 0.0)))

    survived = step >= len(actions_to_use) and not terminated
    avg_vel  = float(np.mean(x_vels)) if x_vels else 0.0
    success  = survived and avg_vel >= min_vel
    reason   = 'ok' if success else ('fall' if terminated else 'low_vel')
    return success, avg_vel, reason


# ── ROA estimation ────────────────────────────────────────────────────────────

def roa_estimation(planner, roa_planner, eval_env, plan_env,
                   nom_qpos, nom_qvel,
                   ref_actions,
                   grid_nz, grid_nang,
                   dz_max, dang_max,
                   eval_steps, min_vel,
                   open_loop: bool,
                   out_dir: Path, seed: int) -> dict:
    """Sweep (Δz, Δang) grid and classify recovery.

    If open_loop=True, use action replay (fast).
    Otherwise use the roa_planner CEM (moderate).
    """
    dz_vals   = np.linspace(-dz_max,   dz_max,   grid_nz)
    dang_vals = np.linspace(-dang_max, dang_max, grid_nang)
    n_total   = grid_nz * grid_nang

    success_grid = np.zeros((grid_nz, grid_nang), dtype=bool)
    vel_grid     = np.zeros((grid_nz, grid_nang))
    reason_grid  = np.full((grid_nz, grid_nang), '', dtype=object)

    nom_z   = float(nom_qpos[1])
    nom_ang = float(nom_qpos[2])

    mode_str = 'open-loop replay' if open_loop else f'CEM(pop={roa_planner.population if roa_planner else "?"}, iters={roa_planner.iterations if roa_planner else "?"})'
    print(f'\n  ROA sweep: {grid_nz}×{grid_nang} = {n_total} points  '
          f'eval_steps={eval_steps}  min_vel={min_vel}  mode={mode_str}')
    print(f'  Nominal: z={nom_z:.3f}  ang={nom_ang:.3f}')
    print(f'  Δz ∈ [{-dz_max:.2f}, {dz_max:.2f}]  '
          f'Δang ∈ [{-dang_max:.2f}, {dang_max:.2f}]')

    done = 0
    t0   = time.time()
    for i, dz in enumerate(dz_vals):
        for j, dang in enumerate(dang_vals):
            if open_loop:
                s, v, r = eval_perturbation_open_loop(
                    eval_env,
                    nom_qpos, nom_qvel,
                    dz, dang,
                    ref_actions, eval_steps, min_vel)
            else:
                s, v, r = eval_perturbation(
                    roa_planner, eval_env, plan_env,
                    nom_qpos, nom_qvel,
                    dz, dang,
                    eval_steps, min_vel, seed + i * grid_nang + j)

            success_grid[i, j] = s
            vel_grid[i, j]     = v
            reason_grid[i, j]  = r
            done += 1
            elapsed = time.time() - t0
            eta     = elapsed / done * (n_total - done)
            print(f'  [{done:3d}/{n_total}] Δz={dz:+.3f} Δang={dang:+.3f} '
                  f'→ {"OK" if s else "FAIL":4s}  vel={v:.2f}  {r}  '
                  f'[ETA {eta:.0f}s]', flush=True)

    # ── figure ────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel 1: binary ROA (green/red)
    ax = axes[0]
    ax.imshow(
        success_grid.T.astype(float),
        origin='lower',
        extent=[-dz_max, dz_max, -dang_max, dang_max],
        aspect='auto',
        cmap='RdYlGn',
        vmin=0, vmax=1,
        alpha=0.85,
    )
    ax.plot(0, 0, 'b*', ms=12, label='nominal state')
    ax.axvline(HEALTHY_Z_MIN - nom_z,   color='k', lw=1.5, ls='--', label='health bound z')
    ax.axvline(HEALTHY_Z_MAX - nom_z,   color='k', lw=1.5, ls='--')
    ax.axhline(-HEALTHY_ANG_MAX - nom_ang, color='gray', lw=1.5, ls=':', label='health bound ang')
    ax.axhline(+HEALTHY_ANG_MAX - nom_ang, color='gray', lw=1.5, ls=':')
    ax.set_xlabel('Δz (torso height, m)')
    ax.set_ylabel('Δang (torso angle, rad)')
    mode_label = 'open-loop' if open_loop else 'CEM (fast)' if roa_planner else 'CEM'
    ax.set_title(f'ROA [{mode_label}] — success (green) / failure (red)')
    ax.legend(fontsize=7)

    # Panel 2: avg velocity heatmap
    ax2 = axes[1]
    vm  = np.nanmax(vel_grid)
    im2 = ax2.imshow(
        vel_grid.T,
        origin='lower',
        extent=[-dz_max, dz_max, -dang_max, dang_max],
        aspect='auto',
        cmap='plasma',
        vmin=0, vmax=max(vm, min_vel),
    )
    plt.colorbar(im2, ax=ax2, label='avg x-velocity (m/s)')
    ax2.contour(
        np.linspace(-dz_max, dz_max, grid_nz),
        np.linspace(-dang_max, dang_max, grid_nang),
        vel_grid.T,
        levels=[min_vel], colors='white', linewidths=1.5,
    )
    ax2.plot(0, 0, 'b*', ms=12, label='nominal')
    ax2.set_xlabel('Δz (torso height, m)')
    ax2.set_ylabel('Δang (torso angle, rad)')
    ax2.set_title(f'Avg x-velocity (m/s)  [white = {min_vel} m/s]')
    ax2.legend(fontsize=7)

    fig.suptitle(f'Region of Attraction — GT iCEM Walker2d [{mode_label}]', fontsize=12)
    plt.tight_layout()
    path = out_dir / 'roa.pdf'
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  [roa] saved {path}')

    n_success = int(success_grid.sum())
    roa_frac  = n_success / n_total
    return {
        'n_total':        n_total,
        'n_success':      n_success,
        'roa_fraction':   roa_frac,
        'mode':           'open_loop' if open_loop else 'cem',
        'dz_values':      dz_vals.tolist(),
        'dang_values':    dang_vals.tolist(),
        'success_grid':   success_grid.tolist(),
        'vel_grid':       vel_grid.tolist(),
        'reason_grid':    reason_grid.tolist(),
        'nominal_z':      nom_z,
        'nominal_ang':    nom_ang,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Poincaré map + ROA probe for GT iCEM Walker2d.')

    # ── speed modes ───────────────────────────────────────────────────────────
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--fast', action='store_true',
                      help='ROA with sparse CEM (pop=50, iters=1, H=30, eval_steps=50). '
                           '~5-10 min for 7×7 grid.')
    mode.add_argument('--open-loop', action='store_true',
                      help='ROA by replaying reference actions open-loop. '
                           'No CEM — completes in <30 s for 7×7 grid.')

    # ── reference caching ─────────────────────────────────────────────────────
    p.add_argument('--load-ref', default=None, metavar='DIR',
                   help='Skip the CEM reference run and load previously saved '
                        'reference arrays from DIR (created by --skip-roa or any '
                        'prior run).  Combine with --open-loop for a near-instant probe.')
    p.add_argument('--skip-roa', action='store_true',
                   help='Run only the reference + Poincaré steps; save reference '
                        'arrays to disk, then exit.  Use --load-ref DIR on the '
                        'next call to skip straight to ROA.')

    # ── reference run ─────────────────────────────────────────────────────────
    p.add_argument('--ref-steps',        type=int,   default=400)
    p.add_argument('--section-stride',   type=int,   default=10,
                   help='Executed steps between Poincaré section recordings')
    p.add_argument('--nominal-step',     type=int,   default=200,
                   help='Which executed step defines the nominal gait state for ROA')

    # ── ROA grid ──────────────────────────────────────────────────────────────
    p.add_argument('--grid-nz',          type=int,   default=7)
    p.add_argument('--grid-nang',        type=int,   default=7)
    p.add_argument('--dz-max',           type=float, default=0.15,
                   help='Max Δz perturbation (m).  Healthy range: z∈(0.8, 2.0)')
    p.add_argument('--dang-max',         type=float, default=0.25,
                   help='Max Δang perturbation (rad).  Healthy range: |ang|<1.0')
    p.add_argument('--eval-steps',       type=int,   default=None,
                   help='Steps to evaluate each perturbed state. '
                        'Default: 200 (full), 50 (--fast), ref_steps (--open-loop)')
    p.add_argument('--success-min-velocity', type=float, default=0.5)

    # ── reference iCEM planner ────────────────────────────────────────────────
    p.add_argument('--planning-horizon',  type=int,   default=60)
    p.add_argument('--executed-steps',    type=int,   default=10)
    p.add_argument('--cem-population',    type=int,   default=500)
    p.add_argument('--cem-elites',        type=int,   default=50)
    p.add_argument('--cem-iters',         type=int,   default=5)
    # iCEM-specific
    p.add_argument('--beta',              type=float, default=2.5,
                   help='Colored-noise exponent (0=white, 2.5=Pinneri default)')
    p.add_argument('--initial-std',       type=float, default=0.5)
    p.add_argument('--keep-fraction',     type=float, default=0.3)
    p.add_argument('--shift-fraction',    type=float, default=0.3)
    p.add_argument('--sample-decay',      type=float, default=1.25)

    # ── ROA iCEM planner (independent from reference planner) ─────────────────
    p.add_argument('--roa-horizon',      type=int,   default=None,
                   help='Planning horizon for ROA iCEM. Default: planning-horizon '
                        '(full), 30 (--fast)')
    p.add_argument('--roa-population',   type=int,   default=None,
                   help='iCEM population for ROA. Default: cem-population '
                        '(full), 50 (--fast)')
    p.add_argument('--roa-elites',       type=int,   default=None,
                   help='iCEM elites for ROA. Default: cem-elites '
                        '(full), 10 (--fast)')
    p.add_argument('--roa-iters',        type=int,   default=None,
                   help='iCEM iterations for ROA. Default: cem-iters '
                        '(full), 1 (--fast)')
    p.add_argument('--roa-executed-steps', type=int, default=None,
                   help='Executed steps per iCEM call for ROA. '
                        'Default: executed-steps (full), 5 (--fast)')

    # ── objective weights (shared by ref & ROA planners) ──────────────────────
    p.add_argument('--wx',   type=float, default=1.0)
    p.add_argument('--wh',   type=float, default=1.0,
                   help='Alive bonus weight (dense reward, matches Gym Walker2d)')
    p.add_argument('--wu',   type=float, default=1e-3)
    p.add_argument('--cf',   type=float, default=10.0)
    p.add_argument('--wz',   type=float, default=0.0)
    p.add_argument('--wang', type=float, default=0.0)
    p.add_argument('--height-target',    type=float, default=1.2)
    p.add_argument('--frame-skip',       type=int,   default=5)

    p.add_argument('--seed',             type=int,   default=42)
    p.add_argument('--output-dir',       default='results/poincare_roa')
    args = p.parse_args()

    # ── apply --fast defaults ─────────────────────────────────────────────────
    if args.fast:
        if args.roa_horizon    is None: args.roa_horizon       = 30
        if args.roa_population is None: args.roa_population    = 50
        if args.roa_elites     is None: args.roa_elites        = 10
        if args.roa_iters      is None: args.roa_iters         = 1
        if args.roa_executed_steps is None: args.roa_executed_steps = 5
        if args.eval_steps     is None: args.eval_steps        = 50
    elif args.open_loop:
        if args.eval_steps is None: args.eval_steps = args.ref_steps
    else:
        if args.roa_horizon    is None: args.roa_horizon       = args.planning_horizon
        if args.roa_population is None: args.roa_population    = args.cem_population
        if args.roa_elites     is None: args.roa_elites        = args.cem_elites
        if args.roa_iters      is None: args.roa_iters         = args.cem_iters
        if args.roa_executed_steps is None: args.roa_executed_steps = args.executed_steps
        if args.eval_steps     is None: args.eval_steps        = 200

    try:
        from sklearn.decomposition import PCA  # noqa: F401
    except ImportError:
        print('ERROR: scikit-learn required.  pip install scikit-learn')
        sys.exit(1)

    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mode_str = ('open-loop replay' if args.open_loop
                else f'fast iCEM(pop={args.roa_population}, iters={args.roa_iters}, '
                     f'H={args.roa_horizon})' if args.fast
                else f'full iCEM(pop={args.roa_population}, iters={args.roa_iters}, '
                     f'H={args.roa_horizon})')

    # ── 1. Reference run (or load from disk) ──────────────────────────────────
    if args.load_ref:
        print(f'[Poincaré/ROA probe] GT iCEM Walker2d  — ROA mode: {mode_str}')
        print(f'[1/3] Loading reference from {args.load_ref} ...')
        ref = load_reference(Path(args.load_ref))
        eval_env  = gym.make('Walker2d-v4')
        plan_env  = gym.make('Walker2d-v4')
        eval_env.reset(seed=args.seed)
        plan_env.reset(seed=args.seed)
        planner = None
    else:
        eval_env = gym.make('Walker2d-v4')
        plan_env = gym.make('Walker2d-v4')
        eval_env.reset(seed=args.seed)
        plan_env.reset(seed=args.seed)
        planner = build_planner(args, plan_env)

        print(f'[Poincaré/ROA probe] GT iCEM Walker2d  — ROA mode: {mode_str}')
        print(f'  Ref iCEM:  H={args.planning_horizon}  exec={args.executed_steps}  '
              f'pop={args.cem_population}  elites={args.cem_elites}  iters={args.cem_iters}  '
              f'β={args.beta}  std={args.initial_std}')
        if not args.open_loop:
            print(f'  ROA iCEM:  H={args.roa_horizon}  exec={args.roa_executed_steps}  '
                  f'pop={args.roa_population}  elites={args.roa_elites}  iters={args.roa_iters}')
        print(f'  Objective: wx={args.wx}  wh={args.wh}  wu={args.wu}  cf={args.cf}  '
              f'wz={args.wz}  wang={args.wang}')

        print('\n[1/3] Reference run ...')
        ref = run_reference(planner, eval_env, args.ref_steps,
                            args.section_stride, args.seed)
        save_reference(ref, out_dir)

    period = estimate_period(ref['z_vels']) if len(ref['z_vels']) > 0 else -1
    print(f'  Estimated gait period: {period} steps '
          f'({period * args.frame_skip * 0.002:.3f} s)' if period > 0
          else '  Gait period: could not detect')

    # ── 2. Poincaré map ───────────────────────────────────────────────────────
    print('\n[2/3] Poincaré map analysis ...')
    poincare_info = poincare_analysis(ref['states'], out_dir, period)

    if args.skip_roa:
        print('\n[--skip-roa] Reference + Poincaré done.  '
              f'Arrays saved to {out_dir}/\n'
              f'Re-run with --load-ref {out_dir} --open-loop (or --fast) to do ROA.')
        plan_env.close()
        eval_env.close()
        return

    # ── 3. ROA estimation ─────────────────────────────────────────────────────
    print('\n[3/3] ROA estimation ...')

    nom_idx   = args.nominal_step // args.section_stride
    nom_idx   = min(nom_idx, len(ref['qpos']) - 1)
    nom_qpos  = ref['qpos'][nom_idx]
    nom_qvel  = ref['qvel'][nom_idx]
    print(f'  Nominal state at section {nom_idx} '
          f'(step {nom_idx * args.section_stride}):  '
          f'z={nom_qpos[1]:.3f}  ang={nom_qpos[2]:.3f}  '
          f'xdot={nom_qvel[0]:.3f}')

    # For open-loop: use actions starting from nominal_step
    nom_step  = nom_idx * args.section_stride
    ref_actions_from_nom = ref['actions'][nom_step:] if len(ref['actions']) > nom_step else ref['actions']

    roa_planner = None
    if not args.open_loop:
        roa_plan_env = gym.make('Walker2d-v4')
        roa_plan_env.reset(seed=args.seed)
        roa_planner = build_roa_planner(args, roa_plan_env)
    else:
        roa_plan_env = None

    t_roa_start = time.time()
    roa_info = roa_estimation(
        planner, roa_planner, eval_env,
        plan_env if not args.open_loop else None,
        nom_qpos, nom_qvel,
        ref_actions_from_nom,
        args.grid_nz, args.grid_nang,
        args.dz_max, args.dang_max,
        args.eval_steps, args.success_min_velocity,
        args.open_loop,
        out_dir, args.seed)
    t_roa = time.time() - t_roa_start
    print(f'  ROA sweep done in {t_roa:.1f}s ({t_roa/60:.1f} min)')

    # ── summary ───────────────────────────────────────────────────────────────
    summary = {
        'controller': {
            'planner':          'gt_icem_mpc',
            'env':              'Walker2d-v4',
            'planning_horizon': args.planning_horizon,
            'executed_steps':   args.executed_steps,
            'population':       args.cem_population,
            'elites':           args.cem_elites,
            'iterations':       args.cem_iters,
            'beta':             args.beta,
            'initial_std':      args.initial_std,
            'keep_fraction':    args.keep_fraction,
            'shift_fraction':   args.shift_fraction,
            'sample_decay':     args.sample_decay,
            'wx': args.wx, 'wh': args.wh, 'wu': args.wu,
            'cf': args.cf, 'wz': args.wz, 'wang': args.wang,
        },
        'roa_mode': ('open_loop' if args.open_loop
                     else 'fast_cem' if args.fast else 'full_cem'),
        'roa_planner': ({
            'horizon':    args.roa_horizon,
            'population': args.roa_population,
            'elites':     args.roa_elites,
            'iterations': args.roa_iters,
            'executed_steps': args.roa_executed_steps,
        } if not args.open_loop else None),
        'reference': {
            'ref_steps':         args.ref_steps,
            'section_stride':    args.section_stride,
            'survived':          bool(ref['survived']),
            'n_steps':           int(ref['n_steps']),
            'avg_vel':           float(np.mean(ref['x_vels'])),
            'gait_period_steps': int(period),
        },
        'poincare':  poincare_info,
        'roa':       {k: v for k, v in roa_info.items()
                      if k not in ('success_grid', 'vel_grid', 'reason_grid',
                                   'dz_values', 'dang_values')},
        'roa_success_grid': roa_info.get('success_grid'),
        'roa_dz_values':    roa_info.get('dz_values'),
        'roa_dang_values':  roa_info.get('dang_values'),
        'roa_sweep_seconds': round(t_roa, 1),
    }

    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n[done]  output → {out_dir}')
    print(f'  Reference: avg_vel={summary["reference"]["avg_vel"]:.3f} m/s  '
          f'gait_period={period} steps')
    print(f'  Poincaré:  {poincare_info.get("n_sections", 0)} sections  '
          f'PC1_var={poincare_info.get("var_pc1", 0):.1%}  '
          f'post-transient PC1 spread={poincare_info.get("spread_pc1_post_transient", 0):.4f}')
    print(f'  ROA [{summary["roa_mode"]}]:  '
          f'{roa_info["n_success"]}/{roa_info["n_total"]} points recovered  '
          f'({roa_info["roa_fraction"]:.1%})  '
          f'[{t_roa:.1f}s]')

    plan_env.close()
    eval_env.close()
    if roa_plan_env is not None:
        roa_plan_env.close()


if __name__ == '__main__':
    main()
