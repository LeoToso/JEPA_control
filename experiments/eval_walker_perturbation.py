#!/usr/bin/env python
"""Mid-rollout perturbation ablation for Walker2D JEPA models.

At step T_PERTURB a torso-velocity kick is injected into the real MuJoCo
environment.  We then run two parallel trajectories for T_RECOVERY steps:

  1. Ground-truth (MuJoCo): SAC policy acts on real proprioceptive states.
  2. Model closed-loop: SAC policy acts on probe-decoded latent states.
     The model predictor rolls forward; the probe decodes z → 17-D state;
     that decoded state is fed back into the SAC at each step.

Both branches start from the same perturbed state and use the SAME SAC
policy — the only difference is whether the SAC sees real or model-predicted
observations.  This distinguishes accurate recovery dynamics from a model
that simply reverts to predicting normal gait, addressing:
  "Could the Walker2D evaluation include a mid-rollout perturbation and
   compare the predicted recovery with the corresponding MuJoCo recovery
   under the same SAC policy?"

Usage
-----
MUJOCO_GL=egl python experiments/eval_walker_perturbation.py \\
    --models \\
        MS+SR   /path/to/ms_sr/model_final.pt   configs/walker2d_smwm_sigreg_rollout_act1.yaml \\
        FWD+EP-AR /path/to/fwd_ep/model_final.pt configs/walker2d_smwm_fwd_endpoint_inverse_act1.yaml \\
        1SP+SR  /path/to/1sp_sr/model_final.pt  configs/walker2d_smwm_sigreg_fwd_act1.yaml \\
    --hdf5-dir data/walker2d_mixed_sac_fs5_64 \\
    --sac-repo sdpkjc/Walker2d-v4-sac_continuous_action-seed4 \\
    --perturb-step 50 \\
    --recovery-steps 80 \\
    --perturb-mag 6.0 \\
    --trials 20 \\
    --out results/walker2d_perturbation.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch

from experiments.walker2d_utils import (
    load_walker_bundle,
    fit_walker_rollout_probe,
    latent_step,
    gym_obs_to_mj_state,
    mj_state_to_gym_obs,
    is_healthy_obs,
    HEALTHY_ANG_MAX,
)
from experiments.probe_utils import encode_obs
from envs.walker2d_visual import Walker2dVisual


# ── perturbation ──────────────────────────────────────────────────────────────

def apply_perturbation(
    visual_env: Walker2dVisual,
    state17: np.ndarray,
    perturb_type: str,
    perturb_mag: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inject a perturbation into the MuJoCo env.

    Returns
    -------
    new_state17 : (17,) float32 — proprioceptive state after perturbation
    new_frame   : (H,W,3) uint8 — visual observation after perturbation
    delta       : (9,) float32  — qvel delta applied
    """
    x_pos = float(visual_env._env.unwrapped.data.qpos[0])
    qpos, qvel = gym_obs_to_mj_state(state17, x_pos)
    delta = np.zeros(9, dtype=np.float64)

    if perturb_type == 'torso_vel':
        # Kick torso angular velocity (qvel[2]) — primary balance destabiliser
        sign = rng.choice([-1, 1])
        delta[2] = sign * perturb_mag
    elif perturb_type == 'body_vel':
        # Gaussian kick across all 9 velocity DOFs
        delta = rng.normal(0.0, perturb_mag, size=9)
    elif perturb_type == 'joint_angle':
        # Perturb hip and knee angles (qpos[3:6])
        qpos[3:6] += rng.normal(0.0, perturb_mag, size=3)
    else:
        raise ValueError(f'Unknown perturb_type: {perturb_type!r}')

    qvel = qvel + delta
    visual_env._env.unwrapped.set_state(qpos, qvel)
    new_state17 = mj_state_to_gym_obs(qpos, qvel)
    new_frame   = visual_env._render()
    return new_state17, new_frame, delta.astype(np.float32)


# ── single trial ──────────────────────────────────────────────────────────────

def run_trial(
    bundle:         dict,
    probe,
    sac_policy,
    visual_env:     Walker2dVisual,
    seed:           int,
    perturb_step:   int,
    recovery_steps: int,
    perturb_type:   str,
    perturb_mag:    float,
) -> dict:
    """Run one perturbation trial (closed-loop comparison).

    Both GT and model branches start from the same perturbed state.
    GT:    SAC(real_state)   → MuJoCo step  (ground truth)
    Model: SAC(probe(z))     → latent_step  (closed-loop prediction)
    The SAC policy is identical in both branches; only the observation source differs.
    """
    rng = np.random.default_rng(seed)

    # ── warm-up: run SAC in real env to perturb_step ──────────────────────────
    frame, state17, _ = visual_env.reset(seed=seed)
    prev_frame = frame.copy()

    for _ in range(perturb_step):
        action     = sac_policy.act(state17)
        frame_new, state17_new, _, done, _ = visual_env.step(action)
        prev_frame = frame.copy()
        frame      = frame_new
        state17    = state17_new
        if done:
            break

    # ── apply perturbation ────────────────────────────────────────────────────
    pert_state17, pert_frame, delta = apply_perturbation(
        visual_env, state17, perturb_type, perturb_mag, rng)

    # Both branches start from the perturbed state
    cur_state17_gt = pert_state17.copy()          # GT uses real env state
    z = encode_obs(bundle, pert_frame, prev_frame, pert_state17)   # (1, D)

    # ── recovery phase ────────────────────────────────────────────────────────
    gt_states:     list[list] = []
    pred_states:   list[list] = []
    gt_actions:    list[list] = []
    pred_actions:  list[list] = []
    gt_survived:   list[bool] = []
    pred_survived: list[bool] = []

    done_gt = False

    for k in range(recovery_steps):
        # ── GT branch: SAC sees real proprioceptive state ─────────────────────
        action_gt = sac_policy.act(cur_state17_gt)
        gt_states.append(cur_state17_gt.tolist())
        gt_actions.append(action_gt.tolist())
        gt_survived.append(is_healthy_obs(cur_state17_gt))

        if not done_gt:
            _, next_state17_gt, _, done_gt, _ = visual_env.step(action_gt)
            cur_state17_gt = next_state17_gt

        # ── Model branch: SAC sees probe-decoded latent state ─────────────────
        pred17      = probe(z[0].cpu().numpy())          # decode current z
        action_pred = sac_policy.act(pred17)             # SAC on model's obs
        z           = latent_step(bundle, z, action_pred)   # advance latent

        pred_states.append(pred17.tolist())
        pred_actions.append(action_pred.tolist())
        pred_survived.append(is_healthy_obs(pred17))

    # ── per-step metrics ──────────────────────────────────────────────────────
    gt_arr   = np.array(gt_states,   dtype=np.float32)   # (T, 17)
    pred_arr = np.array(pred_states, dtype=np.float32)   # (T, 17)
    gt_act   = np.array(gt_actions,  dtype=np.float32)   # (T, 6)
    pr_act   = np.array(pred_actions, dtype=np.float32)  # (T, 6)

    state_mse_per_step   = ((pred_arr - gt_arr) ** 2).mean(axis=1).tolist()
    xvel_mae_per_step    = np.abs(pred_arr[:, 8] - gt_arr[:, 8]).tolist()
    ang_mae_per_step     = np.abs(pred_arr[:, 1] - gt_arr[:, 1]).tolist()
    # Action divergence: does the SAC choose different actions due to obs mismatch?
    action_mae_per_step  = np.abs(pr_act - gt_act).mean(axis=1).tolist()

    return {
        'seed':                 seed,
        'perturb_step':         perturb_step,
        'perturb_delta':        delta.tolist(),
        'gt_states':            gt_states,
        'pred_states':          pred_states,
        'gt_actions':           gt_actions,
        'pred_actions':         pred_actions,
        'gt_survived':          gt_survived,
        'pred_survived':        pred_survived,
        'state_mse_per_step':   state_mse_per_step,
        'xvel_mae_per_step':    xvel_mae_per_step,
        'ang_mae_per_step':     ang_mae_per_step,
        'action_mae_per_step':  action_mae_per_step,
        'mean_state_mse':       float(np.mean(state_mse_per_step)),
        'mean_xvel_mae':        float(np.mean(xvel_mae_per_step)),
        'mean_ang_mae':         float(np.mean(ang_mae_per_step)),
        'mean_action_mae':      float(np.mean(action_mae_per_step)),
        'survival_acc':         float(np.mean(
            [g == p for g, p in zip(gt_survived, pred_survived)])),
        # Did the model's SAC-driven trajectory survive at the same rate?
        'gt_survival_rate':     float(np.mean(gt_survived)),
        'pred_survival_rate':   float(np.mean(pred_survived)),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.split('\n')[0])
    p.add_argument(
        '--models', nargs='+', required=True,
        metavar='NAME CKPT CFG',
        help='Triples of (name checkpoint config) repeated for each model, e.g. '
             '"MS+SR /path/model.pt /path/cfg.yaml FWD+EP-AR ..."')
    p.add_argument('--hdf5-dir', required=True,
                   help='Path to HDF5 dataset directory (train.hdf5 must exist)')
    p.add_argument('--sac-repo',
                   default='sdpkjc/Walker2d-v4-sac_continuous_action-seed4',
                   help='HuggingFace repo for SAC checkpoint (or use --sac-local)')
    p.add_argument('--sac-local', default=None,
                   help='Local path to SAC .cleanrl_model checkpoint (overrides --sac-repo)')
    p.add_argument('--perturb-step', type=int, default=50,
                   help='Episode step at which the perturbation is applied')
    p.add_argument('--recovery-steps', type=int, default=80,
                   help='Steps to simulate after the perturbation')
    p.add_argument('--perturb-type',
                   choices=['torso_vel', 'body_vel', 'joint_angle'],
                   default='torso_vel',
                   help='Type of perturbation to apply')
    p.add_argument('--perturb-mag', type=float, default=6.0,
                   help='Perturbation magnitude (rad/s for vel types, rad for angle)')
    p.add_argument('--trials', type=int, default=20,
                   help='Number of random trials per model')
    p.add_argument('--probe-episodes', type=int, default=200,
                   help='Episodes used for probe fitting')
    p.add_argument('--probe-epochs', type=int, default=30,
                   help='Training epochs for the MLP probe')
    p.add_argument('--rollout-steps', type=int, default=1,
                   help='Rollout horizon used in fit_walker_rollout_probe')
    p.add_argument('--image-size', type=int, default=64)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed-offset', type=int, default=0,
                   help='Added to trial index to get the random seed')
    p.add_argument('--out', required=True,
                   help='Output JSON path')
    return p.parse_args()


def parse_model_triples(raw: list[str]) -> list[tuple[str, str, str]]:
    """Parse flat list ['name', 'ckpt', 'cfg', ...] into triples."""
    if len(raw) % 3 != 0:
        raise ValueError(
            f'--models must be triples of (name ckpt cfg); got {len(raw)} tokens')
    return [(raw[i], raw[i + 1], raw[i + 2]) for i in range(0, len(raw), 3)]


def main():
    args   = parse_args()
    triples = parse_model_triples(args.models)

    # Load SAC policy
    from experiments.walker2d_ppo_utils import download_and_load_sac, load_sac_from_local
    if args.sac_local:
        print(f'[SAC] loading from {args.sac_local}')
        sac = load_sac_from_local(args.sac_local, device='cpu')
    else:
        print(f'[SAC] downloading from {args.sac_repo}')
        sac = download_and_load_sac(repo_id=args.sac_repo, device='cpu')

    results = {
        'config': {
            'perturb_step':   args.perturb_step,
            'recovery_steps': args.recovery_steps,
            'perturb_type':   args.perturb_type,
            'perturb_mag':    args.perturb_mag,
            'trials':         args.trials,
            'rollout_steps':  args.rollout_steps,
        },
        'models': [],
    }

    for name, ckpt_path, cfg_path in triples:
        print(f'\n{"="*60}')
        print(f'Model: {name}')
        print(f'  ckpt: {ckpt_path}')
        print(f'  cfg:  {cfg_path}')

        bundle = load_walker_bundle(ckpt_path, cfg_path, args.device)

        # Fit rollout probe calibrated for predictor outputs
        print(f'[probe] fitting on {args.probe_episodes} episodes '
              f'(rollout_steps={args.rollout_steps}) …')
        probe = fit_walker_rollout_probe(
            bundle,
            hdf5_dir=args.hdf5_dir,
            split='train',
            max_episodes=args.probe_episodes,
            rollout_steps=args.rollout_steps,
            image_size=args.image_size,
            n_epochs=args.probe_epochs,
        )

        visual_env = Walker2dVisual(image_size=args.image_size)

        trial_results = []
        for trial_idx in range(args.trials):
            seed = args.seed_offset + trial_idx
            print(f'  trial {trial_idx+1}/{args.trials}  seed={seed}', end='  ')
            t = run_trial(
                bundle=bundle,
                probe=probe,
                sac_policy=sac,
                visual_env=visual_env,
                seed=seed,
                perturb_step=args.perturb_step,
                recovery_steps=args.recovery_steps,
                perturb_type=args.perturb_type,
                perturb_mag=args.perturb_mag,
            )
            trial_results.append(t)
            print(f'mse={t["mean_state_mse"]:.4f}  '
                  f'xvel_mae={t["mean_xvel_mae"]:.4f}  '
                  f'surv_acc={t["survival_acc"]:.2f}')

        visual_env.close()

        # Summary across trials
        summary = {
            'mean_state_mse':       float(np.mean([r['mean_state_mse']     for r in trial_results])),
            'std_state_mse':        float(np.std( [r['mean_state_mse']     for r in trial_results])),
            'mean_xvel_mae':        float(np.mean([r['mean_xvel_mae']      for r in trial_results])),
            'std_xvel_mae':         float(np.std( [r['mean_xvel_mae']      for r in trial_results])),
            'mean_ang_mae':         float(np.mean([r['mean_ang_mae']       for r in trial_results])),
            'mean_action_mae':      float(np.mean([r['mean_action_mae']    for r in trial_results])),
            'mean_surv_acc':        float(np.mean([r['survival_acc']       for r in trial_results])),
            'mean_gt_surv_rate':    float(np.mean([r['gt_survival_rate']   for r in trial_results])),
            'mean_pred_surv_rate':  float(np.mean([r['pred_survival_rate'] for r in trial_results])),
            # Early-recovery quality (first 20 steps)
            'early_state_mse':      float(np.mean([
                np.mean(r['state_mse_per_step'][:20])
                for r in trial_results])),
            # Late-recovery quality (last 20 steps)
            'late_state_mse':       float(np.mean([
                np.mean(r['state_mse_per_step'][-20:])
                for r in trial_results])),
        }

        print(f'[{name}] summary: mse={summary["mean_state_mse"]:.4f}±{summary["std_state_mse"]:.4f}'
              f'  xvel_mae={summary["mean_xvel_mae"]:.4f}'
              f'  action_mae={summary["mean_action_mae"]:.4f}'
              f'  gt_surv={summary["mean_gt_surv_rate"]:.3f}'
              f'  pred_surv={summary["mean_pred_surv_rate"]:.3f}')

        results['models'].append({
            'name':       name,
            'checkpoint': ckpt_path,
            'config':     cfg_path,
            'trials':     trial_results,
            'summary':    summary,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n[done] results → {out_path}')

    # Print comparison table
    print('\n── Perturbation Recovery Summary (closed-loop: SAC on probe(z) vs SAC on real state) ──')
    print(f'{"Model":<20} {"MSE":>8} {"±":>6} {"xVel MAE":>10} {"Act MAE":>9} '
          f'{"GT surv":>8} {"Pred surv":>10} {"Surv acc":>9}')
    print('-' * 90)
    for m in results['models']:
        s = m['summary']
        print(f'{m["name"]:<20} {s["mean_state_mse"]:8.4f} {s["std_state_mse"]:6.4f} '
              f'{s["mean_xvel_mae"]:10.4f} {s["mean_action_mae"]:9.4f} '
              f'{s["mean_gt_surv_rate"]:8.3f} {s["mean_pred_surv_rate"]:10.3f} '
              f'{s["mean_surv_acc"]:9.3f}')


if __name__ == '__main__':
    main()
