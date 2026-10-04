#!/usr/bin/env python
"""Walker2d Poincaré stability analysis: GT vs MS+SR vs FWD+EP-AR.

Pipeline (12 phases):
  1. Load PPO policy (HuggingFace or local)
  2. Load MS+SR SMWM bundle
  3. Load FWD+EP-AR SMWM bundle
  4. Fit ridge probes for both models (z→obs17) from HDF5 train split
  5. Create GT helper env (Poincaré section: right hip angle zero-crossing)
  6. Collect GT Poincaré crossings via PPO rollout
  7. Collect MS+SR latent crossings (hip angle from decoded obs — no shadow env)
  8. Collect FWD+EP-AR latent crossings
  9. Estimate fixed points (mean crossing in gait space)
  10. Compute Poincaré Jacobians via finite differences
  11. Compute spectral radii and Floquet exponents
  12. Save results JSON + optional plots

Usage
-----
  MUJOCO_GL=egl python experiments/probe_poincare_smwm_walker.py \\
      --ms-sr-ckpt  /mnt/t7shield/jepa_results/walker2d_smwm_sigreg_rollout_act1_seed42/model_final.pt \\
      --fwd-ar-ckpt results/walker2d_smwm_fwd_endpoint_inverse_act1_seed42/model_final.pt \\
      --hdf5-dir    data/walker2d_fs5_64 \\
      --output-dir  results/poincare_walker2d
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch

from experiments.walker2d_ppo_utils   import (
    download_and_load_ppo, load_ppo_from_local,
    download_and_load_sac, load_sac_from_local,
)
from experiments.walker2d_utils  import (
    load_walker_bundle, fit_walker_ridge_probe, fit_walker_mlp_probe,
    gym_obs_to_mj_state, decode_z, latent_step,
)
from experiments.walker2d_poincare_utils import (
    WalkerMuJoCoHelper,
    collect_gt_poincare, collect_latent_poincare,
    build_poincare_map_gt, build_poincare_map_latent,
    poincare_jacobian_fd, spectral_radius, to_gait_state,
    GAIT_DIM,
)
from experiments.probe_utils import encode_obs


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Walker2d Poincaré stability: GT vs SMWM models.')

    # PPO policy
    g = p.add_argument_group('PPO policy')
    g.add_argument('--ppo-repo',
                   default='sdpkjc/Walker2d-v4-ppo_fix_continuous_action-seed3',
                   help='HuggingFace repo ID for PPO checkpoint')
    g.add_argument('--ppo-ckpt',  default=None,
                   help='Local PPO checkpoint (skip HF download if given)')
    g.add_argument('--ppo-device', default='cpu')

    # SAC policy (overrides PPO when provided)
    g = p.add_argument_group('SAC policy (overrides PPO)')
    g.add_argument('--sac-repo', default=None,
                   help='HuggingFace repo ID for SAC checkpoint (overrides PPO)')
    g.add_argument('--sac-ckpt', default=None,
                   help='Local SAC checkpoint path (overrides PPO)')

    # World model checkpoints
    g = p.add_argument_group('World models')
    g.add_argument('--ms-sr-ckpt',  required=True,
                   help='MS+SR model_final.pt')
    g.add_argument('--ms-sr-cfg',   default=None,
                   help='MS+SR env config yaml (auto-detected if None)')
    g.add_argument('--fwd-ar-ckpt', required=True,
                   help='FWD+EP-AR model_final.pt')
    g.add_argument('--fwd-ar-cfg',  default=None,
                   help='FWD+EP-AR env config yaml (auto-detected if None)')
    g.add_argument('--device', default='cuda',
                   help='Device for SMWM models')

    # Dataset
    g = p.add_argument_group('Dataset')
    g.add_argument('--hdf5-dir', required=True,
                   help='Directory with train/val/test.hdf5')
    g.add_argument('--probe-split', default='train')
    g.add_argument('--probe-episodes', type=int, default=300)
    g.add_argument('--probe-ridge', type=float, default=1e-3)
    g.add_argument('--image-size', type=int, default=64)

    # Probe type
    g = p.add_argument_group('Probe')
    g.add_argument('--use-ridge-probe', action='store_true',
                   help='Use linear ridge probe instead of MLP (faster, less accurate)')
    g.add_argument('--mlp-hidden',   type=int, default=128)
    g.add_argument('--mlp-epochs',   type=int, default=30)
    g.add_argument('--mlp-lr',       type=float, default=1e-3)
    g.add_argument('--mlp-batch',    type=int, default=512)

    # Rollout settings
    g = p.add_argument_group('Rollout')
    g.add_argument('--rollout-steps', type=int, default=5000,
                   help='GT/latent roll-out length (after warm-up)')
    g.add_argument('--warmup-steps',  type=int, default=50)
    g.add_argument('--rollout-seed',  type=int, default=42)
    g.add_argument('--min-crossings', type=int, default=5,
                   help='Skip Jacobian if fewer crossings found')

    # Jacobian
    g = p.add_argument_group('Jacobian')
    g.add_argument('--fd-eps',      type=float, default=1e-3,
                   help='Finite-difference step for Poincaré Jacobian')
    g.add_argument('--transient-crossings', type=int, default=2,
                   help='Skip first N crossings (transient) when selecting Jacobian point')

    # Output
    g = p.add_argument_group('Output')
    g.add_argument('--output-dir', default='results/poincare_walker2d')
    g.add_argument('--no-plots', action='store_true')

    return p.parse_args()


# ── config file auto-detection ────────────────────────────────────────────────

def find_cfg(ckpt_path: str) -> str:
    """Try to find env_config.yaml next to checkpoint."""
    p = Path(ckpt_path).parent
    for name in ['env_config.yaml', 'config.yaml', 'cfg.yaml']:
        if (p / name).exists():
            return str(p / name)
    # Fallback: return a minimal inline path that load_walker_bundle can tolerate
    raise FileNotFoundError(
        f'No env config yaml found in {p}. '
        'Pass --ms-sr-cfg / --fwd-ar-cfg explicitly.')


# ── encode initial latent ──────────────────────────────────────────────────────

@torch.no_grad()
def get_initial_latent(bundle: dict, helper: WalkerMuJoCoHelper,
                       obs17: np.ndarray, image_size: int = 64) -> torch.Tensor:
    """Encode a dummy initial frame pair into the model's latent space.

    We don't have access to the real previous frame here, so we render the
    current state twice (current obs used as both current and previous frame).
    """
    qpos, qvel = gym_obs_to_mj_state(obs17, x_pos=0.0)
    helper.set_state(qpos, qvel)
    frame = helper.render_frame()

    # If rendering unavailable, use zeros
    if frame is None:
        dummy = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        z = encode_obs(bundle, dummy, dummy, obs17)
    else:
        import torch.nn.functional as F_
        frame = np.ascontiguousarray(frame)   # remove negative strides
        h, w = frame.shape[:2]
        if h != image_size or w != image_size:
            t = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0)
            t = F_.interpolate(t, (image_size, image_size), mode='bilinear',
                               align_corners=False)
            frame = t[0].permute(1, 2, 0).byte().numpy()
        z = encode_obs(bundle, frame, frame, obs17)

    return z   # (1, latent_dim)


# ── Jacobian helpers ───────────────────────────────────────────────────────────

def compute_jacobian_gt(helper, policy, nominal_qpos, nominal_qvel,
                        fd_eps: float) -> tuple[np.ndarray | None, float | None]:
    def map_fn(g16: np.ndarray) -> np.ndarray | None:
        from experiments.walker2d_poincare_utils import from_gait_state
        qp, qv = from_gait_state(g16, x_pos=float(nominal_qpos[0]))
        return build_poincare_map_gt(helper, policy, qp, qv)

    g0 = to_gait_state(nominal_qpos, nominal_qvel)
    J  = poincare_jacobian_fd(map_fn, g0, eps=fd_eps)
    rho = spectral_radius(J) if J is not None else None
    return J, rho


def compute_jacobian_latent(bundle, ridge_probe, policy,
                             z_nominal: torch.Tensor,
                             fd_eps: float) -> tuple[np.ndarray | None, float | None]:
    """Finite-difference Poincaré Jacobian in gait space for latent model.

    Perturbs decoded gait state → re-encodes via decode only (no image), then rolls.
    """
    obs_nom = decode_z(z_nominal, ridge_probe)
    g0      = to_gait_state(*gym_obs_to_mj_state(obs_nom))

    def map_fn(g16: np.ndarray) -> np.ndarray | None:
        from experiments.walker2d_poincare_utils import from_gait_state
        from experiments.walker2d_utils import mj_state_to_gym_obs
        qp, qv = from_gait_state(g16, x_pos=0.0)
        obs_perturbed = mj_state_to_gym_obs(qp, qv)
        z_perturbed = encode_obs(bundle,
                                  np.zeros((64, 64, 3), dtype=np.uint8),
                                  np.zeros((64, 64, 3), dtype=np.uint8),
                                  obs_perturbed)
        return build_poincare_map_latent(bundle, ridge_probe, policy, z_perturbed)

    J   = poincare_jacobian_fd(map_fn, g0, eps=fd_eps)
    rho = spectral_radius(J) if J is not None else None
    return J, rho


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {'args': vars(args), 'models': {}}

    # ── Phase 1: Policy (SAC takes priority over PPO if args provided) ────────
    print('\n=== Phase 1: Load policy ===')
    t0 = time.time()
    if args.sac_ckpt:
        policy = load_sac_from_local(args.sac_ckpt, device=args.ppo_device)
        print(f'[SAC] loaded from local in {time.time()-t0:.1f}s')
    elif args.sac_repo:
        policy = download_and_load_sac(args.sac_repo, device=args.ppo_device)
        print(f'[SAC] loaded in {time.time()-t0:.1f}s')
    elif args.ppo_ckpt:
        policy = load_ppo_from_local(args.ppo_ckpt, device=args.ppo_device)
        print(f'[PPO] loaded in {time.time()-t0:.1f}s')
    else:
        policy = download_and_load_ppo(args.ppo_repo, device=args.ppo_device)
        print(f'[PPO] loaded in {time.time()-t0:.1f}s')

    # ── Phase 2-3: SMWM bundles ───────────────────────────────────────────────
    print('\n=== Phase 2: Load MS+SR bundle ===')
    ms_sr_cfg  = args.ms_sr_cfg  or find_cfg(args.ms_sr_ckpt)
    ms_bundle  = load_walker_bundle(args.ms_sr_ckpt,  ms_sr_cfg,  args.device)

    print('\n=== Phase 3: Load FWD+EP-AR bundle ===')
    fwd_ar_cfg = args.fwd_ar_cfg or find_cfg(args.fwd_ar_ckpt)
    fwd_bundle = load_walker_bundle(args.fwd_ar_ckpt, fwd_ar_cfg, args.device)

    # ── Phase 4: State probes (MLP by default, ridge as fallback) ────────────
    probe_kind = 'ridge' if args.use_ridge_probe else 'MLP'
    print(f'\n=== Phase 4: Fit {probe_kind} probes ===')
    if args.use_ridge_probe:
        ms_probe  = fit_walker_ridge_probe(ms_bundle,  args.hdf5_dir,
                                            args.probe_split,
                                            args.probe_episodes,
                                            args.probe_ridge,
                                            args.image_size)
        fwd_probe = fit_walker_ridge_probe(fwd_bundle, args.hdf5_dir,
                                            args.probe_split,
                                            args.probe_episodes,
                                            args.probe_ridge,
                                            args.image_size)
    else:
        ms_probe  = fit_walker_mlp_probe(ms_bundle,  args.hdf5_dir,
                                          args.probe_split,
                                          args.probe_episodes,
                                          args.image_size,
                                          hidden=args.mlp_hidden,
                                          n_epochs=args.mlp_epochs,
                                          lr=args.mlp_lr,
                                          batch_size=args.mlp_batch)
        fwd_probe = fit_walker_mlp_probe(fwd_bundle, args.hdf5_dir,
                                          args.probe_split,
                                          args.probe_episodes,
                                          args.image_size,
                                          hidden=args.mlp_hidden,
                                          n_epochs=args.mlp_epochs,
                                          lr=args.mlp_lr,
                                          batch_size=args.mlp_batch)

    # ── Phase 5: Environments ─────────────────────────────────────────────────
    print('\n=== Phase 5: Create environments ===')
    gt_helper     = WalkerMuJoCoHelper(image_size=args.image_size, render=False)
    render_helper = WalkerMuJoCoHelper(image_size=args.image_size, render=True,
                                       render_mode='rgb_array')

    # ── Phase 6: GT Poincaré crossings ───────────────────────────────────────
    print('\n=== Phase 6: Collect GT Poincaré crossings ===')
    t0 = time.time()
    gt_crossings, gt_all_gaits = collect_gt_poincare(
        gt_helper, policy,
        n_steps=args.rollout_steps,
        warmup_steps=args.warmup_steps,
        seed=args.rollout_seed,
    )
    print(f'[GT] {len(gt_crossings)} crossings in {time.time()-t0:.1f}s')
    results['models']['gt'] = {
        'label': 'GT MuJoCo',
        'n_crossings': len(gt_crossings),
        'gait_states': [c['gait'].tolist() for c in gt_crossings],
    }

    # ── Phase 7: MS+SR latent crossings ──────────────────────────────────────
    print('\n=== Phase 7: Collect MS+SR latent crossings ===')
    t0 = time.time()
    obs0, _ = gt_helper.env.reset(seed=args.rollout_seed)
    obs0    = obs0.astype(np.float32)
    z0_ms   = get_initial_latent(ms_bundle,  render_helper, obs0, args.image_size)

    ms_crossings, ms_all_gaits = collect_latent_poincare(
        ms_bundle, ms_probe, policy,
        z0=z0_ms,
        max_steps=args.rollout_steps,
        warmup_steps=args.warmup_steps,
    )
    print(f'[MS+SR] {len(ms_crossings)} crossings in {time.time()-t0:.1f}s')
    results['models']['ms_sr'] = {
        'label': 'MS+SR',
        'n_crossings': len(ms_crossings),
        'gait_states': [c['gait'].tolist() for c in ms_crossings],
    }

    # ── Phase 8: FWD+EP-AR latent crossings ──────────────────────────────────
    print('\n=== Phase 8: Collect FWD+EP-AR latent crossings ===')
    t0 = time.time()
    z0_fwd = get_initial_latent(fwd_bundle, render_helper, obs0, args.image_size)

    fwd_crossings, fwd_all_gaits = collect_latent_poincare(
        fwd_bundle, fwd_probe, policy,
        z0=z0_fwd,
        max_steps=args.rollout_steps,
        warmup_steps=args.warmup_steps,
    )
    print(f'[FWD+EP-AR] {len(fwd_crossings)} crossings in {time.time()-t0:.1f}s')
    results['models']['fwd_ar'] = {
        'label': 'FWD+EP-AR',
        'n_crossings': len(fwd_crossings),
        'gait_states': [c['gait'].tolist() for c in fwd_crossings],
    }

    # ── Phase 9: Fixed points (mean crossing) ─────────────────────────────────
    print('\n=== Phase 9: Estimate fixed points ===')

    def estimate_fixed_point(crossings: list[dict]) -> np.ndarray | None:
        if len(crossings) < 2:
            return None
        gaits = np.stack([c['gait'] for c in crossings])
        return gaits.mean(axis=0)

    gt_fp   = estimate_fixed_point(gt_crossings)
    ms_fp   = estimate_fixed_point(ms_crossings)
    fwd_fp  = estimate_fixed_point(fwd_crossings)

    for name, fp in [('GT', gt_fp), ('MS+SR', ms_fp), ('FWD+EP-AR', fwd_fp)]:
        if fp is not None:
            print(f'[{name}] fixed point z(z,ang,...)={fp[:3]}')

    # ── Phase 10: Poincaré Jacobians ──────────────────────────────────────────
    print('\n=== Phase 10: Compute Poincaré Jacobians ===')

    # Use crossings after the transient for Jacobian — pick the last settled one
    tc = args.transient_crossings

    def _pick_nominal(crossings: list[dict]) -> tuple[int, dict] | tuple[None, None]:
        """Return (idx, crossing) closest to the centroid of settled crossings.

        Using the centroid-nearest point rather than the last crossing gives a
        more representative fixed-point estimate, especially when crossings are
        scattered (FWD+EP-AR) or very few (GT at high walking speed).
        """
        settled = crossings[tc:]
        if not settled:
            return None, None
        gaits    = np.stack([c['gait'] for c in settled])
        centroid = gaits.mean(axis=0)
        dists    = np.linalg.norm(gaits - centroid, axis=1)
        best_local = int(np.argmin(dists))
        nom_idx    = tc + best_local
        return nom_idx, crossings[nom_idx]

    # GT Jacobian at nominal crossing
    gt_J = gt_rho = None
    if gt_fp is not None and len(gt_crossings) >= args.min_crossings:
        nom_idx, nom_c = _pick_nominal(gt_crossings)
        if nom_c is None:
            print('[GT] not enough settled crossings — skipping Jacobian')
        else:
            print(f'[GT] computing Jacobian at crossing {nom_idx} '
                  f'(step {nom_c["step"]}) ...')
            t0 = time.time()
            gt_J, gt_rho = compute_jacobian_gt(
                gt_helper, policy, nom_c['qpos'], nom_c['qvel'], args.fd_eps)
            if gt_rho is not None:
                print(f'[GT] ρ={gt_rho:.4f}  ({time.time()-t0:.1f}s)')
            else:
                print(f'[GT] Jacobian failed (perturbations ended episode)  ({time.time()-t0:.1f}s)')
    else:
        print('[GT] not enough crossings — skipping Jacobian')

    # MS+SR Jacobian
    ms_J = ms_rho = None
    if len(ms_crossings) >= args.min_crossings:
        nom_idx, nom_c = _pick_nominal(ms_crossings)
        if nom_c is not None:
            nom_z = nom_c['z']
            print(f'[MS+SR] computing Jacobian at crossing {nom_idx} ...')
            t0 = time.time()
            ms_J, ms_rho = compute_jacobian_latent(
                ms_bundle, ms_probe, policy, nom_z, args.fd_eps)
            if ms_rho is not None:
                print(f'[MS+SR] ρ={ms_rho:.4f}  ({time.time()-t0:.1f}s)')
            else:
                print(f'[MS+SR] Jacobian failed  ({time.time()-t0:.1f}s)')
    else:
        print('[MS+SR] not enough crossings — skipping Jacobian')

    # FWD+EP-AR Jacobian
    fwd_J = fwd_rho = None
    if len(fwd_crossings) >= args.min_crossings:
        nom_idx, nom_c = _pick_nominal(fwd_crossings)
        if nom_c is not None:
            nom_z = nom_c['z']
            print(f'[FWD+EP-AR] computing Jacobian at crossing {nom_idx} ...')
            t0 = time.time()
            fwd_J, fwd_rho = compute_jacobian_latent(
                fwd_bundle, fwd_probe, policy, nom_z, args.fd_eps)
            if fwd_rho is not None:
                print(f'[FWD+EP-AR] ρ={fwd_rho:.4f}  ({time.time()-t0:.1f}s)')
            else:
                print(f'[FWD+EP-AR] Jacobian failed  ({time.time()-t0:.1f}s)')
    else:
        print('[FWD+EP-AR] not enough crossings — skipping Jacobian')

    # ── Phase 11: Floquet exponents ───────────────────────────────────────────
    print('\n=== Phase 11: Floquet exponents ===')

    def floquet_exponents(J: np.ndarray) -> list[float]:
        """Floquet exponents = log|eigenvalues(J)| (negative ↔ stable mode)."""
        eigs = np.linalg.eigvals(J)
        return sorted(np.log(np.abs(eigs)).tolist(), reverse=True)

    for tag, J, rho in [
        ('GT',      gt_J,  gt_rho),
        ('MS+SR',   ms_J,  ms_rho),
        ('FWD+EP-AR', fwd_J, fwd_rho),
    ]:
        if J is not None:
            floq = floquet_exponents(J)
            print(f'[{tag}]  spectral radius ρ={rho:.4f}  '
                  f'stable={rho < 1}  '
                  f'top-3 Floquet={floq[:3]}')
            key = {'GT': 'gt', 'MS+SR': 'ms_sr', 'FWD+EP-AR': 'fwd_ar'}[tag]
            results['models'][key].update({
                'spectral_radius':   rho,
                'stable':            bool(rho < 1),
                'floquet_exponents': floq,
                'jacobian':          J.tolist(),
                'fixed_point':       (gt_fp if tag == 'GT'
                                      else ms_fp if tag == 'MS+SR'
                                      else fwd_fp).tolist(),
            })

    # ── Phase 12: Save results ────────────────────────────────────────────────
    print('\n=== Phase 12: Save results ===')
    json_path = out_dir / 'poincare_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[saved] {json_path}')

    # Save gait crossing arrays (sparse — for Jacobian / PCA)
    np.save(out_dir / 'gt_gait_crossings.npy',
            np.array([c['gait'] for c in gt_crossings]))
    np.save(out_dir / 'ms_sr_gait_crossings.npy',
            np.array([c['gait'] for c in ms_crossings]))
    np.save(out_dir / 'fwd_ar_gait_crossings.npy',
            np.array([c['gait'] for c in fwd_crossings]))

    # Save dense all-step gait arrays (for HALO-style distribution plots)
    np.save(out_dir / 'gt_all_gaits.npy',    gt_all_gaits)
    np.save(out_dir / 'ms_sr_all_gaits.npy', ms_all_gaits)
    np.save(out_dir / 'fwd_ar_all_gaits.npy', fwd_all_gaits)

    if not args.no_plots:
        try:
            from experiments.plot_poincare_walker import plot_all
            plot_all(results, gt_crossings, ms_crossings, fwd_crossings, out_dir,
                     gt_all_gaits=gt_all_gaits,
                     ms_all_gaits=ms_all_gaits,
                     fwd_all_gaits=fwd_all_gaits)
        except Exception as e:
            import traceback
            print(f'[plot] warning: {e}')
            traceback.print_exc()

    # Cleanup
    for h in [gt_helper, render_helper]:
        h.close()

    print('\n=== Done ===')
    print(f'Results in {out_dir}')


if __name__ == '__main__':
    main()
