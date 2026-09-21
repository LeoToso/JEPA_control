#!/usr/bin/env python
"""Compute fp_err, rho(A_z), and ||B_z|| for a set of frozen-encoder SMWM models.

Definition
----------
fp_err = ||f(z_eq, u=0) - z_eq||

where f is the learned one-step predictor and z_eq = encode(upright CartPole,
state=zeros) is the equilibrium latent.  fp_err measures whether the equilibrium
is a genuine fixed point of the latent dynamics.  Large fp_err means the
predictor does NOT map the equilibrium back to itself, so any LQR built on
A_z = df/dz|_{z_eq} is linearised around a fictitious operating point and will
fail in closed-loop even if rho(A_z) and ||B_z|| look reasonable.

Additional columns
------------------
rho(A_z) : spectral radius of the latent state-transition Jacobian.
            Should be > 1 (CartPole is open-loop unstable).
||B_z||  : Frobenius norm of the latent input Jacobian.
            Should be O(1); near-zero means the predictor is action-insensitive
            and DARE will produce catastrophic gains.

Usage
-----
python experiments/compute_fp_error_table.py \\
    [--device cuda] [--out results/fp_error_table.csv]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent))

import numpy as np

from sensorimotor_probe_utils import (
    equilibrium_latent, load_bundle, local_jacobians,
)

# ── model registry ────────────────────────────────────────────────────────────

MODELS = [
    {
        'label': 'MSpred_IBOT',
        'ckpt': '/mnt/t7shield/jepa_results/cartpole_smwm_ibot_rollout_act1_seed42/model_final.pt',
        'cfg':  'configs/cartpole_smwm_ibot_rollout_act1.yaml',
    },
    {
        'label': '1steppred_IBOT',
        'ckpt': '/mnt/t7shield/jepa_results/cartpole_smwm_ibot_fwd_act1_seed42/model_final.pt',
        'cfg':  'configs/cartpole_smwm_ibot_fwd_act1.yaml',
    },
    {
        'label': 'MSpred_Dinov2',
        'ckpt': '/mnt/t7shield/jepa_results/cartpole_smwm_dinov2_rollout_act1_seed42/model_final.pt',
        'cfg':  'configs/cartpole_smwm_dinov2_rollout_act1.yaml',
    },
    {
        'label': '1steppred_Dinov2',
        'ckpt': '/mnt/t7shield/jepa_results/cartpole_smwm_dinov2_fwd_act1_seed42/model_final.pt',
        'cfg':  'configs/cartpole_smwm_dinov2_fwd_act1.yaml',
    },
    {
        'label': 'MSpred_IBOT_projector_1stepAR',
        'ckpt': '/mnt/t7shield/jepa_results/cartpole_ibot_projector_ar_1step_act1_200ep_seed42/model_final.pt',
        'cfg':  'configs/cartpole_smwm_ibot_projector_ar_1step_act1.yaml',
    },
    {
        'label': 'MSpred_IBOT_projector_EndpointAR',
        'ckpt': '/mnt/t7shield/jepa_results/cartpole_ibot_projector_rollout_endpoint_inv_act1_200ep_seed42/model_final.pt',
        'cfg':  'configs/cartpole_smwm_ibot_projector_rollout_endpoint_inverse_act1.yaml',
    },
    {
        'label': '1steppred_IBOT_projector_EndpointAR',
        'ckpt': '/mnt/t7shield/jepa_results/cartpole_smwm_ibot_fwd_endpoint_inv_act1_seed42/model_final.pt',
        'cfg':  'configs/cartpole_smwm_ibot_fwd_endpoint_inverse_act1.yaml',
    },
    {
        'label': 'MSpred_IBOT_projector_SIGReg',
        'ckpt': '/mnt/t7shield/jepa_results/cartpole_ibot_projector_sigreg_act1_200ep_seed42/model_final.pt',
        'cfg':  'configs/cartpole_smwm_ibot_projector_sigreg_act1.yaml',
    },
    {
        'label': 'MSpred_Dinov2_projector_1stepAR',
        'ckpt': '/mnt/t7shield/jepa_results/cartpole_dinov2_projector_ar_1step_act1_200ep_seed42/model_final.pt',
        'cfg':  'configs/cartpole_smwm_dinov2_projector_ar_1step_act1.yaml',
    },
    {
        'label': 'MSpred_Dinov2_projector_EndpointAR',
        'ckpt': '/mnt/t7shield/jepa_results/cartpole_dinov2_projector_rollout_endpoint_inv_act1_200ep_seed42/model_final.pt',
        'cfg':  'configs/cartpole_smwm_dinov2_projector_rollout_endpoint_inverse_act1.yaml',
    },
    {
        'label': 'MSpred_Dinov2_projector_SIGReg',
        'ckpt': '/mnt/t7shield/jepa_results/cartpole_dinov2_projector_sigreg_act1_200ep_seed42/model_final.pt',
        'cfg':  'configs/cartpole_smwm_dinov2_projector_sigreg_act1.yaml',
    },
]


def _env_patch(bundle):
    if 'environment' not in bundle['env_cfg']:
        bundle['env_cfg']['environment'] = {
            'frame_skip': int(bundle['model_cfg'].get('frame_skip', 5)),
            'image_size': int(bundle['model_cfg'].get('image_size', 128)),
            'action_range': [-10, 10],
            'mass_cart': 1.0, 'mass_pole': 0.1,
            'pole_length': 0.5, 'gravity': 9.8, 'dt': 0.02,
        }


def _print_table(rows):
    header = ['Model', 'fp_err', 'rho(A_z)', '||B_z||', 'status']
    col_w  = [max(len(h), max(len(str(r[i])) for r in rows))
              for i, h in enumerate(header)]
    sep = '  '.join('-' * w for w in col_w)
    fmt = '  '.join(f'{{:<{w}}}' for w in col_w)
    print()
    print(fmt.format(*header))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print()


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', default=None,
                   help='Optional CSV output path')
    args = p.parse_args()

    rows = []
    for spec in MODELS:
        label = spec['label']
        print(f'[{label}] loading …', flush=True)
        try:
            bundle = load_bundle(spec['ckpt'], spec['cfg'], args.device)
        except Exception as e:
            print(f'  ERROR loading: {e}')
            rows.append((label, 'LOAD_ERR', '-', '-', str(e)[:40]))
            continue

        _env_patch(bundle)

        try:
            z_eq = equilibrium_latent(bundle)
            A_z, B_z, fp_err = local_jacobians(bundle, z=z_eq)
        except Exception as e:
            print(f'  ERROR computing Jacobians: {e}')
            rows.append((label, 'JAC_ERR', '-', '-', str(e)[:40]))
            continue

        rho    = float(np.max(np.abs(np.linalg.eigvals(A_z))))
        norm_B = float(np.linalg.norm(B_z))

        status = 'ok'
        if fp_err > 1.0:
            status = 'no_fixed_pt'
        elif rho < 1.0:
            status = 'stable_Az'

        print(f'  fp_err={fp_err:.4e}  rho(A_z)={rho:.4f}  ||B_z||={norm_B:.4f}  [{status}]')
        rows.append((label,
                     f'{fp_err:.4e}',
                     f'{rho:.4f}',
                     f'{norm_B:.4f}',
                     status))

    _print_table(rows)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['model', 'fp_err', 'rho_Az', 'norm_Bz', 'status'])
            for row in rows:
                w.writerow(row)
        print(f'[saved] {out}')


if __name__ == '__main__':
    main()
