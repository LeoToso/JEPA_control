"""Probe a saved linear-system checkpoint without retraining.

Run from the repository root:
    python experiments/probe_unstable_subspace.py --checkpoint PATH.pt

For CartPole, nonstable_dim should be 3. Check rank_F_U AND the singular
values/tolerance sensitivity. A passing rank test does not establish
stability of the LQR designed from the learned predictor.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from checkpoint_io import load_checkpoint_with_env
from jepa_lds.subspace import nonstable_subspace_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--spectral-tol", type=float, default=1e-7)
    parser.add_argument("--rank-rtol", type=float, default=1e-6)
    parser.add_argument("--rank-atol", type=float, default=0.0)
    args = parser.parse_args()

    system, obs, encoder, _predictor, decoder, cfg, extra = (
        load_checkpoint_with_env(args.checkpoint)
    )
    # Do not certify a rank using the helper's historical observation-map
    # fallback: different C can change this diagnostic completely.
    required = {"obs_dim_signal", "n_distractor", "obs_seed"}
    if not required.issubset(extra):
        parser.error(
            "Checkpoint lacks recorded observation-map metadata; recover "
            "the actual C/observation settings before checking rank."
        )

    # These checkpoints use LinearEncoder. Extract W directly in float64
    # rather than evaluating float32 basis observations. Nuisance channels
    # are independent of the physical state, so only signal columns enter F.
    W = encoder.linear.weight.detach().cpu().numpy().astype(np.float64)
    F = W[:, :obs.signal_dim] @ np.asarray(obs.C, dtype=np.float64)
    report = nonstable_subspace_report(
        system.A, F, spectral_tol=args.spectral_tol,
        rank_rtol=args.rank_rtol, rank_atol=args.rank_atol,
    )
    report.update({
        "checkpoint": args.checkpoint,
        "system": system.name,
        "horizon": int(cfg.horizon),
        "decoder_class": type(decoder).__name__ if decoder is not None else None,
        "interpretation": (
            "Tests injectivity on the nonstable generalized invariant subspace; "
            "does not certify exact action reconstruction or controller stability."
        ),
    })
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
