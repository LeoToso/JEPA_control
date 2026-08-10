"""All-pairs eigenvector-alignment probe: computes and plots
`probe_local_stability.py`'s panel 1 (true dominant eigenvector vs. the
learned predictor's own dominant eigenvector, mapped back to state space
and projected to unit length) for EVERY pair of state dimensions, not just
one fixed --dims choice -- a full picture of where the learned and true
directions do (and don't) align across the whole state space, rather than
a single 2D slice of it.

For an n-state system this produces C(n,2) subplots (e.g. 6 for the
4-state cartpole: every combination of cart position, cart velocity, pole
angle, pole angular velocity).

    python experiments/probe_eigenvector_alignment_grid.py \\
        --checkpoint results/example5_actrecon_naive/checkpoint_actrecon_naive_H8_seed0.pt
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from checkpoint_io import load_checkpoint_with_env
from jepa_lds.diagnostics import unstable_eigenvector_alignment
from jepa_lds.plotting import plot_eigenvector_alignment_grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out", type=str, default=None, help="output PDF path (default: alongside the checkpoint)")
    p.add_argument("--ncols", type=int, default=3, help="subplot grid columns")
    args = p.parse_args()

    system, obs_model, encoder, predictor, _decoder, cfg, extra = load_checkpoint_with_env(args.checkpoint)

    if system.is_open_loop_unstable():
        dominant_mode_label, other_mode_label = "ground truth (unstable)", "ground truth (stable)"
    else:
        # Both modes are actually stable (e.g. Example 4's double_mode_stable) --
        # calling the dominant one "unstable" would be wrong, so number them instead.
        dominant_mode_label, other_mode_label = "ground truth (stable 2)", "ground truth (stable 1)"

    print(f"checkpoint: {args.checkpoint}")
    print(f"  system={system.name}  n={system.n}  config={extra.get('config_name', '?')}  trainer={extra.get('trainer', '?')}")

    dim_pairs = list(itertools.combinations(range(system.n), 2))
    print(f"  {len(dim_pairs)} dim pairs: {dim_pairs}")

    eig_panels = []
    for dims in dim_pairs:
        panel = unstable_eigenvector_alignment(system, obs_model, encoder, predictor, dims=dims)
        eig_panels.append(panel)
        print(f"  dims={dims}  cos_sim(learned, dominant) = {panel['cos_sim_unstable']:.4f}")

    out_path = args.out or os.path.splitext(args.checkpoint)[0] + "_eigenvector_alignment_grid.pdf"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plot_eigenvector_alignment_grid(
        eig_panels, out_path, dominant_mode_label=dominant_mode_label, other_mode_label=other_mode_label, ncols=args.ncols,
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
