"""Visualization for MPC rollouts in the learned latent space."""
from __future__ import annotations
import warnings
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


def visualize_mpc_rollout(
    result: Dict,
    out_path: str | Path,
    n_frames: int = 8,
    state_labels: Optional[List[str]] = None,
    title: str = "",
) -> None:
    """Save a PNG visualizing an MPC rollout.

    Layout
    ------
    Row 0 : n_frames observed RGB frames from the actual rollout
    Row 1 : predicted ||Δz|| over the planning horizon at each shown frame
    Row 2 : physical state trajectory (theta, x) + applied actions

    Frames are subsampled evenly from the saved re-planning events.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        warnings.warn("matplotlib unavailable; skipping MPC visualization")
        return

    if state_labels is None:
        state_labels = ["x", "x_dot", "θ", "θ_dot"]

    frames: List[Dict] = result.get("frames", [])
    states: np.ndarray = np.asarray(result.get("states", [[0, 0, 0, 0]]))
    actions: np.ndarray = np.asarray(result.get("actions", [[0]]))
    T = len(states)

    if len(frames) == 0:
        warnings.warn("No frames in result — run rollout with save_frames=True")
        return

    n_show = min(n_frames, len(frames))
    sel_idx = np.linspace(0, len(frames) - 1, n_show, dtype=int)
    sel = [frames[i] for i in sel_idx]

    fig = plt.figure(figsize=(3.2 * n_show, 9))
    gs = gridspec.GridSpec(3, n_show, figure=fig, hspace=0.55, wspace=0.25)

    # ── Row 0: observed frames ────────────────────────────────────────────────
    for col, fd in enumerate(sel):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(fd["obs"])
        stabilized_marker = "✓" if result.get("stabilized") else ""
        ax.set_title(f"t={fd['t']} {stabilized_marker}", fontsize=9)
        ax.axis("off")

    # ── Row 1: predicted ||Δz|| along planning horizon ────────────────────────
    for col, fd in enumerate(sel):
        ax = fig.add_subplot(gs[1, col])
        pred = fd["pred_states"]          # (H+1, d)
        dz = np.linalg.norm(pred - pred[0:1], axis=1)
        ax.plot(dz, color="steelblue", linewidth=1.5)
        ax.fill_between(range(len(dz)), dz, alpha=0.18, color="steelblue")
        ax.set_xlabel("steps ahead", fontsize=7)
        if col == 0:
            ax.set_ylabel("‖Δz‖", fontsize=8)
        ax.set_title("planned ‖Δz‖", fontsize=8)
        ax.tick_params(labelsize=6)

    # ── Row 2 left: state trajectory ─────────────────────────────────────────
    t_ax = np.arange(T)
    ax_s = fig.add_subplot(gs[2, : n_show // 2])
    colors = ["tab:blue", "tab:cyan", "tab:red", "tab:orange"]
    for i, (lbl, col) in enumerate(zip(state_labels, colors)):
        if i < states.shape[1]:
            ax_s.plot(t_ax, states[:, i], label=lbl, color=col, linewidth=1.2)
    ax_s.axhline(0, color="k", linestyle="--", linewidth=0.5)
    ax_s.set_xlabel("timestep")
    ax_s.set_ylabel("state")
    ax_s.legend(fontsize=7, loc="upper right", ncol=2)
    ax_s.set_title("State trajectory")

    # ── Row 2 right: applied actions + re-planning events ────────────────────
    ax_a = fig.add_subplot(gs[2, n_show // 2 :])
    ax_a.plot(t_ax, actions[:, 0], color="darkorange", linewidth=1.2, label="u")
    ax_a.axhline(0, color="k", linestyle="--", linewidth=0.5)
    for fd in frames:
        ax_a.axvline(fd["t"], color="gray", linestyle=":", linewidth=0.4, alpha=0.5)
    ax_a.set_xlabel("timestep")
    ax_a.set_ylabel("action u")
    ax_a.set_title("Control input (│ = re-plan)")
    ax_a.legend(fontsize=7)

    stabilized = result.get("stabilized", False)
    final_err = result.get("final_state_error", float("nan"))
    settling = result.get("settling_time", "?")
    sup = (
        f"{title}  |  stabilized={'yes' if stabilized else 'no'}"
        f"  final_err={final_err:.3f}  settling_t={settling}"
    )
    fig.suptitle(sup.strip("  |  "), fontsize=10)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[vis] Saved MPC visualization → {out_path}")
