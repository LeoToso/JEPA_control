"""Plotting helpers for the experiment scripts. All functions save a PDF/PNG
to `out_path` and also return the Figure in case the caller wants it."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def plot_training_curves(histories: dict[str, list[dict]], out_path: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, hist in histories.items():
        ax.plot([h["total"] for h in hist], label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("total loss")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Training loss")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return fig


def plot_eigenvalues(true_eigvals: np.ndarray, learned_eigvals: dict[str, np.ndarray], out_path: str):
    fig, ax = plt.subplots(figsize=(5, 5))
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=1, label="unit circle")
    ax.scatter(true_eigvals.real, true_eigvals.imag, marker="*", s=200, c="black", label="true A", zorder=5)
    for name, eigs in learned_eigvals.items():
        ax.scatter(eigs.real, eigs.imag, label=f"A_z ({name})", alpha=0.8)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title("Eigenvalues: true dynamics vs learned latent dynamics")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return fig


def plot_closed_loop_trajectories(eval_results: dict[str, dict], out_path: str, max_trials: int = 5):
    n = len(eval_results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, eval_results.items()):
        for xs in res["trajectories"][:max_trials]:
            norms = np.linalg.norm(xs, axis=1)
            norms = np.clip(norms, 1e-8, 1e6)
            ax.plot(norms)
        ax.set_yscale("log")
        ax.set_title(f"{name}\nsuccess={res['success_rate']*100:.0f}%")
        ax.set_xlabel("step")
    axes[0].set_ylabel("||x_t|| (log)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return fig


def plot_robustness_sweep(
    magnitudes: list[float],
    results: dict[str, dict[str, list[float]]],
    out_path: str,
    xlabel: str = "perturbation magnitude",
    title: str = "",
):
    """One line per checkpoint/config, one panel per metric, x-axis is the
    initial-state perturbation magnitude. `results[label]` must have
    "success_rate", "mean_fraction_stable", "final_state_distance_avg" lists
    (one entry per `magnitudes` value, e.g. from repeated `evaluate_controller`
    calls at growing ||x0||) -- makes a basin-of-attraction cliff (success
    dropping off past some perturbation size) visible as a single figure
    instead of a table of numbers."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    specs = [
        ("success_rate", "success rate (%)", 100.0),
        ("mean_fraction_stable", "mean fraction stable", 1.0),
        ("final_state_distance_avg", "final state distance (avg)", 1.0),
    ]
    for ax, (key, ylabel, scale) in zip(axes, specs):
        for label, res in results.items():
            vals = np.array(res[key], dtype=float) * scale
            vals = np.where(np.isfinite(vals), vals, np.nan)  # inf (no controller / blew up) -> gap, not an axis-breaking spike
            ax.plot(magnitudes, vals, marker="o", label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].set_ylim(-5, 105)
    axes[0].legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return fig


def plot_collapse_rate_sweep(
    horizons: list[int],
    results: dict[str, list[float]],
    out_path: str,
    title: str = "",
):
    """One line per config, x-axis is the multistep prediction horizon H,
    y-axis is the fraction of random-seed reruns (at that H) whose learned
    encoder ends up collapsing the unstable mode (unstable_mode_retention
    R^2 below a threshold). `results[label]` is a list of collapse rates in
    [0, 1], one entry per `horizons` value."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, rates in results.items():
        ax.plot(horizons, [r * 100 for r in rates], marker="o", label=label)
    ax.set_xlabel("prediction horizon H")
    ax.set_ylabel("unstable-mode collapse rate (%)")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return fig


# Shared marker style for the origin/equilibrium point, kept identical across all three
# panels (and readable against green/red, plasma, and RdBu_r backgrounds alike).
_EQUILIBRIUM_STYLE = dict(marker="*", s=220, c="white", edgecolors="black", linewidths=1.3, zorder=6)


def _axis_label(dim: int, suffix: str) -> str:
    return f"$x_{{{dim + 1}}}${suffix}"


def plot_local_stability_probe(
    eig_panel: dict,
    roa_panel: dict,
    lyap_panel: dict,
    out_path: str,
    unit_suffix: str = "",
):
    """The 3-panel local-stability probe: unstable-eigenvector alignment
    (true vs. learned, projected back to state space), empirical region of
    attraction under the learned closed-loop controller (with an optional
    oracle-LQR boundary overlay), and a Lyapunov-decrease certificate map --
    adapted from the real pixel-based project's `probe_local_stability_smwm.py`
    (panel 1 replaces its local vector field with a more direct alignment
    check: does the learned dynamics' fastest-growing direction actually
    point along the true unstable mode).

    Deliberately title-free (no per-panel title, no figure suptitle) --
    the quantitative summary (cosine similarities, success rates, decrease
    fraction) is printed to the console by the calling script instead,
    keeping the figure itself uncluttered."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    suffix = f" ({unit_suffix})" if unit_suffix else ""

    # Panel 1: unstable eigenvector alignment -- true (green) vs learned (orange),
    # both unit vectors, with the true stable eigenvector (grey, dashed) as a
    # reference so it's visually obvious if "learned" has aligned with the wrong mode.
    ax = axes[0]
    d0, d1 = eig_panel["dims"]
    # Thin shafts/heads so two nearly-parallel unit arrows don't visually
    # merge into one blob; "learned" is drawn semi-transparent (and on top)
    # so a perfectly-aligned overlap shows as a visibly blended color rather
    # than fully hiding "ground truth (unstable)" underneath.
    arrow_kwargs = dict(angles="xy", scale_units="xy", scale=1, width=0.012, headwidth=3.5, headlength=4.5, headaxislength=4)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="grey", linewidth=0.7, linestyle=":", alpha=0.6)
    if eig_panel.get("v_true_stable") is not None:
        vs = eig_panel["v_true_stable"]
        ax.quiver(0, 0, vs[0], vs[1], color="grey", alpha=0.7, label="ground truth (stable)", **arrow_kwargs)
    vu = eig_panel["v_true_unstable"]
    vl = eig_panel["v_learned"]
    ax.quiver(0, 0, vu[0], vu[1], color="green", label="ground truth (unstable)", zorder=4, **arrow_kwargs)
    ax.quiver(0, 0, vl[0], vl[1], color="darkorange", alpha=0.6, label="learned (dominant)", zorder=5, **arrow_kwargs)
    ax.scatter([0], [0], **_EQUILIBRIUM_STYLE)
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    ax.set_xlabel(_axis_label(d0, suffix))
    ax.set_ylabel(_axis_label(d1, suffix))
    ax.legend(fontsize=8, loc="upper right")
    ax.tick_params(labelsize=8)

    # Panel 2: empirical region of attraction.
    ax = axes[1]
    XX2, YY2 = roa_panel["XX"], roa_panel["YY"]
    d0r, d1r = roa_panel["dims"]
    ax.pcolormesh(XX2, YY2, roa_panel["success"], cmap="RdYlGn", vmin=0, vmax=1, shading="auto")
    legend_handles = [
        Patch(facecolor="green", edgecolor="none", label="success"),
        Patch(facecolor="red", edgecolor="none", label="failure"),
    ]
    if "success_gt" in roa_panel:
        gt = roa_panel["success_gt"]
        if not (np.all(gt > 0.5) or np.all(gt < 0.5)):
            ax.contour(XX2, YY2, gt, levels=[0.5], colors="black", linestyles="dashed", linewidths=1.5)
            legend_handles.append(Line2D([0], [0], color="black", linestyle="dashed", linewidth=1.5, label="oracle-LQR boundary"))
    ax.scatter([0], [0], **_EQUILIBRIUM_STYLE)
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right")
    ax.set_xlabel(_axis_label(d0r, suffix))
    ax.set_ylabel(_axis_label(d1r, suffix))
    ax.tick_params(labelsize=8)

    # Panel 3: Lyapunov decrease.
    ax = axes[2]
    XX3, YY3 = lyap_panel["XX"], lyap_panel["YY"]
    d0l, d1l = lyap_panel["dims"]
    vmax = float(np.percentile(np.abs(lyap_panel["delta_V"]), 95))
    vmax = vmax if vmax > 1e-12 else 1.0
    cf = ax.pcolormesh(XX3, YY3, lyap_panel["delta_V"], cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    ax.contour(XX3, YY3, lyap_panel["delta_V"], levels=[0.0], colors="black", linewidths=1.2)
    ax.scatter([0], [0], **_EQUILIBRIUM_STYLE)
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label=r"$\Delta V = V(x') - V(x)$")
    ax.set_xlabel(_axis_label(d0l, suffix))
    ax.set_ylabel(_axis_label(d1l, suffix))
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return fig


def plot_summary_bars(configs: list[str], r2_values: list[float], success_rates: list[float], out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(configs, r2_values, color="steelblue")
    axes[0].set_ylabel("unstable-mode retention R^2")
    axes[0].set_ylim(min(0, min(r2_values) - 0.1), 1.05)
    axes[0].axhline(0, color="grey", linewidth=0.5)
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(configs, [s * 100 for s in success_rates], color="darkorange")
    axes[1].set_ylabel("closed-loop success rate (%)")
    axes[1].set_ylim(0, 105)
    axes[1].tick_params(axis="x", rotation=30)

    fig.suptitle("Representation collapse vs closed-loop controllability")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return fig


def plot_norm_divergence(true_norms: np.ndarray, rec_norms: np.ndarray, out_path: str, title: str = ""):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(true_norms, label="||z_t|| (re-encoded truth)", color="tab:blue")
    ax.plot(rec_norms, "--", label="||z_t_hat|| (recursive latent rollout)", color="tab:orange")
    ax.set_xlabel("step")
    ax.set_ylabel("latent norm")
    ax.set_title(title or "Latent norm divergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return fig


def plot_five_panels(
    phase_panel: dict,
    pred_err_panel: dict,
    planning_panel: dict,
    true_norms: np.ndarray,
    rec_norms: np.ndarray,
    cos_over_time: np.ndarray,
    out_path: str,
    title: str = "",
    unit_suffix: str = "",
):
    """The 5-panel diagnostic figure: phase portrait, H-step prediction
    error (decoded to the original state space), planning cost (decoded to
    the original state space), latent norm divergence, and cosine alignment
    along the unstable eigenvector -- adapted from the real pixel-based
    project's `plot_checkpoint_summary_physical_smwm.py`.

    `unit_suffix` (e.g. "deg") is appended to the x[dim]/y[dim] axis labels
    of panels 1-3 for display only -- the panels' own numeric values (grid
    coordinates, drift vectors, error, cost) must already be in whatever
    unit that label names; this function does not convert anything."""
    fig, axes = plt.subplots(1, 5, figsize=(34, 6.5))
    suffix = f" ({unit_suffix})" if unit_suffix else ""

    # Panel 1: phase portrait -- true (green) vs learned (plasma) vector field.
    ax = axes[0]
    XX, YY = phase_panel["XX"], phase_panel["YY"]
    d0, d1 = phase_panel["dims"]
    ax.quiver(
        XX, YY, phase_panel["U_true"], phase_panel["V_true"],
        color="green", alpha=0.7, label="true", angles="xy",
    )
    mag = np.hypot(phase_panel["U_learned"], phase_panel["V_learned"])
    q = ax.quiver(
        XX, YY, phase_panel["U_learned"], phase_panel["V_learned"], mag,
        cmap="plasma", alpha=0.9, angles="xy",
    )
    fig.colorbar(q, ax=ax, fraction=0.046, pad=0.04, label="||learned drift||")
    ax.set_xlabel(f"x[{d0}]{suffix}")
    ax.set_ylabel(f"x[{d1}]{suffix}")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Phase portrait: true vs learned drift")

    # Panel 2: H-step prediction error, decoded to the original state space.
    ax = axes[1]
    XX2, YY2 = pred_err_panel["XX"], pred_err_panel["YY"]
    d0e, d1e = pred_err_panel["dims"]
    cf = ax.contourf(XX2, YY2, pred_err_panel["error"], levels=20, cmap="YlOrRd")
    ax.contour(XX2, YY2, pred_err_panel["error"], levels=8, colors="k", linewidths=0.3, alpha=0.5)
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel(f"x[{d0e}]{suffix}")
    ax.set_ylabel(f"x[{d1e}]{suffix}")
    ax.set_title(f"{pred_err_panel['H']}-step prediction error\n||D(f_H(z,0)) - s_GT|| (state space)")

    # Panel 3: planning cost, decoded to the original state space.
    ax = axes[2]
    XX3, YY3 = planning_panel["XX"], planning_panel["YY"]
    d0p, d1p = planning_panel["dims"]
    cf = ax.contourf(XX3, YY3, planning_panel["log_cost"], levels=8, cmap="RdYlBu_r")
    fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel(f"x[{d0p}]{suffix}")
    ax.set_ylabel(f"x[{d1p}]{suffix}")
    ax.set_title(f"log10 ||D(f_{planning_panel['H']}(z,0)) - s_goal||^2 (state space)")

    # Panel 4: latent norm divergence.
    ax = axes[3]
    ax.plot(true_norms, color="steelblue", label="||z_t|| (re-encoded truth)")
    ax.plot(rec_norms, "--", color="darkorange", label="||z_t_hat|| (recursive rollout)")
    ax.set_xlabel("step")
    ax.set_ylabel("latent norm")
    ax.legend(fontsize=8)
    ax.set_title("Latent norm divergence")

    # Panel 5: cosine alignment along the unstable eigenvector.
    ax = axes[4]
    mean = cos_over_time.mean(axis=0)
    std = cos_over_time.std(axis=0)
    steps = np.arange(cos_over_time.shape[1])
    ax.plot(steps, mean, color="#2166ac")
    ax.fill_between(steps, mean - std, mean + std, alpha=0.25, color="#2166ac")
    ax.axhline(1.0, color="grey", linewidth=0.5, linestyle=":")
    ax.axhline(0.0, color="grey", linewidth=0.5)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("step")
    ax.set_title("Cosine alignment (unstable direction)")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return fig


def plot_cosine_alignment(cos_over_time: np.ndarray, out_path: str, title: str = ""):
    fig, ax = plt.subplots(figsize=(5, 4))
    mean = cos_over_time.mean(axis=0)
    std = cos_over_time.std(axis=0)
    steps = np.arange(cos_over_time.shape[1])
    ax.plot(steps, mean, color="tab:green")
    ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color="tab:green")
    ax.axhline(1.0, color="grey", linewidth=0.5, linestyle=":")
    ax.axhline(0.0, color="grey", linewidth=0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("cos(delta z_true, delta z_pred)")
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(title or "Cosine alignment along the unstable eigenvector")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return fig
