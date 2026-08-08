"""Plotting helpers for the experiment scripts. All functions save a PDF/PNG
to `out_path` and also return the Figure in case the caller wants it."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
