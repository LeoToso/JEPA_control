#!/usr/bin/env python
"""Poincaré stability plots for Walker2d GT vs SMWM models.

Generates:
  1. poincare_crossings_2d.pdf  — 2-D projections of gait-space crossings
  2. spectral_radius_bar.pdf    — spectral radius bar chart
  3. floquet_spectrum.pdf       — Floquet exponent distribution
  4. eigenvalue_complex.pdf     — complex plane eigenvalue scatter
  5. gait_pca.pdf               — PCA of crossing cloud (GT vs models)

Usage (standalone):
  python experiments/plot_poincare_walker.py \\
      --results-dir results/poincare_walker2d
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Computer Modern Roman', 'DejaVu Serif'],
    'mathtext.fontset':  'cm',
    'axes.unicode_minus': False,
    'axes.labelsize':    11,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   9,
})

COLORS  = {'GT MuJoCo': '#1f77b4', 'MS+SR': '#ff7f0e', 'FWD+EP-AR': '#2ca02c'}
MARKERS = {'GT MuJoCo': 'o',       'MS+SR': 's',       'FWD+EP-AR': '^'}

# Gait-state axis labels (16-D)
GAIT_LABELS = [
    'z', 'θ_torso', 'θ_r_hip', 'θ_r_knee', 'θ_r_ankle',
    'θ_l_hip', 'θ_l_knee', 'θ_l_ankle',
    'ẋ', 'ż', 'ω_torso', 'ω_r_hip', 'ω_r_knee', 'ω_r_ankle',
    'ω_l_hip', 'ω_l_knee',
]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_arrays(results_dir: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key, fname in [
        ('GT MuJoCo', 'gt_gait_crossings.npy'),
        ('MS+SR',     'ms_sr_gait_crossings.npy'),
        ('FWD+EP-AR', 'fwd_ar_gait_crossings.npy'),
    ]:
        p = results_dir / fname
        if p.exists():
            arr = np.load(p)
            if arr.ndim == 2 and arr.shape[0] > 0:
                arrays[key] = arr
            else:
                print(f'[plot] {fname}: empty — skipping {key}')
        else:
            print(f'[plot] {fname} not found — skipping {key}')
    return arrays


def load_results(results_dir: Path) -> dict:
    p = results_dir / 'poincare_results.json'
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


# ── Figure 1: 2-D crossing projections ───────────────────────────────────────

def plot_crossings_2d(arrays: dict[str, np.ndarray], out_path: Path) -> None:
    pairs = [(0, 8), (0, 1), (8, 9)]   # (z, ẋ), (z, θ_torso), (ẋ, ż)
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.5 * len(pairs), 3.8))

    for ax, (xi, yi) in zip(axes, pairs):
        for label, arr in arrays.items():
            ax.scatter(arr[:, xi], arr[:, yi],
                       c=COLORS[label], marker=MARKERS[label],
                       s=18, alpha=0.7, label=label, linewidths=0)
        ax.set_xlabel(GAIT_LABELS[xi])
        ax.set_ylabel(GAIT_LABELS[yi])
        ax.set_title(f'{GAIT_LABELS[xi]} vs {GAIT_LABELS[yi]}')
        ax.grid(True, lw=0.4, alpha=0.4)

    handles = [Line2D([0], [0], color=COLORS[l], marker=MARKERS[l],
                      linestyle='', ms=6, label=l)
               for l in arrays]
    axes[0].legend(handles=handles, frameon=True, loc='best')

    fig.suptitle('Poincaré Section Crossings (right-hip zero-crossing)',
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {out_path}')


# ── Figure 2: Spectral radius bar chart ──────────────────────────────────────

def plot_spectral_radius(results: dict, out_path: Path) -> None:
    model_keys = [
        ('gt',     'GT MuJoCo'),
        ('ms_sr',  'MS+SR'),
        ('fwd_ar', 'FWD+EP-AR'),
    ]
    labels, rhos, colors = [], [], []
    for key, label in model_keys:
        entry = results.get('models', {}).get(key, {})
        rho   = entry.get('spectral_radius')
        if rho is not None:
            labels.append(label)
            rhos.append(rho)
            colors.append(COLORS.get(label, 'gray'))

    if not labels:
        print('[plot] no spectral radii available — skipping bar chart')
        return

    fig, ax = plt.subplots(figsize=(max(3.5, 1.5 * len(labels)), 3.2))
    xs = np.arange(len(labels))
    bars = ax.bar(xs, rhos, color=colors, width=0.55, edgecolor='black', lw=0.8)
    ax.axhline(1.0, color='crimson', lw=1.5, ls='--', label='ρ = 1 (stability boundary)')

    for bar, rho in zip(bars, rhos):
        ax.text(bar.get_x() + bar.get_width() / 2, rho + 0.01,
                f'{rho:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Spectral radius ρ (Poincaré map)')
    ax.set_title('Local stability at periodic gait', fontsize=11)
    ax.legend()
    ax.set_ylim(0, max(max(rhos) * 1.15, 1.2))
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {out_path}')


# ── Figure 3: Floquet exponent distribution ───────────────────────────────────

def plot_floquet(results: dict, out_path: Path) -> None:
    model_keys = [
        ('gt',     'GT MuJoCo'),
        ('ms_sr',  'MS+SR'),
        ('fwd_ar', 'FWD+EP-AR'),
    ]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for key, label in model_keys:
        entry = results.get('models', {}).get(key, {})
        floq  = entry.get('floquet_exponents')
        if floq:
            xs = np.arange(len(floq))
            ax.plot(xs, floq, color=COLORS.get(label, 'gray'),
                    marker=MARKERS.get(label, 'o'), ms=5, label=label, lw=1.5)

    ax.axhline(0.0, color='crimson', lw=1.2, ls='--', label='0 (neutral)')
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Floquet exponent log|λ_i|')
    ax.set_title('Floquet spectrum of Poincaré map', fontsize=11)
    ax.legend()
    ax.grid(True, lw=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {out_path}')


# ── Figure 4: Complex eigenvalue scatter ─────────────────────────────────────

def plot_eigenvalues(results: dict, out_path: Path) -> None:
    model_keys = [
        ('gt',     'GT MuJoCo'),
        ('ms_sr',  'MS+SR'),
        ('fwd_ar', 'FWD+EP-AR'),
    ]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    theta = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta),
            'k--', lw=1.0, alpha=0.5, label='unit circle')

    for key, label in model_keys:
        entry = results.get('models', {}).get(key, {})
        J     = entry.get('jacobian')
        if J is None:
            continue
        eigs = np.linalg.eigvals(np.array(J))
        ax.scatter(eigs.real, eigs.imag,
                   c=COLORS.get(label, 'gray'),
                   marker=MARKERS.get(label, 'o'),
                   s=30, alpha=0.85, label=label, edgecolors='none')

    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('Re(λ)')
    ax.set_ylabel('Im(λ)')
    ax.set_title('Poincaré-map eigenvalues', fontsize=11)
    ax.set_aspect('equal')
    ax.legend()
    ax.grid(True, lw=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {out_path}')


# ── Figure 5: PCA of crossing clouds ─────────────────────────────────────────

def plot_pca(arrays: dict[str, np.ndarray], out_path: Path) -> None:
    if not arrays:
        return

    all_data = np.concatenate(list(arrays.values()), axis=0)
    mu   = all_data.mean(axis=0)
    cov  = np.cov((all_data - mu).T)
    _, V = np.linalg.eigh(cov)
    V    = V[:, ::-1]   # largest first

    fig, ax = plt.subplots(figsize=(5, 4))
    for label, arr in arrays.items():
        proj = (arr - mu) @ V[:, :2]
        ax.scatter(proj[:, 0], proj[:, 1],
                   c=COLORS.get(label, 'gray'),
                   marker=MARKERS.get(label, 'o'),
                   s=18, alpha=0.7, label=label, linewidths=0)

    ax.set_xlabel('PC 1 (gait space)')
    ax.set_ylabel('PC 2 (gait space)')
    ax.set_title('Poincaré crossings — PCA projection', fontsize=11)
    ax.legend()
    ax.grid(True, lw=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'[saved] {out_path}')


# ── Figure 6: HALO-style 2×3 scatter (GT outer cloud vs model inner cluster) ──
#
# Uses DENSE per-step gait states (not just Poincaré crossings) so that all
# models have equal sample counts and the comparison is purely distributional.

_HALO_PAIRS = [
    # Row 1 – joint positions
    (0, 1),   # z  vs θ_torso
    (0, 2),   # z  vs θ_r_hip
    (2, 5),   # θ_r_hip vs θ_l_hip
    # Row 2 – velocities / mixed
    (0, 8),   # z  vs ẋ
    (8, 9),   # ẋ  vs ż
    (11, 14), # ω_r_hip vs ω_l_hip
]

# subsample GT to at most this many points per panel to keep files small
_GT_MAX_PLOT = 2000


def plot_halo_dense(gt_arr: np.ndarray,
                    model_arr: np.ndarray,
                    model_label: str,
                    out_path: Path,
                    rng: np.random.Generator | None = None) -> None:
    """HALO-style 2×3 state-distribution comparison (dense, per-step sampling).

    Blue small dots  = GT rollout states (outer reference cloud).
    Red larger dots  = decoded latent-model states (inner/shifted cluster).
    White background, no grid, spines only on bottom/left, legend in panel 0.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.0 * n_cols, 3.4 * n_rows),
                             facecolor='white')

    # subsample GT so it doesn't visually swamp the model points
    gt_idx = (rng.choice(len(gt_arr), size=min(_GT_MAX_PLOT, len(gt_arr)),
                          replace=False)
              if len(gt_arr) > _GT_MAX_PLOT else np.arange(len(gt_arr)))
    gt_plot = gt_arr[gt_idx]

    # subsample model similarly for visual balance
    if len(model_arr) > _GT_MAX_PLOT:
        m_idx = rng.choice(len(model_arr), size=_GT_MAX_PLOT, replace=False)
        m_plot = model_arr[m_idx]
    else:
        m_plot = model_arr

    for ax_idx, (xi, yi) in enumerate(_HALO_PAIRS):
        ax = axes[ax_idx // n_cols, ax_idx % n_cols]
        ax.set_facecolor('white')

        ax.scatter(gt_plot[:, xi], gt_plot[:, yi],
                   c='#4c72b0', s=8, alpha=0.4, linewidths=0,
                   label='GT MuJoCo' if ax_idx == 0 else None,
                   zorder=1)

        if len(m_plot) > 0:
            ax.scatter(m_plot[:, xi], m_plot[:, yi],
                       c='#c44e52', s=18, alpha=0.7, linewidths=0,
                       label=model_label if ax_idx == 0 else None,
                       zorder=2)

        ax.set_xlabel(GAIT_LABELS[xi], fontsize=9)
        ax.set_ylabel(GAIT_LABELS[yi], fontsize=9)
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_linewidth(0.7)

        if ax_idx == 0:
            ax.legend(frameon=False, fontsize=8, loc='best',
                      handletextpad=0.4, borderpad=0.2)

    n_gt  = len(gt_arr)
    n_mod = len(model_arr)
    fig.suptitle(f'Gait-state distribution: GT ({n_gt} steps) vs {model_label} ({n_mod} steps)',
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'[saved] {out_path}')


# ── Figure 7: Limit-cycle trajectories ───────────────────────────────────────
#
# Plots the full dense gait trajectory as a time-colored line in three 2-D
# projections so the closed-loop orbit shape is visible rather than just the
# section crossings.  Layout: rows = projections, cols = models.

_LC_PROJ = [
    (2, 11),   # θ_r_hip  vs ω_r_hip   — phase portrait
    (0,  8),   # z        vs ẋ          — height / forward speed
    (2,  5),   # θ_r_hip  vs θ_l_hip   — bilateral coordination
]


def plot_limit_cycle_trajectories(
    model_data: list[tuple[str, np.ndarray]],
    out_path:   Path,
    skip_frac:  float = 0.25,
    n_show:     int   = 800,
) -> None:
    """Limit-cycle trajectories in gait state space, colored by time.

    model_data : list of (label, all_gaits (T, 16)).
    skip_frac  : fraction of each trajectory to discard as transient.
    n_show     : number of steps to show from the settled region.
    """
    from matplotlib.collections import LineCollection

    n_proj   = len(_LC_PROJ)
    n_models = len(model_data)

    fig, axes = plt.subplots(
        n_proj, n_models,
        figsize=(3.6 * n_models, 3.1 * n_proj),
        squeeze=False,
    )

    for col, (label, gaits) in enumerate(model_data):
        color = COLORS.get(label, 'gray')

        if gaits is None or len(gaits) < 10:
            for row in range(n_proj):
                axes[row, col].set_visible(False)
            continue

        T      = len(gaits)
        start  = int(skip_frac * T)
        end    = min(start + n_show, T)
        traj   = gaits[start:end]             # (T', 16)
        T2     = len(traj)
        t_norm = np.linspace(0, 1, max(T2 - 1, 1))

        for row, (xi, yi) in enumerate(_LC_PROJ):
            ax = axes[row, col]
            ax.set_facecolor('white')

            if T2 > 1:
                pts  = np.column_stack([traj[:, xi], traj[:, yi]])
                segs = np.stack([pts[:-1], pts[1:]], axis=1)   # (T'-1, 2, 2)
                lc   = LineCollection(segs, cmap='viridis',
                                      norm=plt.Normalize(0, 1),
                                      linewidths=0.9, alpha=0.85)
                lc.set_array(t_norm)
                ax.add_collection(lc)
                x_lo, x_hi = traj[:, xi].min(), traj[:, xi].max()
                y_lo, y_hi = traj[:, yi].min(), traj[:, yi].max()
                px = max((x_hi - x_lo) * 0.08, 0.02)
                py = max((y_hi - y_lo) * 0.08, 0.02)
                ax.set_xlim(x_lo - px, x_hi + px)
                ax.set_ylim(y_lo - py, y_hi + py)

            ax.set_xlabel(GAIT_LABELS[xi], fontsize=9)
            ax.set_ylabel(GAIT_LABELS[yi], fontsize=9)
            ax.tick_params(labelsize=7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            for sp in ('left', 'bottom'):
                ax.spines[sp].set_linewidth(0.7)
            ax.grid(True, lw=0.3, alpha=0.3)

            if row == 0:
                ax.set_title(label, fontsize=10, color=color,
                             fontweight='bold', pad=4)

    # Shared time colorbar on the right
    sm = plt.cm.ScalarMappable(
        cmap='viridis', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:, -1], shrink=0.45, pad=0.05)
    cbar.set_label('Time →', fontsize=8)
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(['start', 'end'])

    fig.suptitle('Limit-cycle trajectories in gait state space',
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'[saved] {out_path}')


# ── combined entrypoint ───────────────────────────────────────────────────────

def plot_all(results: dict,
             gt_crossings:  list[dict],
             ms_crossings:  list[dict],
             fwd_crossings: list[dict],
             out_dir: Path,
             gt_all_gaits:  np.ndarray | None = None,
             ms_all_gaits:  np.ndarray | None = None,
             fwd_all_gaits: np.ndarray | None = None) -> None:
    """Called from probe_poincare_smwm_walker.py at the end of analysis."""
    # ── sparse crossing arrays (for PCA / 2-D crossing plots) ─────────────────
    arrays: dict[str, np.ndarray] = {}
    for label, crossings in [
        ('GT MuJoCo', gt_crossings),
        ('MS+SR',     ms_crossings),
        ('FWD+EP-AR', fwd_crossings),
    ]:
        if crossings:
            arrays[label] = np.stack([c['gait'] for c in crossings])

    if arrays:
        plot_crossings_2d(arrays, out_dir / 'poincare_crossings_2d.pdf')
        plot_pca(arrays, out_dir / 'gait_pca.pdf')

    # ── HALO-style dense distribution comparison ───────────────────────────────
    gt_dense = gt_all_gaits if (gt_all_gaits is not None and len(gt_all_gaits) > 0) else None
    if gt_dense is not None:
        rng = np.random.default_rng(0)
        for label, dense, slug in [
            ('FWD+EP-AR', fwd_all_gaits, 'fwd_ar'),
            ('MS+SR',     ms_all_gaits,  'ms_sr'),
        ]:
            if dense is not None and len(dense) > 0:
                plot_halo_dense(gt_dense, dense, label,
                                out_dir / f'halo_dense_{slug}.pdf', rng=rng)

    if results:
        plot_spectral_radius(results, out_dir / 'spectral_radius_bar.pdf')
        plot_floquet(results, out_dir / 'floquet_spectrum.pdf')
        plot_eigenvalues(results, out_dir / 'eigenvalue_complex.pdf')

    # ── limit-cycle trajectories ───────────────────────────────────────────────
    lc_data = [
        ('GT MuJoCo', gt_all_gaits),
        ('FWD+EP-AR', fwd_all_gaits),
        ('MS+SR',     ms_all_gaits),
    ]
    lc_valid = [(l, g) for l, g in lc_data if g is not None and len(g) > 0]
    if lc_valid:
        plot_limit_cycle_trajectories(
            lc_valid, out_dir / 'limit_cycle_trajectories.pdf')


# ── standalone CLI ────────────────────────────────────────────────────────────

def _load_dense(results_dir: Path, fname: str) -> np.ndarray | None:
    p = results_dir / fname
    if p.exists():
        arr = np.load(p)
        return arr if arr.ndim == 2 and len(arr) > 0 else None
    return None


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Plot Walker2d Poincaré stability results.')
    p.add_argument('--results-dir', required=True,
                   help='Directory containing poincare_results.json + .npy files')
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    arrays  = load_arrays(results_dir)
    results = load_results(results_dir)

    if arrays:
        plot_crossings_2d(arrays, results_dir / 'poincare_crossings_2d.pdf')
        plot_pca(arrays,          results_dir / 'gait_pca.pdf')

    # HALO-style dense plots
    gt_dense = _load_dense(results_dir, 'gt_all_gaits.npy')
    if gt_dense is not None:
        rng = np.random.default_rng(0)
        for label, fname, slug in [
            ('FWD+EP-AR', 'fwd_ar_all_gaits.npy', 'fwd_ar'),
            ('MS+SR',     'ms_sr_all_gaits.npy',  'ms_sr'),
        ]:
            dense = _load_dense(results_dir, fname)
            if dense is not None:
                plot_halo_dense(gt_dense, dense, label,
                                results_dir / f'halo_dense_{slug}.pdf', rng=rng)

    if results:
        plot_spectral_radius(results, results_dir / 'spectral_radius_bar.pdf')
        plot_floquet(results,         results_dir / 'floquet_spectrum.pdf')
        plot_eigenvalues(results,     results_dir / 'eigenvalue_complex.pdf')

    # ── limit-cycle trajectories ───────────────────────────────────────────────
    lc_data = [
        ('GT MuJoCo', gt_dense),
        ('FWD+EP-AR', _load_dense(results_dir, 'fwd_ar_all_gaits.npy')),
        ('MS+SR',     _load_dense(results_dir, 'ms_sr_all_gaits.npy')),
    ]
    lc_valid = [(l, g) for l, g in lc_data if g is not None and len(g) > 0]
    if lc_valid:
        plot_limit_cycle_trajectories(
            lc_valid, results_dir / 'limit_cycle_trajectories.pdf')

    print('[done]')


if __name__ == '__main__':
    main()
