#!/usr/bin/env python
"""Plot per-trial state trajectories under LQR for multiple models.

Each subfigure corresponds to one model (one JSON result file) and shows
||s_t|| = ||[x, ẋ, θ, θ̇]|| over time for all trials, coloured green for
successful trials and red for failures, with the success threshold as a
horizontal dashed line.

Usage
-----
python experiments/plot_lqr_state_trajectories.py \\
    --jsons "1SP + EP-IDM:results/lqr_1act_endpoint.json" \\
            "MSP + SIG:results/lqr_1act_sigreg.json" \\
    --out results/lqr_state_trajectories.pdf
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

STATE_LABELS  = [r'$x$', r'$\dot{x}$', r'$\theta$', r'$\dot{\theta}$']
STATE_COLORS  = ['#2166ac', '#4dac26', '#111111', '#555555']
PANEL_COLOR   = 'white'
GRID_KW       = dict(color='#cccccc', linewidth=0.8, alpha=0.9)

# Label → color: orange=IDM, blue=SIG, green=other
COLOR_IDM  = '#e07b00'   # orange
COLOR_SIG  = '#2166ac'   # blue
COLOR_OTHER = '#2ca02c'  # green

def _label_color(label: str) -> str:
    u = label.upper()
    if 'SIG' in u:
        return COLOR_SIG
    if 'IDM' in u:
        return COLOR_IDM
    return COLOR_OTHER
LINEWIDTH     = 2.4
TICK_SIZE     = 13
TITLE_SIZE    = 16


def load_trials(json_path):
    """Return (label, success_rate, threshold, trials).

    Handles both compare_lqr_smwm.py and lqr_dinowm_cartpole.py formats.
    Each trial dict must contain 'states' (list of [x,xdot,th,thdot]) and
    'success' (bool).
    """
    with open(json_path) as f:
        data = json.load(f)

    threshold = data.get('protocol', {}).get('success_threshold', 0.7)

    if 'models' in data:
        m = data['models'][0]
        return m['label'], float(m.get('success_rate', 0.0)), threshold, m['trials']

    if 'trials' in data:
        trials = data['trials']
        sr = float(np.mean([t['success'] for t in trials]))
        return data.get('label', Path(json_path).stem), sr, threshold, trials

    raise ValueError(f'Unknown JSON format: {json_path}')


def plot_model_single(ax, trial, label, show_legend=False):
    """Single trial: 4 state variables as coloured lines."""
    states = np.array(trial['states'])   # (T+1, 4)
    t = np.arange(len(states))
    for i, (slabel, scolor) in enumerate(zip(STATE_LABELS, STATE_COLORS)):
        ax.plot(t, states[:, i], color=scolor, linewidth=LINEWIDTH,
                label=slabel if show_legend else None)
    ax.set_facecolor(PANEL_COLOR)
    ax.set_axisbelow(True)
    ax.grid(True, **GRID_KW)
    ax.axhline(0, color='k', linewidth=0.6, alpha=0.3)
    ax.set_title(label, fontsize=TITLE_SIZE, fontweight='bold', pad=4)
    ax.set_xlim(0, len(states) - 1)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.spines[['top', 'right']].set_visible(False)


def plot_model_mean_ci(ax, trials, label, threshold, color='#2166ac'):
    """All trials: mean ± 95 % CI of ||s_t|| over time."""
    # Pad shorter trials to the max length with their last state
    T = max(len(t['states']) for t in trials)
    norms = []
    for trial in trials:
        s = np.array(trial['states'])          # (Ti+1, 4)
        n = np.linalg.norm(s, axis=1)          # (Ti+1,)
        if len(n) < T:
            n = np.concatenate([n, np.full(T - len(n), n[-1])])
        norms.append(n)
    norms = np.array(norms)                    # (n_trials, T)

    t   = np.arange(T)
    mean = norms.mean(axis=0)
    sem  = norms.std(axis=0) / np.sqrt(len(trials))
    lo   = mean - 1.96 * sem
    hi   = mean + 1.96 * sem

    ax.plot(t, mean, color=color, linewidth=LINEWIDTH)
    ax.fill_between(t, lo, hi, color=color, alpha=0.20)
    ax.set_facecolor(PANEL_COLOR)
    ax.set_axisbelow(True)
    ax.grid(True, **GRID_KW)
    ax.axhline(threshold, color='k', linestyle='--', linewidth=1.0, alpha=0.5)
    ax.set_title(label, fontsize=TITLE_SIZE, fontweight='bold', pad=4)
    ax.set_xlim(0, T - 1)
    ax.set_yscale('log')
    ax.tick_params(labelsize=TICK_SIZE)
    ax.spines[['top', 'right']].set_visible(False)


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--jsons', nargs='+', required=True,
                   metavar='LABEL:PATH',
                   help='"Model label:path/to/result.json"  (repeat per model)')
    p.add_argument('--mode', choices=['single', 'mean-ci'], default='single',
                   help='"single": one trial, 4 state vars; '
                        '"mean-ci": mean ± 95%% CI of ||s_t|| over all trials')
    p.add_argument('--trial', type=int, default=9,
                   help='Trial index for --mode single (0-based)')
    p.add_argument('--ncols', type=int, default=None,
                   help='Number of columns (auto if omitted)')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    entries = []
    for spec in args.jsons:
        label, path = spec.split(':', 1)
        entries.append((label.strip(), Path(path.strip())))

    n = len(entries)
    ncols = args.ncols or min(5, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.2 * ncols, 3.5 * nrows),
                             squeeze=False)

    for idx, (override_label, json_path) in enumerate(entries):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        if not json_path.exists():
            ax.set_visible(False)
            continue

        _, sr, threshold, trials = load_trials(json_path)

        if args.mode == 'single':
            trial_idx = min(args.trial, len(trials) - 1)
            plot_model_single(ax, trials[trial_idx], override_label,
                              show_legend=False)
        else:
            color = _label_color(override_label)
            plot_model_mean_ci(ax, trials, override_label, threshold, color=color)

        if col == 0:
            ax.set_ylabel('State' if args.mode == 'single' else r'$\|x_t\|$',
                          fontsize=TITLE_SIZE)
        if row == nrows - 1 or idx + ncols >= n:
            ax.set_xlabel('Step $t$', fontsize=TITLE_SIZE)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    print(f'[done] {out}')


if __name__ == '__main__':
    main()

