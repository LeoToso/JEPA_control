#!/usr/bin/env python
"""Two-panel LQR figure: each panel overlays multiple models as mean ± 95 % CI.

Usage — "from scratch" figure
------------------------------
python experiments/plot_lqr_twopanel.py \\
    --left  "1SP + SIG:results/lqr_1step_pred1act_sigreg.json" \\
            "1SP + EP-IDM:results/lqr_1act_endpoint.json" \\
            "1SP + IDM + MS-IDM:results/lqr_1steppred_MS_AR_1stepAR.json" \\
    --right "MSP + SIG:results/lqr_1act_sigreg.json" \\
            "MSP + EP-IDM + SIG:results/MSpred_EndpointAR_weight05_predloss_SIGREG_act1.json" \\
    --left-title "1-Step Prediction" --right-title "Multi-Step Prediction" \\
    --title "From Scratch" \\
    --out   results/lqr_from_scratch.pdf

Usage — "pre-trained" figure
-----------------------------
python experiments/plot_lqr_twopanel.py \\
    --left  "DINOv2 + SIG:results/lqr_dino_sigreg.json" \\
            "DINOv2 + IDM:results/lqr_dino_idm.json" \\
    --right "iBOT + SIG:results/lqr_ibot_sigreg.json" \\
            "iBOT + IDM:results/lqr_ibot_idm.json" \\
    --title "Pre-trained" \\
    --out   results/lqr_pretrained.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── style constants ────────────────────────────────────────────────────────────
PANEL_COLOR = 'white'
GRID_KW     = dict(color='#cccccc', linewidth=0.8, alpha=0.9)
LINEWIDTH   = 2.4
TICK_SIZE   = 15
LABEL_SIZE  = 17
TITLE_SIZE  = 17
LEGEND_SIZE = 13

COLOR_SIG        = '#2166ac'   # blue  — SIG
COLOR_IDM        = '#d4b100'   # golden yellow — PR-IDM (standard 1-step)
COLOR_EP_IDM_MSP = '#d62728'   # red   — MSP + PR-EP-IDM
COLOR_EP_IDM_1SP = '#e07b00'   # orange — 1SP + PR-EP-IDM
COLOR_OTHER      = ['#2ca02c', '#9467bd', '#8c564b', '#bcbd22', '#e377c2']
COLOR_DINOWM     = '#74c476'   # light green — DINO-WM baseline


def _label_color(label: str) -> str:
    u = label.upper()
    if 'SIG' in u:
        return COLOR_SIG
    if 'EP-IDM' in u:
        return COLOR_EP_IDM_1SP if u.startswith('1SP') else COLOR_EP_IDM_MSP
    if 'IDM' in u:
        return COLOR_IDM
    return COLOR_OTHER[0]


# ── data loading ───────────────────────────────────────────────────────────────

def load_trials(json_path):
    """Return (label, success_rate, threshold, trials).

    Handles both compare_lqr_smwm.py and lqr_dinowm_cartpole.py formats.
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


# ── panel plotting ─────────────────────────────────────────────────────────────

def plot_panel(ax, entries, panel_title: str | None = None,
               legend_loc: str = 'outside bottom', show_ylabel: bool = True,
               legend_ncol: int = 2):
    """Overlay multiple models on one axis.

    entries: list of (label, json_path_str)
    Color priority: DINO-WM (teal) → EP-IDM (red/orange by MSP/1SP) → IDM (yellow)
                    → SIG (blue) → other (green cycle).
    """
    sig_count:   int = 0
    other_count: int = 0
    threshold_val = None

    for label, json_path in entries:
        path = Path(json_path)
        if not path.exists():
            print(f'  [warn] missing {path}, skipping.')
            continue

        _, sr, threshold, trials = load_trials(path)
        if threshold_val is None:
            threshold_val = threshold

        u = label.upper()
        if 'DINO-WM' in u or u.replace('-', '') == 'DINOWM':
            color     = COLOR_DINOWM
            linestyle = '-'
        elif 'EP-IDM' in u:
            color     = COLOR_EP_IDM_1SP if u.startswith('1SP') else COLOR_EP_IDM_MSP
            linestyle = '-'
        elif 'IDM' in u:
            color     = COLOR_IDM
            linestyle = '-'
        elif 'SIG' in u:
            color     = COLOR_SIG
            linestyle = '--' if sig_count > 0 else '-'
            sig_count += 1
        else:
            color     = COLOR_OTHER[other_count % len(COLOR_OTHER)]
            linestyle = '-'
            other_count += 1

        # Build norm matrix, padding shorter trials
        T = max(len(t['states']) for t in trials)
        norms = []
        for trial in trials:
            s = np.array(trial['states'])
            n = np.linalg.norm(s, axis=1)
            if len(n) < T:
                n = np.concatenate([n, np.full(T - len(n), n[-1])])
            norms.append(n)
        norms = np.array(norms)

        t    = np.arange(T)
        mean = norms.mean(axis=0)
        sem  = norms.std(axis=0) / np.sqrt(len(trials))
        lo   = mean - 1.96 * sem
        hi   = mean + 1.96 * sem

        ax.plot(t, mean, color=color, linewidth=LINEWIDTH,
                linestyle=linestyle, label=label)
        ax.fill_between(t, lo, hi, color=color, alpha=0.15)

    if threshold_val is not None:
        ax.axhline(threshold_val, color='#444444', linestyle=':', linewidth=1.8,
                   alpha=0.8, label='_nolegend_')

    ax.set_facecolor(PANEL_COLOR)
    ax.set_axisbelow(True)
    ax.grid(True, **GRID_KW)
    ax.set_yscale('log')
    ax.set_xlabel('Time step $t$', fontsize=LABEL_SIZE)
    if show_ylabel:
        ax.set_ylabel(r'$\|\mathbf{x}_t\|_2$', fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.spines[['top', 'right']].set_visible(False)

    if panel_title:
        ax.set_title(panel_title, fontsize=TITLE_SIZE, fontweight='bold', pad=6)

    if legend_loc not in ('outside bottom', 'outside right'):
        kw = dict(fontsize=LEGEND_SIZE, framealpha=0.9,
                  borderpad=0.5, labelspacing=0.3, handlelength=1.6, ncol=legend_ncol)
        ax.legend(loc=legend_loc, **kw)
    # 'outside bottom' and 'outside right': legend created by main() using
    # figure-level coordinates so multiple panels can't overlap.


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--left', nargs='+', required=True, metavar='LABEL:PATH',
                   help='Models for the left panel, each as "Label:path/to.json"')
    p.add_argument('--right', nargs='+', required=True, metavar='LABEL:PATH',
                   help='Models for the right panel, each as "Label:path/to.json"')
    p.add_argument('--left-title',  default=None,
                   help='Title for the left panel (auto from labels if omitted)')
    p.add_argument('--right-title', default=None,
                   help='Title for the right panel (auto from labels if omitted)')
    p.add_argument('--title', default=None,
                   help='Overall figure suptitle')
    p.add_argument('--left-legend-loc',  default='outside bottom',
                   help='Legend location: matplotlib loc string or "outside right"/"outside bottom"')
    p.add_argument('--right-legend-loc', default='outside bottom',
                   help='Legend location: matplotlib loc string or "outside right"/"outside bottom"')
    p.add_argument('--legend-ncol', type=int, default=2,
                   help='Columns in outside-bottom legends')
    p.add_argument('--out', required=True)
    p.add_argument('--width',  type=float, default=13.0, help='Figure width in inches')
    p.add_argument('--height', type=float, default=5.0,  help='Figure height in inches')
    args = p.parse_args()

    def parse_specs(specs):
        out = []
        for spec in specs:
            label, path = spec.split(':', 1)
            out.append((label.strip(), path.strip()))
        return out

    left_entries  = parse_specs(args.left)
    right_entries = parse_specs(args.right)

    fig, (ax_l, ax_r) = plt.subplots(1, 2,
                                      figsize=(args.width, args.height),
                                      sharey=False)

    left_title  = args.left_title
    right_title = args.right_title

    plot_panel(ax_l, left_entries,  panel_title=left_title,
               legend_loc=args.left_legend_loc,
               legend_ncol=args.legend_ncol)
    plot_panel(ax_r, right_entries, panel_title=right_title,
               legend_loc=args.right_legend_loc, show_ylabel=False,
               legend_ncol=args.legend_ncol)

    if args.title:
        fig.suptitle(args.title, fontsize=TITLE_SIZE + 2, fontweight='bold')

    bottom_locs = {'outside bottom'}
    has_bottom = (args.left_legend_loc in bottom_locs or
                  args.right_legend_loc in bottom_locs)
    # bottom only needs to fit the x-axis tick labels + axis label;
    # the legend itself lives just below y=0 and is captured by bbox_inches='tight'
    bot = 0.14 if has_bottom else 0.12
    fig.subplots_adjust(bottom=bot, wspace=0.25, left=0.10, right=0.97, top=0.95)

    # Place outside legends at figure level so they stay under their own panel
    leg_kw = dict(fontsize=LEGEND_SIZE, framealpha=0.9,
                  borderpad=0.5, labelspacing=0.3, handlelength=1.6,
                  ncol=args.legend_ncol)
    for ax, loc in [(ax_l, args.left_legend_loc), (ax_r, args.right_legend_loc)]:
        if loc not in ('outside bottom', 'outside right'):
            continue
        handles, labels = ax.get_legend_handles_labels()
        pos = ax.get_position()           # figure-fraction bbox
        cx  = (pos.x0 + pos.x1) / 2
        if loc == 'outside bottom':
            # anchor at y=0 (bottom figure edge); bbox_inches='tight' expands canvas
            fig.legend(handles, labels, loc='upper center',
                       bbox_to_anchor=(cx, 0.0), **leg_kw)
        else:  # outside right
            fig.legend(handles, labels, loc='upper left',
                       bbox_to_anchor=(pos.x1 + 0.01, pos.y1), **leg_kw)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    print(f'[done] {out}')


if __name__ == '__main__':
    main()
