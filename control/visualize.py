"""Visualization for MPC rollouts."""
from __future__ import annotations
import warnings
from pathlib import Path
from typing import Dict, Optional
import numpy as np


def save_rollout_frames(result: Dict, out_path, n_frames: int = 8, title: str = '') -> None:
    """Save PNG: top row = n_frames observation images; bottom = theta + action plots."""
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        warnings.warn('matplotlib unavailable'); return

    all_obs = result.get('all_obs', [])
    states  = np.asarray(result.get('states', [[0, 0, 0, 0]]))
    actions = np.asarray(result.get('actions', [[0]]))
    done_at = result.get('done_at', len(states))
    T       = len(states)

    if len(all_obs) == 0:
        warnings.warn('No observations in result; skipping frame save'); return

    # Select n_frames evenly from the real portion of the rollout
    sel_idx = np.linspace(0, min(done_at, len(all_obs)) - 1, n_frames, dtype=int)

    fig = plt.figure(figsize=(2.5 * n_frames, 7))
    gs  = gridspec.GridSpec(3, n_frames, figure=fig, hspace=0.5, wspace=0.2)

    for col, i in enumerate(sel_idx):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(all_obs[i])
        ax.set_title(f't={i}', fontsize=8)
        ax.axis('off')

    t_ax = np.arange(T)
    ax_theta = fig.add_subplot(gs[1, :])
    ax_theta.plot(t_ax, states[:, 2], color='tab:red',    label='θ (rad)')
    ax_theta.plot(t_ax, states[:, 0], color='tab:blue',   label='x (m)', linestyle='--')
    ax_theta.axhline(0, color='k', linestyle=':', linewidth=0.8)
    if done_at < T:
        ax_theta.axvline(done_at, color='red', linestyle='--', linewidth=1.2,
                         label=f'done t={done_at}')
    ax_theta.set_xlabel('timestep'); ax_theta.set_ylabel('state')
    ax_theta.legend(fontsize=8, ncol=3)
    ax_theta.set_title('State trajectory (θ, x)')

    ax_act = fig.add_subplot(gs[2, :])
    ax_act.plot(t_ax, actions[:, 0], color='darkorange', label='u')
    ax_act.axhline(0, color='k', linestyle=':', linewidth=0.8)
    if done_at < T:
        ax_act.axvline(done_at, color='red', linestyle='--', linewidth=1.2)
    ax_act.set_xlabel('timestep'); ax_act.set_ylabel('action u')
    ax_act.set_title('Control input')

    stab = result.get('stabilized', False)
    fe   = result.get('final_state_error', float('nan'))
    fig.suptitle(
        f"{title}  |  stabilized={'yes' if stab else 'no'}  final_err={fe:.3f}",
        fontsize=10,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'[vis] Saved frames -> {out_path}')


def save_rollout_video(result: Dict, out_path, fps: int = 15, title: str = '') -> None:
    """Save GIF using pillow (no ffmpeg needed)."""
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from PIL import Image
        import io
    except ImportError:
        warnings.warn('matplotlib or pillow unavailable; skipping video'); return

    all_obs = result.get('all_obs', [])
    states  = np.asarray(result.get('states', [[0, 0, 0, 0]]))
    actions = np.asarray(result.get('actions', [[0]]))
    done_at = result.get('done_at', len(states))
    T_vid   = min(done_at, len(all_obs))

    if T_vid == 0:
        warnings.warn('No observations; skipping video'); return

    frames_pil = []
    for t in range(T_vid):
        fig = plt.figure(figsize=(9, 3.5))
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

        ax_obs = fig.add_subplot(gs[0, 0])
        ax_obs.imshow(all_obs[t])
        ax_obs.set_title(f't={t}', fontsize=9); ax_obs.axis('off')

        ax_s = fig.add_subplot(gs[0, 1])
        ax_s.plot(range(t + 1), states[:t + 1, 2], color='tab:red',  label='θ')
        ax_s.plot(range(t + 1), states[:t + 1, 0], color='tab:blue', label='x', linestyle='--')
        ax_s.set_xlim(0, len(states)); ax_s.axhline(0, color='k', linewidth=0.5)
        ax_s.legend(fontsize=7); ax_s.set_title('θ, x')

        ax_a = fig.add_subplot(gs[0, 2])
        ax_a.plot(range(t + 1), actions[:t + 1, 0], color='darkorange')
        ax_a.set_xlim(0, len(actions)); ax_a.axhline(0, color='k', linewidth=0.5)
        ax_a.set_title('action u')

        fig.suptitle(title, fontsize=9)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        frames_pil.append(Image.open(buf).copy())
        buf.close()

    out_path = Path(out_path).with_suffix('.gif')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_pil[0].save(
        out_path, save_all=True, append_images=frames_pil[1:],
        duration=int(1000 / fps), loop=0,
    )
    print(f'[vis] Saved video -> {out_path}')
