#!/usr/bin/env python
"""Render 4 × N_FRAMES grid of Walker2d posture frames from cached trajectories.

Layout
------
Row 0 : GT
Row 1 : 1SP+EP-IDM
Row 2 : 1SP+SIG
Row 3 : MSP+SIG

Each row shows N_FRAMES uniformly sampled frames from one episode of the
corresponding cached trajectory (.npz files produced by
plot_model_limit_cycle_sac.py).

Frames are rendered by setting MuJoCo state from the 17-D Walker2d obs:
  obs[0:8]  = qpos[1:9]  (z, tilt, 6 joint angles)
  obs[8:17] = qvel[0:9]  (all velocities)
x-position is set to 0 so the robot appears centred in every frame.

Usage
-----
MUJOCO_GL=egl python experiments/visualize_walker_trajectory_frames.py \\
    --gt-cache      results/cache/gt_trajs.npz \\
    --fwd-cache     results/cache/fwd_trajs.npz \\
    --sig-fwd-cache results/cache/sig_fwd_trajs.npz \\
    --ms-cache      results/cache/ms_trajs.npz \\
    --ms-steps 100 \\
    --n-frames 10 \\
    --out results/walker_trajectory_frames.pdf
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── trajectory helpers ────────────────────────────────────────────────────────

def _load_trajs(path: str) -> list[np.ndarray]:
    d = np.load(path)
    return [d[f't{i}'] for i in range(len(d))]


def _sample_frames(traj: np.ndarray, n_frames: int,
                   max_steps: int | None = None,
                   tail_steps: int | None = None,
                   skip_last: int = 0) -> np.ndarray:
    """Return (n_frames, 17) array of uniformly sampled obs from traj.

    max_steps : keep only the first max_steps steps.
    tail_steps: after max_steps truncation, keep only the last tail_steps steps.
    skip_last : exclude the last N frames from the sampling window.
    """
    t = traj[:max_steps] if max_steps is not None else traj
    if tail_steps is not None:
        t = t[-tail_steps:]
    end = max(0, len(t) - 1 - skip_last)
    idx = np.linspace(0, end, n_frames, dtype=int)
    return t[idx]


# ── rendering ─────────────────────────────────────────────────────────────────

def render_obs_batch(obs_batch: np.ndarray,
                     render_size: int = 200) -> list[np.ndarray]:
    """Render a batch of 17-D Walker2d observations as RGB frames.

    obs_batch : (N, 17) — state vectors from GT env or MLP probe output
    Returns   : list of N (render_size, render_size, 3) uint8 arrays
    """
    import gymnasium as gym
    try:
        from PIL import Image as PilImage
        _has_pil = True
    except ImportError:
        _has_pil = False

    env = gym.make('Walker2d-v4', render_mode='rgb_array')
    env.reset(seed=0)

    std = obs_batch.std(axis=0)
    print(f'  obs std  (qpos-part): {std[:8].round(4)}')
    print(f'  obs std  (qvel-part): {std[8:].round(4)}')

    n_fallback = 0
    frames = []
    for i, obs in enumerate(obs_batch):
        qpos = np.zeros(9, dtype=np.float64)
        qpos[1:] = obs[0:8].astype(np.float64)   # z, tilt, joints
        qvel = obs[8:17].astype(np.float64)
        try:
            env.unwrapped.set_state(qpos, qvel)
        except Exception as e:
            # Clip velocities and retry — do NOT reset (reset gives same pose each time)
            n_fallback += 1
            if n_fallback == 1:
                print(f'  [render warn] set_state failed at frame {i}: {e}')
            try:
                env.unwrapped.set_state(
                    np.clip(qpos, -5, 5), np.clip(qvel, -50, 50))
            except Exception:
                pass  # render whatever state the env is in

        img = env.render()   # (H, W, 3) uint8, default ~480×480

        # resize to render_size × render_size
        if img.shape[0] != render_size or img.shape[1] != render_size:
            if _has_pil:
                img = np.asarray(
                    PilImage.fromarray(img).resize(
                        (render_size, render_size), PilImage.BILINEAR),
                    dtype=np.uint8)
            else:
                import torch
                t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
                t = torch.nn.functional.interpolate(
                    t, (render_size, render_size),
                    mode='bilinear', align_corners=False)
                img = t.squeeze(0).permute(1, 2, 0).byte().numpy()

        frames.append(img)

    if n_fallback:
        print(f'  [render warn] {n_fallback}/{len(obs_batch)} frames used fallback clipping')
    else:
        print(f'  [render ok] all {len(obs_batch)} frames set cleanly')
    env.close()
    return frames


# ── gif ───────────────────────────────────────────────────────────────────────

def build_gif_frame(frames_at_t: list[np.ndarray],
                    labels: list[str],
                    label_fontsize: int = 14,
                    horizontal: bool = False) -> 'PIL.Image':
    """Composite one GIF frame.

    horizontal=False (default): models stacked vertically, rotated labels on left.
    horizontal=True:            models in a row, labels above each frame.
    """
    from PIL import Image as PilImage, ImageDraw, ImageFont
    h, w = frames_at_t[0].shape[:2]
    n = len(frames_at_t)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                                   label_fontsize)
    except Exception:
        font = ImageFont.load_default()

    if horizontal:
        label_h = label_fontsize + 8
        canvas = PilImage.new('RGB', (w * n, label_h + h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        for i, (img, label) in enumerate(zip(frames_at_t, labels)):
            canvas.paste(PilImage.fromarray(img), (i * w, label_h))
            bb = draw.textbbox((0, 0), label, font=font)
            tx = i * w + (w - (bb[2] - bb[0])) // 2
            ty = (label_h - (bb[3] - bb[1])) // 2
            draw.text((tx, ty), label, fill=(30, 30, 30), font=font)
    else:
        label_w = max(120, label_fontsize * 7)
        canvas = PilImage.new('RGB', (label_w + w, h * n), (255, 255, 255))
        for i, (img, label) in enumerate(zip(frames_at_t, labels)):
            canvas.paste(PilImage.fromarray(img), (label_w, i * h))
            txt_img = PilImage.new('RGB', (h, label_w), (240, 240, 240))
            td = ImageDraw.Draw(txt_img)
            bb = td.textbbox((0, 0), label, font=font)
            tx = (h - (bb[2] - bb[0])) // 2
            ty = (label_w - (bb[3] - bb[1])) // 2
            td.text((tx, ty), label, fill=(30, 30, 30), font=font)
            canvas.paste(txt_img.rotate(90, expand=True), (0, i * h))
    return canvas


def save_gif(rows_traj: list[tuple[str, np.ndarray]],
             out_path: str,
             n_frames: int = 100,
             render_size: int = 200,
             fps: int = 10,
             label_fontsize: int = 14,
             from_head: bool = False,
             skip_labels: set | None = None,
             horizontal: bool = False):
    """Render n_frames steps of each trajectory and save as animated GIF.

    from_head=True  : take the first n_frames steps (shows transient).
    from_head=False : take the last  n_frames steps (shows steady state).
    skip_labels     : set of label strings to exclude from the GIF.
    horizontal      : arrange models in a row instead of a column.
    """
    from PIL import Image as PilImage

    skip_labels = skip_labels or set()

    # build per-model frame lists aligned in time
    all_frames: list[list[np.ndarray]] = []
    labels = []
    for label, traj in rows_traj:
        if label in skip_labels:
            print(f'  [{label}] skipped')
            continue
        t = traj[:n_frames] if from_head else traj[-n_frames:]
        print(f'  [{label}] rendering {len(t)} frames ({"head" if from_head else "tail"}) …')
        all_frames.append(render_obs_batch(t, render_size))
        labels.append(label)

    n_t = min(len(f) for f in all_frames)
    gif_frames = []
    for t in range(n_t):
        imgs_at_t = [all_frames[m][t] for m in range(len(labels))]
        gif_frames.append(build_gif_frame(imgs_at_t, labels, label_fontsize,
                                          horizontal=horizontal))

    duration_ms = int(1000 / fps)
    gif_frames[0].save(
        out_path, save_all=True, append_images=gif_frames[1:],
        loop=0, duration=duration_ms)
    print(f'[saved gif] {out_path}  ({n_t} frames @ {fps} fps)')


# ── figure ────────────────────────────────────────────────────────────────────

def build_figure(rows: list[tuple[str, list[np.ndarray]]],
                 label_fontsize: int = 18) -> plt.Figure:
    """Build and return the 4-row × N-col figure.

    rows : list of (row_label, list_of_rgb_frames)
    """
    n_rows = len(rows)
    n_cols = len(rows[0][1])

    cell_w, cell_h = 2.2, 2.2
    left_margin = 1.2  # space for vertical labels
    fig_w = left_margin + n_cols * cell_w
    fig_h = n_rows * cell_h

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=left_margin / fig_w,
                        right=1.0, top=1.0, bottom=0.0,
                        hspace=0.04, wspace=0.04)

    for r, (label, frames) in enumerate(rows):
        for c, frame in enumerate(frames):
            ax = axes[r, c]
            ax.imshow(frame)
            ax.set_axis_off()

        # vertical label centred on this row
        row_axes = axes[r]
        ys = [ax.get_position().y0 + ax.get_position().height / 2
              for ax in row_axes]
        y_mid = sum(ys) / len(ys)
        xs = [ax.get_position().x0 for ax in row_axes]
        x_left = min(xs) - 0.01
        fig.text(x_left, y_mid, label,
                 ha='right', va='center', rotation='vertical',
                 fontsize=label_fontsize, fontweight='bold')

    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--gt-cache',      required=True)
    p.add_argument('--fwd-cache',     required=True,
                   help='1SP+EP-IDM cache')
    p.add_argument('--sig-fwd-cache', default=None,
                   help='1SP+SIG cache (omit to skip that row)')
    p.add_argument('--ms-cache',      required=True,
                   help='MSP+SIG cache')

    p.add_argument('--gt-label',      default='GT')
    p.add_argument('--fwd-label',     default='1SP+EP-IDM')
    p.add_argument('--sig-fwd-label', default='1SP+SIG')
    p.add_argument('--ms-label',      default='MSP+SIG')

    # per-model step truncation (mirrors plot_model_limit_cycle_sac.py)
    p.add_argument('--gt-steps',      type=int, default=None)
    p.add_argument('--fwd-steps',     type=int, default=None)
    p.add_argument('--sig-fwd-steps', type=int, default=None)
    p.add_argument('--ms-steps',      type=int, default=None)

    # episode index to visualise from each cache (per-model overrides --episode)
    p.add_argument('--episode',       type=int, default=0,
                   help='Default episode index for all caches')
    p.add_argument('--gt-episode',      type=int, default=None)
    p.add_argument('--fwd-episode',     type=int, default=None)
    p.add_argument('--sig-fwd-episode', type=int, default=None)
    p.add_argument('--ms-episode',      type=int, default=None)
    p.add_argument('--fwd-select-stay', action='store_true',
                   help='Auto-pick first 1SP+EP-IDM episode that stays within '
                        'the GT phase-portrait bounding box (mirrors plot_model_limit_cycle_sac.py)')

    p.add_argument('--n-frames',      type=int, default=10)
    p.add_argument('--tail-steps',    type=int, default=None,
                   help='Sample frames from only the last N steps of each trajectory')
    p.add_argument('--skip-last',     type=int, default=1,
                   help='Exclude last N frames from sampling window (default 1 avoids terminal pose)')
    p.add_argument('--render-size',   type=int, default=200,
                   help='Pixel size of each rendered frame')
    p.add_argument('--label-fontsize', type=int, default=18)
    p.add_argument('--out', default='results/walker_trajectory_frames.pdf')

    # GIF options
    p.add_argument('--gif',           default=None,
                   help='If set, also save an animated GIF to this path')
    p.add_argument('--gif-frames',    type=int, default=100,
                   help='Number of frames for the GIF')
    p.add_argument('--gif-fps',       type=int, default=10,
                   help='Frames per second for the GIF')
    p.add_argument('--gif-from-head', action='store_true',
                   help='Use first gif-frames steps (default: last gif-frames steps)')
    p.add_argument('--gif-no-gt',     action='store_true',
                   help='Exclude GT row from the GIF')
    p.add_argument('--gif-horizontal', action='store_true',
                   help='Arrange models in a row instead of a column in the GIF')
    args = p.parse_args()

    # ── load and sample ───────────────────────────────────────────────────────
    ep_overrides = {
        'gt':      args.gt_episode,
        'fwd':     args.fwd_episode,
        'sig_fwd': args.sig_fwd_episode,
        'ms':      args.ms_episode,
    }

    specs = [
        ('gt',      args.gt_label,      args.gt_cache,      args.gt_steps),
        ('fwd',     args.fwd_label,     args.fwd_cache,     args.fwd_steps),
    ]
    if args.sig_fwd_cache:
        specs.append(('sig_fwd', args.sig_fwd_label, args.sig_fwd_cache,
                      args.sig_fwd_steps))
    specs.append(('ms', args.ms_label, args.ms_cache, args.ms_steps))

    # pre-load GT trajectories for bbox filtering
    gt_trajs_all = _load_trajs(args.gt_cache)

    def _gt_bbox(trajs, margin=0.15):
        xs = np.concatenate([t[:, 2] for t in trajs])
        ys = np.concatenate([t[:, 11] for t in trajs])
        rx, ry = xs.ptp(), ys.ptp()
        return (xs.min() - margin * rx, xs.max() + margin * rx,
                ys.min() - margin * ry, ys.max() + margin * ry)

    def _escapes_bbox(traj, bbox):
        xmin, xmax, ymin, ymax = bbox
        x, y = traj[:, 2], traj[:, 11]
        return bool(np.any(x < xmin) or np.any(x > xmax) or
                    np.any(y < ymin) or np.any(y > ymax))

    gt_bbox = _gt_bbox(gt_trajs_all)

    rows = []
    gif_trajs = []   # (label, full_traj_after_head_truncation) for GIF
    for key, label, cache_path, max_steps in specs:
        print(f'[{label}] loading {cache_path} …')
        trajs = _load_trajs(cache_path)

        # resolve episode index
        ep_override = ep_overrides.get(key)
        if ep_override is not None:
            ep = min(ep_override, len(trajs) - 1)
        elif key == 'fwd' and args.fwd_select_stay:
            stay = [i for i, t in enumerate(trajs) if not _escapes_bbox(t, gt_bbox)]
            if stay:
                ep = stay[0]
                print(f'  [fwd-select-stay] chose episode {ep} '
                      f'(staying: {stay}, escaping: '
                      f'{[i for i in range(len(trajs)) if i not in stay]})')
            else:
                ep = 0
                print(f'  [fwd-select-stay] no staying episodes found, using 0')
        else:
            ep = min(args.episode, len(trajs) - 1)

        traj = trajs[ep]
        traj_head = traj[:max_steps] if max_steps is not None else traj
        tail = args.tail_steps
        print(f'  episode {ep}: {len(traj)} steps'
              + (f' → head {max_steps}' if max_steps else '')
              + (f' → tail {tail}' if tail else ''))
        obs_batch = _sample_frames(traj_head, args.n_frames, tail_steps=tail,
                                   skip_last=args.skip_last)

        print(f'  rendering {args.n_frames} frames at {args.render_size}px …')
        frames = render_obs_batch(obs_batch, args.render_size)
        rows.append((label, frames))
        gif_trajs.append((label, traj_head))

    # ── plot ──────────────────────────────────────────────────────────────────
    print('\n[figure] assembling grid …')
    fig = build_figure(rows, label_fontsize=args.label_fontsize)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches='tight')
    print(f'[saved] {out}')

    # ── gif ───────────────────────────────────────────────────────────────────
    if args.gif:
        print(f'\n[gif] rendering last {args.gif_frames} frames per model …')
        gif_out = Path(args.gif)
        gif_out.parent.mkdir(parents=True, exist_ok=True)
        skip = {args.gt_label} if args.gif_no_gt else set()
        save_gif(gif_trajs, str(gif_out),
                 n_frames=args.gif_frames,
                 render_size=args.render_size,
                 fps=args.gif_fps,
                 label_fontsize=args.label_fontsize,
                 from_head=args.gif_from_head,
                 skip_labels=skip,
                 horizontal=args.gif_horizontal)


if __name__ == '__main__':
    main()
