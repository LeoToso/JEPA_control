#!/usr/bin/env python
"""Build a side-by-side animated GIF from three model GIFs.

All three animations play in sync. The shorter GIFs loop to match the longest
(or clip to the shortest with --sync min).

Usage
-----
python experiments/make_trajectory_figure.py \
    --gif1 results/renders/latent_icem_H3_success/success_trial_004.gif \
    --gif2 results/renders/latent_icem_H3_ms_sr_success/best_failed_trial_009.gif \
    --gif3 results/renders/latent_icem_H3_fwd_sr/best_failed_trial_000.gif \
    --out  results/trajectory_figure.gif
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_gif(path: str):
    img = Image.open(path)
    frames, durations = [], []
    try:
        while True:
            frames.append(img.copy().convert('RGB'))
            durations.append(img.info.get('duration', 50))
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return frames, durations


def best_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
        '/Library/Fonts/Arial Bold.ttf',
        '/System/Library/Fonts/Helvetica.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gif1',   required=True, help='GIF for column 1 (1SP+EP-IDM)')
    p.add_argument('--gif2',   required=True, help='GIF for column 2 (MSP+SIG)')
    p.add_argument('--gif3',   required=True, help='GIF for column 3 (1SP+SIG)')
    p.add_argument('--label1', default='1SP+EP-IDM')
    p.add_argument('--label2', default='MSP+SIG')
    p.add_argument('--label3', default='1SP+SIG')
    p.add_argument('--label-height', type=int, default=44,
                   help='Pixel height of the label area above each GIF')
    p.add_argument('--gap',    type=int, default=12,
                   help='Pixel gap between columns')
    p.add_argument('--step',   type=int, default=1,
                   help='Keep every Nth frame (reduces file size)')
    p.add_argument('--sync',   choices=['min', 'loop'], default='min',
                   help='min: stop at shortest GIF; loop: loop shorter GIFs')
    p.add_argument('--duration', type=int, default=None,
                   help='Override frame duration in ms (default: from source GIF)')
    p.add_argument('--out',    default='results/trajectory_figure.gif')
    args = p.parse_args()

    cols = [
        (args.gif1, args.label1),
        (args.gif2, args.label2),
        (args.gif3, args.label3),
    ]

    all_frames, all_durations = [], []
    for gif_path, _ in cols:
        print(f'[fig] loading {gif_path} …')
        frames, durations = load_gif(gif_path)
        print(f'      {len(frames)} frames')
        all_frames.append(frames)
        all_durations.append(durations)

    lengths = [len(f) for f in all_frames]
    n_total = min(lengths) if args.sync == 'min' else max(lengths)

    # Subsample
    indices = list(range(0, n_total, args.step))

    # Normalise all columns to the same height
    target_h = max(f[0].height for f in all_frames)
    col_frames = []
    col_widths  = []
    for frames in all_frames:
        h, w = frames[0].height, frames[0].width
        if h != target_h:
            new_w = int(w * target_h / h)
            frames = [f.resize((new_w, target_h), Image.LANCZOS) for f in frames]
        col_frames.append(frames)
        col_widths.append(frames[0].width)

    label_h = args.label_height
    total_w = sum(col_widths) + args.gap * (len(cols) - 1)
    total_h = label_h + target_h

    font = best_font(label_h - 10)

    print(f'[fig] compositing {len(indices)} frames …')
    composite, frame_durations = [], []

    for t in indices:
        canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
        draw   = ImageDraw.Draw(canvas)

        x = 0
        for col_idx, (frames, (_, label)) in enumerate(zip(col_frames, cols)):
            frame = frames[t % len(frames)]
            canvas.paste(frame, (x, label_h))

            # Bold label centred above column
            bbox   = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.text(
                (x + (col_widths[col_idx] - text_w) // 2,
                 (label_h - text_h) // 2),
                label, fill=(0, 0, 0), font=font,
                stroke_width=1, stroke_fill=(0, 0, 0),
            )
            x += col_widths[col_idx] + args.gap

        composite.append(canvas)
        dur = args.duration if args.duration else all_durations[0][t % len(all_durations[0])]
        frame_durations.append(dur * args.step)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composite[0].save(
        out_path,
        save_all=True,
        append_images=composite[1:],
        duration=frame_durations,
        loop=0,
        optimize=False,
    )
    print(f'[fig] saved → {out_path}  ({len(composite)} frames)')


if __name__ == '__main__':
    main()
