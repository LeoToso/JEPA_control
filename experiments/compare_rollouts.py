#!/usr/bin/env python
"""Open-loop rollout comparison: GT frames vs decoded model frames.

Pipeline
--------
1. Run SAC policy in the GT MuJoCo env for N steps → save (frames, actions, obs).
2. For each SMWM model:
   a. Encode the initial frame pair → z_0.
   b. Apply the *same* recorded actions open-loop → z_1 … z_N (no env feedback).
   c. Decode each z_t → pixel frame via trained ConvDecoder.
3. Save frames as individual PNGs in output directory.
   Optionally also save a side-by-side comparison GIF (--gif flag).

Note: SMWM is JEPA-style (no built-in pixel decoder).
      ConvDecoder must be pre-trained with experiments/train_pixel_decoder.py.

Usage
-----
  MUJOCO_GL=egl python experiments/compare_rollouts.py \\
      --ms-sr-ckpt      /mnt/t7shield/.../model_final.pt \\
      --ms-sr-cfg       configs/walker2d_jepa_sigreg_rollout_act1.yaml \\
      --ms-sr-decoder   results/pixel_decoder_ms_sr.pt \\
      --fwd-ar-ckpt     results/.../model_final.pt \\
      --fwd-ar-cfg      configs/walker2d_jepa_fwd_endpoint_inverse_act1.yaml \\
      --fwd-ar-decoder  results/pixel_decoder_fwd_ar.pt \\
      --hdf5-dir        data/walker2d_fs5_64 \\
      --output-dir      results/rollout_comparison \\
      --sac-repo        sdpkjc/Walker2d-v4-sac_continuous_action-seed4 \\
      --n-steps         300
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch
import torch.nn.functional as F

from experiments.walker2d_ppo_utils import download_and_load_sac, load_sac_from_local
from experiments.walker2d_utils import (
    load_walker_bundle, gym_obs_to_mj_state, latent_step,
)
from experiments.probe_utils import encode_obs
from experiments.walker2d_poincare_utils import WalkerMuJoCoHelper
from experiments.train_pixel_decoder import ConvDecoder


# ── pixel decoder helpers ─────────────────────────────────────────────────────

def load_pixel_decoder(path: str, device: torch.device) -> ConvDecoder:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    dec  = ConvDecoder(latent_dim=ckpt['latent_dim'],
                       base_ch=ckpt.get('base_ch', 256))
    dec.load_state_dict(ckpt['state_dict'])
    dec.eval().to(device)
    return dec


@torch.no_grad()
def decode_z_to_frame(z: torch.Tensor, decoder: ConvDecoder,
                      render_size: int) -> np.ndarray:
    """z (1, latent_dim) → HWC uint8 np.ndarray of shape (render_size, render_size, 3)."""
    img = decoder(z)          # (1, 3, 64, 64)  float32 [0,1]
    if img.shape[-1] != render_size or img.shape[-2] != render_size:
        img = F.interpolate(img, (render_size, render_size),
                            mode='bilinear', align_corners=False)
    img_np = (img[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return img_np


# ── GT frame helpers ──────────────────────────────────────────────────────────

def resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    if frame.shape[0] == size and frame.shape[1] == size:
        return frame
    t = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0)
    t = F.interpolate(t, (size, size), mode='bilinear', align_corners=False)
    return t[0].permute(1, 2, 0).byte().numpy()


def add_label(frame: np.ndarray, text: str,
              color=(255, 255, 255)) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont
        img  = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
        except Exception:
            font = ImageFont.load_default()
        draw.text((4, 3), text, fill=color, font=font)
        return np.asarray(img)
    except ImportError:
        return frame


# ── GT rollout ────────────────────────────────────────────────────────────────

def collect_gt_rollout(policy, helper: WalkerMuJoCoHelper,
                       render_helper: WalkerMuJoCoHelper,
                       n_steps: int, seed: int,
                       render_size: int) -> tuple[list, list, list]:
    """Run SAC policy for n_steps; return (frames, actions, obs_list)."""
    obs, _ = helper.env.reset(seed=seed)
    obs = obs.astype(np.float32)

    frames:   list[np.ndarray] = []
    actions:  list[np.ndarray] = []
    obs_list: list[np.ndarray] = []

    for t in range(n_steps):
        qpos, qvel = helper.get_state()
        render_helper.set_state(qpos, qvel)
        frame = render_helper.render_frame()
        if frame is not None:
            frames.append(resize_frame(frame, render_size))

        action = policy.act(obs)
        actions.append(action.copy())
        obs_list.append(obs.copy())

        obs, rew, done = helper.step(action)
        obs = obs.astype(np.float32)
        if done:
            print(f'[GT] episode ended at step {t+1}')
            qpos, qvel = helper.get_state()
            render_helper.set_state(qpos, qvel)
            frame = render_helper.render_frame()
            if frame is not None:
                frames.append(resize_frame(frame, render_size))
            break

    print(f'[GT] collected {len(frames)} frames  {len(actions)} actions')
    return frames, actions, obs_list


# ── model open-loop rollout ───────────────────────────────────────────────────

@torch.no_grad()
def collect_model_rollout(bundle: dict,
                          decoder: ConvDecoder,
                          obs_list: list[np.ndarray],
                          gt_frames: list[np.ndarray],
                          actions: list[np.ndarray],
                          image_size: int,
                          render_size: int,
                          label: str,
                          warmup_steps: int = 1) -> list[np.ndarray]:
    """Open-loop latent rollout with GT warm-up.

    Re-encodes the first `warmup_steps` GT frames to give the model a real
    frame-difference signal before going fully open-loop.  With
    use_frame_diff=True, starting from prev_obs=obs_0 produces a zero diff
    and the latent immediately loses the motion signal.
    """
    def _resize(frame):
        if frame.shape[0] == image_size and frame.shape[1] == image_size:
            return frame
        t = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0)
        t = F.interpolate(t, (image_size, image_size),
                          mode='bilinear', align_corners=False)
        return t[0].permute(1, 2, 0).byte().numpy()

    # Warm-up: re-encode warmup_steps real GT frames so the frame-diff is real
    warmup = min(warmup_steps, len(gt_frames) - 1)
    prev_enc = _resize(gt_frames[0])
    cur_enc  = _resize(gt_frames[warmup])
    z = encode_obs(bundle, cur_enc, prev_enc, obs_list[warmup])

    # Decode from warmup frame onwards; prepend still-frames for t<warmup
    still_z = encode_obs(bundle, prev_enc, prev_enc, obs_list[0])
    prefix = [decode_z_to_frame(still_z, decoder, render_size)] * warmup

    frames: list[np.ndarray] = []
    for action in actions[warmup:]:
        frames.append(decode_z_to_frame(z, decoder, render_size))
        z = latent_step(bundle, z, action)

    print(f'[{label}] rendered {warmup} warm-up + {len(frames)} open-loop frames')
    return prefix + frames


# ── save frames ───────────────────────────────────────────────────────────────

def save_frames(frames: list[np.ndarray], out_dir: Path, prefix: str) -> None:
    from PIL import Image
    sub = out_dir / prefix
    sub.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(sub / f'{i:04d}.png')
    print(f'[saved] {len(frames)} frames → {sub}')


def save_comparison_frames(gt_frames: list[np.ndarray],
                           model_frames: dict[str, list[np.ndarray]],
                           out_dir: Path,
                           gap: int = 4) -> None:
    """Save side-by-side GT | model1 | model2 … as individual PNGs."""
    from PIL import Image
    sub = out_dir / 'comparison'
    sub.mkdir(parents=True, exist_ok=True)

    n = min(len(gt_frames), *(len(f) for f in model_frames.values()))
    h, w = gt_frames[0].shape[:2]
    labels   = ['GT MuJoCo'] + list(model_frames.keys())
    all_seqs = [gt_frames] + list(model_frames.values())
    n_cols   = len(all_seqs)
    total_w  = n_cols * w + (n_cols - 1) * gap

    for i in range(n):
        canvas = np.full((h, total_w, 3), 40, dtype=np.uint8)
        for col, (seq, lbl) in enumerate(zip(all_seqs, labels)):
            f  = add_label(seq[i], lbl)
            x0 = col * (w + gap)
            canvas[:, x0:x0 + w] = f
        Image.fromarray(canvas).save(sub / f'{i:04d}.png')

    print(f'[saved] {n} comparison frames → {sub}')


def save_comparison_gif(gt_frames: list[np.ndarray],
                        model_frames: dict[str, list[np.ndarray]],
                        out_path: Path,
                        fps: int = 15,
                        gap: int = 4) -> None:
    from PIL import Image
    n = min(len(gt_frames), *(len(f) for f in model_frames.values()))
    h, w     = gt_frames[0].shape[:2]
    labels   = ['GT MuJoCo'] + list(model_frames.keys())
    all_seqs = [gt_frames] + list(model_frames.values())
    n_cols   = len(all_seqs)
    total_w  = n_cols * w + (n_cols - 1) * gap

    pil_frames = []
    for i in range(n):
        canvas = np.full((h, total_w, 3), 40, dtype=np.uint8)
        for col, (seq, lbl) in enumerate(zip(all_seqs, labels)):
            f  = add_label(seq[i], lbl)
            x0 = col * (w + gap)
            canvas[:, x0:x0 + w] = f
        pil_frames.append(Image.fromarray(canvas))

    pil_frames[0].save(out_path, save_all=True, append_images=pil_frames[1:],
                       duration=int(1000 / fps), loop=0)
    print(f'[saved] {out_path}  ({n} frames, {n_cols} columns, {fps} fps)')


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Open-loop rollout comparison: GT vs SMWM models.')

    g = p.add_argument_group('SAC policy')
    g.add_argument('--sac-repo', default='sdpkjc/Walker2d-v4-sac_continuous_action-seed4')
    g.add_argument('--sac-ckpt', default=None)
    g.add_argument('--device',   default='cpu')

    g = p.add_argument_group('World models')
    g.add_argument('--ms-sr-ckpt',     required=True)
    g.add_argument('--ms-sr-cfg',      default=None)
    g.add_argument('--ms-sr-decoder',  required=True,
                   help='Path to trained ConvDecoder checkpoint for MS+SR')
    g.add_argument('--fwd-ar-ckpt',    required=True)
    g.add_argument('--fwd-ar-cfg',     default=None)
    g.add_argument('--fwd-ar-decoder', required=True,
                   help='Path to trained ConvDecoder checkpoint for FWD+EP-AR')
    g.add_argument('--model-device',   default='cuda')

    g = p.add_argument_group('Rollout')
    g.add_argument('--n-steps',      type=int, default=300)
    g.add_argument('--seed',         type=int, default=42)
    g.add_argument('--render-size',  type=int, default=256)
    g.add_argument('--image-size',   type=int, default=64)
    g.add_argument('--fps',          type=int, default=15)
    g.add_argument('--warmup-steps', type=int, default=1,
                   help='Number of GT frames to re-encode before going open-loop '
                        '(helps with use_frame_diff=True models; try 5–20 to '
                        'diagnose how many steps the predictor stays accurate)')

    g = p.add_argument_group('Output')
    g.add_argument('--output-dir',  default='results/rollout_comparison')
    g.add_argument('--gif',         action='store_true',
                   help='Also save side-by-side GIF (in addition to PNG frames)')
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev     = torch.device(args.model_device
                           if torch.cuda.is_available() else 'cpu')

    # ── Policy ────────────────────────────────────────────────────────────────
    print('\n=== Load SAC policy ===')
    if args.sac_ckpt:
        policy = load_sac_from_local(args.sac_ckpt, device=args.device)
    else:
        policy = download_and_load_sac(args.sac_repo, device=args.device)

    # ── SMWM bundles ──────────────────────────────────────────────────────────
    print('\n=== Load MS+SR bundle ===')
    from experiments.probe_poincare_walker2d import find_cfg
    ms_cfg    = args.ms_sr_cfg  or find_cfg(args.ms_sr_ckpt)
    ms_bundle = load_walker_bundle(args.ms_sr_ckpt, ms_cfg, args.model_device)

    print('\n=== Load FWD+EP-AR bundle ===')
    fwd_cfg    = args.fwd_ar_cfg or find_cfg(args.fwd_ar_ckpt)
    fwd_bundle = load_walker_bundle(args.fwd_ar_ckpt, fwd_cfg, args.model_device)

    # ── Pixel decoders ────────────────────────────────────────────────────────
    print('\n=== Load pixel decoders ===')
    ms_decoder  = load_pixel_decoder(args.ms_sr_decoder,  dev)
    fwd_decoder = load_pixel_decoder(args.fwd_ar_decoder, dev)

    # ── Environments ──────────────────────────────────────────────────────────
    print('\n=== Create environments ===')
    gt_helper     = WalkerMuJoCoHelper(image_size=args.image_size, render=False)
    render_helper = WalkerMuJoCoHelper(image_size=args.render_size, render=True,
                                       render_mode='rgb_array')

    # ── GT rollout ────────────────────────────────────────────────────────────
    print('\n=== GT rollout ===')
    gt_frames, actions, obs_list = collect_gt_rollout(
        policy, gt_helper, render_helper,
        n_steps=args.n_steps, seed=args.seed,
        render_size=args.render_size)

    # ── Model rollouts ────────────────────────────────────────────────────────
    print('\n=== FWD+EP-AR open-loop rollout ===')
    fwd_frames = collect_model_rollout(
        fwd_bundle, fwd_decoder, obs_list, gt_frames, actions,
        image_size=args.image_size, render_size=args.render_size,
        label='FWD+EP-AR', warmup_steps=args.warmup_steps)

    print('\n=== MS+SR open-loop rollout ===')
    ms_frames = collect_model_rollout(
        ms_bundle, ms_decoder, obs_list, gt_frames, actions,
        image_size=args.image_size, render_size=args.render_size,
        label='MS+SR', warmup_steps=args.warmup_steps)

    # ── Save frames ───────────────────────────────────────────────────────────
    print('\n=== Save frames ===')
    save_frames(gt_frames,  out_dir, 'gt')
    save_frames(fwd_frames, out_dir, 'fwd_ar')
    save_frames(ms_frames,  out_dir, 'ms_sr')

    # Side-by-side comparison PNGs
    save_comparison_frames(
        gt_frames,
        {'FWD+EP-AR': fwd_frames, 'MS+SR': ms_frames},
        out_dir)

    # Optional GIF
    if args.gif:
        save_comparison_gif(
            gt_frames,
            {'FWD+EP-AR': fwd_frames, 'MS+SR': ms_frames},
            out_dir / 'rollout_comparison.gif',
            fps=args.fps)

    gt_helper.close()
    render_helper.close()
    print('\n=== Done ===')


if __name__ == '__main__':
    main()
