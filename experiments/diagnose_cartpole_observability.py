#!/usr/bin/env python3
"""Diagnose CartPole angle observability in rendering and stored datasets."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml
from PIL import Image

from envs.cartpole_visual import ContinuousCartpoleVisual


def _environment(config_path: Path) -> ContinuousCartpoleVisual:
    with config_path.open() as handle:
        cfg = yaml.safe_load(handle)
    env = cfg["environment"]
    return ContinuousCartpoleVisual(
        frame_skip=int(env.get("frame_skip", 1)),
        image_size=int(env.get("image_size", 64)),
        action_range=tuple(env.get("action_range", [-10.0, 10.0])),
        mass_cart=float(env.get("mass_cart", 1.0)),
        mass_pole=float(env.get("mass_pole", 0.1)),
        pole_length=float(env.get("pole_length", 0.5)),
        gravity=float(env.get("gravity", 9.8)),
        dt=float(env.get("dt", 0.02)),
        theta_threshold=float(env.get("theta_threshold", 1.2)),
        seed=0,
    )


def diagnose_render(config_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = _environment(config_path)
    angles = [-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6]
    frames = {}
    try:
        for theta in angles:
            requested = np.array([0.0, 0.0, theta, 0.0], dtype=np.float32)
            frame, returned, _ = env.reset_to_state(requested)
            if not np.allclose(returned, requested):
                raise RuntimeError(
                    f"State-order check failed: requested={requested}, returned={returned}")
            frames[theta] = frame
            suffix = f"{theta:+.1f}".replace("+", "p").replace("-", "m").replace(".", "p")
            Image.fromarray(frame).save(output_dir / f"theta_{suffix}.png")

        montage = np.concatenate([frames[a] for a in angles], axis=1)
        Image.fromarray(montage).save(output_dir / "theta_montage.png")
        print("[render] state order verified: [x, x_dot, theta, theta_dot]")
        print(f"[render] montage: {output_dir / 'theta_montage.png'}")
        for magnitude in (0.1, 0.3, 0.6):
            pos, neg = frames[magnitude], frames[-magnitude]
            mad = np.abs(pos.astype(np.float32) - neg.astype(np.float32)).mean()
            changed = np.mean(np.any(pos != neg, axis=-1))
            mirror_mad = np.abs(
                pos.astype(np.float32) - np.flip(neg, axis=1).astype(np.float32)
            ).mean()
            print(
                f"[render] |theta|={magnitude:.1f}: +/− MAD={mad:.3f}, "
                f"changed_pixels={100*changed:.2f}%, mirror_MAD={mirror_mad:.3f}")
    finally:
        env.close()


def diagnose_dataset(dataset_dir: Path) -> None:
    names = ["x", "x_dot", "theta", "theta_dot"]
    for split in ("train", "val", "test"):
        path = dataset_dir / f"{split}.hdf5"
        if not path.exists():
            continue
        groups = {}
        with h5py.File(path, "r") as handle:
            for ep in handle["episodes"].values():
                typ = ep.attrs.get("trajectory_type", "unknown")
                if isinstance(typ, bytes):
                    typ = typ.decode()
                states = ep["states"][:].astype(np.float32)
                groups.setdefault(str(typ), []).append(states)
        print(f"[dataset:{split}]")
        for typ, chunks in sorted(groups.items()):
            states = np.concatenate(chunks)
            stats = []
            for index, name in enumerate(names):
                values = states[:, index]
                stats.append(
                    f"{name}:mean={values.mean():+.4f},std={values.std():.4f},"
                    f"range=[{values.min():+.3f},{values.max():+.3f}]")
            theta = states[:, 2]
            print(
                f"  {typ:10s} n={len(states):5d} "
                f"theta_pos={100*np.mean(theta > 0):5.1f}%  " + "  ".join(stats))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/cartpole_jepa_recovery_vit_joint.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/cartpole_observable_fs5"))
    parser.add_argument("--output", type=Path, default=Path("results/cartpole_angle_diagnostic"))
    args = parser.parse_args()
    diagnose_render(args.config, args.output)
    diagnose_dataset(args.data)


if __name__ == "__main__":
    main()
