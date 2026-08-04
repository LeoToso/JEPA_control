"""Nonlinear, finite-amplitude CartPole instability probes for JEPA checkpoints.

No model or environment Jacobian is computed.  A frozen post-hoc ridge probe is
fit on encoded training observations and evaluated on held-out observations.
Symmetric nonlinear rollouts then compare the true macro dynamics with decoded
JEPA predictor rollouts using trajectory separation, finite-time divergence
rates, and angle escape times.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import torch
import yaml

from data.dataset import load_discrete_dataset_meta
from experiments.evaluate_cartpole_control import build_model, make_env


STATE_NAMES = ("x", "x_dot", "theta", "theta_dot")


def obs_tensor(obs: np.ndarray, device: torch.device) -> torch.Tensor:
    return (torch.from_numpy(obs).float().permute(2, 0, 1)
            .unsqueeze(0).to(device) / 255.0)


def wrap_angle(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x).copy()
    y[..., 2] = np.arctan2(np.sin(y[..., 2]), np.cos(y[..., 2]))
    return y


def _sample_frame_refs(path: Path, n_samples: int, seed: int):
    """Sample (episode, frame) references without loading image arrays."""
    with h5py.File(path, "r") as f:
        refs = []
        for ep_key in f["episodes"].keys():
            n = len(f["episodes"][ep_key]["states"])
            refs.extend((ep_key, t) for t in range(n))
    rng = np.random.RandomState(seed)
    if n_samples < len(refs):
        idx = rng.choice(len(refs), size=n_samples, replace=False)
        refs = [refs[int(i)] for i in idx]
    return refs


def encode_split(model, dataset_dir: str, split: str, n_samples: int,
                 batch_size: int, seed: int, device: torch.device):
    """Encode sampled real observations with their actual preceding frame."""
    path = Path(dataset_dir) / f"{split}.hdf5"
    refs = _sample_frame_refs(path, n_samples, seed)
    zs, ys = [], []
    with h5py.File(path, "r") as f:
        for start in range(0, len(refs), batch_size):
            rows = refs[start:start + batch_size]
            curr, prev, states = [], [], []
            for ep_key, t in rows:
                ep = f["episodes"][ep_key]
                curr.append(ep["observations"][t])
                prev.append(ep["observations"][max(t - 1, 0)])
                states.append(ep["states"][t])
            curr_t = (torch.from_numpy(np.stack(curr)).float()
                      .permute(0, 3, 1, 2).to(device) / 255.0)
            prev_t = (torch.from_numpy(np.stack(prev)).float()
                      .permute(0, 3, 1, 2).to(device) / 255.0)
            with torch.no_grad():
                z = model.encode_obs(curr_t, prev_t).cpu().numpy()
            zs.append(z)
            ys.append(np.asarray(states, dtype=np.float64))
    return np.concatenate(zs), wrap_angle(np.concatenate(ys))


def fit_ridge_probe(z: np.ndarray, states: np.ndarray, state_mean: np.ndarray,
                    state_std: np.ndarray, ridge: float):
    """Fit normalized-state ridge regression with an unregularized intercept."""
    y = (states - state_mean) / state_std
    z_mean, y_mean = z.mean(0), y.mean(0)
    zc, yc = z - z_mean, y - y_mean
    gram = zc.T @ zc
    w = np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), zc.T @ yc)
    b = y_mean - z_mean @ w
    return w, b


def decode(z: np.ndarray, w: np.ndarray, b: np.ndarray,
           state_mean: np.ndarray, state_std: np.ndarray):
    pred = (np.asarray(z) @ w + b) * state_std + state_mean
    return wrap_angle(pred)


def probe_metrics(truth: np.ndarray, pred: np.ndarray):
    err = pred - truth
    err[:, 2] = np.arctan2(np.sin(err[:, 2]), np.cos(err[:, 2]))
    mse = np.mean(err ** 2, axis=0)
    ss_res = np.sum(err ** 2, axis=0)
    ss_tot = np.sum((truth - truth.mean(0)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return {
        "n": int(len(truth)),
        "r2": r2.tolist(),
        "r2_mean": float(np.mean(r2)),
        "rmse": np.sqrt(mse).tolist(),
    }


def learned_step(model, z: torch.Tensor, raw_action: float,
                 action_scale: float, window: int):
    """One predictor macro-step; current experiments use W=1."""
    if window != 1:
        raise NotImplementedError(
            "Nonlinear instability probe currently requires predictor_window=1")
    u = torch.full((z.shape[0], 1, model.config.action_dim),
                   float(raw_action) / float(action_scale), device=z.device)
    with torch.no_grad():
        return model.predict(z.unsqueeze(1), u)


def finite_rate(distance: np.ndarray, dt: float):
    """Finite-amplitude rate at each time, relative to the initial separation."""
    d = np.asarray(distance, dtype=np.float64)
    out = np.full_like(d, np.nan)
    if len(d) == 0 or not np.isfinite(d[0]) or d[0] <= 1e-12:
        return out
    for k in range(1, len(d)):
        if np.isfinite(d[k]) and d[k] > 1e-12:
            out[k] = np.log(d[k] / d[0]) / (k * dt)
    out[0] = 0.0
    return out


def first_escape(theta_abs: np.ndarray, threshold: float):
    idx = np.flatnonzero(np.asarray(theta_abs) >= threshold)
    return int(idx[0]) if len(idx) else None


def paired_rollout(model, cfg, w, b, state_mean, state_std, epsilon,
                   direction, horizon, action_scale, seed, device):
    """Run a symmetric true/learned pair from dynamically valid frame histories."""
    env_p, env_m = make_env(cfg, seed), make_env(cfg, seed + 1)
    pre_p = epsilon * direction
    pre_m = -epsilon * direction
    prev_p, _, _ = env_p.reset_to_state(pre_p.astype(np.float32))
    prev_m, _, _ = env_m.reset_to_state(pre_m.astype(np.float32))
    obs_p, state_p, _, done_p, _ = env_p.step(0.0)
    obs_m, state_m, _, done_m, _ = env_m.step(0.0)
    with torch.no_grad():
        zp = model.encode_obs(obs_tensor(obs_p, device), obs_tensor(prev_p, device))
        zm = model.encode_obs(obs_tensor(obs_m, device), obs_tensor(prev_m, device))

    # Center post-hoc decoded trajectories at the decoder's own equilibrium.
    env_eq = make_env(cfg, seed + 2)
    eq_obs, _, _ = env_eq.reset_to_state(np.zeros(4, dtype=np.float32))
    with torch.no_grad():
        z_eq = model.encode_obs(obs_tensor(eq_obs, device),
                                obs_tensor(eq_obs, device)).cpu().numpy()
    decoded_eq = decode(z_eq, w, b, state_mean, state_std)[0]
    env_eq.close()

    true_p, true_m = [state_p.copy()], [state_m.copy()]
    pred_p = [decode(zp.cpu().numpy(), w, b, state_mean, state_std)[0] - decoded_eq]
    pred_m = [decode(zm.cpu().numpy(), w, b, state_mean, state_std)[0] - decoded_eq]
    window = int(cfg["model"].get("predictor_window", 1))
    for _ in range(horizon):
        if done_p or done_m:
            break
        obs_p, state_p, _, done_p, _ = env_p.step(0.0)
        obs_m, state_m, _, done_m, _ = env_m.step(0.0)
        zp = learned_step(model, zp, 0.0, action_scale, window)
        zm = learned_step(model, zm, 0.0, action_scale, window)
        true_p.append(state_p.copy()); true_m.append(state_m.copy())
        pred_p.append(decode(zp.cpu().numpy(), w, b, state_mean, state_std)[0]
                      - decoded_eq)
        pred_m.append(decode(zm.cpu().numpy(), w, b, state_mean, state_std)[0]
                      - decoded_eq)
    env_p.close(); env_m.close()
    return tuple(np.asarray(x) for x in (true_p, true_m, pred_p, pred_m))


def aggregate_rollouts(rows, state_std, dt, thresholds):
    records = []
    rate_errors, curve_errors = [], []
    escape = {str(t): [] for t in thresholds}
    for row in rows:
        tp, tm, pp, pm = (row[k] for k in ("true_plus", "true_minus",
                                           "pred_plus", "pred_minus"))
        n = min(len(tp), len(tm), len(pp), len(pm))
        tp, tm, pp, pm = tp[:n], tm[:n], pp[:n], pm[:n]
        td = np.linalg.norm((tp - tm) / state_std, axis=1)
        pd = np.linalg.norm((pp - pm) / state_std, axis=1)
        tr, pr = finite_rate(td, dt), finite_rate(pd, dt)
        valid = np.isfinite(tr) & np.isfinite(pr)
        if np.any(valid[1:]):
            rate_errors.append(float(np.mean(np.abs(tr[1:][valid[1:]] - pr[1:][valid[1:]]))))
            curve_errors.append(float(np.mean(np.abs(
                np.log(np.maximum(td[valid], 1e-12) / max(td[0], 1e-12))
                - np.log(np.maximum(pd[valid], 1e-12) / max(pd[0], 1e-12))))))
        true_theta = np.maximum(np.abs(tp[:, 2]), np.abs(tm[:, 2]))
        pred_theta = np.maximum(np.abs(pp[:, 2]), np.abs(pm[:, 2]))
        for threshold in thresholds:
            et = first_escape(true_theta, threshold)
            ep = first_escape(pred_theta, threshold)
            escape[str(threshold)].append({"true_step": et, "learned_step": ep})
        records.append({
            "epsilon": row["epsilon"], "direction": row["direction"],
            "true_distance": td.tolist(), "learned_distance": pd.tolist(),
            "true_rate": tr.tolist(), "learned_rate": pr.tolist(),
        })

    escape_summary = {}
    for threshold in thresholds:
        vals = escape[str(threshold)]
        censor = max(max((len(r["true_distance"]) for r in records), default=1) - 1, 0) + 1
        true_steps = np.array([v["true_step"] if v["true_step"] is not None else censor for v in vals])
        pred_steps = np.array([v["learned_step"] if v["learned_step"] is not None else censor for v in vals])
        escaped_true = np.array([v["true_step"] is not None for v in vals])
        escaped_pred = np.array([v["learned_step"] is not None for v in vals])
        escape_summary[str(threshold)] = {
            "mean_abs_step_error_censored": float(np.mean(np.abs(true_steps - pred_steps))),
            "escape_classification_agreement": float(np.mean(escaped_true == escaped_pred)),
            "true_escape_fraction": float(np.mean(escaped_true)),
            "learned_escape_fraction": float(np.mean(escaped_pred)),
            "pairs": vals,
        }
    return {
        "finite_rate_mae": float(np.mean(rate_errors)) if rate_errors else None,
        "log_growth_curve_mae": float(np.mean(curve_errors)) if curve_errors else None,
        "escape_times": escape_summary,
        "curves": records,
    }


def save_plot(result, path: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib unavailable; skipping divergence plot")
        return
    curves = result["nonlinear_instability"]["curves"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for row in curves:
        label = f'{row["direction"]}, eps={row["epsilon"]:g}'
        axes[0].plot(row["true_distance"], "-", alpha=.8, label="true " + label)
        axes[0].plot(row["learned_distance"], "--", alpha=.8, label="learned " + label)
        axes[1].plot(row["true_rate"], "-", alpha=.8)
        axes[1].plot(row["learned_rate"], "--", alpha=.8)
    axes[0].set_yscale("log"); axes[0].set_title("Symmetric separation")
    axes[0].set_xlabel("macro step"); axes[0].set_ylabel("normalized distance")
    axes[1].set_title("Finite-amplitude divergence rate")
    axes[1].set_xlabel("macro step"); axes[1].set_ylabel("rate [1/s]")
    if len(curves) <= 8:
        axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--probe-train-samples", type=int, default=4000)
    p.add_argument("--probe-test-samples", type=int, default=1500)
    p.add_argument("--probe-batch-size", type=int, default=128)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--theta-eps", type=float, nargs="+", default=[.01, .02, .05, .1])
    p.add_argument("--theta-dot-eps", type=float, nargs="+", default=[.05, .1, .2])
    p.add_argument("--escape-thresholds", type=float, nargs="+", default=[.02, .05, .1, .2])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model, arch = build_model(cfg["model"], checkpoint, device)
    cfg = dict(cfg); cfg["model"] = arch
    meta = load_discrete_dataset_meta(args.data)
    state_mean = np.asarray(meta.get("state_mean", np.zeros(4)), dtype=np.float64)
    state_std = np.asarray(meta.get("state_std", np.ones(4)), dtype=np.float64)
    action_scale = float(meta.get("action_scale", 1.0))

    print("[probe] encoding frozen train/test representations ...")
    z_train, s_train = encode_split(model, args.data, "train",
                                    args.probe_train_samples, args.probe_batch_size,
                                    args.seed, device)
    z_test, s_test = encode_split(model, args.data, "test",
                                  args.probe_test_samples, args.probe_batch_size,
                                  args.seed + 1, device)
    w, b = fit_ridge_probe(z_train, s_train, state_mean, state_std, args.ridge)
    test_pred = decode(z_test, w, b, state_mean, state_std)
    state_probe = probe_metrics(s_test, test_pred)
    print("[state probe] R2=" + ", ".join(
        f"{n}={v:.3f}" for n, v in zip(STATE_NAMES, state_probe["r2"])))

    rows = []
    specs = [("theta", e, np.array([0., 0., 1., 0.])) for e in args.theta_eps]
    specs += [("theta_dot", e, np.array([0., 0., 0., 1.]))
              for e in args.theta_dot_eps]
    for i, (name, epsilon, direction) in enumerate(specs):
        tp, tm, pp, pm = paired_rollout(
            model, cfg, w, b, state_mean, state_std, epsilon, direction,
            args.horizon, action_scale, args.seed + 10 * i, device)
        rows.append({"epsilon": float(epsilon), "direction": name,
                     "true_plus": tp, "true_minus": tm,
                     "pred_plus": pp, "pred_minus": pm})
        print(f"[rollout] {name} eps={epsilon:g} steps={len(tp)-1}")

    macro_dt = (float(cfg["environment"].get("dt", .02))
                * int(cfg["environment"].get("frame_skip", 1)))
    instability = aggregate_rollouts(rows, state_std, macro_dt,
                                     args.escape_thresholds)
    result = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "state_names": STATE_NAMES,
        "macro_dt": macro_dt,
        "posthoc_state_probe": state_probe,
        "nonlinear_instability": instability,
        "probe_protocol": {
            "uses_jacobian": False,
            "action_sequence": "zero",
            "symmetric_pairs": True,
            "probe_train_samples": len(z_train),
            "probe_test_samples": len(z_test),
            "ridge": args.ridge,
        },
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=True))
    plot_path = out.with_suffix(".png")
    save_plot(result, plot_path)
    print(f'[nonlinear] rate_MAE={instability["finite_rate_mae"]}  '
          f'curve_MAE={instability["log_growth_curve_mae"]}')
    print(f"[done] {out}")
    if plot_path.exists():
        print(f"[done] {plot_path}")


if __name__ == "__main__":
    main()
