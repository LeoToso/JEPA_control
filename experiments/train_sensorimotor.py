#!/usr/bin/env python
"""Train the Ivashkov et al. sensorimotor world model on visual CartPole."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from data.dataset import load_discrete_dataset_meta, make_discrete_dataloaders
from models.sensorimotor_world_model import SensorimotorWorldModel
from losses.sigreg import sigreg_loss


def prepare_obs(x, device):
    return x.to(device, non_blocking=True).float().div_(255.)


def losses(model, batch, device, lambda_inverse, lambda_sigreg,
           sigreg_num_slices, sigreg_num_points,
           sigreg_variance_floor_weight, lambda_state_recon=0.0,
           lambda_rollout=0.0, lambda_fwd=1.0, lambda_inverse_1step=0.0):
    obs = batch['obs_seq']
    actions = batch['actions'].to(device, non_blocking=True).float()
    b, steps = obs.shape[:2]
    current = prepare_obs(obs.reshape(b * steps, *obs.shape[2:]), device)
    previous = torch.cat((obs[:, :1], obs[:, :-1]), dim=1)
    previous = prepare_obs(
        previous.reshape(b * steps, *previous.shape[2:]), device)
    proprio = None
    if model.use_proprio:
        proprio = batch['states'].reshape(b * steps, -1).to(
            device, non_blocking=True).float()
        if model.proprio_indices is not None:
            proprio = proprio[:, model.proprio_indices]
    z = model.encode(current, previous, proprio)
    z = z.reshape(b, steps, -1)
    z_t, z_tp1 = z[:, :-1], z[:, 1:]

    # Build action context: (b, steps-1, action_context_dim).
    ctx = model.action_context_dim
    a_seq = actions if actions.ndim == 3 else actions.unsqueeze(-1)  # (b, steps-1, raw_d)
    raw_d = a_seq.shape[-1]
    if model.use_action_history:
        # History of k raw_d-dim actions per context window; k = ctx // raw_d.
        # Works for scalar (raw_d=1, k=ctx) and multi-dim (raw_d=2, k=ctx//2) actions.
        k = ctx // raw_d
        prev_act = batch.get('prev_actions')
        if prev_act is not None:
            prev_act = prev_act.to(device, non_blocking=True).float()
            # Dataset supplies ctx-1 prev steps; trim to the k-1 we actually need.
            prev_act = prev_act[:, -(k - 1):, :] if k > 1 else prev_act[:, :0, :]
            a_full = torch.cat([prev_act, a_seq], dim=1)          # (b, T+k-1, raw_d)
        else:
            pad = a_seq.new_zeros(b, k - 1, raw_d)
            a_full = torch.cat([pad, a_seq], dim=1)               # (b, T+k-1, raw_d)
        # Sliding windows oldest→newest, flattened: (b, T, raw_d, k) → (b, T, ctx)
        a_windows = a_full.unfold(1, k, 1)                        # (b, T, raw_d, k)
        action_ctx = a_windows.permute(0, 1, 3, 2).reshape(b, steps - 1, ctx)
    else:
        # Tile each raw_d-dim action to fill ctx slots.
        action_ctx = model.expand_action(
            a_seq.reshape(b * (steps - 1), raw_d)).reshape(b, steps - 1, ctx)

    pred = model.predict(z_t, action_ctx)
    fwd = F.mse_loss(pred, z_tp1)
    if model.inverse_model is not None and lambda_inverse > 0:
        # Inverse model takes the full latent sequence z (b, steps, d) and
        # predicts all steps-1 actions in one shot.
        if model.inverse_model.seq_len != steps:
            raise ValueError(
                f'inverse_seq_len={model.inverse_model.seq_len} != steps={steps}. '
                f'Set inverse_seq_len=horizon+1 in model config.')
        pred_action = model.inverse_model(z)   # (b, steps-1, action_dim)
        inv = F.mse_loss(pred_action, action_ctx)
    elif model.endpoint_inverse_model is not None and lambda_inverse > 0:
        # Endpoint inverse: decoder sees only z_0 and z_H (no intermediate latents).
        H = model.endpoint_inverse_model.horizon
        if H != steps - 1:
            raise ValueError(
                f'endpoint_inverse_horizon={H} != horizon={steps-1}. '
                f'Set endpoint_inverse_horizon=horizon in model config.')
        pred_action = model.endpoint_inverse_model(z[:, 0], z[:, -1])  # (b, H, action_dim)
        inv = F.mse_loss(pred_action, action_ctx)
    else:
        inv = fwd.new_zeros(())
    if lambda_sigreg > 0:
        z_sig = z.reshape(-1, z.shape[-1])
        sig = sigreg_loss(
            z_sig, num_slices=sigreg_num_slices,
            num_points=sigreg_num_points,
            variance_floor_weight=sigreg_variance_floor_weight)
    else:
        sig = fwd.new_zeros(())
    if model.inverse_model_1step is not None and lambda_inverse_1step > 0:
        # Single-step inverse: applied to every consecutive (z_t, z_{t+1}) pair.
        z_pairs = torch.stack([z[:, :-1], z[:, 1:]], dim=2)  # (b, steps-1, 2, d)
        bp = b * (steps - 1)
        pred_1step = model.inverse_model_1step(
            z_pairs.reshape(bp, 2, -1))                       # (bp, 1, action_dim)
        inv1 = F.mse_loss(pred_1step[:, 0], action_ctx.reshape(bp, -1))
    else:
        inv1 = fwd.new_zeros(())
    if model.state_decoder is not None and lambda_state_recon > 0:
        state_target = batch['states'].to(device, non_blocking=True).float()
        state_pred = model.state_decoder(z.reshape(b * steps, -1)).reshape(b, steps, -1)
        sr = F.mse_loss(state_pred, state_target)
    else:
        sr = fwd.new_zeros(())
    if lambda_rollout > 0 and steps > 1:
        # Recursive rollout: chain predictions ẑ_{k+1} = f(ẑ_k, ā_k), no teacher forcing.
        z_curr = z[:, :1]  # (b, 1, d) — start from encoded z_0
        rl = fwd.new_zeros(())
        for k in range(steps - 1):
            z_pred = model.predict(z_curr, action_ctx[:, k:k+1])  # (b, 1, d)
            rl = rl + F.mse_loss(z_pred[:, 0], z[:, k + 1])
            z_curr = z_pred  # full BPTT: gradient flows through the chain
        rl = rl / (steps - 1)
    else:
        rl = fwd.new_zeros(())
    total = (lambda_fwd * fwd + lambda_inverse * inv + lambda_sigreg * sig
             + lambda_state_recon * sr + lambda_rollout * rl
             + lambda_inverse_1step * inv1)
    return total, fwd, inv, sig, sr, rl, z[:, 0]


@torch.no_grad()
def validate(model, loader, device, loss_cfg, collect_probe=False):
    model.eval()
    total = np.zeros(6, dtype=np.float64)
    count = 0
    zs, states = [], []
    for batch in loader:
        loss, fwd, inv, sig, sr, rl, z = losses(model, batch, device, **loss_cfg)
        n = batch['obs_seq'].shape[0]
        total += n * np.array(
            [float(loss), float(fwd), float(inv), float(sig), float(sr), float(rl)])
        count += n
        if collect_probe:
            zs.append(z.cpu().numpy())
            states.append(batch['states'][:, 0].numpy())
    out = total / max(count, 1)
    if collect_probe:
        return out, np.concatenate(zs), np.concatenate(states)
    return out, None, None


def ridge_probe(z_train, y_train, z_test, y_test, ridge=1e-3):
    zm, zs = z_train.mean(0), np.maximum(z_train.std(0), 1e-6)
    ym = y_train.mean(0)
    x = (z_train - zm) / zs
    xt = (z_test - zm) / zs
    x = np.concatenate((x, np.ones((len(x), 1))), axis=1)
    xt = np.concatenate((xt, np.ones((len(xt), 1))), axis=1)
    reg = ridge * np.eye(x.shape[1]); reg[-1, -1] = 0.
    w = np.linalg.solve(x.T @ x + reg, x.T @ (y_train - ym))
    pred = xt @ w + ym
    denom = np.sum((y_test - y_test.mean(0)) ** 2, axis=0)
    return 1. - np.sum((pred - y_test) ** 2, axis=0) / np.maximum(denom, 1e-12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--save-dir', required=True)
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', default=None,
                   help='Path to checkpoint to resume from (e.g. checkpoints/checkpoint_best.pt)')
    p.add_argument('--num-workers', type=int, default=None,
                   help='Override num_workers in config (use 0 when system RAM is tight)')
    p.add_argument('--probe-every', type=int, default=None,
                   help='Override probe_every in config (use a large number or 9999 to skip probes)')
    p.add_argument('--checkpoint-every', type=int, default=None,
                   help='Override checkpoint_every in config')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mc, tc = cfg['model'], dict(cfg['training'])
    if args.epochs is not None:
        tc['epochs'] = args.epochs
    if args.batch_size is not None:
        tc['batch_size'] = args.batch_size
    if args.num_workers is not None:
        tc['num_workers'] = args.num_workers
    if args.probe_every is not None:
        tc['probe_every'] = args.probe_every
    if args.checkpoint_every is not None:
        tc['checkpoint_every'] = args.checkpoint_every
    out = Path(args.save_dir)
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (out / 'config.yaml').write_text(yaml.safe_dump({'model': mc, 'training': tc}))

    meta = load_discrete_dataset_meta(args.data)
    action_scale = float(meta.get('action_scale', 1.0)) \
        if tc.get('normalize_actions', True) else 1.0
    use_proprio = bool(mc.get('use_proprio', False))
    state_mean = meta['state_mean'] if use_proprio else None
    state_std = meta['state_std'] if use_proprio else None
    action_context_dim = int(mc.get('action_context_dim', 5))
    use_action_history = bool(mc.get('use_action_history', False))
    loaders = make_discrete_dataloaders(
        args.data, batch_size=int(tc['batch_size']),
        num_workers=int(tc.get('num_workers', 2)),
        horizon=int(tc.get('horizon', 1)), frame_stack=int(mc.get('frame_stack', 1)),
        state_mean=state_mean, state_std=state_std,
        action_scale=action_scale, target_image_size=int(mc.get('image_size', 64)),
        balanced_sampling=bool(tc.get('balanced_sampling', False)),
        trajectory_type_sampling_weights=tc.get(
            'trajectory_type_sampling_weights'),
        action_context_dim=action_context_dim if use_action_history else 1)
    if 'train' not in loaders:
        import os
        found = [f for f in os.listdir(args.data) if f.endswith('.hdf5')]
        raise RuntimeError(
            f'No train split loaded from {args.data!r}. '
            f'HDF5 files found: {found}. '
            f'Expected train.hdf5 / val.hdf5 / test.hdf5 in that directory.')
    print(f'Device: {device}')
    print(f'[data] train={len(loaders["train"].dataset)} '
          f'val={len(loaders["val"].dataset)} action_scale={action_scale:g}')

    model = SensorimotorWorldModel(mc).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    d_actual = next(iter(model.predictor.parameters())).shape[-1]
    print(f'[model] SensorimotorWM {n_params:,} trainable params '
          f'latent={d_actual} '
          f'encoder={mc.get("encoder_type", "vit")} '
          f'predictor(depth={mc["predictor_depth"]})')
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(tc['lr']),
        weight_decay=float(tc.get('weight_decay', 1e-3)))
    lambda_fwd = float(tc.get('lambda_fwd', 1.0))
    lambda_inv = float(tc.get('lambda_inverse', 1.0))
    lambda_inv1 = float(tc.get('lambda_inverse_1step', 0.0))
    lambda_sigreg = float(tc.get('lambda_sigreg', 0.0))
    lambda_sr = float(tc.get('lambda_state_recon', 0.0))
    lambda_rl = float(tc.get('lambda_rollout', 0.0))
    if lambda_inv > 0 and model.inverse_model is None and model.endpoint_inverse_model is None:
        raise ValueError(
            'lambda_inverse > 0 requires model.use_inverse_model=true '
            'or model.use_endpoint_inverse=true')
    if lambda_inv1 > 0 and model.inverse_model_1step is None:
        raise ValueError(
            'lambda_inverse_1step > 0 requires model.use_inverse_model_1step=true')
    if lambda_sr > 0 and model.state_decoder is None:
        raise ValueError(
            'lambda_state_recon > 0 requires model.use_state_decoder=true')
    loss_cfg = {
        'lambda_inverse': lambda_inv,
        'lambda_sigreg': lambda_sigreg,
        'sigreg_num_slices': int(tc.get('sigreg_num_slices', 128)),
        'sigreg_num_points': int(tc.get('sigreg_num_points', 17)),
        'sigreg_variance_floor_weight': float(
            tc.get('sigreg_variance_floor_weight', 0.0)),
        'lambda_state_recon': lambda_sr,
        'lambda_rollout': lambda_rl,
        'lambda_fwd': lambda_fwd,
        'lambda_inverse_1step': lambda_inv1,
    }
    print(f'[loss] lambda_fwd={lambda_fwd:g} lambda_inverse={lambda_inv:g} '
          f'lambda_inverse_1step={lambda_inv1:g} '
          f'lambda_sigreg={lambda_sigreg:g} lambda_state_recon={lambda_sr:g} '
          f'lambda_rollout={lambda_rl:g}')
    epochs = int(tc['epochs'])
    warmup_epochs = int(tc.get('warmup_epochs', 5))
    base_lr = float(tc['lr'])
    min_lr = float(tc.get('min_lr', 1e-6))

    def lr_multiplier(epoch_index):
        if epoch_index < warmup_epochs:
            return float(epoch_index + 1) / max(warmup_epochs, 1)
        progress = ((epoch_index - warmup_epochs) /
                    max(epochs - warmup_epochs - 1, 1))
        cosine = .5 * (1. + math.cos(math.pi * progress))
        return (min_lr + (base_lr - min_lr) * cosine) / base_lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    start_epoch = 1
    best_val = float('inf')
    best_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = int(ckpt['epoch']) + 1
        best_val = float(ckpt.get('val_loss', float('inf')))
        print(f'[resume] from epoch={ckpt["epoch"]} val={best_val:.5f} '
              f'→ continuing from epoch {start_epoch}')
    history = []

    for epoch in range(start_epoch, epochs + 1):
        start = time.time()
        model.train()
        sums = np.zeros(6, dtype=np.float64); count = 0
        for batch in loaders['train']:
            loss, fwd, inv, sig, sr, rl, _ = losses(
                model, batch, device, **loss_cfg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            n = batch['obs_seq'].shape[0]
            sums += n * np.array([
                float(loss.detach()), float(fwd.detach()),
                float(inv.detach()), float(sig.detach()),
                float(sr.detach()), float(rl.detach())])
            count += n
        train = sums / count
        do_probe = epoch % int(tc.get('probe_every', 10)) == 0
        val, zv, yv = validate(
            model, loaders['val'], device, loss_cfg, do_probe)
        row = {'epoch': epoch, 'train': train.tolist(), 'val': val.tolist()}
        current_lr = float(optimizer.param_groups[0]['lr'])
        msg = (f'[Epoch {epoch:3d}/{epochs}] train={train[0]:.5f} '
               f'fwd={train[1]:.5f} inv={train[2]:.5f} '
               f'sig={train[3]:.5f} sr={train[4]:.5f} rl={train[5]:.5f} '
               f'val={val[0]:.5f} vfwd={val[1]:.5f} vinv={val[2]:.5f} '
               f'vsig={val[3]:.5f} vsr={val[4]:.5f} vrl={val[5]:.5f} '
               f'lr={current_lr:.2e}')
        if do_probe:
            _, ztr, ytr = validate(
                model, loaders['train'], device, loss_cfg, True)
            r2 = ridge_probe(ztr, ytr, zv, yv)
            row['probe_r2'] = r2.tolist()
            msg += (' R2=[' + ','.join(f'{x:.3f}' for x in r2) + ']')
        print(msg + f' dt={time.time() - start:.1f}s')
        row['lr'] = current_lr
        history.append(row)

        checkpoint = {
            'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'model_config': mc,
            'training_config': tc, 'action_scale': action_scale,
            'state_mean': state_mean, 'state_std': state_std,
            'val_loss': float(val[0]),
        }

        if val[0] < best_val:
            best_val = float(val[0])
            best_epoch = epoch
            torch.save(checkpoint, out / 'checkpoints' / 'checkpoint_best.pt')
            torch.save({'model': model.state_dict(), 'model_config': mc,
                        'action_scale': action_scale, 'epoch': epoch,
                        'state_mean': state_mean, 'state_std': state_std,
                        'val_loss': best_val}, out / 'model_selected.pt')
            print(f'[best] epoch={epoch} val={best_val:.5f}')

        if epoch % int(tc.get('checkpoint_every', 10)) == 0 or epoch == epochs:
            torch.save(checkpoint,
                       out / 'checkpoints' / f'checkpoint_epoch{epoch:04d}.pt')
        (out / 'history.json').write_text(json.dumps(history, indent=2))
        scheduler.step()

    torch.save({'model': model.state_dict(), 'model_config': mc,
                'action_scale': action_scale, 'state_mean': state_mean,
                'state_std': state_std}, out / 'model_final.pt')
    print(f'[selection] best_epoch={best_epoch} val={best_val:.5f}')
    print(f'[done] {out / "model_final.pt"}')


if __name__ == '__main__':
    main()
