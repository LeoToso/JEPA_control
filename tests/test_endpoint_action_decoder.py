"""Tests for EndpointActionDecoder and endpoint action reconstruction mode.

Covers:
  - output shape [B, H_act, action_dim]
  - gradients reach the encoder through both endpoint encodings
  - no intermediate latent is passed to the endpoint decoder
  - windows do not cross episode boundaries
  - action normalization round-trip is correct
  - endpoint_sequence mode coexists with the old local inv_head
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch
import torch.nn.functional as F


# ── Decoder architecture tests ────────────────────────────────────────────────

def test_output_shape():
    """Output must be (B, H_act, action_dim)."""
    from models.endpoint_action_decoder import EndpointActionDecoder
    B, d, H_act = 8, 8, 5
    dec = EndpointActionDecoder(latent_dim=d, action_dim=1, H_act=H_act)
    out = dec(torch.randn(B, d), torch.randn(B, d))
    assert out.shape == (B, H_act, 1), f'Expected ({B},{H_act},1), got {out.shape}'


def test_output_shape_multi_action_dim():
    """Works for action_dim > 1."""
    from models.endpoint_action_decoder import EndpointActionDecoder
    B, d, H_act, a = 4, 16, 3, 2
    dec = EndpointActionDecoder(latent_dim=d, action_dim=a, H_act=H_act)
    out = dec(torch.randn(B, d), torch.randn(B, d))
    assert out.shape == (B, H_act, a)


def test_gradient_flows_through_z_start():
    """Loss gradient must reach z_start."""
    from models.endpoint_action_decoder import EndpointActionDecoder
    d, H_act = 8, 5
    dec = EndpointActionDecoder(latent_dim=d, action_dim=1, H_act=H_act)
    z_s = torch.randn(4, d, requires_grad=True)
    z_e = torch.randn(4, d)
    loss = F.mse_loss(dec(z_s, z_e), torch.zeros(4, H_act, 1))
    loss.backward()
    assert z_s.grad is not None and z_s.grad.abs().sum() > 0, 'No gradient through z_start'


def test_gradient_flows_through_z_end():
    """Loss gradient must reach z_end."""
    from models.endpoint_action_decoder import EndpointActionDecoder
    d, H_act = 8, 5
    dec = EndpointActionDecoder(latent_dim=d, action_dim=1, H_act=H_act)
    z_s = torch.randn(4, d)
    z_e = torch.randn(4, d, requires_grad=True)
    loss = F.mse_loss(dec(z_s, z_e), torch.zeros(4, H_act, 1))
    loss.backward()
    assert z_e.grad is not None and z_e.grad.abs().sum() > 0, 'No gradient through z_end'


def test_gradient_reaches_encoder_via_endpoint_loss():
    """Endpoint action loss must propagate gradient into encoder parameters."""
    from models.jepa import make_jepa
    from models.endpoint_action_decoder import EndpointActionDecoder
    model = make_jepa(latent_dim=8, action_latent_dim=1, action_encoder='linear',
                      frame_stack=1, image_size=32, patch_size=8,
                      vit_embed_dim=32, vit_depth=2, vit_num_heads=2,
                      predictor_hidden_dim=32, predictor_n_layers=2, predictor_window=1)
    dec = EndpointActionDecoder(latent_dim=8, action_dim=1, H_act=3, use_delta=True)

    obs_s = torch.randn(2, 3, 32, 32)
    obs_e = torch.randn(2, 3, 32, 32)
    z_s = model.encoder(obs_s)
    z_e = model.encoder(obs_e)
    loss = F.mse_loss(dec(z_s, z_e), torch.zeros(2, 3, 1))
    loss.backward()

    enc_grad_total = sum(
        p.grad.abs().sum().item()
        for p in model.encoder.parameters()
        if p.grad is not None
    )
    assert enc_grad_total > 0, 'Endpoint action loss produced no gradient in encoder'


def test_no_intermediate_latents_in_decoder():
    """Decoder first-layer input width must be 2*d (no_delta) or 3*d (delta) — no extras."""
    from models.endpoint_action_decoder import EndpointActionDecoder
    d = 8
    dec_nd = EndpointActionDecoder(latent_dim=d, H_act=5, use_delta=False)
    dec_d  = EndpointActionDecoder(latent_dim=d, H_act=5, use_delta=True)
    first_nd = dec_nd.net[0]
    first_d  = dec_d.net[0]
    assert first_nd.in_features == 2 * d, \
        f'use_delta=False: expected {2*d}, got {first_nd.in_features}'
    assert first_d.in_features == 3 * d, \
        f'use_delta=True: expected {3*d}, got {first_d.in_features}'


# ── Action normalization ───────────────────────────────────────────────────────

def test_action_normalization_round_trip():
    """Dividing raw actions by action_scale and multiplying back gives originals."""
    action_scale = 10.0
    raw = torch.FloatTensor([[-10.0], [0.0], [10.0], [5.3]])
    normalized = raw / action_scale
    assert normalized.abs().max().item() <= 1.0 + 1e-6
    recovered  = normalized * action_scale
    assert torch.allclose(raw, recovered, atol=1e-5)


def test_normalized_actions_zero_mean_baseline():
    """For uniformly random actions in [-1, 1], baseline (predict 0) MSE = variance."""
    actions = torch.zeros(1000, 5, 1).uniform_(-1.0, 1.0)
    baseline_mse = (actions ** 2).mean().item()
    # E[u²] for U[-1,1] = 1/3
    assert abs(baseline_mse - 1.0 / 3.0) < 0.05, \
        f'Baseline MSE={baseline_mse:.3f} not close to 1/3'


# ── Episode-boundary safety ───────────────────────────────────────────────────

def test_windows_do_not_cross_episode_boundary():
    """TrajectoryDataset valid_starts must never span two different episodes."""
    from data.dataset import TrajectoryDataset
    n = 10  # steps per episode
    ep_ids = np.array([0] * n + [1] * n, dtype=np.int32)
    data = {
        'obs':         np.zeros((2 * n, 64, 64, 3), dtype=np.uint8),
        'states':      np.zeros((2 * n, 4),           dtype=np.float32),
        'actions':     np.zeros((2 * n, 1),            dtype=np.float32),
        'next_obs':    np.zeros((2 * n, 64, 64, 3),   dtype=np.uint8),
        'next_states': np.zeros((2 * n, 4),            dtype=np.float32),
        'episode_ids': ep_ids,
        'splits':      {'train': np.arange(2 * n)},
    }
    H = 5
    ds = TrajectoryDataset(data, split='train', horizon=H)
    for start in ds.valid_starts:
        ep_window = ep_ids[start:start + H]
        assert np.all(ep_window == ep_window[0]), (
            f'Window at start={start} crosses episode: {ep_window}')


# ── Coexistence with old local IDM ────────────────────────────────────────────

def test_endpoint_sequence_coexists_with_local_inv_head():
    """Trainer must build both inv_head and endpoint_action_decoder when both are active."""
    from models.jepa import make_jepa
    from training.trainer import Trainer

    model = make_jepa(
        latent_dim=8, action_latent_dim=1, action_encoder='linear',
        frame_stack=2, image_size=64, patch_size=8,
        vit_embed_dim=64, vit_depth=2, vit_num_heads=2,
        predictor_hidden_dim=32, predictor_n_layers=2, predictor_window=3,
    )
    cfg = {
        'lambda_pred':                       1.0,
        'lambda_inv':                        1.0,
        'inv_frames':                        5,
        'inv_action_scale':                  10.0,
        'inv_hidden_dim':                    64,
        'lambda_action_reconstruction':      1.0,
        'action_reconstruction_mode':        'endpoint_sequence',
        'action_reconstruction_horizon':     5,
        'action_reconstruction_hidden_dim':  64,
        'action_reconstruction_n_layers':    2,
        'action_reconstruction_use_delta':   True,
        'action_reconstruction_action_scale': 1.0,
        'detach_targets':                    False,
        'use_target_encoder':                False,
        'predictor_window':                  3,
        'jacobian_every':                    10000,
        'epochs':                            1,
        'lr':                                1e-4,
        'weight_decay':                      0.0,
    }
    trainer = Trainer(model, cfg, save_dir='/tmp/test_endpoint_coexist', device='cpu')

    assert trainer.inv_head is not None,                 'inv_head must exist (lambda_inv > 0)'
    assert trainer.endpoint_action_decoder is not None,  'endpoint decoder must exist'
    assert trainer.phys_endpoint_decoder is not None,    'phys decoder must exist'

    # Run a forward pass — must produce both inv_loss and endpoint_action_loss
    B, H, C, HW = 2, 10, 6, 64
    batch = {
        'obs_seq': torch.randn(B, H + 1, C, HW, HW),
        'actions': torch.randn(B, H, 1) * 0.3,
        'states':  torch.randn(B, H + 1, 4),
    }
    loss, info = trainer._compute_loss(batch, is_train=True)

    assert 'inv_loss'              in info, 'inv_loss missing from info'
    assert 'endpoint_action_loss'  in info, 'endpoint_action_loss missing from info'
    assert loss.item() > 0


def test_endpoint_decoder_output_when_lambda_zero():
    """When lambda_action_reconstruction=0, endpoint_action_decoder must be None."""
    from models.jepa import make_jepa
    from training.trainer import Trainer

    model = make_jepa(latent_dim=8, action_latent_dim=1, action_encoder='linear',
                      frame_stack=1, image_size=64, patch_size=8,
                      vit_embed_dim=64, vit_depth=2, vit_num_heads=2,
                      predictor_hidden_dim=32, predictor_n_layers=2, predictor_window=1)
    cfg = {'lambda_action_reconstruction': 0.0, 'epochs': 1, 'lr': 1e-4, 'weight_decay': 0.0}
    trainer = Trainer(model, cfg, save_dir='/tmp/test_endpoint_disabled', device='cpu')
    assert trainer.endpoint_action_decoder is None
    assert trainer.phys_endpoint_decoder   is None
