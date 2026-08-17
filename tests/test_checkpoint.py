import tempfile
import os

import numpy as np
import torch

from jepa_lds.checkpoint import load_checkpoint, save_checkpoint
from jepa_lds.data import make_observation_model, generate_dataset
from jepa_lds.systems import make_double_mode_system
from jepa_lds.train import TrainConfig, train_jepa


def test_save_load_round_trip_encoder_predictor_only():
    system = make_double_mode_system()
    obs_model = make_observation_model(system, obs_dim_signal=6, n_distractor=8, seed=0)
    train_batch = generate_dataset(system, obs_model, 20, 10, seed=0, x0_std=0.03, action_std=0.3, state_clip=15.0)

    cfg = TrainConfig(latent_dim=2, horizon=4, outer_rounds=2, inner_epochs=2, lambda_pred_ms=1.0, seed=0)
    enc, pred, dec, _hist = train_jepa(system, obs_model, train_batch, cfg, verbose=False)
    assert dec is None

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.pt")
        save_checkpoint(path, enc, pred, dec, cfg, extra={"note": "test"})
        assert os.path.exists(path)

        enc2, pred2, dec2, cfg2, extra = load_checkpoint(path, obs_dim=obs_model.p, action_dim=system.m)

    assert dec2 is None
    assert extra == {"note": "test"}
    assert cfg2.latent_dim == cfg.latent_dim
    assert cfg2.horizon == cfg.horizon

    y = torch.randn(5, obs_model.p)
    with torch.no_grad():
        z1 = enc(y)
        z2 = enc2(y)
    assert torch.allclose(z1, z2)

    A_z1, B_z1 = pred.matrices()
    A_z2, B_z2 = pred2.matrices()
    assert np.allclose(A_z1, A_z2)
    assert np.allclose(B_z1, B_z2)


def test_save_load_round_trip_with_decoder():
    system = make_double_mode_system()
    obs_model = make_observation_model(system, obs_dim_signal=6, n_distractor=8, seed=0)
    train_batch = generate_dataset(system, obs_model, 20, 10, seed=0, x0_std=0.03, action_std=0.3, state_clip=15.0)

    cfg = TrainConfig(
        latent_dim=2, horizon=4, outer_rounds=2, inner_epochs=2,
        lambda_pred_ms=1.0, lambda_actrecon_ms=1.0, seed=0,
    )
    enc, pred, dec, _hist = train_jepa(system, obs_model, train_batch, cfg, verbose=False)
    assert dec is not None

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.pt")
        save_checkpoint(path, enc, pred, dec, cfg)
        enc2, pred2, dec2, cfg2, extra = load_checkpoint(path, obs_dim=obs_model.p, action_dim=system.m)

    assert dec2 is not None
    assert type(dec2).__name__ == type(dec).__name__
    assert extra == {}

    z_window = torch.randn(3, cfg.horizon + 1, cfg.latent_dim)
    with torch.no_grad():
        a1 = dec(z_window)
        a2 = dec2(z_window)
    assert torch.allclose(a1, a2)

