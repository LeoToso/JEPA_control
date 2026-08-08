import numpy as np

from jepa_lds.control import design_latent_controller, evaluate_controller
from jepa_lds.data import make_observation_model, generate_dataset
from jepa_lds.diagnostics import unstable_mode_retention
from jepa_lds.models import FixedLinearEncoder
from jepa_lds.systems import make_double_mode_system
from jepa_lds.train import TrainConfig, train_jepa


def _setup(seed=0):
    system = make_double_mode_system()
    obs_model = make_observation_model(system, obs_dim_signal=6, n_distractor=8, seed=0)
    train_batch = generate_dataset(system, obs_model, 300, 10, seed=seed, x0_std=0.03, action_std=0.3, state_clip=15.0)
    val_batch = generate_dataset(system, obs_model, 80, 10, seed=seed + 10_000, x0_std=0.03, action_std=0.3, state_clip=15.0)
    return system, obs_model, train_batch, val_batch


def test_oracle_closed_form_fit_recovers_true_eigenvalues():
    """With a fixed, faithful encoder, the closed-form (ALS) predictor fit
    should recover the true system's eigenvalues almost exactly -- this is
    the necessary baseline before asking what a regularizer distorts."""
    system, obs_model, train_batch, _val = _setup()
    oracle_encoder = FixedLinearEncoder(obs_model.oracle_decoder_weight(system.n))
    cfg = TrainConfig(latent_dim=system.n, horizon=9, lambda_pred_1step=1.0, lambda_pred_ms=1.0, seed=0)
    _enc, pred, _dec, _hist = train_jepa(
        system, obs_model, train_batch, cfg, encoder=oracle_encoder, train_encoder=False, verbose=False
    )
    A_z, _B_z = pred.matrices()
    learned_eigs = sorted(np.linalg.eigvals(A_z).real)
    true_eigs = sorted(system.eigvals())
    assert np.allclose(learned_eigs, true_eigs, atol=0.05)


def test_unregularized_joint_training_runs_and_returns_finite_latent_system():
    system, obs_model, train_batch, _val = _setup()
    cfg = TrainConfig(
        latent_dim=system.n, horizon=9, outer_rounds=8, inner_epochs=3,
        lambda_pred_1step=1.0, lambda_pred_ms=1.0, seed=0,
    )
    enc, pred, dec, hist = train_jepa(system, obs_model, train_batch, cfg, verbose=False)
    A_z, B_z = pred.matrices()
    assert np.isfinite(A_z).all()
    assert np.isfinite(B_z).all()
    assert dec is None
    assert len(hist) == 8 * 3


def test_strong_sigreg_degrades_unstable_mode_retention_relative_to_action_recon():
    """Core claim of this codebase: at matched (tuned) regularizer strength,
    SIGReg -- which only constrains the marginal distribution of z -- can
    destroy the unstable mode's recoverability, while action-reconstruction
    -- which is tied to what the applied actions did -- preserves it. This
    runs at reduced scale for test speed; see experiments/ for the full
    (more robust, multi-seed) demonstration."""
    system, obs_model, train_batch, val_batch = _setup(seed=0)

    cfg_sigreg = TrainConfig(
        latent_dim=system.n, horizon=9, outer_rounds=20, inner_epochs=6, lr=1e-2,
        lambda_pred_1step=1.0, lambda_pred_ms=1.0, lambda_sigreg=150.0, seed=0,
    )
    enc_sigreg, pred_sigreg, _dec, _hist = train_jepa(system, obs_model, train_batch, cfg_sigreg, verbose=False)

    cfg_ar = TrainConfig(
        latent_dim=system.n, horizon=9, outer_rounds=20, inner_epochs=6, lr=1e-2,
        lambda_pred_1step=1.0, lambda_pred_ms=1.0,
        lambda_actrecon_1step=10.0, lambda_actrecon_ms=10.0, seed=0,
    )
    enc_ar, pred_ar, _dec, _hist = train_jepa(system, obs_model, train_batch, cfg_ar, verbose=False)

    r2_sigreg = unstable_mode_retention(system, enc_sigreg, train_batch, val_batch)
    r2_ar = unstable_mode_retention(system, enc_ar, train_batch, val_batch)
    assert r2_ar > 0.95
    assert r2_ar - r2_sigreg > 0.3

    ctrl_ar = design_latent_controller(pred_ar)
    ev_ar = evaluate_controller(
        system, obs_model, enc_ar, ctrl_ar["K_z"], n_trials=10, n_steps=80,
        x0_std=0.03, success_threshold=0.5, hold_steps=15, seed=555,
    )
    assert ev_ar["success_rate"] == 1.0

    ctrl_sigreg = design_latent_controller(pred_sigreg)
    ev_sigreg = evaluate_controller(
        system, obs_model, enc_sigreg, ctrl_sigreg["K_z"], n_trials=10, n_steps=80,
        x0_std=0.03, success_threshold=0.5, hold_steps=15, seed=555,
    )
    assert ev_sigreg["success_rate"] < ev_ar["success_rate"]
