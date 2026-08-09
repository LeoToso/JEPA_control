import numpy as np

from jepa_lds.control import design_latent_controller, evaluate_controller
from jepa_lds.data import make_observation_model, generate_dataset, make_modal_contaminated_x0_sampler
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


def test_undercomplete_latent_ties_pred_loss_and_sigreg_reliably_collapses_unstable_mode():
    """The 'killer' construction (see experiments/run_example3_killer_collapse.py):
    with latent_dim=1 (deliberately SMALLER than system.n=2) and a noiseless
    double_mode system, the ONLY 1-D encodings achieving exactly zero
    multistep prediction loss (for any horizon) are the two eigenspaces
    themselves -- L_pred is exactly tied at 0 between "keep only unstable"
    and "keep only stable". Biasing the initial condition so the STABLE
    modal coordinate has much larger variance than the unstable one then
    reliably (across random encoder inits) tips SIGReg-regularized training
    into the unstable-collapsing solution, while action reconstruction
    reliably keeps the unstable mode instead (its response to actions
    amplifies over the window rather than decaying)."""
    system = make_double_mode_system()
    obs_model = make_observation_model(system, obs_dim_signal=6, n_distractor=0, measurement_noise_std=0.0, seed=0)
    x0_sampler = make_modal_contaminated_x0_sampler(
        system, sigma_stable=0.3, sigma_unstable_small=0.01, sigma_unstable_large=0.01, contamination_prob=0.0,
    )
    T = 5
    train_batch = generate_dataset(
        system, obs_model, 300, T, seed=0, action_std=0.05, process_noise_std=0.0,
        state_clip=50.0, x0_sampler=x0_sampler, mixture={"passive": 0.3, "random": 0.7},
    )
    val_batch = generate_dataset(
        system, obs_model, 80, T, seed=10_000, action_std=0.05, process_noise_std=0.0,
        state_clip=50.0, x0_sampler=x0_sampler, mixture={"passive": 0.3, "random": 0.7},
    )

    n_seeds = 5
    n_sigreg_collapsed = 0
    n_actrecon_collapsed = 0
    for seed in range(n_seeds):
        cfg_sigreg = TrainConfig(
            latent_dim=1, horizon=4, outer_rounds=40, inner_epochs=6, batch_size=4096, lr=1e-2,
            lambda_pred_1step=0.0, lambda_pred_ms=1.0, lambda_sigreg=25.0, seed=seed,
        )
        enc_sigreg, _pred, _dec, _hist = train_jepa(system, obs_model, train_batch, cfg_sigreg, verbose=False)
        r2_sigreg = unstable_mode_retention(system, enc_sigreg, train_batch, val_batch)
        n_sigreg_collapsed += int(r2_sigreg < 0.5)

        cfg_ar = TrainConfig(
            latent_dim=1, horizon=4, outer_rounds=40, inner_epochs=6, batch_size=4096, lr=1e-2,
            lambda_pred_1step=0.0, lambda_pred_ms=1.0, lambda_actrecon_ms=25.0, seed=seed,
        )
        enc_ar, _pred, _dec, _hist = train_jepa(system, obs_model, train_batch, cfg_ar, verbose=False)
        r2_ar = unstable_mode_retention(system, enc_ar, train_batch, val_batch)
        n_actrecon_collapsed += int(r2_ar < 0.5)

    # empirically (see experiments/run_example3_killer_collapse.py) this is 15/15 for sigreg and
    # 0/15 for actrecon at n_seeds=15; use a conservative margin here to avoid test flakiness.
    assert n_sigreg_collapsed >= 3
    assert n_actrecon_collapsed == 0
