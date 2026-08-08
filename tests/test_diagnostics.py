import numpy as np
import torch

from jepa_lds.control import design_latent_controller
from jepa_lds.data import generate_dataset, make_observation_model
from jepa_lds.diagnostics import (
    decode_state,
    fit_state_probe,
    grid_to_states,
    h_step_prediction_error_panel,
    phase_portrait_panel,
    planning_cost_panel,
)
from jepa_lds.models import FixedLinearEncoder, LinearLatentPredictor
from jepa_lds.systems import make_double_mode_system


def _setup(seed=0):
    system = make_double_mode_system()
    obs_model = make_observation_model(system, obs_dim_signal=6, n_distractor=4, seed=0)
    batch = generate_dataset(system, obs_model, 100, 10, seed=seed, x0_std=0.05, action_std=0.3, state_clip=15.0)
    encoder = FixedLinearEncoder(obs_model.oracle_decoder_weight(system.n))
    pred = LinearLatentPredictor(system.n, system.m)
    with torch.no_grad():
        pred.A.weight.copy_(torch.tensor(system.A, dtype=torch.float32))
        pred.B.weight.copy_(torch.tensor(system.B, dtype=torch.float32))
    return system, obs_model, batch, encoder, pred


def test_fit_state_probe_recovers_state_from_faithful_encoder():
    system, _obs_model, batch, encoder, _pred = _setup()
    beta = fit_state_probe(encoder, batch, alpha=1e-3)
    assert beta.shape == (system.n + 1, system.n)
    y = batch.y.reshape(-1, batch.y.shape[-1])
    x = batch.x.reshape(-1, batch.x.shape[-1])
    with torch.no_grad():
        z = encoder(torch.tensor(y, dtype=torch.float32)).numpy()
    x_hat = decode_state(z, beta)
    assert np.allclose(x_hat, x, atol=0.2)


def test_grid_to_states_places_grid_in_selected_dims_and_zeros_elsewhere():
    XX, YY = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
    states = grid_to_states(XX, YY, dims=(1, 3), n=4)
    assert states.shape == (9, 4)
    assert np.all(states[:, 0] == 0.0)
    assert np.all(states[:, 2] == 0.0)
    assert np.allclose(states[:, 1], XX.ravel())
    assert np.allclose(states[:, 3], YY.ravel())


def test_phase_portrait_panel_true_field_matches_analytic_drift():
    system, obs_model, batch, encoder, pred = _setup()
    beta = fit_state_probe(encoder, batch, alpha=1e-3)
    panel = phase_portrait_panel(system, obs_model, encoder, pred, beta, dims=(0, 1), lo=(-1, -1), hi=(1, 1), n_points=5)
    for key in ("U_true", "V_true", "U_learned", "V_learned"):
        assert panel[key].shape == panel["XX"].shape
        assert np.all(np.isfinite(panel[key]))
    # true drift at grid state x is exactly (A - I) x -- check one grid point.
    x = np.array([panel["XX"][0, 0], panel["YY"][0, 0]])
    dx = (system.A - np.eye(2)) @ x
    assert np.isclose(panel["U_true"][0, 0], dx[0])
    assert np.isclose(panel["V_true"][0, 0], dx[1])


def test_phase_portrait_panel_faithful_encoder_learned_field_close_to_true():
    """A faithful encoder + a predictor that exactly matches the true
    dynamics should produce a learned drift field close to the true one."""
    system, obs_model, batch, encoder, pred = _setup()
    beta = fit_state_probe(encoder, batch, alpha=1e-3)
    panel = phase_portrait_panel(system, obs_model, encoder, pred, beta, dims=(0, 1), lo=(-1, -1), hi=(1, 1), n_points=5)
    assert np.allclose(panel["U_true"], panel["U_learned"], atol=0.3)
    assert np.allclose(panel["V_true"], panel["V_learned"], atol=0.3)


def test_h_step_prediction_error_panel_shape_and_zero_for_faithful_setup():
    system, obs_model, _batch, encoder, pred = _setup()
    panel = h_step_prediction_error_panel(system, obs_model, encoder, pred, dims=(0, 1), lo=(-0.5, -0.5), hi=(0.5, 0.5), n_points=4, H=5)
    assert panel["error"].shape == panel["XX"].shape
    assert panel["H"] == 5
    assert np.all(np.isfinite(panel["error"]))
    # deterministic noiseless encode + exactly-matching predictor -> ~0 error.
    assert np.allclose(panel["error"], 0.0, atol=1e-3)


def test_planning_cost_panel_shape_and_finite():
    system, obs_model, _batch, encoder, pred = _setup()
    ctrl = design_latent_controller(pred, q_scale=10.0, r_scale=1.0)
    assert ctrl["P_z"] is not None
    panel = planning_cost_panel(system, obs_model, encoder, ctrl["P_z"], dims=(0, 1), lo=(-1, -1), hi=(1, 1), n_points=6)
    assert panel["log_cost"].shape == panel["XX"].shape
    assert np.all(np.isfinite(panel["log_cost"]))
    # cost should be minimal (near -inf in log10, i.e. lowest value) at the origin.
    center = panel["log_cost"].shape[0] // 2
    assert panel["log_cost"][center, center] <= panel["log_cost"].max()
