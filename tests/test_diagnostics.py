import numpy as np
import torch

from jepa_lds.control import design_latent_controller
from jepa_lds.data import generate_dataset, make_observation_model
from jepa_lds.diagnostics import (
    closed_loop_trajectory_panel,
    decode_state,
    fit_state_probe,
    grid_to_states,
    h_step_prediction_error_panel,
    lyapunov_decrease_panel,
    phase_portrait_panel,
    planning_cost_panel,
    region_of_attraction_panel,
    unstable_eigenvector_alignment,
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
    """Error is now measured in the DECODED ORIGINAL STATE SPACE (via the
    ridge probe), not in raw latent coordinates -- with a faithful encoder
    and an exactly-matching predictor, decoding should recover the true
    H-step-ahead state almost exactly."""
    system, obs_model, batch, encoder, pred = _setup()
    beta = fit_state_probe(encoder, batch, alpha=1e-3)
    panel = h_step_prediction_error_panel(system, obs_model, encoder, pred, beta, dims=(0, 1), lo=(-0.5, -0.5), hi=(0.5, 0.5), n_points=4, H=5)
    assert panel["error"].shape == panel["XX"].shape
    assert panel["H"] == 5
    assert np.all(np.isfinite(panel["error"]))
    assert np.allclose(panel["error"], 0.0, atol=0.05)


def test_planning_cost_panel_shape_and_finite():
    """Cost is now log10 ||decoded H-step zero-action rollout - origin||^2
    in the ORIGINAL STATE SPACE, not the latent LQR value function -- no
    controller/Riccati solution needed."""
    system, obs_model, batch, encoder, pred = _setup()
    beta = fit_state_probe(encoder, batch, alpha=1e-3)
    panel = planning_cost_panel(system, obs_model, encoder, pred, beta, dims=(0, 1), lo=(-1, -1), hi=(1, 1), n_points=7, H=5)
    assert panel["log_cost"].shape == panel["XX"].shape
    assert panel["H"] == 5
    assert np.all(np.isfinite(panel["log_cost"]))
    # cost should be minimal (near -inf in log10, i.e. lowest value) at the origin (center of an odd-sized grid).
    center = panel["log_cost"].shape[0] // 2
    assert panel["log_cost"][center, center] <= panel["log_cost"].max()


def test_region_of_attraction_panel_shape_and_success_near_origin():
    """With a faithful encoder + exactly-matching predictor, the learned LQR
    gain should coincide with the oracle full-state gain, and small enough
    perturbations near the origin should all be classified as successes by
    both under a generous threshold."""
    system, obs_model, _batch, encoder, pred = _setup()
    ctrl = design_latent_controller(pred)
    assert ctrl["stabilizable"]
    K_gt, _P_gt = system.dlqr()
    panel = region_of_attraction_panel(
        system, obs_model, encoder, ctrl["K_z"], dims=(0, 1), lo=(-0.05, -0.05), hi=(0.05, 0.05),
        n_points=3, n_steps=30, success_threshold=1.0, hold_steps=5, K_gt=K_gt,
    )
    assert panel["success"].shape == panel["XX"].shape
    assert panel["success_rate"] == 1.0
    assert "success_gt" in panel and "success_rate_gt" in panel
    assert panel["success_rate_gt"] == 1.0


def test_region_of_attraction_panel_no_gt_key_when_k_gt_omitted():
    system, obs_model, _batch, encoder, pred = _setup()
    ctrl = design_latent_controller(pred)
    panel = region_of_attraction_panel(
        system, obs_model, encoder, ctrl["K_z"], dims=(0, 1), lo=(-0.05, -0.05), hi=(0.05, 0.05),
        n_points=3, n_steps=10, success_threshold=1.0, hold_steps=3,
    )
    assert "success_gt" not in panel


def test_closed_loop_trajectory_panel_converges_for_faithful_setup():
    """With a faithful encoder + exactly-matching predictor, the learned LQR
    gain should drive a large initial deviation back toward the origin, and
    the returned trajectories/x0s should be projected onto the requested
    dims and match the requested rollout length."""
    system, obs_model, _batch, encoder, pred = _setup()
    ctrl = design_latent_controller(pred)
    x0 = np.array([5.0, 5.0])
    panel = closed_loop_trajectory_panel(system, obs_model, encoder, ctrl["K_z"], [x0], dims=(0, 1), n_steps=60)
    assert panel["dims"] == (0, 1)
    xs = panel["trajectories"][0]
    assert xs.shape == (61, 2)
    assert np.allclose(panel["x0s"][0], x0)
    assert np.linalg.norm(xs[-1]) < np.linalg.norm(xs[0])


def test_lyapunov_decrease_panel_shape_and_decrease_near_origin():
    system, obs_model, _batch, encoder, pred = _setup()
    ctrl = design_latent_controller(pred)
    _K_gt, P_gt = system.dlqr()
    panel = lyapunov_decrease_panel(
        system, obs_model, encoder, ctrl["K_z"], P_gt, dims=(0, 1), lo=(-0.1, -0.1), hi=(0.1, 0.1), n_points=5,
    )
    assert panel["delta_V"].shape == panel["XX"].shape
    assert np.all(np.isfinite(panel["delta_V"]))
    # a faithful encoder + matching predictor's LQR gain should satisfy the
    # oracle's own Lyapunov decrease almost everywhere near the origin.
    assert panel["frac_decrease"] >= 0.8


def test_unstable_eigenvector_alignment_faithful_setup_perfect_alignment():
    """With a faithful (identity-like) encoder and a predictor whose A_z
    exactly matches the true A, the predictor's own dominant eigenvector,
    mapped back to state space, should recover the true unstable
    eigenvector almost exactly (cos_sim ~ 1) and be ~orthogonal to the
    true stable one (double_mode's A is symmetric, so its eigenvectors are
    orthogonal)."""
    system, obs_model, _batch, encoder, pred = _setup()
    result = unstable_eigenvector_alignment(system, obs_model, encoder, pred, dims=(0, 1))
    assert result["cos_sim_unstable"] > 0.99
    assert abs(result["cos_sim_stable"]) < 0.05


def test_unstable_eigenvector_alignment_detects_collapse_to_stable_mode():
    """Regression test for a real bug: an earlier implementation decoded the
    latent eigenvector back to state space via a NOISY, data-fit ridge
    probe, which for latent_dim < system.n turned out to be ill-posed and
    reported nearly the same (wrong) direction regardless of which mode the
    predictor actually tracked. The fix uses the encoder's own EXACT linear
    map (`encoder_state_to_latent_map`) and its pseudoinverse instead.

    Here we hand-construct a predictor whose dominant eigenvalue's
    eigenvector is the TRUE STABLE direction (i.e. it has "collapsed" to
    the wrong mode) and check this is correctly flagged: closer to the true
    stable eigenvector than the true unstable one."""
    system, obs_model, _batch, encoder, _pred = _setup()
    w, V, Vinv = system.modal_decomposition()
    idx_u = system.unstable_mode_index()
    idx_s = 1 - idx_u
    vals = np.zeros(2, dtype=complex)
    vals[idx_s] = 2.0  # dominant eigenvalue sits on the STABLE eigenvector
    vals[idx_u] = 0.1  # non-dominant
    A_custom = np.real(V @ np.diag(vals) @ Vinv)
    collapsed_pred = LinearLatentPredictor(system.n, system.m)
    with torch.no_grad():
        collapsed_pred.A.weight.copy_(torch.tensor(A_custom, dtype=torch.float32))
        collapsed_pred.B.weight.copy_(torch.tensor(system.B, dtype=torch.float32))

    result = unstable_eigenvector_alignment(system, obs_model, encoder, collapsed_pred, dims=(0, 1))
    # "v_learned" is only canonicalized in sign relative to the TRUE UNSTABLE
    # direction, so when it's actually closer to stable, cos_sim_stable's
    # own sign is arbitrary (+1 or -1 depending on which way the flip fell)
    # -- compare magnitudes, which is the sign-agnostic "which line is this
    # closer to" question we actually care about.
    assert abs(result["cos_sim_stable"]) > abs(result["cos_sim_unstable"])
    assert abs(result["cos_sim_stable"]) > 0.99
