import numpy as np
import torch

from jepa_lds.control import design_latent_controller
from jepa_lds.data import generate_dataset, make_observation_model
from jepa_lds.diagnostics import (
    closed_loop_trajectory_panel,
    grid_to_states,
    lyapunov_decrease_panel,
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


def test_grid_to_states_places_grid_in_selected_dims_and_zeros_elsewhere():
    XX, YY = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
    states = grid_to_states(XX, YY, dims=(1, 3), n=4)
    assert states.shape == (9, 4)
    assert np.all(states[:, 0] == 0.0)
    assert np.all(states[:, 2] == 0.0)
    assert np.allclose(states[:, 1], XX.ravel())
    assert np.allclose(states[:, 3], YY.ravel())


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

