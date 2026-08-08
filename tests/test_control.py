import numpy as np
import torch

from jepa_lds.control import (
    design_latent_controller,
    evaluate_controller,
    is_stabilizable,
    pbh_uncontrollable_modes,
)
from jepa_lds.data import make_observation_model
from jepa_lds.models import LinearLatentPredictor
from jepa_lds.systems import make_double_mode_system


def _predictor_from_matrices(A: np.ndarray, B: np.ndarray) -> LinearLatentPredictor:
    d, m = A.shape[0], B.shape[1]
    pred = LinearLatentPredictor(d, m)
    with torch.no_grad():
        pred.A.weight.copy_(torch.tensor(A, dtype=torch.float32))
        pred.B.weight.copy_(torch.tensor(B, dtype=torch.float32))
    return pred


def test_pbh_detects_uncontrollable_unstable_mode():
    # x1 unstable and driven by input; x2 unstable and NOT driven -> uncontrollable unstable mode
    A = np.diag([1.3, 1.2])
    B = np.array([[1.0], [0.0]])
    bad = pbh_uncontrollable_modes(A, B)
    assert len(bad) == 1
    assert not is_stabilizable(A, B)


def test_design_latent_controller_fails_gracefully_when_unstable_mode_uncontrollable():
    A = np.diag([1.3, 1.2])
    B = np.array([[1.0], [0.0]])
    pred = _predictor_from_matrices(A, B)
    result = design_latent_controller(pred)
    assert result["stabilizable"] is False
    assert result["K_z"] is None
    assert len(result["uncontrollable_unstable_modes"]) == 1


def test_design_latent_controller_succeeds_when_controllable():
    A = np.diag([1.3, 0.8])
    B = np.array([[1.0], [1.0]])
    pred = _predictor_from_matrices(A, B)
    result = design_latent_controller(pred)
    assert result["stabilizable"] is True
    assert result["K_z"] is not None
    assert result["latent_closed_loop_spectral_radius"] < 1.0


def test_closed_loop_evaluation_succeeds_with_faithful_encoder():
    """A faithful (identity-like) encoder + a predictor that matches the true
    dynamics should let LQR stabilize the true system in closed loop."""
    system = make_double_mode_system()
    obs_model = make_observation_model(system, seed=0, n_distractor=4, measurement_noise_std=0.0)
    from jepa_lds.models import FixedLinearEncoder

    W = obs_model.oracle_decoder_weight(system.n)
    encoder = FixedLinearEncoder(W)
    pred = _predictor_from_matrices(system.A, system.B)
    ctrl = design_latent_controller(pred)
    assert ctrl["stabilizable"]
    ev = evaluate_controller(
        system, obs_model, encoder, ctrl["K_z"],
        n_trials=10, n_steps=60, x0_std=0.05, success_threshold=1.0, hold_steps=10, seed=1,
    )
    assert ev["success_rate"] == 1.0


def test_evaluate_controller_handles_no_controller():
    system = make_double_mode_system()
    obs_model = make_observation_model(system, seed=0)
    from jepa_lds.models import LinearEncoder

    encoder = LinearEncoder(obs_model.p, system.n)
    ev = evaluate_controller(system, obs_model, encoder, None, n_trials=3, n_steps=10)
    assert ev["success_rate"] == 0.0
    assert ev["final_state_distance_avg"] == float("inf")


def test_evaluate_controller_rejects_hold_steps_longer_than_trajectory():
    """hold_steps > n_steps+1 can never be satisfied (there aren't that many
    steps to check), which would otherwise silently report success_rate=0
    regardless of how stable the trajectory actually was -- must raise
    instead of silently misleading."""
    system = make_double_mode_system()
    obs_model = make_observation_model(system, seed=0)
    from jepa_lds.models import LinearEncoder

    encoder = LinearEncoder(obs_model.p, system.n)
    K_z = np.zeros((system.m, system.n))
    import pytest

    with pytest.raises(ValueError, match="hold_steps"):
        evaluate_controller(system, obs_model, encoder, K_z, n_trials=2, n_steps=10, hold_steps=500)
