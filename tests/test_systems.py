import numpy as np
import pytest

from jepa_lds.systems import make_double_mode_system, make_linearized_cartpole_system


@pytest.mark.parametrize("maker", [make_double_mode_system, make_linearized_cartpole_system])
def test_open_loop_unstable(maker):
    s = maker()
    assert s.is_open_loop_unstable()
    assert s.spectral_radius() > 1.0


@pytest.mark.parametrize("maker", [make_double_mode_system, make_linearized_cartpole_system])
def test_stabilizable(maker):
    s = maker()
    assert s.is_stabilizable()


def test_double_mode_fully_controllable():
    s = make_double_mode_system()
    assert s.is_controllable()
    assert s.pbh_uncontrollable_modes() == []


def test_dlqr_stabilizes_ground_truth():
    for maker in [make_double_mode_system, make_linearized_cartpole_system]:
        s = maker()
        K, _P = s.dlqr()
        rho_cl = s.closed_loop_spectral_radius(K)
        assert rho_cl < 1.0, f"{s.name}: LQR should stabilize the true system, got rho={rho_cl}"


def test_simulate_matches_step():
    s = make_double_mode_system()
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal(s.n)
    actions = rng.standard_normal((5, s.m))
    xs = s.simulate(x0, actions)
    assert xs.shape == (6, s.n)
    x_manual = x0
    for t in range(5):
        x_manual = s.step(x_manual, actions[t])
    assert np.allclose(xs[-1], x_manual)


def test_unstable_modal_coordinate_shapes():
    s = make_double_mode_system()
    x = np.random.default_rng(0).standard_normal((100, s.n))
    xi = s.unstable_modal_coordinate(x)
    assert xi.shape == (100,)
