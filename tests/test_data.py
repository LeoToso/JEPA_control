import numpy as np

from jepa_lds.data import WindowDataset, generate_dataset, make_modal_contaminated_x0_sampler, make_observation_model
from jepa_lds.systems import make_double_mode_system


def test_generate_dataset_shapes_and_finite():
    s = make_double_mode_system()
    obs = make_observation_model(s, seed=1)
    batch = generate_dataset(s, obs, n_episodes=15, T=10, seed=0, x0_std=0.03, action_std=0.3)
    assert batch.y.shape == (15, 11, obs.p)
    assert batch.x.shape == (15, 11, s.n)
    assert batch.a.shape == (15, 10, s.m)
    assert np.isfinite(batch.y).all()
    assert np.isfinite(batch.x).all()


def test_passive_policy_is_zero_action():
    s = make_double_mode_system()
    obs = make_observation_model(s, seed=1)
    batch = generate_dataset(s, obs, n_episodes=50, T=10, seed=0, mixture={"passive": 1.0})
    assert np.allclose(batch.a, 0.0)


def test_window_dataset_length():
    s = make_double_mode_system()
    obs = make_observation_model(s, seed=1)
    batch = generate_dataset(s, obs, n_episodes=10, T=10, seed=0)
    horizon = 4
    ds = WindowDataset(batch, horizon)
    expected = 10 * (10 - horizon + 1)
    assert len(ds) == expected
    y_win, a_win = ds[0]
    assert y_win.shape == (horizon + 1, obs.p)
    assert a_win.shape == (horizon, s.m)


def test_oracle_decoder_recovers_state_without_noise():
    s = make_double_mode_system()
    obs = make_observation_model(s, seed=1, measurement_noise_std=0.0, n_distractor=4)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(s.n)
    y = obs.observe(x, rng)
    W = obs.oracle_decoder_weight(s.n)
    x_hat = W @ y
    assert np.allclose(x, x_hat, atol=1e-8)


def test_modal_contaminated_x0_sampler_shape_and_kurtosis_asymmetry():
    """The unstable modal coordinate should be markedly heavier-tailed
    (higher excess kurtosis) than the stable one when the unstable draw is a
    contaminated Gaussian mixture and the stable draw is a plain Gaussian."""
    s = make_double_mode_system()
    sampler = make_modal_contaminated_x0_sampler(
        s, sigma_stable=0.03, sigma_unstable_small=0.01, sigma_unstable_large=3.0, contamination_prob=0.05,
    )
    rng = np.random.default_rng(0)
    X0 = np.array([sampler(rng) for _ in range(5000)])
    assert X0.shape == (5000, s.n)
    assert np.all(np.isfinite(X0))

    _w, _V, Vinv = s.modal_decomposition()
    idx_u = s.unstable_mode_index()
    idx_s = 1 - idx_u
    xi_u = X0 @ Vinv[idx_u].real
    xi_s = X0 @ Vinv[idx_s].real

    def excess_kurtosis(v):
        v = v - v.mean()
        return np.mean(v**4) / (np.var(v) ** 2 + 1e-12) - 3.0

    assert excess_kurtosis(xi_u) > 10.0
    assert abs(excess_kurtosis(xi_s)) < 1.0


def test_modal_contaminated_x0_sampler_rejects_non_2d_system():
    from jepa_lds.systems import make_linearized_cartpole_system

    s = make_linearized_cartpole_system()
    import pytest

    with pytest.raises(ValueError, match="2-state"):
        make_modal_contaminated_x0_sampler(s)


def test_generate_dataset_accepts_custom_x0_sampler():
    s = make_double_mode_system()
    obs = make_observation_model(s, seed=1)
    sampler = make_modal_contaminated_x0_sampler(s)
    batch = generate_dataset(s, obs, n_episodes=20, T=5, seed=0, x0_sampler=sampler)
    assert batch.x.shape == (20, 6, s.n)
    assert np.isfinite(batch.x).all()
