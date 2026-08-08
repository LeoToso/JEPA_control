import numpy as np

from jepa_lds.data import WindowDataset, generate_dataset, make_observation_model
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
