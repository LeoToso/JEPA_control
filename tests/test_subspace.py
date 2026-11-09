import numpy as np

from jepa_lds.subspace import nonstable_subspace_report
from jepa_lds.systems import make_linearized_cartpole_system


def test_cartpole_retains_generalized_marginal_direction_and_drops_only_stable():
    A = make_linearized_cartpole_system().A
    eigenvalues, eigenvectors = np.linalg.eig(A)
    unstable = eigenvectors[:, np.argmax(np.abs(eigenvalues))].real
    # Unlike eigenvectors of the defective lambda=1 block, e_pos/e_vel
    # form a full basis of its two-dimensional generalized eigenspace.
    U, _ = np.linalg.qr(
        np.column_stack((np.eye(4)[:, :2], unstable))
    )
    report = nonstable_subspace_report(A, U.T)
    assert report["nonstable_dim"] == 3
    assert report["rank_F_U"] == 3
    assert report["injective_at_tolerance"]
    np.testing.assert_allclose(report["singular_values_F_U"], 1, atol=1e-10)
    assert report["invariant_subspace_residual"] < 1e-10


def test_missing_generalized_direction_is_not_mistaken_for_full_preservation():
    A = np.array([[1, 1, 0, 0], [0, 1, 0, 0],
                  [0, 0, 1.2, 0], [0, 0, 0, 0.8]], dtype=float)
    # Retain the marginal eigenvector and unstable mode, but lose the
    # generalized marginal direction. Latent dimension is still three.
    F = np.eye(4)[[0, 2, 3]]
    report = nonstable_subspace_report(A, F)
    assert report["nonstable_dim"] == 3
    assert report["rank_F_U"] == 2
    assert not report["injective_at_tolerance"]


def test_small_latent_dimension_reports_zero_minimum_gain():
    report = nonstable_subspace_report(np.diag([1.2, 1.1]), [[1, 1]])
    assert report["nonstable_dim"] == 2
    assert report["rank_F_U"] == 1
    assert report["minimum_gain"] == 0


def test_weak_direction_is_visible_in_tolerance_sensitivity():
    report = nonstable_subspace_report(
        np.diag([1.2, 0.8]), np.diag([1e-7, 1.0])
    )
    assert report["rank_F_U"] == 0
    assert report["rank_sensitivity"]["1e-08"] == 1
    np.testing.assert_allclose(report["minimum_gain"], 1e-7)


def test_complex_pair_and_stable_only_system():
    A = np.array([[0, -1.1, 0], [1.1, 0, 0], [0, 0, 0.8]])
    report = nonstable_subspace_report(A, np.eye(3)[:2])
    assert report["nonstable_dim"] == 2
    assert report["rank_F_U"] == 2
    stable = nonstable_subspace_report(np.diag([0.8, 0.9]), [[1, 0]])
    assert stable["nonstable_dim"] == 0
    assert stable["injective_at_tolerance"]
    assert stable["minimum_gain"] is None
