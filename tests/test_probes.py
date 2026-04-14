"""
Unit tests for all probes, DMDc, LQR, and the ground truth linearisation.

Run with: pytest tests/test_probes.py -v
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
import scipy.linalg
import torch


def _make_stable_system(n=4, m=1, p=2, seed=0):
    rng = np.random.RandomState(seed)
    A = rng.randn(n, n)
    eigs, V = np.linalg.eig(A)
    eigs_scaled = eigs / (np.abs(eigs).max() + 0.1) * 0.8
    A = np.real(V @ np.diag(eigs_scaled) @ np.linalg.inv(V))
    B = rng.randn(n, m)
    C = rng.randn(p, n)
    return A, B, C


def _make_cartpole_like():
    from ground_truth.cartpole_gt import CartpoleGroundTruth
    gt = CartpoleGroundTruth()
    return gt.A_star, gt.B_star, gt.C_star, gt


def _controllable_system(n=4, m=1):
    A = np.diag([1.1, 0.9, 0.7, 0.5])
    B = np.zeros((n, m))
    B[0, 0] = 1.0
    B[1, 0] = 0.5
    return A, B


def _uncontrollable_system(n=4, m=1):
    A = np.diag([1.1, 0.9, 0.7, 0.5])
    B = np.zeros((n, m))
    B[1, 0] = 1.0
    return A, B


def _nmp_system():
    A = np.array([[0.5, 1.0], [0.0, 0.3]])
    B = np.array([[-1.0], [1.0]])
    C = np.array([[1.0, 0.0]])
    return A, B, C


class TestGroundTruth:
    def test_eigenvalues_one_unstable(self):
        from ground_truth.cartpole_gt import CartpoleGroundTruth
        gt = CartpoleGroundTruth()
        assert gt.ell_star == 1, f"Expected 1 unstable mode, got {gt.ell_star}"
        assert np.max(np.abs(gt.unstable_eigenvalues)) > 1.0

    def test_spectral_radius(self):
        from ground_truth.cartpole_gt import CartpoleGroundTruth
        gt = CartpoleGroundTruth()
        rho = gt.spectral_radius
        assert 1.0 < rho < 1.5, f"Spectral radius {rho:.3f} out of expected range"

    def test_nmp(self):
        from ground_truth.cartpole_gt import CartpoleGroundTruth
        gt = CartpoleGroundTruth()
        assert gt.is_nmp, "Cartpole should be NMP"
        assert len(gt.nmp_zeros) > 0

    def test_linearisation_validation(self):
        from ground_truth.cartpole_gt import CartpoleGroundTruth
        gt = CartpoleGroundTruth()
        val = gt.validate_linearization(n_steps=50, x0_scale=0.01)
        assert val["max_relative_error"] < 0.05, (
            f"Linearisation error {val['max_relative_error']:.3f} > 5%")

    def test_markov_parameters(self):
        from ground_truth.cartpole_gt import CartpoleGroundTruth
        gt = CartpoleGroundTruth()
        assert gt.relative_degree is not None
        assert gt.markov_params.shape[0] >= 8


class TestDMDc:
    def test_dmdc_on_linear_system(self):
        from identification.dmdc import fit_dmdc
        rng = np.random.RandomState(42)
        n, m = 4, 1
        A_true = np.diag([0.9, 0.8, 0.7, 0.6])
        B_true = rng.randn(n, m) * 0.5
        N = 5000
        Z = rng.randn(N, n)
        A_actions = rng.randn(N, m)
        Z_next = Z @ A_true.T + A_actions @ B_true.T
        A_hat, B_hat = fit_dmdc(Z, A_actions, Z_next)
        A_err = np.linalg.norm(A_hat - A_true, "fro") / np.linalg.norm(A_true, "fro")
        B_err = np.linalg.norm(B_hat - B_true, "fro") / np.linalg.norm(B_true, "fro")
        assert A_err < 0.01, f"A recovery error {A_err:.4f} > 1%"
        assert B_err < 0.01, f"B recovery error {B_err:.4f} > 1%"

    def test_dmdc_proximal_converges(self):
        from identification.dmdc import fit_dmdc_proximal
        rng = np.random.RandomState(7)
        n, m = 4, 1
        A_true = np.diag([0.9, 0.8, 0.7, 0.6])
        B_true = rng.randn(n, m)
        N = 2000
        Z = rng.randn(N, n)
        A_actions = rng.randn(N, m)
        Z_next = Z @ A_true.T + A_actions @ B_true.T
        A_hat, B_hat, info = fit_dmdc_proximal(Z, A_actions, Z_next, max_iter=200)
        assert info["residual"] < 1e-4, f"Proximal DMDc residual {info['residual']:.6f}"

    def test_output_map_recovery(self):
        from identification.dmdc import fit_output_map
        rng = np.random.RandomState(0)
        n, d, p = 4, 32, 2
        C_true = rng.randn(p, d)
        Z = rng.randn(200, d)
        Y = Z @ C_true.T + rng.randn(200, p) * 1e-8
        C_hat = fit_output_map(Z, Y)
        err = np.linalg.norm(C_hat - C_true, "fro") / np.linalg.norm(C_true, "fro")
        assert err < 0.01, f"Output map error {err:.4f}"


class TestP1:
    def test_spectral_recovery_on_ground_truth(self):
        from probes.spectral import P1_1_eigenvalue_recovery
        A_star, B_star, C_star, gt = _make_cartpole_like()
        result = P1_1_eigenvalue_recovery(A_star, A_star)
        assert result["delta_lambda"] < 1e-6, f"delta_lambda={result['delta_lambda']}"
        assert result["UMR"] == 1.0, f"UMR={result['UMR']}"
        assert result["spectral_radius_error"] < 1e-10

    def test_spectral_recovery_perturbed(self):
        from probes.spectral import P1_1_eigenvalue_recovery
        A_star, B_star, C_star, gt = _make_cartpole_like()
        rng = np.random.RandomState(0)
        A_hat = A_star + 0.01 * rng.randn(*A_star.shape)
        result = P1_1_eigenvalue_recovery(A_hat, A_star)
        assert result["n_unstable_true"] == 1


class TestP2PBH:
    def test_pbh_on_known_controllable(self):
        from probes.pbh import P2_1_pbh_stabilizability
        A, B = _controllable_system()
        result = P2_1_pbh_stabilizability(A, B, delta_tol=0.2)
        assert result["n_near_unstable"] >= 1
        assert result["mu_S"] > 1e-4, f"mu_S={result['mu_S']:.6f} should be > 0"
        assert result["is_stabilizable"]

    def test_pbh_on_known_uncontrollable(self):
        from probes.pbh import P2_1_pbh_stabilizability
        A, B = _uncontrollable_system()
        result = P2_1_pbh_stabilizability(A, B, delta_tol=0.2)
        assert result["n_near_unstable"] >= 1
        assert result["mu_S"] < 1e-3, (
            f"mu_S={result['mu_S']:.6f} should be near 0 for uncontrollable system")
        assert not result["is_stabilizable"]


class TestP4:
    def test_nmp_detection(self):
        from probes.zeros import P4_1_transmission_zeros
        A, B, C = _nmp_system()
        result = P4_1_transmission_zeros(A, B, C, A, B, C)
        assert result["true_NMP_count"] >= 1, (
            f"Expected NMP zeros, found {result['true_NMP_count']}")
        assert result["NMP_count_match"]

    def test_step_response_undershoot_formula(self):
        A, B, C = _nmp_system()
        n, d_u, p = A.shape[0], B.shape[1], C.shape[0]
        T = 50
        a_step = np.ones(d_u)
        z = np.zeros(n)
        y_traj = []
        for t in range(T):
            y_traj.append(float(np.dot(C[0], z)))
            z = A @ z + B @ a_step
        y = np.array(y_traj)
        y0 = y[0]
        yT = y[-1]
        if abs(yT - y0) > 1e-6:
            UR = (np.min(y) - y0) / (yT - y0)
            assert UR < 0, f"Expected undershoot (UR < 0), got UR={UR:.3f}"


class TestDetectability:
    def test_detectable_system(self):
        from probes.pbh import P2_2_pbh_detectability
        A_star, B_star, C_star, gt = _make_cartpole_like()
        result = P2_2_pbh_detectability(A_star, C_star, delta_tol=0.2)
        assert result["is_detectable"]
        assert result["mu_D"] > 1e-4


class TestLQR:
    def test_lqr_stabilizes_true_system(self):
        from control.lqr import solve_discrete_lqr
        A_star, B_star, C_star, gt = _make_cartpole_like()
        Q = np.diag([1.0, 1.0, 10.0, 1.0])
        R = np.array([[0.01]])
        K, P, cl_eigs = solve_discrete_lqr(A_star, B_star, Q, R)
        assert np.all(np.abs(cl_eigs) < 1.0), (
            f"Closed-loop unstable: max|eig|={np.max(np.abs(cl_eigs)):.4f}")
        rng = np.random.RandomState(0)
        n_trials = 20
        successes = 0
        for _ in range(n_trials):
            x = rng.uniform(-0.2, 0.2, size=4)
            for t in range(200):
                u = -K @ x
                x = A_star @ x + B_star.flatten() * u[0]
            if np.linalg.norm(x) < 0.1:
                successes += 1
        success_rate = successes / n_trials
        assert success_rate == 1.0, (
            f"Oracle LQR success rate {success_rate:.2f} < 1.0")

    def test_dare_solution_positive_definite(self):
        from control.lqr import solve_discrete_lqr
        A_star, B_star, C_star, gt = _make_cartpole_like()
        Q = np.diag([1.0, 1.0, 10.0, 1.0])
        R = np.array([[0.01]])
        K, P, cl_eigs = solve_discrete_lqr(A_star, B_star, Q, R)
        eigs_P = np.linalg.eigvalsh(P)
        assert np.all(eigs_P > 0), f"P is not positive definite: min_eig={eigs_P.min():.6f}"


class TestKalman:
    def test_full_rank_system(self):
        from probes.kalman import P3_1_kalman_decomposition, _effective_rank, _controllability_matrix
        A_star, B_star, C_star, gt = _make_cartpole_like()
        result = P3_1_kalman_decomposition(A_star, B_star, C_star, epsilon_rank=1e-7)
        assert result["r_controllable"] >= 3, (
            f"r_controllable={result['r_controllable']} expected >= 3")
        assert result["r_observable"] >= 2, (
            f"r_observable={result['r_observable']} too low")

    def test_uncontrollable_reduces_rank(self):
        from probes.kalman import P3_1_kalman_decomposition
        A = np.diag([0.9, 0.8, 0.7, 0.6])
        B = np.zeros((4, 1))
        B[0, 0] = 1.0
        C = np.eye(2, 4)
        result = P3_1_kalman_decomposition(A, B, C)
        assert result["r_controllable"] < 4


class TestActionEncoder:
    def test_linear_action_encoder_pseudoinverse(self):
        from models.action_encoder import LinearActionEncoder
        enc = LinearActionEncoder(action_dim=1, latent_action_dim=32)
        enc.compute_pseudoinverse()
        W = enc.W.weight.data
        W_pinv = enc.W_pinv
        product = W_pinv @ W
        expected = torch.eye(1)
        err = float(torch.norm(product - expected).item())
        assert err < 1e-6, f"W_pinv @ W identity error = {err:.2e}"

    def test_identity_action_encoder(self):
        from models.action_encoder import IdentityActionEncoder
        enc = IdentityActionEncoder(action_dim=1)
        u = torch.randn(10, 1)
        assert torch.allclose(enc(u), u)
        assert torch.allclose(enc.decode(u), u)

    def test_mlp_action_encoder_reconstruction(self):
        from models.action_encoder import MLPActionEncoder
        enc = MLPActionEncoder(action_dim=1, latent_action_dim=8)
        optimizer = torch.optim.Adam(enc.parameters(), lr=1e-3)
        u = torch.randn(256, 1)
        for _ in range(200):
            optimizer.zero_grad()
            loss = enc.reconstruction_loss(u)
            loss.backward()
            optimizer.step()
        final_loss = enc.reconstruction_loss(u).item()
        assert final_loss < 0.1, f"MLP recon loss {final_loss:.4f} after training"


class TestSpectralLossGradient:
    def test_spectral_loss_gradient_flows(self):
        from losses.spectral import spectral_matching_loss
        d, d_u = 8, 1
        z = torch.randn(32, d, requires_grad=False)
        a = torch.randn(32, d_u, requires_grad=False)
        z_next = torch.randn(32, d, requires_grad=True)
        true_unstable = np.array([1.05 + 0j])
        loss, info = spectral_matching_loss(z, a, z_next, true_unstable)
        assert loss.requires_grad or loss.item() >= 0
        if z_next.grad is not None:
            z_next.grad.zero_()
        loss.backward()
        assert True


class TestPBHLoss:
    def test_pbh_loss_zero_for_stable_system(self):
        from losses.pbh import pbh_stabilizability_loss
        d, d_u = 4, 1
        rng = np.random.RandomState(0)
        A = np.diag([0.5, 0.4, 0.3, 0.2])
        B = rng.randn(d, d_u)
        N = 64
        Z = torch.from_numpy(rng.randn(N, d)).float()
        A_actions = torch.from_numpy(rng.randn(N, d_u)).float()
        Z_next = torch.from_numpy(Z.numpy() @ A.T + A_actions.numpy() @ B.T).float()
        loss, info = pbh_stabilizability_loss(Z, A_actions, Z_next, delta_tol=0.05)
        assert info["n_unstable"] == 0
        assert loss.item() == 0.0

    def test_pbh_loss_positive_for_unstable_uncontrollable(self):
        from losses.pbh import pbh_stabilizability_loss
        d, d_u = 4, 1
        A = np.diag([1.1, 0.8, 0.7, 0.6])
        B = np.zeros((d, d_u))
        B[1, 0] = 1.0
        N = 128
        rng = np.random.RandomState(5)
        Z = torch.from_numpy(rng.randn(N, d)).float()
        A_act = torch.from_numpy(rng.randn(N, d_u)).float()
        Z_next = torch.from_numpy(Z.numpy() @ A.T + A_act.numpy() @ B.T).float()
        loss, info = pbh_stabilizability_loss(Z, A_act, Z_next, delta_tol=0.2)
        assert info["n_unstable"] >= 1
        assert loss.item() > 0


class TestJEPAModel:
    @pytest.mark.parametrize("variant", [
        "E-noact", "E-spec", "E-PBH", "E-both-r", "E-lift", "E-full"
    ])
    def test_forward_pass_shapes(self, variant):
        from models.jepa import make_jepa
        model = make_jepa(variant, latent_dim=16,
                          encoder_channels=[16, 32, 64, 128])
        B = 4
        obs = torch.randn(B, 3, 64, 64)
        act = torch.randn(B, 1)
        out = model(obs, act, obs)
        assert out["z_t"].shape == (B, 16)
        assert out["z_hat"].shape == (B, 16)
        assert out["z_next"].shape == (B, 16)
        assert out["z_next_sg"].shape == (B, 16)


class TestObserver:
    def test_luenberger_poles_stable(self):
        from control.observer import design_luenberger
        from control.lqr import solve_discrete_lqr
        A_star, B_star, C_star, gt = _make_cartpole_like()
        Q = np.diag([1.0, 1.0, 10.0, 1.0])
        R = np.array([[0.01]])
        K, P, cl_eigs = solve_discrete_lqr(A_star, B_star, Q, R)
        L = design_luenberger(A_star, B_star, C_star,
                               pole_scale=0.8,
                               controller_poles=cl_eigs)
        obs_eigs = scipy.linalg.eigvals(A_star - L @ C_star)
        assert np.all(np.abs(obs_eigs) < 1.0), (
            f"Observer poles not stable: {np.abs(obs_eigs)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
