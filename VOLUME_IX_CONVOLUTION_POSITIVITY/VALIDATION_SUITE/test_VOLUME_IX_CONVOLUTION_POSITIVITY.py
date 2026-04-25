#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# test_VOLUME_IX_CONVOLUTION_POSITIVITY.py
#
# VALIDATION SUITE FOR VOLUME IX: CONVOLUTION POSITIVITY
# 100% Coverage TDD Suite for the Exact Positivity Framework.
# Evaluates algebraic properties, quadrature bounds, time-frequency 
# (discrete Plancherel) consistency, and the three new obligation handlers 
# (XIII, XIV, XV) implementing exact error accounting and TAP-HO operator bounds.

import sys
import os
import math
import numpy as np
import mpmath as mp
import pytest
from unittest import mock

# Inject the proof directory into sys.path
PROOF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'VOLUME_IX_CONVOLUTION_POSITIVITY_PROOF'))
sys.path.insert(0, PROOF_DIR)

try:
    import VOLUME_IX_CONVOLUTION_POSITIVITY as ix
except ImportError:
    # Fallback for naming variations
    import CONVOLUTION_POSITIVITY as ix

mp.mp.dps = 70
TOL = 1e-10

class TestKernelPropertiesAndLambdaStar:
    """
    Requirement 1.5, 2.1, 2.3: Kernel derivatives, negativity region, and lambda*.
    """

    @pytest.mark.parametrize("H", [0.5, 1.0, 2.0])
    def test_kernel_positivity_algebraic(self, H):
        """k_H(t) must be strictly positive for all t (algebraic identity)."""
        for t in np.linspace(-5.0, 5.0, 20):
            k_val = float(ix.k_H(t, H))
            assert k_val > 0.0, f"k_H({t}) is not positive: {k_val}"

    def test_negativity_region_bounds(self):
        """
        Verify the negativity region N_H where w_H''(t) < 0.
        u0 = arcosh(sqrt(1.5)) approx 0.65847.
        """
        H = 1.0
        region = ix.compute_negativity_region(H)
        
        expected_t0 = float(mp.acosh(mp.sqrt(mp.mpf("1.5"))))
        
        assert abs(region.t_max - expected_t0) < 1e-12
        assert abs(region.t_min - (-expected_t0)) < 1e-12
        assert abs(region.length - 2*expected_t0) < 1e-12
        
        # Check w_H'' is indeed negative inside, and positive outside
        assert float(ix.w_H_second_derivative(0.0, H)) < 0.0
        assert float(ix.w_H_second_derivative(region.t_max * 1.5, H)) > 0.0

    @pytest.mark.parametrize("H", [0.5, 1.0, 3.0])
    def test_lambda_star_computation(self, H):
        r"""
        Verify that \hat{k}_H(xi) / \hat{w}_H(xi) >= lambda*.
        The minimum ratio should be exactly 4/H^2.
        """
        computed_lam = ix.compute_lambda_star(H, xi_max=5.0, samples=100)
        expected_lam = 4.0 / (H**2)
        assert abs(computed_lam - expected_lam) < 1e-10

    def test_compute_lambda_star_error(self):
        """H must be positive."""
        with pytest.raises(ValueError):
            ix.compute_lambda_star(0.0)


class TestDirichletBoundsAndTails:
    """
    Requirement 2.2, 2.4: Truncation tail bounds and sup bounds on |D_N|^2.
    """

    def test_uniform_D_sq_bound(self):
        """Verify that sup_D_sq strictly bounds |D_N(t)|^2 on the interval."""
        cfg = ix.DirichletConfig(N=5, sigma=0.5, window_type="sharp")
        t_min, t_max = -5.0, 5.0
        
        sup_bound = ix.sup_D_sq(cfg, t_min, t_max, samples=500)
        
        # Check against random points in the interval
        rng = np.random.default_rng(42)
        random_ts = rng.uniform(t_min, t_max, 50)
        
        for t in random_ts:
            actual_val = ix.D_N_abs_sq_from_cfg(t, cfg)
            assert actual_val <= sup_bound + 1e-10, f"Sup bound violated at t={t}"

    def test_tail_bound_accuracy(self):
        """Truncation error bound should decrease exponentially with L."""
        H = 1.0
        bound_D_sq = 10.0 # Arbitrary positive bound
        
        tail_L2 = ix.tail_bound_convolution(H, L=2.0, bound_on_D_sq=bound_D_sq)
        tail_L5 = ix.tail_bound_convolution(H, L=5.0, bound_on_D_sq=bound_D_sq)
        tail_L10 = ix.tail_bound_convolution(H, L=10.0, bound_on_D_sq=bound_D_sq)
        
        assert tail_L5 < tail_L2
        assert tail_L10 < tail_L5
        assert tail_L10 < 1e-6 # Should be extremely small for L=10*H

    def test_tail_bound_invalid(self):
        """L <= 0 returns infinity."""
        assert math.isinf(ix.tail_bound_convolution(1.0, 0.0, 10.0))


class TestConvolutionPositivity:
    """
    Requirement 1.1, 1.4, 4.2, 4.3: Direct convolution, net positivity, 
    and curvature leakage bounding.
    """

    @pytest.mark.parametrize("T0", [0.0, 14.1347])
    def test_convolution_integral_positive(self, T0):
        """Direct numerical evaluation yields a strictly positive value."""
        cfg = ix.DirichletConfig(N=5, sigma=0.5, window_type="gaussian", window_params={"alpha": 2.0})
        H = 1.0
        L = 8.0
        
        val, tail_err = ix.convolution_integral(cfg, H, T0, L, tol=1e-10)
        
        # The integral of a strictly positive kernel against |D_N|^2 is positive
        assert val > 0.0
        # The lower bound of the interval must also be > 0
        assert val - tail_err > 0.0

    def test_convolution_integral_invalid(self):
        cfg = ix.DirichletConfig(N=5, sigma=0.5, window_type="sharp")
        with pytest.raises(ValueError):
            ix.convolution_integral(cfg, 1.0, 0.0, 0.0, 1e-10)

    def test_curvature_leakage_bound_holds(self):
        """
        Compare the conservative leakage bound against the actual integral 
        over the negativity region N_H.
        """
        cfg = ix.DirichletConfig(N=5, sigma=0.5, window_type="sharp")
        H = 1.0
        T0 = 0.0
        
        # 1. Get the theoretical bound
        leakage_bound = ix.curvature_leakage_bound(cfg, H, T0)
        
        # 2. Compute the actual integral over N_H: \int_{N_H} |w_H''(t)| |D_N(T0+t)|^2 dt
        region = ix.compute_negativity_region(H)
        
        def actual_leakage_integrand(t_mp):
            t = float(t_mp)
            wpp = float(ix.w_H_second_derivative(t, H))
            dsq = ix.D_N_abs_sq_from_cfg(T0 + t, cfg)
            return abs(wpp) * dsq
            
        actual_leakage_mp = mp.quad(actual_leakage_integrand, [region.t_min, region.t_max])
        actual_leakage = float(actual_leakage_mp)
        
        assert actual_leakage <= leakage_bound + 1e-10, "Leakage theoretical bound violated!"

    def test_net_positivity_verifier(self):
        """
        Test the end-to-end verify_net_positivity function. 
        It must show that the direct integral is positive within error bounds.
        """
        cfg = ix.DirichletConfig(N=5, sigma=0.5, window_type="sharp")
        H = 1.0
        T0 = 14.1347
        L = 8.0
        
        res = ix.verify_net_positivity(cfg, H, T0, L, tol=1e-10)
        
        # Direct integral is mathematically guaranteed to be positive because k_H(t) > 0
        assert res.convolution_value > 0.0
        assert res.direct_positive_within_error is True
        
        # Ensure the interval is correctly constructed
        assert res.interval_for_Q[0] == res.convolution_value - res.convolution_tail_error
        assert res.interval_for_Q[1] == res.convolution_value + res.convolution_tail_error


class TestTimeFreqConsistency:
    """
    Requirement 4.1: Time vs Frequency domain (discrete Plancherel) equality.
    """

    def test_time_freq_consistency(self):
        r"""
        Ensure Plancherel equality holds within quadrature tolerance:
        \int k_H(t)|D_N(T0+t)|^2 dt == \sum a_n a_m cos(T0 log n/m) \hat{k}_H
        """
        # Small N and smooth window to ensure rapid frequency decay
        cfg = ix.DirichletConfig(N=5, sigma=0.5, window_type="gaussian", window_params={"alpha": 3.0})
        H = 1.0
        T0 = 0.0
        
        # We need large enough L in frequency to capture the signal
        L_t = 8.0
        L_xi = 6.0 
        
        comp = ix.compare_time_freq_domains(cfg, H, T0, L_t=L_t, L_xi=L_xi, tol=1e-10)
        
        # The discrete sum is analytically exact, so the difference is purely
        # quadrature truncation error from the time domain integral.
        assert abs(comp["difference"]) < 1e-6, \
            f"Time/Freq domain mismatch! Time: {comp['Q_time']}, Freq: {comp['Q_freq']}, Diff: {comp['difference']}"

    def test_hat_w_H_analytic_large_arg(self):
        r"""Ensure the large-argument approximation branch is hit in \hat{w}_H."""
        H = 1.0
        xi = 20.0  # pi^2 * 20 * 1 = 197 > 50
        val = ix.hat_w_H_analytic(xi, H)
        assert float(val) > 0.0


class TestObligationsHandlers:
    """
    Tests the Obligation XIII, XIV, and XV verification layers.
    """

    def test_derive_xi_to_Q_H(self):
        """Obligation XIII: Verify the xi -> Q_H derivation numerical check."""
        cfg = ix.DirichletConfig(N=5, sigma=0.5, window_type="sharp")
        res = ix.derive_xi_to_Q_H(H=1.0, cfg=cfg, T0=0.0, dps=30)
        
        assert res.H == 1.0
        assert res.min_spectral_value > 0.0, "Spectral representation must be positive."
        assert res.derivation_verified is True

    def test_mean_value_with_remainder(self):
        """Obligation XIV: Verify the mean-value bound proxy."""
        cfg = ix.DirichletConfig(N=5, sigma=0.5, window_type="sharp")
        res = ix.mean_value_with_remainder(cfg=cfg, H=1.0, T=100.0, dps=30)
        
        assert res.mv_sum > 0.0
        assert res.remainder_rate > 0.0
        assert res.remainder_bound == res.remainder_rate / 100.0
        assert res.T_for_epsilon(1e-2) == res.remainder_rate / 1e-2
        
        with pytest.raises(ValueError):
            res.T_for_epsilon(-0.5)

    def test_build_K_N_raw_kernel(self):
        """Test the raw kernel matrix fallback surrogate."""
        K = ix._build_K_N_raw_kernel(H=1.0, N=5)
        assert K.shape == (5, 5)
        # Verify zero diagonal
        np.testing.assert_allclose(np.diag(K), np.zeros(5), atol=1e-12)
        # Verify symmetry
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    @mock.patch('VOLUME_IX_CONVOLUTION_POSITIVITY._build_K_N_from_kernel')
    def test_verify_operator_norm_bound(self, mock_build):
        """Obligation XV: Verify operator norm computation and model fitting."""
        # Mock returning a dummy operator
        def dummy_build(H, N):
            return np.eye(N) * 0.5, False, "Dummy"
        mock_build.side_effect = dummy_build
        
        # Test with N_max = 100
        res = ix.verify_operator_norm_bound(H=1.0, N_max=100, N_values=[10, 20, 50], power_iters=10)
        
        assert res.operator_description == "Dummy"
        assert res.kH0 == 6.0  # 6 / 1.0^2
        # np.eye has op norm 0.5
        np.testing.assert_allclose(res.op_norms, [0.5, 0.5, 0.5], atol=1e-6)
        
        # Margin = 6.0 - 0.5 = 5.5
        np.testing.assert_allclose(res.margins, [5.5, 5.5, 5.5], atol=1e-6)
        
        # Model eval for N=10
        pred = res.margin_model(10)
        assert math.isfinite(pred)

    def test_power_iteration_break_condition(self):
        """Test the power iteration early break when matrix is functionally zero."""
        K = np.zeros((5, 5))
        op_norm = ix._power_iteration_op_norm(K)
        assert op_norm == 0.0


class TestDiagnostics:
    """Test the integrated diagnostics output script."""

    @mock.patch('VOLUME_IX_CONVOLUTION_POSITIVITY.verify_operator_norm_bound')
    def test_demo_execution(self, mock_verify, capsys):
        """Execute the _demo script to ensure all print paths run cleanly."""
        
        # Mock out the heavy operator verification
        mock_verify.return_value = mock.MagicMock(
            operator_description="Mock Desc",
            using_tapho_operator=False,
            kH0=6.0,
            ks=[10],
            op_norms=[1.0],
            margins=[5.0],
            verified_up_to_N=10,
            min_margin=5.0,
            A_hat=1.0,
            B_hat=2.0,
            analytically_open=True
        )
        
        ix._demo()
        captured = capsys.readouterr().out
        assert "=== Volume IX Convolution Positivity Demo ===" in captured
        assert "Obligation XIII" in captured
        assert "Obligation XIV" in captured
        assert "Obligation XV" in captured
        assert "NOTE: Volume I TAP-HO module not found" in captured

if __name__ == '__main__':
    pytest.main([__file__, "-v"])