#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# test_VOLUME_IX_CONVOLUTION_POSITIVITY.py
#
# VALIDATION SUITE FOR VOLUME IX: CONVOLUTION POSITIVITY
# Tests the exact positive floor, curvature leakage bounds, tail truncation
# error bounds, net positivity verification, and Time vs Frequency 
# (Parseval/Plancherel) consistency.

import sys
import os
import math
import numpy as np
import mpmath as mp
import pytest

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
        """
        Verify that hat{k}_H(xi) / hat{w}_H(xi) >= lambda*.
        The minimum ratio should be exactly 4/H^2.
        """
        computed_lam = ix.compute_lambda_star(H, xi_max=5.0, samples=100)
        expected_lam = 4.0 / (H**2)
        
        # Since hat{k}_H(xi) / hat{w}_H(xi) = (2pi xi)^2 + 4/H^2,
        # the minimum is exactly 4/H^2 at xi = 0.
        assert abs(computed_lam - expected_lam) < 1e-10


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
    Requirement 4.1: Time vs Frequency domain (Plancherel/Parseval) equality.
    """

    def test_time_freq_consistency(self):
        """
        Ensure Plancherel equality holds within quadrature tolerance:
        \int k_H(t)|D_N(T0+t)|^2 dt == \int \hat{k}_H(xi) |S(xi)e^{-i xi T0}|^2 dxi
        """
        # Small N and smooth window to ensure rapid frequency decay
        cfg = ix.DirichletConfig(N=5, sigma=0.5, window_type="gaussian", window_params={"alpha": 3.0})
        H = 1.0
        T0 = 0.0
        
        # We need large enough L in frequency to capture the signal
        L_t = 8.0
        L_xi = 6.0 
        
        comp = ix.compare_time_freq_domains(cfg, H, T0, L_t=L_t, L_xi=L_xi, tol=1e-10)
        
        # The difference should be driven entirely by truncation tail errors
        assert abs(comp["difference"]) < 1e-6, \
            f"Time/Freq domain mismatch! Time: {comp['Q_time']}, Freq: {comp['Q_freq']}, Diff: {comp['difference']}"

if __name__ == '__main__':
    pytest.main([__file__, "-v"])