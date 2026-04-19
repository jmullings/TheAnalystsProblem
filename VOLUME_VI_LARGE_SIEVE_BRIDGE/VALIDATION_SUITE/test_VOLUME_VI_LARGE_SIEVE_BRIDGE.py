#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# test_VOLUME_VI_LARGE_SIEVE_BRIDGE.py
#
# VALIDATION SUITE FOR VOLUME VI: LARGE SIEVE BRIDGE
# Tests the implementation of the Montgomery-Vaughan large sieve bounds, 
# kernel-decay off-diagonal bounds, discrete-to-continuous transition error 
# control, and explicit constant tracking.

import sys
import os
import math
import numpy as np
import mpmath as mp
import pytest

# Inject the proof directory into sys.path
PROOF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'VOLUME_VI_LARGE_SIEVE_BRIDGE_PROOF'))
sys.path.insert(0, PROOF_DIR)

try:
    import VOLUME_VI_LARGE_SIEVE_BRIDGE as ls
except ImportError:
    # Fallback in case of naming variation
    import LARGE_SIEVE_BRIDGE as ls

mp.mp.dps = 80
TOL = 1e-12

class TestLargeSieveConstants:
    """
    Requirement 2 & 5: Explicit constant tracking and No Hidden Big-O.
    """

    def test_explicit_constants_are_finite(self):
        """All bound functions must return finite floats (no NaNs, no infs)."""
        cfg = ls.DirichletConfig(N=50, sigma=0.5, window_type="sharp")
        xi_values = [0.0, 1.0, 5.0]
        
        constants, comps = ls.validate_large_sieve_bounds(cfg, H=1.0, xi_values=xi_values)
        
        # Verify constants object fields are finite floats
        assert isinstance(constants.min_separation, float) and math.isfinite(constants.min_separation)
        assert isinstance(constants.sum_abs_sq, float) and math.isfinite(constants.sum_abs_sq)
        assert isinstance(constants.MV_constant, float) and math.isfinite(constants.MV_constant)
        assert isinstance(constants.MV_bound, float) and math.isfinite(constants.MV_bound)
        assert isinstance(constants.kernel_bound_constant, float) and math.isfinite(constants.kernel_bound_constant)
        assert isinstance(constants.kernel_bound, float) and math.isfinite(constants.kernel_bound)
        assert isinstance(constants.discrete_to_cont_error, float) and math.isfinite(constants.discrete_to_cont_error)
        
        # Verify bound comparison fields are finite floats
        for c in comps:
            assert isinstance(c.off_diag_exact, float) and math.isfinite(c.off_diag_exact)
            assert isinstance(c.ratio_off_to_MV, float) and math.isfinite(c.ratio_off_to_MV)
            assert isinstance(c.ratio_off_to_kernel, float) and math.isfinite(c.ratio_off_to_kernel)

    def test_min_separation_computation(self):
        """Ensure δ is correctly computed for log frequencies."""
        N = 10
        # γ_n = log n
        gammas = ls.log_frequencies(N)
        
        # For γ_n = log n, the minimum separation occurs at the largest n
        # δ = log(N) - log(N-1) = log(N/(N-1))
        expected_delta = math.log(10.0 / 9.0)
        computed_delta = ls.min_separation(gammas)
        
        assert abs(computed_delta - expected_delta) < 1e-14


class TestMontgomeryVaughanInequality:
    """
    Requirement 1: Montgomery-Vaughan large sieve bound implementation.
    """

    def test_montgomery_vaughan_inequality(self):
        """Verify |S(ξ)|^2 - sum|a_n|^2 ≤ MV_bound for several ξ."""
        cfg = ls.DirichletConfig(N=30, sigma=0.5, window_type="sharp")
        xi_values = np.linspace(-5.0, 5.0, 20)
        
        constants, comps = ls.validate_large_sieve_bounds(cfg, H=1.0, xi_values=xi_values)
        
        for c in comps:
            # The MV inequality states:
            # |S(ξ)|^2 <= (N + 1/δ) sum |a_n|^2
            # Since |S(ξ)|^2 = diag + off_diag, and diag = sum |a_n|^2
            # |off_diag| <= |S(ξ)|^2 + diag <= MV_bound + diag
            # The function `compute_exact_off_diagonal` returns exactly | |S(ξ)|^2 - diag |
            
            # The ratio |off| / MV_bound should theoretically be <= 1.0 
            # (In practice, MV bounds the *total* sum, so |off| is strictly bounded by MV)
            assert c.ratio_off_to_MV <= 1.0 + TOL, f"MV bound violated at ξ={c.xi}: ratio={c.ratio_off_to_MV}"


class TestKernelDecayBound:
    """
    Requirement 1.3 & 3: Kernel-decay bound for off-diagonal interference.
    """

    def test_kernel_bound_dominates_off_diagonal(self):
        """
        Show that the kernel-decay bound strictly bounds the actual off-diagonal 
        computation for all ξ.
        """
        cfg = ls.DirichletConfig(N=40, sigma=0.5, window_type="sharp")
        xi_values = np.linspace(-2.0, 2.0, 10)
        
        constants, comps = ls.validate_large_sieve_bounds(cfg, H=1.0, xi_values=xi_values)
        
        for c in comps:
            # The ratio |off| / kernel_bound must be <= 1.0
            assert c.ratio_off_to_kernel <= 1.0 + TOL, \
                f"Kernel bound violated at ξ={c.xi}: ratio={c.ratio_off_to_kernel}"

    def test_kernel_bound_tighter_than_MV_for_large_H(self):
        """
        For a highly localized kernel (large H), the kernel bound should be 
        tighter (smaller) than the general MV bound.
        """
        cfg = ls.DirichletConfig(N=50, sigma=0.5, window_type="sharp")
        
        # Test with H=2.0 (tighter spectral localization)
        constants, _ = ls.validate_large_sieve_bounds(cfg, H=2.0, xi_values=[0.0])
        
        assert constants.kernel_bound < constants.MV_bound, \
            "Kernel bound failed to improve upon the loose MV bound"


class TestDiscreteToContinuousTransition:
    """
    Requirement 1.4: Discrete-to-continuous quadrature error bounds.
    """

    def test_discrete_to_continuous_error(self):
        """Compare discrete sum to integral and verify error bound."""
        # For a(n) = n^{-0.5}, |a_n|^2 = 1/n
        # S_N = sum_{n=1}^N 1/n  (Harmonic number)
        N = 100
        cfg = ls.DirichletConfig(N=N, sigma=0.5, window_type="sharp")
        raw_a, _ = ls.build_coefficients(cfg)
        
        I, error_bound = ls.discrete_to_continuous_quadrature(raw_a)
        
        # The sum of 1/n for N=100 is ~ 5.187
        assert abs(I - sum(1.0/n for n in range(1, N+1))) < 1e-12
        
        # The error bound is defined as: 0.5(a_1^2 + a_N^2) + sum |a_{n+1}^2 - a_n^2|
        # For 1/n, the sequence is monotonically decreasing, so the sum of absolute differences
        # telescopes to a_1^2 - a_N^2 = 1 - 1/N
        expected_error = 0.5 * (1.0 + 1.0/N) + (1.0 - 1.0/N)
        assert abs(error_bound - expected_error) < 1e-12


class TestWindowScalingEffects:
    """
    Requirement 4.3 & 5: Verify smooth windows improve large sieve constants.
    """

    def test_window_improves_large_sieve_constant(self):
        """
        Demonstrate that a smooth window reduces the sum|a_n|^2, thereby 
        lowering the overall MV bound compared to a sharp cutoff.
        """
        N = 50
        H = 1.0
        
        # Sharp window
        cfg_sharp = ls.DirichletConfig(N=N, sigma=0.5, window_type="sharp")
        const_sharp, _ = ls.validate_large_sieve_bounds(cfg_sharp, H, [0.0])
        
        # Smooth Gaussian window
        cfg_smooth = ls.DirichletConfig(N=N, sigma=0.5, window_type="gaussian", window_params={"alpha": 2.0})
        const_smooth, _ = ls.validate_large_sieve_bounds(cfg_smooth, H, [0.0])
        
        # The smooth window suppresses the higher frequency coefficients, 
        # resulting in a strictly smaller MV bound.
        assert const_smooth.MV_bound < const_sharp.MV_bound, \
            "Smoothing failed to reduce the MV Large Sieve bound"


if __name__ == '__main__':
    pytest.main([__file__, "-v"])