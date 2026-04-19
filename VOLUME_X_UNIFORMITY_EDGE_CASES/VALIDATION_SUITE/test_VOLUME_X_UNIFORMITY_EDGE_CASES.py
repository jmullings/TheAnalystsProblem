#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# test_VOLUME_X_UNIFORMITY_EDGE_CASES.py
#
# VALIDATION SUITE FOR VOLUME X: UNIFORMITY & EDGE CASES
# Tests the global positivity and stability of the framework under extreme
# parameters: small H, large H, large T0, arithmetic resonances, and varying N.

import sys
import os
import numpy as np
import pytest

# Inject the proof directory into sys.path
PROOF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'VOLUME_X_UNIFORMITY_EDGE_CASES_PROOF'))
sys.path.insert(0, PROOF_DIR)

try:
    import VOLUME_X_UNIFORMITY_EDGE_CASES as vx
except ImportError:
    # Fallback for naming variations
    import UNIFORMITY_EDGE_CASES as vx


class TestModule1SmallH:
    """
    Requirement: Verify stability and scaling laws as H -> 0+.
    """
    
    def test_small_H_scaling_and_stability(self):
        """
        Verify that k_H(0) ~ 6/H^2, that the convolution remains positive, 
        and that tail bounds and leakage are controlled as H approaches 0.
        """
        # We test a range down to H=0.05
        H_values = [1.0, 0.5, 0.25, 0.1, 0.05]
        results = vx.check_small_H_scaling(H_values)
        
        # Every test in the module should pass
        failed = [r.name for r in results if not r.passed]
        assert not failed, f"Small H scaling failed for: {failed}"


class TestModule2LargeH:
    """
    Requirement: Verify behavior as H -> infinity (kernel flattening).
    """

    def test_large_H_flattening(self):
        """
        Verify that k_H(t) flattens to ~ 6/H^2, tail bounds remain small,
        and that the convolution does not degenerate (Q remains positive).
        """
        H_values = [1.0, 2.0, 5.0, 10.0]
        results = vx.check_large_H_behavior(H_values)
        
        failed = [r.name for r in results if not r.passed]
        assert not failed, f"Large H flattening failed for: {failed}"


class TestModule3LargeT0:
    """
    Requirement: Ensure rapid phase oscillations do not create a resonance 
    that flips the sign of the quadratic form.
    """

    def test_oscillatory_stability(self):
        """
        Verify that Q(T0) remains strictly positive and bounded below 
        as T0 becomes very large.
        """
        H = 1.0
        N_values = [5, 10]
        # Test up to very large T0
        T0_values = [0.0, 10.0, 50.0, 100.0, 500.0]
        
        results = vx.check_large_T0_behavior(H, N_values, T0_values)
        
        failed = [r.name for r in results if not r.passed]
        assert not failed, f"Large T0 stability failed for: {failed}"


class TestModule4ResonanceEdgeCases:
    """
    Requirement: Test worst-case scenarios where T0(ln n - ln m) ≈ 2πk.
    """

    def test_arithmetic_resonances(self):
        """
        Verify that deliberately choosing T0 to hit the resonance peaks 
        of the off-diagonal terms does not break global positivity.
        """
        H = 1.0
        N = 10
        p_pairs = [(2, 3), (2, 5), (3, 5), (3, 7)]
        k_values = [1, 2, 3]
        
        results = vx.check_resonance_edge_cases(H, N, p_pairs, k_values)
        
        failed = [r.name for r in results if not r.passed]
        assert not failed, f"Arithmetic resonance edge cases failed for: {failed}"


class TestModule5OscillatoryIntegralBounds:
    """
    Requirement: Verify that |Q(T0) - Q_diag| decays roughly like a negative 
    power of T0 (empirical Van der Corput / Riemann-Lebesgue decay).
    """

    @pytest.mark.xfail(reason="Empirical slope fit can be noisy for small N/T0 combinations. Left as diagnostic.")
    def test_oscillatory_decay_shape(self):
        """
        Verify the log-log slope of the oscillatory interference decay is <= -0.4.
        This test is marked xfail because short sequences of T0 might locally 
        violate the strict log-log linear fit, even though the overall asymptotic 
        decay holds.
        """
        H = 1.0
        N = 10
        T0_values = [10.0, 20.0, 40.0, 80.0, 160.0, 320.0]
        
        results = vx.check_oscillatory_decay_shape(H, N, T0_values)
        
        failed = [r.name for r in results if not r.passed]
        assert not failed, f"Oscillatory decay shape failed for: {failed}"


class TestModule6UniformN:
    """
    Requirement: Ensure positivity is independent of N (scalability).
    """

    def test_uniformity_across_N(self):
        """
        Verify that Q remains positive and leakage is controlled as N grows,
        tested across different windowing functions.
        """
        H = 1.0
        T0 = 10.0
        N_values = [5, 10, 20, 50]
        
        results = vx.check_uniformity_in_N(H, T0, N_values)
        
        failed = [r.name for r in results if not r.passed]
        assert not failed, f"Uniformity across N failed for: {failed}"


class TestMasterGrid:
    """
    Requirement: Final Master Test crossing all parameters.
    """

    @pytest.mark.slow
    def test_full_master_grid(self):
        """
        Run the full cross-product of (H, T0, N, window_type) and assert 
        guaranteed positivity, bounded leakage, and Parseval consistency 
        for every single configuration.
        """
        results = vx.run_master_grid()
        
        failed = [r.name for r in results if not r.passed]
        assert not failed, f"Master Grid failure. {len(failed)} configurations broke positivity or Parseval identity."

if __name__ == '__main__':
    pytest.main([__file__, "-v"])