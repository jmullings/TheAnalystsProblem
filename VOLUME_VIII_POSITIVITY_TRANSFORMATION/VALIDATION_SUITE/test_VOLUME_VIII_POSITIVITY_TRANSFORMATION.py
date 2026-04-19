#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# test_VOLUME_VIII_POSITIVITY_TRANSFORMATION.py
#
# VALIDATION SUITE FOR VOLUME VIII: POSITIVITY TRANSFORMATION
# Tests the extension to the full line, integration by parts engine,
# derivative shifting for positivity, and the final positive operator
# representation with strict error bounds.

import sys
import os
import math
import numpy as np
import mpmath as mp
import pytest

# Inject the proof directory into sys.path
PROOF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'VOLUME_VIII_POSITIVITY_TRANSFORMATION_PROOF'))
sys.path.insert(0, PROOF_DIR)

try:
    import VOLUME_VIII_POSITIVITY_TRANSFORMATION as pt
except ImportError:
    # Fallback for naming variations
    import PositivityTransformation as pt

mp.mp.dps = 70
TOL = 1e-10

# --- Helper functions for tests ---

def mock_k_hat(xi, H):
    """A simple positive, symmetric, exponentially decaying kernel for testing."""
    xi_mp = mp.mpf(xi)
    H_mp = mp.mpf(H)
    return mp.exp(-mp.pi * H_mp * mp.fabs(xi_mp))

def mock_S_abs_sq(xi):
    """A simple positive, symmetric signal for testing."""
    xi_mp = mp.mpf(xi)
    return mp.exp(-0.5 * xi_mp**2)

class TestTruncationToFullLine:
    """
    Requirement 1.1, 2.3: Truncation to full-line integral and tail bounds.
    """

    def test_tail_bound_integral_validity(self):
        """
        Verify that the tail bound correctly bounds the integral of the
        exponential tail.
        """
        H = 1.0
        L = 2.0
        bound_on_S = 2.5 # Arbitrary constant bound on |S|^2

        # The tail bound assumes k_hat(xi) <= k(0) * exp(-pi^2 H |xi|)
        # Let's compute the exact integral of the upper bound for x > L
        # int_L^inf e^{-pi^2 H x} dx = e^{-pi^2 H L} / (pi^2 H)
        
        H_mp = mp.mpf(H)
        L_mp = mp.mpf(L)
        k0 = mp.fabs(pt.k_hat(0.0, H))
        C = k0 if k0 > 0 else mp.mpf("1")
        
        expected_one_side = C * mp.exp(-mp.pi**2 * H_mp * L_mp) / (mp.pi**2 * H_mp)
        expected_total = float(2 * expected_one_side) * bound_on_S

        calculated_bound = pt.tail_bound_integral(H, L, bound_on_S)
        
        assert abs(calculated_bound - expected_total) < TOL

    def test_extend_to_full_line_computation(self):
        """
        Verify that extending to the full line computes the truncated integral
        correctly and returns a finite, positive tail bound.
        """
        H = 1.0
        L = 5.0
        
        # Compute exact truncated integral with mock functions
        def integrand(x):
            return mock_k_hat(x, H) * mock_S_abs_sq(x)
            
        exact_trunc_integral = float(mp.quad(integrand, [-L, L]))
        
        val, err_bound = pt.extend_to_full_line(
            kernel=mock_k_hat,
            S_abs_sq=mock_S_abs_sq,
            L=L,
            H=H,
            tol=1e-12
        )
        
        assert abs(val - exact_trunc_integral) < TOL
        assert err_bound > 0
        assert math.isfinite(err_bound)

class TestIntegrationByPartsEngine:
    """
    Requirement 1.2, 2.2: Integration by parts and antiderivatives of |S|^2.
    """

    def test_antiderivative_computation(self):
        """
        Verify that S_abs_sq_antideriv computes the correct antiderivatives.
        Using a simple mock config where S(xi) creates a known |S(xi)|^2.
        """
        cfg = pt.DirichletConfig(N=1, sigma=0.5, window_type="sharp")
        # For N=1, a_1 = 1, S(xi) = 1*e^{-i xi log 1} = 1.
        # So |S(xi)|^2 = 1.
        
        # A_0(xi) = 1
        assert abs(pt.S_abs_sq_antideriv(2.0, cfg, 0, xi0=0.0) - 1.0) < TOL
        
        # A_1(xi) = int_0^xi 1 dt = xi
        assert abs(pt.S_abs_sq_antideriv(2.0, cfg, 1, xi0=0.0) - 2.0) < TOL
        
        # A_2(xi) = int_0^xi t dt = 0.5 * xi^2
        assert abs(pt.S_abs_sq_antideriv(2.0, cfg, 2, xi0=0.0) - 2.0) < TOL
        assert abs(pt.S_abs_sq_antideriv(3.0, cfg, 2, xi0=0.0) - 4.5) < TOL

    def test_integration_by_parts_identity(self):
        """
        Check that IBP preserves the integral value exactly (up to numerical error)
        for order=0 and order=1.
        """
        cfg = pt.DirichletConfig(N=2, sigma=0.5, window_type="sharp")
        H = 1.0
        L = 2.0
        
        # Base integral (order 0)
        res0 = pt.integration_by_parts(pt.k_hat, lambda xi: pt.S_abs_sq_from_cfg(xi, cfg), L, H, order=0)
        
        # IBP once (order 1)
        res1 = pt.integration_by_parts(pt.k_hat, lambda xi: pt.S_abs_sq_from_cfg(xi, cfg), L, H, order=1)
        
        assert abs(res0.transformed_integral - res1.transformed_integral) < 1e-8


class TestDerivativeShiftingAndPositivity:
    """
    Requirement 1.3, 1.4, 2.4: Shift derivatives, ensure positive weight,
    and verify PositiveOperator representation.
    """

    def test_positive_weight_after_shift(self):
        """
        After applying shift_derivatives_to_S, the resulting weight function
        w(xi) must be non-negative on the interval [-L, L].
        """
        cfg = pt.DirichletConfig(N=2, sigma=0.5, window_type="sharp")
        H = 1.0
        L = 2.0
        
        # Use the actual k_hat
        pos_form = pt.shift_derivatives_to_S(
            kernel=pt.k_hat,
            S_abs_sq=lambda xi: pt.S_abs_sq_from_cfg(xi, cfg),
            L=L,
            H=H,
            max_order=2,
            truncation_error=1e-10
        )
        
        # Check positivity of the weight function on a grid
        xs = np.linspace(-L, L, 50)
        for x in xs:
            assert pos_form.weight_function(x) >= -1e-12

    def test_positive_operator_evaluation(self):
        """
        Verify that PositiveOperator.evaluate computes the quadratic form correctly
        by comparing it to the standard truncated integral.
        """
        cfg = pt.DirichletConfig(N=2, sigma=0.5, window_type="sharp")
        H = 1.0
        L = 2.0
        
        S_fn = lambda xi: pt.S_abs_sq_from_cfg(xi, cfg)
        
        # Direct truncated integral
        direct_val, _ = pt.extend_to_full_line(pt.k_hat, S_fn, L, H, tol=1e-12)
        
        # Shift derivatives (order 0 just uses k_hat)
        pos_form = pt.shift_derivatives_to_S(
            kernel=pt.k_hat,
            S_abs_sq=S_fn,
            L=L,
            H=H,
            max_order=0,
            truncation_error=1e-10
        )
        
        op = pt.PositiveOperator(
            weight_function=pos_form.weight_function,
            boundary_terms=pos_form.boundary_terms,
            truncation_error=pos_form.truncation_error
        )
        
        eval_val = op.evaluate(S_fn, L)
        
        assert abs(direct_val - eval_val) < 1e-10


class TestErrorBudgetAndConsistency:
    """
    Requirement 1.5, 4.1: Explicit error tracking and consistency checks.
    """

    def test_error_budget_finite(self):
        """
        Ensure all error components (truncation, numerical) are finite
        and sum to a valid interval.
        """
        cfg = pt.DirichletConfig(N=3, sigma=0.5, window_type="sharp")
        H = 1.5
        L = 3.0
        
        res = pt.positivity_transformation(cfg, H, L, max_order=1)
        
        assert math.isfinite(res.truncation_error)
        assert math.isfinite(res.numerical_error)
        assert res.truncation_error > 0
        assert res.numerical_error >= 0
        
        lower, upper = res.error_interval
        assert lower <= res.total_positive_form <= upper
        assert abs((upper - lower) / 2 - (res.truncation_error + res.numerical_error)) < 1e-12

    def test_consistency_with_volume_V(self):
        """
        Compare the original integral with the transformed positive form.
        The difference must be within the declared numerical error.
        """
        cfg = pt.DirichletConfig(N=5, sigma=0.5, window_type="gaussian", window_params={"alpha": 1.0})
        H = 1.0
        L = 4.0
        
        res = pt.positivity_transformation(cfg, H, L, max_order=1)
        
        diff = abs(res.original_integral - res.transformed_integral)
        # Transformed integral should match original integral exactly up to numerical quadrature limits
        # We use a slightly looser tolerance here because order=1 involves numerical derivatives and integrals
        assert diff < 1e-8
        
        # The numerical error tracked in the result object should bound the actual difference
        assert diff <= res.numerical_error + 1e-14

    def test_boundary_terms_decay(self):
        """
        Show that boundary terms decrease as L increases (for fixed H).
        """
        cfg = pt.DirichletConfig(N=2, sigma=0.5, window_type="sharp")
        H = 1.0
        
        # Note: To see boundary term decay, we need to do at least 1 IBP
        res_small_L = pt.integration_by_parts(pt.k_hat, lambda xi: pt.S_abs_sq_from_cfg(xi, cfg), L=2.0, H=H, order=1)
        res_large_L = pt.integration_by_parts(pt.k_hat, lambda xi: pt.S_abs_sq_from_cfg(xi, cfg), L=4.0, H=H, order=1)
        
        # |k_hat(L)| decays exponentially, so boundary terms should shrink significantly
        assert abs(res_large_L.boundary_terms) < abs(res_small_L.boundary_terms)

if __name__ == '__main__':
    pytest.main([__file__, "-v"])