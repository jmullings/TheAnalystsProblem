#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_kernel_decomposition.py

VALIDATION SUITE FOR VOLUME II: KERNEL DECOMPOSITION

This test suite validates strict T1-level kernel algebra including:
- Curvature/floor decomposition
- Fourier symbol analysis
- Bochner-PSD checks
- Analytical constants and identities

All tests use high-precision arithmetic (mpmath with 50 decimal places) to ensure
mathematical rigor at the T1 verification level.
"""

import sys
import os
import math
import numpy as np
import mpmath as mp
import pytest

# Inject the proof directory into sys.path
PROOF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'KERNAL_DECOMPOSITION_PROOF'))
sys.path.insert(0, PROOF_DIR)

# Note: The application file was renamed to kernel_decomposition.py to match Python conventions
# If the file is still named KERNAL_DECOMPOSITION_PROBLEM.py on your system, change this import
try:
    import kernel_decomposition as kd
except ImportError:
    import KERNAL_DECOMPOSITION_PROBLEM as kd

# Configure precision for T1-level rigorous verification
mp.mp.dps = 50
TOL = mp.mpf('1e-45')
NUM_TOL = 1e-9  # For numpy/float operations

# ============================================================================
# PYTEST FIXTURES
# ============================================================================

@pytest.fixture
def high_precision_values():
    """Provide common high-precision test values."""
    return {
        'small': mp.mpf('0.1'),
        'unit': mp.mpf('1.0'),
        'medium': mp.mpf('2.5'),
        'pi': mp.pi,
        'test_point': mp.mpf('0.777')
    }

@pytest.fixture
def H_values():
    """Provide standard H parameter values for testing."""
    return [0.5, 1.0, 2.0, math.pi]

# ============================================================================
# TEST CLASSES
# ============================================================================

class TestHyperbolicPrimitives:
    """Validate core hyperbolic primitives at high precision (T1)."""
    
    def test_sech_identities(self):
        """Verify sech(x) = 1/cosh(x) and power relationships."""
        x = mp.mpf('1.2345')
        # sech(x) = 1/cosh(x)
        assert mp.almosteq(kd.sech(x), 1.0 / mp.cosh(x), TOL), \
            "sech(x) should equal 1/cosh(x)"
        # sech2(x) = sech(x)^2
        assert mp.almosteq(kd.sech2(x), kd.sech(x)**2, TOL), \
            "sech2(x) should equal sech(x)^2"
        # sech4(x) = sech2(x)^2
        assert mp.almosteq(kd.sech4(x), kd.sech2(x)**2, TOL), \
            "sech4(x) should equal sech2(x)^2"
        
    def test_tanh_identity(self):
        """Verify tanh(x) = sinh(x)/cosh(x)."""
        x = mp.mpf('1.2345')
        assert mp.almosteq(kd.tanh_(x), mp.sinh(x) / mp.cosh(x), TOL), \
            "tanh(x) should equal sinh(x)/cosh(x)"

    def test_sech_symmetry(self):
        """Verify sech is an even function: sech(-x) = sech(x)."""
        x = mp.mpf('2.5')
        assert mp.almosteq(kd.sech(x), kd.sech(-x), TOL), \
            "sech should be an even function"

    def test_sech_bounds(self):
        """Verify 0 < sech(x) <= 1 for all x."""
        for x_val in [0.0, 1.0, 5.0, 10.0]:
            x = mp.mpf(x_val)
            sech_val = kd.sech(x)
            assert 0 < sech_val <= 1, \
                f"sech({x}) = {sech_val} should be in (0, 1]"


class TestKernelDecompositionAlgebra:
    """
    Requirement 1 & 10: Exact analytic decomposition (T1) and 
    Negativity Localization Theorem.
    """
    
    @pytest.mark.parametrize("H", [0.5, 1.0, math.pi])
    def test_derivatives_by_finite_difference(self, H):
        """Strict verification of symbolic derivatives using high-precision FD (T1)."""
        H_mp = mp.mpf(H)
        t = mp.mpf('0.777')
        dt = mp.mpf('1e-20')
        
        # First derivative
        fd_prime = (kd.w_H(t + dt, H_mp) - kd.w_H(t - dt, H_mp)) / (2 * dt)
        sym_prime = kd.w_H_prime(t, H_mp)
        assert mp.almosteq(fd_prime, sym_prime, mp.mpf('1e-25')), \
            f"First derivative mismatch at H={H}, t={t}"
        
        # Second derivative
        fd_double_prime = (kd.w_H_prime(t + dt, H_mp) - kd.w_H_prime(t - dt, H_mp)) / (2 * dt)
        sym_double_prime = kd.w_H_double_prime(t, H_mp)
        assert mp.almosteq(fd_double_prime, sym_double_prime, mp.mpf('1e-25')), \
            f"Second derivative mismatch at H={H}, t={t}"

    @pytest.mark.parametrize("H", [0.1, 1.0, 2.5])
    def test_exact_decomposition_identity(self, H):
        r"""
        Prove k_H(t) = -w_H''(t) + (4/H^2)w_H(t) = (6/H^2)sech^4(t/H) (T1).
        """
        H_mp = mp.mpf(H)
        for t_val in [0.0, 0.5, 1.0, 5.0]:
            t = mp.mpf(t_val)
            
            w_h = kd.w_H(t, H_mp)
            w_h_pp = kd.w_H_double_prime(t, H_mp)
            
            # LHS: -w_H''(t) + (4/H^2)w_H(t)
            lhs = -w_h_pp + (4.0 / (H_mp * H_mp)) * w_h
            
            # RHS: (6/H^2)sech^4(t/H)
            rhs = kd.k_H_sech4_closed_form(t, H_mp)
            
            # Implementation wrapper
            k_t = kd.k_H(t, H_mp)
            
            assert mp.almosteq(lhs, rhs, TOL), \
                f"Decomposition identity failed at H={H}, t={t}"
            assert mp.almosteq(k_t, rhs, TOL), \
                f"k_H implementation mismatch at H={H}, t={t}"

    def test_kernel_symmetry(self):
        """Verify k_H is an even function: k_H(-t) = k_H(t)."""
        H_mp = mp.mpf('1.0')
        for t_val in [0.5, 1.0, 2.0]:
            t = mp.mpf(t_val)
            assert mp.almosteq(kd.k_H(t, H_mp), kd.k_H(-t, H_mp), TOL), \
                f"k_H should be symmetric at t={t}"


class TestSignAnalysisAndOptimality:
    """
    Requirement 2, 3 & 4: Sign analysis, floor term positivity, 
    and sharpness of lambda correction (T1).
    """
    
    @pytest.mark.parametrize("H", [1.0, 2.0])
    def test_w_double_prime_sign_transition(self, H):
        """Prove exact transition point for curvature sign change (T1)."""
        H_mp = mp.mpf(H)
        info = kd.w_double_prime_sign_info(H_mp)
        t_trans = info["transition_t"]
        
        # Curvature should be exactly 0 at the transition point
        assert mp.almosteq(kd.curvature_term(t_trans, H_mp), 0.0, TOL), \
            f"Curvature should be zero at transition point t={t_trans}"
        
        # Curvature should be negative strictly inside the interval
        t_inside = t_trans * mp.mpf('0.5')
        curv_inside = kd.curvature_term(t_inside, H_mp)
        assert curv_inside < 0.0, \
            f"Curvature should be negative at t={t_inside} (inside interval)"
        
        # Curvature should be positive strictly outside the interval
        t_outside = t_trans * mp.mpf('1.5')
        curv_outside = kd.curvature_term(t_outside, H_mp)
        assert curv_outside > 0.0, \
            f"Curvature should be positive at t={t_outside} (outside interval)"

    @pytest.mark.parametrize("H", [1.0, 2.0])
    def test_floor_term_positivity_and_dominance(self, H):
        """Prove (4/H^2)w_H(t) > 0 and dominates the negative curvature (T1)."""
        H_mp = mp.mpf(H)
        lam = kd.lambda_star(H_mp)
        
        for t_val in np.linspace(0, 5, 20):
            t = mp.mpf(t_val)
            floor = kd.floor_term(t, H_mp, lam)
            assert floor > 0.0, \
                f"Floor term should be positive at t={t}"
            
            curv = kd.curvature_term(t, H_mp)
            # Dominance: floor + curvature = k_H(t) > 0
            total = floor + curv
            assert total > 0.0, \
                f"Floor term should dominate curvature at t={t}: floor={floor}, curv={curv}"

    def test_minimal_lambda_sharpness(self):
        """
        Requirement 4 & 8: For any λ < 4/H^2, kernel becomes negative (T1).
        """
        H = 1.0
        lam_star = float(kd.lambda_star(mp.mpf(H)))
        
        # Exact lam_star should show numeric minimal lambda is <= lam_star
        numeric_min = kd.minimal_lambda_numeric(H=H, t_max=10.0, n_grid=5000)
        assert numeric_min <= lam_star + NUM_TOL, \
            f"Numeric minimal lambda {numeric_min} should not exceed theoretical {lam_star}"
        
        # Perturbation breaks positivity
        lam_perturbed = lam_star - 0.001
        res = kd.finds_negative_for_lambda(H=H, lam=lam_perturbed, t_max=10.0, n_grid=5000)
        assert res is not None, \
            f"Should find negative values for lambda={lam_perturbed} < lambda_star={lam_star}"
        
        t_neg, val_neg = res
        assert val_neg < 0.0, \
            f"Found value {val_neg} at t={t_neg} should be negative"


class TestFourierDomain:
    """
    Requirement 5: Fourier-side decomposition and positivity (T1).
    """
    
    @pytest.mark.parametrize("H", [1.0, 3.0])
    def test_fourier_decomposition_identity(self, H):
        r"""
        Show \hat{k}_H(ξ) = ((2πξ)^2 + λ*) \hat{w}_H(ξ) (T1).
        """
        H_mp = mp.mpf(H)
        lam = kd.lambda_star(H_mp)
        
        for xi_val in [0.0, 0.5, 1.0, 10.0]:
            xi = mp.mpf(xi_val)
            w_hat = kd.w_H_hat(xi, H_mp)
            k_hat = kd.k_H_hat(xi, H_mp)
            
            # Decompose
            energy_term = ((2 * mp.pi * xi) ** 2) * w_hat
            floor_term = lam * w_hat
            
            assert mp.almosteq(k_hat, energy_term + floor_term, TOL), \
                f"Fourier decomposition failed at H={H}, ξ={xi}"
            
            # Both terms must be non-negative
            assert energy_term >= 0.0, \
                f"Energy term should be non-negative at ξ={xi}"
            assert floor_term >= 0.0, \
                f"Floor term should be non-negative at ξ={xi}"

    def test_fourier_symbol_global_positivity(self):
        r"""Verify \hat{k}_H(ξ) >= 0 on a dense grid (T1 verification)."""
        result = kd.fourier_symbol_nonnegative(H=1.0, xi_max=50.0, n_grid=10000)
        assert result, "Fourier symbol should be non-negative on entire grid"


class TestBochnerEquivalence:
    """
    Requirement 6: Bochner equivalence closure (T1).
    """
    
    def test_toeplitz_matrix_symmetry(self):
        r"""Toeplitz matrix built from k_H must be perfectly symmetric (T1)."""
        H = 1.0
        N = 15
        log_ns = np.log(np.arange(1, N + 1, dtype=float))
        K = kd.toeplitz_matrix_from_kernel(log_ns, H)
        
        assert np.allclose(K, K.T, atol=NUM_TOL), \
            "Toeplitz matrix should be symmetric"
        
    def test_toeplitz_matrix_psd(self):
        """
        Pointwise k_H >= 0 with Fourier symbol >= 0 implies PSD Toeplitz (T1).
        """
        H = 1.5
        N = 30
        result = kd.bochner_psd_check(N=N, H=H)
        assert result, f"Toeplitz matrix should be positive semi-definite for H={H}, N={N}"

    @pytest.mark.parametrize("N", [5, 10, 20])
    def test_toeplitz_eigenvalues_positive(self, N):
        """Verify all eigenvalues of Toeplitz matrix are non-negative."""
        H = 1.0
        log_ns = np.log(np.arange(1, N + 1, dtype=float))
        K = kd.toeplitz_matrix_from_kernel(log_ns, H)
        eigenvalues = np.linalg.eigvalsh(K)
        
        assert np.all(eigenvalues >= -NUM_TOL), \
            f"All eigenvalues should be non-negative (min eigenvalue: {eigenvalues.min()})"


class TestIntegralsAndDecay:
    """
    Requirement 7: Integral / norm control (T1).
    """
    
    @pytest.mark.parametrize("H", [0.5, 1.0, 2.0])
    def test_L1_norm_analytic_vs_numeric(self, H):
        r"""Prove \int k_H(t) dt = 8/H (T1)."""
        H_mp = mp.mpf(H)
        exact_L1 = kd.k_H_L1(H_mp)
        
        # Verify analytical formula
        expected_L1 = 8 / H_mp
        assert mp.almosteq(exact_L1, expected_L1, TOL), \
            f"Analytical L1 norm should equal 8/H for H={H}"
        
        # Numerical quadrature check
        numeric_L1 = mp.quad(lambda t: kd.k_H(t, H_mp), [-mp.inf, mp.inf])
        assert mp.almosteq(exact_L1, numeric_L1, mp.mpf('1e-20')), \
            f"Numeric L1 integral should match analytical value for H={H}"
        
    @pytest.mark.parametrize("H", [1.0, 1.5])
    def test_L2_norm_analytic_vs_numeric(self, H):
        r"""Prove \int k_H(t)^2 dt = (1152/35) * H^{-3} (T1)."""
        H_mp = mp.mpf(H)
        exact_L2_sq = kd.k_H_L2_squared(H_mp)
        
        # Verify analytical formula
        expected_L2_sq = (mp.mpf('1152') / mp.mpf('35')) / (H_mp ** 3)
        assert mp.almosteq(exact_L2_sq, expected_L2_sq, TOL), \
            f"Analytical L2^2 norm should equal (1152/35)/H^3 for H={H}"
        
        numeric_L2_sq = mp.quad(lambda t: kd.k_H(t, H_mp)**2, [-mp.inf, mp.inf])
        assert mp.almosteq(exact_L2_sq, numeric_L2_sq, mp.mpf('1e-20')), \
            f"Numeric L2^2 integral should match analytical value for H={H}"

    def test_exponential_decay(self):
        r"""Establish decay rate ~ e^{-4|t|/H} (T1)."""
        H = 1.0
        samples = kd.k_H_decay_sample(H=H, t_values=[0.0, 5.0, 10.0, 20.0])
        
        # Verify extreme decay at t=20
        t_20_val = next(v for t, v in samples if t == 20.0)
        assert t_20_val < 1e-15, \
            f"Kernel should decay to near-zero at t=20: {t_20_val}"
        
        # Check relative decay rate fits exponential profile
        t_5_val = next(v for t, v in samples if t == 5.0)
        t_10_val = next(v for t, v in samples if t == 10.0)
        
        # sech^4(t/H) decays like (2e^{-t/H})^4 = 16e^{-4t/H}
        # k_H(t) = 6/H^2 sech^4(t/H) ~ 96/H^2 e^{-4t/H}
        # ratio k_H(10) / k_H(5) for H=1 should be approx e^{-4*5} = e^{-20}
        ratio = t_10_val / t_5_val
        expected_ratio = math.exp(-20.0)
        
        # Adjusted tolerance for double precision float arithmetic at this scale
        assert abs(ratio - expected_ratio) < 1e-11, \
            f"Decay ratio {ratio} should match exponential prediction {expected_ratio}"


if __name__ == '__main__':
    pytest.main([__file__, "-v", "--tb=short"])