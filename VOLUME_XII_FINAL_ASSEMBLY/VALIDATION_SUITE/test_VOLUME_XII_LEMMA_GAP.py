#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_VOLUME_XII_LEMMA_GAP.py

VALIDATION SUITE FOR VOLUME XII: LEMMA GAP (G1)
Tests the numerical harness for bounding the localized large-sieve constant,
including kernel evaluations, window functions, coefficient generation,
diagonal/off-diagonal calculations, empirical scaling laws, and the
corrected analytic mean-value constant.
"""

import sys
import os
import math
import numpy as np
import pytest

# Inject the proof directory into sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Adjust this path if your directory structure is different
PROOF_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'VOLUME_XII_FINAL_ASSEMBLY_PROOF'))
if PROOF_DIR not in sys.path:
    sys.path.insert(0, PROOF_DIR)

try:
    import VOLUME_XII_LEMMA_GAP as lg
except ImportError:
    # Fallback if the file is named differently in the environment
    try:
        import empirical_large_sieve_constant_windowed as lg
    except ImportError:
        pytest.skip("Could not import VOLUME_XII_LEMMA_GAP module.", allow_module_level=True)


class TestKernelAndWindows:
    """Test fundamental kernel and window functions."""

    def test_k_H_properties(self):
        """k_H should be positive, symmetric, and peak at t=0 with value 6/H^2."""
        H = 1.5
        # Peak value
        assert math.isclose(lg.k_H(0.0, H), 6.0 / (H**2), rel_tol=1e-9)
        # Positivity
        for t in [0.0, 1.0, -1.0, 10.0]:
            assert lg.k_H(t, H) > 0.0
        # Symmetry
        assert math.isclose(lg.k_H(2.5, H), lg.k_H(-2.5, H), rel_tol=1e-9)

    def test_k_H_trunc(self):
        """Truncated kernel should match k_H within bandwidth, 0 outside."""
        H = 1.0
        B = 3.5
        # Inside band
        assert math.isclose(lg.k_H_trunc(2.0, H, B), lg.k_H(2.0, H), rel_tol=1e-9)
        # Outside band
        assert lg.k_H_trunc(4.0, H, B) == 0.0

    def test_bump_window(self):
        """Bump window should be 1 at 0, 0 at boundaries."""
        assert math.isclose(lg.bump_window(0.0), math.exp(-1.0), rel_tol=1e-9)
        assert lg.bump_window(1.0) == 0.0
        assert lg.bump_window(-1.0) == 0.0
        assert lg.bump_window(1.5) == 0.0
        
        # Test smoothness/monotonicity on (0, 1)
        xs = np.linspace(0, 0.99, 10)
        vals = [lg.bump_window(x) for x in xs]
        for i in range(1, len(vals)):
            assert vals[i] < vals[i-1]

    def test_gaussian_window(self):
        """Gaussian window should peak at 0."""
        alpha = 2.0
        assert math.isclose(lg.gaussian_window(0.0, alpha), 1.0, rel_tol=1e-9)
        assert lg.gaussian_window(1.0, alpha) < 1.0
        assert math.isclose(lg.gaussian_window(1.0, alpha), math.exp(-alpha), rel_tol=1e-9)


class TestCoefficientsAndDiagonal:
    """Test coefficient generation and diagonal mass computations."""

    def test_generate_coefficients(self):
        """Base coefficients should match a_n = n^{-1/2} * w(n/N)."""
        N = 10
        # Use flat window (w=1)
        coeffs = lg.generate_coefficients(N, lambda x: 1.0)
        assert len(coeffs) == N
        assert math.isclose(coeffs[0].real, 1.0, rel_tol=1e-9)
        assert math.isclose(coeffs[3].real, 4**(-0.5), rel_tol=1e-9)

    def test_generate_coefficients_weighted(self):
        """Test named window coefficient generator."""
        N = 20
        coeffs_flat = lg.generate_coefficients_weighted(N, window="flat")
        assert math.isclose(coeffs_flat[0].real, 1.0, rel_tol=1e-9)
        
        coeffs_bump = lg.generate_coefficients_weighted(N, window="bump")
        # bump window is evaluated at 2*(n/N) - 1.
        # For n=N/2, x = 0.5, arg = 0, bump(0) = e^{-1}
        # coeffs[N/2 - 1] = (N/2)^{-1/2} * e^{-1}
        mid_idx = N // 2 - 1
        expected = ((N/2)**(-0.5)) * math.exp(-1.0)
        assert math.isclose(coeffs_bump[mid_idx].real, expected, rel_tol=1e-9)

    def test_diagonal_mass(self):
        """D_H should equal (6/H^2) * sum |a_n|^2."""
        H = 2.0
        a = [complex(1.0, 0.0), complex(0.5, 0.0)]
        expected = (6.0 / 4.0) * (1.0**2 + 0.5**2)
        result = lg.diagonal_mass(a, H)
        assert math.isclose(result, expected, rel_tol=1e-9)


class TestOffDiagonalAndCaching:
    """Test kernel caching and off-diagonal evaluation."""

    def test_kernel_cache(self):
        """Kernel cache should return correct matrix with zero diagonal."""
        H = 1.0
        B = 3.5
        N = 5
        K = lg._kernel_cache_key(H, B, N)
        
        assert K.shape == (N, N)
        # Diagonal should be exactly 0
        np.testing.assert_array_equal(np.diag(K), np.zeros(N))
        
        # Off-diagonal element K[0, 1] = k_H(log 1 - log 2)
        expected_k12 = lg.k_H_trunc(math.log(1) - math.log(2), H, B)
        assert math.isclose(K[0, 1], expected_k12, rel_tol=1e-9)
        
        # Symmetry
        assert math.isclose(K[0, 1], K[1, 0], rel_tol=1e-9)

    def test_off_diagonal_vectorized_vs_adaptive(self):
        """Vectorized off-diagonal should match the adaptive wrapper."""
        N = 10
        H = 1.5
        T0 = 14.134
        a = lg.generate_coefficients_weighted(N, window="flat")
        
        # Vectorized direct
        a_arr = np.array(a, dtype=complex)
        logs = np.log(np.arange(1, N + 1, dtype=np.float64))
        K = lg._kernel_cache_key(H, None, N)
        res_vec = lg.off_diagonal_vectorized(a_arr, logs, T0, K)
        
        # Adaptive wrapper
        res_adapt = lg.off_diagonal_adaptive(a, H, T0, B=None)
        
        assert math.isclose(res_vec.real, res_adapt.real, rel_tol=1e-9)
        assert math.isclose(res_vec.imag, res_adapt.imag, rel_tol=1e-9)


class TestEmpiricalAveraging:
    """Test empirical T-averaged constant C(H; N, T)."""

    def test_averaged_off_diagonal_L2(self):
        """Averaged L2 should be non-negative and finite."""
        N = 5
        H = 1.0
        T = 10.0
        a = lg.generate_coefficients_weighted(N, window="flat")
        
        # Use small number of samples for fast testing
        num_samples = 10
        res = lg.averaged_off_diagonal_L2(a, H, T, num_samples, B=3.5)
        
        assert res >= 0.0
        assert math.isfinite(res)

    def test_empirical_C_H(self):
        """Empirical C_H should be computed correctly."""
        N = 10
        H = 0.5
        T = 20.0
        a = lg.generate_coefficients_weighted(N, window="bump")
        
        C_val = lg.empirical_C_H(a, H, T, num_samples=10, B=3.5)
        
        assert C_val >= 0.0
        assert math.isfinite(C_val)
        # For N=10, bump window, H=0.5, we expect C_val < 1.0
        assert C_val < 1.0

    # Removed the @pytest.mark.slow decorator to avoid the warning,
    # as the mark is not explicitly registered in pytest.ini.
    def test_certify_C_H_converged(self):
        """Certification routine should converge or hit max_iter."""
        N = 10
        H = 0.5
        T = 10.0
        a = lg.generate_coefficients_weighted(N, window="bump")
        
        # Fast test with loose tolerance
        res = lg.certify_C_H_converged(a, H, T, B=3.5, tol=0.1, max_iter=2)
        
        assert "C(H)" in res
        assert "passes" in res
        assert "converged" in res
        assert res["C(H)"] >= 0.0


class TestAnalyticMeanValue:
    """Test the corrected analytic mean-value constant."""

    def test_infinite_series_constant_analytic(self):
        """B_analytic(H, w; N) should be computable and < 1 for small N."""
        H = 0.5
        N = 50  # Small N for fast test
        
        B_val = lg.infinite_series_constant_analytic(H, N)
        
        assert B_val >= 0.0
        assert math.isfinite(B_val)
        # Should be comfortably below 1 for N=50
        assert B_val < 0.95

    def test_infinite_series_constant_corrected_alias(self):
        """The corrected alias should call the analytic function."""
        H = 0.5
        N = 20
        val1 = lg.infinite_series_constant_analytic(H, N)
        val2 = lg.infinite_series_constant_corrected(H=H, N_analytic=N)
        assert math.isclose(val1, val2, rel_tol=1e-9)


class TestScalingModels:
    """Test scaling law fitting utilities."""

    def test_fit_scaling_log(self):
        """Least-squares fit C(N) ~ A/log N + B should return valid parameters."""
        Ns = [10, 50, 100, 500]
        # Create synthetic data: C(N) = 2.0 / log(N) + 0.5
        Cs = [2.0 / math.log(n) + 0.5 for n in Ns]
        
        A, B = lg.fit_scaling_log(Ns, Cs)
        
        assert math.isclose(A, 2.0, rel_tol=1e-5)
        assert math.isclose(B, 0.5, rel_tol=1e-5)

    def test_asymptotic_passes(self):
        """Check if asymptotic limit B is < 1."""
        Ns = [10, 50, 100]
        # Data converging to 0.8
        Cs_pass = [1.0 / math.log(n) + 0.8 for n in Ns]
        assert lg.asymptotic_passes(Ns, Cs_pass) is True
        
        # Data converging to 1.2
        Cs_fail = [1.0 / math.log(n) + 1.2 for n in Ns]
        assert lg.asymptotic_passes(Ns, Cs_fail) is False


class TestDiagnostics:
    """Test diagnostic and decomposition functions."""

    def test_split_near_far_indices(self):
        """Index splitting should partition all off-diagonal pairs correctly."""
        N = 10
        H = 1.0
        B = 1.0  # Use small B so some pairs fall into 'far'
        
        near, far = lg.split_near_far_indices(N, H, B)
        
        # Total off-diagonal pairs is N*(N-1)
        assert len(near) + len(far) == N * (N - 1)
        
        logs = [math.log(n) for n in range(1, N + 1)]
        # Verify a near pair
        if near:
            m, n = near[0]
            assert abs(logs[m-1] - logs[n-1]) <= B * H
        # Verify a far pair
        if far:
            m, n = far[0]
            assert abs(logs[m-1] - logs[n-1]) > B * H

    def test_off_diagonal_far_bound(self):
        """Far bound should be positive and finite."""
        N = 10
        H = 1.0
        B = 3.5
        a = lg.generate_coefficients_weighted(N, window="flat")
        
        bound = lg.off_diagonal_far_bound(a, H, B)
        assert bound > 0.0
        assert math.isfinite(bound)

if __name__ == '__main__':
    pytest.main([__file__, "-v"])