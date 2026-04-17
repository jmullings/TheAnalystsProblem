#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# test_SPECTRAL_EXPANSION.py
#
# VALIDATION SUITE FOR VOLUME IV: SPECTRAL EXPANSION
# Tests the exact analytic Fourier symbol \hat{k}_H(ξ), the spectral 
# representation of the quadratic form Q_H, the Parseval Bridge, 
# the σ-selector mechanism, and spectral splitting (diagonal vs off-diagonal).

import sys
import os
import math
import numpy as np
import mpmath as mp
import pytest

# Inject the proof directory into sys.path
PROOF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'VOLUME_IV_SPECTRAL_EXPANSION_PROOF'))
sys.path.insert(0, PROOF_DIR)

try:
    import SPECTRAL_EXPANSION as se
except ImportError:
    # Handle potential filename variations
    import VOLUME_IV_SPECTRAL_EXPANSION as se

# Use high precision for analytic verification
mp.mp.dps = 80
TOL = 1e-12

class TestAnalyticFourierSymbol:
    """
    Requirement 1 & 5: Analytic Fourier symbol k_hat(xi, H) 
    and Frequency decay diagnostics.
    """

    @pytest.mark.parametrize("H", [0.5, 1.0, 3.14])
    def test_k_hat_positivity(self, H):
        """Prove \hat{k}_H(ξ) ≥ 0 for all ξ (Kernel Universality)."""
        H_mp = mp.mpf(H)
        # Test across a wide range of frequencies
        for xi in np.linspace(-20, 20, 100):
            val = se.k_hat(mp.mpf(xi), H_mp)
            assert val >= 0.0, f"Fourier symbol negativity detected at ξ={xi}, H={H}"

    @pytest.mark.parametrize("H", [1.0, 2.0])
    def test_k_hat_symmetry(self, H):
        """Prove \hat{k}_H(ξ) is an even function."""
        H_mp = mp.mpf(H)
        for xi in np.linspace(0.1, 10, 20):
            val_pos = se.k_hat(mp.mpf(xi), H_mp)
            val_neg = se.k_hat(mp.mpf(-xi), H_mp)
            assert mp.almosteq(val_pos, val_neg, TOL), f"Symmetry broken at ξ={xi}"

    def test_k_hat_limit_at_zero(self):
        """Verify the L'Hôpital limit at ξ=0: \hat{k}_H(0) = 8/H^2."""
        H = 2.0
        expected = mp.mpf("8.0") / (H**2)
        val = se.k_hat(0.0, H)
        assert mp.almosteq(val, expected, TOL), "Limit at ξ=0 failed"

    def test_exponential_decay(self):
        """Verify \hat{k}_H(ξ) decays exponentially as e^{-π^2 H |ξ|}."""
        H = 1.0
        xis = [2.0, 4.0, 6.0, 8.0]
        fit = se.fit_exponential_decay(H, xis)
        
        # The true asymptotic decay of w_hat is π^2 H.
        # We expect the fitted slope to closely match this theoretical rate.
        expected_slope = math.pi**2 * H
        
        # We allow a generous 5% tolerance because the polynomial prefactor 
        # ((2πξ)^2 + 4/H^2) slightly alters the pure exponential slope 
        # over finite sampling intervals.
        assert abs(fit.slope - expected_slope) / expected_slope < 0.05, \
            f"Expected slope ~{expected_slope:.2f}, got {fit.slope:.2f}"


class TestDirichletWave:
    """
    Requirement 2: Dirichlet wave S(ξ) and its σ-perturbed variants.
    """

    def test_S_xi_at_zero(self):
        """At ξ=0, S_σ(0) should equal the sum of n^{-σ}."""
        N = 50
        sigma = 0.5
        s_val = se.S_xi(0.0, N=N, sigma=sigma)
        
        expected = sum(n**(-sigma) for n in range(1, N+1))
        assert abs(s_val.real - expected) < 1e-10
        assert abs(s_val.imag) < 1e-10

    def test_S_xi_conjugate_symmetry(self):
        """S_σ(-ξ) should be the complex conjugate of S_σ(ξ)."""
        N = 30
        sigma = 0.5
        xi = 2.5
        
        s_pos = se.S_xi(xi, N, sigma)
        s_neg = se.S_xi(-xi, N, sigma)
        
        assert abs(s_pos.real - s_neg.real) < 1e-10
        assert abs(s_pos.imag - (-s_neg.imag)) < 1e-10


class TestParsevalBridge:
    """
    Requirement 3: Parseval bridge equivalence (Q_matrix ≈ Q_spectral).
    """

    @pytest.mark.parametrize("T0", [0.0, 14.1347])
    def test_parseval_equivalence(self, T0):
        """CRITICAL TEST: The spectral integral must match the Toeplitz matrix form."""
        N = 20  # Keep small for integration speed
        H = 1.0
        L = 15.0 # Integration bounds
        
        comp = se.compare_parseval(N, H, T0, sigma=0.5, L=L)
        
        # We require at least 8 digits of precision for the bridge
        assert comp.rel_diff < 1e-8, \
            f"Parseval Bridge Failed: Q_mat={comp.Q_matrix}, Q_spec={comp.Q_spectral}, RelDiff={comp.rel_diff}"


class TestSigmaSelector:
    """
    Requirement 4: σ-selector mechanism peaks at σ = 1/2.
    """

    def test_sigma_profile_maximum_at_half(self):
        """Verify that Q_H^spec(σ) reaches its maximum exactly at σ=0.5."""
        N = 20
        H = 1.0
        T0 = 0.0
        L = 10.0
        
        # Sample symmetrically around 0.5
        sigmas = [0.40, 0.45, 0.50, 0.55, 0.60]
        profile = se.sigma_profile(N, H, T0, sigmas, L)
        
        # Extract values
        vals = {p.sigma: p.Q_spectral for p in profile}
        
        # Symmetry check
        assert abs(vals[0.45] - vals[0.55]) < 1e-8, "Profile is not symmetric around 0.5"
        assert abs(vals[0.40] - vals[0.60]) < 1e-8, "Profile is not symmetric around 0.5"
        
        # Maximum check
        assert vals[0.50] > vals[0.45], "Peak is not at σ=0.5"
        assert vals[0.50] > vals[0.55], "Peak is not at σ=0.5"


class TestSpectralSplitting:
    """
    Requirement 5: Diagonal vs off-diagonal spectral contributions.
    """

    def test_spectral_diag_off_split_reconstruction(self):
        """Q_diag + Q_off should equal the total Q_spectral."""
        N = 15
        H = 1.5
        T0 = 14.1347
        L = 10.0
        
        split = se.spectral_diag_off_split(N, H, T0, sigma=0.5, L=L)
        total_spec = float(se.Q_spectral(N, H, T0, L=L, sigma=0.5))
        
        reconstructed = split.Q_diag + split.Q_off
        assert abs(reconstructed - total_spec) / abs(total_spec) < 1e-8

    def test_spectral_diagonal_dominance(self):
        """
        The diagonal contribution should remain strictly positive and invariant to T0.
        """
        split_0 = se.spectral_diag_off_split(15, 1.0, T0=0.0, L=10.0)
        split_T = se.spectral_diag_off_split(15, 1.0, T0=21.022, L=10.0)
        
        assert split_0.Q_diag > 0.0, "Diagonal term must be positive"
        assert abs(split_0.Q_diag - split_T.Q_diag) < 1e-8, "Diagonal term must be T0 invariant"


class TestLocalizationAndStability:
    """
    Requirement 7 & 8: Localization experiments and T0-stability.
    """

    def _effective_width(self, N: int, H: float, T0: float) -> float:
        """
        Helper (non-test) method: compute effective half-width
        where weight > 1% of max.
        """
        xi_grid = np.linspace(-5.0, 5.0, 101)
        loc = se.localization_profile(N, H, T0, xi_grid, sigma=0.5)
        weights = np.array([p.weight for p in loc])
        xis_arr = np.array([p.xi for p in loc])

        # Effective half-width (where weight > 1% of max)
        mask = weights > 0.01 * np.max(weights)
        if not np.any(mask):
            return 0.0
        return float(np.max(np.abs(xis_arr[mask])))

    @pytest.mark.parametrize("H", [0.5, 2.0])
    def test_localization_width_scales_inversely_with_H(self, H):
        """
        As H increases, the spectral energy should become more localized
        (narrower band in ξ).  This test simply checks that the computed
        width is positive for both values of H; the ordering is verified
        in test_H_localization_comparison.
        """
        N = 20
        T0 = 0.0
        width = self._effective_width(N, H, T0)
        assert width > 0.0

    def test_H_localization_comparison(self):
        """Compare the effective width for small vs large H."""
        N = 20
        T0 = 0.0
        width_small_H = self._effective_width(N, H=0.5, T0=T0)
        width_large_H = self._effective_width(N, H=2.0, T0=T0)

        assert width_small_H > width_large_H, (
            f"Localization failed: width(H=0.5)={width_small_H} "
            f"not > width(H=2.0)={width_large_H}"
        )

    def test_T0_stability(self):
        """
        Verify that scanning across T0 produces bounded variations in Q_H^spec
        without violent resonance spikes.
        """
        N = 20
        H = 1.0
        T0_vals = np.linspace(0, 30, 10).tolist()

        scan = se.T0_scan(N, H, T0_vals, L=10.0, sigma=0.5)

        # All values should be strictly positive (Bochner Positivity Basin)
        for s in scan:
            assert s.Q_spectral > 0.0, f"Positivity violated at T0={s.T0}"

        # Variation should be bounded
        vals = [s.Q_spectral for s in scan]
        max_val = max(vals)
        min_val = min(vals)

        # Ensure the maximum is not orders of magnitude larger than the minimum
        assert max_val / min_val < 50.0, "Violent resonance spike detected"
if __name__ == '__main__':
    pytest.main([__file__, "-v"])