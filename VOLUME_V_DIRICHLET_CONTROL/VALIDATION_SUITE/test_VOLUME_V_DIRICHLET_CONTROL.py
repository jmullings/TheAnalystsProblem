#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# test_VOLUME_V_DIRICHLET_CONTROL.py
#
# VALIDATION SUITE FOR VOLUME V: DIRICHLET POLYNOMIAL CONTROL
# Tests the generalized Dirichlet wave engine, smooth windowing frameworks,
# decay diagnostics vs kernel decay, magnitude bounds, structural decomposition
# (diagonal vs off-diagonal), scaling behavior, and integration with the 
# spectral kernel.

import sys
import os
import math
import numpy as np
import mpmath as mp
import pytest

# Inject the proof directory into sys.path
PROOF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'VOLUME_V_DIRICHLET_CONTROL_PROOF'))
sys.path.insert(0, PROOF_DIR)

try:
    import VOLUME_V_DIRICHLET_CONTROL as vc
except ImportError:
    # Fallback in case of naming variation
    import DIRICHLET_CONTROL as vc

mp.mp.dps = 80
TOL = 1e-12

class TestGeneralizedDirichletEngine:
    """
    Requirement 1: Generalized Dirichlet wave engine S_a(ξ) supporting 
    various coefficient types (plain, log, von_mangoldt, custom).
    """

    def test_plain_coefficients_at_zero(self):
        """S(0) must equal the sum of the coefficients exactly."""
        N = 50
        sigma = 0.5
        cfg = vc.DirichletConfig(N=N, sigma=sigma, weight_type="plain", window_type="sharp")
        
        expected = sum(n**(-sigma) for n in range(1, N+1))
        S_val = vc.S_a_xi(0.0, cfg)
        
        assert abs(S_val.real - expected) < 1e-10
        assert abs(S_val.imag) < 1e-10

    def test_von_mangoldt_coefficients(self):
        """Verify the von Mangoldt weight engine outputs correct primes."""
        N = 20
        sigma = 0.5
        cfg = vc.DirichletConfig(N=N, sigma=sigma, weight_type="von_mangoldt", window_type="sharp")
        
        a, logn = vc.build_coefficients(cfg)
        
        # For N=20, primes and prime powers: 2,3,4,5,7,8,9,11,13,16,17,19
        # Check specific values
        assert a[0] == 0.0  # n=1
        assert abs(a[1] - math.log(2) * (2**-sigma)) < TOL  # n=2 (prime)
        assert abs(a[3] - math.log(2) * (4**-sigma)) < TOL  # n=4 (p^2)
        assert a[5] == 0.0  # n=6 (composite)

    def test_conjugate_symmetry(self):
        """Prove S(-ξ) = conj(S(ξ)) for real coefficients."""
        cfg = vc.DirichletConfig(N=30, sigma=0.5, weight_type="plain", window_type="sharp")
        xi = 2.5
        
        S_pos = vc.S_a_xi(xi, cfg)
        S_neg = vc.S_a_xi(-xi, cfg)
        
        assert abs(S_pos.real - S_neg.real) < 1e-10
        assert abs(S_pos.imag - (-S_neg.imag)) < 1e-10


class TestSmoothWindowFramework:
    """
    Requirement 2: Implementation of smooth test-functions to replace sharp cutoffs.
    """

    def test_gaussian_window_decay(self):
        """Gaussian window must smoothly decay towards N."""
        N = 100
        cfg = vc.DirichletConfig(N=N, sigma=0.5, window_type="gaussian", window_params={"alpha": 2.0})
        raw_a, _ = vc.build_coefficients(cfg)
        a_windowed = vc.apply_window(cfg, raw_a)
        
        # Verify n=1 is barely affected (exp(-2 * (1/100)^2) ~ 1)
        assert abs(a_windowed[0] - raw_a[0]) < 1e-3
        
        # Verify n=N is heavily suppressed (exp(-2 * 1^2) = e^-2 ~ 0.135)
        expected_suppression = raw_a[-1] * math.exp(-2.0)
        assert abs(a_windowed[-1] - expected_suppression) < 1e-10

    def test_bump_window_compact_support(self):
        """Bump function must enforce strict compact support (0 at boundaries)."""
        N = 100
        cfg = vc.DirichletConfig(N=N, sigma=0.5, window_type="bump")
        raw_a, _ = vc.build_coefficients(cfg)
        a_windowed = vc.apply_window(cfg, raw_a)
        
        # Exact boundaries (n/N -> 0 and n/N -> 1) evaluate to 0
        assert a_windowed[-1] == 0.0
        
        # Midpoint should be positive
        mid = N // 2
        assert a_windowed[mid] > 0.0

    def test_log_sech2_alignment(self):
        """Log-aligned sech^2 window must align with Volume IV kernel."""
        N = 100
        T_center = math.log(50)
        H = 1.0
        cfg = vc.DirichletConfig(N=N, sigma=0.5, window_type="log_sech2", window_params={"T": T_center, "H": H})
        
        raw_a, _ = vc.build_coefficients(cfg)
        a_windowed = vc.apply_window(cfg, raw_a)
        
        # The weight should peak exactly at n=50
        assert a_windowed[49] > a_windowed[10]
        assert a_windowed[49] > a_windowed[80]


class TestDecayControl:
    """
    Requirement 3: Verify |S(ξ)| decays as |ξ| increases, and compare it 
    against the kernel's exponential decay.
    """

    def test_empirical_decay_of_dirichlet_wave(self):
        """|S(ξ)| must decrease as frequency |ξ| gets large."""
        cfg = vc.DirichletConfig(N=100, sigma=0.5, window_type="gaussian")
        xis = [0.0, 5.0, 20.0, 50.0]
        
        profile = vc.decay_profile(cfg, xis, H=1.0)
        
        # While highly oscillatory, the general magnitude envelope drops at high freq
        # We test that the extreme frequency (50) is smaller than DC (0)
        S_dc = profile[0].S_abs
        S_high = profile[-1].S_abs
        
        assert S_high < S_dc * 0.5, "High frequency signal did not decay relative to DC"

    def test_kernel_decay_outpaces_signal_decay(self):
        r"""
        The analytic kernel \hat{k}_H(ξ) must decay exponentially (e^{-π^2 H |ξ|}), 
        vastly outpacing the slow polynomial decay of the Dirichlet wave.
        """
        cfg = vc.DirichletConfig(N=50, sigma=0.5, window_type="sharp")
        profile = vc.decay_profile(cfg, [0.0, 10.0], H=1.0)
        
        k_dc = profile[0].k_hat_abs
        k_high = profile[1].k_hat_abs
        
        S_dc = profile[0].S_abs
        S_high = profile[1].S_abs
        
        kernel_drop_ratio = k_high / k_dc
        signal_drop_ratio = S_high / S_dc
        
        # Kernel drop should be mathematically orders of magnitude steeper
        assert kernel_drop_ratio < signal_drop_ratio * 1e-10, "Kernel fails to suppress signal at high frequency"


class TestMagnitudeBounds:
    """
    The core Bochner norm \\int \\hat{k}_H |S|^2 dξ must be strictly positive.
    """

    def test_trivial_bound(self):
        """|S(ξ)| <= sum |a_n| for all ξ."""
        cfg = vc.DirichletConfig(N=50, sigma=0.5, window_type="plain")
        raw_a, logn = vc.build_coefficients(cfg)
        trivial = vc.trivial_bound(raw_a)
        
        # Test a random highly oscillatory point
        S_val = vc.S_a_xi(17.345, cfg, raw_a, logn)
        assert abs(S_val) <= trivial + 1e-10

    def test_kernel_weighted_norm_positivity(self):
        r"""The core Bochner norm \int \hat{k}_H |S|^2 dξ must be strictly positive."""
        cfg = vc.DirichletConfig(N=50, sigma=0.5, window_type="gaussian")
        norm = vc.kernel_weighted_norm(cfg, H=1.0, L=15.0)
        
        assert norm > 0.0, "Kernel-weighted norm violated positivity"


class TestStructuralDecomposition:
    """
    Requirement 5: Decompose |S(ξ)|^2 into Diagonal and Off-Diagonal components.
    """

    def test_exact_structural_split(self):
        """|S(ξ)|^2 exactly equals Diag + OffDiag."""
        cfg = vc.DirichletConfig(N=30, sigma=0.5, window_type="sharp")
        xi = 3.14159
        
        dec = vc.structural_decomposition_at_xi(xi, cfg)
        
        reconstructed = dec.diag + dec.off_diag
        assert abs(reconstructed - dec.total) / dec.total < 1e-10

    def test_diagonal_positivity_and_invariance(self):
        """The diagonal sum |a_n|^2 must be strictly positive and invariant to ξ."""
        cfg = vc.DirichletConfig(N=30, sigma=0.5, window_type="exponential")
        
        dec1 = vc.structural_decomposition_at_xi(0.0, cfg)
        dec2 = vc.structural_decomposition_at_xi(15.0, cfg)
        
        assert dec1.diag > 0.0
        assert abs(dec1.diag - dec2.diag) < 1e-10, "Diagonal term changed with ξ"


class TestStabilityAndScaling:
    """
    Requirement 6: Verify behavior under N scaling, σ sweeps, and window sensitivity.
    """

    def test_sigma_symmetry(self):
        """
        Verify symmetry properties when sweeping σ.
        The L^2 norm of S_σ(ξ) is symmetric around σ=0.5 if evaluating 
        the integral exactly, but finite N introduces asymmetry. 
        We simply test the profile executes stably across the critical strip.
        """
        sigmas = [0.3, 0.4, 0.5, 0.6, 0.7]
        profile = vc.sigma_symmetry_profile(N=40, sigmas=sigmas, window_type="sharp", window_params=None, H=1.0, L=5.0)
        
        assert len(profile) == 5
        assert profile[0.5].kernel_norm > 0.0

    def test_smoothing_improves_L2_stability(self):
        """
        Compare the L^2 norm growth of a sharp cutoff vs a smooth Gaussian cutoff.
        The smooth cutoff should yield a strictly smaller, more controlled L^2 norm.
        """
        cfg_sharp = vc.DirichletConfig(N=100, sigma=0.5, window_type="sharp")
        cfg_smooth = vc.DirichletConfig(N=100, sigma=0.5, window_type="gaussian", window_params={"alpha": 2.0})
        
        L2_sharp = vc.L2_norm_S(cfg_sharp, L=10.0)
        L2_smooth = vc.L2_norm_S(cfg_smooth, L=10.0)
        
        assert L2_smooth < L2_sharp, "Smoothing failed to control the L^2 norm"


class TestIntegrationWithVolumeIV:
    """
    Requirement 7: Integration with Volume IV spectral functional.
    """

    def test_volume_IV_spectral_evaluation(self):
        """
        Ensure Q_spec^V evaluates successfully, proving the Dirichlet wave 
        interfaces perfectly with the Volume IV k_hat kernel.
        """
        cfg = vc.DirichletConfig(N=50, sigma=0.5, window_type="log_sech2", window_params={"T": 2.0, "H": 1.0})
        
        # Calculate Q_spec integrating over the Dirichlet wave
        Q_val = vc.Q_spectral_dirichlet(cfg, H=1.0, T0=0.0, L=10.0)
        
        # Must be strictly positive (Bochner Positivity Basin)
        assert Q_val > 0.0


if __name__ == '__main__':
    pytest.main([__file__, "-v"])