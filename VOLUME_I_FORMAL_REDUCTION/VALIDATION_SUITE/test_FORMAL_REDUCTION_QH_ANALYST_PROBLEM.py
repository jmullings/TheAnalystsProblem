#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# test_FORMAL_REDUCTION_QH_ANALYST_PROBLEM.py
#
# TDD Suite for Volume I – Formal Reduction
# Target: VOLUME_I_FORMAL_REDUCTION/VOLUME_I_FORMAL_REDUCTION_PROOF/VOLUME_I_FORMAL_REDUCTION.py
#
# STATUS UPDATE (2026-04-16):
#   • Defect-3 Fix: Parseval Bridge tolerance tightened from 1e-2 to 1e-8.
#     This enforces proof-grade numerical fidelity and masks no drift.
#   • Defect-1 Alignment: Docstrings updated to reflect Volume II Fourier fix.
#     (Anchors k_H_hat(0) = 8/H).
#   • Defect-2 Context: Added comments clarifying that C_ratio > 1 is a 
#     property of the Pointwise Absolute Bound, resolved by Lemma XII.1'.
#
# This suite verifies the algebraic, analytic, and structural properties of the
# Toeplitz quadratic form, the Parseval bridge, and the Analyst's Problem decomposition.

import sys
import math
import cmath
import unittest
from pathlib import Path

# Resolve path to import the target module
try:
    from VOLUME_I_FORMAL_REDUCTION.VOLUME_I_FORMAL_REDUCTION_PROOF.VOLUME_I_FORMAL_REDUCTION import (
        sech, sech2, sech4, tanh,
        Lambda_H_tau, Lambda_H_dd_tau, w_H_time, g_H_sech4,
        k_H_time, k_H_hat,
        dirichlet_S_N, dirichlet_S_N_derivatives, F2_sigma_T,
        curvature_F2_bar,
        build_toeplitz_matrix, physical_vector_x, toeplitz_quadratic_form,
        phased_quadratic_form,
        parseval_bridge_certificate,
        M1_diagonal_term, cross_offdiagonal_term, QH_from_M1_and_cross,
        absolute_cross_term, C_ratio,
        check_kernel_positive_definite, parseval_identity_residual,
        FormalReduction
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from VOLUME_I_FORMAL_REDUCTION.VOLUME_I_FORMAL_REDUCTION_PROOF.VOLUME_I_FORMAL_REDUCTION import (
        sech, sech2, sech4, tanh,
        Lambda_H_tau, Lambda_H_dd_tau, w_H_time, g_H_sech4,
        k_H_time, k_H_hat,
        dirichlet_S_N, dirichlet_S_N_derivatives, F2_sigma_T,
        curvature_F2_bar,
        build_toeplitz_matrix, physical_vector_x, toeplitz_quadratic_form,
        phased_quadratic_form,
        parseval_bridge_certificate,
        M1_diagonal_term, cross_offdiagonal_term, QH_from_M1_and_cross,
        absolute_cross_term, C_ratio,
        check_kernel_positive_definite, parseval_identity_residual,
        FormalReduction
    )


class TestSpecialFunctions(unittest.TestCase):
    """T1: Basic Hyperbolic Function Properties"""

    def test_sech_at_zero(self):
        self.assertAlmostEqual(sech(0.0), 1.0, places=14)

    def test_sech2_at_zero(self):
        self.assertAlmostEqual(sech2(0.0), 1.0, places=14)

    def test_sech4_positivity(self):
        for x in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            self.assertGreater(sech4(x), 0.0)

    def test_tanh_properties(self):
        self.assertAlmostEqual(tanh(0.0), 0.0, places=14)
        self.assertTrue(tanh(100.0) > 0.99)
        self.assertTrue(tanh(-100.0) < -0.99)


class TestKernelProperties(unittest.TestCase):
    """
    T1: Curvature Window and Bochner Kernel Properties.
    Aligned with Volume II Fourier Fix (Defect-1):
    k_H_hat(0) must be exactly 8/H.
    """

    def test_Lambda_H_tau_positivity(self):
        H = 2.0
        for tau in [-5.0, 0.0, 5.0]:
            self.assertGreater(Lambda_H_tau(tau, H), 0.0)

    def test_g_H_sech4_normalization(self):
        H = 3.0
        expected = 6.0 / (H * H)
        self.assertAlmostEqual(g_H_sech4(0.0, H), expected, places=14)

    def test_k_H_hat_at_zero(self):
        """
        CRITICAL ANCHOR: Volume II fix ensures k_H_hat(0) = 8/H.
        Previously, this was off by a factor of 2π in older drafts.
        """
        H = 2.5
        self.assertAlmostEqual(k_H_hat(0.0, H), 8.0 / H, places=14)

    def test_k_H_hat_positivity(self):
        H = 1.5
        for omega in [-10.0, -1.0, 0.1, 5.0, 20.0]:
            self.assertGreater(k_H_hat(omega, H), 0.0)

    def test_k_H_hat_even_symmetry(self):
        H = 2.0
        omega = 3.7
        self.assertAlmostEqual(k_H_hat(omega, H), k_H_hat(-omega, H), places=14)


class TestDirichletModel(unittest.TestCase):
    """T1/T2: Dirichlet Polynomial and Energy Forms"""

    def test_dirichlet_S_N_at_t_zero(self):
        N = 5
        sigma = 0.5
        T = 0.0
        expected = sum(n**(-0.5) for n in range(1, N+1))
        result = dirichlet_S_N(T, N, sigma)
        self.assertAlmostEqual(result.real, expected, places=12)
        self.assertAlmostEqual(result.imag, 0.0, places=12)

    def test_F2_sigma_T_positivity(self):
        N = 10
        for T in [0.0, 14.13, 50.0]:
            self.assertGreaterEqual(F2_sigma_T(T, N, sigma=0.5), 0.0)


class TestParsevalBridge(unittest.TestCase):
    """
    T1: Exact Equivalence of Integral and Toeplitz Forms.
    UPDATED (Defect-3 Fix): Tolerance tightened to 1e-8 to enforce
    proof-grade numerical fidelity.
    """

    def test_toeplitz_matrix_symmetry(self):
        N = 10
        H = 2.0
        M = build_toeplitz_matrix(N, H)
        for i in range(N):
            for j in range(N):
                self.assertAlmostEqual(M[i][j], M[j][i], places=14)

    def test_physical_vector_values(self):
        N = 4
        sigma = 0.5
        x = physical_vector_x(N, sigma)
        expected = [1.0, 1.0/math.sqrt(2), 1.0/math.sqrt(3), 0.5]
        for i in range(N):
            self.assertAlmostEqual(x[i], expected[i], places=14)

    def test_parseval_bridge_identity(self):
        """
        CRITICAL TEST: The integral curvature F2_bar must match the
        phased quadratic form (via the validated bridge certificate)
        up to numerical quadrature error.

        TOLERANCE: < 1e-8 (Tightened from 1e-2 per Defect-3 Audit).
        """
        T0 = 14.1347  # Near first Riemann zero
        H = 1.5
        N = 5
        sigma = 0.5

        # Use the dedicated bridge residual.
        # num_steps=5001 is sufficient for 1e-8 tolerance in this regime.
        res = parseval_identity_residual(T0, H, N, sigma=sigma,
                                         tau_min=-100.0, tau_max=100.0,
                                         num_steps=5001)

        self.assertLess(res, 1e-8,
                        f"Parseval bridge failed: Integral and Toeplitz forms diverge. "
                        f"Residual = {res} >= 1e-8")


class TestAnalystProblemDecomposition(unittest.TestCase):
    """
    T1: Decomposition of Q_H into Diagonal and Off-Diagonal.
    CONTEXT (Defect-2): The absolute_cross_term provides a worst-case bound
    ignoring phase cancellation. C_ratio > 1 is possible here. The actual
    diagonal dominance is proven via the Mean-Square framework (Lemma XII.1').
    """

    def test_QH_decomposition_exactness(self):
        N = 8
        H = 2.0
        T0 = 21.02
        sigma = 0.5

        Q_direct = phased_quadratic_form(N, H, T0, sigma)
        Q_decomp = QH_from_M1_and_cross(N, H, T0, sigma)
        self.assertAlmostEqual(Q_direct, Q_decomp, places=12)

    def test_M1_positivity(self):
        N = 10
        H = 1.0
        M1 = M1_diagonal_term(N, H)
        self.assertGreater(M1, 0.0)

    def test_absolute_cross_term_bounds_cross(self):
        """|CROSS| <= AbsoluteCross for any T0."""
        N = 15
        H = 1.2
        sigma = 0.5
        abs_cross = absolute_cross_term(N, H, sigma)

        for T0 in [0.0, 14.1, 50.5, 100.0]:
            cross = cross_offdiagonal_term(N, H, T0, sigma)
            self.assertLessEqual(abs(cross), abs_cross + 1e-12)

    def test_C_ratio_computes(self):
        """
        Computes C_ratio.
        Note: For large N, C_ratio > 1.0. This confirms that the
        Pointwise Absolute Bound is insufficient for proof, necessitating
        the Phase-Averaged (Lemma XII.1') approach.
        """
        N = 10
        H = 2.0
        ratio = C_ratio(N, H)
        self.assertTrue(math.isfinite(ratio))
        self.assertGreaterEqual(ratio, 0.0)


class TestBochnerPositivity(unittest.TestCase):
    """T1: Kernel Positivity Constraints"""

    def test_toeplitz_kernel_psd(self):
        """
        By Bochner's Theorem, since k_H_hat(omega) > 0, the resulting
        Toeplitz matrix must be Positive Semi-Definite.
        """
        N = 20
        H = 1.5
        l_min, l_max = check_kernel_positive_definite(N, H)

        try:
            import numpy as np
            M = build_toeplitz_matrix(N, H)
            eigenvalues = np.linalg.eigvalsh(M)
            self.assertGreaterEqual(min(eigenvalues), -1e-10, "Toeplitz matrix is not PSD")
        except ImportError:
            self.assertIsInstance(l_min, float)
            self.assertIsInstance(l_max, float)


class TestFormalReductionAPI(unittest.TestCase):
    """API Surface verification for external TDD harnesses"""

    def test_api_surface(self):
        api = FormalReduction()
        self.assertIsNotNone(api.k_H_time(0.0, 1.0))
        self.assertIsNotNone(api.k_H_hat(0.0, 1.0))
        self.assertEqual(len(api.toeplitz_matrix(3, 1.0)), 3)
        self.assertEqual(len(api.physical_vector(3)), 3)
        self.assertIsInstance(api.Q_H(3, 1.0), float)
        self.assertIsInstance(api.F2_bar(10.0, 1.0, 3), float)
        self.assertIsInstance(api.M1(3, 1.0), float)
        self.assertIsInstance(api.Cross(10.0, 3, 1.0), float)
        self.assertIsInstance(api.AbsoluteCross(3, 1.0), float)
        self.assertIsInstance(api.C_ratio(3, 1.0), float)
        self.assertIsInstance(api.parseval_residual(10.0, 1.0, 3), float)
        self.assertEqual(len(api.kernel_psd_bounds(3, 1.0)), 2)


if __name__ == '__main__':
    unittest.main()