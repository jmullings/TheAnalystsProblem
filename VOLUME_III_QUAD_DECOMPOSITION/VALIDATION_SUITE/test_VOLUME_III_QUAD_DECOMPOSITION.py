#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# test_QuadraticFormDecomposition.py
#
# VALIDATION SUITE FOR VOLUME III: QUADRATIC FORM DECOMPOSITION
# Tests structural identities, matrix decomposition, symmetric/antisymmetric 
# splits, and growth limits for the Analyst's Problem.

import sys
import os
import math
import numpy as np
import mpmath as mp
import pytest

# Inject the proof directory into sys.path
PROOF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'VOLUME_III_QUAD_DECOMPOSITION_PROOF'))
sys.path.insert(0, PROOF_DIR)

try:
    import VOLUME_III_QUAD_DECOMPOSITION as qf
except ImportError:
    # If the file was saved with a different name, adjust here
    import VOLUME_III_QUAD_DECOMPOSITION as qf

# Configure precision for rigorous verification
mp.mp.dps = 80
TOL = 1e-12

class TestAlgebraicIdentities:
    """
    Requirement 1: Algebraic log identities (symbolic + numeric checks).
    """

    def test_log_identities_symbolic_and_numeric(self):
        """Verify the exactness of the algebraic identities."""
        result = qf.verify_algebraic_identities(samples=100)
        
        # 1. Log difference representation
        assert result.log_diff_ok, "log(m) - log(n) = log(m/n) failed"
        
        # 2. Square decomposition
        assert result.square_decomp_ok, "(log m - log n)^2 identity failed"
        
        # 3. Hard Algebraic Identity (HAI)
        assert result.hard_identity_ok, "4(log m)(log n) + (log m - log n)^2 = (log mn)^2 failed"
        
        # 4. Symmetric / Antisymmetric split
        assert result.sym_antisym_ok, "Symmetric/Antisymmetric decomposition failed"

class TestKernelStructure:
    """
    Requirement 2: Kernel k_H(t) = 6/H^2 * sech^4(t/H) and its Toeplitz structure.
    """

    @pytest.mark.parametrize("H", [0.5, 1.0, 3.14])
    def test_kernel_positivity_and_decay(self, H):
        """Kernel must be strictly positive and exhibit exponential decay."""
        H_mp = mp.mpf(H)
        
        # Positivity
        for t in np.linspace(-10, 10, 50):
            val = qf.k_H(mp.mpf(t), H_mp)
            assert val > 0, f"Kernel negativity detected at t={t}"
            
        # Decay properties
        checks = qf.verify_kernel_properties(H=H)
        assert checks.decay_ok, f"Kernel exponential decay failed for H={H}"

    @pytest.mark.parametrize("H", [0.1, 1.0, 2.5])
    def test_kernel_symmetry(self, H):
        """Kernel must be perfectly symmetric k_H(t) == k_H(-t)."""
        checks = qf.verify_kernel_properties(H=H)
        assert checks.symmetry_ok, f"Kernel symmetry failed for H={H}"

class TestQuadraticFormMatrices:
    """
    Requirement 3 & 4 & 5: Quadratic form Q_H(x; T0) over x_n = n^{-1/2},
    Diagonal / off-diagonal decomposition, and Matrix Symmetry.
    """

    def test_matrix_construction_and_symmetry(self):
        """Validate K_mn and A_mn matrix symmetries."""
        cfg = qf.QuadraticFormConfig(N=50, H=1.0, T0=14.1347)
        mats = qf.build_quadratic_form(cfg)
        
        # Requirement 5: Symmetry checks
        sym_check = qf.check_matrix_symmetry(mats)
        assert sym_check.K_sym_ok, f"Kernel matrix K asymmetric: max error {sym_check.max_K_asym}"
        assert sym_check.A_sym_ok, f"Weighted matrix A asymmetric: max error {sym_check.max_A_asym}"
        
        # Explicit test of the phase component symmetry
        # P_mn = cos(T0 * (log m - log n))
        # Since cos is even, P_mn = P_nm
        assert np.allclose(mats.P, mats.P.T, atol=TOL)

    def test_diagonal_offdiagonal_decomposition(self):
        """Verify Q_H = D_H + O_H perfectly."""
        cfg = qf.QuadraticFormConfig(N=100, H=2.0, T0=0.0)
        mats = qf.build_quadratic_form(cfg)
        
        assert abs(mats.Q_H - (mats.D_H + mats.O_H)) < TOL, "Q_H decomposition failed"
        
        # Validate D_H computation against exact harmonic sum
        k_0 = float(qf.k_H(mp.mpf("0"), mp.mpf(2.0)))
        harmonic_sum = sum(1.0 / n for n in range(1, 101))
        expected_D_H = k_0 * harmonic_sum
        
        assert abs(mats.D_H - expected_D_H) < TOL, "Diagonal term D_H incorrectly computed"

    @pytest.mark.parametrize("T0", [0.0, 14.1347, 50.0])
    def test_phase_matrix_impact(self, T0):
        """Check that T0 modulates the off-diagonal terms but preserves the diagonal."""
        cfg_0 = qf.QuadraticFormConfig(N=30, H=1.0, T0=0.0)
        cfg_T = qf.QuadraticFormConfig(N=30, H=1.0, T0=T0)
        
        mats_0 = qf.build_quadratic_form(cfg_0)
        mats_T = qf.build_quadratic_form(cfg_T)
        
        # Diagonal must remain invariant under T0
        assert abs(mats_0.D_H - mats_T.D_H) < TOL, "Diagonal term affected by T0 phase"
        
        # Off-diagonal should change (unless T0 = 0)
        if T0 != 0.0:
            assert abs(mats_0.O_H - mats_T.O_H) > TOL, "Off-diagonal term unaffected by T0 phase"

class TestGrowthDiagnostics:
    """
    Requirement 6 & 7: Growth diagnostics in N and H,
    and Numerical experiments for D_H, O_H, Q_H.
    """

    def test_harmonic_number_approximation(self):
        """Verify the harmonic_number helper function."""
        # Exact for small N
        exact = sum(1.0 / n for n in range(1, 11))
        assert abs(qf.harmonic_number(10) - exact) < TOL
        
        # Approximate for large N
        approx = qf.harmonic_number(100000)
        expected = math.log(100000) + 0.57721566490153286
        assert abs(approx - expected) < 1e-4

    def test_parameter_sweep_execution(self):
        """Ensure parameter sweep executes correctly across grid."""
        Ns = (10, 20)
        Hs = (0.5, 1.0)
        sweep = qf.parameter_sweep(Ns, Hs, T0=0.0)
        
        assert len(sweep.records) == 4
        
        # Verify specific structural properties of the sweep
        for (N, H), diag in sweep.records.items():
            assert diag.N == N
            assert diag.H == H
            assert diag.D_H > 0
            
            # Theoretical diagonal vs Computed diagonal
            assert abs(diag.D_H - diag.diag_theory) < TOL

    @pytest.mark.xfail(reason="Analyst's Problem open: Off-diagonal growth currently dominates diagonal at finite N.")
    def test_diagonal_dominance(self):
        """
        The critical inequality of the Analyst's Problem:
        Prove D_H > |O_H| globally.
        
        This test is explicitly marked to XFAIL (Expected Fail) because
        the raw un-smoothed off-diagonal terms currently outgrow the 
        log(N) diagonal at finite N.
        
        Closing this gap requires the Large Sieve / Euler-Maclaurin 
        machinery slated for Volumes VI and VII.
        """
        cfg = qf.QuadraticFormConfig(N=200, H=1.0, T0=0.0)
        mats, diag = qf.analyse_growth(cfg)
        
        assert diag.ratio_D_to_absO >= 1.0, f"Dominance failure: D_H = {diag.D_H}, |O_H| = {abs(diag.O_H)}"

if __name__ == '__main__':
    pytest.main([__file__, "-v"])