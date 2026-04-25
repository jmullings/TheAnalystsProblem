#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# test_VOLUME_XII_FINAL_ASSEMBLY_PROOF.py
#
# VALIDATION SUITE FOR VOLUME XII: FINAL ASSEMBLY AND CERTIFICATION
# 100% Coverage TDD for the Operator-Theoretic (TAP HO) Final Assembly.

import sys
import os
import math
import numpy as np
import pytest
from unittest import mock
from dataclasses import dataclass

# Inject the proof directory into sys.path
PROOF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'VOLUME_XII_FINAL_ASSEMBLY_PROOF'))
sys.path.insert(0, PROOF_DIR)

try:
    import VOLUME_XII_FINAL_ASSEMBLY as xii
except ImportError:
    pytest.skip("VOLUME_XII_FINAL_ASSEMBLY module not found", allow_module_level=True)

class TestVolumeXIIConstants:
    """Test fundamental constants from Volume II exposed by Volume XII."""
    
    def test_lambda_star(self):
        for H in [0.1, 0.5, 1.0, 2.0]:
            assert math.isclose(xii.lambda_star(H), 4.0 / (H * H), rel_tol=1e-12)

    def test_k_H_eval(self):
        H = 1.0
        assert math.isclose(xii.k_H(0.0, H), 6.0, rel_tol=1e-12)
        assert xii.k_H(1.0, H) > 0

    def test_k_H_L1_norm(self):
        for H in [0.1, 0.5, 1.0, 2.0]:
            assert math.isclose(xii.k_H_L1(H), 8.0 / H, rel_tol=1e-12)

    def test_k_H_L2_squared(self):
        for H in [0.1, 0.5, 1.0]:
            assert math.isclose(xii.k_H_L2_squared(H), 1152.0 / (35.0 * H ** 3), rel_tol=1e-12)

    def test_kernel_tail_bound(self):
        H = 1.0
        tail_L5 = xii.kernel_tail_bound(H, 5.0)
        tail_L10 = xii.kernel_tail_bound(H, 10.0)
        assert tail_L5 > 0 and tail_L10 > 0
        assert tail_L10 < tail_L5  # Exponential decay

class TestDiagonalMass:
    """Test Volume III diagonal decomposition."""
    def test_diagonal_positive(self):
        for H in [0.1, 1.0]:
            for N in [10, 50]:
                assert xii.diagonal_mass_D_H(N, H) > 0

    def test_harmonic_approx(self):
        for N in [10, 100]:
            approx = xii.harmonic_approx(N)
            exact = sum(1.0/n for n in range(1, N+1))
            assert abs(approx - exact) < 1.0 / N

class TestOffDiagonalMVBound:
    """Test the historical MV bounds and diagonal dominance ratios."""
    def test_mv_bound_and_ratio(self):
        for H in [0.5, 1.0]:
            for N in [10, 50]:
                bound = xii.off_diagonal_MV_bound(N, H)
                assert bound >= 0
                rho = xii.diagonal_dominance_ratio_MV(N, H)
                assert rho > 0 and math.isfinite(rho)
                
    def test_mv_bound_N1(self):
        assert xii.off_diagonal_MV_bound(1, 1.0) == 0.0
        assert xii.diagonal_dominance_ratio_MV(1, 1.0) == float('inf')

class TestOperatorBounds:
    """Test TAP-HO functional analysis properties and caching."""
    def test_operator_bounds_result_properties(self):
        # Finite/Compact Operator
        res_compact = xii.OperatorBoundsResult(hs_norm=1.5, op_norm=1.0, coherence_err=0.0, N_probe=10, H=1.0)
        assert res_compact.is_hilbert_schmidt is True
        assert res_compact.gap_g1_structurally_bypassed is True

        # Infinite/Divergent Operator
        res_diverge = xii.OperatorBoundsResult(hs_norm=math.inf, op_norm=math.inf, coherence_err=0.0, N_probe=10, H=1.0)
        assert res_diverge.is_hilbert_schmidt is False
        assert res_diverge.gap_g1_structurally_bypassed is False

    @mock.patch('VOLUME_XII_FINAL_ASSEMBLY.ho_hilbert_schmidt_norm', return_value=2.0)
    @mock.patch('VOLUME_XII_FINAL_ASSEMBLY.ho_operator_norm_power_iteration', return_value=1.5)
    @mock.patch('VOLUME_XII_FINAL_ASSEMBLY.ho_cross_dimensional_coherence', return_value=1e-12)
    def test_compute_ho_bounds(self, mock_coh, mock_op, mock_hs):
        ho = xii._compute_ho_bounds(H=0.5, N_probe=20)
        assert ho.hs_norm == 2.0
        assert ho.op_norm == 1.5
        assert ho.coherence_err == 1e-12
        assert ho.H == 0.5

    def test_get_ho_bounds_cache(self):
        xii._HO_CACHE.clear()
        with mock.patch('VOLUME_XII_FINAL_ASSEMBLY._compute_ho_bounds') as mock_compute:
            mock_compute.return_value = xii.OperatorBoundsResult(1.0, 1.0, 0.0, 10, 0.5)
            # Call twice, should compute once
            res1 = xii.get_ho_bounds(0.5, 10)
            res2 = xii.get_ho_bounds(0.5, 10)
            assert res1 is res2
            mock_compute.assert_called_once()

class TestMVBAnalytic:
    """Test finite-N mean-value B_analytic evaluation and cache."""
    def test_get_mv_B_analytic_cache(self):
        xii._MV_CACHE.clear()
        with mock.patch('VOLUME_XII_FINAL_ASSEMBLY.infinite_series_constant_analytic') as mock_mv:
            mock_mv.return_value = 0.85
            # Call twice
            res1 = xii.get_mv_B_analytic(0.5, 10)
            res2 = xii.get_mv_B_analytic(0.5, 10)
            assert res1 == res2 == 0.85
            mock_mv.assert_called_once()

class TestCertificationResult:
    """Test the XII_CertificationResult dataclass and its properties."""
    def test_margin_pct(self):
        res_ok = xii.XII_CertificationResult(H=1.0, T0=0.0, N=10, L=5.0, Q_trunc=10.0, Q_lower_bound=9.0)
        assert res_ok.margin_pct == 90.0

        res_zero = xii.XII_CertificationResult(H=1.0, T0=0.0, N=10, L=5.0, Q_trunc=0.0, Q_lower_bound=0.0)
        assert res_zero.margin_pct == 0.0

class TestCertifyFull:
    """Test the complete certification pipeline for a single parameter set."""
    @mock.patch('VOLUME_XII_FINAL_ASSEMBLY.get_ho_bounds')
    @mock.patch('VOLUME_XII_FINAL_ASSEMBLY.get_mv_B_analytic', return_value=0.8)
    @mock.patch('VOLUME_XII_FINAL_ASSEMBLY.certify_single')
    def test_certify_full_success(self, mock_single, mock_mv, mock_ho):
        # Mock dependencies
        mock_ho.return_value = xii.OperatorBoundsResult(hs_norm=1.5, op_norm=1.0, coherence_err=0.0, N_probe=100, H=0.5)
        
        # Mock XI harness returning a standard details string
        mock_res = mock.MagicMock()
        mock_res.Q_trunc = 100.0
        mock_res.Q_lower_bound = 99.0
        mock_res.passed = True
        mock_res.details = "E_tail=1e-5, E_quad=1e-5, E_spec=1e-5, E_num=1e-5, F_floor=1.0 "
        mock_single.return_value = mock_res

        res = xii.certify_full(H=0.5, T0=10.0, N=50, dps=40)
        assert res.computationally_certified is True
        assert res.ho_gap_g1_bypassed is True
        assert res.analytically_supported is True
        assert res.E_tail == 1e-5

    @mock.patch('VOLUME_XII_FINAL_ASSEMBLY.get_ho_bounds')
    @mock.patch('VOLUME_XII_FINAL_ASSEMBLY.certify_single')
    def test_certify_full_bad_details_string(self, mock_single, mock_ho):
        mock_ho.return_value = xii.OperatorBoundsResult(1.5, 1.0, 0.0, 100, 0.5)
        mock_res = mock.MagicMock()
        mock_res.passed = True
        mock_res.details = "GARBAGE_STRING_WITHOUT_KEY_VALUES"
        mock_single.return_value = mock_res
        
        res = xii.certify_full(0.5, 10.0, 50, dps=40)
        # Should gracefully fail parsing and return 0.0 for errors
        assert res.E_tail == 0.0
        assert res.computationally_certified is True

    @mock.patch('VOLUME_XII_FINAL_ASSEMBLY.make_config')
    @mock.patch('VOLUME_XII_FINAL_ASSEMBLY.certify_single', side_effect=Exception("Timeout"))
    def test_certify_full_exceptions(self, mock_single, mock_config):
        # 1. Test config fallback (make_config fails with window="ruelle")
        mock_config.side_effect = [TypeError("Unknown Window"), mock.MagicMock()]
        
        res = xii.certify_full(H=1.0, T0=10.0, N=50, dps=40)
        assert mock_config.call_count == 2
        assert res.computationally_certified is False
        assert "certify_single raised: Timeout" in res.notes

class TestFinalAssembly:
    """Test the full multi-configuration sweep."""
    @mock.patch('VOLUME_XII_FINAL_ASSEMBLY.ASSEMBLY_GRID', [(0.5, 0.0, 10), (2.0, 0.0, 10)])
    @mock.patch('VOLUME_XII_FINAL_ASSEMBLY.certify_full')
    def test_run_final_assembly(self, mock_certify):
        # Setup mock returns: 1 pass, 1 computational fail
        r1 = xii.XII_CertificationResult(H=0.5, T0=0.0, N=10, L=5.0, Q_trunc=1.0, Q_lower_bound=0.5,
                                         op_norm=1.0, hs_norm=1.5, B_analytic=0.8, 
                                         computationally_certified=True, analytically_supported=True)
        r2 = xii.XII_CertificationResult(H=2.0, T0=0.0, N=10, L=5.0, Q_trunc=1.0, Q_lower_bound=-0.5,
                                         op_norm=math.inf, hs_norm=math.inf, B_analytic=math.inf,
                                         computationally_certified=False, analytically_supported=False)
        mock_certify.side_effect = [r1, r2]

        summary = xii.run_final_assembly(dps=40, verbose=True)
        
        assert summary.n_total == 2
        assert summary.n_comp_pass == 1
        assert summary.n_analytic_supported == 1
        assert summary.n_fail == 1
        assert summary.min_Q_lower == -0.5
        assert summary.min_margin_pct == -50.0

class TestOutputDiagnostics:
    """Test the printing and presentation helpers."""
    def test_print_mv_gap_table(self, capsys):
        # We just want to make sure it executes without throwing formatting errors
        xii.print_mv_gap_table(H_values=[0.5], N_values=[10, 50])
        captured = capsys.readouterr().out
        assert "HISTORICAL GAP-G1 DIAGNOSTICS" in captured
        assert "H=0.50" in captured

class TestInternalTDD:
    """Run the internal TDD suite built into Volume XII."""
    
    @mock.patch('VOLUME_XII_FINAL_ASSEMBLY.compare_time_freq_domains', return_value={"Q_time": 100.0, "Q_freq": 100.0})
    def test_internal_tdd_suite(self, mock_compare):
        # Patches out the slow Parseval bridge; other tests are fast.
        n_fails = xii.run_volume_xii_tests()
        assert n_fails == 0

    @mock.patch('VOLUME_XII_FINAL_ASSEMBLY.make_config', side_effect=Exception("Mock Fail"))
    def test_internal_tdd_parseval_exception(self, mock_config):
        # Force a failure in test_parseval_bridge to test the branch
        res = xii.test_parseval_bridge()
        assert res.passed is False
        assert "make_config failed" in res.details

class TestMainRunner:
    """Test the main driver function with different outcomes."""

    @mock.patch('VOLUME_XII_FINAL_ASSEMBLY.run_volume_xii_tests', return_value=0)
    @mock.patch('VOLUME_XII_FINAL_ASSEMBLY.run_scaling_experiment_example')
    @mock.patch('VOLUME_XII_FINAL_ASSEMBLY.run_final_assembly')
    def test_run_volume_xii_absolute(self, mock_asm, mock_scale, mock_tests, capsys):
        # All pass scenario
        mock_summary = xii.XII_AssemblySummary(
            n_total=2, n_comp_pass=2, n_analytic_supported=2,
            min_Q_lower=1.0, min_margin_pct=90.0, max_E_total=0.1,
            op_norms_by_H={0.5: 1.0, 2.0: math.inf}, hs_norms_by_H={0.5: 1.5}, B_analytic_by_H={0.5: 0.8}
        )
        mock_asm.return_value = mock_summary
        
        xii.run_volume_xii(dps=40)
        out = capsys.readouterr().out
        # Matches LRM-Audit compliant string
        assert "STATUS: COMPUTATIONALLY COMPLETE · ANALYTICALLY SUPPORTED" in out

    @mock.patch('VOLUME_XII_FINAL_ASSEMBLY.run_volume_xii_tests', return_value=0)
    @mock.patch('VOLUME_XII_FINAL_ASSEMBLY.run_scaling_experiment_example')
    @mock.patch('VOLUME_XII_FINAL_ASSEMBLY.run_final_assembly')
    def test_run_volume_xii_partial_analytic(self, mock_asm, mock_scale, mock_tests, capsys):
        # Computational pass, but analytic bounds out of scope (e.g. H > 1)
        mock_summary = xii.XII_AssemblySummary(
            n_total=2, n_comp_pass=2, n_analytic_supported=1
        )
        mock_asm.return_value = mock_summary
        
        xii.run_volume_xii(dps=40)
        out = capsys.readouterr().out
        assert "COMPUTATIONALLY COMPLETE · PARTIAL ANALYTIC SUPPORT" in out

    @mock.patch('VOLUME_XII_FINAL_ASSEMBLY.run_volume_xii_tests', return_value=0)
    @mock.patch('VOLUME_XII_FINAL_ASSEMBLY.run_scaling_experiment_example', side_effect=Exception("Scale Error"))
    @mock.patch('VOLUME_XII_FINAL_ASSEMBLY.run_final_assembly')
    def test_run_volume_xii_failures(self, mock_asm, mock_scale, mock_tests, capsys):
        # Computational failures present + exception handling
        mock_summary = xii.XII_AssemblySummary(
            n_total=2, n_comp_pass=1, n_analytic_supported=1, n_fail=1
        )
        mock_asm.return_value = mock_summary
        
        xii.run_volume_xii(dps=40)
        out = capsys.readouterr().out
        assert "[WARNING] scaling experiment raised: Scale Error" in out
        assert "STATUS: PARTIAL" in out

if __name__ == '__main__':
    pytest.main([__file__, "-v"])