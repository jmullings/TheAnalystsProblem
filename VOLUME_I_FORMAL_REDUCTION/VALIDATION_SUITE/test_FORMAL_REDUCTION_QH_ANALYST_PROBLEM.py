#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# test_FORMAL_REDUCTION_QH_ANALYST_PROBLEM.py
#
# TDD Suite for Volume I – Formal Reduction
# Target: VOLUME_I_FORMAL_REDUCTION_PROOF/FORMAL_REDUCTION_QH_ANALYST_PROBLEM.py
#
# ============================================================================
# VERIFICATION LAYER V1.TDD — PROOF INFRASTRUCTURE (NOT ANALYTIC PROOF)
# ============================================================================
#
# ROLE: This module is a machine-checkable certificate that the Volume I
# computational engine faithfully implements the T1 identities proved in
# the manuscript. It is part of the formal proof apparatus as a verification
# layer, not as a substitute for analytic theorems.
#
# EPISTEMIC TAXONOMY FOR TESTS:
#   • T1 tests: Verify unconditional algebraic/numeric identities.
#               These are machine-verifiable certainties.
#   • T2 tests: Verify properties conditional on standard analytic inputs
#               (e.g., Parseval bridge via Weil explicit formula).
#   • T3 tests: Diagnostic checks for open inequalities; pass/fail informs
#               empirical support but does not constitute proof.
#
# WHAT THIS SUITE CERTIFIES:
#   ✓ Kernel implementation: k_H(t) = (6/H²) sech⁴(t/H) and normalization k̂_H(0) = 8/H
#   ✓ Parseval bridge: Integral and Toeplitz forms match to < 1e-8 residual
#   ✓ Toeplitz decomposition: Q_H = M1 + Cross holds exactly in finite arithmetic
#   ✓ Bochner positivity: Toeplitz kernel matrix is PSD (numerically)
#   ✓ Dirichlet polynomial: Correct implementation of D_N(σ, T)
#   ✓ API surface: FormalReduction class exposes required methods with correct types
#
# WHAT THIS SUITE DOES NOT PROVE:
#   ✗ The explicit formula representation EF.H (Volume IX)
#   ✗ The ΔA negativity lemma for off-critical zeros (Volume VIII)
#   ✗ The finite-N → ∞ convergence theorem 6.1′ (Volume X)
#   ✗ The equivalence RH ⇔ Q_H^∞ > 0 (Theorem 6.2)
#
# INTEGRATION:
#   • CI/CD: Suite must pass for any release tagged "proof-grade"
#   • ProofPipeline: Call VerificationLayer.run_engine_checks() to certify
#     that the underlying engine is faithful before evaluating analytic preconditions
#   • Versioning: Tag commits with v1.0-proof-grade only when tests pass AND
#     volumes are in sync with implemented identities
#
# STATUS UPDATE (2026-04-16):
#   • Defect-3 Fix: Parseval Bridge tolerance tightened from 1e-2 to 1e-8.
#     This enforces proof-grade numerical fidelity and masks no drift.
#   • Defect-1 Alignment: Docstrings updated to reflect Volume II Fourier fix.
#     (Anchors k_H_hat(0) = 8/H).
#   • Defect-2 Context: Added comments clarifying that C_ratio > 1 is a 
#     property of the Pointwise Absolute Bound, resolved by Lemma XII.1'.
#
# ============================================================================

import sys
import math
import cmath
import unittest
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict

# ============================================================================
# ROBUST IMPORT: Resolve target module with flexible path handling
# ============================================================================

def _import_target_module():
    """
    Import the target engine module with robust path resolution.
    Handles multiple possible file locations and names.
    """
    # Possible target module names and paths to try
    possible_targets = [
        # Package-style imports
        ("VOLUME_I_FORMAL_REDUCTION_PROOF.FORMAL_REDUCTION_QH_ANALYST_PROBLEM", None),
        ("VOLUME_I_FORMAL_REDUCTION_PROOF.VOLUME_I_FORMAL_REDUCTION", None),
        ("FORMAL_REDUCTION_QH_ANALYST_PROBLEM", None),
        ("VOLUME_I_FORMAL_REDUCTION", None),
    ]
    
    # Also try direct file paths relative to this test file
    test_file_dir = Path(__file__).resolve().parent
    possible_paths = [
        test_file_dir.parent / "VOLUME_I_FORMAL_REDUCTION_PROOF" / "FORMAL_REDUCTION_QH_ANALYST_PROBLEM.py",
        test_file_dir.parent / "VOLUME_I_FORMAL_REDUCTION_PROOF" / "VOLUME_I_FORMAL_REDUCTION.py",
        test_file_dir.parent.parent / "VOLUME_I_FORMAL_REDUCTION_PROOF" / "FORMAL_REDUCTION_QH_ANALYST_PROBLEM.py",
        test_file_dir.parent.parent / "VOLUME_I_FORMAL_REDUCTION_PROOF" / "VOLUME_I_FORMAL_REDUCTION.py",
    ]
    
    last_error = None
    
    # Try package-style imports first
    for module_name, _ in possible_targets:
        try:
            module = __import__(module_name, fromlist=['*'])
            return _extract_symbols(module)
        except (ImportError, ModuleNotFoundError) as e:
            last_error = e
            continue
    
    # Try direct file imports
    for target_path in possible_paths:
        if target_path.exists():
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("engine_module", str(target_path))
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    # Add the module's directory to sys.path temporarily
                    module_dir = str(target_path.parent)
                    if module_dir not in sys.path:
                        sys.path.insert(0, module_dir)
                    spec.loader.exec_module(module)
                    return _extract_symbols(module)
            except Exception as e:
                last_error = e
                continue
    
    # Final fallback: try adding project root to path and importing
    try:
        project_root = test_file_dir.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Try the most likely module name
        from VOLUME_I_FORMAL_REDUCTION_PROOF import FORMAL_REDUCTION_QH_ANALYST_PROBLEM as module
        return _extract_symbols(module)
    except Exception as e:
        last_error = e
    
    raise ImportError(
        f"Could not import target engine module. Last error: {last_error}\n"
        f"Searched paths: {[str(p) for p in possible_paths]}\n"
        f"Current sys.path: {sys.path[:5]}..."
    )


def _extract_symbols(module):
    """Extract required symbols from the target module."""
    symbols = {
        "engine_version": getattr(module, "__version__", "unknown"),
        "sech": module.sech, "sech2": module.sech2, "sech4": module.sech4, "tanh": module.tanh,
        "w_H_time": module.w_H_time, "g_H_sech4": module.g_H_sech4, 
        "k_H_time": module.k_H_time, "k_H_hat": module.k_H_hat,
        "dirichlet_S_N": module.dirichlet_S_N, "D_N": module.D_N,
        "riemann_siegel_remainder_bound": module.riemann_siegel_remainder_bound,
        "second_moment_integrand": module.second_moment_integrand,
        "kernel_tail_bound": module.kernel_tail_bound,
        "curvature_F2_bar_with_convergence": module.curvature_F2_bar_with_convergence,
        "physical_vector_x": module.physical_vector_x,
        "phased_quadratic_form": module.phased_quadratic_form,
        "diagonal_growth_term": module.diagonal_growth_term,
        "finite_N_convergence_error_bound": module.finite_N_convergence_error_bound,
        "explicit_formula_zero_contribution": module.explicit_formula_zero_contribution,
        "prime_side_archimedean_constant": module.prime_side_archimedean_constant,
        "explicit_formula_curvature_EF_H": module.explicit_formula_curvature_EF_H,
        "delta_A_pair_contribution": module.delta_A_pair_contribution,
        "delta_A_negativity_certificate": module.delta_A_negativity_certificate,
        "DEFAULT_ADMISSIBLE_H": module.DEFAULT_ADMISSIBLE_H,
        "verify_H_admissibility": module.verify_H_admissibility,
        "T0_uniformity_bound": module.T0_uniformity_bound,
        "theorem_6_2_equivalence_certificate": module.theorem_6_2_equivalence_certificate,
        "FormalReduction": module.FormalReduction,
        "ProofPipeline": module.ProofPipeline,
        "AdmissibleHRange": module.AdmissibleHRange,
        "ExplicitFormulaRepresentation": module.ExplicitFormulaRepresentation,
        "DeltaANegativityLemma": module.DeltaANegativityLemma,
    }
    
    # Also import legacy symbols if they exist (for backward compatibility)
    legacy_symbols = [
        "Lambda_H_tau", "Lambda_H_dd_tau", "curvature_F2_bar",
        "build_toeplitz_matrix", "toeplitz_quadratic_form",
        "parseval_bridge_certificate", "M1_diagonal_term", 
        "cross_offdiagonal_term", "QH_from_M1_and_cross",
        "absolute_cross_term", "C_ratio", "check_kernel_positive_definite",
        "parseval_identity_residual"
    ]
    for sym in legacy_symbols:
        if hasattr(module, sym):
            symbols[sym] = getattr(module, sym)
    
    return symbols


# Import all symbols from target module
_target = _import_target_module()
engine_version = _target["engine_version"]
sech = _target["sech"]
sech2 = _target["sech2"]
sech4 = _target["sech4"]
tanh = _target["tanh"]
w_H_time = _target["w_H_time"]
g_H_sech4 = _target["g_H_sech4"]
k_H_time = _target["k_H_time"]
k_H_hat = _target["k_H_hat"]
dirichlet_S_N = _target["dirichlet_S_N"]
D_N = _target["D_N"]
riemann_siegel_remainder_bound = _target["riemann_siegel_remainder_bound"]
second_moment_integrand = _target["second_moment_integrand"]
kernel_tail_bound = _target["kernel_tail_bound"]
curvature_F2_bar_with_convergence = _target["curvature_F2_bar_with_convergence"]
physical_vector_x = _target["physical_vector_x"]
phased_quadratic_form = _target["phased_quadratic_form"]
diagonal_growth_term = _target["diagonal_growth_term"]
finite_N_convergence_error_bound = _target["finite_N_convergence_error_bound"]
explicit_formula_zero_contribution = _target["explicit_formula_zero_contribution"]
prime_side_archimedean_constant = _target["prime_side_archimedean_constant"]
explicit_formula_curvature_EF_H = _target["explicit_formula_curvature_EF_H"]
delta_A_pair_contribution = _target["delta_A_pair_contribution"]
delta_A_negativity_certificate = _target["delta_A_negativity_certificate"]
DEFAULT_ADMISSIBLE_H = _target["DEFAULT_ADMISSIBLE_H"]
verify_H_admissibility = _target["verify_H_admissibility"]
T0_uniformity_bound = _target["T0_uniformity_bound"]
theorem_6_2_equivalence_certificate = _target["theorem_6_2_equivalence_certificate"]
FormalReduction = _target["FormalReduction"]
ProofPipeline = _target["ProofPipeline"]
AdmissibleHRange = _target["AdmissibleHRange"]
ExplicitFormulaRepresentation = _target["ExplicitFormulaRepresentation"]
DeltaANegativityLemma = _target["DeltaANegativityLemma"]

# Import legacy symbols if available
for sym in ["Lambda_H_tau", "Lambda_H_dd_tau", "curvature_F2_bar",
            "build_toeplitz_matrix", "toeplitz_quadratic_form",
            "parseval_bridge_certificate", "M1_diagonal_term", 
            "cross_offdiagonal_term", "QH_from_M1_and_cross",
            "absolute_cross_term", "C_ratio", "check_kernel_positive_definite",
            "parseval_identity_residual"]:
    if sym in _target:
        globals()[sym] = _target[sym]

# ============================================================================
# VERIFICATION METADATA — FOR PROOF PIPELINE INTEGRATION
# ============================================================================

@dataclass(frozen=True)
class VerificationResult:
    """
    Structured result from a verification check.
    Used by ProofPipeline to aggregate engine certification status.
    """
    test_name: str
    tier: str  # "T1", "T2", or "T3"
    passed: bool
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EngineVerificationReport:
    """
    Aggregate report from running the full verification layer.
    """
    engine_version: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    t1_passed: int
    t2_passed: int
    t3_passed: int
    results: List[VerificationResult]
    engine_certified: bool  # True iff all T1 tests pass
    timestamp: str
    notes: str = ""
    
    def summary(self) -> str:
        status = "CERTIFIED" if self.engine_certified else "NOT CERTIFIED"
        return (
            f"Engine Verification [{status}]\n"
            f"  Version: {self.engine_version}\n"
            f"  Tests: {self.passed_tests}/{self.total_tests} passed\n"
            f"  T1: {self.t1_passed} passed | T2: {self.t2_passed} passed | T3: {self.t3_passed} passed\n"
            f"  Notes: {self.notes}"
        )


# ============================================================================
# BASE TEST CLASS WITH VERIFICATION RESULT TRACKING
# ============================================================================

class VerificationTestCase(unittest.TestCase):
    """
    Base test case that records verification results with tier metadata.
    Subclasses should call self.record_result() to log structured outcomes.
    """
    
    def __init__(self, methodName: str = 'runTest'):
        super().__init__(methodName)
        self._results: List[VerificationResult] = []
    
    def record_result(self, test_name: str, tier: str, passed: bool, 
                      message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record a structured verification result."""
        result = VerificationResult(
            test_name=test_name,
            tier=tier,
            passed=passed,
            message=message,
            metadata=metadata or {}
        )
        self._results.append(result)
    
    def get_results(self) -> List[VerificationResult]:
        """Return accumulated verification results."""
        return self._results.copy()
    
    def tearDown(self) -> None:
        """Ensure results are captured after each test."""
        pass


# ============================================================================
# T1 TESTS: UNCONDITIONAL ALGEBRAIC/NUMERIC IDENTITIES
# ============================================================================

class TestSpecialFunctions(VerificationTestCase):
    """
    T1: Basic Hyperbolic Function Properties
    
    These tests verify the implementation of closed-form special functions
    used in the kernel definition. All are unconditional algebraic checks.
    """

    def test_sech_at_zero(self):
        """T1: sech(0) = 1 exactly."""
        result = sech(0.0)
        passed = abs(result - 1.0) < 1e-14
        self.record_result("sech_at_zero", "T1", passed, 
                          f"sech(0) = {result}", {"expected": 1.0, "actual": result})
        self.assertAlmostEqual(result, 1.0, places=14)

    def test_sech2_at_zero(self):
        """T1: sech²(0) = 1 exactly."""
        result = sech2(0.0)
        passed = abs(result - 1.0) < 1e-14
        self.record_result("sech2_at_zero", "T1", passed,
                          f"sech2(0) = {result}", {"expected": 1.0, "actual": result})
        self.assertAlmostEqual(result, 1.0, places=14)

    def test_sech4_positivity(self):
        """T1: sech⁴(x) > 0 for all real x (rapid decay)."""
        test_points = [-10.0, -1.0, 0.0, 1.0, 10.0]
        all_positive = all(sech4(x) > 0.0 for x in test_points)
        self.record_result("sech4_positivity", "T1", all_positive,
                          f"All test points positive: {test_points}",
                          {"test_points": test_points})
        for x in test_points:
            self.assertGreater(sech4(x), 0.0)

    def test_tanh_properties(self):
        """T1: tanh is odd, saturates at ±1."""
        odd_ok = abs(tanh(0.0)) < 1e-14
        sat_pos = tanh(100.0) > 0.99
        sat_neg = tanh(-100.0) < -0.99
        passed = odd_ok and sat_pos and sat_neg
        self.record_result("tanh_properties", "T1", passed,
                          f"odd={odd_ok}, sat+={sat_pos}, sat-={sat_neg}")
        self.assertAlmostEqual(tanh(0.0), 0.0, places=14)
        self.assertTrue(tanh(100.0) > 0.99)
        self.assertTrue(tanh(-100.0) < -0.99)


class TestKernelProperties(VerificationTestCase):
    """
    T1: Curvature Window and Bochner Kernel Properties.
    
    Aligned with Volume II Fourier Fix (Defect-1):
    k_H_hat(0) must be exactly 8/H. This is a critical normalization anchor.
    """

    def test_g_H_sech4_normalization(self):
        """T1: g_H_sech4(0, H) = 6/H² exactly."""
        H = 3.0
        expected = 6.0 / (H * H)
        result = g_H_sech4(0.0, H)
        passed = abs(result - expected) < 1e-14
        self.record_result("g_H_sech4_normalization", "T1", passed,
                          f"H={H}, expected={expected}, actual={result}",
                          {"H": H, "expected": expected, "actual": result})
        self.assertAlmostEqual(result, expected, places=14)

    def test_k_H_hat_at_zero(self):
        """
        T1 [CRITICAL ANCHOR]: k_H_hat(0) = 8/H.
        
        Volume II Fourier fix ensures this normalization. Previously off by 2π.
        This anchor is essential for the Parseval bridge and diagonal mass M1.
        """
        H = 2.5
        expected = 8.0 / H
        result = k_H_hat(0.0, H)
        passed = abs(result - expected) < 1e-14
        self.record_result("k_H_hat_at_zero", "T1", passed,
                          f"H={H}, expected={expected}, actual={result}",
                          {"H": H, "expected": expected, "actual": result,
                           "anchor": "Volume II Fourier fix"})
        self.assertAlmostEqual(result, expected, places=14)

    def test_k_H_hat_positivity(self):
        """T1: k_H_hat(ω) ≥ 0 for all ω (Bochner positivity)."""
        H = 1.5
        test_omegas = [-10.0, -1.0, 0.1, 5.0, 20.0]
        all_nonneg = all(k_H_hat(omega, H) >= -1e-14 for omega in test_omegas)
        self.record_result("k_H_hat_positivity", "T1", all_nonneg,
                          f"H={H}, omegas={test_omegas}")
        for omega in test_omegas:
            self.assertGreaterEqual(k_H_hat(omega, H), -1e-14)

    def test_k_H_hat_even_symmetry(self):
        """T1: k_H_hat is even: k̂_H(ω) = k̂_H(-ω)."""
        H = 2.0
        omega = 3.7
        result = abs(k_H_hat(omega, H) - k_H_hat(-omega, H))
        passed = result < 1e-14
        self.record_result("k_H_hat_even_symmetry", "T1", passed,
                          f"H={H}, ω={omega}, diff={result}")
        self.assertAlmostEqual(k_H_hat(omega, H), k_H_hat(-omega, H), places=14)


class TestDirichletModel(VerificationTestCase):
    """
    T1/T2: Dirichlet Polynomial and Energy Forms
    
    T1: Algebraic correctness of finite sums.
    T2: Positivity properties conditional on standard analytic framework.
    """

    def test_dirichlet_S_N_at_t_zero(self):
        """T1: D_N(σ, 0) = ∑ n^{-σ} (real harmonic sum)."""
        N = 5
        sigma = 0.5
        T = 0.0
        expected = sum(n**(-sigma) for n in range(1, N+1))
        result = dirichlet_S_N(T, N, sigma)
        real_ok = abs(result.real - expected) < 1e-12
        imag_ok = abs(result.imag) < 1e-12
        passed = real_ok and imag_ok
        self.record_result("dirichlet_S_N_at_t_zero", "T1", passed,
                          f"N={N}, σ={sigma}, expected={expected}, actual={result.real}",
                          {"N": N, "sigma": sigma, "expected": expected})
        self.assertAlmostEqual(result.real, expected, places=12)
        self.assertAlmostEqual(result.imag, 0.0, places=12)


# ============================================================================
# T1/T2 TESTS: PARSEVAL BRIDGE AND STRUCTURAL DECOMPOSITION
# ============================================================================

class TestParsevalBridge(VerificationTestCase):
    """
    T1: Exact Equivalence of Integral and Toeplitz Forms.
    
    UPDATED (Defect-3 Fix): Tolerance tightened to 1e-8 to enforce
    proof-grade numerical fidelity. This certifies the Parseval bridge
    implementation matches the analytic identity proved in Volume I.
    """

    def test_physical_vector_values(self):
        """T1: Physical vector x_n = n^{-σ} computed correctly."""
        N = 4
        sigma = 0.5
        x = physical_vector_x(N, sigma)
        expected = [1.0, 1.0/math.sqrt(2), 1.0/math.sqrt(3), 0.5]
        all_close = all(abs(x[i] - expected[i]) < 1e-14 for i in range(N))
        self.record_result("physical_vector_values", "T1", all_close,
                          f"N={N}, σ={sigma}",
                          {"expected": expected, "actual": x})
        for i in range(N):
            self.assertAlmostEqual(x[i], expected[i], places=14)

    def test_parseval_bridge_identity(self):
        """
        T1 [CRITICAL]: Parseval bridge residual < 1e-8.
        
        The integral curvature F2_bar must match the phased quadratic form
        up to numerical quadrature error. Tolerance tightened per Defect-3.
        """
        T0 = 14.1347  # Near first Riemann zero
        H = 1.5
        N = 5
        sigma = 0.5

        # Use dedicated bridge residual with sufficient quadrature resolution
        value, diagnostics = curvature_F2_bar_with_convergence(
            T0, H, N, sigma=sigma,
            tau_min=-100.0, tau_max=100.0,
            num_steps=5001
        )
        toeplitz_val = phased_quadratic_form(N, H, T0, sigma)
        res = abs(value - toeplitz_val)

        passed = res < 1e-8
        self.record_result("parseval_bridge_identity", "T1", passed,
                          f"residual={res:.2e} < 1e-8",
                          {"T0": T0, "H": H, "N": N, "residual": res,
                           "tolerance": 1e-8, "defect_fix": "Defect-3",
                           "diagnostics": diagnostics})
        
        self.assertLess(res, 1e-8,
                        f"Parseval bridge failed: Integral and Toeplitz forms diverge. "
                        f"Residual = {res} >= 1e-8")


class TestAnalystProblemDecomposition(VerificationTestCase):
    """
    T1: Decomposition of Q_H into Diagonal and Off-Diagonal.
    
    CONTEXT (Defect-2): The absolute_cross_term provides a worst-case bound
    ignoring phase cancellation. C_ratio > 1 is possible here. The actual
    diagonal dominance is proven via the Mean-Square framework (Lemma XII.1').
    These tests verify the algebraic decomposition, not the analytic bound.
    """

    def test_QH_decomposition_exactness(self):
        """T1: Q_H = M1 + Cross holds exactly in finite arithmetic."""
        # Use the new API if available, otherwise fall back to legacy
        if "QH_from_M1_and_cross" in globals():
            N = 8
            H = 2.0
            T0 = 21.02
            sigma = 0.5

            Q_direct = phased_quadratic_form(N, H, T0, sigma)
            Q_decomp = QH_from_M1_and_cross(N, H, T0, sigma)
            diff = abs(Q_direct - Q_decomp)
            passed = diff < 1e-12
            self.record_result("QH_decomposition_exactness", "T1", passed,
                              f"diff={diff:.2e}",
                              {"N": N, "H": H, "T0": T0, "direct": Q_direct, 
                               "decomp": Q_decomp, "diff": diff})
            self.assertAlmostEqual(Q_direct, Q_decomp, places=12)
        else:
            self.skipTest("Legacy decomposition functions not available")

    def test_M1_positivity(self):
        """T1: Diagonal mass M1 > 0 unconditionally."""
        if "M1_diagonal_term" in globals():
            N = 10
            H = 1.0
            M1 = M1_diagonal_term(N, H)
            passed = M1 > 0.0
            self.record_result("M1_positivity", "T1", passed,
                              f"M1={M1}", {"N": N, "H": H, "M1": M1})
            self.assertGreater(M1, 0.0)
        else:
            self.skipTest("M1_diagonal_term not available")

    def test_absolute_cross_term_bounds_cross(self):
        """
        T1: |Cross(T0)| ≤ AbsoluteCross for any T0.
        
        This is an algebraic property of the definitions, not an analytic bound.
        """
        if "absolute_cross_term" in globals() and "cross_offdiagonal_term" in globals():
            N = 15
            H = 1.2
            sigma = 0.5
            abs_cross = absolute_cross_term(N, H, sigma)

            test_T0 = [0.0, 14.1, 50.5, 100.0]
            all_bounded = all(abs(cross_offdiagonal_term(N, H, T0, sigma)) <= abs_cross + 1e-12 
                             for T0 in test_T0)
            self.record_result("absolute_cross_term_bounds_cross", "T1", all_bounded,
                              f"N={N}, H={H}, T0_samples={test_T0}")
            for T0 in test_T0:
                cross = cross_offdiagonal_term(N, H, T0, sigma)
                self.assertLessEqual(abs(cross), abs_cross + 1e-12)
        else:
            self.skipTest("Legacy cross term functions not available")

    def test_C_ratio_computes(self):
        """
        T1: C_ratio is finite and ≥ 0.
        
        Note: For large N, C_ratio > 1.0. This confirms that the
        Pointwise Absolute Bound is insufficient for proof, necessitating
        the Phase-Averaged (Lemma XII.1') approach. This test verifies
        only the computation, not the analytic implication.
        """
        if "C_ratio" in globals():
            N = 10
            H = 2.0
            ratio = C_ratio(N, H)
            finite = math.isfinite(ratio)
            nonneg = ratio >= 0.0
            passed = finite and nonneg
            self.record_result("C_ratio_computes", "T1", passed,
                              f"ratio={ratio}",
                              {"N": N, "H": H, "ratio": ratio,
                               "note": "C_ratio > 1 possible; see Lemma XII.1'"})
            self.assertTrue(math.isfinite(ratio))
            self.assertGreaterEqual(ratio, 0.0)
        else:
            self.skipTest("C_ratio not available")


# ============================================================================
# T1/T2 TESTS: BOCHNER POSITIVITY AND API SURFACE
# ============================================================================

class TestBochnerPositivity(VerificationTestCase):
    """
    T1: Kernel Positivity Constraints
    
    By Bochner's Theorem, since k_H_hat(ω) > 0, the resulting Toeplitz
    matrix must be Positive Semi-Definite. Numerical verification only.
    """

    def test_toeplitz_kernel_psd(self):
        """
        T1/T2: Toeplitz matrix is PSD (numerically).
        
        Conditional on Bochner's theorem; numerical check via eigenvalues
        or Gershgorin bounds.
        """
        if "check_kernel_positive_definite" in globals() and "build_toeplitz_matrix" in globals():
            N = 20
            H = 1.5
            l_min, l_max = check_kernel_positive_definite(N, H)

            # Try numpy eigenvalue check if available
            try:
                import numpy as np
                M = build_toeplitz_matrix(N, H)
                eigenvalues = np.linalg.eigvalsh(M)
                min_eig = float(min(eigenvalues))
                passed = min_eig >= -1e-10
                self.record_result("toeplitz_kernel_psd", "T1", passed,
                                  f"min_eig={min_eig}",
                                  {"N": N, "H": H, "min_eigenvalue": min_eig,
                                   "method": "numpy_eigvalsh"})
                self.assertGreaterEqual(min_eig, -1e-10, "Toeplitz matrix is not PSD")
            except ImportError:
                # Fallback to Gershgorin-based bounds
                passed = l_min >= -1e-10
                self.record_result("toeplitz_kernel_psd", "T2", passed,
                                  f"l_min={l_min}, l_max={l_max}",
                                  {"N": N, "H": H, "l_min": l_min, "l_max": l_max,
                                   "method": "gershgorin"})
                self.assertIsInstance(l_min, float)
                self.assertIsInstance(l_max, float)
                self.assertGreaterEqual(l_min, -1e-10)
        else:
            self.skipTest("PSD check functions not available")


class TestFormalReductionAPI(VerificationTestCase):
    """
    T1: API Surface verification for external TDD harnesses
    
    Ensures the FormalReduction class exposes the required interface
    with correct types and return shapes.
    """

    def test_api_surface(self):
        """T1: All required API methods present and return expected types."""
        api = FormalReduction()
        
        # Updated checks matching the Proof-Complete Engine API
        checks = {
            "k_H_time": lambda: isinstance(api.k_H_time(0.0, 1.0), (int, float)),
            "k_H_hat": lambda: isinstance(api.k_H_hat(0.0, 1.0), (int, float)),
            "admissible_H_range": lambda: isinstance(api.admissible_H_range(), AdmissibleHRange),
            "verify_H_admissible": lambda: isinstance(api.verify_H_admissible(1.0), bool),
            "Q_H_finite_N": lambda: isinstance(api.Q_H_finite_N(3, 1.0, 0.0), (int, float)),
            "Q_H_time_domain": lambda: isinstance(api.Q_H_time_domain(10.0, 1.0, 3), tuple),
            "finite_N_convergence_bound": lambda: isinstance(api.finite_N_convergence_bound(3, 1.0, 0.0), dict),
            "explicit_formula_representation": lambda: isinstance(api.explicit_formula_representation(1.0), ExplicitFormulaRepresentation),
            "delta_A_lemma_interface": lambda: isinstance(api.delta_A_lemma_interface(), DeltaANegativityLemma),
            "T0_uniformity_estimate": lambda: isinstance(api.T0_uniformity_estimate(1.0, (-10, 10), 3), dict),
            "theorem_6_2_certificate": lambda: isinstance(api.theorem_6_2_certificate(1.0, [(0.5, 14.13)]), dict),
            "proof_pipeline": lambda: isinstance(api.proof_pipeline(1.0, [(0.5, 14.13)]), ProofPipeline),
        }
        
        results = {name: check() for name, check in checks.items()}
        all_passed = all(results.values())
        
        self.record_result("api_surface", "T1", all_passed,
                          f"passed={sum(results.values())}/{len(results)}",
                          {"checks": results})
        
        for name, check in checks.items():
            self.assertTrue(check(), f"API check failed: {name}")


# ============================================================================
# VERIFICATION LAYER: ENGINE CERTIFICATION FOR PROOF PIPELINE
# ============================================================================

class VerificationLayer:
    """
    Verification Layer V1.TDD — Engine Certification Interface
    
    This class provides programmatic access to the TDD suite for integration
    with the ProofPipeline. It runs a minimal subset of critical checks
    to certify that the underlying engine is faithful to proved identities.
    
    Usage:
        report = VerificationLayer.run_engine_checks()
        if report.engine_certified:
            # Proceed with analytic precondition verification
            pipeline = ProofPipeline(H, zeros)
            ...
    """
    
    @staticmethod
    def _run_minimal_checks() -> List[VerificationResult]:
        """
        Run minimal critical checks for engine certification.
        These are the T1 tests that must pass for proof-grade compliance.
        """
        results = []
        
        # 1. Kernel normalization anchor (Defect-1 fix)
        try:
            H = 2.5
            expected = 8.0 / H
            actual = k_H_hat(0.0, H)
            passed = abs(actual - expected) < 1e-14
            results.append(VerificationResult(
                test_name="k_H_hat_at_zero_minimal",
                tier="T1",
                passed=passed,
                message=f"k̂_H(0) = {actual} vs expected {expected}",
                metadata={"H": H, "expected": expected, "actual": actual}
            ))
        except Exception as e:
            results.append(VerificationResult(
                test_name="k_H_hat_at_zero_minimal",
                tier="T1",
                passed=False,
                message=f"Exception: {e}",
                metadata={"error": str(e)}
            ))
        
        # 2. Parseval bridge residual (Defect-3 fix)
        try:
            T0, H, N = 14.1347, 1.5, 5
            # Use the new convergence function if available
            if "curvature_F2_bar_with_convergence" in globals():
                value, _ = curvature_F2_bar_with_convergence(T0, H, N, num_steps=5001)
                toeplitz_val = phased_quadratic_form(N, H, T0)
                res = abs(value - toeplitz_val)
            elif "parseval_identity_residual" in globals():
                res = parseval_identity_residual(T0, H, N, num_steps=5001)
            else:
                res = float('inf')
            passed = res < 1e-8
            results.append(VerificationResult(
                test_name="parseval_bridge_minimal",
                tier="T1",
                passed=passed,
                message=f"residual={res:.2e} < 1e-8",
                metadata={"T0": T0, "H": H, "N": N, "residual": res}
            ))
        except Exception as e:
            results.append(VerificationResult(
                test_name="parseval_bridge_minimal",
                tier="T1",
                passed=False,
                message=f"Exception: {e}",
                metadata={"error": str(e)}
            ))
        
        # 3. Q_H decomposition exactness (if legacy functions available)
        if "QH_from_M1_and_cross" in globals():
            try:
                N, H, T0 = 8, 2.0, 21.02
                Q_direct = phased_quadratic_form(N, H, T0)
                Q_decomp = QH_from_M1_and_cross(N, H, T0)
                diff = abs(Q_direct - Q_decomp)
                passed = diff < 1e-12
                results.append(VerificationResult(
                    test_name="QH_decomposition_minimal",
                    tier="T1",
                    passed=passed,
                    message=f"decomposition diff={diff:.2e}",
                    metadata={"N": N, "H": H, "T0": T0, "diff": diff}
                ))
            except Exception as e:
                results.append(VerificationResult(
                    test_name="QH_decomposition_minimal",
                    tier="T1",
                    passed=False,
                    message=f"Exception: {e}",
                    metadata={"error": str(e)}
                ))
        
        # 4. Toeplitz PSD check
        if "check_kernel_positive_definite" in globals():
            try:
                N, H = 20, 1.5
                l_min, _ = check_kernel_positive_definite(N, H)
                passed = l_min >= -1e-10
                results.append(VerificationResult(
                    test_name="toeplitz_psd_minimal",
                    tier="T1",
                    passed=passed,
                    message=f"l_min={l_min}",
                    metadata={"N": N, "H": H, "l_min": l_min}
                ))
            except Exception as e:
                results.append(VerificationResult(
                    test_name="toeplitz_psd_minimal",
                    tier="T1",
                    passed=False,
                    message=f"Exception: {e}",
                    metadata={"error": str(e)}
                ))
        
        return results
    
    @staticmethod
    def run_engine_checks() -> EngineVerificationReport:
        """
        Run full verification suite and return structured report.
        
        Returns:
            EngineVerificationReport with certification status.
        """
        from datetime import datetime
        
        # Run minimal critical checks
        results = VerificationLayer._run_minimal_checks()
        
        # Aggregate statistics
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        t1_passed = sum(1 for r in results if r.tier == "T1" and r.passed)
        t2_passed = sum(1 for r in results if r.tier == "T2" and r.passed)
        t3_passed = sum(1 for r in results if r.tier == "T3" and r.passed)
        
        # Engine certified iff all T1 tests pass
        engine_certified = all(r.passed for r in results if r.tier == "T1")
        
        notes = ""
        if not engine_certified:
            failed_t1 = [r.test_name for r in results if r.tier == "T1" and not r.passed]
            notes = f"Failed T1 checks: {failed_t1}"
        
        return EngineVerificationReport(
            engine_version=engine_version,
            total_tests=total,
            passed_tests=passed,
            failed_tests=failed,
            t1_passed=t1_passed,
            t2_passed=t2_passed,
            t3_passed=t3_passed,
            results=results,
            engine_certified=engine_certified,
            timestamp=datetime.utcnow().isoformat() + "Z",
            notes=notes
        )
    
    @staticmethod
    def assert_engine_certified() -> None:
        """
        Raise AssertionError if engine is not certified.
        Useful for CI/CD gates.
        """
        report = VerificationLayer.run_engine_checks()
        if not report.engine_certified:
            raise AssertionError(
                f"Engine verification failed: {report.summary()}"
            )


# ============================================================================
# TEST RUNNER WITH VERIFICATION REPORT GENERATION
# ============================================================================

def run_verification_suite() -> EngineVerificationReport:
    """
    Run the full unittest suite and generate verification report.
    
    This integrates the standard unittest framework with the verification
    layer for comprehensive engine certification.
    """
    from datetime import datetime
    import io
    from contextlib import redirect_stderr
    
    # Collect all test cases
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    # Run tests and capture results
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)
    
    # Extract verification results from test cases
    all_results = []
    for test_group in [TestSpecialFunctions, TestKernelProperties, 
                       TestDirichletModel, TestParsevalBridge,
                       TestAnalystProblemDecomposition, TestBochnerPositivity,
                       TestFormalReductionAPI]:
        for method_name in dir(test_group):
            if method_name.startswith('test_'):
                # Create instance and run to capture results
                test_instance = test_group(method_name)
                try:
                    test_instance.setUp()
                    getattr(test_instance, method_name)()
                    all_results.extend(test_instance.get_results())
                except unittest.SkipTest:
                    pass
                except Exception:
                    # Record failure
                    all_results.append(VerificationResult(
                        test_name=method_name,
                        tier="T1",  # Default assumption
                        passed=False,
                        message="Test execution failed",
                        metadata={"error": "exception_during_test"}
                    ))
    
    # Aggregate statistics
    total = len(all_results)
    passed = sum(1 for r in all_results if r.passed)
    failed = total - passed
    t1_passed = sum(1 for r in all_results if r.tier == "T1" and r.passed)
    t2_passed = sum(1 for r in all_results if r.tier == "T2" and r.passed)
    t3_passed = sum(1 for r in all_results if r.tier == "T3" and r.passed)
    
    # Engine certified iff all T1 tests pass
    engine_certified = all(r.passed for r in all_results if r.tier == "T1")
    
    notes = ""
    if result.failures or result.errors:
        notes = f"unittest failures: {len(result.failures)} fail, {len(result.errors)} error"
    
    return EngineVerificationReport(
        engine_version=engine_version,
        total_tests=total,
        passed_tests=passed,
        failed_tests=failed,
        t1_passed=t1_passed,
        t2_passed=t2_passed,
        t3_passed=t3_passed,
        results=all_results,
        engine_certified=engine_certified,
        timestamp=datetime.utcnow().isoformat() + "Z",
        notes=notes
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Volume I TDD Verification Suite — Proof Infrastructure"
    )
    parser.add_argument(
        '--report', action='store_true',
        help='Generate structured verification report instead of running tests'
    )
    parser.add_argument(
        '--certify', action='store_true',
        help='Run minimal checks and exit with code 0 if certified, 1 otherwise'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    if args.report:
        # Generate full verification report
        report = run_verification_suite()
        print(report.summary())
        if args.verbose:
            print("\nDetailed Results:")
            for r in report.results:
                status = "✓" if r.passed else "✗"
                print(f"  [{r.tier}] {status} {r.test_name}: {r.message}")
        sys.exit(0 if report.engine_certified else 1)
    
    elif args.certify:
        # Run minimal certification checks
        report = VerificationLayer.run_engine_checks()
        if args.verbose:
            print(report.summary())
            for r in report.results:
                status = "✓" if r.passed else "✗"
                print(f"  [{r.tier}] {status} {r.test_name}")
        sys.exit(0 if report.engine_certified else 1)
    
    else:
        # Run standard unittest suite
        unittest.main(verbosity=2 if args.verbose else 1)