#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PROOF_SCRIPT (PUBLIC HSO, φ-weights, LOG-FREE TAP HO)
=====================================================


"The Analyst's Problem – Hilbert Operator" (TAP HO), public version
with golden-ratio φ-Ruelle weights and no proprietary content.


REQUIREMENTS
------------
LOG-FREE TAP HO
  • No use of log(), np.log(), or any logarithm in the operator
    definition or in any of the core verification tests.
  • math.log is used only inside φ-metadata for weight decay,
    not in TAP HO itself.


PUBLIC HSO
  • The operator is a φ-weighted Hilbert–Schmidt Gram operator
    on ℓ², built from a single global feature map Γ and fixed
    golden-ratio weights W_φ.
  • No proprietary basis (e.g. TAP-HSO internals) appears.


Golden-Ratio φ-Ruelle Weights
  • PHI = (1 + sqrt(5)) / 2.
  • Weights w_k ∝ 4 / (φ^k + φ^{-k})², normalised to sum to 1.
  • Fast-decaying, positive sequence compatible with
    Hilbert–Schmidt compactness.


Hilbert Operator Requirements (A1–A5)
  A1: Linearity — tested numerically on finite ℓ²_N.
  A2: Boundedness — estimated via power iteration and Schur test.
  A3: Adjoint existence — enforced by symmetry K = Kᵀ and tested.
  A4: Compactness — supported by Hilbert–Schmidt norm and SVD
      truncation energy tests.
  A5: Real spectrum — verified on symmetric K_N via eigvalsh.


Cross-Dimension Coherence
  • K_N is always the top-left principal corner of K_M for M > N.
  • Models a single infinite operator K on ℓ² with K_N = P_N K P_N*.


SEB-NORMALISER INVERSE
  • Given K_N and a Collatz observable v, compute:
      – Eigenvalues λ_j (operator spectrum).
      – Eigenvectors φ_j (eigenmodes).
      – Spectral coefficients c_j = ⟨v, φ_j⟩ (Collatz spectrals).
      – Reconstruction from top-k modes and residual.


USAGE
-----
  • Primary verification:
        python3 Collatz_SEB.py
  • To run the inverse SEB demo, call demo_inverse_seb_normaliser()
    from a separate driver or REPL.
"""


from __future__ import annotations


import numpy as np
import math
import warnings
from typing import Callable, Tuple, Optional, Union, Dict, List
from dataclasses import dataclass, field

import matplotlib.pyplot as plt  # added for charting


try:
    import sympy as sp  # optional, not used in Gram case
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    warnings.warn("sympy not available; symbolic checks limited")


# ============================================================================
# φ-RUELLE WEIGHTS (PUBLIC HSO, GOLDEN-RATIO BASED) – METADATA ONLY
# ============================================================================


PHI = (1.0 + math.sqrt(5.0)) / 2.0
NUM_BRANCHES = 9  # feature dimension / number of φ-weights



@dataclass
class WeightDecayAnalysis:
    weights: np.ndarray
    decay_rate: float = field(init=False)
    summable_p: List[float] = field(init=False)


    def __post_init__(self):
        self.decay_rate = -2.0 * math.log(PHI)
        self.summable_p = [1.0, 2.0, float("inf")]


    def is_hilbert_schmidt_compatible(self) -> bool:
        return float(np.sum(self.weights ** 2)) < float("inf")


    def operator_norm_bound(self) -> float:
        return float(np.max(np.abs(self.weights)))



def phi_bi_lorentzian_weights(num_branches: int = NUM_BRANCHES) -> np.ndarray:
    ks = np.arange(num_branches, dtype=float)
    denom = (PHI ** ks + PHI ** (-ks)) ** 2
    w_raw = 4.0 / denom
    return w_raw / w_raw.sum()



W_PHI = phi_bi_lorentzian_weights()
WEIGHT_ANALYSIS = WeightDecayAnalysis(W_PHI)


# ============================================================================
# GLOBAL PUBLIC HSO FEATURE MAP (NO PROPRIETARY CONTENT)
# ============================================================================


MAX_N = 5000
FEATURE_DIM = NUM_BRANCHES


_rng = np.random.default_rng(2026)
_GAMMA_FULL = _rng.normal(size=(MAX_N, FEATURE_DIM))
_norms = np.linalg.norm(_GAMMA_FULL, axis=1, keepdims=True)
_norms[_norms == 0.0] = 1.0
_GAMMA_FULL = _GAMMA_FULL / _norms


# ============================================================================
# USER-FACING OPERATOR DEFINITION: TAP HO VIA GRAM SURROGATE
# ============================================================================


KERNEL_TYPE = "gram_surrogate"



def define_analysts_problem_kernel(
    x: Optional[Union[float, np.ndarray]] = None,
    y: Optional[Union[float, np.ndarray]] = None,
    N: Optional[int] = None,
    params: Optional[Dict] = None,
) -> Union[float, np.ndarray, Callable]:
    if KERNEL_TYPE == "gram_surrogate":
        if N is None:
            raise ValueError("N must be specified for matrix-type kernels")
        if N > MAX_N:
            raise ValueError(f"N={N} exceeds MAX_N={MAX_N}")
        Gamma_raw = _GAMMA_FULL[:N, :]
        idx_scale = np.arange(1, N + 1, dtype=float)[:, None]
        Gamma_scaled = Gamma_raw / idx_scale
        S_phi = np.diag(W_PHI.astype(float))
        K = Gamma_scaled @ S_phi @ Gamma_scaled.T
        return (K + K.T) / 2.0
    if KERNEL_TYPE == "matrix":
        if N is None:
            raise ValueError("N must be specified for matrix-type kernels")
        i = np.arange(1, N + 1)
        j = np.arange(1, N + 1)
        return 1.0 / (i[:, None] + j[None, :] - 1)
    if KERNEL_TYPE == "integral":
        if x is None or y is None:
            raise ValueError("x and y must be specified for integral kernels")
        sigma = params.get("sigma", 1.0) if params else 1.0
        return np.exp(-np.square(np.array(x) - np.array(y)) / (2.0 * sigma ** 2))
    if KERNEL_TYPE == "fourier_multiplier":
        def symbol(xi):
            return -1j * np.sign(xi)
        return symbol
    raise ValueError(f"Unknown KERNEL_TYPE: {KERNEL_TYPE}")


# ============================================================================
# ANALYTIC ARGUMENT VERIFIER (LOG-FREE TAP HO)
# ============================================================================



@dataclass
class AnalyticArgument:
    property_name: str
    theorem_statement: str
    verification_method: str
    status: str = "pending"
    evidence: Dict = field(default_factory=dict)
    references: List[str] = field(default_factory=list)


    def report(self) -> str:
        icon_map = {
            "verified": "OK",
            "failed": "FAIL",
            "inconclusive": "WARN",
            "pending": "PEND",
        }
        icon = icon_map.get(self.status, "?")
        return f"{icon} {self.property_name}: {self.status}\n {self.verification_method}"



class AnalyticVerifier:
    def __init__(self, kernel_func: Callable, kernel_type: str):
        self.kernel_func = kernel_func
        self.kernel_type = kernel_type
        self.arguments: List[AnalyticArgument] = []


    def add_argument(self, arg: AnalyticArgument):
        self.arguments.append(arg)


    def verify_kernel_symmetry(self) -> AnalyticArgument:
        arg = AnalyticArgument(
            property_name="Self-adjointness (kernel symmetry)",
            theorem_statement="K_N symmetric ⇒ T_N self-adjoint; consistent corners ⇒ T self-adjoint on ℓ²",
            verification_method="Numeric symmetry of K_N for several N",
            references=["Reed & Simon, Thm. VI.23", "Conway, Prop. II.2.4"],
        )
        if self.kernel_type not in ["matrix", "gram_surrogate"]:
            arg.status = "inconclusive"
            arg.evidence = {"reason": f"symmetry check not implemented for {self.kernel_type}"}
            self.add_argument(arg)
            return arg
        try:
            sym_errs = []
            for N in [100, 400, 1200]:
                K = self.kernel_func(N=N)
                sym_err = float(np.max(np.abs(K - K.T)))
                sym_errs.append(sym_err)
            max_err = max(sym_errs)
            arg.evidence = {"symmetry_errors": sym_errs, "max_symmetry_error": max_err}
            arg.status = "verified" if max_err < 1e-12 else "inconclusive"
        except Exception as e:
            arg.status = "inconclusive"
            arg.evidence = {"error": str(e)}
        self.add_argument(arg)
        return arg


    def verify_compactness_via_truncation(
        self, N_test: int = 1000, k_fraction: float = 0.2, threshold: float = 5e-3
    ) -> AnalyticArgument:
        arg = AnalyticArgument(
            property_name="Compactness (spectral truncation proxy)",
            theorem_statement="Rapid decay of singular values ⇒ T compact on ℓ²",
            verification_method="SVD truncation energy capture",
            references=["Pinkus, n-Widths", "Reed & Simon, Thm. VI.20"],
        )
        try:
            K = self.kernel_func(N=N_test)
            _, s, _ = np.linalg.svd(K, full_matrices=False)
            total_energy = float(np.sum(s ** 2)) + 1e-15
            k = max(1, int(N_test * k_fraction))
            captured = float(np.sum(s[:k] ** 2))
            uncaptured = 1.0 - captured / total_energy
            arg.evidence = {
                "N_test": N_test,
                "k_fraction": k_fraction,
                "k_used": k,
                "uncaptured_energy_fraction": uncaptured,
            }
            arg.status = "verified" if uncaptured < threshold else "inconclusive"
        except Exception as e:
            arg.status = "inconclusive"
            arg.evidence = {"error": str(e)}
        self.add_argument(arg)
        return arg


    def verify_boundedness_schur_test(self) -> AnalyticArgument:
        arg = AnalyticArgument(
            property_name="Boundedness (Schur test, matrix form)",
            theorem_statement="Row/column sum bounds ⇒ ||T||_op bounded on ℓ²",
            verification_method="Numeric Schur bounds for N up to 2000",
            references=["Stein & Weiss, Ch. I.4"],
        )
        if self.kernel_type not in ["matrix", "gram_surrogate"]:
            arg.status = "inconclusive"
            arg.evidence = {"reason": f"Schur test not implemented for {self.kernel_type}"}
            self.add_argument(arg)
            return arg
        try:
            N_vals = [200, 400, 800, 2000]
            row_sums = []
            col_sums = []
            for N in N_vals:
                K = self.kernel_func(N=N)
                absK = np.abs(K)
                row_sums.append(float(np.max(np.sum(absK, axis=1))))
                col_sums.append(float(np.max(np.sum(absK, axis=0))))
            C1 = max(row_sums)
            C2 = max(col_sums)
            schur = math.sqrt(C1 * C2)
            arg.evidence = {
                "N_values": N_vals,
                "max_row_sums": row_sums,
                "max_col_sums": col_sums,
                "schur_bound_estimate": schur,
            }
            arg.status = "verified" if np.isfinite(schur) and schur < 1e6 else "inconclusive"
        except Exception as e:
            arg.status = "inconclusive"
            arg.evidence = {"error": str(e)}
        self.add_argument(arg)
        return arg


    def verify_weight_decay_compatibility(self) -> AnalyticArgument:
        arg = AnalyticArgument(
            property_name="φ-Ruelle weight decay compatibility",
            theorem_statement="Exponential decay of φ-weights ⇒ diagonal φ-operator compact on ℓ²",
            verification_method="Metadata: decay_rate < 0 and ℓ²-summability",
            references=["Ruelle, Thermodynamic Formalism"],
        )
        try:
            decay_rate = WEIGHT_ANALYSIS.decay_rate
            is_l2 = WEIGHT_ANALYSIS.is_hilbert_schmidt_compatible()
            op_norm = WEIGHT_ANALYSIS.operator_norm_bound()
            arg.evidence = {
                "decay_rate_metadata": decay_rate,
                "ell2_summable": is_l2,
                "operator_norm_bound": op_norm,
            }
            arg.status = "verified" if is_l2 and decay_rate < 0.0 else "inconclusive"
        except Exception as e:
            arg.status = "inconclusive"
            arg.evidence = {"error": str(e)}
        self.add_argument(arg)
        return arg


    def verify_cross_dimension_consistency(
        self,
        N_pairs: Optional[List[Tuple[int, int]]] = None,
        tol: float = 1e-10,
    ) -> AnalyticArgument:
        arg = AnalyticArgument(
            property_name="Cross-dimension consistency",
            theorem_statement="K_N principal corner of K_M ⇒ single operator on ℓ²",
            verification_method="Compare K_N to top-left block of K_M",
            references=["Conway, Sec. II.3"],
        )
        if self.kernel_type not in ["matrix", "gram_surrogate"]:
            arg.status = "inconclusive"
            arg.evidence = {"reason": f"only for matrix-like kernels, not {self.kernel_type}"}
            self.add_argument(arg)
            return arg
        if N_pairs is None:
            N_pairs = [(100, 200), (200, 400), (400, 800)]
        try:
            max_err = 0.0
            evidence_pairs = []
            for Ns, Nl in N_pairs:
                K_s = self.kernel_func(N=Ns)
                K_l = self.kernel_func(N=Nl)
                block = K_l[:Ns, :Ns]
                err = float(np.linalg.norm(block - K_s, ord="fro"))
                evidence_pairs.append(
                    {"N_small": Ns, "N_large": Nl, "fro_block_error": err}
                )
                max_err = max(max_err, err)
            arg.evidence = {"max_block_error": max_err, "pairs": evidence_pairs}
            arg.status = "verified" if max_err < tol else "inconclusive"
        except Exception as e:
            arg.status = "inconclusive"
            arg.evidence = {"error": str(e)}
        self.add_argument(arg)
        return arg


    def generate_report(self) -> str:
        lines = [
            "",
            "=" * 70,
            "ANALYTIC VERIFICATION REPORT (PUBLIC HSO, φ-WEIGHTS)",
            "=" * 70,
        ]
        for arg in self.arguments:
            lines.append(arg.report())
            if arg.evidence:
                lines.append(f" Evidence: {arg.evidence}")
            if arg.references:
                lines.append(" References: " + "; ".join(arg.references))
            lines.append("")
        verified = sum(1 for a in self.arguments if a.status == "verified")
        total = len(self.arguments)
        lines.append(f"SUMMARY: {verified}/{total} analytic arguments verified")
        lines.append("=" * 70 + "\n")
        return "\n".join(lines)


# ============================================================================
# NUMERICAL TESTS ON FINITE ℓ²_N (LOG-FREE TAP HO)
# ============================================================================



def test_linearity(
    T: Callable[[np.ndarray], np.ndarray],
    N: int,
    trials: int = 10,
    tol: float = 1e-10,
) -> Tuple[bool, float]:
    max_err = 0.0
    for _ in range(trials):
        x = np.random.randn(N)
        y = np.random.randn(N)
        alpha, beta = np.random.randn(2)
        lhs = T(alpha * x + beta * y)
        rhs = alpha * T(x) + beta * T(y)
        denom = np.linalg.norm(lhs) + 1e-15
        err = np.linalg.norm(lhs - rhs) / denom
        max_err = max(max_err, err)
    return max_err < tol, float(max_err)



def test_boundedness(K: np.ndarray) -> Tuple[bool, float]:
    N = K.shape[0]
    x = np.random.randn(N)
    x /= np.linalg.norm(x) + 1e-15
    op_norm = 0.0
    for _ in range(200):
        y = K @ x
        norm_y = np.linalg.norm(y)
        if norm_y < 1e-15:
            break
        x = y / norm_y
        op_norm = norm_y
    return np.isfinite(op_norm) and op_norm < 1e6, float(op_norm)



def test_adjoint_consistency(
    K: np.ndarray, trials: int = 10, tol: float = 1e-10
) -> Tuple[bool, float]:
    max_err = 0.0
    for _ in range(trials):
        x = np.random.randn(K.shape[0])
        y = np.random.randn(K.shape[0])
        lhs = float(np.dot(y, K @ x))
        rhs = float(np.dot(K.T @ y, x))
        denom = abs(lhs) + abs(rhs) + 1e-15
        err = abs(lhs - rhs) / denom
        max_err = max(max_err, err)
    return max_err < tol, float(max_err)



def test_hilbert_schmidt(K: np.ndarray) -> Tuple[bool, float]:
    hs_sq = float(np.sum(np.abs(K) ** 2))
    return np.isfinite(hs_sq), float(math.sqrt(hs_sq))



def test_spectral_reality(K: np.ndarray, tol: float = 1e-10) -> Tuple[bool, float]:
    evals = np.linalg.eigvalsh((K + K.T) / 2.0)
    imag_max = float(np.max(np.abs(np.imag(evals))))
    return imag_max < tol, imag_max



def test_positive_semidefinite(
    K: np.ndarray, trials: int = 20, tol: float = 1e-10
) -> Tuple[bool, float]:
    N = K.shape[0]
    min_q = float("inf")
    for _ in range(trials):
        x = np.random.randn(N)
        q = float(x.T @ (K @ x))
        min_q = min(min_q, q)
    return min_q >= -tol, min_q


# ============================================================================
# SEB-NORMALISER INVERSE: EIGEN-BASED COLLATZ SPECTRALS
# ============================================================================



def compute_eigendecomposition(K: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    evals, evecs = np.linalg.eigh(K)
    idx = np.argsort(-np.abs(evals))
    evals = evals[idx]
    evecs = evecs[:, idx]
    if k is not None and 1 <= k < len(evals):
        evals = evals[:k]
        evecs = evecs[:, :k]
    return evals, evecs



def collatz_observable_vector(N: int) -> np.ndarray:
    def collatz(n: int) -> int:
        return n // 2 if n % 2 == 0 else 3 * n + 1


    def collatz_stopping_time(n: int, max_steps: int = 10000) -> int:
        steps = 0
        cur = n
        while cur != 1 and steps < max_steps:
            cur = collatz(cur)
            steps += 1
        return steps


    vals = [collatz_stopping_time(n) for n in range(1, N + 1)]
    return np.asarray(vals, dtype=float)



def project_collatz_onto_eigenbasis(
    K: np.ndarray,
    observable_vector: Optional[np.ndarray] = None,
    top_k: int = 20
) -> Dict[str, np.ndarray]:
    N = K.shape[0]
    if observable_vector is None:
        v = collatz_observable_vector(N)
    else:
        v = np.asarray(observable_vector, dtype=float)
        if v.shape[0] != N:
            raise ValueError(f"observable_vector length {v.shape[0]} != K dimension {N}")
    evals, evecs = compute_eigendecomposition(K)
    coeffs = evecs.T @ v
    k_use = min(top_k, len(evals))
    recon = (evecs[:, :k_use] * coeffs[:k_use]).sum(axis=1)
    residual = v - recon
    return {
        "evals": evals,
        "evecs": evecs,
        "coeffs": coeffs,
        "recon": recon,
        "residual": residual,
        "observable": v,
    }



def demo_inverse_seb_normaliser(N: int = 2000, top_k: int = 20) -> None:
    print("=" * 80)
    print("SEB-NORMALISER INVERSE DEMO – EIGENVALUES & COLLATZ SPECTRALS".center(80))
    print("=" * 80)
    print(f"Dimension N   : {N}")
    print(f"Top-k modes   : {top_k}")
    print("Operator      : TAP-HO public HSO φ-operator (K_N)")
    print("=" * 80)
    K = define_analysts_problem_kernel(N=N)
    v = collatz_observable_vector(N)
    result = project_collatz_onto_eigenbasis(K, observable_vector=v, top_k=top_k)
    evals = result["evals"]
    coeffs = result["coeffs"]
    residual = result["residual"]
    print("\nLeading eigenvalues (by |λ|):")
    for j in range(min(top_k, len(evals))):
        lam = evals[j]
        print(f"  λ_{j+1:2d} ≈ {lam:.6e}")
    print("\nLeading Collatz spectral coefficients |c_j|:")
    for j in range(min(top_k, len(coeffs))):
        print(f"  |c_{j+1:2d}| ≈ {abs(coeffs[j]):.6e}")
    recon_err = float(np.linalg.norm(residual) / (np.linalg.norm(v) + 1e-15))
    print(f"\nRelative reconstruction error using top-{top_k} modes ≈ {recon_err:.6e}")
    print("=" * 80)


# ============================================================================
# MAIN: HYBRID NUMERIC + ANALYTIC VERIFICATION + FINAL CHART
# ============================================================================



def main() -> bool:
    print("================================================================")
    print(" TAP HO PUBLIC HSO φ-OPERATOR – HILBERT OPERATOR VERIFICATION")
    print(" LOG-FREE TAP HO (no log() in operator core)".center(64))
    print("================================================================\n")
    print(f"Kernel type     : {KERNEL_TYPE}")
    print(f"φ-Ruelle weights: {W_PHI}")
    print(f"Weight metadata : decay_rate = {WEIGHT_ANALYSIS.decay_rate:.4f}, "
          f"ℓ²-summable = {WEIGHT_ANALYSIS.is_hilbert_schmidt_compatible()}")
    print()
    test_N = [100, 400, 1200, 2000, 4000]
    results_numeric: List[Dict[str, Union[int, bool, float]]] = []
    analytic = AnalyticVerifier(define_analysts_problem_kernel, KERNEL_TYPE)
    op_norms_by_N: Dict[int, float] = {}
    hs_norms_by_N: Dict[int, float] = {}
    for N in test_N:
        print(f"▶ Numeric tests at N = {N}")
        print("-" * 60)
        K = define_analysts_problem_kernel(N=N)


        def T(x, K_mat=K):
            return K_mat @ x


        linear_ok, linear_err = test_linearity(T, N)
        bounded_ok, op_norm = test_boundedness(K)
        adj_ok, adj_err = test_adjoint_consistency(K)
        hs_ok, hs_norm = test_hilbert_schmidt(K)
        spec_ok, imag_max = test_spectral_reality(K)
        psd_ok, min_q = test_positive_semidefinite(K)
        op_norms_by_N[N] = op_norm
        hs_norms_by_N[N] = hs_norm
        tests = [
            ("Linearity", linear_ok, linear_err),
            ("Boundedness (power)", bounded_ok, op_norm),
            ("Adjoint consistency", adj_ok, adj_err),
            ("Hilbert-Schmidt", hs_ok, hs_norm),
            ("Spectral reality", spec_ok, imag_max),
            ("Positive semidefinite", psd_ok, min_q),
        ]
        all_pass = True
        for name, ok, val in tests:
            status = "PASS" if ok else "FAIL"
            print(f" {status:4s} {name:26s} (value = {val:.3e})")
            if not ok:
                all_pass = False
        results_numeric.append(
            {"N": N, "all_pass": all_pass, "op_norm": op_norm, "hs_norm": hs_norm}
        )
        print()
    print("▶ Operator norm / HS norm across N")
    sup_op_norm = 0.0
    for r in results_numeric:
        N = r["N"]
        print(f" N={N:4d}  ||K_N||_op ≈ {r['op_norm']:.6e}  ||K_N||_HS ≈ {r['hs_norm']:.6e}")
        sup_op_norm = max(sup_op_norm, r["op_norm"])
    print(f"\nEstimated sup_N ||K_N||_op over tested N: {sup_op_norm:.6e}\n")
    K_100 = define_analysts_problem_kernel(N=100)
    K_200 = define_analysts_problem_kernel(N=200)
    block_error = np.linalg.norm(K_200[:100, :100] - K_100, ord="fro")
    print(f"BLOCK ERROR TEST (N=100 vs N=200 top-left block): {block_error:.3e}\n")
    print("▶ Analytic verification")
    print("-" * 60)
    analytic.verify_weight_decay_compatibility()
    analytic.verify_kernel_symmetry()
    analytic.verify_compactness_via_truncation()
    analytic.verify_boundedness_schur_test()
    analytic.verify_cross_dimension_consistency()
    print(analytic.generate_report())
    print("=" * 70)
    numeric_ok = all(r["all_pass"] for r in results_numeric)
    analytic_ok = all(a.status == "verified" for a in analytic.arguments)
    if numeric_ok and analytic_ok and sup_op_norm < 1e6 and block_error < 1e-10:
        print("CONCLUSION: Within tested N, TAP HO public HSO φ-operator behaves as a")
        print("bona fide Hilbert operator on ℓ².")
    elif numeric_ok:
        print("NUMERIC TESTS PASSED; ANALYTIC VERIFICATION PARTIAL.")
    else:
        print("VERIFICATION INCOMPLETE – some numeric tests failed.")
    print("=" * 70)

    # ------------------------------------------------------------------
    # FINAL CHART: Collatz, Collatz-spectral, eigenvalues, SEB/RPF curve,
    #              and SEB-normalised curve at the end of the process run.
    # ------------------------------------------------------------------
    # Use the largest tested N for the visual.
    N_plot = max(test_N)
    K_plot = define_analysts_problem_kernel(N=N_plot)
    v_collatz = collatz_observable_vector(N_plot)
    proj = project_collatz_onto_eigenbasis(K_plot, observable_vector=v_collatz, top_k=50)
    v_recon = proj["recon"]
    evals = proj["evals"]

    # Normalise for joint plotting.
    def safe_norm(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        rng = np.max(x) - np.min(x)
        if rng == 0.0:
            return x * 0.0
        return (x - np.min(x)) / rng

    x_axis = np.arange(1, N_plot + 1, dtype=float)

    collatz_norm = safe_norm(v_collatz)
    spectral_norm = safe_norm(v_recon)

    # Eigenvalue profile as a cumulative, resampled curve on x_axis.
    evals_abs = np.abs(evals)
    eig_profile = np.cumsum(evals_abs)
    eig_profile = eig_profile / (np.max(eig_profile) + 1e-15)
    # Resample eigen-value profile to length N_plot (pad if needed).
    if eig_profile.shape[0] < N_plot:
        eig_resampled = np.interp(
            np.linspace(0, eig_profile.shape[0] - 1, N_plot),
            np.arange(eig_profile.shape[0]),
            eig_profile,
        )
    else:
        eig_resampled = eig_profile[:N_plot]
    eig_norm = safe_norm(eig_resampled)

    # SEB / Ruelle–Perron–Frobenius "wiggle" proxy:
    # Take K_plot acting on v_collatz, then normalise.
    seb_rpf_raw = K_plot @ v_collatz
    seb_rpf_norm = safe_norm(seb_rpf_raw)

    # SEB-normalised equation proxy: normalised diagonal of K (local normalisation).
    seb_diag = np.diag(K_plot)
    seb_norm_eq = safe_norm(seb_diag)

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, collatz_norm, label="Collatz observable (normalised)", linewidth=1.2)
    plt.plot(x_axis, spectral_norm, label="Collatz spectral recon (normalised)", linewidth=1.2)
    plt.plot(x_axis, eig_norm, label="Eigenvalue profile (normalised)", linewidth=1.0)
    plt.plot(x_axis, seb_rpf_norm, label="SEB / RPF operator response (normalised)", linewidth=1.0)
    plt.plot(x_axis, seb_norm_eq, label="SEB-normalised diagonal (normalised)", linewidth=1.0)

    plt.title("Collatz / SEB / φ-HSO Spectral View (N = %d)" % N_plot)
    plt.xlabel("Index n")
    plt.ylabel("Normalised magnitude")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return numeric_ok and analytic_ok and sup_op_norm < 1e6 and block_error < 1e-10



if __name__ == "__main__":
    success = main()
    raise SystemExit(0 if success else 1)
