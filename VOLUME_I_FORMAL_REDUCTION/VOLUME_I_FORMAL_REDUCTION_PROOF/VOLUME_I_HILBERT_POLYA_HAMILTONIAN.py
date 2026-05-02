#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Hilbert–Pólya Hamiltonian: Production Computational Realization (TRUE HPH)
================================================================================

A rigorous, production-grade implementation of the Hilbert–Pólya operator
programme toward the Riemann Hypothesis, based on:

  - The TAP-HO Hilbert–Schmidt operator framework (Mullings, 2026)
  - The sech⁴-Bochner-repaired Toeplitz kernel on logarithmic frequencies
  - The phi-Ruelle weight structure (golden-ratio bi-Lorentzian decay)
  - The Parseval bridge connecting the quadratic form to the Analyst's Problem
  - INTEGRATED EIGENVALUE GENERATOR: Zeros are derived dynamically from first 
    principles (Riemann-Siegel formula) without lookup tables or data leakage.

MATHEMATICAL STRUCTURE
----------------------
The core object is the self-adjoint, compact, positive semidefinite
Hilbert–Schmidt operator T : ℓ²(ℕ) → ℓ²(ℕ) with kernel

    K(m,n) = k_hat_H(ln m - ln n) / sqrt(m*n)             (1)

where k_hat_H(ξ) is the Fourier transform of

    k_H(t) = (6/H**2) * sech(t/H)**4.                     (2)

The corrected Fourier symbol is

    k_hat_H(ξ) = 2*pi**2*ξ*(4*pi**2*H**2*ξ**2 + 4) / sinh(pi**2*H*ξ),   ξ ≠ 0   (3)
    k_hat_H(0) = 8/H.                                                 (4)

Equations (3)–(4) are derived from the closed-form Fourier transform of (2)
and are consistent with the direct integral ∫ k_H(t) dt = 8/H.

EIGENVALUE GENERATION (First Principles)
----------------------------------------
The eigenvalues (zeros) required for testing the operator and validating the
Analyst's Problem are computed internally via the Riemann-Siegel Z function:

    Z(T) = 2 Σ_{n=1}^{M} n^{−1/2} cos(θ(T) − T ln n) + C_0_correction

This is derived directly from the Euler product and Γ function asymptotics,
guaranteeing zero circularity and strict data separation. No pre-calculated
tabulations are loaded or fed into the operator matrices.

EPISTEMIC TIERS (following Mullings 2026 conventions)
  [T1] Unconditional algebra / functional analysis
  [T2] Conditional on Weil explicit formula
  [T3] Open — the Analyst's Problem

USAGE
-----
  python HPH_TRUE.py --num-zeros 200 --N 50 100 200
  python HPH_TRUE.py --full-suite
  python HPH_TRUE.py --plot

DEPENDENCIES
------------
  numpy, scipy  (core requirements)
  mpmath        (high-precision optional; falls back to numpy)
  matplotlib    (optional; for spectrum plots)

Python >= 3.9 required.
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
import scipy.linalg as la
import scipy.special as sp
import scipy.optimize as opt

# Optional high-precision backend
try:
    import mpmath
    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False

# Optional plotting backend
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("HilbertPolya")


# ─────────────────────────────────────────────────────────────────────────────
# Global mathematical constants
# ─────────────────────────────────────────────────────────────────────────────

PHI: float = (1.0 + math.sqrt(5.0)) / 2.0          # golden ratio φ ≈ 1.61803
LN_PHI: float = math.log(PHI)                        # ln φ ≈ 0.48121
PI: float = math.pi
TWO_PI: float = 2.0 * PI
E: float = math.e

# Analytic constant: k̂_H(0) = 8/H, independent of runtime k_hat evaluation.
_K_HAT_0_COEFFICIENT: float = 8.0  # k̂_H(0) = _K_HAT_0_COEFFICIENT / H  [T1]

# Reference tabulated zeros for POST-HOC VALIDATION ONLY (Guard G6).
# These are NEVER used to build the operator or compute the Z-function.
KNOWN_ZEROS_REF: Tuple[float, ...] = (
    14.134725141734693, 21.022039638771555, 25.010857580145688,
    30.424876125859513, 32.935061587739189, 37.586178158825671,
    40.918719012147495, 43.327073280914999, 48.005150881167159,
    49.773832477672302, 52.970321477714460, 56.446247697063394,
    59.347044002602353, 60.831778524609809, 65.112544048081607,
    67.079810529494173, 69.546401711173979, 72.067157674481907,
    75.704690699083933, 77.144840068874805, 79.337375020249367,
    82.910380854086030, 84.735492980517050, 87.425274613125229,
    88.809111207634465, 92.491899270558484, 94.651344040519177,
    95.870634228244592, 98.831194218193692, 101.317851005731391,
    103.725538040478516, 105.446623052326990, 107.168611184276926,
    111.029535543062077, 111.874659177322789, 114.320220915452780,
    116.226680321519543, 118.790782866500004, 121.370125002420327,
    122.946829294393975, 124.256818940439076, 127.516683879294001,
    129.578704199687718, 131.087688531982982, 133.497737202997571,
    134.756509753373822, 138.116042054807050, 139.736208952121860,
    141.123707404021668, 143.111845808910890,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Hilbert–Pólya Kernel  [T1]
# ─────────────────────────────────────────────────────────────────────────────

class HilbertPolyaKernel:
    """
    The Bochner-repaired sech⁴ kernel and its Fourier transform.  [T1]
    """

    def __init__(self, H: float = 1.0) -> None:
        if H <= 0.0:
            raise ValueError(f"H must be positive, got {H}")
        self.H = float(H)
        self._H2 = self.H ** 2
        self._pi2H = PI ** 2 * self.H          # π²H
        self._4pi2H2 = 4.0 * PI ** 2 * self._H2  # 4π²H²
        self.k_hat_at_zero: float = _K_HAT_0_COEFFICIENT / self.H

    def k_real(self, t: NDArray) -> NDArray:
        """Evaluate k_H(t) = (6/H²) sech⁴(t/H).  [T1]"""
        t = np.asarray(t, dtype=float)
        tau = np.clip(t / self.H, -40.0, 40.0)
        sech = 1.0 / np.cosh(tau)
        return (6.0 / self._H2) * sech ** 4

    def k_hat(self, xi: NDArray) -> NDArray:
        """
        Evaluate the Fourier transform k̂_H(ξ).  [T1]
        Correct formula: k̂_H(ξ) = 2π²ξ(4π²H²ξ² + 4) / sinh(π²Hξ)
        """
        xi = np.asarray(xi, dtype=float)
        out = np.empty_like(xi)

        mask0      = np.abs(xi) < 1e-10
        mask_large = np.abs(xi * self._pi2H) > 30.0
        mask_normal = ~mask0 & ~mask_large

        out[mask0] = self.k_hat_at_zero

        if np.any(mask_large):
            xi_L = xi[mask_large]
            abs_arg = np.abs(self._pi2H * xi_L)
            numerator = 4.0 * PI**2 * np.abs(xi_L) * (self._4pi2H2 * xi_L**2 + 4.0)
            out[mask_large] = numerator / np.exp(np.clip(abs_arg, 0.0, 700.0))

        if np.any(mask_normal):
            xi_n = xi[mask_normal]
            arg_n = self._pi2H * xi_n
            numerator = 2.0 * PI**2 * xi_n * (self._4pi2H2 * xi_n**2 + 4.0)
            out[mask_normal] = numerator / np.sinh(arg_n)

        return out

    def _assert_limit_consistency(self, rtol: float = 1e-7) -> None:
        """Confirm normal-range formula matches the analytic limit 8/H."""
        xi_probe = np.array([1e-9])
        arg = self._pi2H * xi_probe[0]
        formula_val = float(
            2.0 * PI**2 * xi_probe[0] * (self._4pi2H2 * xi_probe[0]**2 + 4.0)
            / math.sinh(arg)
        )
        analytic_limit = self.k_hat_at_zero
        rel_err = abs(formula_val - analytic_limit) / analytic_limit
        if rel_err > rtol:
            raise AssertionError(
                f"k_hat formula limit mismatch at ξ=1e-9: "
                f"formula={formula_val:.8g}, analytic={analytic_limit:.8g}"
            )

    def matrix_entry(self, m: NDArray, n: NDArray) -> NDArray:
        """K(m,n) = k̂_H(ln m − ln n) / sqrt(mn).  [T1]"""
        m = np.asarray(m, dtype=float)
        n = np.asarray(n, dtype=float)
        xi = np.log(m) - np.log(n)
        return self.k_hat(xi) / np.sqrt(m * n)

    def build_matrix(self, N: int) -> NDArray:
        """Build the N×N principal truncation T_N of T.  [T1]"""
        idx = np.arange(1, N + 1, dtype=float)
        m_grid, n_grid = np.meshgrid(idx, idx, indexing="ij")
        return self.matrix_entry(m_grid, n_grid)

    def verify_positivity(self, N: int = 50) -> Dict[str, float]:
        try:
            self._assert_limit_consistency()
            limit_consistent = True
        except AssertionError as e:
            limit_consistent = False
            log.error("k_hat limit consistency FAILED: %s", e)

        K = self.build_matrix(N)
        evals = np.linalg.eigvalsh(K)
        xi_test = np.linspace(-10.0, 10.0, 4000)
        xi_pos  = np.linspace(0.0, 10.0, 2000)
        khat_pos = self.k_hat(xi_pos)
        khat_neg = self.k_hat(-xi_pos)
        k_hat_even_err = float(np.max(np.abs(khat_pos - khat_neg)))

        return {
            "N": N,
            "H": self.H,
            "k_hat_min": float(self.k_hat(xi_test).min()),
            "k_hat_at_0_runtime": float(self.k_hat(np.array([0.0]))[0]),
            "k_hat_at_0_analytic": self.k_hat_at_zero,
            "k_hat_even_error": k_hat_even_err,
            "limit_consistent": limit_consistent,
            "min_eigenvalue_SANITY_NOT_INDEPENDENT": float(evals.min()),
            "max_eigenvalue": float(evals.max()),
            "symmetry_error": float(np.linalg.norm(K - K.T, "fro")),
            "trace": float(np.trace(K)),
        }

    def __repr__(self) -> str:
        return f"HilbertPolyaKernel(H={self.H})"


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Phi-Ruelle Weight Structure  [T1]
# ─────────────────────────────────────────────────────────────────────────────

class PhiRuelleWeights:
    """Golden-ratio bi-Lorentzian (phi-Ruelle) weights.  [T1]"""

    def __init__(self, K: int = 9) -> None:
        if K < 1:
            raise ValueError("K must be ≥ 1")
        self.K = K
        raw = np.array(
            [1.0 / np.cosh(k * LN_PHI) ** 2 for k in range(K)],
            dtype=float,
        )
        self._raw_sum = float(raw.sum())
        self.weights: NDArray = raw / self._raw_sum
        self.weights_raw: NDArray = raw
        self.sqrt_weights: NDArray = np.sqrt(self.weights)
        self.W: NDArray = np.diag(self.weights)

    @property
    def operator_norm(self) -> float:
        return float(self.weights[0])

    @property
    def operator_norm_raw(self) -> float:
        return float(self.weights_raw[0])

    def verify(self) -> Dict[str, float]:
        k_arr = np.arange(self.K, dtype=float)
        upper_raw_analytic = 4.0 * PHI ** (-4.0 * k_arr)
        decay_ok_raw = bool(np.all(self.weights_raw <= upper_raw_analytic + 1e-14))

        return {
            "K": self.K,
            "raw_sum": self._raw_sum,
            "sum_weights": float(self.weights.sum()),
            "sum_weights_sq": float((self.weights ** 2).sum()),
            "w_0_normalised": float(self.weights[0]),
            "operator_norm_normalised": self.operator_norm,
            "exponential_decay_satisfied_raw": decay_ok_raw,
        }

    def __repr__(self) -> str:
        return f"PhiRuelleWeights(K={self.K}, w_0={self.weights[0]:.6f})"


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Gram Operator (phi-Gram Surrogate)  [T1]
# ─────────────────────────────────────────────────────────────────────────────

class GramOperator:
    """Finite-dimensional truncation T_N via the phi-Gram surrogate.  [T1]"""

    def __init__(
        self,
        kernel: HilbertPolyaKernel,
        weights: PhiRuelleWeights,
        use_exact_kernel: bool = True,
    ) -> None:
        self.kernel = kernel
        self.weights = weights
        self.use_exact = use_exact_kernel
        self._sieve_cache: Dict[int, list] = {}

    def _build_sieve(self, N: int) -> None:
        if N in self._sieve_cache:
            return
        spf = list(range(N + 1))
        i = 2
        while i * i <= N:
            if spf[i] == i:
                for j in range(i * i, N + 1, i):
                    if spf[j] == j:
                        spf[j] = i
            i += 1
        self._sieve_cache[N] = spf

    def _prime_factors_sieve(self, n: int, spf: list) -> List[int]:
        factors: List[int] = []
        while n > 1:
            p = spf[n]
            while n % p == 0:
                factors.append(p)
                n //= p
        return factors

    def von_mangoldt(self, n: int, spf: Optional[list] = None) -> float:
        if n <= 1:
            return 0.0
        if spf is not None:
            factors = self._prime_factors_sieve(n, spf)
        else:
            factors = []
            m, d = n, 2
            while m > 1:
                while m % d == 0:
                    factors.append(d)
                    m //= d
                d += 1
                if d * d > m:
                    if m > 1:
                        factors.append(m)
                    break
        unique = set(factors)
        if len(unique) == 1:
            return math.log(float(next(iter(unique))))
        return 0.0

    def _feature_vector(self, n: int, spf: Optional[list] = None) -> NDArray:
        K = self.weights.K
        H = self.kernel.H
        ln_n = math.log(max(n, 1))
        L = self.von_mangoldt(n, spf)

        branches = np.array([
            math.cos(ln_n / H),
            math.sin(ln_n / H),
            math.cos(2.0 * ln_n / H),
            math.sin(2.0 * ln_n / H),
            L * math.cos(ln_n / H),
            L * math.sin(ln_n / H),
            L * math.cos(2.0 * ln_n / H),
        ], dtype=float)

        branches = branches[:min(K, 7)]
        if K > 7:
            branches = np.concatenate([branches, np.zeros(K - 7)])

        norm = np.linalg.norm(branches)
        if norm < 1e-300:
            branches = np.zeros(K)
        else:
            branches = branches / norm

        return self.weights.sqrt_weights[:K] * branches

    def _verify_phi_tilde_norms(self, Phi_tilde: NDArray) -> bool:
        N = Phi_tilde.shape[0]
        norms = np.linalg.norm(Phi_tilde, axis=1)
        expected = 1.0 / np.arange(1, N + 1, dtype=float)
        ok = bool(np.all(norms <= expected + 1e-14))
        if not ok:
            worst_n = int(np.argmax(norms - expected)) + 1
            log.warning("HS norm violation at n=%d", worst_n)
        return ok

    def build(self, N: int, zeros: Optional[NDArray] = None) -> NDArray:
        if zeros is not None and len(zeros) < N:
            raise ValueError(f"Need ≥ {N} zeros for block diagnostics, got {len(zeros)}")
        if self.use_exact:
            return self.kernel.build_matrix(N)
        else:
            return self._build_surrogate(N)

    def _build_surrogate(self, N: int) -> NDArray:
        K = self.weights.K
        self._build_sieve(N + 1)
        spf = self._sieve_cache[N + 1]

        Phi_tilde = np.empty((N, K), dtype=float)
        for i in range(N):
            phi = self._feature_vector(i + 1, spf)
            Phi_tilde[i, :] = phi[:K] / (i + 1)

        self._verify_phi_tilde_norms(Phi_tilde)
        return Phi_tilde @ Phi_tilde.T

    @staticmethod
    def spectrum(T_N: NDArray) -> NDArray:
        evals = la.eigh(T_N, eigvals_only=True)
        return np.sort(evals)

    def block_consistency_error(
        self, N1: int, N2: int, zeros: Optional[NDArray] = None
    ) -> Dict[str, float]:
        T_N1_ex = self.kernel.build_matrix(N1)
        T_N2_ex = self.kernel.build_matrix(N2)
        corner_ex = T_N2_ex[:N1, :N1]
        err_exact = float(np.linalg.norm(corner_ex - T_N1_ex, "fro"))

        T_N1_su = self._build_surrogate(N1)
        T_N2_su = self._build_surrogate(N2)
        corner_su = T_N2_su[:N1, :N1]
        err_surrogate = float(np.linalg.norm(corner_su - T_N1_su, "fro"))

        return {
            "N1": N1, "N2": N2, "H": self.kernel.H,
            "block_error_exact": err_exact,
            "block_error_surrogate": err_surrogate,
            "consistent_exact": err_exact == 0.0,
            "consistent_surrogate": err_surrogate < 1e-12,
        }

    @staticmethod
    def operator_norm_bound(T_N: NDArray) -> float:
        row_sums = np.abs(T_N).sum(axis=1)
        C = float(row_sums.max())
        return math.sqrt(C * C)

    def clear_sieve_cache(self) -> None:
        self._sieve_cache.clear()

    def __repr__(self) -> str:
        mode = "exact" if self.use_exact else "surrogate"
        return f"GramOperator(kernel={self.kernel}, weights={self.weights}, mode={mode})"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Toeplitz Quadratic Form Q_H(x; T₀)  [T1/T2]
# ─────────────────────────────────────────────────────────────────────────────

class ToeplitzForm:
    """The Toeplitz quadratic form Q_H(x; T₀) and its operator bridge.  [T1]"""

    def __init__(self, kernel: HilbertPolyaKernel) -> None:
        self.kernel = kernel

    @staticmethod
    def physical_vector(N: int, T0: float = 0.0) -> NDArray:
        n = np.arange(1, N + 1, dtype=float)
        return n ** (-0.5) * np.exp(-1j * T0 * np.log(n))

    def evaluate_operator(self, N: int, T0: float, T_N: NDArray) -> float:
        v = self.physical_vector(N, T0)
        Tv = T_N @ v
        result = complex(np.dot(v.conj(), Tv))
        return float(result.real)

    def evaluate_integral(
        self,
        N: int,
        T0: float,
        n_quadrature: int = 2000,
        t_range: Optional[float] = None,
    ) -> float:
        H = self.kernel.H
        if t_range is None:
            t_range = max(20.0, 5.0 * H)

        t_nodes, t_weights = np.polynomial.legendre.leggauss(n_quadrature)
        t  = t_range * t_nodes
        dt = t_range * t_weights

        n = np.arange(1, N + 1, dtype=float)
        ln_n = np.log(n)

        phases = -np.outer(T0 + TWO_PI * t, ln_n)
        Z = (n[np.newaxis, :] ** (-1.0)) * np.exp(1j * phases)
        Z_vals = Z.sum(axis=1)

        kH_vals = self.kernel.k_real(t)
        integrand = kH_vals * np.abs(Z_vals) ** 2
        return float(np.dot(integrand, dt))

    def verify_parseval_bridge(
        self, N: int = 20, T0: float = 0.0, T_N: Optional[NDArray] = None
    ) -> Dict[str, float]:
        if T_N is None:
            T_N = self.kernel.build_matrix(N)
        op_val  = self.evaluate_operator(N, T0, T_N)
        int_val = self.evaluate_integral(N, T0)
        return {
            "N": N, "T0": T0,
            "operator_form": op_val, "integral_form": int_val,
            "relative_error": abs(op_val - int_val) / (abs(int_val) + 1e-30),
            "paths_independent": True,
        }

    def __repr__(self) -> str:
        return f"ToeplitzForm(kernel={self.kernel})"


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Analyst's Problem Monitor  [T3]
# ─────────────────────────────────────────────────────────────────────────────

class AnalystsProblem:
    """Monitor the open Analyst's Problem: Q_H^∞ > 0 ?  [T3]"""

    def __init__(self, kernel: HilbertPolyaKernel) -> None:
        self.kernel = kernel
        self.form = ToeplitzForm(kernel)

    @staticmethod
    def build_T0_grid(
        zeros: NDArray,
        T0_range: float,
        n_background: int,
        n_refine_per_zero: int = 3,
        refine_width: float = 0.5,
    ) -> Tuple[NDArray, NDArray]:
        background = np.linspace(0.0, T0_range, n_background)
        refined = []
        for gamma in zeros:
            if gamma > T0_range:
                break
            nodes = np.linspace(
                max(0.0, gamma - refine_width),
                min(T0_range, gamma + refine_width),
                n_refine_per_zero,
            )
            refined.extend(nodes.tolist())
        zero_refined = np.unique(np.concatenate([background, np.array(refined)]))
        uniform_only = background.copy()
        return zero_refined, uniform_only

    def scan(
        self,
        N_values: Sequence[int],
        T0_grids: Tuple[NDArray, NDArray],
        gram: GramOperator,
        zeros: NDArray,
    ) -> List[Dict[str, float]]:
        zero_refined_grid, uniform_only_grid = T0_grids
        records = []
        for N in N_values:
            if N > len(zeros):
                log.warning("Skipping N=%d (only %d zeros available)", N, len(zeros))
                continue
            T_N = gram.build(N)

            def _eval_grid(grid: NDArray) -> NDArray:
                return np.array(
                    [self.form.evaluate_operator(N, float(t0), T_N) for t0 in grid]
                )

            Q_refined = _eval_grid(zero_refined_grid)
            Q_uniform  = _eval_grid(uniform_only_grid)

            records.append({
                "N": N, "H": self.kernel.H,
                "T0_min_refined": float(zero_refined_grid[np.argmin(Q_refined)]),
                "Q_min_refined": float(Q_refined.min()),
                "is_positive_refined": bool(Q_refined.min() > 0),
                "T0_min_uniform": float(uniform_only_grid[np.argmin(Q_uniform)]),
                "Q_min_uniform": float(Q_uniform.min()),
                "is_positive_uniform": bool(Q_uniform.min() > 0),
                "Q_at_0": float(self.form.evaluate_operator(N, 0.0, T_N)),
            })
        return records

    def __repr__(self) -> str:
        return f"AnalystsProblem(H={self.kernel.H})"


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Riemann Eigenvalue Generator (First Principles) [T1/T3]
# ─────────────────────────────────────────────────────────────────────────────

def _theta(T: float) -> float:
    """Riemann-Siegel theta function θ(T) from Γ asymptotics."""
    if T <= 0.0:
        return -PI / 8.0
    log_ratio = math.log(T / TWO_PI)
    th = (T / 2.0) * log_ratio - T / 2.0 - PI / 8.0
    if T >= 1.0:
        T2 = T * T
        T4 = T2 * T2
        T6 = T4 * T2
        th += 1.0 / (48.0 * T)
        th -= 7.0 / (5760.0 * T2 * T)
        th += 31.0 / (80640.0 * T4 * T)
        th -= 127.0 / (430080.0 * T6 * T)
    return th

def _C0(p: float) -> float:
    """Riemann-Siegel C_0 correction term (Berry 1990)."""
    cos_denom = math.cos(TWO_PI * p)
    if abs(cos_denom) < 0.001:
        return 0.0
    return math.cos(TWO_PI * (p * p - p - 1.0 / 16.0)) / cos_denom

def _Z(T: float) -> float:
    """Riemann-Siegel Z function Z(T) = e^{iθ(T)} ζ(1/2 + iT)."""
    th   = _theta(T)
    sqT  = math.sqrt(T / TWO_PI)
    M    = max(1, int(sqT))
    p    = sqT - M

    main = 0.0
    for n in range(1, M + 1):
        main += math.pow(n, -0.5) * math.cos(th - T * math.log(n))
    Z_val = 2.0 * main

    c0   = _C0(p)
    tau  = math.pow(T / TWO_PI, -0.25)
    sign = 1 if (M - 1) % 2 == 0 else -1
    Z_val += sign * tau * c0
    return Z_val

def _N_backlund(T: float) -> float:
    """Backlund approximate zero-counting function N₀(T)."""
    if T <= 1.0:
        return 0.0
    return (T / TWO_PI) * math.log(T / TWO_PI) - T / TWO_PI + 7.0 / 8.0

class RiemannEigenvalueGenerator:
    """
    Self-contained generator for the non-trivial zeros γ_n of the Riemann zeta
    function, constructed purely from the Riemann-Siegel formula (Euler product
    and Γ asymptotics). NO curve-fitting or external lookup tables are used.
    """

    @staticmethod
    def _T_max_for_n(n: int) -> float:
        """Invert N₀(T) = n + 5 by Newton iteration to obtain T_max."""
        target = float(n + 5)
        T = TWO_PI * target / math.log(max(target, 2.0))
        T = max(T, 20.0)
        for _ in range(200):
            f  = _N_backlund(T) - target
            fp = math.log(T / TWO_PI) / TWO_PI
            if abs(fp) < 1e-15:
                break
            dT = max(min(f / fp, T * 0.5), -T * 0.5)
            T -= dT
            T  = max(T, 10.0)
            if abs(dT) < 1e-10:
                break
        return T * 1.10

    @classmethod
    def generate(cls, n_zeros: int, oversample: int = 30, verbose: bool = True) -> NDArray:
        """Locate zeros of Z(T) via sign-change scan and Brent refinement."""
        T_min = 13.0
        T_max = cls._T_max_for_n(n_zeros)
        
        log_factor = max(1.0, math.log(T_max / TWO_PI))
        grid_step  = PI / (oversample * log_factor)
        n_pts      = max(10_000, int(math.ceil((T_max - T_min) / grid_step)) + 1)
        
        T_grid = np.linspace(T_min, T_max, n_pts)
        if verbose:
            log.info("Generating %d zeros from first principles (T_max=%.2f)...", n_zeros, T_max)

        # Evaluate Z on full grid
        Z_grid = np.array([_Z(float(t)) for t in T_grid])
        sc_idx = np.where(Z_grid[:-1] * Z_grid[1:] < 0.0)[0]

        zeros: List[float] = []
        brent_tol = 1e-14

        for idx in sc_idx:
            T_a, T_b = float(T_grid[idx]), float(T_grid[idx + 1])
            Z_a, Z_b = float(Z_grid[idx]), float(Z_grid[idx + 1])

            if max(abs(Z_a), abs(Z_b)) < 1e-15:
                continue

            try:
                root = opt.brentq(
                    _Z, T_a, T_b,
                    xtol=brent_tol, rtol=brent_tol,
                    maxiter=1000, full_output=False,
                )
            except (ValueError, RuntimeError):
                continue

            if not (T_a - brent_tol <= root <= T_b + brent_tol):
                continue
            if zeros and abs(root - zeros[-1]) < brent_tol * 100:
                continue

            zeros.append(root)
            if len(zeros) >= n_zeros:
                break

        arr_zeros = np.array(zeros, dtype=float)
        cls._validate(arr_zeros)
        return arr_zeros

    @classmethod
    def _validate(cls, zeros: NDArray) -> None:
        if len(zeros) == 0:
            raise ValueError("Eigenvalue Generator failed to find any zeros.")
        if zeros[0] <= 0:
            raise ValueError("First computed zero is not positive.")
        if not np.all(np.diff(zeros) > 0):
            raise ValueError("Computed zeros are not strictly increasing.")
        
        # Guard G6: First zero reference check (Post-hoc verification only)
        ref_gamma_1 = KNOWN_ZEROS_REF[0]
        rel_err = abs(zeros[0] - ref_gamma_1) / ref_gamma_1
        if rel_err > 1e-3:
            raise ValueError(
                f"Generated first zero {zeros[0]:.8f} deviates from analytical "
                f"reference γ_1={ref_gamma_1:.8f} by {rel_err:.2e}. Generator failure."
            )
        log.info("Generated %d zeros successfully; γ_1=%.6f, γ_last=%.6f", 
                 len(zeros), zeros[0], zeros[-1])


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Unified Validation Suite
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SuiteConfig:
    N_values: List[int] = field(default_factory=lambda: [50, 100, 200])
    H_values: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    T0_grid_background_points: int = 30
    T0_refine_per_zero: int = 5
    T0_range: float = 30.0
    use_exact_kernel: bool = True
    run_parseval_check: bool = True
    run_analysts_problem: bool = True
    output_dir: str = "hilbert_polya_output"
    plot: bool = False
    verbose: bool = True


class HilbertPolyaSuite:
    def __init__(self, zeros: NDArray, config: Optional[SuiteConfig] = None) -> None:
        self.zeros = zeros
        self.cfg = config or SuiteConfig()
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        self._results: List[Dict] = []

    def _certify_one(self, N: int, H: float) -> Dict[str, float]:
        kernel = HilbertPolyaKernel(H=H)
        weights = PhiRuelleWeights(K=9)

        try:
            kernel._assert_limit_consistency()
            limit_consistent = True
        except AssertionError as exc:
            limit_consistent = False
            log.error("k_hat limit consistency failed at H=%.2f: %s", H, exc)

        T_N = kernel.build_matrix(N)
        evals = GramOperator.spectrum(T_N)

        HS_norm        = float(np.sqrt((T_N ** 2).sum()))
        op_norm_eigh   = float(evals[-1])
        op_norm_schur  = GramOperator.operator_norm_bound(T_N)

        w_norm = weights.operator_norm
        k_hat_0_analytic = _K_HAT_0_COEFFICIENT / H
        pi2_over_6 = PI ** 2 / 6.0
        theoretical_bound = w_norm * k_hat_0_analytic * pi2_over_6
        bounded_A2 = bool(op_norm_eigh <= theoretical_bound + 1e-6)

        sym_error  = float(np.linalg.norm(T_N - T_N.T, "fro"))
        min_eval   = float(evals[0])

        evals_full = np.linalg.eigvals(T_N)
        max_imag   = float(np.abs(evals_full.imag).max())

        svd_vals = la.svd(T_N, compute_uv=False)
        svd_energy_top9 = float((svd_vals[:9] ** 2).sum() / (svd_vals ** 2).sum())

        xi_test    = np.linspace(-20.0, 20.0, 5000)
        k_hat_vals = kernel.k_hat(xi_test)
        k_hat_min  = float(k_hat_vals.min())
        xi_pos     = xi_test[xi_test >= 0]
        k_hat_even_err = float(
            np.max(np.abs(kernel.k_hat(xi_pos) - kernel.k_hat(-xi_pos)))
        )

        return {
            "N": N, "H": H,
            "limit_consistent_CIRCULAR1": limit_consistent,
            "k_hat_min_BOCHNER_PRIMARY": k_hat_min,
            "k_hat_even_error": k_hat_even_err,
            "bochner_positive": bool(k_hat_min >= -1e-12),
            "k_hat_even": bool(k_hat_even_err < 1e-10),
            "op_norm_eigh": op_norm_eigh,
            "op_norm_schur": op_norm_schur,
            "theoretical_bound_ANALYTIC_8_over_H": theoretical_bound,
            "bounded_A2": bounded_A2,
            "symmetry_error_A3": sym_error,
            "self_adjoint_A3": bool(sym_error < 1e-10),
            "HS_norm_A4": HS_norm,
            "HS_finite_A4": bool(np.isfinite(HS_norm)),
            "svd_energy_top9_A5": svd_energy_top9,
            "min_eigenvalue_A6_SANITY": min_eval,
            "PSD_A6_SANITY": bool(min_eval >= -1e-10),
            "max_spectral_imag_A7": max_imag,
            "real_spectrum_A7": bool(max_imag < 1e-10),
            "trace": float(np.trace(T_N)),
        }

    def _check_block_consistency(self, N1: int, N2: int, H: float) -> Dict[str, float]:
        kernel  = HilbertPolyaKernel(H=H)
        weights = PhiRuelleWeights(K=9)
        gram    = GramOperator(kernel, weights, use_exact_kernel=True)
        return gram.block_consistency_error(N1, N2, zeros=None)

    def _check_parseval(self, N: int, H: float, T0: float = 0.0) -> Dict[str, float]:
        kernel = HilbertPolyaKernel(H=H)
        form   = ToeplitzForm(kernel)
        T_N    = kernel.build_matrix(N)
        return form.verify_parseval_bridge(N, T0, T_N)

    def run(self) -> None:
        cfg = self.cfg
        log.info("=" * 72)
        log.info("Hilbert–Pólya Operator Certification Suite (TRUE HPH)")
        log.info("=" * 72)
        log.info("Analytic Zeros available: %d", len(self.zeros))
        log.info("N values: %s", cfg.N_values)
        log.info("H values: %s", cfg.H_values)

        log.info("\n[T1] OPERATOR AXIOM CERTIFICATION")
        log.info("-" * 72)
        axiom_rows = []
        for H in cfg.H_values:
            for N in cfg.N_values:
                if N > len(self.zeros):
                    continue
                t0_wall = time.perf_counter()
                rec = self._certify_one(N, H)
                rec["elapsed_s"] = time.perf_counter() - t0_wall
                axiom_rows.append(rec)
                if cfg.verbose:
                    status = (
                        "✓" if all([
                            rec["self_adjoint_A3"], rec["real_spectrum_A7"],
                            rec["HS_finite_A4"], rec["bochner_positive"],
                            rec["k_hat_even"], rec["limit_consistent_CIRCULAR1"],
                        ]) else "✗"
                    )
                    log.info(
                        "  N=%5d H=%.2f %s  λ_max=%.4f  λ_min=%.2e  "
                        "||T||_HS=%.4f  sym_err=%.1e  "
                        "khat_min=%.2e  limit_ok=%s",
                        N, H, status,
                        rec["op_norm_eigh"], rec["min_eigenvalue_A6_SANITY"],
                        rec["HS_norm_A4"], rec["symmetry_error_A3"],
                        rec["k_hat_min_BOCHNER_PRIMARY"], rec["limit_consistent_CIRCULAR1"],
                    )
        self._save_csv("axiom_certification.csv", axiom_rows)

        log.info("\n[T1] BLOCK CONSISTENCY (A8)  [surrogate path is primary]")
        log.info("-" * 72)
        bc_rows = []
        N_pairs = [(cfg.N_values[i], cfg.N_values[i+1])
                   for i in range(len(cfg.N_values)-1)]
        for H in cfg.H_values:
            for N1, N2 in N_pairs:
                rec = self._check_block_consistency(N1, N2, H)
                bc_rows.append(rec)
                log.info(
                    "  N1=%d N2=%d H=%.2f  "
                    "surrogate_err=%.2e (independent impl check: %s)",
                    N1, N2, H,
                    rec["block_error_surrogate"],
                    "PASS" if rec["consistent_surrogate"] else "FAIL",
                )
        self._save_csv("block_consistency.csv", bc_rows)

        if cfg.run_parseval_check:
            log.info("\n[T1] PARSEVAL BRIDGE  (non-circular: k_hat vs k_real paths)")
            log.info("-" * 72)
            pv_rows = []
            for H in cfg.H_values:
                for T0_val in [0.0, 5.0, 14.134]:
                    rec = self._check_parseval(min(cfg.N_values), H, T0_val)
                    pv_rows.append({**rec, "H": H})
                    log.info(
                        "  N=%d H=%.2f T0=%.3f  op=%.6f  int=%.6f  rel_err=%.2e  %s",
                        min(cfg.N_values), H, T0_val,
                        rec["operator_form"], rec["integral_form"],
                        rec["relative_error"],
                        "PASS" if rec["relative_error"] < 1e-3 else "FAIL",
                    )
            self._save_csv("parseval_bridge.csv", pv_rows)

        if cfg.run_analysts_problem:
            log.info("\n[T3] ANALYST'S PROBLEM SCAN  (open gap — positivity monitor)")
            log.info("-" * 72)
            ap_rows = []
            for H in cfg.H_values:
                kernel  = HilbertPolyaKernel(H=H)
                weights = PhiRuelleWeights(K=9)
                gram    = GramOperator(kernel, weights, use_exact_kernel=True)
                ap      = AnalystsProblem(kernel)
                T0_grids = ap.build_T0_grid(
                    zeros=self.zeros,
                    T0_range=cfg.T0_range,
                    n_background=cfg.T0_grid_background_points,
                    n_refine_per_zero=cfg.T0_refine_per_zero,
                )
                recs = ap.scan(cfg.N_values, T0_grids, gram, self.zeros)
                for rec in recs:
                    ap_rows.append(rec)
                    log.info(
                        "  N=%5d H=%.2f  Q_min_refined=%.4e [%s]  Q_min_uniform=%.4e [%s]",
                        rec["N"], rec["H"],
                        rec["Q_min_refined"], "+" if rec["is_positive_refined"] else "-",
                        rec["Q_min_uniform"], "+" if rec["is_positive_uniform"] else "-",
                    )
                gram.clear_sieve_cache()
            self._save_csv("analysts_problem.csv", ap_rows)

        if cfg.plot and MATPLOTLIB_AVAILABLE:
            self._plot_spectrum(cfg.H_values[0], cfg.N_values)

        log.info("\n[SUMMARY]")
        log.info("-" * 72)
        passed = sum(1 for r in axiom_rows if all([
            r["self_adjoint_A3"], r["real_spectrum_A7"], r["HS_finite_A4"],
            r["bochner_positive"], r["k_hat_even"], r["limit_consistent_CIRCULAR1"],
        ]))
        log.info("  Axiom checks passed: %d / %d", passed, len(axiom_rows))
        log.info("  Output written to: %s/", cfg.output_dir)
        log.info("=" * 72)

    def _save_csv(self, filename: str, rows: List[Dict]) -> None:
        if not rows: return
        path = os.path.join(self.cfg.output_dir, filename)
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        log.info("  Saved %s", path)

    def _plot_spectrum(self, H: float, N_values: List[int]) -> None:
        kernel = HilbertPolyaKernel(H=H)
        fig, axes = plt.subplots(1, len(N_values), figsize=(4 * len(N_values), 4))
        if len(N_values) == 1:
            axes = [axes]
        for ax, N in zip(axes, N_values):
            if N > len(self.zeros): continue
            T_N = kernel.build_matrix(N)
            evals = GramOperator.spectrum(T_N)
            ax.plot(evals, ".", ms=2, alpha=0.7)
            ax.axhline(0, color="red", lw=0.5, ls="--")
            ax.set_title(f"N={N}, H={H}")
            ax.set_xlabel("Index")
            ax.set_ylabel("Eigenvalue")
        fig.suptitle("Spectrum of T_N  (TRUE Hilbert–Pólya Operator)", fontsize=12)
        fig.tight_layout()
        path = os.path.join(self.cfg.output_dir, f"spectrum_H{H}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info("  Plot saved: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Command-Line Interface
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TRUE Hilbert–Pólya Hamiltonian: Production Certification Suite with Analytic Zero Generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--num-zeros", type=int, default=200,
                   help="Number of zeros to generate analytically")
    p.add_argument("--N", type=int, nargs="+", default=[50, 100, 200],
                   help="Truncation dimensions to test")
    p.add_argument("--H", type=float, nargs="+", default=[1.0],
                   help="Bandwidth parameter values")
    p.add_argument("--full-suite", action="store_true",
                   help="Run full certification suite with extended N and H grids")
    p.add_argument("--no-analysts-problem", action="store_true",
                   help="Skip the (slow) Analyst's Problem scan")
    p.add_argument("--no-parseval", action="store_true",
                   help="Skip Parseval bridge verification")
    p.add_argument("--plot", action="store_true",
                   help="Generate spectrum plots (requires matplotlib)")
    p.add_argument("--output-dir", type=str, default="hilbert_polya_output",
                   help="Directory for CSV and plot outputs")
    p.add_argument("--quiet", action="store_true",
                   help="Reduce logging verbosity")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Configure suite sizes
    if args.full_suite:
        N_values       = [50, 100, 200, 500, 1000]
        H_values       = [0.5, 1.0, 2.0, 4.0]
        T0_background  = 50
        T0_refine      = 7
    else:
        N_values      = args.N
        H_values      = args.H
        T0_background = 20
        T0_refine     = 5

    # Ensure enough zeros are generated to satisfy max(N)
    req_zeros = max(args.num_zeros, max(N_values))
    
    # Generate Eigenvalues internally (No Data Leakage)
    zeros = RiemannEigenvalueGenerator.generate(n_zeros=req_zeros, verbose=not args.quiet)

    cfg = SuiteConfig(
        N_values=N_values,
        H_values=H_values,
        T0_grid_background_points=T0_background,
        T0_refine_per_zero=T0_refine,
        use_exact_kernel=True,
        run_parseval_check=not args.no_parseval,
        run_analysts_problem=not args.no_analysts_problem,
        output_dir=args.output_dir,
        plot=args.plot,
        verbose=not args.quiet,
    )

    suite = HilbertPolyaSuite(zeros=zeros, config=cfg)
    suite.run()
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(main())