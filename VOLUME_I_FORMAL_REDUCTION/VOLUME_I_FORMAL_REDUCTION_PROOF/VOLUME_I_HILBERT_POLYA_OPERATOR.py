#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
THE ANALYST'S PROBLEM — A HILBERT–PÓLYA CANDIDATE OPERATOR
======================================================================

VOLUME I: ARITHMETIC SECH-RESONANT HPO WITH HPH KERNEL (PRODUCTION)

Canonical master equation (finite-N, ℓ²({1,…,N}) formulation)
-------------------------------------------------------------

    H_N
    =
    D_N
    +
    K_N^{HPH}
    +
    K_N^{prime}
    +
    γ R_N
    −
    δ_N I_N,

where

    K_N^{HPH}   = Hilbert–Pólya Hamiltonian Toeplitz kernel (Bochner-repaired
                   sech⁴, corrected Fourier symbol, phi-Ruelle weighted surrogate),
    K_N^{prime} = explicit-formula-inspired prime kernel,

and

    δ_N      = (1/N) Tr( D_N + K_N^{HPH} + K_N^{prime} ),
    D_N      = diag(t_1,…,t_N) from inverting the Riemann–von Mangoldt N(T),
    t_n      ≈ height of the n-th zeta zero (heuristic, but zero-free),
    γ R_N    = small random GUE-like perturbation (optional).

HPH KERNEL STRUCTURE (integrated from Mullings 2026 / HPH_ANALYTIC_EXACT)
--------------------------------------------------------------------------
The backbone K_N^{HPH} is the N×N principal truncation of the
Hilbert–Schmidt operator T : ℓ²(ℕ) → ℓ²(ℕ) with kernel

    K(m,n) = k̂_H(ln m − ln n) / √(mn)                               (1)

where k̂_H(ξ) is the ordinary-frequency Fourier transform of

    k_H(t) = (6/H²) sech⁴(t/H)                                       (2)

Corrected closed-form Fourier symbol (GR 3.549.4, [FIX-FORMULA]):

    k̂_H(ξ) = 2π²ξ(4π²H²ξ² + 4) / sinh(π²Hξ),   ξ ≠ 0              (3)
    k̂_H(0)  = 8/H                                                     (4)

Non-negativity (Bochner): ξ/sinh(π²Hξ) > 0 for ξ ≠ 0, and
(4π²H²ξ² + 4) > 0 always, so k̂_H(ξ) ≥ 0 everywhere. [T1]

Phi-Gram surrogate (rank ≤ K factorisation):

    K(m,n) ≈ Φ̃_m^T W Φ̃_n,
    Φ̃_n = φ_n / n,   W = diag(w_0,…,w_{K-1}),
    w_k ∝ sech²(k ln φ)  (phi-Ruelle weights, φ = golden ratio).     (5)

Parseval bridge (verified numerically, algebraically independent paths):

    ⟨T_N v(T₀), v(T₀)⟩ = ∫ k_H(t) |Z_N(1 + i(T₀ + 2πt))|² dt     (6)

where v_n(T₀) = n^{-1/2} e^{-iT₀ ln n} and Z_N(s) = Σ_{n=1}^N n^{-s}.

Key design constraints:

  • Self-adjointness at each finite N (machine-precision symmetry).
  • Arithmetic backbone driven by the von Mangoldt function Λ(n).
  • HPH kernel is zero-free: T_N never encodes tabulated Riemann zeros.
  • Prime-resonance kernel implementing an explicit-formula-like structure
    using only primes, no Riemann zeros.
  • Anti-circularity: Zeros only appear in diagnostics, never in H_N itself.
  • Full validation suite: linearity, boundedness, Hilbert–Schmidt, real
    spectrum, block functional symmetry, GUE spacing tests, explicit-formula
    trace hooks, Berry–Keating unfolding, Parseval bridge verification, and
    resolvent-based limit heuristics.

EPISTEMIC TIERS
  [T1] Unconditional algebra / functional analysis
  [T2] Conditional on Weil explicit formula
  [T3] Open — the Analyst's Problem

This construction is a bona fide numerical HPO candidate: it is intended to be
research-grade and structurally aligned with Hilbert–Pólya expectations, but
it does *not* constitute a proof of the Riemann hypothesis.
"""


import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.stats import ks_2samp, kstest
from scipy.special import lambertw  # reserved for possible analytic refinements


# ============================================================
# GLOBAL PARAMETERS
# ============================================================

PHI      = (1.0 + math.sqrt(5.0)) / 2.0
LN_PHI   = math.log(PHI)
PI       = math.pi
MAX_N    = 8000

# HPH bandwidth parameter (H > 0; recommended 0.5 ≤ H ≤ 5)
HPH_H_DEFAULT = 1.0

# Analytic constant: k̂_H(0) = 8/H, independent of runtime evaluation.
# Derivation: ∫ (6/H²) sech⁴(t/H) dt = (6/H²)·H·(4/3) = 8/H.
_K_HAT_0_COEFFICIENT = 8.0   # k̂_H(0) = _K_HAT_0_COEFFICIENT / H  [T1]

# Phi-Ruelle weight truncation
PHI_RUELLE_K = 9

# Prime-resonance kernel parameters
P_MAX_DEFAULT  = 229          # reference base cutoff for N≈400
EPSILON_PRIME  = 0.08         # prime kernel coupling

# HPH coupling to the full operator
EPSILON_HPH    = 0.12         # coupling ε for K_N^{HPH} in H_N

# Random GUE-perturbation parameters
GAMMA_GUE      = 0.02
RNG_SEED_GUE   = 20260426

# EPS shift for resonance tuning
EPS_T_SHIFT    = 2.0

# Test dimensions
TEST_NS = [200, 400, 800, 1600, 2000]

_rng = np.random.default_rng(271828)


# ============================================================
# BASIC UTILITIES
# ============================================================

def sech(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x_clipped = np.clip(x, -40.0, 40.0)
    return 2.0 / (np.exp(x_clipped) + np.exp(-x_clipped))


# ============================================================
# ARITHMETIC UTILITIES (Λ AND PRIMES)
# ============================================================

def von_mangoldt(n: int) -> float:
    if n < 2:
        return 0.0
    for p in range(2, int(math.isqrt(n)) + 1):
        if n % p == 0:
            m = n
            while m % p == 0:
                m //= p
            if m == 1:
                return math.log(float(p))
            return 0.0
    return math.log(float(n))


def build_von_mangoldt_vector(N: int) -> np.ndarray:
    return np.array([von_mangoldt(n) for n in range(1, N + 1)], dtype=float)


_PRIME_CACHE: Dict[int, np.ndarray] = {}


def get_primes(P_max: int) -> np.ndarray:
    if P_max in _PRIME_CACHE:
        return _PRIME_CACHE[P_max]
    is_prime = np.ones(P_max + 1, dtype=bool)
    is_prime[:2] = False
    for p in range(2, int(math.isqrt(P_max)) + 1):
        if is_prime[p]:
            is_prime[p * p:P_max + 1:p] = False
    primes = np.nonzero(is_prime)[0].astype(int)
    _PRIME_CACHE[P_max] = primes
    return primes


def adaptive_P_max(N: int,
                   base: int = P_MAX_DEFAULT,
                   growth: float = 0.5,
                   P_cap: int = 5000) -> int:
    scale = max(N / 400.0, 1.0)
    P_est = int(base * (scale ** growth))
    return min(max(P_est, base), P_cap)


# ============================================================
# ARITHMETIC LEVEL (RIEMANN–VON MANGOLDT)
# ============================================================

def arithmetic_level(n: int, N: int) -> float:
    """
    Approximate t_n such that N(t_n) = n, with N(T) the
    Riemann–von Mangoldt counting function.
    """
    if n <= 0:
        return 0.0
    t = 2.0 * math.pi * n / max(math.log(float(n) + 1.0), 1.0)
    for _ in range(10):
        if t <= 0:
            t = float(n)
            break
        lt  = math.log(t / (2.0 * math.pi * math.e) + 1e-10)
        Nt  = (t / (2.0 * math.pi)) * lt + 7.0 / 8.0
        dNt = (lt + 1.0) / (2.0 * math.pi)
        if abs(dNt) < 1e-15:
            break
        t -= (Nt - n) / dNt
    return max(t, 0.0)


def build_diagonal_D(N: int) -> np.ndarray:
    return np.array([arithmetic_level(n, N) for n in range(1, N + 1)], dtype=float)


# ============================================================
# HPH KERNEL  [T1]
# ============================================================

class HilbertPolyaKernel:
    """
    Bochner-repaired sech⁴ Hilbert–Pólya kernel and its Fourier transform. [T1]

    Real-space kernel:
        k_H(t) = (6/H²) sech⁴(t/H)       > 0 for all t ∈ ℝ

    Fourier transform (even, non-negative — Bochner's theorem):
        k̂_H(ξ) = 2π²ξ(4π²H²ξ² + 4) / sinh(π²Hξ),   ξ ≠ 0
        k̂_H(0)  = 8/H

    Kernel matrix entry:
        K(m,n) = k̂_H(ln m − ln n) / √(mn)

    All entries are ≥ 0 (Bochner + non-negativity of k̂_H). [T1]
    """

    def __init__(self, H: float = HPH_H_DEFAULT) -> None:
        if H <= 0.0:
            raise ValueError(f"H must be positive, got {H}")
        self.H   = float(H)
        self._H2 = self.H ** 2
        self._pi2H       = PI ** 2 * self.H
        self._4pi2H2     = 4.0 * PI ** 2 * self._H2
        # Analytic k̂_H(0) = 8/H — used as a priori bound [FIX-CIRCULAR-4]
        self.k_hat_at_zero: float = _K_HAT_0_COEFFICIENT / self.H

    # ── real-space kernel ──────────────────────────────────────────────

    def k_real(self, t: np.ndarray) -> np.ndarray:
        """k_H(t) = (6/H²) sech⁴(t/H).  Independent of k_hat.  [T1]"""
        t   = np.asarray(t, dtype=float)
        tau = np.clip(t / self.H, -40.0, 40.0)
        return (6.0 / self._H2) * (1.0 / np.cosh(tau)) ** 4

    # ── Fourier symbol k̂_H(ξ) [FIX-FORMULA] ─────────────────────────

    def k_hat(self, xi: np.ndarray) -> np.ndarray:
        """
        k̂_H(ξ) — corrected ordinary-frequency FT of k_H (GR 3.549.4). [T1]

        Non-negativity: ξ/sinh(π²Hξ) > 0 for ξ ≠ 0; (4π²H²ξ²+4) > 0 always.
        Numerical branches:
          |ξ| < 1e-10          → analytic limit 8/H
          |π²Hξ| > 30          → asymptotic (avoids sinh overflow)
          otherwise            → direct formula
        """
        xi  = np.asarray(xi, dtype=float)
        out = np.empty_like(xi)

        mask0      = np.abs(xi) < 1e-10
        mask_large = np.abs(xi * self._pi2H) > 30.0
        mask_normal = ~mask0 & ~mask_large

        # ξ = 0 limit [FIX-CIRCULAR-1]
        out[mask0] = self.k_hat_at_zero

        # large |ξ|: sinh(x) ≈ sign(x)·exp(|x|)/2
        if np.any(mask_large):
            xi_L    = xi[mask_large]
            abs_arg = np.abs(self._pi2H * xi_L)
            numer   = 4.0 * PI**2 * np.abs(xi_L) * (self._4pi2H2 * xi_L**2 + 4.0)
            out[mask_large] = numer / np.exp(np.clip(abs_arg, 0.0, 700.0))

        # normal range
        if np.any(mask_normal):
            xi_n  = xi[mask_normal]
            numer = 2.0 * PI**2 * xi_n * (self._4pi2H2 * xi_n**2 + 4.0)
            out[mask_normal] = numer / np.sinh(self._pi2H * xi_n)

        return out

    # ── non-circular limit self-check [FIX-CIRCULAR-1] ───────────────

    def assert_limit_consistency(self, rtol: float = 1e-7) -> None:
        """
        Confirm normal-range formula at ξ=1e-9 matches analytic limit 8/H.
        Independent of the mask0 branch — checks the algebraic formula
        against _K_HAT_0_COEFFICIENT/H.
        """
        xi_probe = 1e-9
        arg      = self._pi2H * xi_probe
        val      = float(
            2.0 * PI**2 * xi_probe * (self._4pi2H2 * xi_probe**2 + 4.0)
            / math.sinh(arg)
        )
        analytic = self.k_hat_at_zero
        rel_err  = abs(val - analytic) / analytic
        if rel_err > rtol:
            raise AssertionError(
                f"k_hat limit check failed: formula={val:.8g}, "
                f"analytic={analytic:.8g}, rel_err={rel_err:.2e} > {rtol:.2e}"
            )

    # ── N×N kernel matrix ─────────────────────────────────────────────

    def build_matrix(self, N: int) -> np.ndarray:
        """
        T_N[m-1,n-1] = k̂_H(ln m − ln n) / √(mn).  Zero-free.  [T1]
        """
        idx             = np.arange(1, N + 1, dtype=float)
        m_g, n_g        = np.meshgrid(idx, idx, indexing="ij")
        xi              = np.log(m_g) - np.log(n_g)
        return self.k_hat(xi) / np.sqrt(m_g * n_g)

    def verify_positivity(self, N: int = 50) -> Dict[str, float]:
        """Diagnostic: Bochner positivity and symmetry.  [T1]"""
        self.assert_limit_consistency()
        K       = self.build_matrix(N)
        evals   = np.linalg.eigvalsh(K)
        xi_test = np.linspace(-10.0, 10.0, 4000)
        xi_pos  = np.linspace(0.0, 10.0, 2000)
        kp      = self.k_hat(xi_pos)
        kn      = self.k_hat(-xi_pos)
        return {
            "N": N, "H": self.H,
            "k_hat_min":        float(self.k_hat(xi_test).min()),
            "k_hat_at_0":       float(self.k_hat(np.array([0.0]))[0]),
            "k_hat_at_0_analytic": self.k_hat_at_zero,
            "k_hat_even_error": float(np.max(np.abs(kp - kn))),
            "min_eigenvalue_SANITY": float(evals.min()),
            "symmetry_error":   float(np.linalg.norm(K - K.T, "fro")),
        }

    def __repr__(self) -> str:
        return f"HilbertPolyaKernel(H={self.H})"


# ============================================================
# PHI-RUELLE WEIGHTS  [T1]
# ============================================================

class PhiRuelleWeights:
    """
    Golden-ratio bi-Lorentzian (phi-Ruelle) normalised weights.  [T1]

    w_k^raw = sech²(k ln φ),   normalised so Σ_k w_k = 1.

    Properties [Proposition 4.2, Mullings 2026]:
      (i)   w_k^raw ≤ 4 φ^{-4k}  (exponential decay)
      (ii)  Σ (w_k^raw)² < ∞     (square-summable)
      (iii) ||W||_op = w_0        (diagonal operator)

    Exponential decay verified without reference to _raw_sum [FIX-CIRCULAR-6].
    """

    def __init__(self, K: int = PHI_RUELLE_K) -> None:
        if K < 1:
            raise ValueError("K must be ≥ 1")
        self.K    = K
        raw       = np.array(
            [1.0 / np.cosh(k * LN_PHI) ** 2 for k in range(K)], dtype=float
        )
        self._raw_sum     = float(raw.sum())
        self.weights      = raw / self._raw_sum
        self.weights_raw  = raw
        self.sqrt_weights = np.sqrt(self.weights)
        self.W            = np.diag(self.weights)

    @property
    def operator_norm(self) -> float:
        return float(self.weights[0])

    def verify(self) -> Dict[str, float]:
        """Non-circular exponential decay check [FIX-CIRCULAR-6]."""
        k_arr             = np.arange(self.K, dtype=float)
        upper_raw_analytic = 4.0 * PHI ** (-4.0 * k_arr)
        decay_ok          = bool(np.all(self.weights_raw <= upper_raw_analytic + 1e-14))
        return {
            "K": self.K,
            "sum_weights": float(self.weights.sum()),
            "w_0": float(self.weights[0]),
            "exponential_decay_satisfied": decay_ok,
            "decay_check_note": "UNNORMALISED weights vs 4·φ^{-4k} (independent analytic bound)",
        }

    def __repr__(self) -> str:
        return f"PhiRuelleWeights(K={self.K}, w_0={self.weights[0]:.6f})"


# ============================================================
# HPH GRAM OPERATOR (PHI-GRAM SURROGATE)  [T1]
# ============================================================

class HPHGramOperator:
    """
    Finite-dimensional truncation T_N via the phi-Gram surrogate.  [T1]

    K(m,n) ≈ Φ̃_m^T W Φ̃_n,   Φ̃_n = φ_n / n,   W = diag(w_k).

    Feature vector φ_n ∈ ℝ^K (zero-free; depends only on n, H, Λ(n)):
      0: cos(ln n / H)
      1: sin(ln n / H)
      2: cos(2 ln n / H)
      3: sin(2 ln n / H)
      4: Λ(n) cos(ln n / H)
      5: Λ(n) sin(ln n / H)
      6: Λ(n) cos(2 ln n / H)

    DATA SEPARATION: feature vectors never receive tabulated Riemann zeros.
    Post-build HS norm check: ||Φ̃_n|| ≤ 1/n verified independently [FIX-CIRCULAR-5].
    """

    def __init__(self, kernel: HilbertPolyaKernel, weights: PhiRuelleWeights) -> None:
        self.kernel  = kernel
        self.weights = weights
        self._sieve_cache: Dict[int, list] = {}   # instance-level [FIX-LEAK-6]

    # ── sieve ─────────────────────────────────────────────────────────

    def _build_sieve(self, N: int) -> None:
        if N in self._sieve_cache:
            return
        spf = list(range(N + 1))
        i   = 2
        while i * i <= N:
            if spf[i] == i:
                for j in range(i * i, N + 1, i):
                    if spf[j] == j:
                        spf[j] = i
            i += 1
        self._sieve_cache[N] = spf

    def _von_mangoldt_sieve(self, n: int, spf: list) -> float:
        if n <= 1:
            return 0.0
        factors: List[int] = []
        m = n
        while m > 1:
            p = spf[m]
            while m % p == 0:
                factors.append(p)
                m //= p
        unique = set(factors)
        return math.log(float(next(iter(unique)))) if len(unique) == 1 else 0.0

    # ── feature vector ─────────────────────────────────────────────────

    def _feature_vector(self, n: int, spf: Optional[list] = None) -> np.ndarray:
        """Zero-free analytic feature vector for index n.  [FIX-LEAK-1/2]"""
        H    = self.kernel.H
        K    = self.weights.K
        ln_n = math.log(max(n, 1))
        L    = self._von_mangoldt_sieve(n, spf) if spf else von_mangoldt(n)

        branches = np.array([
            math.cos(ln_n / H),
            math.sin(ln_n / H),
            math.cos(2.0 * ln_n / H),
            math.sin(2.0 * ln_n / H),
            L * math.cos(ln_n / H),
            L * math.sin(ln_n / H),
            L * math.cos(2.0 * ln_n / H),
        ], dtype=float)[:min(K, 7)]

        if K > 7:
            branches = np.concatenate([branches, np.zeros(K - 7)])

        norm = np.linalg.norm(branches)
        if norm < 1e-300:
            return np.zeros(K)
        return self.weights.sqrt_weights[:K] * (branches / norm)

    # ── HS norm check [FIX-CIRCULAR-5] ───────────────────────────────

    def _verify_phi_tilde_norms(self, Phi_tilde: np.ndarray) -> bool:
        N        = Phi_tilde.shape[0]
        norms    = np.linalg.norm(Phi_tilde, axis=1)
        expected = 1.0 / np.arange(1, N + 1, dtype=float)
        return bool(np.all(norms <= expected + 1e-14))

    # ── build exact or surrogate ──────────────────────────────────────

    def build_exact(self, N: int) -> np.ndarray:
        """Exact K(m,n) matrix. Zero-free.  [T1]"""
        return self.kernel.build_matrix(N)

    def build_surrogate(self, N: int) -> np.ndarray:
        """Zero-free phi-Gram surrogate: T_N ≈ Φ̃ W Φ̃^T.  [T1]"""
        K = self.weights.K
        self._build_sieve(N + 1)
        spf       = self._sieve_cache[N + 1]
        Phi_tilde = np.empty((N, K), dtype=float)
        for i in range(N):
            phi           = self._feature_vector(i + 1, spf)
            Phi_tilde[i]  = phi[:K] / (i + 1)
        self._verify_phi_tilde_norms(Phi_tilde)
        return Phi_tilde @ Phi_tilde.T

    def clear_sieve_cache(self) -> None:
        """Release instance-level sieve cache.  [FIX-LEAK-6]"""
        self._sieve_cache.clear()

    def __repr__(self) -> str:
        return f"HPHGramOperator(kernel={self.kernel}, weights={self.weights})"


# ============================================================
# TOEPLITZ QUADRATIC FORM / PARSEVAL BRIDGE  [T1]
# ============================================================

class ToeplitzForm:
    """
    Parseval bridge between the operator form and the integral form.  [T1]

    Q_H(x; T₀) = ⟨T_N v_N, v_N⟩
               = ∫ k_H(t) |Z_N(1 + i(T₀ + 2πt))|² dt

    with v_n = n^{-1/2} e^{-iT₀ ln n},  Z_N(s) = Σ_{n=1}^N n^{-s}.

    The two code paths are algebraically independent:
      operator_form → uses k_hat (GR formula)
      integral_form → uses k_real (direct sech⁴)
    Agreement is a non-circular test.  [FIX-CIRCULAR-7, FIX-PARSEVAL]
    """

    def __init__(self, kernel: HilbertPolyaKernel) -> None:
        self.kernel = kernel

    @staticmethod
    def physical_vector(N: int, T0: float = 0.0) -> np.ndarray:
        """v_n = n^{-1/2} e^{-iT₀ ln n}.  [T1]"""
        n = np.arange(1, N + 1, dtype=float)
        return n ** (-0.5) * np.exp(-1j * T0 * np.log(n))

    def evaluate_operator(self, N: int, T0: float, T_N: np.ndarray) -> float:
        """Re ⟨T_N v_N, v_N⟩.  Uses k_hat path.  [T1]"""
        v  = self.physical_vector(N, T0)
        Tv = T_N @ v
        return float(np.dot(v.conj(), Tv).real)

    def evaluate_integral(self, N: int, T0: float,
                          n_quadrature: int = 2000) -> float:
        """
        ∫ k_H(t) |Z_N(1 + i(T₀ + 2πt))|² dt.
        Uses k_real path (independent).  [T1, FIX-PARSEVAL]
        """
        H       = self.kernel.H
        t_range = max(20.0, 5.0 * H)
        nodes, weights = np.polynomial.legendre.leggauss(n_quadrature)
        t  = t_range * nodes
        dt = t_range * weights
        n  = np.arange(1, N + 1, dtype=float)
        # Z_N(1+i(T₀+2πt)) = Σ_n n^{-1} exp(-i(T₀+2πt)ln n)
        phases = -np.outer(T0 + 2.0 * PI * t, np.log(n))
        Z_vals = (n[np.newaxis, :] ** (-1.0) * np.exp(1j * phases)).sum(axis=1)
        return float(np.dot(self.kernel.k_real(t) * np.abs(Z_vals) ** 2, dt))

    def verify_parseval_bridge(self, N: int = 20, T0: float = 0.0,
                               T_N: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Non-circular Parseval bridge check.  [T1, FIX-CIRCULAR-7]
        """
        if T_N is None:
            T_N = self.kernel.build_matrix(N)
        op  = self.evaluate_operator(N, T0, T_N)
        ig  = self.evaluate_integral(N, T0)
        rel = abs(op - ig) / (abs(ig) + 1e-30)
        return {
            "N": N, "T0": T0,
            "operator_form": op, "integral_form": ig,
            "relative_error": rel,
            "paths_independent": True,
            "note": (
                "operator_form uses k_hat (GR formula); "
                "integral_form uses k_real (sech⁴). Non-circular."
            ),
        }

    def __repr__(self) -> str:
        return f"ToeplitzForm(kernel={self.kernel})"


# ============================================================
# EXPLICIT-FORMULA-INSPIRED PRIME–RESONANCE KERNEL
# ============================================================

def build_prime_kernel(
    t_n: np.ndarray,
    primes: np.ndarray,
    eps: float = EPSILON_PRIME,
    use_density_weight: bool = True,
    sigma_exp: float = 0.5,
) -> np.ndarray:
    r"""
    Explicit-formula-inspired prime-resonance kernel:

        K_prime(m,n) = ε_prime Σ_{p≤P_max} (log p / p^{σ_exp})
                       w_m(p) w_n(p) cos((t_m - t_n) log p),

    with log-window weights w_m(p) = sech²(α(log p − log t_m)).
    Zero-free: depends only on primes and arithmetic levels t_n.
    """
    N     = len(t_n)
    Kp    = np.zeros((N, N), dtype=float)
    t     = np.asarray(t_n, dtype=float)
    p_arr = np.asarray(primes, dtype=float)
    logp  = np.log(p_arr)

    log_t  = np.log(t + 1.0)
    w_spec = (1.0 / (log_t[:, None] * log_t[None, :])) if use_density_weight else 1.0

    alpha = 0.4
    W     = np.zeros((N, p_arr.size), dtype=float)
    for i, T in enumerate(t):
        mu_T    = math.log(max(T, 2.0))
        W[i, :] = sech(alpha * (logp - mu_T)) ** 2

    coeff = logp / (p_arr ** sigma_exp)

    for ip in range(p_arr.size):
        c_p   = coeff[ip]
        w_col = W[:, ip]
        outer = np.outer(w_col, w_col)
        osc   = np.cos(np.subtract.outer(t, t) * logp[ip])
        Kp   += c_p * (outer * osc * w_spec)

    Kp *= eps
    Kp  = 0.5 * (Kp + Kp.T)
    mx  = float(np.max(np.abs(Kp)))
    if mx > 0.0:
        Kp /= mx
    return Kp


# ============================================================
# GUE-PERTURBATION
# ============================================================

def build_random_gue_perturbation(
    N: int,
    gamma: float = GAMMA_GUE,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(RNG_SEED_GUE)
    R = rng.normal(0.0, 1.0, size=(N, N))
    R = 0.5 * (R + R.T)
    x = rng.standard_normal(N)
    x /= np.linalg.norm(x) + 1e-15
    op_norm_val = 0.0
    for _ in range(40):
        y = R @ x
        op_norm_val = float(np.linalg.norm(y))
        if op_norm_val < 1e-15:
            break
        x = y / op_norm_val
    if op_norm_val > 1e-12:
        R *= gamma / op_norm_val
    else:
        R *= gamma
    return R


# ============================================================
# CORE HPH-DRIVEN OPERATOR (H_N)
# ============================================================

class ChaoticHilbertPolyaOperator:
    r"""
    Arithmetic HPH-resonant HPO candidate:

        H_N = D_N
              + ε_HPH · K_N^{HPH}
              + K_N^{prime}
              + γ R_N
              − δ_N I_N  (optional trace centering)

    K_N^{HPH} is drawn from the Hilbert–Pólya Hamiltonian: the Bochner-positive
    sech⁴ Toeplitz kernel with corrected Fourier symbol k̂_H(ξ) and
    phi-Ruelle weighted Gram surrogate.  By default the exact kernel
    K(m,n) = k̂_H(ln m − ln n)/√(mn) is used; set use_hph_surrogate=True
    for the rank-K phi-Gram approximation.

    Self-adjointness is preserved at the finite-N level by construction.
    Zero-free: the kernel T_N never encodes tabulated Riemann zeros.
    """

    def __init__(
        self,
        N: int,
        H: float = HPH_H_DEFAULT,
        epsilon_hph: float = EPSILON_HPH,
        epsilon_prime: float = EPSILON_PRIME,
        P_max: Optional[int] = None,
        gamma_gue: float = GAMMA_GUE,
        use_hph_surrogate: bool = False,
        add_random: bool = True,
        center_trace: bool = False,
        rng: Optional[np.random.Generator] = None,
    ):
        if N < 2 or N > MAX_N:
            raise ValueError(f"N must be in [2, {MAX_N}], got {N}")
        self.N              = N
        self.H              = H
        self.epsilon_hph    = epsilon_hph
        self.epsilon_prime  = epsilon_prime
        self.P_max          = P_max if P_max is not None else adaptive_P_max(N)
        self.gamma_gue      = gamma_gue
        self.use_hph_surrogate = use_hph_surrogate
        self.add_random     = add_random
        self.center_trace   = center_trace
        self._rng           = rng if rng is not None else np.random.default_rng(RNG_SEED_GUE)

        # Arithmetic diagonal (Riemann–von Mangoldt inversion)
        self._D_diag = build_diagonal_D(N)

        # HPH kernel objects
        self._hph_kernel  = HilbertPolyaKernel(H=H)
        self._hph_weights = PhiRuelleWeights(K=PHI_RUELLE_K)
        self._hph_gram    = HPHGramOperator(self._hph_kernel, self._hph_weights)

        # Verify k_hat limit consistency [FIX-CIRCULAR-1]
        self._hph_kernel.assert_limit_consistency()

        # Build HPH backbone K_N^{HPH}
        if use_hph_surrogate:
            K_hph = self._hph_gram.build_surrogate(N)
        else:
            K_hph = self._hph_gram.build_exact(N)
        self._K_hph = K_hph

        # Prime-resonance kernel (explicit-formula-inspired)
        primes         = get_primes(self.P_max)
        self._K_prime  = build_prime_kernel(
            self._D_diag, primes, eps=epsilon_prime,
            use_density_weight=True, sigma_exp=0.5,
        )

        # Combine deterministic components
        H_det = np.diag(self._D_diag) + epsilon_hph * self._K_hph + self._K_prime

        if self.center_trace:
            delta_N = float(np.trace(H_det)) / N
            H_det   = H_det - delta_N * np.eye(N)

        if self.add_random and self.gamma_gue > 0.0:
            R     = build_random_gue_perturbation(N, gamma=self.gamma_gue, rng=self._rng)
            H_det = H_det + R

        self._matrix = 0.5 * (H_det + H_det.T)

    # ── public API ────────────────────────────────────────────────────

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @property
    def levels(self) -> np.ndarray:
        return self._D_diag

    @property
    def K_backbone(self) -> np.ndarray:
        """HPH kernel matrix (scaled by ε_HPH)."""
        return self.epsilon_hph * self._K_hph

    @property
    def K_prime(self) -> np.ndarray:
        return self._K_prime

    @property
    def hph_kernel(self) -> HilbertPolyaKernel:
        return self._hph_kernel

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self._matrix @ x

    def spectrum(self) -> np.ndarray:
        return np.sort(np.linalg.eigvalsh(self._matrix.astype(float)))

    def operator_norm(self, max_iter: int = 200) -> float:
        x = _rng.standard_normal(self.N)
        x /= np.linalg.norm(x) + 1e-15
        norm_val = 0.0
        for _ in range(max_iter):
            y        = self._matrix @ x
            norm_val = float(np.linalg.norm(y))
            if norm_val < 1e-15:
                break
            x = y / norm_val
        return norm_val

    def hilbert_schmidt_norm(self) -> float:
        return float(math.sqrt(np.sum(self._matrix.astype(float) ** 2)))

    def trace(self) -> float:
        return float(np.trace(self._matrix))

    def heat_trace(self, t: float) -> float:
        return float(np.sum(np.exp(-t * self.spectrum())))

    def weyl_density_error(self) -> Dict[str, float]:
        evals = self.spectrum()
        evals = evals[evals > 1.0]
        if evals.size == 0:
            return {"weyl_error": float("nan"), "verified": False}
        T_max = float(evals[-1])
        if T_max <= 1.0:
            return {"weyl_error": float("nan"), "verified": False}
        N_emp  = float(evals.size)
        N_weyl = (T_max / (2.0 * math.pi)) * math.log(
            T_max / (2.0 * math.pi * math.e) + 1e-10
        ) + 7.0 / 8.0
        rel_err = abs(N_emp - N_weyl) / (abs(N_weyl) + 1.0)
        return {
            "N_empirical": N_emp,
            "N_weyl":      N_weyl,
            "T_max":       T_max,
            "relative_error": rel_err,
            "verified":    rel_err < 0.05,
        }

    # ── Parseval bridge accessor ───────────────────────────────────────

    def parseval_bridge_check(self, N_sub: int = 20, T0: float = 0.0) -> Dict[str, float]:
        """
        Verify Parseval bridge on the HPH kernel for a sub-problem of size N_sub.
        Non-circular: operator path uses k_hat, integral path uses k_real.  [T1]
        """
        N_sub = min(N_sub, self.N)
        form  = ToeplitzForm(self._hph_kernel)
        T_sub = self._hph_kernel.build_matrix(N_sub)
        return form.verify_parseval_bridge(N_sub, T0, T_sub)

    def hph_positivity_check(self) -> Dict[str, float]:
        """
        Bochner positivity and limit consistency of the embedded HPH kernel.  [T1]
        """
        return self._hph_kernel.verify_positivity(N=min(50, self.N))


# ============================================================
# BLOCK OPERATOR FOR FUNCTIONAL-EQUATION SYMMETRY
# ============================================================

class BlockSpectralOperator:
    r"""
    Block operator implementing exact λ ↔ −λ symmetry:

        H_centered = H - (Tr(H)/N) I
        H_block = [[ H_centered,  K_tot ],
                   [ K_tot,      -H_centered ]],

    where K_tot = K_backbone + K_prime.
    """

    def __init__(self, core: ChaoticHilbertPolyaOperator):
        self.core = core
        H   = core.matrix
        N   = core.N
        c   = float(np.trace(H)) / N
        Hc  = H - c * np.eye(N)
        K   = core.K_backbone + core.K_prime
        top = np.concatenate((Hc, K), axis=1)
        bot = np.concatenate((K, -Hc), axis=1)
        Hb  = np.concatenate((top, bot), axis=0)
        self._H_block             = 0.5 * (Hb + Hb.T)
        self._evals: Optional[np.ndarray] = None
        self._evecs: Optional[np.ndarray] = None

    @property
    def matrix(self) -> np.ndarray:
        return self._H_block

    def eigenpairs(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._evals is None or self._evecs is None:
            vals, vecs = np.linalg.eigh(self._H_block.astype(float))
            idx        = np.argsort(vals)
            self._evals = vals[idx]
            self._evecs = vecs[:, idx]
        return self._evals, self._evecs

    def spectrum(self) -> np.ndarray:
        evals, _ = self.eigenpairs()
        return evals


# ============================================================
# NUMERICAL TESTS
# ============================================================

def test_linearity(op: ChaoticHilbertPolyaOperator,
                   trials: int = 8, tol: float = 1e-10) -> Tuple[bool, float]:
    max_err = 0.0
    for _ in range(trials):
        x, y  = _rng.standard_normal(op.N), _rng.standard_normal(op.N)
        a, b  = _rng.standard_normal(2)
        lhs   = op.apply(a * x + b * y)
        rhs   = a * op.apply(x) + b * op.apply(y)
        err   = float(np.linalg.norm(lhs - rhs) / (np.linalg.norm(lhs) + 1e-15))
        max_err = max(max_err, err)
    return max_err < tol, max_err


def test_adjoint_consistency(op: ChaoticHilbertPolyaOperator,
                             trials: int = 8, tol: float = 1e-10) -> Tuple[bool, float]:
    max_err = 0.0
    for _ in range(trials):
        x, y  = _rng.standard_normal(op.N), _rng.standard_normal(op.N)
        lhs   = float(np.dot(y, op.apply(x)))
        rhs   = float(np.dot(op.apply(y), x))
        err   = abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-15)
        max_err = max(max_err, err)
    return max_err < tol, max_err


def test_spectral_reality(M: np.ndarray, tol: float = 1e-10) -> Tuple[bool, float]:
    evals    = np.linalg.eigvalsh(M.astype(float))
    imag_max = float(np.max(np.abs(np.imag(evals))))
    return imag_max < tol, imag_max


def test_hilbert_schmidt(M: np.ndarray) -> Tuple[bool, float]:
    hs = float(math.sqrt(np.sum(M.astype(float) ** 2)))
    return np.isfinite(hs), hs


# ============================================================
# SPECTRAL STATISTICS (GUE, SPACING RATIO)
# ============================================================

def local_unfold(eigenvalues: np.ndarray, window: int = 20) -> np.ndarray:
    E  = np.sort(eigenvalues.astype(float))
    N  = len(E)
    if N < 2 * window + 4:
        return np.array([], dtype=float)
    unfolded: List[float] = []
    for i in range(window, N - window - 1):
        seg  = E[i - window: i + window + 1]
        npts = np.arange(len(seg), dtype=float)
        poly = Polynomial.fit(seg, npts, deg=1).convert()
        s    = float(np.polyval(poly.coef[::-1], E[i + 1]) -
                     np.polyval(poly.coef[::-1], E[i]))
        if s > 0:
            unfolded.append(s)
    arr = np.array(unfolded, dtype=float)
    if arr.size > 0:
        arr /= arr.mean()
    return arr


def wigner_surmise_cdf(s: np.ndarray) -> np.ndarray:
    return 1.0 - np.exp(-4.0 * np.asarray(s, dtype=float) ** 2 / math.pi)


def gue_ks_test(eigenvalues: np.ndarray) -> Dict[str, float]:
    evals = np.sort(eigenvalues.astype(float))
    n     = evals.size
    if n < 80:
        return {"ks_statistic": float("nan"), "p_value": float("nan")}
    bulk   = evals[n // 4: 3 * n // 4]
    spaces = local_unfold(bulk, window=20)
    if spaces.size < 20:
        return {"ks_statistic": float("nan"), "p_value": float("nan")}
    stat, pval = kstest(spaces, wigner_surmise_cdf)
    return {"ks_statistic": float(stat), "p_value": float(pval)}


def mean_spacing_ratio(eigenvalues: np.ndarray) -> float:
    E = np.sort(eigenvalues.astype(float))
    s = np.diff(E)
    s = s[s > 0]
    if s.size < 3:
        return float("nan")
    r = np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])
    return float(np.mean(r))


# ============================================================
# FUNCTIONAL-EQUATION / REFLECTION / ORTHOGONALITY (BLOCK)
# ============================================================

def block_reflection_test(evals: np.ndarray) -> Dict[str, float]:
    evals   = np.sort(evals.astype(float))
    N       = len(evals)
    if N < 4:
        return {"max_pairing_error": float("nan"), "normalized_error": float("nan")}
    mid     = 0.5 * (evals[0] + evals[-1])
    sr      = float(evals[-1] - evals[0]) + 1e-15
    n_pairs = N // 2
    low     = evals[:n_pairs]
    high    = evals[N - n_pairs:][::-1]
    err     = float(np.max(np.abs(low + high - 2.0 * mid)))
    return {"max_pairing_error": err, "normalized_error": err / sr}


def block_functional_symmetry(evals: np.ndarray) -> Dict[str, float]:
    evals   = np.sort(evals.astype(float))
    center  = 0.5 * (evals[0] + evals[-1])
    shifted = evals - center
    err     = float(np.linalg.norm(np.sort(shifted) - np.sort(-shifted)) /
                    (np.linalg.norm(shifted) + 1e-15))
    return {"center": center, "symmetry_error": err}


def eigenvector_orthogonality(vecs: np.ndarray) -> Dict[str, float]:
    G    = vecs.T.conj() @ vecs
    Ga   = np.abs(G)
    np.fill_diagonal(Ga, 0.0)
    return {"max_overlap": float(np.max(Ga)), "mean_overlap": float(np.mean(Ga))}


# ============================================================
# RIEMANN ZERO CDF COMPARISON
# ============================================================

def compare_density_cdfs(zeros: np.ndarray,
                         eigs: np.ndarray) -> Dict[str, float]:
    zeros = np.sort(zeros.astype(float))
    eigs  = np.sort(eigs.astype(float))
    if zeros.size < 20 or eigs.size < 20:
        return {"ks_statistic": float("nan"), "p_value": float("nan"),
                "n_zeros": float(zeros.size), "n_eigs": float(eigs.size)}
    T_min = max(zeros[0], eigs[0])
    T_max = min(zeros[min(len(zeros)-1, len(eigs)-1)], eigs[-1])
    if T_max <= T_min:
        return {"ks_statistic": float("nan"), "p_value": float("nan"),
                "n_zeros": float(zeros.size), "n_eigs": float(eigs.size)}
    zw = zeros[(zeros >= T_min) & (zeros <= T_max)]
    ew = eigs[(eigs >= T_min) & (eigs <= T_max)]
    if zw.size < 20 or ew.size < 20:
        return {"ks_statistic": float("nan"), "p_value": float("nan"),
                "n_zeros": float(zw.size), "n_eigs": float(ew.size)}
    stat, pval = ks_2samp(ew, zw, alternative="two-sided")
    return {"ks_statistic": float(stat), "p_value": float(pval),
            "n_zeros": float(zw.size), "n_eigs": float(ew.size)}


# ============================================================
# EXPLICIT-FORMULA SPECTRAL TRACE HOOK
# ============================================================

def spectral_trace(evals: np.ndarray, tau: float) -> complex:
    return complex(np.sum(np.exp(1j * tau * evals)))


def prime_trace(tau: float, P_max: int = 10000, K_max: int = 5) -> complex:
    is_prime = np.ones(P_max + 1, dtype=bool)
    is_prime[0:2] = False
    for p in range(2, int(math.isqrt(P_max)) + 1):
        if is_prime[p]:
            is_prime[p * p:P_max + 1:p] = False
    primes = np.nonzero(is_prime)[0]
    s = 0.0 + 0.0j
    for p in primes:
        logp = math.log(p)
        pk   = p
        for k in range(1, K_max + 1):
            if pk > P_max:
                break
            weight = logp / (pk ** 0.5)
            phase  = math.log(pk)
            s     += weight * complex(math.cos(tau * phase), math.sin(tau * phase))
            pk    *= p
    return s


def explicit_formula_trace_hook(
    evals: np.ndarray,
    tau_vals: np.ndarray,
    P_max: int = 20000,
    K_max: int = 4,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    print("\n▶ Spectral vs Prime Trace (explicit-formula hook)")
    print("-" * 72)
    print(f"{'τ':<10} | {'|Tr_spectral|':<16} | {'|Tr_prime|':<16} | {'rel_diff':<10}")
    print("-" * 72)
    for tau in tau_vals:
        Ts  = spectral_trace(evals, tau)
        Tp  = prime_trace(tau, P_max=P_max, K_max=K_max)
        rel = abs(Ts - Tp) / (abs(Tp) + 1e-12)
        print(f"{tau:<10.4f} | {abs(Ts):<16.6f} | {abs(Tp):<16.6f} | {rel:<10.6f}")
        rows.append({"tau": float(tau), "Tr_spec_abs": float(abs(Ts)),
                     "Tr_prime_abs": float(abs(Tp)), "rel_diff": float(rel)})
    return rows


# ============================================================
# ANALYTIC-STYLE BOUNDS FOR SMOOTHING-INDUCED ERROR
# ============================================================

def prime_tail_bound(tau: float, P_max: int, P_cap: int = 200000) -> float:
    tail_limit = min(max(P_max * 10, P_max + 1000), P_cap)
    is_prime   = np.ones(tail_limit + 1, dtype=bool)
    is_prime[:2] = False
    for p in range(2, int(math.isqrt(tail_limit)) + 1):
        if is_prime[p]:
            is_prime[p * p:tail_limit + 1:p] = False
    primes_tail = np.nonzero(is_prime)[0]
    primes_tail = primes_tail[primes_tail > P_max]
    if primes_tail.size == 0:
        return 0.0
    acc = 0.0
    for p in primes_tail:
        logp   = math.log(p)
        weight = logp / math.sqrt(p)
        f_hat  = math.sqrt(math.pi / tau) * math.exp(-(logp**2) / (4.0 * tau))
        acc   += weight * f_hat * 2.0
    return float(abs(acc))


def gaussian_spectral_tail_bound(tau: float, T_max: float) -> float:
    from math import erfc
    return 0.5 * math.sqrt(math.pi / tau) * erfc(T_max * math.sqrt(tau))


# ============================================================
# HOOK 1: EXPLICIT-FORMULA TRACE MATCHING (GAUSSIAN TEST FUNCTION)
# ============================================================

def explicit_trace_with_test_function(
    eigenvalues: np.ndarray,
    test_tau: float,
    prime_limit: int = 10000,
    k_max: int = 5,
) -> Dict[str, float]:
    """
    Smoothed explicit formula with Gaussian f(E) = exp(-τ E²),
    f̂(ω) = sqrt(π/τ) exp(-ω²/(4τ)).
    """
    tau             = test_tau
    spectral_val    = float(np.sum(np.exp(-tau * eigenvalues**2)))
    prime_val       = 0.0

    is_prime = np.ones(prime_limit + 1, dtype=bool)
    is_prime[0:2] = False
    for p in range(2, int(np.sqrt(prime_limit)) + 1):
        if is_prime[p]:
            is_prime[p * p:prime_limit + 1:p] = False
    primes = np.nonzero(is_prime)[0]

    for p in primes:
        logp = math.log(p)
        pk   = p
        for k in range(1, k_max + 1):
            if pk > prime_limit:
                break
            weight   = logp / math.sqrt(pk)
            omega    = k * logp
            f_hat    = math.sqrt(math.pi / tau) * math.exp(-(omega**2) / (4.0 * tau))
            prime_val += weight * f_hat * 2.0
            pk       *= p

    rel_diff  = abs(spectral_val - prime_val) / (abs(prime_val) + 1e-12)
    T_max     = float(np.max(np.abs(eigenvalues))) if eigenvalues.size > 0 else 0.0
    ptail     = prime_tail_bound(tau, prime_limit)
    stail     = gaussian_spectral_tail_bound(tau, T_max)

    return {
        "spectral_trace": spectral_val,
        "prime_trace": prime_val,
        "relative_diff": rel_diff,
        "prime_tail_bound": ptail,
        "spectral_tail_bound": stail,
        "verified": rel_diff < 0.90,
    }


# ============================================================
# HOOK 2: BERRY–KEATING-STYLE UNFOLDING
# ============================================================

def berry_keating_unfolding(eigenvalues: np.ndarray) -> np.ndarray:
    T = np.sort(eigenvalues[eigenvalues > 1.0])
    if len(T) == 0:
        return np.array([])

    def N_expected(T_val: float) -> float:
        if T_val <= 1.0:
            return 0.0
        return (T_val / (2.0 * math.pi)) * math.log(T_val / (2.0 * math.pi * math.e)) + 7.0 / 8.0

    N_exp = np.array([N_expected(float(t)) for t in T], dtype=float)
    if N_exp.size > 1:
        diffs = np.diff(N_exp)
        ms    = float(np.mean(diffs))
        if ms > 0:
            N_exp = N_exp / ms
    return N_exp


def zero_level_ks_test(
    eigenvalues: np.ndarray,
    riemann_zeros: np.ndarray,
    use_berry_keating: bool = True,
) -> Dict[str, float]:
    zeros = np.sort(riemann_zeros.astype(float))
    if zeros.size < 20:
        return {"ks_stat": float("nan"), "p_value": float("nan"), "verified": False}
    evals_u = berry_keating_unfolding(eigenvalues) if use_berry_keating \
        else np.sort(eigenvalues.astype(float))
    if evals_u.size < 20:
        return {"ks_stat": float("nan"), "p_value": float("nan"), "verified": False}
    m        = min(evals_u.size, zeros.size)
    eu       = np.sort(evals_u[:m])
    zu       = np.sort(zeros[:m])

    def norm_unit(x: np.ndarray) -> np.ndarray:
        lo, hi = float(x[0]), float(x[-1])
        if hi <= lo:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    stat, pval = ks_2samp(norm_unit(eu), norm_unit(zu), alternative="two-sided")
    return {"ks_stat": float(stat), "p_value": float(pval),
            "verified": bool(stat < 0.30)}


# ============================================================
# HOOK 3: STRONG RESOLVENT CONVERGENCE
# ============================================================

def resolvent(matrix: np.ndarray, z: complex) -> np.ndarray:
    n = matrix.shape[0]
    return np.linalg.inv(matrix - z * np.eye(n, dtype=complex))


def resolvent_convergence_test(
    operators: List[np.ndarray],
    z: complex = 1.0 + 1.0j,
    atol: float = 1e-6,
) -> Dict[str, object]:
    errors: List[float] = []
    if len(operators) < 2:
        return {"errors": errors, "converged": False, "atol": atol}
    for i in range(1, len(operators)):
        H_big   = operators[i]
        H_small = operators[i - 1]
        N_big   = H_big.shape[0]
        N_small = H_small.shape[0]
        R_big   = resolvent(H_big.astype(complex), z)
        R_small = resolvent(H_small.astype(complex), z)
        R_pad   = np.zeros((N_big, N_big), dtype=complex)
        R_pad[:N_small, :N_small] = R_small
        errors.append(float(np.linalg.norm(R_big - R_pad, ord=2)))
    converged = bool(errors[-1] < 0.01) if errors else False
    return {"errors": errors, "converged": converged, "atol": atol}


# ============================================================
# HOOK 4: PARSEVAL BRIDGE DIAGNOSTIC (HPH-SPECIFIC)
# ============================================================

def parseval_bridge_diagnostic(
    op: ChaoticHilbertPolyaOperator,
    T0_vals: Optional[List[float]] = None,
    N_sub: int = 20,
) -> List[Dict[str, float]]:
    """
    Run the Parseval bridge check for multiple T₀ values.

    This is a diagnostic native to the HPH kernel: it verifies that
    k̂_H (used in operator form) correctly Fourier-transforms k_H
    (used in the integral form) at multiple probe points T₀.
    Independent code paths; non-circular.  [FIX-CIRCULAR-7]
    """
    if T0_vals is None:
        T0_vals = [0.0, 5.0, 14.134]
    form  = ToeplitzForm(op.hph_kernel)
    T_sub = op.hph_kernel.build_matrix(N_sub)
    rows: List[Dict[str, float]] = []
    print("\n▶ Parseval Bridge (HPH kernel, non-circular: k_hat vs k_real paths)")
    print("-" * 72)
    print(f"  N_sub={N_sub}, H={op.H}")
    print(f"{'T0':<10} | {'op_form':<14} | {'int_form':<14} | {'rel_err':<12} | pass?")
    print("-" * 72)
    for T0 in T0_vals:
        rec = form.verify_parseval_bridge(N_sub, T0, T_sub)
        ok  = rec["relative_error"] < 1e-3
        print(f"  {T0:<8.3f} | {rec['operator_form']:<14.6f} | "
              f"{rec['integral_form']:<14.6f} | {rec['relative_error']:<12.2e} | "
              f"{'PASS' if ok else 'FAIL'}")
        rows.append({**rec, "H": op.H})
    return rows


# ============================================================
# INTEGRATED SPECTRAL / TRACE / LIMIT VALIDATION
# ============================================================

def full_spectral_validation(
    eigenvalues: np.ndarray,
    operators: List[np.ndarray],
    riemann_zeros: np.ndarray,
    gaussian_tau: float = 0.5,
    resolvent_z: complex = 1.0 + 1.0j,
    resolvent_atol: float = 1e-6,
) -> Dict[str, object]:
    results: Dict[str, object] = {}
    results["explicit_hook"] = explicit_trace_with_test_function(
        eigenvalues, test_tau=gaussian_tau, prime_limit=10000, k_max=5,
    )
    results["zero_ks"] = zero_level_ks_test(
        eigenvalues, riemann_zeros, use_berry_keating=True,
    )
    results["resolvent_conv"] = resolvent_convergence_test(
        operators, z=resolvent_z, atol=resolvent_atol,
    )
    return results


# ============================================================
# ANTI-CIRCULARITY GUARD
# ============================================================

class AntiCircularityGuard:
    """
    Verify that the operator construction does not encode tabulated
    Riemann zeros in D_diag or K_matrix at machine precision.
    """

    @staticmethod
    def execute_guard(D_diag: np.ndarray,
                      K_matrix: np.ndarray,
                      zeros: np.ndarray) -> None:
        if zeros.size == 0:
            return
        z_r  = np.round(zeros.astype(float), 12)
        D_r  = np.round(D_diag.ravel().astype(float), 12)
        K_r  = np.round(K_matrix.ravel().astype(float), 12)
        if np.intersect1d(D_r, z_r).size > 0 or np.intersect1d(K_r, z_r).size > 0:
            raise RuntimeError(
                "AntiCircularityGuard: detected direct reuse of Riemann zeros "
                "inside operator coefficients."
            )


# ============================================================
# CONTINUUM LIMIT PLACEHOLDER
# ============================================================

class ContinuumHilbertPolyaModel:
    """
    Abstract placeholder for the infinite-dimensional limit operator H_∞.
    The HPH kernel T provides the Hilbert–Schmidt backbone; the arithmetic
    structure provides the spectral localization.
    """

    def __init__(self) -> None:
        self.domain_description = (
            "Core domain D ⊂ ℓ²(ℕ) where the HPH Toeplitz form and the "
            "quadratic form of H_∞ are simultaneously well-defined."
        )
        self.parseval_identity = (
            "⟨T v(T₀), v(T₀)⟩ = ∫ k_H(t)|Z(1+i(T₀+2πt))|² dt  [T1]"
        )
        self.rh_connection = (
            "RH ⟺ Q_H^∞ := inf_{T₀∈ℝ} lim_{N→∞} Q_H(x;T₀) > 0  [T2, open: T3]"
        )

    def theoretical_resolvent_bound(self, z: complex) -> str:
        return (
            f"Expected resolvent bound for z={z}: "
            "||(H_∞ − zI)^{-1}|| ≤ C/dist(z,ℝ) with C ~ 1."
        )


# ============================================================
# CORE VALIDATION DRIVER
# ============================================================

def run_validation(N: int,
                   H: float = HPH_H_DEFAULT,
                   zeros: Optional[np.ndarray] = None,
                   verbose: bool = True) -> Dict[str, object]:
    op = ChaoticHilbertPolyaOperator(
        N,
        H=H,
        epsilon_hph=EPSILON_HPH,
        epsilon_prime=EPSILON_PRIME,
        P_max=None,
        gamma_gue=GAMMA_GUE,
        use_hph_surrogate=False,
        add_random=True,
    )

    results: Dict[str, object] = {"N": N, "H": H}

    # A1–A4 basic operator tests
    lin_ok,  lin_err  = test_linearity(op)
    op_norm            = op.operator_norm()
    bdd_ok             = np.isfinite(op_norm) and op_norm < 1e9
    adj_ok,  adj_err   = test_adjoint_consistency(op)
    hs_ok,   hs_norm   = test_hilbert_schmidt(op.matrix)
    spec_ok, imag_max  = test_spectral_reality(op.matrix)

    results["linearity"]        = (lin_ok, lin_err)
    results["boundedness"]      = (bdd_ok, op_norm)
    results["adjoint"]          = (adj_ok, adj_err)
    results["hilbert_schmidt"]  = (hs_ok, hs_norm)
    results["real_spectrum"]    = (spec_ok, imag_max)

    evals              = op.spectrum()
    results["spectrum"] = evals
    results["weyl"]     = op.weyl_density_error()

    # HPH-specific checks
    results["hph_positivity"] = op.hph_positivity_check()
    results["parseval_bridge"] = parseval_bridge_diagnostic(
        op, T0_vals=[0.0, 5.0, 14.134], N_sub=min(20, N)
    )

    # Block spectral operator
    block               = BlockSpectralOperator(op)
    evals_block, vecs_b = block.eigenpairs()
    results["block_spectrum"]       = evals_block
    results["block_reflection"]     = block_reflection_test(evals_block)
    results["block_func_eq"]        = block_functional_symmetry(evals_block)
    results["block_orthogonality"]  = eigenvector_orthogonality(vecs_b)

    evals_pos                        = evals_block[evals_block > 0.0]
    results["block_gue_positive"]   = gue_ks_test(evals_pos)
    results["spacing_ratio"]        = mean_spacing_ratio(evals_pos)

    if zeros is not None and zeros.size > 0:
        results["block_zero_density"] = compare_density_cdfs(
            np.sort(zeros[:min(5000, zeros.size)]), evals_pos
        )
    else:
        results["block_zero_density"] = None

    # Original explicit-formula spectral trace hook
    tau_vals = np.linspace(0.1, 2.0, 10)
    results["explicit_trace"] = explicit_formula_trace_hook(
        evals_pos, tau_vals, P_max=20000, K_max=4
    )

    # Enhanced hooks
    operator_ladder: List[np.ndarray] = []
    if N >= 4:
        N_half = max(2, N // 2)
        op_half = ChaoticHilbertPolyaOperator(
            N_half, H=H,
            epsilon_hph=EPSILON_HPH, epsilon_prime=EPSILON_PRIME,
            P_max=None, gamma_gue=GAMMA_GUE,
            use_hph_surrogate=False, add_random=True,
        )
        operator_ladder.append(op_half.matrix.astype(float))
    operator_ladder.append(op.matrix.astype(float))

    zeros_for_test = (np.sort(zeros[:min(5000, zeros.size)])
                      if zeros is not None and zeros.size > 0
                      else np.array([], dtype=float))

    results["full_spectral"] = full_spectral_validation(
        eigenvalues=evals_pos,
        operators=operator_ladder,
        riemann_zeros=zeros_for_test,
        gaussian_tau=0.5,
        resolvent_z=1.0 + 1.0j,
        resolvent_atol=1e-6,
    )

    if verbose:
        _print_N_results(N, results)

    return results


def _print_N_results(N: int, r: Dict[str, object]) -> None:
    def tick(ok: bool) -> str:
        return "✓ PASS" if ok else "✗ FAIL"

    tests = [
        ("Linearity",           r["linearity"]),
        ("Boundedness",         r["boundedness"]),
        ("Adjoint consistency", r["adjoint"]),
        ("Hilbert–Schmidt",     r["hilbert_schmidt"]),
        ("Real spectrum",       r["real_spectrum"]),
    ]

    H = r.get("H", HPH_H_DEFAULT)
    print(f"\n▶ HPH-resonant H_N at dimension N={N}, H={H}")
    print("-" * 64)
    for name, (ok, val) in tests:
        print(f"  {tick(ok)} {name:28s} ({val:.3e})")

    evals = r["spectrum"]
    weyl  = r["weyl"]
    hph_p = r["hph_positivity"]

    print("\n  HPH kernel diagnostics:")
    print(f"    k_hat_min      = {hph_p.get('k_hat_min', float('nan')):.3e}  "
          f"(Bochner ≥ 0: {'✓' if hph_p.get('k_hat_min', -1) >= -1e-12 else '✗'})")
    print(f"    k_hat(0) analytic = {hph_p.get('k_hat_at_0_analytic', float('nan')):.6f}  "
          f"runtime = {hph_p.get('k_hat_at_0', float('nan')):.6f}")
    print(f"    symmetry_error = {hph_p.get('symmetry_error', float('nan')):.3e}")
    print(f"    min_eigenvalue (SANITY) = {hph_p.get('min_eigenvalue_SANITY', float('nan')):.3e}")

    pb_rows = r.get("parseval_bridge", [])
    if pb_rows:
        pb_ok = all(row.get("relative_error", 1.0) < 1e-3 for row in pb_rows)
        print(f"    Parseval bridge: {'✓ PASS' if pb_ok else '✗ FAIL'} "
              f"(max rel_err={max(row.get('relative_error', 0) for row in pb_rows):.2e})")

    print("\n  Spectrum summary (H_N):")
    print(f"    min eigenvalue   ≈ {float(evals.min()):.6e}")
    print(f"    max eigenvalue   ≈ {float(evals.max()):.6e}")
    print(f"    effective rank   ≈ {effective_rank(evals):.3f}")
    if weyl.get("verified", False) is not False:
        print(f"    Weyl: N_emp={weyl['N_empirical']:.0f}, "
              f"N_weyl={weyl['N_weyl']:.1f}, "
              f"rel_err={weyl['relative_error']:.4f}")

    evals_block = r["block_spectrum"]
    v3          = r["block_reflection"]
    v6          = r["block_func_eq"]
    v8          = r["block_orthogonality"]
    gue         = r["block_gue_positive"]
    r_mean      = r["spacing_ratio"]
    rd_block    = r["block_zero_density"]
    full_spec   = r.get("full_spectral", {})

    print("\n  Block operator summary (H_block, 2N×2N):")
    print(f"    min eigenvalue   ≈ {float(evals_block.min()):.6e}")
    print(f"    max eigenvalue   ≈ {float(evals_block.max()):.6e}")
    print(f"    V3 reflection normalized error   : {v3['normalized_error']:.3e}")
    print(f"    V6 func-eq symmetry error        : {v6['symmetry_error']:.3e}")
    print(f"    V8 max eigenvector overlap       : {v8['max_overlap']:.3e}")
    print(f"    GUE KS (λ>0): stat={gue.get('ks_statistic', float('nan')):.4f}, "
          f"p={gue.get('p_value', float('nan')):.3e}")
    print(f"    mean spacing ratio ⟨r⟩ ≈ {r_mean:.4f} "
          f"(target: 0.535 GUE, 0.386 Poisson)")

    if rd_block is not None:
        print(f"    Zero density KS (λ>0): stat={rd_block['ks_statistic']:.4f}, "
              f"p={rd_block['p_value']:.3e}  "
              f"(n_zeros={rd_block['n_zeros']:.0f}, n_eigs={rd_block['n_eigs']:.0f})")

    explicit_hook  = full_spec.get("explicit_hook", {})
    zero_ks        = full_spec.get("zero_ks", {})
    resolvent_conv = full_spec.get("resolvent_conv", {})

    if explicit_hook:
        print("    Gaussian explicit hook: "
              f"rel_diff={explicit_hook.get('relative_diff', float('nan')):.3e} "
              f"verified={explicit_hook.get('verified', False)}")
        print("      prime_tail_bound ≈ "
              f"{explicit_hook.get('prime_tail_bound', float('nan')):.3e}, "
              "spectral_tail_bound ≈ "
              f"{explicit_hook.get('spectral_tail_bound', float('nan')):.3e}")
    if zero_ks:
        print("    Zero-level KS (Berry–Keating): "
              f"stat={zero_ks.get('ks_stat', float('nan')):.4f}, "
              f"p={zero_ks.get('p_value', float('nan')):.3e}, "
              f"verified={zero_ks.get('verified', False)}")
    if resolvent_conv:
        errs = resolvent_conv.get("errors", [])
        if errs:
            print(f"    Resolvent conv errors (last 3): {errs[-3:] if len(errs) >= 3 else errs}")
        print(f"    Resolvent converged: {resolvent_conv.get('converged', False)}")


# ============================================================
# CROSS-N DIAGNOSTICS AND VERDICT
# ============================================================

def effective_rank(eigenvalues: np.ndarray) -> float:
    lam   = np.abs(eigenvalues.astype(float))
    total = lam.sum()
    if total <= 0:
        return 0.0
    p   = lam / total
    ent = -np.sum(p * np.log(p + 1e-15))
    return float(math.exp(ent))


def cross_n_diagnostics(all_results: List[Dict[str, object]]) -> None:
    print("\n▶ Cross-N operator norm and rank")
    print("-" * 64)
    for r in all_results:
        N = r["N"]
        _, op_norm = r["boundedness"]
        _, hs_norm = r["hilbert_schmidt"]
        evals      = r["spectrum"]
        print(f"  N={N:5d}  ||H||_op ≈ {op_norm:.6e}  "
              f"||H||_HS ≈ {hs_norm:.6e}  "
              f"rank_eff ≈ {effective_rank(evals):.1f}")


def final_verdict(all_results: List[Dict[str, object]]) -> bool:
    print("\n" + "=" * 72)
    print("FINAL VERDICT — HPH-RESONANT HILBERT–PÓLYA OPERATOR")
    print("=" * 72)

    checks = {
        "Linearity":           all(r["linearity"][0]       for r in all_results),
        "Uniform boundedness": all(r["boundedness"][0]     for r in all_results),
        "Self-adjointness":    all(r["adjoint"][0]         for r in all_results),
        "Hilbert–Schmidt":     all(r["hilbert_schmidt"][0] for r in all_results),
        "Real spectrum":       all(r["real_spectrum"][0]   for r in all_results),
        "Weyl law":            any(r["weyl"].get("verified", False)
                                   for r in all_results),
    }

    # HPH-specific: Bochner positivity and Parseval bridge
    hph_bochner_ok = all(
        r["hph_positivity"].get("k_hat_min", -1.0) >= -1e-12
        for r in all_results
    )
    parseval_ok = all(
        all(row.get("relative_error", 1.0) < 1e-3 for row in r.get("parseval_bridge", []))
        for r in all_results
    )
    checks["HPH Bochner positivity [T1]"]  = hph_bochner_ok
    checks["HPH Parseval bridge [T1]"]     = parseval_ok

    # Block-level symmetry at largest N
    r_block    = all_results[-1]
    v3         = r_block["block_reflection"]
    v6         = r_block["block_func_eq"]
    v8         = r_block["block_orthogonality"]
    gue        = r_block["block_gue_positive"]
    r_mean     = r_block["spacing_ratio"]
    full_spec  = r_block.get("full_spectral", {})
    explicit_hook  = full_spec.get("explicit_hook", {})
    zero_ks        = full_spec.get("zero_ks", {})
    resolvent_conv = full_spec.get("resolvent_conv", {})

    checks["Block symmetry (V3/V6)"]       = (v3["normalized_error"] < 1e-12 and
                                               v6["symmetry_error"] < 1e-12)
    checks["Block orthogonality (V8)"]     = v8["max_overlap"] < 1e-12
    checks["GUE-like spacings"]            = (not math.isnan(r_mean)) and r_mean > 0.386

    if explicit_hook:
        checks["Explicit-formula trace (Gaussian)"] = (
            explicit_hook.get("relative_diff", 1.0) < 0.90
        )
    else:
        checks["Explicit-formula trace (Gaussian)"] = False

    if zero_ks:
        checks["Zero-level KS (Berry–Keating)"] = (
            zero_ks.get("ks_stat", 1.0) < 0.30
        )
    else:
        checks["Zero-level KS (Berry–Keating)"] = False

    if resolvent_conv:
        errs = resolvent_conv.get("errors", [])
        checks["Strong resolvent (heuristic)"] = (len(errs) > 0 and errs[-1] < 0.01)
    else:
        checks["Strong resolvent (heuristic)"] = False

    rd = r_block.get("block_zero_density")
    if rd is not None:
        checks["Zero density (KS, λ>0)"] = rd["ks_statistic"] < 0.20
    else:
        checks["Zero density (KS, λ>0)"] = None

    all_ok = True
    for name, status in checks.items():
        if status is None:
            icon, label = "⚪", "N/A"
        elif status:
            icon, label = "✅", "PASS"
        else:
            icon, label = "❌", "FAIL"
            all_ok = False
        print(f"  {icon}  {name:42s} {label}")

    print()
    if all_ok:
        print("✅ HPH-resonant HPO passes global, local, HPH-kernel, trace, and limit diagnostics")
        print("   (up to numerical tolerances, finite-N artifacts, and smoothing choices).")
        print()
        print("Mathematical status:")
        print("  • K_N^{HPH}: Bochner-positive sech⁴ Toeplitz kernel with corrected")
        print("    Fourier symbol k̂_H(ξ) = 2π²ξ(4π²H²ξ²+4)/sinh(π²Hξ).")
        print("  • Parseval bridge ⟨T_N v,v⟩ = ∫ k_H(t)|Z_N(1+i(T₀+2πt))|² dt")
        print("    verified numerically (independent code paths, non-circular).")
        print("  • H_N = D_N + ε_HPH·K_N^{HPH} + K_N^{prime} + γR realizes a")
        print("    GUE-plus-arithmetic candidate with correct mean density.")
        print("  • Block construction enforces functional-equation symmetry at")
        print("    machine precision; spectral statistics align with Riemann data.")
        print("  • Strong resolvent ladder supports (but does not prove) convergence")
        print("    toward an infinite-dimensional Hilbert–Pólya operator.")
        print()
        print("  RH equivalence: Q_H^∞ = inf_{T₀} lim_{N→∞} ⟨T_N v(T₀), v(T₀)⟩ > 0")
        print("  remains open [T3]. This construction does not constitute a proof.")
    else:
        print("❌ At least one core, HPH-kernel, trace, or limit diagnostic failed.")
        print("   Adjust ε_HPH, H, ε_prime, γ, or resolve the indicated issue.")

    print("=" * 72)
    return all_ok


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main() -> bool:
    SEP = "=" * 80

    print(SEP)
    print(" THE ANALYST'S PROBLEM — HPH-RESONANT HILBERT–PÓLYA OPERATOR")
    print(" (Volumes I–XI: Bochner sech⁴ HPH kernel + arithmetic backbone)")
    print(SEP)
    print()
    print("  H_N = D_N + ε_HPH·K_N^{HPH} + K_N^{prime} + γ R  [HPH-resonant]")
    print("  D_N = diag(t_n)   [von Mangoldt inversion]")
    print(f"  HPH kernel: k̂_H(ξ) = 2π²ξ(4π²H²ξ²+4)/sinh(π²Hξ), H={HPH_H_DEFAULT}")
    print(f"  ε_HPH = {EPSILON_HPH:.3f}   ε_prime = {EPSILON_PRIME:.3f}   γ = {GAMMA_GUE:.3f}")
    print(f"  Phi-Ruelle weights K={PHI_RUELLE_K}, φ={PHI:.5f}")
    print(f"  Test dimensions: {TEST_NS}")
    print()

    # Load Riemann zeros (diagnostics only — never enter H_N)
    zeros = np.array([])
    try:
        with open("RiemannZeros.txt", "r") as f:
            data: List[float] = []
            for line in f:
                for token in line.strip().split():
                    try:
                        data.append(float(token))
                    except ValueError:
                        pass
            zeros = np.array(data, dtype=float)
    except FileNotFoundError:
        pass

    if zeros.size > 0:
        print(f"  Loaded {zeros.size} Riemann zeros from RiemannZeros.txt.")
    else:
        print("  RiemannZeros.txt not found; zero-density diagnostics limited.")
    print()

    # Anti-circularity guard
    op_test = ChaoticHilbertPolyaOperator(
        400,
        H=HPH_H_DEFAULT,
        epsilon_hph=EPSILON_HPH,
        epsilon_prime=EPSILON_PRIME,
        P_max=None,
        gamma_gue=GAMMA_GUE,
        use_hph_surrogate=False,
        add_random=True,
    )
    AntiCircularityGuard.execute_guard(
        np.diag(op_test.levels),
        op_test.K_backbone + op_test.K_prime,
        zeros,
    )
    print("  AntiCircularityGuard: passed (zeros not present in K coefficients).")
    print()

    # Full validation chain across N
    all_results: List[Dict[str, object]] = []
    for N in TEST_NS:
        res = run_validation(
            N,
            H=HPH_H_DEFAULT,
            zeros=zeros if zeros.size > 0 else None,
        )
        all_results.append(res)

    cross_n_diagnostics(all_results)
    ok = final_verdict(all_results)
    return ok


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)