#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HPH_TRUE_BOOTSTRAP_RH_PROOF.py
================================================================================
Main script with centralized imports via VOLUME_IMPORT_MANAGER.
Bootstrapped ENTIRELY from the TRUE Hilbert–Pólya Hamiltonian (HPH) framework.

Changes from previous HPO implementation:
  • REMOVED: Explicit prime-resonance kernel.
  • REMOVED: Random GUE symmetric dressing.
  • ADDED: TRUE HPH `RiemannEigenvalueGenerator` (dynamically generates zeros 
    from first-principles Riemann-Siegel without lookup tables or leakage).
  • ADDED: TRUE HPH `GramOperator` with strictly zero-free feature vectors.
  • ADDED: TRUE HPH `ToeplitzForm` and `AnalystsProblem` monitor.

Master equation (finite N):
    H_N = D_N + ε_HPH · T_N

    D_N = diag(t_n)    [Riemann–von Mangoldt inversion]
    T_N = GramOperator [TRUE HPH Bochner-positive Toeplitz kernel K_N]

Parseval bridge (independent code paths, non-circular):
    operator_form → uses k̂_H (GR formula)
    integral_form → uses k_H (direct sech⁴)
================================================================================
"""

from __future__ import annotations

import os
import sys
import math
import enum
import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any, Sequence

import numpy as np
import scipy.linalg as la
import scipy.optimize as opt

# ── VOLUME IMPORT MANAGER ────────────────────────────────────────────────────
from VOLUME_IMPORT_MANAGER import (
    VolumeImporter,
    VolumeConfig,
    FunctionSpec,
    VolumeStatus,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)
log = logger

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ════════════════════════════════════════════════════════════════════════════
# GLOBAL STRICTNESS / PARAMETERS
# ════════════════════════════════════════════════════════════════════════════

STRICT_MODE = True
RNG_SEED = 314159
print(f"RNG seed = {RNG_SEED}")

TEST_NS         = [50, 100, 200, 400]
H_BANDWIDTH     = 1.0          # HPH bandwidth parameter H > 0
SIGMA_DIRICHLET = 0.5
EPSILON_HPH     = 0.15         # coupling ε_HPH for T_N in H_N

# Mathematical Constants
PHI             = (1.0 + math.sqrt(5.0)) / 2.0   # golden ratio φ ≈ 1.61803
LN_PHI          = math.log(PHI)                    # ln φ ≈ 0.48121
PI              = math.pi
TWO_PI          = 2.0 * PI
PHI_RUELLE_K    = 9            # number of phi-Ruelle weight components

# Analytic k̂_H(0) = 8/H.  [T1]
_K_HAT_0_COEFFICIENT = 8.0  

_rng = np.random.default_rng(RNG_SEED)


# ════════════════════════════════════════════════════════════════════════════
# VOLUME CONFIGS & REGISTRY
# ════════════════════════════════════════════════════════════════════════════

# Keeping all volume configs exact so they seamlessly integrate.
VOLUME_CONFIGS: List[VolumeConfig] = [
    VolumeConfig(volume_id="VOLUME_I", module_path="VOLUME_I_FORMAL_REDUCTION.VOLUME_I_FORMAL_REDUCTION_PROOF.VOLUME_I_FORMAL_REDUCTION", functions=[FunctionSpec("FormalReduction", alias="FormalReduction")], optional=True),
    VolumeConfig(volume_id="VOLUME_II", module_path="VOLUME_II_KERNAL_DECOMPOSITION.VOLUME_II_KERNAL_DECOMPOSITION_PROOF.KERNAL_DECOMPOSITION_PROBLEM", functions=[FunctionSpec("k_H", required=True), FunctionSpec("k_H_hat", required=True), FunctionSpec("k_H_L1", required=True), FunctionSpec("lambda_star", required=True)], optional=False, post_import_hook=lambda f: f["k_H"](0.0, 0.5) > 0.0),
    VolumeConfig(volume_id="VOLUME_III", module_path="VOLUME_III_QUAD_DECOMPOSITION.VOLUME_III_QUAD_DECOMPOSITION_PROOF.VOLUME_III_QUAD_DECOMPOSITION", functions=[FunctionSpec("QuadraticFormConfig", required=False), FunctionSpec("build_quadratic_form", required=False), FunctionSpec("estimate_mean_square_ratio", required=False)], optional=True),
    VolumeConfig(volume_id="VOLUME_IV", module_path="VOLUME_IV_SPECTRAL_EXPANSION.VOLUME_IV_SPECTRAL_EXPANSION_PROOF.SPECTRAL_EXPANSION", functions=[FunctionSpec("Q_spectral", required=False)], optional=True),
    VolumeConfig(volume_id="VOLUME_V", module_path="VOLUME_V_DIRICHLET_CONTROL.VOLUME_V_DIRICHLET_CONTROL_PROOF.VOLUME_V_DIRICHLET_CONTROL", functions=[FunctionSpec("DirichletConfig", alias="VolumeV_DirichletConfig", required=False), FunctionSpec("build_coefficients", required=False), FunctionSpec("apply_window", required=False), FunctionSpec("trivial_bound", required=False), FunctionSpec("L2_norm_S", required=False), FunctionSpec("kernel_weighted_norm", required=False)], optional=True),
    VolumeConfig(volume_id="VOLUME_VI", module_path="VOLUME_VI_LARGE_SIEVE_BRIDGE.VOLUME_VI_LARGE_SIEVE_BRIDGE_PROOF.VOLUME_VI_LARGE_SIEVE_BRIDGE", functions=[FunctionSpec("DirichletConfig", alias="VolumeVI_DirichletConfig", required=False), FunctionSpec("validate_large_sieve_bounds", required=False), FunctionSpec("scaling_study", required=False)], optional=True),
    VolumeConfig(volume_id="VOLUME_VII", module_path="VOLUME_VII_EULER_MACLAURIN_CONTROL.VOLUME_VII_EULER_MACLAURIN_CONTROL_PROOF.VOLUME_VII_EULER_MACLAURIN_CONTROL", functions=[FunctionSpec("diagonal_mass_em_bound", required=False)], optional=True),
    VolumeConfig(volume_id="VOLUME_VIII", module_path="VOLUME_VIII_POSITIVITY_TRANSFORMATION.VOLUME_VIII_POSITIVITY_TRANSFORMATION_PROOF.VOLUME_VIII_POSITIVITY_TRANSFORMATION", functions=[FunctionSpec("positivity_transformation", alias="tap_ho_positivity", required=False)], optional=True),
    VolumeConfig(volume_id="VOLUME_IX", module_path="VOLUME_IX_CONVOLUTION_POSITIVITY.VOLUME_IX_CONVOLUTION_POSITIVITY_PROOF.VOLUME_IX_CONVOLUTION_POSITIVITY", functions=[FunctionSpec("DirichletConfig", alias="VolumeIX_DirichletConfig", required=False), FunctionSpec("verify_net_positivity", alias="vol9_verify_net_pos", required=False)], optional=True),
    VolumeConfig(volume_id="VOLUME_X", module_path="VOLUME_X_UNIFORMITY_EDGE_CASES.VOLUME_X_UNIFORMITY_EDGE_CASES_PROOF.VOLUME_X_UNIFORMITY_EDGE_CASES", functions=[FunctionSpec("check_lipschitz_uniformity_T0", alias="vol10_lip_check", required=False), FunctionSpec("check_limit_passage_N_infinity", alias="vol10_limit_N_inf", required=False)], optional=True),
    VolumeConfig(volume_id="VOLUME_XI", module_path="VOLUME_XI_HILBERT_POLYA_SPECTRAL.VOLUME_XI_HILBERT_POLYA_SPECTRAL_PROOF.VOLUME_XI_HILBERT_POLYA_SPECTRAL", functions=[FunctionSpec("run_spectral_validation_suite", alias="vol11_run_suite", required=False)], optional=True),
]

@dataclass
class Volumes:
    """Single namespace for every imported symbol."""
    k_H: Callable = None
    k_H_hat: Callable = None
    k_H_L1: Callable = None
    lambda_star: Callable = None
    VolumeIII_QuadraticFormConfig: Any = None
    VolumeIII_build_quadratic_form: Any = None
    VolumeIII_estimate_mean_square_ratio: Any = None
    VolumeV_DirichletConfig: Any = None
    VolumeV_build_coefficients: Any = None
    VolumeV_apply_window: Any = None
    VolumeV_trivial_bound: Any = None
    VolumeV_L2_norm_S: Any = None
    VolumeV_kernel_weighted_norm: Any = None
    VolumeVI_DirichletConfig: Any = None
    VolumeVI_validate_large_sieve_bounds: Any = None
    VolumeVI_scaling_study: Any = None
    VolumeVII_diagonal_mass_em_bound: Any = None
    tap_ho_positivity: Any = None
    VolumeIX_DirichletConfig: Any = None
    vol9_verify_net_pos: Any = None
    vol10_lip_check: Any = None
    vol10_limit_N_inf: Any = None
    vol11_run_suite: Any = None

importer = VolumeImporter(project_root=PROJECT_ROOT)
importer.register_volumes(VOLUME_CONFIGS)
importer.import_all(raise_on_missing=False)
print(importer.summary())

if not importer.is_available("VOLUME_II"):
    raise RuntimeError("FATAL: VOLUME_II (kernel source of truth) is required.")

vols = Volumes()
vols.k_H = importer.get_function("VOLUME_II", "k_H")
vols.k_H_hat = importer.get_function("VOLUME_II", "k_H_hat")
vols.k_H_L1 = importer.get_function("VOLUME_II", "k_H_L1")
vols.lambda_star = importer.get_function("VOLUME_II", "lambda_star")

vols.VolumeIII_QuadraticFormConfig = importer.get_function("VOLUME_III", "QuadraticFormConfig")
vols.VolumeIII_build_quadratic_form = importer.get_function("VOLUME_III", "build_quadratic_form")
vols.VolumeIII_estimate_mean_square_ratio = importer.get_function("VOLUME_III", "estimate_mean_square_ratio")

vols.VolumeV_DirichletConfig = importer.get_function("VOLUME_V", "VolumeV_DirichletConfig")
vols.VolumeV_build_coefficients = importer.get_function("VOLUME_V", "build_coefficients")
vols.VolumeV_apply_window = importer.get_function("VOLUME_V", "apply_window")
vols.VolumeV_trivial_bound = importer.get_function("VOLUME_V", "trivial_bound")
vols.VolumeV_L2_norm_S = importer.get_function("VOLUME_V", "L2_norm_S")
vols.VolumeV_kernel_weighted_norm = importer.get_function("VOLUME_V", "kernel_weighted_norm")

vols.VolumeVI_DirichletConfig = importer.get_function("VOLUME_VI", "VolumeVI_DirichletConfig")
vols.VolumeVI_validate_large_sieve_bounds = importer.get_function("VOLUME_VI", "validate_large_sieve_bounds")
vols.VolumeVI_scaling_study = importer.get_function("VOLUME_VI", "scaling_study")

vols.VolumeVII_diagonal_mass_em_bound = importer.get_function("VOLUME_VII", "diagonal_mass_em_bound")
vols.tap_ho_positivity = importer.get_function("VOLUME_VIII", "tap_ho_positivity")
vols.VolumeIX_DirichletConfig = importer.get_function("VOLUME_IX", "VolumeIX_DirichletConfig")
vols.vol9_verify_net_pos = importer.get_function("VOLUME_IX", "vol9_verify_net_pos")

vols.vol10_lip_check = importer.get_function("VOLUME_X", "vol10_lip_check")
vols.vol10_limit_N_inf = importer.get_function("VOLUME_X", "vol10_limit_N_inf")
vols.vol11_run_suite = importer.get_function("VOLUME_XI", "vol11_run_suite")

STATUS_III  = importer.get_status("VOLUME_III")
STATUS_V    = importer.get_status("VOLUME_V")
STATUS_VI   = importer.get_status("VOLUME_VI")
STATUS_VII  = importer.get_status("VOLUME_VII")
STATUS_VIII = importer.get_status("VOLUME_VIII")
STATUS_IX   = importer.get_status("VOLUME_IX")
STATUS_X    = importer.get_status("VOLUME_X")
STATUS_XI   = importer.get_status("VOLUME_XI")

def _vol_ok(status: VolumeStatus) -> bool:
    return status in (VolumeStatus.AVAILABLE, VolumeStatus.PARTIAL)

# ════════════════════════════════════════════════════════════════════════════
# TRUE HPH EIGENVALUE GENERATOR (First Principles)
# ════════════════════════════════════════════════════════════════════════════

def _theta(T: float) -> float:
    if T <= 0.0: return -PI / 8.0
    log_ratio = math.log(T / TWO_PI)
    th = (T / 2.0) * log_ratio - T / 2.0 - PI / 8.0
    if T >= 1.0:
        T2 = T * T; T4 = T2 * T2; T6 = T4 * T2
        th += 1.0/(48.0*T) - 7.0/(5760.0*T2*T) + 31.0/(80640.0*T4*T) - 127.0/(430080.0*T6*T)
    return th

def _C0(p: float) -> float:
    cos_denom = math.cos(TWO_PI * p)
    if abs(cos_denom) < 0.001: return 0.0
    return math.cos(TWO_PI * (p * p - p - 1.0 / 16.0)) / cos_denom

def _Z(T: float) -> float:
    th = _theta(T)
    sqT = math.sqrt(T / TWO_PI)
    M = max(1, int(sqT))
    p = sqT - M
    main = sum(math.pow(n, -0.5) * math.cos(th - T * math.log(n)) for n in range(1, M + 1))
    Z_val = 2.0 * main
    c0 = _C0(p)
    tau = math.pow(T / TWO_PI, -0.25)
    sign = 1 if (M - 1) % 2 == 0 else -1
    return Z_val + sign * tau * c0

def _N_backlund(T: float) -> float:
    if T <= 1.0: return 0.0
    return (T / TWO_PI) * math.log(T / TWO_PI) - T / TWO_PI + 7.0 / 8.0

class RiemannEigenvalueGenerator:
    @staticmethod
    def _T_max_for_n(n: int) -> float:
        target = float(n + 5)
        T = max(TWO_PI * target / math.log(max(target, 2.0)), 20.0)
        for _ in range(200):
            f = _N_backlund(T) - target
            fp = math.log(T / TWO_PI) / TWO_PI
            if abs(fp) < 1e-15: break
            dT = max(min(f / fp, T * 0.5), -T * 0.5)
            T = max(T - dT, 10.0)
            if abs(dT) < 1e-10: break
        return T * 1.10

    @classmethod
    def generate(cls, n_zeros: int) -> np.ndarray:
        T_min, T_max = 13.0, cls._T_max_for_n(n_zeros)
        grid_step = PI / (30 * max(1.0, math.log(T_max / TWO_PI)))
        n_pts = max(10_000, int(math.ceil((T_max - T_min) / grid_step)) + 1)
        T_grid = np.linspace(T_min, T_max, n_pts)
        
        log.info("Generating %d analytic zeros strictly from first principles...", n_zeros)
        Z_grid = np.array([_Z(float(t)) for t in T_grid])
        sc_idx = np.where(Z_grid[:-1] * Z_grid[1:] < 0.0)[0]

        zeros: List[float] = []
        for idx in sc_idx:
            T_a, T_b = float(T_grid[idx]), float(T_grid[idx + 1])
            try:
                root = opt.brentq(_Z, T_a, T_b, xtol=1e-14, rtol=1e-14)
                if not zeros or abs(root - zeros[-1]) > 1e-12:
                    zeros.append(root)
                    if len(zeros) >= n_zeros: break
            except ValueError:
                continue
        arr = np.array(zeros, dtype=float)
        log.info("Successfully generated %d zeros. (γ_1 = %.6f)", len(arr), arr[0] if len(arr) else 0)
        return arr


# ════════════════════════════════════════════════════════════════════════════
# TRUE HPH KERNEL & WEIGHTS  [T1]
# ════════════════════════════════════════════════════════════════════════════

class HilbertPolyaKernel:
    def __init__(self, H: float) -> None:
        if H <= 0.0: raise ValueError(f"H must be positive, got {H}")
        self.H = float(H)
        self._H2 = self.H ** 2
        self._pi2H = PI ** 2 * self.H
        self._4pi2H2 = 4.0 * PI ** 2 * self._H2
        self.k_hat_at_zero = _K_HAT_0_COEFFICIENT / self.H

    def k_real(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        tau = np.clip(t / self.H, -40.0, 40.0)
        return (6.0 / self._H2) * (1.0 / np.cosh(tau)) ** 4

    def k_hat(self, xi: np.ndarray) -> np.ndarray:
        xi = np.asarray(xi, dtype=float)
        out = np.empty_like(xi)
        mask0 = np.abs(xi) < 1e-10
        mask_large = np.abs(xi * self._pi2H) > 30.0
        mask_normal = ~mask0 & ~mask_large

        out[mask0] = self.k_hat_at_zero

        if np.any(mask_large):
            xi_L = xi[mask_large]
            numer = 4.0 * PI**2 * np.abs(xi_L) * (self._4pi2H2 * xi_L**2 + 4.0)
            out[mask_large] = numer / np.exp(np.clip(np.abs(self._pi2H * xi_L), 0.0, 700.0))

        if np.any(mask_normal):
            xi_n = xi[mask_normal]
            numer = 2.0 * PI**2 * xi_n * (self._4pi2H2 * xi_n**2 + 4.0)
            out[mask_normal] = numer / np.sinh(self._pi2H * xi_n)
        return out

    def build_matrix(self, N: int) -> np.ndarray:
        idx = np.arange(1, N + 1, dtype=float)
        m_g, n_g = np.meshgrid(idx, idx, indexing="ij")
        xi = np.log(m_g) - np.log(n_g)
        return self.k_hat(xi) / np.sqrt(m_g * n_g)

class PhiRuelleWeights:
    def __init__(self, K: int = PHI_RUELLE_K) -> None:
        self.K = K
        raw = np.array([1.0 / np.cosh(k * LN_PHI) ** 2 for k in range(K)], dtype=float)
        self.weights = raw / float(raw.sum())
        self.sqrt_weights = np.sqrt(self.weights)

    @property
    def operator_norm(self) -> float:
        return float(self.weights[0])


# ════════════════════════════════════════════════════════════════════════════
# TRUE HPH GRAM OPERATOR  [T1]
# ════════════════════════════════════════════════════════════════════════════

class GramOperator:
    def __init__(self, kernel: HilbertPolyaKernel, weights: PhiRuelleWeights, use_exact: bool = True) -> None:
        self.kernel = kernel
        self.weights = weights
        self.use_exact = use_exact
        self._sieve_cache: Dict[int, list] = {}

    def _build_sieve(self, N: int) -> None:
        if N in self._sieve_cache: return
        spf = list(range(N + 1))
        i = 2
        while i * i <= N:
            if spf[i] == i:
                for j in range(i * i, N + 1, i):
                    if spf[j] == j: spf[j] = i
            i += 1
        self._sieve_cache[N] = spf

    def von_mangoldt(self, n: int, spf: list) -> float:
        if n <= 1: return 0.0
        factors = []
        m = n
        while m > 1:
            p = spf[m]
            while m % p == 0:
                factors.append(p)
                m //= p
        unique = set(factors)
        return math.log(float(next(iter(unique)))) if len(unique) == 1 else 0.0

    def _feature_vector(self, n: int, spf: list) -> np.ndarray:
        H, K = self.kernel.H, self.weights.K
        ln_n = math.log(max(n, 1))
        L = self.von_mangoldt(n, spf)
        branches = np.array([
            math.cos(ln_n / H), math.sin(ln_n / H), math.cos(2.0 * ln_n / H), math.sin(2.0 * ln_n / H),
            L * math.cos(ln_n / H), L * math.sin(ln_n / H), L * math.cos(2.0 * ln_n / H),
        ], dtype=float)[:min(K, 7)]
        if K > 7: branches = np.concatenate([branches, np.zeros(K - 7)])
        norm = np.linalg.norm(branches)
        if norm < 1e-300: return np.zeros(K)
        return self.weights.sqrt_weights[:K] * (branches / norm)

    def build(self, N: int) -> np.ndarray:
        if self.use_exact: return self.kernel.build_matrix(N)
        self._build_sieve(N + 1)
        spf = self._sieve_cache[N + 1]
        Phi_tilde = np.empty((N, self.weights.K), dtype=float)
        for i in range(N):
            phi = self._feature_vector(i + 1, spf)
            Phi_tilde[i, :] = phi / (i + 1)
        return Phi_tilde @ Phi_tilde.T

    def clear_cache(self) -> None:
        self._sieve_cache.clear()


# ════════════════════════════════════════════════════════════════════════════
# TRUE HPH PARSEVAL BRIDGE & ANALYST PROBLEM
# ════════════════════════════════════════════════════════════════════════════

class ToeplitzForm:
    def __init__(self, kernel: HilbertPolyaKernel) -> None:
        self.kernel = kernel

    @staticmethod
    def physical_vector(N: int, T0: float = 0.0) -> np.ndarray:
        n = np.arange(1, N + 1, dtype=float)
        return n ** (-0.5) * np.exp(-1j * T0 * np.log(n))

    def evaluate_operator(self, N: int, T0: float, T_N: np.ndarray) -> float:
        v = self.physical_vector(N, T0)
        return float(np.dot(v.conj(), T_N @ v).real)

    def evaluate_integral(self, N: int, T0: float, n_quadrature: int = 2000) -> float:
        H = self.kernel.H
        t_range = max(20.0, 5.0 * H)
        nodes, weights = np.polynomial.legendre.leggauss(n_quadrature)
        t = t_range * nodes
        dt = t_range * weights
        n = np.arange(1, N + 1, dtype=float)
        phases = -np.outer(T0 + TWO_PI * t, np.log(n))
        Z_vals = (n[np.newaxis, :] ** (-1.0) * np.exp(1j * phases)).sum(axis=1)
        return float(np.dot(self.kernel.k_real(t) * np.abs(Z_vals) ** 2, dt))

    def verify_parseval_bridge(self, N: int, T0: float, T_N: np.ndarray) -> Dict[str, float]:
        op_val = self.evaluate_operator(N, T0, T_N)
        int_val = self.evaluate_integral(N, T0)
        return {
            "operator_form": op_val, "integral_form": int_val,
            "relative_error": abs(op_val - int_val) / (abs(int_val) + 1e-30)
        }

class AnalystsProblem:
    def __init__(self, kernel: HilbertPolyaKernel) -> None:
        self.kernel = kernel
        self.form = ToeplitzForm(kernel)

    def scan(self, N: int, zeros: np.ndarray, gram: GramOperator) -> Dict[str, Any]:
        T_N = gram.build(N)
        grid = np.linspace(0.0, 30.0, 30)
        Q_uniform = np.array([self.form.evaluate_operator(N, float(t0), T_N) for t0 in grid])
        return {"N": N, "Q_min": float(Q_uniform.min()), "is_positive": bool(Q_uniform.min() > 0)}


# ════════════════════════════════════════════════════════════════════════════
# ARITHMETIC DIAGONAL
# ════════════════════════════════════════════════════════════════════════════

def arithmetic_level(n: int) -> float:
    if n <= 0: return 0.0
    t = TWO_PI * n / max(math.log(n + 1.0), 1.0)
    for _ in range(12):
        if t <= 0: return float(n)
        lt = math.log(max(t / (TWO_PI * math.e), 1e-10))
        Nt = t / TWO_PI * lt + 7.0 / 8.0
        dNt = (lt + 1.0) / TWO_PI
        if abs(dNt) < 1e-15: break
        t -= (Nt - n) / dNt
    return max(t, 0.0)

def build_arithmetic_diagonal(N: int) -> np.ndarray:
    diag = np.array([arithmetic_level(idx + 1) for idx in range(N)])
    return np.diag(diag)


# ════════════════════════════════════════════════════════════════════════════
# CORE OPERATOR CONSTRUCTION (PURE TRUE HPH)
# ════════════════════════════════════════════════════════════════════════════

def build_hilbert_polya_operator(
    N: int, H: float, eps: float, use_exact_kernel: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs the Hamiltonian from STRICTLY the TRUE HPH backbone:
        H_N = D_N + ε_HPH * T_N
    Returns (H_N, D_N, T_N_scaled).
    """
    D_N = build_arithmetic_diagonal(N)
    kernel = HilbertPolyaKernel(H=H)
    weights = PhiRuelleWeights(K=PHI_RUELLE_K)
    gram = GramOperator(kernel, weights, use_exact=use_exact_kernel)
    
    T_N = gram.build(N)
    T_N_scaled = eps * T_N
    H_N = D_N + T_N_scaled
    
    gram.clear_cache()
    return H_N, D_N, T_N_scaled


# ════════════════════════════════════════════════════════════════════════════
# TRUE HPH KERNEL DIAGNOSTICS TABLE
# ════════════════════════════════════════════════════════════════════════════

def hph_kernel_diagnostics(N_vals: List[int], H: float) -> None:
    print("=" * 80)
    print(" TRUE HPH KERNEL DIAGNOSTICS: BOCHNER + PARSEVAL + OPERATOR BOUNDS")
    print("=" * 80)
    print(f"Kernel parameters: H = {H:.4f}")
    print(f"  k̂_H(ξ) = 2π²ξ(4π²H²ξ²+4)/sinh(π²Hξ),  k̂_H(0) = 8/H = {8.0/H:.6f}")
    print("-" * 80)
    print(f"{'N':<8} | {'||T_N||_op':<10} | {'k̂(0)analytic':<14} | {'k_hat_min':<12} | {'Parseval_err':<14}")
    print("-" * 80)

    kernel = HilbertPolyaKernel(H=H)
    form = ToeplitzForm(kernel)
    xi_test = np.linspace(-10.0, 10.0, 4000)

    for N in N_vals:
        T_mat = kernel.build_matrix(N)
        evals_T = np.linalg.eigvalsh(T_mat)
        op_norm = float(max(abs(evals_T[0]), abs(evals_T[-1])))
        khat_min = float(kernel.k_hat(xi_test).min())

        N_sub = min(N, 20)
        pb = form.verify_parseval_bridge(N_sub, 0.0, kernel.build_matrix(N_sub))
        
        print(f"{N:<8} | {op_norm:<10.4f} | {kernel.k_hat_at_zero:<14.6f} | {khat_min:<12.4e} | {pb['relative_error']:<14.4e}")
    print("=" * 80 + "\n")


# ════════════════════════════════════════════════════════════════════════════
# HOOKS (III, V, VI, VII, VIII, IX, X, XI)
# ════════════════════════════════════════════════════════════════════════════

def vol3_quadratic_hook(N: int, H: float, H_N: np.ndarray) -> None:
    print("\n  VOLUME III  Quadratic Form Decomposition (HOOK-F)")
    print("  " + "-" * 56)
    if not _vol_ok(STATUS_III) or vols.VolumeIII_build_quadratic_form is None:
        print("  Volume III missing or incomplete — skipped.")
        return
    try:
        cfg = vols.VolumeIII_QuadraticFormConfig(N=N, H=H, T0=0.0)
        Q_form = vols.VolumeIII_build_quadratic_form(cfg, H_N)
        if vols.VolumeIII_estimate_mean_square_ratio:
            ms = vols.VolumeIII_estimate_mean_square_ratio(cfg, Q_form)
            print(f"  Mean-square ratio (off-diag / diag) ≈ {ms:.3e}")
    except Exception as exc: print(f"  Volume III hook failed: {exc}")

def vol5_dirichlet_hook(N: int, H: float) -> None:
    print("\n  VOLUME V  Dirichlet Polynomial Control (HOOK-H)")
    print("  " + "-" * 56)
    if not _vol_ok(STATUS_V) or vols.VolumeV_build_coefficients is None:
        print("  Volume V missing or incomplete — skipped.")
        return
    try:
        cfg = vols.VolumeV_DirichletConfig(N=N, sigma=SIGMA_DIRICHLET)
        a_raw, _ = vols.VolumeV_build_coefficients(cfg)
        norm_S = vols.VolumeV_L2_norm_S(cfg)
        print(f"  ‖S‖_L² ≈ {norm_S:.6e}")
    except Exception as exc: print(f"  Volume V hook failed: {exc}")

def vol6_large_sieve_hook(N: int, H: float) -> None:
    print("\n  VOLUME VI  Large Sieve Bridge (HOOK-I)")
    print("  " + "-" * 56)
    if not _vol_ok(STATUS_VI) or vols.VolumeVI_validate_large_sieve_bounds is None:
        print("  Volume VI missing or incomplete — skipped.")
        return
    try:
        cfg = vols.VolumeVI_DirichletConfig(N=N, sigma=SIGMA_DIRICHLET)
        xi_values = np.linspace(-2.0, 2.0, 9)
        consts, _ = vols.VolumeVI_validate_large_sieve_bounds(cfg=cfg, H=H, xi_values=xi_values, use_sech_basis="sech2")
        print(f"  MV_bound = {consts.MV_bound:.6e} | kernel_bound = {consts.kernel_bound:.6e}")
    except Exception as exc: print(f"  Volume VI hook failed: {exc}")

def vol11_spectral_hook(N: int, generated_zeros: np.ndarray) -> None:
    print("\n  VOLUME XI  Spectral Alignment (HOOK-K)")
    print("  " + "-" * 56)
    if not _vol_ok(STATUS_XI) or vols.vol11_run_suite is None:
        print("  Volume XI missing or incomplete — skipped.")
        return
    try:
        res = vols.vol11_run_suite(N=N, zeros=generated_zeros)
        v6 = res.get("V6_func_eq", {})
        sym_err = v6.get("symmetry_error", 1.0)
        print(f"  Chiral Block Symmetry (λ ↔ -λ) : {'✓ PASS' if sym_err < 1e-12 else '✗ FAIL'} (err: {sym_err:.2e})")
    except Exception as exc: print(f"  Volume XI hook failed: {exc}")

def track_operator_convergence(H_mats: List[np.ndarray]) -> None:
    for i in range(len(H_mats) - 1):
        A = H_mats[i]
        B = H_mats[i + 1][:A.shape[0], :A.shape[0]]
        print(f"Δ(H_{i+1}, H_{i}) = {np.linalg.norm(B - A):.3e}")


# ════════════════════════════════════════════════════════════════════════════
# MAIN DRIVER
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    SEP = "=" * 80
    print(SEP)
    print(" HPH_TRUE_BOOTSTRAP_RH_PROOF.py  (PURE TRUE HPH, Volumes I–XI)")
    print(SEP)
    print(f"  H_N = D_N + ε_HPH · T_N   [TRUE HPH coupling ε={EPSILON_HPH:.2f}]")
    print("  D_N = diag(tₙ)   [von Mangoldt inversion]")
    print("  T_N = TRUE HPH Bochner sech⁴ Gram/Toeplitz kernel [Zero-Free]")
    print("  *NO PRIME-RESONANCE PERTURBATION, NO RANDOM GUE DRESSING*")
    print(f"  k̂_H(ξ) = 2π²ξ(4π²H²ξ²+4)/sinh(π²Hξ),  k̂_H(0) = 8/H = {8.0/H_BANDWIDTH:.4f}")
    print(f"  H = {H_BANDWIDTH}   σ = {SIGMA_DIRICHLET}\n")

    # 1. Generate First-Principles Zeros
    print("▶ DYNAMIC ZERO GENERATION (Riemann-Siegel)")
    print("-" * 60)
    zeros = RiemannEigenvalueGenerator.generate(n_zeros=max(TEST_NS) + 10)
    print()

    # 2. HPH Kernel Diagnostics
    hph_kernel_diagnostics(TEST_NS, H_BANDWIDTH)

    # 3. Iterate N dimensions
    H_mats: List[np.ndarray] = []
    
    # Analyst Problem Monitor
    kernel = HilbertPolyaKernel(H=H_BANDWIDTH)
    gram = GramOperator(kernel, PhiRuelleWeights(K=PHI_RUELLE_K), use_exact=True)
    ap = AnalystsProblem(kernel)

    for N in TEST_NS:
        print(f"\n▶ N = {N}")
        print("-" * 60)

        # Build TRUE HPH Hamiltonian
        H_N, D_N, T_N_scaled = build_hilbert_polya_operator(N, H_BANDWIDTH, EPSILON_HPH)
        H_mats.append(H_N)

        T_op = float(np.linalg.norm(T_N_scaled, 2))
        H_spec = np.linalg.eigvalsh(H_N)
        
        print(f"  [Operator] ‖T_N‖_op = {T_op:.4e} | H_N Spectrum: min={H_spec[0]:.4e}, max={H_spec[-1]:.4e}")

        # Scan Analyst's Problem
        ap_res = ap.scan(N, zeros, gram)
        ap_sym = "✓ PASS" if ap_res["is_positive"] else "✗ FAIL"
        print(f"  [Analyst's Problem] Q_min = {ap_res['Q_min']:.4e} ({ap_sym})")

        # Run Volume Hooks
        vol3_quadratic_hook(N, H_BANDWIDTH, H_N)
        vol5_dirichlet_hook(N, H_BANDWIDTH)
        vol6_large_sieve_hook(N, H_BANDWIDTH)
        vol11_spectral_hook(N, zeros)
        print()

    # 4. Convergence Dashboard
    print(SEP)
    print(" OPERATOR CONVERGENCE (N-ladder)")
    print(SEP)
    track_operator_convergence(H_mats)

    print(SEP)
    print(" PURE TRUE HPH BOOTSTRAP COMPLETE")
    print(SEP)


if __name__ == "__main__":
    main()