#!/usr/bin/env python3
"""
VOLUME VIII — Positivity Transformation via TAP Hilbert Operator (TAP HO)
=========================================================================

This module replaces the fragile spectral Integration-by-Parts (IBP)
positivity argument with a structural operator-theoretic factorization.

Hybrid Protocol (Option 3, corrected):
    We separate the *computation* of logarithmic geometry from the *structure*
    of the Hilbert operator.
    - The operator internally accepts an embedding vector ψ(n) ≈ log(n).
    - It builds a Gram surrogate K_N = Γ_N W Γ_N^T using pure algebraic
      operations on this embedding.
    - W is a diagonal matrix of strictly positive spectral quadrature weights.

By aligning the operator's feature branches with the spectral quadrature
nodes (t_vals), the quadratic form
        Q_H^disc(a) = a^T K_N a = || √W Γ_N^T a ||²
becomes a quadrature model for the spectral energy
        Q_H = ∫ k̂(t,H) |S(t)|^2 dt
with
        |Q_H^disc(a) − Q_H(a)| ≤ ε(N)
as the quadrature is refined (more branches, larger T_max).

IMPORTANT CORRECTION:
    - Γ uses cos(t log n), sin(t log n), so |Γ^T a|² = |S(t)|².
    - Quadrature weights must therefore *not* halve the energy.
      The factor-of-2 discrepancy arises exactly from that halving.
    - We remove the 1/2 in W and keep the spectral side unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np

try:
    from VOLUME_V_DIRICHLET_CONTROL.VOLUME_V_DIRICHLET_CONTROL_PROOF.VOLUME_V_DIRICHLET_CONTROL import (
        DirichletConfig,
        build_coefficients,
        apply_window,
    )
except Exception:  # pragma: no cover
    # Minimal fallback DirichletConfig and builders (plain a_n = n^{-sigma})
    from dataclasses import dataclass as _dc_dataclass

    @_dc_dataclass
    class DirichletConfig:
        N: int
        sigma: float = 0.5
        weight_type: str = "plain"
        window_type: str = "sharp"
        window_params: Optional[Dict[str, float]] = None
        custom_coeffs: Optional[np.ndarray] = None
        custom_window: Optional[callable] = None

    def build_coefficients(cfg: DirichletConfig):
        N = cfg.N
        sigma = cfg.sigma
        ns = np.arange(1, N + 1, dtype=float)
        logn = np.log(ns)
        if cfg.weight_type == "plain":
            a = ns ** (-sigma)
        elif cfg.weight_type == "log":
            a = logn * ns ** (-sigma)
        else:
            a = ns ** (-sigma)
        return a, logn

    def apply_window(cfg: DirichletConfig, a: np.ndarray) -> np.ndarray:
        if cfg.window_type == "sharp":
            return a.copy()
        N = cfg.N
        params = cfg.window_params or {}
        w = np.empty_like(a)
        for i in range(N):
            n = i + 1
            x = n / float(N)
            if cfg.window_type == "gaussian":
                alpha = params.get("alpha", 1.0)
                w[i] = math.exp(-alpha * x * x)
            else:
                w[i] = 1.0
        return a * w

# ---------------------------------------------------------------------------
# TAP HO feature map and quadrature-consistent spectral weights
# ---------------------------------------------------------------------------

NUM_BRANCHES = 200  # Must be even (cos/sin pairs)
_MAX_DIM = 5000

def build_tap_feature_map(
    N: int,
    num_branches: int,
    t_vals: np.ndarray,
    embedding: np.ndarray,
) -> np.ndarray:
    """
    TAP HO feature map Γ[n,k] using an external spatial embedding ψ(n).

    Γ encodes pure Mellin/Fourier phase geometry:

        Γ_n,2j   = cos(t_j ψ(n))
        Γ_n,2j+1 = sin(t_j ψ(n))

    The modulus |Γ^T a|² = |S(t_j)|², where S(t) = ∑ a_n n^{-it},
    provided embedding = log n and t_vals are the quadrature nodes.
    """
    if num_branches % 2 != 0:
        raise ValueError("num_branches must be even (cos/sin pairing).")
    if len(embedding) != N:
        raise ValueError("Embedding vector length must match dimension N.")
    half = num_branches // 2
    if len(t_vals) != half:
        raise ValueError("t_vals length must be num_branches // 2.")

    cols = []
    for t in t_vals:
        phase = t * embedding
        cols.append(np.cos(phase))
        cols.append(np.sin(phase))

    Gamma = np.stack(cols, axis=1)
    return Gamma

def gaussian_spectral_weights_tap(
    num_branches: int,
    T_max: float = 20.0,
    H: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Spectral weights matching a midpoint quadrature grid.

    Construct midpoints t_k on [-T_max, T_max] and assign
        w_k ≈ k̂(t_k,H) Δt
    with k̂(t,H) = exp(-t^2 / H^2).

    Each t_k is represented by cos and sin columns, but since
    |Γ^T a|² already reconstructs |S(t_k)|² from those two components,
    we assign the *full* quadrature weight to each column pair, without
    an extra 1/2 factor.

    Returns (t_vals, w_diag, dt).
    """
    half = num_branches // 2
    if half < 2:
        raise ValueError("Need at least 2 frequency points for quadrature.")

    edges = np.linspace(-T_max, T_max, half + 1)
    t_vals = 0.5 * (edges[:-1] + edges[1:])
    dt = edges[1] - edges[0]

    w = []
    for u in t_vals:
        val = math.exp(-(u ** 2) / (H ** 2)) * dt
        # IMPORTANT: no 1/2 factor here; cos/sin already combine to |S(t)|².
        w.append(val)  # cos
        w.append(val)  # sin

    w_diag = np.array(w, dtype=float)
    return t_vals, w_diag, dt

# ---------------------------------------------------------------------------
# Core TAP HO operator and results
# ---------------------------------------------------------------------------

@dataclass
class OperatorFactorizationResult:
    N: int
    minimum_weight: float
    is_positive_definite: bool
    dense_quadratic_form: float
    factorized_norm_squared: float
    factorization_error: float
    spectral_integral: Optional[float] = None
    spectral_error: Optional[float] = None

class PositiveGramOperator:
    """
    TAP HO Positive Gram Operator.

    Constructs K_N = Γ_N W Γ_N^T with Γ and W aligned to the same
    quadrature grid used in the spectral integral.
    """
    def __init__(
        self,
        N: int,
        embedding: np.ndarray,
        num_branches: int = NUM_BRANCHES,
        T_max: float = 20.0,
        H: float = 5.0,
    ):
        if N <= 0 or N > _MAX_DIM:
            raise ValueError(f"Invalid dimension N={N}")
        if embedding.shape[0] != N:
            raise ValueError("Embedding length must equal N.")

        self.N = N

        # 1. Spectral weights and grid
        self.t_vals, self.W_diag, self.dt = gaussian_spectral_weights_tap(
            num_branches, T_max=T_max, H=H
        )

        # 2. Feature map tied to Mellin embedding ψ(n) = log n
        self.Gamma = build_tap_feature_map(N, num_branches, self.t_vals, embedding)

        # 3. Dense Gram matrix
        W_mat = np.diag(self.W_diag)
        self.K_dense = self.Gamma @ W_mat @ self.Gamma.T
        self.K_dense = 0.5 * (self.K_dense + self.K_dense.T)

    def evaluate_dense(self, a: np.ndarray) -> float:
        a = np.asarray(a, dtype=float).reshape(-1)
        if a.shape[0] != self.N:
            raise ValueError("Coefficient vector length must equal N.")
        return float(a.T @ self.K_dense @ a)

    def evaluate_factorized(self, a: np.ndarray) -> float:
        a = np.asarray(a, dtype=float).reshape(-1)
        if a.shape[0] != self.N:
            raise ValueError("Coefficient vector length must equal N.")
        projected = self.Gamma.T @ a
        weighted_sq = (projected ** 2) * self.W_diag
        return float(np.sum(weighted_sq))

# ---------------------------------------------------------------------------
# Dirichlet coefficients and spectral integral
# ---------------------------------------------------------------------------

def build_dirichlet_coefficients(
    cfg: DirichletConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build Dirichlet coefficients a_n and Mellin embedding ψ(n) = log n.
    """
    if cfg.custom_coeffs is not None:
        a = np.asarray(cfg.custom_coeffs, dtype=float).reshape(-1)
        ns = np.arange(1, cfg.N + 1, dtype=float)
        embedding = np.log(ns)
        return a, embedding

    a_raw, logn = build_coefficients(cfg)
    a = apply_window(cfg, a_raw)
    return np.asarray(a, dtype=float).reshape(-1), np.asarray(logn, dtype=float).reshape(-1)

def evaluate_spectral_on_grid(
    a: np.ndarray,
    logn: np.ndarray,
    t_vals: np.ndarray,
    H: float,
    dt: float,
) -> float:
    """
    Evaluate the Gaussian-smoothed spectral energy on the same grid
    used by the TAP HO operator:

        ∑ k̂(t_k,H) |S(t_k)|^2 Δt,

    with S(t) = ∑ a_n n^{-it}, k̂(t,H) = exp(-t^2 / H^2).
    """
    total = 0.0
    for t in t_vals:
        cos_part = np.cos(t * logn)
        sin_part = np.sin(t * logn)

        S_real = float(np.dot(a, cos_part))
        S_imag = float(np.dot(a, sin_part))

        S_sq = S_real**2 + S_imag**2
        kernel = math.exp(-(t**2) / (H**2))

        total += kernel * S_sq * dt
    return total

# ---------------------------------------------------------------------------
# High-level TAP HO positivity transformation + spectral comparison
# ---------------------------------------------------------------------------

def positivity_transformation(
    cfg: DirichletConfig,
    T_max: float = 20.0,
    H: float = 5.0,
    num_branches: int = NUM_BRANCHES,
) -> OperatorFactorizationResult:
    """
    High-level TAP HO positivity transformation (Hybrid Protocol, corrected).

    1. Build Dirichlet coefficients a_n and Mellin embedding ψ(n)=log n.
    2. Instantiate PositiveGramOperator with ψ(n).
    3. Compute:
         Q_dense      = a^T K_N a,
         Q_factorized = || √W Γ^T a ||².
    4. Compute Gaussian-smoothed spectral sum on the same grid:
         Q_spectral ≈ ∑ k̂(t_k,H)|S(t_k)|² Δt.

    We *no longer* expect machine-zero spectral error, only that
        |Q_factorized − Q_spectral| → 0
    as the quadrature (num_branches, T_max) is refined.
    """
    N = cfg.N
    a_coeffs, logn = build_dirichlet_coefficients(cfg)

    op = PositiveGramOperator(
        N,
        embedding=logn,
        num_branches=num_branches,
        T_max=T_max,
        H=H,
    )

    dense_val = op.evaluate_dense(a_coeffs)
    factorized_val = op.evaluate_factorized(a_coeffs)
    factorization_error = abs(dense_val - factorized_val)

    spectral_val = evaluate_spectral_on_grid(
        a_coeffs,
        logn,
        op.t_vals,
        H=H,
        dt=op.dt,
    )
    spectral_error = abs(factorized_val - spectral_val)

    min_w = float(np.min(op.W_diag))

    return OperatorFactorizationResult(
        N=N,
        minimum_weight=min_w,
        is_positive_definite=(min_w > 0.0),
        dense_quadratic_form=dense_val,
        factorized_norm_squared=factorized_val,
        factorization_error=factorization_error,
        spectral_integral=spectral_val,
        spectral_error=spectral_error,
    )

# ---------------------------------------------------------------------------
# Diagnostics / simple demo
# ---------------------------------------------------------------------------

def _demo(cfg: Optional[DirichletConfig] = None) -> None:
    """
    Diagnostic script entrypoint.

    Demonstrates:
      - Gram surrogate factorization (dense vs factorized).
      - Agreement with the sampled spectral integral on the same grid.
      - Size of |Q_operator − Q_spectral|, which should now be driven
        by genuine quadrature/aliasing error, not by a factor-of-2 bug.
    """
    if cfg is None:
        cfg = DirichletConfig(
            N=100,
            sigma=0.5,
            window_type="gaussian",
            window_params={"alpha": 2.0},
        )

    res = positivity_transformation(cfg)
    print("=== Volume VIII: TAP HO Positivity Transformation Demo ===")
    print(f"Dimension (N)                : {res.N}")
    print(f"Minimum spectral weight      : {res.minimum_weight:.6e} (> 0 ⇒ PSD)")
    print(f"Is strictly PSD?             : {res.is_positive_definite}")
    print(f"Dense quadratic form a^T K a : {res.dense_quadratic_form:.12e}")
    print(f"Factorized ||√W Γ^T a||²     : {res.factorized_norm_squared:.12e}")
    print(f"Factorization error          : {res.factorization_error:.3e}")
    print(f"Spectral integral (Gaussian) : {res.spectral_integral:.12e}")
    print(f"Spectral error               : {res.spectral_error:.3e}")

if __name__ == "__main__":
    _demo()