#!/usr/bin/env python3
"""
VOLUME_VI_LARGE_SIEVE_BRIDGE.py
================================

Volume VI: Large Sieve Bridge

Computational bridge between discrete Dirichlet polynomials (Volume V)
and continuous spectral integrals, using explicit Montgomery–Vaughan
bounds and the sech^2-based kernel from Volume IV.

Core capabilities

1. Frequency set and coefficients
   - Frequencies γ_n = log n.
   - Coefficients a_n from Volume V DirichletConfig (plain, log,
     von Mangoldt, or custom), with windows applied.

2. Montgomery–Vaughan inequality
   - For real frequencies γ_n with minimum separation δ,
       sup_ξ |∑ a_n e^{2π i ξ γ_n}|^2 ≤ (N + 1/δ) ∑ |a_n|^2.

3. Kernel-decay bound for off-diagonal
   - Uses Volume IV kernel k_hat(ξ, H) to bound the off-diagonal
     Dirichlet quadratic form in frequency space.

4. Discrete-to-continuous transition
   - Approximates discrete ∑ f(n) by ∫ f(t) dt with explicit error
     control using a simple Euler–Maclaurin style bound.

5. Explicit constant tracking
   - All bounds return explicit finite floats; a dedicated data class
     records (N, δ, ∑|a_n|^2, MV constants, kernel constants, error).

6. Diagnostics
   - Direct exact off-diagonal computation.
   - Validation of MV bound and kernel bound against reality.
   - Scaling tests w.r.t. N and σ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import mpmath as mp
import numpy as np

# ---------------------------------------------------------------------------
# 0. Global settings
# ---------------------------------------------------------------------------

mp.mp.dps = 80  # high precision for kernel evaluations


# ---------------------------------------------------------------------------
# 1. Import / fallback for Volume V structures
# ---------------------------------------------------------------------------

try:
    # Preferred: import from your existing Volume V module
    from VOLUME_V_DIRICHLET_CONTROL.VOLUME_V_DIRICHLET_CONTROL_PROOF.VOLUME_V_DIRICHLET_CONTROL import (  # type: ignore # noqa: E501
        DirichletConfig,
        build_coefficients,
        apply_window,
    )
except Exception:  # fallback if real module not importable

    @dataclass
    class DirichletConfig:  # minimal fallback
        N: int
        sigma: float = 0.5
        weight_type: str = "plain"  # "plain", "log", "von_mangoldt", "custom"
        window_type: str = "sharp"  # "sharp", "gaussian", "exponential", "bump", "log_sech2"
        window_params: Dict[str, float] | None = None
        custom_coeffs: np.ndarray | None = None
        custom_window: callable | None = None

    def _von_mangoldt(n: int) -> float:
        if n < 2:
            return 0.0
        m = n
        p = 2
        while p * p <= m:
            if m % p == 0:
                while m % p == 0:
                    m //= p
                return math.log(p) if m == 1 else 0.0
            p += 1 if p == 2 else 2
        return math.log(m)

    def build_coefficients(cfg: DirichletConfig) -> Tuple[np.ndarray, np.ndarray]:
        N = cfg.N
        sigma = cfg.sigma
        ns = np.arange(1, N + 1, dtype=float)
        logn = np.log(ns)
        if cfg.weight_type == "plain":
            a = ns ** (-sigma)
        elif cfg.weight_type == "log":
            a = logn * ns ** (-sigma)
        elif cfg.weight_type == "von_mangoldt":
            lam = np.array([_von_mangoldt(int(n)) for n in ns], dtype=float)
            a = lam * ns ** (-sigma)
        elif cfg.weight_type == "custom":
            if cfg.custom_coeffs is None or len(cfg.custom_coeffs) != N:
                raise ValueError("custom_coeffs must be provided and length N")
            a = np.array(cfg.custom_coeffs, dtype=float)
        else:
            raise ValueError(f"Unknown weight_type {cfg.weight_type}")
        return a, logn

    def apply_window(cfg: DirichletConfig, a: np.ndarray) -> np.ndarray:
        N = cfg.N
        wt = cfg.window_type
        params = cfg.window_params or {}
        if wt == "sharp":
            return a.copy()
        w = np.empty_like(a)
        for i in range(N):
            n = i + 1
            x = n / float(N)
            if wt == "gaussian":
                alpha = params.get("alpha", 1.0)
                w[i] = math.exp(-alpha * x * x)
            elif wt == "exponential":
                alpha = params.get("alpha", 1.0)
                w[i] = math.exp(-alpha * x)
            elif wt == "bump":
                if x <= 0.0 or x >= 1.0:
                    w[i] = 0.0
                else:
                    t = x * (1.0 - x)
                    w[i] = math.exp(-1.0 / t)
            elif wt == "log_sech2":
                T = params.get("T", math.log(N))
                H = params.get("H", 1.0)
                z = (math.log(n) - T) / H
                w[i] = 1.0 / math.cosh(z) ** 2
            elif wt == "custom":
                if cfg.custom_window is None:
                    raise ValueError("custom_window is required")
                w[i] = cfg.custom_window(n, N)
            else:
                raise ValueError(f"Unknown window_type {wt}")
        return a * w


# ---------------------------------------------------------------------------
# 2. Volume IV kernel k_hat(ξ, H)
# ---------------------------------------------------------------------------

def k_hat(xi: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    """
    Sech^2-based spectral kernel:

        k_hat(ξ, H) = ((2πξ)^2 + 4/H^2) * w_hat(ξ, H),

    with w_H(t) = sech^2(t/H) and

        w_hat(ξ, H) = π H * (2π ξ H) / sinh(π^2 ξ H).

    Mirrors the Volume IV implementation, including asymptotic handling.
    """
    xi = mp.mpf(xi)
    H = mp.mpf(H)

    if xi == 0:
        return mp.mpf("8") / (H ** 2)

    a = mp.fabs(xi)
    num = (2 * mp.pi * a) ** 2 + 4 / (H ** 2)
    arg = (mp.pi ** 2) * a * H

    if arg > 50:
        exp_term = mp.e ** (-arg)
        w_hat_val = mp.pi * H * (2 * mp.pi * a * H) * 2 * exp_term
    else:
        w_hat_val = mp.pi * H * (2 * mp.pi * a * H) / mp.sinh(arg)

    val = num * w_hat_val
    return val if val >= 0 else mp.mpf("0")


# ---------------------------------------------------------------------------
# 3. Core Large Sieve objects
# ---------------------------------------------------------------------------

def log_frequencies(N: int) -> np.ndarray:
    """
    Frequencies γ_n = log n, n = 1..N.
    """
    ns = np.arange(1, N + 1, dtype=float)
    return np.log(ns)


def min_separation(gamma: np.ndarray) -> float:
    """
    Minimum separation δ = min_{r != s} |γ_r - γ_s|.

    For γ_n = log n this is asymptotically ~ 1/N; here we compute it
    directly on the sorted array.
    """
    if gamma.size < 2:
        return float("inf")
    diffs = np.diff(np.sort(gamma))
    diffs_pos = diffs[diffs > 0]
    if diffs_pos.size == 0:
        return float("inf")
    return float(np.min(diffs_pos))


# ---------------------------------------------------------------------------
# 4. Explicit constant container
# ---------------------------------------------------------------------------

@dataclass
class LargeSieveConstants:
    N: int
    min_separation: float
    sum_abs_sq: float
    MV_constant: float           # N + 1/δ
    MV_bound: float              # (N + 1/δ) * sum_abs_sq
    kernel_bound_constant: float # max k_hat over off-diagonal pairs
    kernel_bound: float          # numeric kernel off-diagonal bound
    discrete_to_cont_error: float


# ---------------------------------------------------------------------------
# 5. Montgomery–Vaughan bound
# ---------------------------------------------------------------------------

def montgomery_vaughan_bound(
    gamma: np.ndarray,
    a: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Montgomery–Vaughan large sieve bound

        sup_ξ |S(ξ)|^2 ≤ (N + 1/δ) ∑|a_n|^2,

    where S(ξ) = ∑ a_n e^{2π i ξ γ_n} and
          δ = min_{n != m} |γ_n - γ_m|.

    Returns (δ, MV_constant, MV_bound).
    """
    N = len(a)
    if N == 0:
        return float("inf"), 0.0, 0.0
    δ = min_separation(gamma)
    sum_abs_sq = float(np.sum(np.abs(a) ** 2))
    MV_constant = N + (1.0 / δ if δ > 0 else 0.0)
    MV_bound = MV_constant * sum_abs_sq
    return δ, MV_constant, MV_bound


# ---------------------------------------------------------------------------
# 6. Kernel-decay off-diagonal bound
# ---------------------------------------------------------------------------

def kernel_decay_off_diagonal_bound(
    a: np.ndarray,
    logn: np.ndarray,
    H: float,
) -> Tuple[float, float]:
    """
    Kernel-based off-diagonal bound.

    For n != m we bound the off-diagonal contribution by

        kernel_bound ≤ ∑_{n != m} |a_n a_m| k_hat(log n - log m, H).

    Returns (kernel_bound_constant, kernel_bound) where
      kernel_bound_constant = max_{n != m} k_hat(log n - log m, H).
    """
    N = len(a)
    if N <= 1:
        return 0.0, 0.0

    # pairwise differences of logn
    logn_col = logn.reshape(-1, 1)
    Δ = logn_col - logn_col.T  # N x N

    mask_off = ~np.eye(N, dtype=bool)
    Δ_off = Δ[mask_off]

    # evaluate kernel
    k_vals = np.array([float(k_hat(x, H)) for x in Δ_off], dtype=float)
    kernel_bound_constant = float(np.max(k_vals))

    a_abs = np.abs(a)
    prod = (a_abs.reshape(-1, 1) * a_abs.reshape(1, -1))[mask_off]
    weight = prod * k_vals
    kernel_bound = float(np.sum(weight))

    return kernel_bound_constant, kernel_bound


# ---------------------------------------------------------------------------
# 7. Discrete-to-continuous quadrature (coarse Euler–Maclaurin bound)
# ---------------------------------------------------------------------------

def discrete_to_continuous_quadrature(
    a: np.ndarray,
) -> Tuple[float, float]:
    """
    Approximate S_N = ∑_{n=1}^N |a_n|^2 by an integral and bound the error.

    We take the step-function model a(t) on [1,N] and note that the midpoint
    rule with grid spacing 1 gives I ≈ S_N. A conservative Euler–Maclaurin
    style error bound is

        |S_N - I| <= 0.5 (|a_1|^2 + |a_N|^2) + ∑_{n=1}^{N-1} | |a_{n+1}|^2 - |a_n|^2 |.

    Returns (I, error_bound). Here I = S_N for simplicity.
    """
    N = len(a)
    if N == 0:
        return 0.0, 0.0

    a_sq = np.abs(a) ** 2
    S_N = float(np.sum(a_sq))

    I = S_N
    edge = 0.5 * (float(a_sq[0]) + float(a_sq[-1]))
    diffs = np.diff(a_sq)
    total_var = float(np.sum(np.abs(diffs)))
    error_bound = edge + total_var
    return I, error_bound


# ---------------------------------------------------------------------------
# 8. Exact off-diagonal computation
# ---------------------------------------------------------------------------

def compute_exact_off_diagonal(
    xi: float,
    a: np.ndarray,
    logn: np.ndarray,
) -> float:
    """
    Exact off-diagonal magnitude at a given ξ.

    S(ξ) = ∑ a_n e^{2π i ξ log n}.

    |S(ξ)|^2 = ∑|a_n|^2 + ∑_{n != m} a_n \bar{a_m} e^{2π i ξ (log n - log m)}.

    We return |off-diagonal| = | |S(ξ)|^2 - ∑|a_n|^2 |.
    """
    xi = float(xi)
    N = len(a)
    if N == 0:
        return 0.0

    phase = np.exp(2j * math.pi * xi * logn)
    S_val = np.dot(a, phase)
    total = abs(S_val) ** 2
    diag = float(np.sum(np.abs(a) ** 2))
    off = total - diag
    return float(abs(off))


# ---------------------------------------------------------------------------
# 9. Large sieve validation / diagnostics
# ---------------------------------------------------------------------------

@dataclass
class BoundComparison:
    xi: float
    off_diag_exact: float
    MV_bound: float
    kernel_bound: float
    ratio_off_to_MV: float
    ratio_off_to_kernel: float


def validate_large_sieve_bounds(
    cfg: DirichletConfig,
    H: float,
    xi_values: Iterable[float],
) -> Tuple[LargeSieveConstants, List[BoundComparison]]:
    """
    Main Volume VI diagnostic.

    Steps:
      1. Build a_n and log n from cfg (with window).
      2. Compute MV bound (δ, MV_constant, MV_bound).
      3. Compute kernel off-diagonal bound.
      4. Compute discrete-to-continuous error bound.
      5. For each ξ, compare exact off-diagonal vs both bounds.
    """
    raw_a, logn = build_coefficients(cfg)
    a = apply_window(cfg, raw_a)
    gamma = logn

    δ, MV_constant, MV_bound = montgomery_vaughan_bound(gamma, a)
    sum_abs_sq = float(np.sum(np.abs(a) ** 2))

    kernel_const, kernel_bound = kernel_decay_off_diagonal_bound(a, logn, H)

    _, error_bound = discrete_to_continuous_quadrature(a)

    constants = LargeSieveConstants(
        N=len(a),
        min_separation=δ,
        sum_abs_sq=sum_abs_sq,
        MV_constant=MV_constant,
        MV_bound=MV_bound,
        kernel_bound_constant=kernel_const,
        kernel_bound=kernel_bound,
        discrete_to_cont_error=error_bound,
    )

    comparisons: List[BoundComparison] = []
    for xi in xi_values:
        off = compute_exact_off_diagonal(xi, a, logn)
        ratio_off_MV = off / (MV_bound + 1e-30)
        ratio_off_kernel = off / (kernel_bound + 1e-30)
        comparisons.append(
            BoundComparison(
                xi=float(xi),
                off_diag_exact=off,
                MV_bound=MV_bound,
                kernel_bound=kernel_bound,
                ratio_off_to_MV=ratio_off_MV,
                ratio_off_to_kernel=ratio_off_kernel,
            )
        )

    return constants, comparisons


# ---------------------------------------------------------------------------
# 10. Scaling study (N, σ, windows)
# ---------------------------------------------------------------------------

@dataclass
class ScalingRecord:
    N: int
    sigma: float
    window_type: str
    min_separation: float
    MV_constant: float
    MV_bound: float
    kernel_bound: float
    discrete_to_cont_error: float


def scaling_study(
    Ns: List[int],
    sigma: float,
    window_type: str,
    window_params: Dict[str, float] | None,
    H: float,
) -> List[ScalingRecord]:
    """
    Run a scaling study over N for fixed σ and window, collecting
    the key large-sieve constants.
    """
    records: List[ScalingRecord] = []
    for N in Ns:
        cfg = DirichletConfig(
            N=N,
            sigma=sigma,
            weight_type="plain",
            window_type=window_type,
            window_params=window_params,
        )
        raw_a, logn = build_coefficients(cfg)
        a = apply_window(cfg, raw_a)
        δ, MV_const, MV_bound = montgomery_vaughan_bound(logn, a)
        kernel_const, kernel_bound = kernel_decay_off_diagonal_bound(a, logn, H)
        _, err = discrete_to_continuous_quadrature(a)

        records.append(
            ScalingRecord(
                N=N,
                sigma=sigma,
                window_type=window_type,
                min_separation=δ,
                MV_constant=MV_const,
                MV_bound=MV_bound,
                kernel_bound=kernel_bound,
                discrete_to_cont_error=err,
            )
        )
    return records


# ---------------------------------------------------------------------------
# 11. Demo driver
# ---------------------------------------------------------------------------

def run_volume_vi_demo() -> None:
    """
    Simple CLI demo for manual inspection of Volume VI behaviour.
    """
    print("=== VOLUME VI: Large Sieve Bridge Demo ===")

    H = 1.0
    sigma = 0.5
    cfg = DirichletConfig(
        N=100,
        sigma=sigma,
        weight_type="plain",
        window_type="sharp",
        window_params=None,
    )

    xi_values = np.linspace(-2.0, 2.0, 9)
    constants, comps = validate_large_sieve_bounds(cfg, H=H, xi_values=xi_values)

    print("\nLarge Sieve Constants:")
    print(f"N = {constants.N}")
    print(f"min separation δ ≈ {constants.min_separation:.6e}")
    print(f"∑|a_n|^2 = {constants.sum_abs_sq:.6e}")
    print(f"MV constant N + 1/δ = {constants.MV_constant:.6e}")
    print(f"MV bound = {constants.MV_bound:.6e}")
    print(f"Kernel bound constant = {constants.kernel_bound_constant:.6e}")
    print(f"Kernel off-diagonal bound = {constants.kernel_bound:.6e}")
    print(f"Discrete→continuous error bound ≈ {constants.discrete_to_cont_error:.6e}")

    print("\nBound comparison at sample ξ:")
    for c in comps:
        print(
            f"ξ={c.xi:+.2f}, |off|={c.off_diag_exact:.3e}, "
            f"|off|/MV={c.ratio_off_to_MV:.3e}, "
            f"|off|/kernel={c.ratio_off_to_kernel:.3e}"
        )

    print("\nScaling study (plain, sharp window):")
    Ns = [50, 100, 200, 400]
    records = scaling_study(Ns, sigma=sigma, window_type="sharp",
                            window_params=None, H=H)
    for r in records:
        print(
            f"N={r.N:4d}, δ≈{r.min_separation:.3e}, "
            f"MV_const={r.MV_constant:.3e}, "
            f"MV_bound={r.MV_bound:.3e}, "
            f"kernel_bound={r.kernel_bound:.3e}, "
            f"disc→cont_err={r.discrete_to_cont_error:.3e}"
        )


if __name__ == "__main__":
    run_volume_vi_demo()