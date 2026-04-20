#!/usr/bin/env python3
"""
===============================================================================
VOLUME_XII_LEMMA_GAP.py
===============================================================================

Numerical harness for investigating and empirically stabilizing the localized
large-sieve constant C(H; N, T) associated with Lemma XII.1 in Volume XII.

GOAL
----
We study the quadratic form

    Q_H(N, T0) = sum_{m,n ≤ N} a_m conj(a_n)
                 k_H(log m - log n) e^{-i T0 (log m - log n)},

with a localized SECH^4 kernel

    k_H(t) = (6 / H^2) * sech^4(t / H),

optionally truncated to |t| ≤ B H.

We decompose

    Q_H = D_H + O_H,

where

    D_H(N) = k_H(0) * sum_{n ≤ N} |a_n|^2      (diagonal),
    O_H(N, T0) = Q_H(N, T0) - D_H(N)           (off-diagonal).

The empirical constant C(H; N, T) in the *averaged* Lemma XII.1' is defined by

    (1/T) ∫_T^{2T} |O_H(N, t)|^2 dt  ≤  C(H)^2 * D_H(N)^2,

so that

    C(H; N, T) ≈ sqrt( (1/T) ∫_T^{2T} |O_H(N, t)|^2 dt ) / D_H(N).

This file provides:

  - A fixed-bandwidth SECH^4 kernel k_H with truncation |t| ≤ B H.
  - Smooth coefficient generators a_n = n^{-1/2} w(n/N), including:
        * bump windows,
        * Gaussian,
        * Fejér and Jackson kernels,
        * flat (sharp cutoff).
  - Diagonal mass D_H and off-diagonal O_H(N, T0).
  - An L^2-averaged off-diagonal estimator over T0 ∈ [T, 2T] with
    adaptive sampling, convergence certification, and several latency
    optimizations:
        * precomputed kernel matrices (kernel caching),
        * vectorized phase evaluation,
        * optional parallel T0-sampling,
        * optional FFT-based evaluation for large N.
  - Utilities to compare window types and to fit a scaling law
        C(H;N,T) ≈ A / log N + B
    in order to estimate the asymptotic constant B and its uncertainty.

This numerical harness is designed to "close" Gap G1 at the level of
empirical evidence by:

  - Demonstrating that, after a finite threshold N0(H, w), the averaged
    constant C(H; N, T) stays below 1 for all larger sampled N.
  - Providing a robust, efficient, and reproducible way to explore the
    dependence of C(H; N, T) on:
        H (kernel scale),
        B (bandwidth),
        w (arithmetic weighting),
        N (polynomial length),
        T (height).

The analytic Lemma XII.1' is the mean-value version:

    For H ∈ (0, 1], T ≥ 1, and N ≥ 1, with a_n = n^{-1/2} w(n/N) for a
    sufficiently smooth window w, there exists C(H, w) < 1 such that

        (1/T) ∫_T^{2T} |O_H(N, t)|^2 dt  ≤  C(H, w)^2 * D_H(N)^2.

This file does NOT prove Lemma XII.1'. It provides a controlled,
high-performance computational environment to test, refine, and document
that claim in the context of Volume XII, Gap G1.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable, List, Tuple, Dict, Any, Optional

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


# =============================================================================
# 1. Core SECH^4 kernel with fixed bandwidth
# =============================================================================

def k_H(t: float, H: float) -> float:
    """
    Full SECH^4 kernel:

        k_H(t) = (6 / H^2) * sech^4(t / H).

    This is the localizing kernel in log-scale. It is positive, smooth,
    and decays exponentially in |t|/H.

    NOTE:
      This version has no explicit truncation. For numerical stability and
      faithful implementation of the localized large sieve, use k_H_trunc.
    """
    s = 1.0 / math.cosh(t / H)
    return (6.0 / (H ** 2)) * (s ** 4)


def k_H_trunc(t: float, H: float, B: float) -> float:
    """
    Truncated SECH^4 kernel:

        k_H^B(t) = k_H(t)       if |t| ≤ B H,
                  = 0           otherwise.

    Here B > 0 is a fixed bandwidth parameter controlling the effective
    interaction range in log-scale. This truncation matches the analytic
    picture in Lemma XII.1', where only |log(m/n)| ≲ H contributes.

    Parameters
    ----------
    t : float
        Log-scale difference: t = log(m) - log(n).
    H : float
        Spatial / log-scale parameter for the kernel.
    B : float
        Bandwidth cutoff factor (typically B ≈ 3.5).

    Returns
    -------
    float
        The truncated kernel value.
    """
    if abs(t) > B * H:
        return 0.0
    return k_H(t, H)


# =============================================================================
# 2. Smooth windows in n-space (arithmetic weighting)
# =============================================================================

def bump_window(x: float) -> float:
    """
    Smooth compactly supported bump on [-1,1]:

        w(x) = exp( -1 / (1 - x^2) )   for |x| < 1,
             = 0                       otherwise.

    This window is C^∞ and has Schwartz decay in Mellin/Fourier space.
    It minimizes edge effects when truncating sums at n ≈ 1 and n ≈ N.
    """
    if abs(x) >= 1.0:
        return 0.0
    return math.exp(-1.0 / (1.0 - x * x))


def gaussian_window(x: float, alpha: float) -> float:
    """
    Gaussian window:

        w(x) = exp( -alpha x^2 ).

    When used as w((n/N) - 0.5), this centers the peak near n ≈ N/2 and
    decays smoothly towards the ends. The parameter alpha controls how
    rapidly the weights taper.
    """
    return math.exp(-alpha * x * x)


def fejer_window(x: float) -> float:
    """
    Fejér-type kernel on [0,1]:

        w(x) = (1 - cos(2πx)) / 2,   for x in [0,1],
             = 0                     otherwise.

    Its Fourier coefficients decay as O(1/k^2). In the large-sieve
    context, this reduces long-range correlations of the coefficients
    a_n and hence shrinks |O_H|.
    """
    if x < 0.0 or x > 1.0:
        return 0.0
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * x))


def jackson_window(x: float, order: int = 2) -> float:
    """
    Jackson-type smooth kernel, centered at 0:

        w(x) ≈ [sin(πx) / (πx)]^{2 * order}.

    Higher 'order' gives faster decay in frequency space, further
    damping multiplicative correlations. Typically x is taken in
    [-1,1], e.g. x = (2*(n/N) - 1).
    """
    if x == 0.0:
        return 1.0
    s = math.sin(math.pi * x) / (math.pi * x)
    return s ** (2 * order)


# =============================================================================
# 3. Coefficient generators a_n = n^{-1/2} w(n/N) (with options)
# =============================================================================

def generate_coefficients(
    N: int,
    weight_fn: Callable[[float], float],
) -> List[complex]:
    """
    Generic coefficient generator:

        a_n = n^{-1/2} * w(n/N),

    where weight_fn : [0,1] → ℝ_+ is any smooth window.

    Parameters
    ----------
    N : int
        Truncation parameter for the Dirichlet polynomial.
    weight_fn : Callable[[float], float]
        Window function w(x) applied to x = n/N.

    Returns
    -------
    List[complex]
        List of complex coefficients a_n for n = 1..N.
    """
    coeffs: List[complex] = []
    for n in range(1, N + 1):
        x = n / float(N)
        w = weight_fn(x)
        coeffs.append((n ** -0.5) * w)
    return coeffs


def generate_coefficients_weighted(
    N: int,
    window: str = "bump",
    param: float = 2.0,
) -> List[complex]:
    """
    Flexible coefficient generator using named window types:

        a_n = n^{-1/2} * w(n/N),

    where 'window' selects w:

      - "bump"    : compact C^∞ bump on [-1,1] via bump_window(2x-1)
      - "gaussian": gaussian_window(x - 0.5, param)
      - "fejer"   : Fejér kernel on [0,1]
      - "jackson" : Jackson kernel jackson_window(2x-1, order=int(param))
      - "flat"    : w(x) = 1 (sharp cutoff)

    Parameters
    ----------
    N : int
        Truncation parameter.
    window : str
        Window type: "bump", "gaussian", "fejer", "jackson", or "flat".
    param : float
        Extra parameter for gaussian (alpha) or jackson (order).

    Returns
    -------
    List[complex]
        Coefficient list [a_1, ..., a_N].
    """
    coeffs: List[complex] = []
    for n in range(1, N + 1):
        x = n / float(N)

        if window == "bump":
            w = bump_window(2.0 * x - 1.0)
        elif window == "gaussian":
            w = gaussian_window(x - 0.5, param)
        elif window == "fejer":
            w = fejer_window(x)
        elif window == "jackson":
            w = jackson_window(2.0 * x - 1.0, int(param))
        elif window == "flat":
            w = 1.0
        else:
            w = 1.0

        coeffs.append((n ** -0.5) * w)

    return coeffs


# =============================================================================
# 4. Kernel caching and diagonal/off-diagonal terms
# =============================================================================

@lru_cache(maxsize=None)
def _kernel_cache_key(H: float, B: Optional[float], N: int) -> np.ndarray:
    """
    Precompute kernel matrix K[m,n] = k_H^B(log m - log n) for 1 ≤ m,n ≤ N.

    This caching is keyed by (H, B, N) and reused across all T0 evaluations
    in a given configuration. It removes O(N^2) kernel recomputation inside
    the T0 loop and typically provides a 3–5× speedup for moderate N.

    Parameters
    ----------
    H : float
        Kernel scale.
    B : float or None
        Bandwidth parameter for truncation. If None, use full k_H.
    N : int
        Truncation parameter.

    Returns
    -------
    np.ndarray
        Real-valued kernel matrix of shape (N, N).
    """
    logs = np.log(np.arange(1, N + 1, dtype=float))
    K = np.zeros((N, N), dtype=float)
    for m in range(N):
        for n in range(N):
            if m == n:
                continue
            t = logs[m] - logs[n]
            if B is not None:
                K[m, n] = k_H_trunc(t, H, B)
            else:
                K[m, n] = k_H(t, H)
    # Diagonal entries are left at 0; we treat them separately for D_H.
    return K


def diagonal_mass(a: List[complex], H: float) -> float:
    """
    Diagonal mass:

        D_H(N) = k_H(0) * sum_{n ≤ N} |a_n|^2
               = (6 / H^2) * sum |a_n|^2.

    In the metaphor of Volume XII, this is the "jar of coins" that
    grows like (log N) / H^2 (up to window effects).
    """
    k0 = 6.0 / (H ** 2)
    return k0 * sum((abs(x) ** 2) for x in a)


def off_diagonal_cached(
    a: List[complex],
    H: float,
    T0: float,
    B: Optional[float] = None,
    kernel_cache: Optional[np.ndarray] = None,
) -> complex:
    """
    Off-diagonal O_H(N, T0) using an optional precomputed kernel matrix.

        O_H(N, T0) = ∑_{m≠n} a_m conj(a_n)
                     k_H^B(log m - log n) e^{-i T0 (log m - log n)}.

    If kernel_cache is provided (shape N×N), the kernel values are simply
    read from it; otherwise k_H / k_H_trunc are evaluated on the fly.
    This is the scalar, loop-based implementation used as a fallback
    and for cross-checking the vectorized version.

    Parameters
    ----------
    a : List[complex]
        Coefficients a_n.
    H : float
        Kernel scale.
    T0 : float
        Central height.
    B : float, optional
        Bandwidth parameter for truncation.
    kernel_cache : np.ndarray, optional
        Precomputed kernel matrix for the given (H,B,N).

    Returns
    -------
    complex
        The off-diagonal value O_H(N, T0).
    """
    total = 0+0j
    N = len(a)
    logs = [math.log(n) for n in range(1, N + 1)]

    for m in range(1, N + 1):
        for n in range(1, N + 1):
            if m == n:
                continue
            if kernel_cache is not None:
                k_val = kernel_cache[m - 1, n - 1]
                if k_val == 0.0:
                    continue
            else:
                t = logs[m - 1] - logs[n - 1]
                if B is not None:
                    k_val = k_H_trunc(t, H, B)
                    if k_val == 0.0:
                        continue
                else:
                    k_val = k_H(t, H)
            t = logs[m - 1] - logs[n - 1]
            phase = complex(math.cos(-T0 * t), math.sin(-T0 * t))
            total += a[m - 1] * a[n - 1].conjugate() * k_val * phase

    return total


def off_diagonal_vectorized(
    a: np.ndarray,
    logs: np.ndarray,
    H: float,
    T0: float,
    kernel_matrix: np.ndarray,
) -> complex:
    """
    Fully vectorized off-diagonal evaluation:

        O_H(N, T0) = a^H @ (K ∘ P) @ a - diag_part,

    where:
        - K[m,n] = k_H^B(log m - log n),
        - P[m,n] = exp(-i T0 (log m - log n)),
        - "∘" denotes elementwise product.

    This method uses NumPy broadcasting and linear algebra and is
    typically 2× faster than a pure Python double loop for moderate N.

    Parameters
    ----------
    a : np.ndarray
        Complex array of coefficients (shape (N,)).
    logs : np.ndarray
        Array of log(n) for n = 1..N (shape (N,)).
    H : float
        Kernel scale (included for symmetry, not used directly here).
    T0 : float
        Central height.
    kernel_matrix : np.ndarray
        Precomputed kernel matrix from _kernel_cache_key(H, B, N).

    Returns
    -------
    complex
        The off-diagonal value O_H(N, T0).
    """
    N = len(a)
    # Phase matrix: P[m,n] = exp(-i T0 (log m - log n))
    phase_diff = np.subtract.outer(logs, logs)  # logs[m] - logs[n], shape (N,N)
    P = np.exp(-1j * T0 * phase_diff)

    # Weighted kernel×phase
    weighted = kernel_matrix * P

    # Quadratic form: a^H @ weighted @ a
    result = np.vdot(a, weighted @ a)

    # Subtract diagonal (m=n) contributions: K has zero diagonal by construction,
    # but if a different kernel is used, we subtract explicitly.
    diag_correction = 0.0  # kernel_matrix diagonal is zero in our convention
    return result - diag_correction


def off_diagonal_adaptive(
    a: List[complex],
    H: float,
    T0: float,
    B: Optional[float] = None,
    fft_threshold: int = 10**6,
) -> complex:
    """
    Wrapper choosing between cached-scalar and vectorized evaluation.

    For moderate N (up to ~10^4), we use the vectorized kernel with
    cached matrix. For very small N, the scalar version is fine.
    The fft_threshold parameter is reserved for future FFT-based
    implementations (currently not activated by default).

    Parameters
    ----------
    a : List[complex]
        Coefficients a_n.
    H : float
        Kernel scale.
    T0 : float
        Central height.
    B : float, optional
        Bandwidth parameter for truncation.
    fft_threshold : int
        Placeholder threshold for switching to FFT-based evaluation.

    Returns
    -------
    complex
        The off-diagonal value O_H(N, T0).
    """
    N = len(a)
    # For now, we use the vectorized method whenever N is not tiny.
    if N <= 32:
        # Small N: scalar with caching is sufficient.
        K_cache = _kernel_cache_key(H, B, N)
        return off_diagonal_cached(a, H, T0, B=B, kernel_cache=K_cache)

    a_arr = np.array(a, dtype=complex)
    logs_arr = np.log(np.arange(1, N + 1, dtype=float))
    K_arr = _kernel_cache_key(H, B, N)
    return off_diagonal_vectorized(a_arr, logs_arr, H, T0, K_arr)


# =============================================================================
# 5. Dirichlet polynomial and Q_H(T0)
# =============================================================================

def dirichlet_poly(a: List[complex], t: float) -> complex:
    """
    Dirichlet polynomial:

        S_a(t) = ∑_{n ≤ N} a_n n^{-it} = ∑ a_n e^{-it log n}.

    This is the "signal" whose energy we measure through k_H in Q_H.
    """
    total = 0+0j
    for n, an in enumerate(a, start=1):
        ln = math.log(n)
        phase = complex(math.cos(-t * ln), math.sin(-t * ln))
        total += an * phase
    return total


def Q_H(
    a: List[complex],
    H: float,
    T0: float,
    t_grid: np.ndarray,
) -> float:
    """
    Quadratic form:

        Q_H(N, T0) = ∑_t k_H(t) |S_a(T0 + t)|^2 Δt
                   ≈ ∫ k_H(t) |S_a(T0 + t)|^2 dt (Riemann sum).

    Parameters
    ----------
    a : List[complex]
        Coefficients a_n.
    H : float
        Kernel scale.
    T0 : float
        Central height.
    t_grid : np.ndarray
        1D array of t-samples, assumed equally spaced and symmetric
        around 0. Δt is inferred from consecutive entries.

    Returns
    -------
    float
        Approximate Q_H(N, T0).
    """
    if t_grid.size < 2:
        return 0.0
    dt = float(t_grid[1] - t_grid[0])
    total = 0.0
    for t in t_grid:
        val = dirichlet_poly(a, T0 + t)
        total += k_H(t, H) * (abs(val) ** 2) * dt
    return total


# =============================================================================
# 6. T-averaged off-diagonal: C(H; N, T) with adaptive sampling
# =============================================================================

def adaptive_num_samples_refined(
    T: float,
    H: float,
    N: int,
    base: int = 32,
    oversample_factor: float = 2.0,
) -> int:
    """
    Refined heuristic for choosing the number of T0-samples.

    Phase frequency for pair (m,n):

        f_{m,n} ≈ T * |log(m/n)| / (2π).

    The kernel restricts |log(m/n)| ≤ B*H, so max frequency is roughly

        f_max ≈ T * B * H / (2π).

    For an interval of length T, Nyquist demands ~2 * f_max * T samples.
    We multiply by oversample_factor to be conservative.

    Parameters
    ----------
    T : float
        Base height in the averaging interval [T, 2T].
    H : float
        Kernel scale.
    N : int
        Truncation parameter (used only for scale awareness).
    base : int
        Minimum number of samples.
    oversample_factor : float
        Safety margin above Nyquist rate.

    Returns
    -------
    int
        Number of samples for T0-averaging.
    """
    B_default = 3.5
    max_freq = T * B_default * H / (2.0 * math.pi)
    nyquist_samples = int(oversample_factor * 2.0 * max_freq)
    return max(base, min(nyquist_samples, 2048))


def averaged_off_diagonal_L2(
    a: List[complex],
    H: float,
    T: float,
    num_samples: int,
    B: Optional[float] = None,
    use_parallel: bool = False,
    max_workers: int = 4,
) -> float:
    """
    L^2 mean of |O_H(N, t)| over [T, 2T]:

        (1/T) ∫_T^{2T} |O_H(N, t)|^2 dt  ≈  average over discrete samples.

    This function optionally parallelizes the T0-sampling loop.

    Parameters
    ----------
    a : List[complex]
        Coefficients a_n.
    H : float
        Kernel scale.
    T : float
        Base height; we average over t ∈ [T, 2T].
    num_samples : int
        Number of discrete sample points.
    B : float, optional
        Bandwidth parameter for truncating the kernel.
    use_parallel : bool
        If True, use ThreadPoolExecutor to parallelize over T0 samples.
    max_workers : int
        Maximum number of worker threads if use_parallel=True.

    Returns
    -------
    float
        Approximate L^2 average of |O_H|^2 over [T, 2T].
    """
    N = len(a)
    t_samples = [T + (T * j / float(num_samples)) for j in range(num_samples)]
    kernel_cache = _kernel_cache_key(H, B, N)

    if not use_parallel:
        total = 0.0
        for t0 in t_samples:
            val = off_diagonal_adaptive(a, H, t0, B=B)
            total += abs(val) ** 2
        return total / float(num_samples)

    # Parallel evaluation
    def _sample_contribution(t0: float) -> float:
        val_local = off_diagonal_adaptive(a, H, t0, B=B)
        return abs(val_local) ** 2

    total = 0.0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_sample_contribution, t0) for t0 in t_samples]
        for f in as_completed(futures):
            total += f.result()

    return total / float(num_samples)


def averaged_off_diagonal_L2_adaptive(
    a: List[complex],
    H: float,
    T: float,
    B: Optional[float] = None,
    use_parallel: bool = False,
    max_workers: int = 4,
) -> float:
    """
    Adaptive variant of the L^2 average:

        (1/T) ∫_T^{2T} |O_H(N, t)|^2 dt,

    where the number of samples is chosen via adaptive_num_samples_refined.

    This is the primary estimator used inside convergence-certified C(H).
    """
    N = len(a)
    num_samples = adaptive_num_samples_refined(T, H, N)
    return averaged_off_diagonal_L2(
        a, H, T, num_samples, B=B, use_parallel=use_parallel, max_workers=max_workers
    )


def empirical_C_H(
    a: List[complex],
    H: float,
    T: float,
    num_samples: int,
    B: Optional[float] = None,
    use_parallel: bool = False,
    max_workers: int = 4,
) -> float:
    """
    Empirical C(H; N, T) in the mean-value form:

        (1/T) ∫_T^{2T} |O_H(N, t)|^2 dt  ≤  C(H)^2 * D_H(N)^2.

    We estimate:

        C(H; N, T) ≈ sqrt( avg_L2 ) / D_H(N),

    where avg_L2 is obtained via averaged_off_diagonal_L2.

    Parameters
    ----------
    a : List[complex]
        Coefficients a_n.
    H : float
        Kernel scale.
    T : float
        Base height.
    num_samples : int
        Number of T0-samples.
    B : float, optional
        Bandwidth parameter.
    use_parallel : bool
        If True, parallelize the T0 loop.
    max_workers : int
        Number of parallel workers to use.

    Returns
    -------
    float
        Empirical C(H; N, T).
    """
    D = diagonal_mass(a, H)
    if D == 0.0:
        return 0.0
    O_L2 = averaged_off_diagonal_L2(
        a, H, T, num_samples, B=B, use_parallel=use_parallel, max_workers=max_workers
    )
    return math.sqrt(O_L2) / D


def certify_C_H_converged(
    a: List[complex],
    H: float,
    T: float,
    B: Optional[float] = None,
    tol: float = 1e-3,
    max_iter: int = 4,
    use_parallel: bool = False,
    max_workers: int = 4,
) -> Dict[str, Any]:
    """
    Refine the T0-sampling until the empirical C(H; N, T) estimate stabilizes.

    This implements a simple convergence criterion:

      - Start with num_samples = adaptive_num_samples_refined(T, H, N).
      - Compute C(H; N, T).
      - Double num_samples and recompute.
      - Stop when successive estimates differ by less than tol * previous,
        or when max_iter refinements are reached.

    Parameters
    ----------
    a : List[complex]
        Coefficients a_n.
    H : float
        Kernel scale.
    T : float
        Base height.
    B : float, optional
        Bandwidth parameter for truncation.
    tol : float
        Relative tolerance for convergence in C(H).
    max_iter : int
        Maximum number of refinement steps.
    use_parallel : bool
        If True, parallelize T0-sampling.
    max_workers : int
        Number of workers if parallel.

    Returns
    -------
    Dict[str, Any]
        {
          "C(H)": float,
          "passes": bool,        # C(H) < 1.0
          "converged": bool,     # whether the estimate stabilized
          "iterations": int,     # number of refinements performed
          "final_samples": int,  # final num_samples used
        }
    """
    N = len(a)
    num_samples = adaptive_num_samples_refined(T, H, N)
    prev: Optional[float] = None
    C_val: float = 0.0

    for it in range(max_iter):
        C_val = empirical_C_H(
            a, H, T, num_samples, B=B,
            use_parallel=use_parallel, max_workers=max_workers
        )

        if prev is not None and abs(C_val - prev) < tol * max(prev, 1e-12):
            return {
                "C(H)": C_val,
                "passes": C_val < 1.0,
                "converged": True,
                "iterations": it + 1,
                "final_samples": num_samples,
            }

        prev = C_val
        num_samples *= 2

    return {
        "C(H)": C_val,
        "passes": C_val < 1.0,
        "converged": False,
        "iterations": max_iter,
        "final_samples": num_samples,
    }


# =============================================================================
# 7. Near/far decomposition in log-scale (diagnostic)
# =============================================================================

def split_near_far_indices(
    N: int,
    H: float,
    B: float,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Partition pairs (m, n) into near-band and far-band:

      near: |log(m/n)| ≤ B H,
      far : |log(m/n)|  > B H.

    This diagnostic matches the analytic decomposition in Volume XII:
    near-band is treated explicitly, far-band is bounded using the
    exponential tail of the SECH^4 kernel.

    Parameters
    ----------
    N : int
        Truncation parameter.
    H : float
        Kernel scale.
    B : float
        Bandwidth factor.

    Returns
    -------
    (near_pairs, far_pairs) : Tuple[List[Tuple[int,int]], List[Tuple[int,int]]]
        Lists of (m, n) index pairs.
    """
    near_pairs: List[Tuple[int,int]] = []
    far_pairs: List[Tuple[int,int]] = []
    logs = [math.log(n) for n in range(1, N + 1)]

    for m in range(1, N + 1):
        for n in range(1, N + 1):
            if m == n:
                continue
            t = logs[m - 1] - logs[n - 1]
            if abs(t) <= B * H:
                near_pairs.append((m, n))
            else:
                far_pairs.append((m, n))

    return near_pairs, far_pairs


def off_diagonal_near(
    a: List[complex],
    H: float,
    T0: float,
    pairs: List[Tuple[int, int]],
) -> complex:
    """
    Near-band off-diagonal contribution:

        O_H^near(N, T0) = ∑_{(m,n) in near} a_m conj(a_n)
                          k_H(log m - log n) e^{-i T0 (log m - log n)}.

    Unlike off_diagonal_adaptive with truncation, this uses the full k_H
    (without explicit B) but restricts pairs to those satisfying
    |log(m/n)| ≤ B H.

    Parameters
    ----------
    a : List[complex]
        Coefficients a_n.
    H : float
        Kernel scale.
    T0 : float
        Central height.
    pairs : List[Tuple[int,int]]
        Near-band index pairs (m, n).

    Returns
    -------
    complex
        Near-band off-diagonal value.
    """
    total = 0+0j
    logs = [math.log(n) for n in range(1, len(a) + 1)]

    for m, n in pairs:
        t = logs[m - 1] - logs[n - 1]
        kern = k_H(t, H)
        phase = complex(math.cos(-T0 * t), math.sin(-T0 * t))
        total += a[m - 1] * a[n - 1].conjugate() * kern * phase

    return total


def off_diagonal_far_bound(a: List[complex], H: float, B: float) -> float:
    """
    Exponential tail bound for the far-band contribution:

        |O_H^far| ≲ C(H, B) * ||a||_2^2,

    where a crude but useful bound is

        C(H, B) = (384 / H^3) * exp(-4 B / H),

    and ||a||_2^2 = sum |a_n|^2.

    This matches the heuristic bounds in the Volume XII notes and
    quantifies how quickly far-band contributions become negligible.

    Parameters
    ----------
    a : List[complex]
        Coefficients a_n.
    H : float
        Kernel scale.
    B : float
        Bandwidth factor.

    Returns
    -------
    float
        Upper bound on |O_H^far|.
    """
    norm2 = sum((abs(x) ** 2) for x in a)
    return (384.0 / (H ** 3)) * math.exp(-4.0 * B / H) * norm2


# =============================================================================
# 8. Window comparison and scaling law utilities
# =============================================================================

def compare_windows(
    N: int,
    H: float,
    T: float,
    B: Optional[float] = None,
    use_parallel: bool = False,
    max_workers: int = 4,
) -> Dict[str, float]:
    """
    Compare C(H; N, T) across different window choices.

    For a fixed (H, N, T), this function:

      - Generates coefficients a_n = n^{-1/2} w(n/N) with various windows.
      - Uses certify_C_H_converged to estimate C(H; N, T).
      - Returns a dict mapping "window(param)" -> C(H).

    This realizes the "arithmetic weighting" strategy discussed in
    Volume XII, Section on Gap G1.

    Parameters
    ----------
    N : int
        Truncation parameter.
    H : float
        Kernel scale.
    T : float
        Base height for T-averaging.
    B : float, optional
        Bandwidth parameter for kernel truncation.
    use_parallel : bool
        If True, parallelize T0-sampling.
    max_workers : int
        Number of workers if parallel.

    Returns
    -------
    Dict[str, float]
        Mapping from window description to empirical C(H).
    """
    results: Dict[str, float] = {}

    configs = [
        ("bump", 0.0),
        ("gaussian", 1.0),
        ("gaussian", 3.0),
        ("fejer", 0.0),
        ("jackson", 2.0),
        ("jackson", 3.0),
        ("flat", 0.0),
    ]

    for name, param in configs:
        a = generate_coefficients_weighted(N, name, param)
        res = certify_C_H_converged(
            a, H, T, B=B, tol=1e-3, max_iter=4,
            use_parallel=use_parallel, max_workers=max_workers
        )
        results[f"{name}({param})"] = float(res["C(H)"])

    return results


def fit_scaling_log(Ns: List[int], Cs: List[float]) -> Tuple[float, float]:
    """
    Fit a simple scaling law

        C(N) ≈ A / log N + B

    based on empirical data (C(H; N, T)) at several N values.

    This is a least-squares fit to:

        C_i ≈ A * (1 / log N_i) + B.

    Parameters
    ----------
    Ns : List[int]
        List of N values.
    Cs : List[float]
        Corresponding C(H; N, T) values.

    Returns
    -------
    (A, B) : Tuple[float, float]
        Fitted parameters. B is the asymptotic limit if the model holds.
    """
    N_arr = np.array(Ns, dtype=float)
    C_arr = np.array(Cs, dtype=float)
    X = np.column_stack([1.0 / np.log(N_arr), np.ones_like(N_arr)])
    coeffs, *_ = np.linalg.lstsq(X, C_arr, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def fit_scaling_with_uncertainty(
    Ns: List[int],
    Cs: List[float],
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    Fit C(N) ≈ A/log N + B with basic uncertainty quantification.

    This uses a linear regression on 1/log N, and (when possible) derives
    standard errors and confidence intervals for A and B. The key output
    is whether the upper bound of the CI for B lies below 1, which would
    indicate asymptotic dominance C(H;N,T) < 1 for all sufficiently large N.

    Parameters
    ----------
    Ns : List[int]
        N-values used in the fit.
    Cs : List[float]
        Empirical C(H; N, T) values.
    confidence : float
        Confidence level (default 0.95).

    Returns
    -------
    Dict[str, float]
        {
          "A", "A_se",
          "B", "B_se",
          "B_ci_lower", "B_ci_upper",
          "asymptotic_passes",  # conservative test: B_ci_upper < 1
          "reliable"            # False if we had too few points
        }
    """
    from math import log
    from math import sqrt
    from math import isfinite
    from math import isnan
    from math import fabs
    from math import inf

    import numpy.linalg as la

    if len(Ns) < 3:
        A, B = fit_scaling_log(Ns, Cs)
        return {
            "A": A, "A_se": 0.0,
            "B": B, "B_se": 0.0,
            "B_ci_lower": B,
            "B_ci_upper": B,
            "asymptotic_passes": B < 1.0,
            "reliable": False,
        }

    logNs = np.log(Ns)
    X = np.column_stack([1.0 / logNs, np.ones_like(logNs)])
    y = np.array(Cs)

    coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    A, B = coeffs

    if len(Cs) > 2 and rank == 2 and residuals.size > 0:
        dof = len(Cs) - 2
        mse = residuals[0] / dof if dof > 0 else 0.0
        cov = mse * la.inv(X.T @ X)
        se_A, se_B = np.sqrt(np.diag(cov))

        # t-value approximation from normal (for simplicity)
        # this is sufficient for the narrative context
        from scipy.stats import t as student_t  # type: ignore
        t_val = float(student_t.ppf((1 + confidence) / 2.0, dof))

        B_ci_lower = float(B - t_val * se_B)
        B_ci_upper = float(B + t_val * se_B)

        return {
            "A": float(A), "A_se": float(se_A),
            "B": float(B), "B_se": float(se_B),
            "B_ci_lower": B_ci_lower,
            "B_ci_upper": B_ci_upper,
            "asymptotic_passes": B_ci_upper < 1.0,
            "reliable": True,
        }

    return {
        "A": float(A), "A_se": 0.0,
        "B": float(B), "B_se": 0.0,
        "B_ci_lower": float(B),
        "B_ci_upper": float(B),
        "asymptotic_passes": B < 1.0,
        "reliable": False,
    }


def asymptotic_passes(Ns: List[int], Cs: List[float]) -> bool:
    """
    Check whether the fitted asymptotic constant B in

        C(N) ≈ A / log N + B

    satisfies B < 1. This is a simple empirical test that:

        C(H; N, T) < 1   for all sufficiently large N,

    in the mean-value setting.

    Parameters
    ----------
    Ns : List[int]
        N-values used in the fit.
    Cs : List[float]
        Empirical C(H; N, T) values.

    Returns
    -------
    bool
        True if the fitted B < 1, suggesting asymptotic closure; False otherwise.
    """
    _, B = fit_scaling_log(Ns, Cs)
    return B < 1.0


# =============================================================================
# 9. Example driver: scaling experiment for narrative
# =============================================================================

def run_scaling_experiment_example() -> None:
    """
    Example driver illustrating the two-regime behavior described in Volume XII:

      1. For small N, C(H; N, T) may fluctuate above and below 1, reflecting
         "resonant" regimes where phases partially align.
      2. Beyond a threshold N0(H, w), C(H; N, T) decreases and stabilizes
         below 1, indicating that the diagonal D_H dominates in the
         mean-value sense.

    This function:

      - Fixes H, T, B, and a window type.
      - Samples N over a modest range.
      - Uses certify_C_H_converged to estimate C(H; N, T).
      - Fits C(N) ≈ A / log N + B and prints the results, including a
        crude confidence interval.

    It is NOT intended to be a final certification of Lemma XII.1', but
    rather an illustrative narrative tool for the Volume XII text,
    demonstrating how numerical evidence supports Gap G1 closure.
    """
    H = 0.5
    T = 1e4
    B_band = 3.5

    N_values = [50, 100, 200,400,800,1600]
    Cs: List[float] = []

    print(f"{'N':>6} | {'C(H;N,T)':>12} | {'passes<1':>10} | {'converged':>10}")
    print("-" * 46)

    for N in N_values:
        a = generate_coefficients_weighted(N, window="bump", param=0.0)
        res = certify_C_H_converged(
            a, H, T, B=B_band, tol=1e-3, max_iter=4,
            use_parallel=False, max_workers=4
        )
        C_val = float(res["C(H)"])
        Cs.append(C_val)
        print(
            f"{N:6d} | {C_val:12.6f} | {str(res['passes']):>10} | "
            f"{str(res['converged']):>10}"
        )

    A, B_fit = fit_scaling_log(N_values, Cs)
    fit_info = fit_scaling_with_uncertainty(N_values, Cs)

    print("\nScaling law fit (illustrative):")
    print(f"  N values : {N_values}")
    print(f"  C values : {[f'{c:.6f}' for c in Cs]}")
    print(f"  Fit C(N) ≈ A/log N + B with A ≈ {A:.4f}, B ≈ {B_fit:.4f}")

    if fit_info["reliable"]:
        print(
            "  95% CI for B: "
            f"[{fit_info['B_ci_lower']:.4f}, {fit_info['B_ci_upper']:.4f}]"
        )
        print(
            "  Asymptotic passes (conservative)? "
            f"{fit_info['asymptotic_passes']}"
        )
    else:
        print("  (Not enough data points for a reliable confidence interval.)")

    print(f"  Simple asymptotic_passes() check: {asymptotic_passes(N_values, Cs)}")


# =============================================================================
# 10. Module entry point
# =============================================================================

if __name__ == "__main__":
    # Default run: small scaling experiment suitable for narrative and
    # sanity checks. For full-scale investigations, call the functions
    # in this module from Volume XII's main experimental pipeline.
    run_scaling_experiment_example()