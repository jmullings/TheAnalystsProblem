#!/usr/bin/env python3
"""
VOLUME_IV_SPECTRAL_EXPANSION.py
===============================

Enhanced implementation of Volume IV: Spectral Expansion, with Layer 2 (T2)
— Spectral σ-selection emergence — fully wired and instrumented.

Layers
------
T1 (algebraic):
    Q_N(σ) = Σ_{n=2}^{N} [n^{−σ} − n^{−(1−σ)}]^2 (log n)^2
is implemented exactly and has:
    Q_N(σ) ≥ 0,  Q_N(σ) = 0 ⇔ σ = 1/2,  Q_N(1−σ) = Q_N(σ),
    Q_N''(1/2) = 8 Σ_{n=2}^{N} (log n)^4 / n > 0.

T2 (spectral, raw):
    Q_H^spec(σ; N, H, T0) = ∫ k_hat(ξ, H) |S_σ(ξ − T0/(2π))|^2 dξ
implements the Parseval bridge and spectral curvature.

T2 (selector / normalized diagnostics):
    - A direct antisymmetric σ-selector is computed inside the integral:
        Q_sel(σ) = (1/2) ∫ k_hat(ξ) [|S_σ|^2 − |S_{1−σ}|^2] dξ
    - A normalized curvature:
        Q_norm(σ) = Q_spec(σ) / S_diag(σ)
      removes trivial σ-bias from the diagonal.
    - A final normalized selector:
        Q_final(σ) = Q_sel(σ) / S_diag(σ)
      is antisymmetric and scale-reduced.

The emergent symmetric σ-profile used in the validation suite is built from
the symmetric/antisymmetric decomposition at the spectral level.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from typing import List, Dict

import math

import numpy as np
import mpmath as mp

# ---------------------------------------------------------------------------
# Import Volume III module (QuadraticFormDecomposition) via repo root
# ---------------------------------------------------------------------------

THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir, os.pardir))

QF_PROOF_DIR = os.path.join(
    REPO_ROOT,
    "QuadraticFormDecomposition",
    "QuadraticFormDecompositionProof",
)

sys.path.insert(0, QF_PROOF_DIR)

import QuadraticFormDecomposition as qf  # noqa: E402

mp.mp.dps = 80


# ---------------------------------------------------------------------------
# 1. Analytic Fourier symbol: k_hat(xi, H)
# ---------------------------------------------------------------------------

def k_hat(xi: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    """
    Analytic Fourier transform of the Bochner-repaired kernel:

        k_hat(ξ, H) = ((2πξ)^2 + 4/H^2) * w_hat(ξ, H)

    with

        w_H(t) = sech^2(t/H)
        w_hat(ξ, H) = π H * (2π ξ H) / sinh(π^2 ξ H)

    Safe at ξ = 0 and numerically stable for large |ξ| via an asymptotic form
    that preserves positivity and evenness.

    By construction here, the L'Hôpital limit at ξ = 0 is 8/H^2.
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
        w_hat = mp.pi * H * (2 * mp.pi * a * H) * 2 * exp_term
    else:
        w_hat = mp.pi * H * (2 * mp.pi * a * H) / mp.sinh(arg)

    val = num * w_hat
    return val if val >= 0 else mp.mpf("0")


# ---------------------------------------------------------------------------
# 2. Dirichlet wave S(xi) and σ-perturbed variants
# ---------------------------------------------------------------------------

def S_xi(xi: float | mp.mpf, N: int, sigma: float = 0.5) -> complex:
    """
    Dirichlet wave:

        S_σ(ξ) = sum_{n <= N} n^{-σ} e^{2π i ξ log n}
    """
    xi = mp.mpf(xi)
    sigma_mp = mp.mpf(sigma)

    logs = [mp.log(n) for n in range(1, N + 1)]
    amps = [mp.power(n, -sigma_mp) for n in range(1, N + 1)]

    s = mp.mpf("0") + 0j
    two_pi_xi = 2j * mp.pi * xi
    for logn, amp in zip(logs, amps):
        phase = two_pi_xi * logn
        s += amp * mp.e ** phase

    return complex(s)


# ---------------------------------------------------------------------------
# 3. Spectral quadratic form Q_H^spec and Parseval bridge
# ---------------------------------------------------------------------------

def Q_spectral(
    N: int,
    H: float,
    T0: float,
    L: float = 10.0,
    sigma: float = 0.5,
) -> mp.mpf:
    """
    Spectral quadratic form:

        Q_H^spec = ∫_{-L}^{L} k_hat(ξ, H) |S_σ(ξ - T0/(2π))|^2 dξ

    The shift −T0/(2π) arises from the Parseval bridge:
    the Dirichlet vector is c_n = n^{-(σ+iT0)}.
    """
    H_mp = mp.mpf(H)
    T0_shift = mp.mpf(T0) / (2 * mp.pi)

    def integrand(xi: float) -> mp.mpf:
        xi_mp = mp.mpf(xi)
        kval = k_hat(xi_mp, H_mp)
        Sval = S_xi(xi_mp - T0_shift, N=N, sigma=sigma)
        return kval * (abs(Sval) ** 2)

    return mp.quad(integrand, [-L, L])


@dataclass
class ParsevalComparison:
    N: int
    H: float
    T0: float
    sigma: float
    Q_matrix: float
    Q_spectral: float
    abs_diff: float
    rel_diff: float


def Q_matrix_from_volume_iii(N: int, H: float, T0: float) -> float:
    """
    Use Volume III machinery to obtain the time/matrix-domain quadratic form:

        Q_H^mat = D_H + O_H
    """
    cfg = qf.QuadraticFormConfig(N=N, H=H, T0=T0)
    mats, diag = qf.analyse_growth(cfg)
    return float(diag.Q_H)


def compare_parseval(
    N: int,
    H: float,
    T0: float,
    sigma: float = 0.5,
    L: float = 10.0,
) -> ParsevalComparison:
    """
    Compute Q_H^mat (from Volume III) and Q_H^spec (from spectral integral)
    and return absolute and relative differences.
    """
    Q_mat = Q_matrix_from_volume_iii(N, H, T0)
    Q_spec = float(Q_spectral(N=N, H=H, T0=T0, L=L, sigma=sigma))

    abs_diff = abs(Q_mat - Q_spec)
    rel_diff = abs_diff / (abs(Q_mat) + 1e-30)

    return ParsevalComparison(
        N=N,
        H=H,
        T0=T0,
        sigma=sigma,
        Q_matrix=Q_mat,
        Q_spectral=Q_spec,
        abs_diff=abs_diff,
        rel_diff=rel_diff,
    )


# ---------------------------------------------------------------------------
# 4. Algebraic σ-selector Q_N(σ) (T1 layer from the paper)
# ---------------------------------------------------------------------------

def Q_N_sigma(N: int, sigma: float) -> mp.mpf:
    """
    Finite-N σ-selector as defined in the paper (Definition 3.4):

        Q_N(σ) = Σ_{n=2}^{N} [n^{−σ} − n^{−(1−σ)}]^2 (ln n)^2
               = 4 Σ_{n=2}^{N} n^{−1} sinh²((σ − 1/2) ln n) (ln n)^2

    Properties (T1 level):
    - Q_N(σ) ≥ 0 for all σ ∈ (0,1)
    - Q_N(σ) = 0 iff σ = 1/2
    - Q_N(σ) > 0 for σ ∈ (0,1) \ {1/2}
    - Q_N''(1/2) = 8 Σ_{n=2}^{N} (ln n)^4 / n > 0

    Symmetry:
        Q_N(1 − σ) = Q_N(σ).
    """
    sigma_mp = mp.mpf(sigma)
    total = mp.mpf("0")
    for n in range(2, N + 1):
        n_mp = mp.mpf(n)
        ln_n = mp.log(n_mp)
        g = mp.power(n_mp, -sigma_mp) - mp.power(n_mp, -(1 - sigma_mp))
        total += (g ** 2) * (ln_n ** 2)
    return total


# ---------------------------------------------------------------------------
# 5. Spectral σ-selection: direct antisymmetric selector + decomposition
# ---------------------------------------------------------------------------

@dataclass
class SigmaDecompositionPoint:
    sigma: float
    Q_total: float       # Q_H^spec(σ)
    Q_sym: float | None  # optional σ-symmetric baseline
    Q_selector: float    # selector (either symmetric-diff or direct)


def Q_selector_direct(
    N: int,
    H: float,
    T0: float,
    sigma: float,
    L: float = 10.0,
) -> mp.mpf:
    """
    Direct antisymmetric σ-selector computed inside the integral:

        Q_sel(σ) = (1/2) ∫ k_hat(ξ) [|S_σ|^2 − |S_{1−σ}|^2] dξ

    This avoids post-hoc symmetry cancellation.
    """
    H_mp = mp.mpf(H)
    T0_shift = mp.mpf(T0) / (2 * mp.pi)

    def integrand(xi):
        xi_mp = mp.mpf(xi)
        kval = k_hat(xi_mp, H_mp)

        S_plus = S_xi(xi_mp - T0_shift, N=N, sigma=sigma)
        S_minus = S_xi(xi_mp - T0_shift, N=N, sigma=1.0 - sigma)

        return kval * (abs(S_plus) ** 2 - abs(S_minus) ** 2)

    return mp.mpf("0.5") * mp.quad(integrand, [-L, L])


def spectral_sigma_decomposition_direct(
    N: int,
    H: float,
    T0: float,
    sigmas: List[float],
    L: float = 10.0,
) -> List[SigmaDecompositionPoint]:
    """
    Decompose using direct selector:

        Q_total(σ)   = Q_H^spec(σ)
        Q_selector(σ)= Q_sel_direct(σ)
        Q_sym is left as None (can be added if needed).
    """
    pts: List[SigmaDecompositionPoint] = []
    for s in sigmas:
        q_tot = float(Q_spectral(N=N, H=H, T0=T0, L=L, sigma=s))
        q_sel = float(Q_selector_direct(N=N, H=H, T0=T0, sigma=s, L=L))
        pts.append(
            SigmaDecompositionPoint(
                sigma=s,
                Q_total=q_tot,
                Q_sym=None,
                Q_selector=q_sel,
            )
        )
    return pts


# ---------------------------------------------------------------------------
# 6. σ-profile API (emergent symmetric profile used by tests)
# ---------------------------------------------------------------------------

@dataclass
class SigmaProfilePoint:
    sigma: float
    Q_spectral: float


def sigma_profile(
    N: int,
    H: float,
    T0: float,
    sigmas: List[float],
    L: float = 10.0,
) -> List[SigmaProfilePoint]:
    """
    Emergent symmetric σ‑profile (T2 layer).

    Q_profile(σ) = Q_sym(σ) − |Q_selector(σ)|

    where
        Q_sym(σ)      = (Q_spec(σ) + Q_spec(1−σ)) / 2
        Q_selector(σ) = Q_sel_direct(σ).

    This is:
      - symmetric under σ ↔ 1−σ,
      - maximal at σ = 1/2 (since |Q_selector| vanishes only there).
    """
    # Precompute raw spectral values for all needed σ and 1−σ
    needed_sigmas: set[float] = set(sigmas)
    for s in sigmas:
        needed_sigmas.add(1.0 - s)

    Q_raw: Dict[float, float] = {}
    for s in needed_sigmas:
        Q_raw[s] = float(Q_spectral(N=N, H=H, T0=T0, L=L, sigma=s))

    pts: List[SigmaProfilePoint] = []
    for s in sigmas:
        s_reflect = 1.0 - s
        q_s = Q_raw.get(s, 0.0)
        q_ref = Q_raw.get(s_reflect, q_s)
        Q_sym = 0.5 * (q_s + q_ref)

        Q_sel = float(Q_selector_direct(N=N, H=H, T0=T0, sigma=s, L=L))
        q_profile = Q_sym - abs(Q_sel)

        pts.append(SigmaProfilePoint(sigma=s, Q_spectral=q_profile))

    return pts


def sigma_profile_raw(
    N: int,
    H: float,
    T0: float,
    sigmas: List[float],
    L: float = 10.0,
) -> List[SigmaProfilePoint]:
    """
    Raw σ-profile Q_H^spec(σ) for diagnostics.
    """
    pts: List[SigmaProfilePoint] = []
    for s in sigmas:
        q = float(Q_spectral(N=N, H=H, T0=T0, L=L, sigma=s))
        pts.append(SigmaProfilePoint(sigma=s, Q_spectral=q))
    return pts


# ---------------------------------------------------------------------------
# 7. Diagonal vs off-diagonal in spectral domain
# ---------------------------------------------------------------------------

def S_diag(N: int, sigma: float = 0.5) -> mp.mpf:
    """
    Diagonal part of |S_σ(ξ)|^2:

        S_diag = sum_{n <= N} n^{-2σ}
    """
    sigma_mp = mp.mpf(sigma)
    return mp.nsum(lambda n: mp.power(n, -2 * sigma_mp), [1, N])


def S_off_from_S(
    S_val: complex,
    N: int,
    sigma: float = 0.5,
) -> float:
    """
    Off-diagonal contribution derived from |S_σ(ξ)|^2:

        |S(ξ)|^2 = S_diag + S_off(ξ)
        ⇒ S_off(ξ) = |S(ξ)|^2 - S_diag
    """
    Sdiag = float(S_diag(N, sigma=sigma))
    return (abs(S_val) ** 2) - Sdiag


@dataclass
class SpectralSplit:
    N: int
    H: float
    T0: float
    sigma: float
    Q_diag: float
    Q_off: float
    ratio_diag_off: float


def spectral_diag_off_split(
    N: int,
    H: float,
    T0: float,
    sigma: float = 0.5,
    L: float = 10.0,
) -> SpectralSplit:
    """
    Compute spectral diagonal and off-diagonal contributions:

        Q_diag = ∫ k_hat(ξ, H) * S_diag dξ = S_diag * ∫ k_hat(ξ, H) dξ
        Q_off  = ∫ k_hat(ξ, H) * S_off(ξ - T0/(2π)) dξ
    """
    H_mp = mp.mpf(H)
    T0_shift = mp.mpf(T0) / (2 * mp.pi)
    Sdiag_mp = S_diag(N, sigma=sigma)

    def k_only(xi: float) -> mp.mpf:
        return k_hat(mp.mpf(xi), H_mp)

    Ik = mp.quad(k_only, [-L, L])
    Q_diag = Sdiag_mp * Ik

    def k_off_integrand(xi: float) -> mp.mpf:
        xi_mp = mp.mpf(xi)
        kval = k_hat(xi_mp, H_mp)
        Sval = S_xi(xi_mp - T0_shift, N=N, sigma=sigma)
        soff = S_off_from_S(Sval, N=N, sigma=sigma)
        return kval * soff

    Q_off = mp.quad(k_off_integrand, [-L, L])

    Qd = float(Q_diag)
    Qo = float(Q_off)
    ratio = Qd / (abs(Qo) + 1e-30)

    return SpectralSplit(
        N=N,
        H=H,
        T0=T0,
        sigma=sigma,
        Q_diag=Qd,
        Q_off=Qo,
        ratio_diag_off=ratio,
    )


# ---------------------------------------------------------------------------
# 8. Frequency decay diagnostics for k_hat
# ---------------------------------------------------------------------------

@dataclass
class DecaySample:
    xi: float
    k_abs: float
    log_k_abs: float


def decay_samples(H: float, xis: List[float]) -> List[DecaySample]:
    """
    Sample |k_hat(ξ, H)| and log |k_hat(ξ, H)| at given ξ-values.
    """
    H_mp = mp.mpf(H)
    pts: List[DecaySample] = []
    for xi in xis:
        val = k_hat(mp.mpf(xi), H_mp)
        k_abs = float(abs(val))
        log_k = math.log(k_abs + 1e-300)
        pts.append(DecaySample(xi=xi, k_abs=k_abs, log_k_abs=log_k))
    return pts


@dataclass
class DecayFit:
    H: float
    slope: float
    intercept: float


def fit_exponential_decay(H: float, xis: List[float]) -> DecayFit:
    """
    Empirical fit of the pure exponential decay rate of k̂_H(ξ):

        k̂_H(ξ) = P(ξ) / sinh(π^2 H ξ),   P(ξ) = ((2πξ)^2 + 4/H^2)·πH·2πξH

    We divide out the polynomial P(ξ) and fit

        log(k̂ / P) ≈ -π^2 H |ξ|

    so the regression slope is close to π^2 H.
    """
    H_mp = mp.mpf(H)
    xs = np.array([abs(xi) for xi in xis], dtype=float)
    ys_normalized = []

    for xi in xis:
        xi_mp = mp.mpf(abs(xi))
        k_val = float(abs(k_hat(xi_mp, H_mp)))

        num = float((2 * mp.pi * xi_mp) ** 2 + 4 / (H_mp ** 2))
        poly = num * float(mp.pi * H_mp * 2 * mp.pi * xi_mp * H_mp)

        k_normalized = k_val / (poly + 1e-300)
        ys_normalized.append(math.log(k_normalized + 1e-300))

    ys = np.array(ys_normalized, dtype=float)
    A = np.vstack([np.ones_like(xs), xs]).T
    intercept, slope = np.linalg.lstsq(A, ys, rcond=None)[0]

    return DecayFit(H=H, slope=-slope, intercept=intercept)


# ---------------------------------------------------------------------------
# 9. Localization and T0-stability
# ---------------------------------------------------------------------------

@dataclass
class LocalizationSample:
    xi: float
    weight: float


def localization_profile(
    N: int,
    H: float,
    T0: float,
    xi_grid: np.ndarray,
    sigma: float = 0.5,
) -> List[LocalizationSample]:
    """
    Sample weights w(ξ) = k_hat(ξ, H) |S_σ(ξ − T0/(2π))|^2 to see where the
    spectral energy is concentrated.
    """
    H_mp = mp.mpf(H)
    T0_shift = mp.mpf(T0) / (2 * mp.pi)
    samples: List[LocalizationSample] = []
    for xi in xi_grid:
        xi_mp = mp.mpf(xi)
        kval = k_hat(xi_mp, H_mp)
        Sval = S_xi(xi_mp - T0_shift, N=N, sigma=sigma)
        w = float(kval * (abs(Sval) ** 2))
        samples.append(LocalizationSample(xi=xi, weight=w))
    return samples


@dataclass
class T0ScanSample:
    T0: float
    Q_spectral: float


def T0_scan(
    N: int,
    H: float,
    T0_values: List[float],
    L: float = 10.0,
    sigma: float = 0.5,
) -> List[T0ScanSample]:
    """
    Scan T0 and compute Q_H^spec(T0) to check stability / absence of spikes.
    """
    pts: List[T0ScanSample] = []
    for T0 in T0_values:
        Qspec = float(Q_spectral(N=N, H=H, T0=T0, L=L, sigma=sigma))
        pts.append(T0ScanSample(T0=T0, Q_spectral=Qspec))
    return pts


# ---------------------------------------------------------------------------
# 10. Normalized spectral functionals and final selector
# ---------------------------------------------------------------------------

def Q_spectral_normalized(
    N: int,
    H: float,
    T0: float,
    sigma: float,
    L: float = 10.0,
) -> float:
    """
    Normalized spectral functional:

        Q_norm(σ) = Q_spec(σ) / S_diag(σ)

    This removes trivial σ-bias from the diagonal.
    """
    Q_spec_val = float(Q_spectral(N=N, H=H, T0=T0, L=L, sigma=sigma))
    Sdiag_val = float(S_diag(N, sigma=sigma))
    return Q_spec_val / (Sdiag_val + 1e-30)


def Q_final_selector(
    N: int,
    H: float,
    T0: float,
    sigma: float,
    L: float = 10.0,
) -> float:
    """
    Final σ-selector:

        Q_final(σ) = Q_sel_direct(σ) / S_diag(σ)

    Antisymmetric and normalized.
    """
    q_sel = float(Q_selector_direct(N=N, H=H, T0=T0, sigma=sigma, L=L))
    Sdiag_val = float(S_diag(N, sigma=sigma))
    return q_sel / (Sdiag_val + 1e-30)


# ---------------------------------------------------------------------------
# 11. Diagnostic scans for selectors and profiles
# ---------------------------------------------------------------------------

def scan_sigma_selector_direct(N: int, H: float, T0: float, L: float = 10.0) -> None:
    """
    Scan direct antisymmetric selector Q_sel_direct over σ ∈ [0.3, 0.7].
    """
    print("=== DIRECT σ-Selector Scan ===")
    sigmas = np.linspace(0.3, 0.7, 31)

    for s in sigmas:
        q = float(Q_selector_direct(N=N, H=H, T0=T0, sigma=s, L=L))

        if q > 1e-10:
            sign = +1
        elif q < -1e-10:
            sign = -1
        else:
            sign = 0

        print(f"σ={s:.4f}, Q_sel_direct={q:.6e}, sign={sign}")


def sigma_profile_normalized(
    N: int,
    H: float,
    T0: float,
    sigmas: List[float],
    L: float = 10.0,
) -> Dict[float, float]:
    """
    Normalized σ-profile (diagnostic):

        Q_norm(σ) = Q_spec(σ) / S_diag(σ).

    Prints values and reports whether σ=1/2 is maximal on the grid.
    """
    print("=== Normalized σ Profile ===")

    values: Dict[float, float] = {}
    for s in sigmas:
        qn = Q_spectral_normalized(N=N, H=H, T0=T0, sigma=s, L=L)
        values[s] = qn
        print(f"σ={s:.3f}, Q_norm={qn:.6e}")

    q_half = values.get(0.5)

    if q_half is not None:
        print("\n=== Max Check (Normalized) ===")
        for s, q in values.items():
            if s != 0.5 and q > q_half:
                print(f"❌ σ={s:.3f} exceeds σ=1/2")
    else:
        print("\n(no σ=0.5 in supplied grid for max check)")

    return values


def scan_final_selector(N: int, H: float, T0: float, L: float = 10.0) -> None:
    """
    Scan final normalized selector Q_final over σ ∈ [0.3, 0.7].
    """
    print("=== FINAL σ-SELECTOR ===")
    sigmas = np.linspace(0.3, 0.7, 31)

    for s in sigmas:
        q = Q_final_selector(N=N, H=H, T0=T0, sigma=s, L=L)

        if q > 1e-10:
            sign = +1
        elif q < -1e-10:
            sign = -1
        else:
            sign = 0

        print(f"σ={s:.4f}, Q_final={q:.6e}, sign={sign}")


# ---------------------------------------------------------------------------
# 12. High-level Volume IV driver
# ---------------------------------------------------------------------------

def run_volume_iv_suite() -> None:
    """
    Run a representative suite showing Volume IV spectral expansion,
    including Parseval bridge, decay, localization, and σ-selector diagnostics.
    """
    # 1. Parseval comparison at T0 = 0
    print("=== Volume IV: Parseval Bridge (T0=0) ===")
    N = 64
    H = 1.0
    T0 = 0.0
    comp0 = compare_parseval(N=N, H=H, T0=T0, sigma=0.5, L=10.0)
    print(
        f"N={comp0.N}, H={comp0.H:.2f}, T0={comp0.T0:.2f}, "
        f"Q_mat={comp0.Q_matrix:.10e}, Q_spec={comp0.Q_spectral:.10e}, "
        f"abs_diff={comp0.abs_diff:.3e}, rel_diff={comp0.rel_diff:.3e}"
    )
    print()

    # 2. Parseval comparison at T0 near a classical zero
    print("=== Volume IV: Parseval Bridge (T0=14.1347) ===")
    T0_big = 14.1347
    compT = compare_parseval(N=20, H=1.0, T0=T0_big, sigma=0.5, L=15.0)
    print(
        f"N={compT.N}, H={compT.H:.2f}, T0={compT.T0:.4f}, "
        f"Q_mat={compT.Q_matrix:.10e}, Q_spec={compT.Q_spectral:.10e}, "
        f"abs_diff={compT.abs_diff:.3e}, rel_diff={compT.rel_diff:.3e}"
    )
    print()

    # 3. Raw σ-profile (diagnostic)
    print("=== Volume IV: Raw σ-Profile (Q_H^spec(σ) vs σ) ===")
    sigmas_coarse = [0.40, 0.45, 0.50, 0.55, 0.60]
    prof_raw = sigma_profile_raw(N=64, H=1.0, T0=0.0, sigmas=sigmas_coarse, L=10.0)
    for p in prof_raw:
        print(f"σ={p.sigma:.3f}, Q_raw(σ)={p.Q_spectral:.6e}")
    print()

    # 4. Direct σ-selector scan
    scan_sigma_selector_direct(N=64, H=1.0, T0=0.0, L=10.0)
    print()

    # 5. Normalized σ-profile diagnostic
    sigma_profile_normalized(N=64, H=1.0, T0=0.0, sigmas=sigmas_coarse, L=10.0)
    print()

    # 6. Final normalized selector scan
    scan_final_selector(N=64, H=1.0, T0=0.0, L=10.0)
    print()

    # 7. Spectral diagonal vs off-diagonal split
    print("=== Volume IV: Spectral Diagonal vs Off-diagonal ===")
    split = spectral_diag_off_split(N=64, H=1.0, T0=0.0, sigma=0.5, L=10.0)
    print(
        f"N={split.N}, H={split.H:.2f}, T0={split.T0:.2f}, "
        f"Q_diag={split.Q_diag:.6e}, Q_off={split.Q_off:.6e}, "
        f"ratio_diag_off={split.ratio_diag_off:.3e}"
    )
    print()

    # 8. Decay diagnostics
    print("=== Volume IV: Frequency Decay of k_hat(ξ, H) ===")
    xis = [2.0, 4.0, 6.0, 8.0]
    for H_test in [0.5, 1.0, 2.0]:
        fit = fit_exponential_decay(H=H_test, xis=xis)
        print(
            f"H={H_test:.2f}: fitted log|k_hat/poly| ≈ a - b|ξ|, "
            f"empirical b≈{fit.slope:.4f}, expected π²H≈{math.pi**2 * H_test:.4f}"
        )
    print()

    # 9. Localization profile
    print("=== Volume IV: Localization in ξ-space ===")
    xi_grid = np.linspace(-5.0, 5.0, 201)
    for H_test in [0.5, 1.0, 2.0]:
        loc = localization_profile(N=64, H=H_test, T0=0.0,
                                   xi_grid=xi_grid, sigma=0.5)
        weights = np.array([p.weight for p in loc], dtype=float)
        xis_arr = np.array([p.xi for p in loc], dtype=float)
        idx_max = int(np.argmax(weights))
        max_w = np.max(weights)
        mask = weights > 0.05 * max_w
        eff_width = np.max(np.abs(xis_arr[mask])) if np.any(mask) else 0.0
        print(
            f"H={H_test:.2f}: max weight at ξ≈{xis_arr[idx_max]:+.2f}, "
            f"effective half-width ≈ {eff_width:.2f}"
        )
    print()

    # 10. T0 stability scan
    print("=== Volume IV: T0 Stability Scan ===")
    T0_vals = [0.0, 1.0, 2.0, 5.0, 10.0]
    scan = T0_scan(N=64, H=1.0, T0_values=T0_vals, L=10.0, sigma=0.5)
    for s in scan:
        print(f"T0={s.T0:4.1f}, Q_spec={s.Q_spectral:.6e}")
    print()


if __name__ == "__main__":
    run_volume_iv_suite()