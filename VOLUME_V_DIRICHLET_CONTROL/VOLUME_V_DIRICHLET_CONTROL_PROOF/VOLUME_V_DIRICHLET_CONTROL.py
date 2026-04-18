#!/usr/bin/env python3
"""
VOLUME_V_DIRICHLET_CONTROL.py
=============================

Volume V: Dirichlet Polynomial Control

This module turns Dirichlet polynomials

    S_a(ξ) = sum_{n <= N} a_n * exp(2π i ξ log n)

into controlled spectral signals suitable for interaction with the
sech^2-based spectral kernel developed in Volume IV.

It provides:

  1. A generalized Dirichlet wave engine S_a(ξ) with arbitrary coefficients.
  2. Smooth test-function (window) framework (Gaussian, exponential, bump, and
     sech^2-aligned log-weight).
  3. Decay diagnostics and comparison with the kernel decay e^{-π^2 H |ξ|}.
  4. Magnitude bounds: trivial, L^2, and kernel-weighted norms.
  5. Structural decomposition |S(ξ)|^2 = diag + off-diag and diagnostics.
  6. Stability & scaling tests in N, σ, and windows.
  7. Integration hooks to Volume IV's spectral kernel k_hat(ξ, H).

The module is self-contained but assumes Volume IV's k_hat is available;
for integration, import it from your SPECTRAL_EXPANSION module or redefine it
here in exactly the same way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import mpmath as mp
import numpy as np

# ---------------------------------------------------------------------------
# 0. Global high-precision setting
# ---------------------------------------------------------------------------

mp.mp.dps = 80


# ---------------------------------------------------------------------------
# 1. Utility: von Mangoldt and basic tables
# ---------------------------------------------------------------------------

def von_mangoldt(n: int) -> float:
    """
    Von Mangoldt function Λ(n):

        Λ(n) = log p if n = p^k for some prime p, k>=1
             = 0 otherwise

    Implemented via a trial division factorization up to sqrt(n).
    Intended for moderate sizes N (e.g. N up to a few 10^4) for experimentation.
    """
    if n < 2:
        return 0.0

    # Trial division
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            # Factor out all powers of p
            k = 0
            while m % p == 0:
                m //= p
                k += 1
            # If the residue is 1, n is a pure prime power
            return math.log(p) if m == 1 else 0.0
        p += 1 if p == 2 else 2  # 2 then odds

    # If we reach here, m is prime and m^1 = n, so n itself is prime
    return math.log(m)


def build_log_table(N: int) -> np.ndarray:
    """
    Precompute log n for 1 <= n <= N.
    """
    ns = np.arange(1, N + 1, dtype=float)
    return np.log(ns)


def build_lambert_table(N: int) -> np.ndarray:
    """
    Precompute Λ(n) for 1 <= n <= N.
    """
    lam = np.zeros(N, dtype=float)
    for n in range(1, N + 1):
        lam[n - 1] = von_mangoldt(n)
    return lam


# ---------------------------------------------------------------------------
# 2. Volume IV kernel: k_hat(ξ, H)
# ---------------------------------------------------------------------------

def k_hat(xi: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    """
    Analytic Fourier transform of the Bochner-repaired kernel (as in Volume IV):

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
        w_hat_val = mp.pi * H * (2 * mp.pi * a * H) * 2 * exp_term
    else:
        w_hat_val = mp.pi * H * (2 * mp.pi * a * H) / mp.sinh(arg)

    val = num * w_hat_val
    return val if val >= 0 else mp.mpf("0")


# ---------------------------------------------------------------------------
# 3. Window / test functions w(n/N)
# ---------------------------------------------------------------------------

def window_sharp(n: int, N: int) -> float:
    """
    Sharp cutoff window: w(n/N) = 1 for n <= N, 0 otherwise.
    For our arrays of length N, this is effectively all ones.
    """
    return 1.0 if 1 <= n <= N else 0.0


def window_gaussian(n: int, N: int, alpha: float = 1.0) -> float:
    """
    Gaussian window:

        w(n/N) = exp(-alpha * (n/N)^2)
    """
    x = n / float(N)
    return math.exp(-alpha * x * x)


def window_exponential(n: int, N: int, alpha: float = 1.0) -> float:
    """
    Exponential window:

        w(n/N) = exp(-alpha * n/N)
    """
    x = n / float(N)
    return math.exp(-alpha * x)


def window_bump(n: int, N: int) -> float:
    """
    Compact C^∞ bump on (0,1):

        w(x) = exp(-1 / (x(1-x))) for 0 < x < 1
             = 0 otherwise

    evaluated at x = n/N.
    """
    x = n / float(N)
    if x <= 0.0 or x >= 1.0:
        return 0.0
    t = x * (1.0 - x)
    return math.exp(-1.0 / t)


def window_log_sech2(n: int, N: int, T: float, H: float) -> float:
    """
    Log-aligned sech^2 window:

        w(n) = sech^2( (log n - T) / H )

    This is aligned with the Volume IV kernel in the log-domain.
    """
    ln_n = math.log(n)
    z = (ln_n - T) / H
    # sech^2(z) = 1 / cosh^2(z)
    return 1.0 / math.cosh(z) ** 2


# ---------------------------------------------------------------------------
# 4. Generalized Dirichlet wave S_a(ξ)
# ---------------------------------------------------------------------------

@dataclass
class DirichletConfig:
    N: int
    sigma: float = 0.5
    weight_type: str = "plain"  # "plain", "log", "von_mangoldt", "custom"
    window_type: str = "sharp"  # "sharp", "gaussian", "exponential", "bump", "log_sech2"
    window_params: Dict[str, float] | None = None
    custom_coeffs: np.ndarray | None = None
    custom_window: Callable[[int, int], float] | None = None


def build_coefficients(cfg: DirichletConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build coefficient array a_n and log-table log n for given N and sigma.

    Cases:
      - weight_type="plain": a_n = n^{-sigma}
      - weight_type="log":   a_n = (log n) * n^{-sigma}
      - weight_type="von_mangoldt": a_n = Λ(n) * n^{-sigma}
      - weight_type="custom": use cfg.custom_coeffs (length N)

    Returns (a, logn) as numpy arrays of length N.
    """
    N = cfg.N
    sigma = cfg.sigma
    logn = build_log_table(N)
    ns = np.arange(1, N + 1, dtype=float)

    if cfg.weight_type == "plain":
        a = ns ** (-sigma)
    elif cfg.weight_type == "log":
        a = (logn) * ns ** (-sigma)
    elif cfg.weight_type == "von_mangoldt":
        lam = build_lambert_table(N)
        a = lam * ns ** (-sigma)
    elif cfg.weight_type == "custom":
        if cfg.custom_coeffs is None or len(cfg.custom_coeffs) != N:
            raise ValueError("custom_coeffs must be provided and length N for weight_type='custom'.")
        a = np.array(cfg.custom_coeffs, dtype=float)
    else:
        raise ValueError(f"Unknown weight_type: {cfg.weight_type}")

    return a, logn


def apply_window(cfg: DirichletConfig, a: np.ndarray) -> np.ndarray:
    """
    Apply window w(n/N) to coefficients a_n.

    Supported window_type:
      - "sharp":       w(n/N) = 1
      - "gaussian":    w(n/N) = exp(-alpha * (n/N)^2)
      - "exponential": w(n/N) = exp(-alpha * n/N)
      - "bump":        compact C^∞ bump on (0,1)
      - "log_sech2":   w(n) = sech^2((log n - T)/H)
      - "custom":      cfg.custom_window(n, N)
    """
    N = cfg.N
    window_type = cfg.window_type
    params = cfg.window_params or {}

    if window_type == "sharp":
        return a.copy()

    w = np.empty_like(a)
    for idx in range(N):
        n = idx + 1
        if window_type == "gaussian":
            alpha = params.get("alpha", 1.0)
            w[idx] = window_gaussian(n, N, alpha=alpha)
        elif window_type == "exponential":
            alpha = params.get("alpha", 1.0)
            w[idx] = window_exponential(n, N, alpha=alpha)
        elif window_type == "bump":
            w[idx] = window_bump(n, N)
        elif window_type == "log_sech2":
            T = params.get("T", math.log(N))
            H = params.get("H", 1.0)
            w[idx] = window_log_sech2(n, N, T=T, H=H)
        elif window_type == "custom":
            if cfg.custom_window is None:
                raise ValueError("custom_window must be provided for window_type='custom'.")
            w[idx] = cfg.custom_window(n, N)
        else:
            raise ValueError(f"Unknown window_type: {window_type}")

    return a * w


def S_a_xi(
    xi: float | mp.mpf,
    cfg: DirichletConfig,
    a: np.ndarray | None = None,
    logn: np.ndarray | None = None,
) -> complex:
    """
    Generalized Dirichlet wave:

        S_a(ξ) = sum_{n <= N} a_n * w(n/N) * exp(2π i ξ log n)

    If a and logn are not provided, they are constructed from cfg.
    """
    if a is None or logn is None:
        raw_a, logn = build_coefficients(cfg)
        a = apply_window(cfg, raw_a)

    xi_mp = mp.mpf(xi)
    two_pi_i_xi = 2j * mp.pi * xi_mp

    # Evaluate sum in mpmath, but use numpy arrays for logs and coefficients.
    s = mp.mpf("0") + 0j
    for coeff, ln_n in zip(a, logn):
        phase = two_pi_i_xi * mp.mpf(ln_n)
        s += coeff * mp.e ** phase

    return complex(s)


# ---------------------------------------------------------------------------
# 5. ξ-grid and sampling utilities
# ---------------------------------------------------------------------------

def xi_grid(L: float, num: int, adaptive: bool = False) -> np.ndarray:
    """
    Build a ξ-grid over [-L, L].

    If adaptive=True, use a simple scheme that densifies near 0:
      - sample more densely in [-L/4, L/4]
      - more sparsely outside.

    This is a lightweight alternative to full-blown adaptive quadrature.
    """
    if not adaptive:
        return np.linspace(-L, L, num)

    # Half the points in the central region.
    num_center = num // 2
    num_outer = num - num_center

    xs_center = np.linspace(-L / 4.0, L / 4.0, num_center)
    xs_outer_left = np.linspace(-L, -L / 4.0, num_outer // 2, endpoint=False)
    xs_outer_right = np.linspace(L / 4.0, L, num_outer - num_outer // 2)

    return np.concatenate([xs_outer_left, xs_center, xs_outer_right])


# ---------------------------------------------------------------------------
# 6. Decay diagnostics and kernel comparison
# ---------------------------------------------------------------------------

@dataclass
class DecayDiagnostic:
    xi: float
    S_abs: float
    log_S_abs: float
    k_hat_abs: float
    log_k_hat_abs: float


def decay_profile(
    cfg: DirichletConfig,
    xis: Iterable[float],
    H: float,
) -> List[DecayDiagnostic]:
    """
    Sample |S_a(ξ)| and |k_hat(ξ,H)| on a grid and return log-abs diagnostics.

    Used to:
      - empirically verify decay of S_a(ξ),
      - compare against kernel decay.
    """
    raw_a, logn = build_coefficients(cfg)
    a = apply_window(cfg, raw_a)

    diagnostics: List[DecayDiagnostic] = []
    for x in xis:
        S_val = S_a_xi(x, cfg, a=a, logn=logn)
        S_abs = abs(S_val)
        k_val = k_hat(x, H)
        k_abs = float(abs(k_val))
        diagnostics.append(
            DecayDiagnostic(
                xi=x,
                S_abs=S_abs,
                log_S_abs=math.log(S_abs + 1e-300),
                k_hat_abs=k_abs,
                log_k_hat_abs=math.log(k_abs + 1e-300),
            )
        )
    return diagnostics


# ---------------------------------------------------------------------------
# 7. Norms: trivial, L^2, and kernel-weighted
# ---------------------------------------------------------------------------

def trivial_bound(a: np.ndarray) -> float:
    """
    Trivial bound:

        |S_a(ξ)| <= sum |a_n|
    """
    return float(np.sum(np.abs(a)))


def L2_norm_S(
    cfg: DirichletConfig,
    L: float,
    num_xi: int = 2048,
) -> float:
    """
    Approximate L^2 norm:

        ||S_a||_2^2 ≈ ∫_{-L}^L |S_a(ξ)|^2 dξ

    via trapezoidal rule over a uniform grid.
    """
    xs = np.linspace(-L, L, num_xi)
    raw_a, logn = build_coefficients(cfg)
    a = apply_window(cfg, raw_a)

    vals = np.empty_like(xs, dtype=float)
    for i, x in enumerate(xs):
        S_val = S_a_xi(x, cfg, a=a, logn=logn)
        vals[i] = abs(S_val) ** 2

    # trapezoidal rule
    dx = xs[1] - xs[0]
    integral = float(np.trapz(vals, dx=dx))
    return integral


def kernel_weighted_norm(
    cfg: DirichletConfig,
    H: float,
    L: float,
    num_xi: int = 1024,
) -> float:
    """
    Approximate kernel-weighted norm:

        ∫_{-L}^L k_hat(ξ,H) |S_a(ξ)|^2 dξ

    via trapezoidal rule.
    """
    xs = np.linspace(-L, L, num_xi)
    raw_a, logn = build_coefficients(cfg)
    a = apply_window(cfg, raw_a)

    vals = np.empty_like(xs, dtype=float)
    for i, x in enumerate(xs):
        S_val = S_a_xi(x, cfg, a=a, logn=logn)
        kval = k_hat(x, H)
        vals[i] = float(kval) * (abs(S_val) ** 2)

    dx = xs[1] - xs[0]
    integral = float(np.trapz(vals, dx=dx))
    return integral


# ---------------------------------------------------------------------------
# 8. Structural decomposition |S(ξ)|² = diag + off-diag
# ---------------------------------------------------------------------------

@dataclass
class StructuralDecomposition:
    diag: float
    off_diag: float
    total: float


def structural_decomposition_at_xi(
    xi: float,
    cfg: DirichletConfig,
    a: np.ndarray | None = None,
    logn: np.ndarray | None = None,
) -> StructuralDecomposition:
    """
    Compute the exact decomposition:

        |S_a(ξ)|^2 = sum_n |a_n w_n|^2 + sum_{n != m} a_n w_n conj(a_m w_m) e^{2π i ξ (log n - log m)}

    but return diag and off_diag as scalars. Here we compute:

        diag = sum |b_n|^2
        off_diag = |S_a(ξ)|^2 - diag

    where b_n = a_n w(n/N).
    """
    if a is None or logn is None:
        raw_a, logn = build_coefficients(cfg)
        a = apply_window(cfg, raw_a)

    # diag
    diag = float(np.sum(np.abs(a) ** 2))

    # total directly via S_a_xi
    S_val = S_a_xi(xi, cfg, a=a, logn=logn)
    total = abs(S_val) ** 2

    off_diag = total - diag

    return StructuralDecomposition(diag=diag, off_diag=off_diag, total=total)


@dataclass
class StructuralProfilePoint:
    xi: float
    diag: float
    off_diag: float
    total: float
    ratio_diag_off: float


def structural_profile(
    cfg: DirichletConfig,
    xis: Iterable[float],
) -> List[StructuralProfilePoint]:
    """
    Compute structural decomposition for a list of ξ values and form diagnostics.
    """
    raw_a, logn = build_coefficients(cfg)
    a = apply_window(cfg, raw_a)

    profile: List[StructuralProfilePoint] = []
    for x in xis:
        dec = structural_decomposition_at_xi(x, cfg, a=a, logn=logn)
        ratio = dec.diag / (abs(dec.off_diag) + 1e-30)
        profile.append(
            StructuralProfilePoint(
                xi=x,
                diag=dec.diag,
                off_diag=dec.off_diag,
                total=dec.total,
                ratio_diag_off=ratio,
            )
        )
    return profile


# ---------------------------------------------------------------------------
# 9. Stability & scaling diagnostics
# ---------------------------------------------------------------------------

@dataclass
class ScalingDiagnostic:
    N: int
    sigma: float
    window_type: str
    trivial_bound: float
    L2_norm: float
    kernel_norm: float
    max_abs_S: float


def max_abs_S_on_grid(
    cfg: DirichletConfig,
    L: float,
    num_xi: int = 2048,
) -> float:
    """
    Find max_{ξ in [-L,L]} |S_a(ξ)| on a grid.
    """
    xs = np.linspace(-L, L, num_xi)
    raw_a, logn = build_coefficients(cfg)
    a = apply_window(cfg, raw_a)

    max_val = 0.0
    for x in xs:
        S_val = S_a_xi(x, cfg, a=a, logn=logn)
        val = abs(S_val)
        if val > max_val:
            max_val = val
    return max_val


def scaling_diagnostics(
    Ns: List[int],
    sigma: float,
    window_type: str,
    window_params: Dict[str, float] | None,
    H: float,
    L: float,
    num_xi: int = 1024,
) -> List[ScalingDiagnostic]:
    """
    For a list of N, compute:

        - trivial bound
        - L^2 norm over [-L,L]
        - kernel-weighted norm over [-L,L]
        - max |S_a(ξ)| on [-L,L]

    to understand scaling and stability.
    """
    diagnostics: List[ScalingDiagnostic] = []

    for N in Ns:
        cfg = DirichletConfig(
            N=N,
            sigma=sigma,
            weight_type="plain",
            window_type=window_type,
            window_params=window_params,
        )
        raw_a, _ = build_coefficients(cfg)
        a = apply_window(cfg, raw_a)

        tb = trivial_bound(a)
        l2 = L2_norm_S(cfg, L=L, num_xi=num_xi)
        kn = kernel_weighted_norm(cfg, H=H, L=L, num_xi=num_xi)
        maxS = max_abs_S_on_grid(cfg, L=L, num_xi=num_xi)

        diagnostics.append(
            ScalingDiagnostic(
                N=N,
                sigma=sigma,
                window_type=window_type,
                trivial_bound=tb,
                L2_norm=l2,
                kernel_norm=kn,
                max_abs_S=maxS,
            )
        )

    return diagnostics


# ---------------------------------------------------------------------------
# 10. σ-symmetry diagnostics
# ---------------------------------------------------------------------------

@dataclass
class SigmaSymmetryPoint:
    sigma: float
    L2_norm: float
    kernel_norm: float
    max_abs_S: float


def sigma_symmetry_profile(
    N: int,
    sigmas: Iterable[float],
    window_type: str,
    window_params: Dict[str, float] | None,
    H: float,
    L: float,
) -> Dict[float, SigmaSymmetryPoint]:
    """
    For fixed N, compute norms and max |S_a| for each σ in sigmas, to probe
    symmetry under σ ↔ 1−σ and monotonic behaviors.
    """
    profile: Dict[float, SigmaSymmetryPoint] = {}

    for sigma in sigmas:
        cfg = DirichletConfig(
            N=N,
            sigma=sigma,
            weight_type="plain",
            window_type=window_type,
            window_params=window_params,
        )
        l2 = L2_norm_S(cfg, L=L, num_xi=1024)
        kn = kernel_weighted_norm(cfg, H=H, L=L, num_xi=1024)
        maxS = max_abs_S_on_grid(cfg, L=L, num_xi=1024)

        profile[sigma] = SigmaSymmetryPoint(
            sigma=sigma,
            L2_norm=l2,
            kernel_norm=kn,
            max_abs_S=maxS,
        )

    return profile


# ---------------------------------------------------------------------------
# 11. Integration with Volume IV spectral functional
# ---------------------------------------------------------------------------

def Q_spectral_dirichlet(
    cfg: DirichletConfig,
    H: float,
    T0: float,
    L: float = 10.0,
    num_xi: int = 1024,
) -> float:
    """
    Dirichlet-based spectral quadratic form analogous to Volume IV:

        Q_spec^V(σ) = ∫_{-L}^{L} k_hat(ξ,H) |S_a(ξ - T0/(2π))|^2 dξ

    where S_a is defined by cfg (coefficients + window).
    """
    xs = np.linspace(-L, L, num_xi)
    raw_a, logn = build_coefficients(cfg)
    a = apply_window(cfg, raw_a)

    vals = np.empty_like(xs, dtype=float)
    shift = T0 / (2 * math.pi)
    for i, x in enumerate(xs):
        xi_shift = x - shift
        S_val = S_a_xi(xi_shift, cfg, a=a, logn=logn)
        kval = k_hat(x, H)
        vals[i] = float(kval) * (abs(S_val) ** 2)

    dx = xs[1] - xs[0]
    integral = float(np.trapz(vals, dx=dx))
    return integral


# ---------------------------------------------------------------------------
# 12. Example driver (for manual experimentation, not used by tests)
# ---------------------------------------------------------------------------

def run_volume_v_demo() -> None:
    """
    Demonstration routine for Volume V diagnostics.

    This is a minimal, non-exhaustive example. Proper TDD for Volume V should
    be implemented in a separate test module.
    """
    print("=== VOLUME V: Dirichlet Control Demo ===")

    N = 100
    sigma = 0.5
    H = 1.0
    L = 5.0

    cfg_plain = DirichletConfig(
        N=N,
        sigma=sigma,
        weight_type="plain",
        window_type="sharp",
    )

    raw_a, logn = build_coefficients(cfg_plain)
    a = apply_window(cfg_plain, raw_a)
    tb = trivial_bound(a)
    print(f"Trivial bound (plain, sharp): {tb:.6e}")

    l2 = L2_norm_S(cfg_plain, L=L, num_xi=2048)
    print(f"L^2 norm over [-{L},{L}]: {l2:.6e}")

    kn = kernel_weighted_norm(cfg_plain, H=H, L=L, num_xi=2048)
    print(f"Kernel-weighted norm (H={H}): {kn:.6e}")

    xs = xi_grid(L=L, num=17, adaptive=True)
    decays = decay_profile(cfg_plain, xs, H=H)
    print("\nDecay sample (ξ, log|S|, log|k_hat|):")
    for d in decays:
        print(f"ξ={d.xi:+.2f}, log|S|={d.log_S_abs:.3f}, log|k|={d.log_k_hat_abs:.3f}")

    sp = structural_profile(cfg_plain, xs)
    print("\nStructural split at a few ξ:")
    for p in sp:
        print(
            f"ξ={p.xi:+.2f}, diag={p.diag:.3e}, off={p.off_diag:.3e}, total={p.total:.3e}, "
            f"ratio_diag_off={p.ratio_diag_off:.3e}"
        )

    Ns = [50, 100, 200]
    print("\nScaling diagnostics (plain, sharp):")
    scale = scaling_diagnostics(
        Ns=Ns,
        sigma=sigma,
        window_type="sharp",
        window_params=None,
        H=H,
        L=L,
        num_xi=1024,
    )
    for d in scale:
        print(
            f"N={d.N:4d}, trivial={d.trivial_bound:.3e}, "
            f"L2={d.L2_norm:.3e}, kernel={d.kernel_norm:.3e}, max|S|={d.max_abs_S:.3e}"
        )

    sigmas = [0.3, 0.4, 0.5, 0.6, 0.7]
    print("\nσ-symmetry diagnostics (plain, sharp):")
    sym = sigma_symmetry_profile(
        N=100,
        sigmas=sigmas,
        window_type="sharp",
        window_params=None,
        H=H,
        L=L,
    )
    for s in sigmas:
        p = sym[s]
        print(
            f"σ={s:.2f}, L2={p.L2_norm:.3e}, "
            f"kernel={p.kernel_norm:.3e}, max|S|={p.max_abs_S:.3e}"
        )

    print("\nDirichlet-based Q_spec^V demo:")
    qv = Q_spectral_dirichlet(cfg_plain, H=H, T0=0.0, L=L, num_xi=1024)
    print(f"Q_spec^V(σ={sigma:.2f}) ≈ {qv:.6e}")


if __name__ == "__main__":
    run_volume_v_demo()