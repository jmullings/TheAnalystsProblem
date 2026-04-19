#!/usr/bin/env python3
"""
VOLUME IX — Convolution Positivity
==================================

Final, test-aligned implementation with:

- Exact λ* = 4/H^2 with test-compatible signature.
- Convolution positivity via k_H(t) = (6/H^2) sech^4(t/H).
- Curvature leakage via the negativity region of w_H''.
- Pointwise domination check λ* w_H ≥ |w_H''| on the negativity region.
- Time–frequency consistency using the algebraically exact discrete identity

      Q_time = Σ_{n,m} a_n a_m* (n/m)^{-iT0} hat{k}_H((log m − log n)/(2π)),

  where hat{k}_H(ξ) = ((2πξ)^2 + 4/H^2) hat{w}_H(ξ) with hat{w}_H given by
  the analytic closed-form formula for the Fourier transform of sech^2(t/H).

  Because D_N(t) = Σ_n a_n n^{-it} is a finite sum of pure tones (not an L²
  function), its distributional Fourier transform is a sum of Dirac deltas.
  Plancherel's theorem therefore does not reduce to a continuous ξ-integral;
  it collapses exactly to the discrete double sum above.  The previous
  implementation incorrectly treated S(ξ) = Σ_n a_n n^{-iξ} as a proxy for
  the spectral density and integrated it continuously — that is the source of
  the ~4-unit mismatch observed in the failing test.

The demo at the bottom runs a sanity check with moderate parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional

import mpmath as mp
import numpy as np

mp.mp.dps = 70

# ---------------------------------------------------------------------------
# Dirichlet polynomial infrastructure (Volume V) — with fallbacks
# ---------------------------------------------------------------------------

try:
    from VOLUME_V_DIRICHLET_CONTROL.VOLUME_V_DIRICHLET_CONTROL_PROOF.VOLUME_V_DIRICHLET_CONTROL import (  # type: ignore  # noqa: E501
        DirichletConfig,
        build_coefficients,
        apply_window,
    )
except Exception:  # pragma: no cover
    from dataclasses import dataclass as _dc_dataclass

    @_dc_dataclass
    class DirichletConfig:
        N: int
        sigma: float = 0.5
        weight_type: str = "plain"
        window_type: str = "sharp"
        window_params: Optional[Dict[str, float]] = None
        custom_coeffs: Optional[np.ndarray] = None
        custom_window: Optional[Callable[[int, int], float]] = None

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


def D_N_from_config(t: float, cfg: DirichletConfig) -> complex:
    """
    D_N(t) = ∑_{n ≤ N} a_n n^{-i t}, coefficients and windows from cfg.
    """
    a_raw, logn = build_coefficients(cfg)
    a = apply_window(cfg, a_raw)
    t_mp = mp.mpf(t)
    re = mp.mpf("0")
    im = mp.mpf("0")
    for coeff, ln in zip(a, logn):
        ln_mp = mp.mpf(ln)
        phase = t_mp * ln_mp
        c = mp.cos(phase)
        s = mp.sin(phase)
        coeff_mp = mp.mpf(coeff)
        re += coeff_mp * c
        im += -coeff_mp * s
    return complex(re, im)


def D_N_abs_sq_from_cfg(t: float, cfg: DirichletConfig) -> float:
    val = D_N_from_config(t, cfg)
    return float((val.conjugate() * val).real)


# ---------------------------------------------------------------------------
# Kernel w_H(t) = sech^2(t/H) and Bochner‑repaired k_H(t)
# ---------------------------------------------------------------------------

def sech(x: mp.mpf) -> mp.mpf:
    return 1 / mp.cosh(x)


def w_H(t: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    """
    Base kernel w_H(t) = sech^2(t/H).
    """
    t_mp = mp.mpf(t)
    H_mp = mp.mpf(H)
    u = t_mp / H_mp
    s = sech(u)
    return s * s


def w_H_second_derivative(t: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    """
    Second derivative of w_H(t) = sech^2(t/H) with respect to t.

    If w(u) = sech^2(u), then
      w''(u) = 4 sech^2(u) tanh^2(u) - 2 sech^4(u).

    For u = t/H:
      d^2/dt^2 w_H(t) = (1/H^2) w''(t/H).
    """
    t_mp = mp.mpf(t)
    H_mp = mp.mpf(H)
    u = t_mp / H_mp
    s = sech(u)
    s2 = s * s
    th = mp.tanh(u)
    th2 = th * th
    wpp_u = 4 * s2 * th2 - 2 * s2 * s2
    return wpp_u / (H_mp * H_mp)


def k_H(t: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    """
    Bochner‑repaired kernel:

      k_H(t) = -w_H''(t) + 4/H^2 * w_H(t).

    Using the explicit algebraic identity this equals:

      k_H(t) = (6/H^2) * sech^4(t/H),

    which is strictly positive for all t.
    """
    t_mp = mp.mpf(t)
    H_mp = mp.mpf(H)
    u = t_mp / H_mp
    s = sech(u)
    s2 = s * s
    s4 = s2 * s2
    return (6 / (H_mp * H_mp)) * s4


# ---------------------------------------------------------------------------
# Negativity region for w_H'' and associated constants
# ---------------------------------------------------------------------------

@dataclass
class NegativityRegion:
    t_min: float
    t_max: float
    max_abs_second_derivative: float
    length: float
    integral_abs_second_derivative: float


def compute_negativity_region(H: float) -> NegativityRegion:
    """
    For w_H(t) = sech^2(t/H), w_H''(t) changes sign where
      4 - 6 sech^2(t/H) = 0.

    Solving for u = t/H:
      sech^2(u) = 2/3
      cosh^2(u) = 3/2
      cosh(u) = sqrt(3/2)
      u0 = arcosh(sqrt(3/2)).

    The negativity region for w''(u) is |u| < u0.
    For w_H''(t), the region is |t| < H * u0.
    """
    H_mp = mp.mpf(H)
    u0 = mp.acosh(mp.sqrt(mp.mpf("3") / mp.mpf("2")))
    t0 = float(H_mp * u0)
    t_min = -t0
    t_max = t0

    xs = np.linspace(t_min, t_max, 401)
    max_abs = 0.0
    for x in xs:
        val = float(mp.fabs(w_H_second_derivative(x, H_mp)))
        if val > max_abs:
            max_abs = val

    def abs_wpp(t_mp):
        return mp.fabs(w_H_second_derivative(t_mp, H_mp))

    integral_abs = mp.quad(abs_wpp, [t_min, t_max])
    length = t_max - t_min

    return NegativityRegion(
        t_min=t_min,
        t_max=t_max,
        max_abs_second_derivative=max_abs,
        length=length,
        integral_abs_second_derivative=float(integral_abs),
    )


# ---------------------------------------------------------------------------
# Supremum bound on |D_N|^2
# ---------------------------------------------------------------------------

def sup_D_sq(cfg: DirichletConfig, t_min: float = -10.0, t_max: float = 10.0, samples: int = 1024) -> float:
    """
    Crude sup bound on |D_N(t)|^2 on [t_min, t_max] by sampling.
    In practice, you may replace this with a Volume VI large sieve bound.
    """
    ts = np.linspace(t_min, t_max, samples)
    sup_val = 0.0
    for t in ts:
        val = D_N_abs_sq_from_cfg(t, cfg)
        if val > sup_val:
            sup_val = val
    return sup_val


# ---------------------------------------------------------------------------
# Tail bound for convolution integral
# ---------------------------------------------------------------------------

def tail_bound_convolution(H: float, L: float, bound_on_D_sq: float) -> float:
    """
    Bound ∫_{|t|>L} w_H(t) |D_N(T0 + t)|^2 dt using:

      ∫_{|t|>L} w_H(t) |D|^2 dt ≤ sup |D|^2 * 2 ∫_L^∞ sech^2(t/H) dt.

    Let u = t/H. Then:

      ∫_L^∞ sech^2(t/H) dt = H ∫_{L/H}^∞ sech^2(u) du
                           = H [tanh(u)]_{L/H}^∞
                           = H (1 - tanh(L/H)).

    So the two‑sided tail is 2H (1 - tanh(L/H)).
    """
    if L <= 0 or H <= 0:
        return float("inf")
    H_mp = mp.mpf(H)
    L_mp = mp.mpf(L)
    u0 = L_mp / H_mp
    tail_one_side = H_mp * (1 - mp.tanh(u0))
    tail_two_sides = 2 * tail_one_side
    return float(tail_two_sides) * float(bound_on_D_sq)


# ---------------------------------------------------------------------------
# Lambda* computation (exact, test-compatible signature)
# ---------------------------------------------------------------------------

def compute_lambda_star(H: float, xi_max: float = 5.0, samples: int = 100) -> float:
    """
    Exact positivity constant λ* such that:

      hat_k_H(xi) = ((2*pi*xi)^2 + 4/H^2) * hat_w_H(xi)
                 >= (4/H^2) * hat_w_H(xi),

    so:

      λ* = 4 / H^2.

    The extra parameters (xi_max, samples) are accepted for compatibility
    with the validation tests but are not used in the exact formula.
    """
    if H <= 0:
        raise ValueError("H must be positive")
    return 4.0 / (H * H)


# ---------------------------------------------------------------------------
# Pointwise domination check: λ* w_H ≥ |w_H''| on negativity region
# ---------------------------------------------------------------------------

def verify_pointwise_domination(H: float, samples: int = 1000) -> bool:
    """
    Verify numerically that:

      λ* w_H(t) ≥ |w_H''(t)|

    for t in the negativity region N_H, where λ* = 4/H^2.
    """
    lam = compute_lambda_star(H)
    neg = compute_negativity_region(H)
    xs = np.linspace(neg.t_min, neg.t_max, samples)
    for t in xs:
        w = float(w_H(t, H))
        wpp = abs(float(w_H_second_derivative(t, H)))
        if lam * w + 1e-14 < wpp:
            return False
    return True


# ---------------------------------------------------------------------------
# Convolution integral evaluator
# ---------------------------------------------------------------------------

def convolution_integrand(
    t_mp,
    cfg: DirichletConfig,
    H: float,
    T0: float,
) -> mp.mpf:
    t = float(t_mp)
    return k_H(t, H) * mp.mpf(D_N_abs_sq_from_cfg(T0 + t, cfg))


def convolution_integral(
    cfg: DirichletConfig,
    H: float,
    T0: float,
    L: float,
    tol: float,
) -> Tuple[float, float]:
    """
    Compute ∫_{-L}^{L} k_H(t) |D_N(T0 + t)|^2 dt with high‑precision quadrature,
    and return (value, truncation_error_bound) where the latter controls
    the tail |t| > L using the base kernel w_H.
    """
    if L <= 0:
        raise ValueError("L must be positive")

    mp.mp.dps = max(mp.mp.dps, int(-math.log10(max(tol, 1e-20))) + 20)

    I = mp.quad(
        lambda t_mp: convolution_integrand(t_mp, cfg, H, T0),
        [-L, L],
    )
    conv_val = float(I)

    sup_D2 = sup_D_sq(cfg, t_min=T0 - L, t_max=T0 + L, samples=1024)
    tail_err = tail_bound_convolution(H, L, sup_D2)

    return conv_val, tail_err


# ---------------------------------------------------------------------------
# Positive floor term
# ---------------------------------------------------------------------------

def positive_floor(
    cfg: DirichletConfig,
    H: float,
    T0: float,
    L: float,
    tol: float,
) -> float:
    """
    Compute a positive floor term of the form:

      λ* ∫ w_H(t) |D_N(T0 + t)|^2 dt,

    with exact λ* = 4/H^2, integral over [-L, L].
    """
    lam_star = compute_lambda_star(H)

    def integrand(t_mp):
        t = float(t_mp)
        return w_H(t, H) * mp.mpf(D_N_abs_sq_from_cfg(T0 + t, cfg))

    I = mp.quad(integrand, [-L, L])
    return lam_star * float(I)


# ---------------------------------------------------------------------------
# Curvature leakage bound (tightened)
# ---------------------------------------------------------------------------

def curvature_leakage_bound(
    cfg: DirichletConfig,
    H: float,
    T0: float,
) -> float:
    """
    Bound the integral of |w_H''(t)| |D_N(T0 + t)|^2 over the negativity
    region N_H using:

      ∫_{N_H} |w_H''(t)| |D_N(T0 + t)|^2 dt
        ≤ (∫_{N_H} |w_H''(t)| dt) · sup_{N_H} |D_N|^2.
    """
    neg = compute_negativity_region(H)
    sup_D2 = sup_D_sq(cfg, t_min=T0 + neg.t_min, t_max=T0 + neg.t_max, samples=512)
    leakage = neg.integral_abs_second_derivative * sup_D2
    return leakage


# ---------------------------------------------------------------------------
# Positivity result data structure and verifier
# ---------------------------------------------------------------------------

@dataclass
class PositivityResult:
    cfg: DirichletConfig
    H: float
    T0: float
    L: float
    convolution_value: float
    convolution_tail_error: float
    positive_floor_value: float
    curvature_leakage_bound: float
    net_bound_floor_minus_leakage: float
    guaranteed_positive: bool
    direct_positive_within_error: bool
    total_error_bound: float
    interval_for_Q: Tuple[float, float]
    lambda_star: float
    pointwise_domination_holds: bool


def verify_net_positivity(
    cfg: DirichletConfig,
    H: float,
    T0: float,
    L: float,
    tol: float = 1e-10,
) -> PositivityResult:
    """
    Main verifier combining:

      - Convolution integral and its tail error.
      - Positive floor term.
      - Curvature leakage bound.
      - λ* and pointwise domination check.
    """
    conv_val, tail_err = convolution_integral(cfg, H, T0, L, tol)
    floor_val = positive_floor(cfg, H, T0, L, tol)
    leakage_bound = curvature_leakage_bound(cfg, H, T0)
    lam_star = compute_lambda_star(H)
    domination_ok = verify_pointwise_domination(H)

    net_bound = floor_val - leakage_bound
    guaranteed_positive = net_bound > 0.0

    lower_direct = conv_val - tail_err
    direct_positive = lower_direct >= -abs(conv_val) * 1e-12

    total_err = tail_err
    interval = (conv_val - total_err, conv_val + total_err)

    return PositivityResult(
        cfg=cfg,
        H=H,
        T0=T0,
        L=L,
        convolution_value=conv_val,
        convolution_tail_error=tail_err,
        positive_floor_value=floor_val,
        curvature_leakage_bound=leakage_bound,
        net_bound_floor_minus_leakage=net_bound,
        guaranteed_positive=guaranteed_positive,
        direct_positive_within_error=direct_positive,
        total_error_bound=total_err,
        interval_for_Q=interval,
        lambda_star=lam_star,
        pointwise_domination_holds=domination_ok,
    )


# ---------------------------------------------------------------------------
# Analytic Fourier transform of w_H(t) = sech^2(t/H)
# ---------------------------------------------------------------------------

def hat_w_H_analytic(xi: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    """
    Exact Fourier transform (unitary convention, exp(-2*pi*i*xi*t)):

        hat_w_H(xi) = integral sech^2(t/H) exp(-2*pi*i*xi*t) dt.

    Derived via the residue theorem (poles of sech^2(z/H) at z = i*pi*H*(k+1/2),
    k in Z):

        hat{w}_H(ξ) = 2π²H²|ξ| / sinh(π²|ξ|H)   for ξ ≠ 0,
        hat{w}_H(0) = 2H.

    Equivalently written as π·H·(2π|ξ|H) / sinh(π²|ξ|H).

    For |ξ|H so large that π²|ξ|H > 50 we use the one-sided exponential
    approximation sinh(x) ≈ e^x/2 to avoid overflow.
    """
    xi = mp.mpf(xi)
    H = mp.mpf(H)
    if xi == 0:
        return 2 * H
    a = mp.fabs(xi)
    arg = (mp.pi ** 2) * a * H
    if arg > 50:
        # sinh(arg) ≈ exp(arg)/2, so 1/sinh(arg) ≈ 2*exp(-arg)
        exp_term = mp.exp(-arg)
        return mp.pi * H * (2 * mp.pi * a * H) * 2 * exp_term
    else:
        return mp.pi * H * (2 * mp.pi * a * H) / mp.sinh(arg)


def hat_k_H_analytic(xi: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    """
    Exact Fourier transform of k_H(t) = (6/H^2) sech^4(t/H).

    Uses the algebraic identity (valid because k_H = -w_H'' + (4/H^2) w_H):

        hat{k}_H(ξ) = ((2πξ)^2 + 4/H^2) * hat{w}_H(ξ).
    """
    xi = mp.mpf(xi)
    H = mp.mpf(H)
    factor = (2 * mp.pi * xi) ** 2 + 4 / (H * H)
    return factor * hat_w_H_analytic(xi, H)


# ---------------------------------------------------------------------------
# Time vs frequency domain comparison (discrete Plancherel identity)
# ---------------------------------------------------------------------------

def compare_time_freq_domains(
    cfg: DirichletConfig,
    H: float,
    T0: float,
    L_t: float,
    L_xi: float,        # accepted for API compatibility; unused in the discrete sum
    tol: float = 1e-10,
) -> Dict[str, float]:
    """
    Compare the time-domain convolution Q_time against the mathematically
    equivalent discrete frequency-domain expression Q_freq.

    **Time domain**::

        Q_time = ∫_{-L_t_eff}^{L_t_eff} k_H(t) |D_N(T0 + t)|^2 dt,

    where L_t_eff = max(L_t, 8) to reduce truncation error.

    **Frequency domain (discrete Plancherel)**::

        D_N(t) = ∑_n a_n n^{-it} = ∑_n a_n e^{-it log n}

    has distributional Fourier transform

        hat{D_N}(ξ) = ∑_n a_n δ(ξ + log(n)/(2π)).

    Plancherel's theorem therefore gives:

        Q_time = ∑_{n,m} a_n a_m* (n/m)^{-iT0} hat{k}_H((log m − log n)/(2π)),

    which for real-valued coefficients simplifies to:

        Q_freq = ∑_{n,m} a_n a_m cos(T0 (log n − log m))
                          * hat{k}_H((log m − log n)/(2π)).

    This is an exact finite double sum — no continuous ξ-integral is needed.

    .. note::
        The previous implementation integrated hat{k}_H(ξ)|S(ξ)|² dξ where
        S(ξ) = ∑_n a_n n^{-iξ} is a Dirichlet polynomial in the *spectral*
        variable ξ.  That is NOT the Fourier transform of D_N and produces a
        systematically different (and incorrect) value.  The correct picture
        is that every (n, m) cross-frequency pair contributes a delta at
        ξ = (log m − log n)/(2π), picking out hat{k}_H at that exact point.

    Parameters
    ----------
    cfg : DirichletConfig
        Dirichlet polynomial configuration (N, weights, window).
    H : float
        Kernel bandwidth parameter (H > 0).
    T0 : float
        Centre point for the convolution.
    L_t : float
        Integration half-width in the time domain.
    L_xi : float
        Accepted for API compatibility; not used (the discrete sum is exact).
    tol : float
        Quadrature tolerance passed to :func:`convolution_integral`.

    Returns
    -------
    dict with keys:
        ``Q_time``          – time-domain numerical integral,
        ``Q_time_tail_err`` – tail truncation bound,
        ``Q_freq``          – discrete Plancherel double sum,
        ``difference``      – Q_time − Q_freq.
    """
    # Use slightly enlarged domain for reduced truncation error
    L_t_eff = max(L_t, 8.0)

    # Time domain
    Q_time, tail_err_time = convolution_integral(cfg, H, T0, L_t_eff, tol)

    # Frequency domain: discrete Plancherel double sum
    a_raw, logn = build_coefficients(cfg)
    a = apply_window(cfg, a_raw)          # shape (N,), real-valued
    N = cfg.N
    H_mp = mp.mpf(H)

    Q_freq_mp = mp.mpf("0")
    for n_idx in range(N):
        for m_idx in range(N):
            log_n = mp.mpf(logn[n_idx])
            log_m = mp.mpf(logn[m_idx])
            # Frequency at which the (n,m) pair contributes:
            # hat{D_N}(ξ) has a delta at ξ = -log(n)/(2π), so the cross
            # term (n,m) picks out hat{k}_H at ξ = (log m - log n)/(2π).
            xi_nm = (log_m - log_n) / (2 * mp.pi)

            # Phase factor: a_n * a_m* * (n/m)^{-iT0}
            # = a_n * a_m * e^{-iT0(log n - log m)}   (coefficients are real)
            # Real part: a_n * a_m * cos(T0 * (log n - log m))
            phase = mp.mpf(T0) * (log_n - log_m)
            weight = mp.mpf(a[n_idx]) * mp.mpf(a[m_idx]) * mp.cos(phase)

            k_hat_val = hat_k_H_analytic(xi_nm, H_mp)
            Q_freq_mp += weight * k_hat_val

    Q_freq = float(Q_freq_mp)

    return {
        "Q_time": Q_time,
        "Q_time_tail_err": tail_err_time,
        "Q_freq": Q_freq,
        "difference": Q_time - Q_freq,
    }


# ---------------------------------------------------------------------------
# Demo / sanity check
# ---------------------------------------------------------------------------

def _demo():
    """
    Sanity check demo:

      - Prints λ*, negativity region, and curvature/leakage balance.
      - Shows time-domain Q(T0) and the discrete Plancherel frequency value,
        along with their difference (should be < 1e-6 after truncation).

    Parameters are chosen to run in a few seconds while exercising the
    full Volume IX machinery.
    """
    cfg = DirichletConfig(
        N=8,
        sigma=0.5,
        window_type="gaussian",
        window_params={"alpha": 3.0},
    )
    H = 1.0
    T0 = 0.0
    L_t = 4.0
    L_xi = 4.0

    print("=== Volume IX Convolution Positivity Demo ===")
    print(f"H = {H}, T0 = {T0}, L_t = {L_t}, L_xi = {L_xi}")

    lam = compute_lambda_star(H)
    print(f"lambda_star (exact)       = {lam:.12e}")

    neg = compute_negativity_region(H)
    print(f"negativity_region [t_min, t_max] = [{neg.t_min:.6f}, {neg.t_max:.6f}]")
    print(f"negativity |w_H''| integral      = {neg.integral_abs_second_derivative:.12e}")

    res = verify_net_positivity(cfg, H, T0, L_t, tol=1e-10)
    print("\n--- Net positivity ---")
    print(f"convolution_value         = {res.convolution_value:.12e}")
    print(f"convolution_tail_error    = {res.convolution_tail_error:.12e}")
    print(f"positive_floor_value      = {res.positive_floor_value:.12e}")
    print(f"curvature_leakage_bound   = {res.curvature_leakage_bound:.12e}")
    print(f"net_floor_minus_leakage   = {res.net_bound_floor_minus_leakage:.12e}")
    print(f"guaranteed_positive       = {res.guaranteed_positive}")
    print(f"pointwise_domination      = {res.pointwise_domination_holds}")
    print(f"interval_for_Q            = [{res.interval_for_Q[0]:.12e}, {res.interval_for_Q[1]:.12e}]")

    print("\n--- Time vs Frequency (discrete Plancherel) ---")
    comp = compare_time_freq_domains(cfg, H, T0, L_t=L_t, L_xi=L_xi, tol=1e-10)
    print(f"Q_time                    = {comp['Q_time']:.12e}")
    print(f"Q_freq                    = {comp['Q_freq']:.12e}")
    print(f"difference (time - freq)  = {comp['difference']:.12e}")


if __name__ == "__main__":
    _demo()