"""
Volume II — Kernel Stabilisation (Production Version)

This module implements the stabilised kernel framework used in
"The Analyst's Problem: A Program Toward the Riemann Hypothesis".

It is designed to be *proof-grade* in the following sense:

  • The kernel k_H(t) is defined and manipulated symbolically in a way that
    matches the analytic manuscript exactly.

  • All identities that are used later in the program (Volumes III–XII) are
    either proved in closed form or numerically verified at high precision
    as a *sanity check* of this implementation, not as a substitute for proof.

  • The role of this volume is precisely delimited:

      (V2-1) Construct an admissible test function k_H(t) for the explicit
             formula, with clean analytic properties (even, smooth, rapid
             decay).

      (V2-2) Prove and expose the exact identity
                 k_H(t) = -w_H''(t) + (4/H^2) w_H(t)
                        = (6/H^2) sech^4(t/H),

             together with its Fourier transform and norms.

      (V2-3) Establish that, for this kernel, both the time-domain function
             k_H(t) and its Fourier transform \hat{k}_H(ξ) are nonnegative,
             so that Bochner–Toeplitz positivity holds for all finite N.

      (V2-4) Provide a clean interface to Volume I (reduction) and Volume IV
             (spectral expansion): any use of this kernel in the "Analyst's
             Problem" quadratic form is mathematically consistent with the
             explicit formula and with the RH ⇔ Analyst’s Problem reduction
             established in Volume I.

This module does NOT itself claim or attempt to prove the Riemann Hypothesis.
It supplies a rigorously characterised kernel that is *used* by later volumes.
"""

from __future__ import annotations
import numpy as np
import mpmath as mp
from dataclasses import dataclass

# High-precision arithmetic for validation checks.
mp.mp.dps = 80


# ===========================================================================
# 1. Hyperbolic primitives (as in the analytic manuscript)
# ===========================================================================

def sech(x: mp.mpf) -> mp.mpf:
    """
    sech(x) = 1 / cosh(x), basic hyperbolic building block.
    """
    return 1 / mp.cosh(x)


def sech2(x: mp.mpf) -> mp.mpf:
    """
    sech^2(x), used as the base window w_H(t) in this volume.
    """
    s = sech(x)
    return s * s


def sech4(x: mp.mpf) -> mp.mpf:
    """
    sech^4(x), closed-form for the stabilised kernel k_H(t).
    """
    s2 = sech2(x)
    return s2 * s2


def tanh_(x: mp.mpf) -> mp.mpf:
    """
    tanh(x), keeping notation parallel to the analytic text.
    """
    return mp.tanh(x)


@dataclass(frozen=True)
class KernelParams:
    """
    Simple container for kernel parameters.

    H > 0 is the only shape parameter for the window w_H and kernel k_H.
    """
    H: float  # H > 0


# ===========================================================================
# 2. Base window w_H and derivatives, curvature/floor split
# ===========================================================================

def w_H(t: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    """
    Window function:

        w_H(t) = sech^2(t / H).

    This is the "raw" hyperbolic window whose curvature is analysed in Volume II.
    """
    u = t / H
    return sech2(u)


def w_H_prime(t: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    """
    First derivative of w_H(t) with respect to t:

        w_H'(t) = d/dt sech^2(u) with u = t/H
                = -2 sech^2(u) tanh(u) * (1/H).
    """
    u = t / H
    return (-2 / H) * sech2(u) * tanh_(u)


def w_H_double_prime(t: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    """
    Second derivative of w_H(t) with respect to t:

        w_H''(t) = (2/H^2) * sech^2(u) * (3 tanh^2(u) - 1),  u = t/H.

    Sign pattern:
      • Near t = 0, w_H''(t) < 0.
      • Outside a compact transition region, w_H''(t) > 0.

    This sign change in curvature is the source of the instability
    of -w_H'' as a stand-alone kernel.
    """
    u = t / H
    s2 = sech2(u)
    th = tanh_(u)
    return (2 / (H * H)) * s2 * (3 * th * th - 1)


def curvature_term(t: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    """
    The "curvature term" in the Volume II decomposition.

    In this implementation, curvature_term(t, H) is simply w_H''(t, H).
    The stabilised kernel will be formed as a curvature-plus-floor split:

        k_H(t) = - w_H''(t) + λ_* w_H(t),

    with λ_* chosen minimally to enforce global positivity.
    """
    return w_H_double_prime(t, H)


# The transition point in u where 3 tanh^2(u) - 1 = 0 ⇒ tanh^2(u) = 1/3.
TRANSITION_U = mp.atanh(1 / mp.sqrt(3))


def w_double_prime_sign_info(H: float | mp.mpf) -> dict:
    """
    Sign structure of w_H''(t):

      • For |u| < atanh(1/√3): 3 tanh^2(u) - 1 < 0 ⇒ w_H''(t) < 0.
      • For |u| > atanh(1/√3): 3 tanh^2(u) - 1 > 0 ⇒ w_H''(t) > 0.

    This diagnostic is used to confirm that the implementation matches
    the analytic sign diagram in the Volume II manuscript.
    """
    info = {
        "transition_u": TRANSITION_U,
        "transition_t": TRANSITION_U * H,
        "wpp_at_0": w_H_double_prime(0, H),
        "wpp_at_transition": w_H_double_prime(TRANSITION_U * H, H),
    }
    return info


def curvature_negative_interval(H: float | mp.mpf) -> tuple[mp.mpf, mp.mpf]:
    """
    Central interval in t where the curvature term is negative:

        { t : |t/H| < atanh(1/√3) }.

    On this interval, w_H''(t) < 0 and hence -w_H''(t) > 0.
    Outside, the sign flips.
    """
    a = -TRANSITION_U * H
    b = TRANSITION_U * H
    return a, b


# ===========================================================================
# 3. Minimal λ* = 4/H^2 and sharpness within the family -w_H'' + λ w_H
# ===========================================================================

def lambda_star(H: float | mp.mpf) -> mp.mpf:
    """
    Minimal stabilisation constant in the admissible family:

        k_λ(t) = -w_H''(t) + λ w_H(t).

    Analytically, λ_* = 4/H^2 is the exact infimum of λ such that
    k_λ(t) ≥ 0 for all t in R. The code-level uses this constant
    everywhere; numerical checks only *illustrate* the sharpness.
    """
    return mp.mpf(4) / (H * H)


def k_lambda(t: float | mp.mpf, H: float | mp.mpf, lam: float | mp.mpf) -> mp.mpf:
    """
    Generic member of the stabilisation family:

        k_λ(t) = -w_H''(t) + λ w_H(t).

    For λ = λ_*, this agrees with the closed-form sech^4 kernel.
    """
    return -w_H_double_prime(t, H) + lam * w_H(t, H)


def minimal_lambda_numeric(H: float = 1.0, t_max: float = 10.0, n_grid: int = 10001) -> float:
    """
    Diagnostic numerical approximation to the minimal λ
    such that k_λ(t) ≥ 0 on [-t_max, t_max].

    This is NOT used in any proof component; it is a sanity check
    that the implementation of w_H and w_H'' is consistent with the
    analytic claim λ_* = 4/H^2.

    Method: approximate sup_t w_H''(t) / w_H(t) over a grid.
    """
    H = float(H)
    ts = np.linspace(-t_max, t_max, n_grid)
    vals = []
    for t in ts:
        wt = float(w_H(t, H))
        if wt < 1e-20:
            continue
        ratio = float(w_H_double_prime(t, H)) / wt
        vals.append(ratio)
    return max(vals) if vals else 0.0


def finds_negative_for_lambda(H: float = 1.0, lam: float = 3.5,
                              t_max: float = 10.0, n_grid: int = 20001):
    """
    Diagnostic: search for t where k_λ(t) < 0.

    Used to illustrate sharpness: for λ slightly below λ_* we find
    a negative dip; for λ ≥ λ_* we do not.
    """
    ts = np.linspace(-t_max, t_max, n_grid)
    for t in ts:
        val = float(k_lambda(t, H, lam))
        if val < -1e-10:
            return float(t), val
    return None


def lambda_sharpness_verbose(H: float = 1.0):
    """
    Human-facing diagnostic for λ-sharpness:

        (λ*, t_neg, k_λ(t_neg))

    where λ* = 4/H^2 and λ = λ* - 0.001 is tested for negativity.

    This is included purely for transparency and debugging, not for proof.
    """
    H_mp = mp.mpf(H)
    lam_star_val = float(lambda_star(H_mp))
    lam_test = lam_star_val - 0.001
    res = finds_negative_for_lambda(H=H, lam=lam_test, t_max=10.0, n_grid=20001)
    if res is None:
        return lam_star_val, None, None
    t_neg, val_neg = res
    return lam_star_val, t_neg, val_neg


# ===========================================================================
# 4. Stabilised kernel k_H and closed form identity
# ===========================================================================

def floor_term(t: float | mp.mpf, H: float | mp.mpf, lam: float | mp.mpf) -> mp.mpf:
    """
    "Floor term" λ w_H(t) in the curvature-plus-floor split.

    In this program, we always take λ = λ_* = 4/H^2, but this function
    retains the parameter for clarity and experimentation.
    """
    return lam * w_H(t, H)


def k_H(t: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    """
    Stabilised kernel:

        k_H(t) = -w_H''(t) + (4/H^2) w_H(t).

    Analytically, Volume II proves that this can be written exactly as

        k_H(t) = (6/H^2) sech^4(t/H),

    which is strictly positive for all real t. The implementation below
    uses the curvature/floor split so that TDD can exercise both sides
    of the identity independently.
    """
    H2 = H * H
    lam_star_val = mp.mpf(4) / H2
    return -w_H_double_prime(t, H) + floor_term(t, H, lam_star_val)


def k_H_sech4_closed_form(t: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    """
    Closed-form expression:

        k_H(t) = (6/H^2) sech^4(t/H).

    This is used in tests to confirm that the curvature-based construction
    matches the algebraic identity to high precision.
    """
    u = t / H
    return (mp.mpf(6) / (H * H)) * sech4(u)


# ===========================================================================
# 5. Fourier-side symbols and Bochner positivity
# ===========================================================================

def w_H_hat(xi: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    r"""
    Fourier transform of w_H(t) = sech^2(t/H) under the convention

        \hat{f}(ξ) = ∫_{ℝ} f(t) e^{-2π i ξ t} dt.

    Explicit formula (corrected to match analytic derivation):

        \hat{w}_H(ξ) = 2π²H²ξ / sinh(π²Hξ),

    with the limiting value \hat{w}_H(0) = 2H obtained from the
    first-order expansion sinh(z) ~ z at z = 0.
    """
    xi = mp.mpf(xi)
    H = mp.mpf(H)
    if xi == 0:
        return 2 * H
    return 2 * mp.pi**2 * H**2 * xi / mp.sinh(mp.pi**2 * H * xi)


def k_H_hat(xi: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    r"""
    Fourier transform of k_H under the same convention.

    Using k_H = -w_H'' + (4/H^2) w_H and \widehat{f''}(ξ) = -(2π ξ)^2 \hat{f}(ξ),
    we obtain

        \hat{k}_H(ξ) = ((2π ξ)^2 + 4/H^2) \hat{w}_H(ξ).

    Since \hat{w}_H(ξ) ≥ 0 and (2π ξ)^2 + 4/H^2 > 0, we have \hat{k}_H(ξ) ≥ 0
    for all real ξ. This is the Fourier-side nonnegativity needed for the
    Bochner → Toeplitz PSD step.
    """
    xi = mp.mpf(xi)
    H = mp.mpf(H)
    lam = lambda_star(H)
    return ((2 * mp.pi * xi) ** 2 + lam) * w_H_hat(xi, H)


def fourier_symbol_nonnegative(H: float = 1.0, xi_max: float = 20.0, n_grid: int = 4001) -> bool:
    """
    Grid-based sanity check that \hat{k}_H(ξ) is nonnegative.

    Analytically, this is guaranteed by the formula in k_H_hat.
    Numerically, this function checks for violations above a small tolerance.
    """
    xis = np.linspace(-xi_max, xi_max, n_grid)
    for xi in xis:
        val = float(k_H_hat(xi, H))
        if val < -1e-10:
            return False
    return True


# ===========================================================================
# 6. Bochner → Toeplitz PSD (finite N)
# ===========================================================================

def toeplitz_matrix_from_kernel(log_ns: np.ndarray, H: float) -> np.ndarray:
    """
    Build the Toeplitz Gram matrix

        K_{mn} = k_H(log m - log n)

    over a finite log-integer grid.

    By Bochner's theorem and the nonnegativity of \hat{k}_H, this matrix
    is positive semi-definite (PSD) for every finite N. This function is
    used heavily in Volumes III–IV and in the validation suite.
    """
    N = len(log_ns)
    K = np.empty((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            t = log_ns[i] - log_ns[j]
            K[i, j] = float(k_H(t, H))
    return K


def is_psd_matrix(A: np.ndarray, tol: float = 1e-9) -> bool:
    """
    Numerical PSD check via eigenvalues.

    This is a validation tool only; the mathematical PSD property follows
    from the Fourier representation and Bochner's theorem.
    """
    w = np.linalg.eigvalsh(A)
    return np.all(w >= -tol)


def bochner_psd_check(N: int = 20, H: float = 1.0) -> bool:
    """
    Convenience wrapper: check PSD of the Toeplitz Gram matrix for
    ns = 1,...,N.

    Used in diagnostics to confirm that the implementation is consistent
    with the analytic Bochner argument.
    """
    ns = np.arange(1, N + 1, dtype=float)
    log_ns = np.log(ns)
    K = toeplitz_matrix_from_kernel(log_ns, H)
    K = 0.5 * (K + K.T)
    return is_psd_matrix(K)


# ===========================================================================
# 7. L^1 / L^2 norms and decay (analytic constants)
# ===========================================================================

def k_H_L1(H: float | mp.mpf) -> mp.mpf:
    r"""
    L^1 norm:

        ∫_{ℝ} k_H(t) dt = (6/H^2) ∫ sech^4(t/H) dt.

    Substituting u = t/H ⇒ dt = H du, and using
        ∫_{ℝ} sech^4(u) du = 4/3,
    we obtain the closed form

        ∫ k_H = 8/H.

    This constant is used in tail bounds in later volumes.
    """
    H = mp.mpf(H)
    return mp.mpf(8) / H


def k_H_L2_squared(H: float | mp.mpf) -> mp.mpf:
    r"""
    L^2 norm:

        ∫_{ℝ} k_H(t)^2 dt.

    Analytically,

        ∫ k_H(t)^2 dt = (1152 / 35) * H^{-3}.

    The code returns this closed form, which is matched against numerical
    quadrature in the validation suite.
    """
    H = mp.mpf(H)
    return (mp.mpf(1152) / mp.mpf(35)) / (H ** 3)


def k_H_decay_sample(H: float = 1.0, t_values: list[float] | None = None) -> list[tuple[float, float]]:
    """
    Sample k_H(t) at a small set of multiples of H to illustrate decay.

    As |t| grows, k_H(t) decays like e^{-4|t|/H}, which is used in tail
    estimates in the positivity and truncation analyses.
    """
    if t_values is None:
        t_values = [0.0, 1.0 * H, 2.0 * H, 3.0 * H, 4.0 * H]
    out = []
    for t in t_values:
        val = float(k_H(t, H))
        out.append((float(t), val))
    return out


# ===========================================================================
# 8. Classical comparators: Gaussian and Fejér-type kernels
#    (used only for contrast and diagnostics)
# ===========================================================================

def w_G_H(t: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    """
    Gaussian window:

        w_G,H(t) = exp(-(t/H)^2).

    Included as a classical comparator to illustrate that dual-positivity
    (k ≥ 0 and \hat{k} ≥ 0) is special: the Gaussian-based k_G can have
    time-domain sign changes even though its Fourier symbol is positive.
    """
    t = mp.mpf(t)
    H = mp.mpf(H)
    return mp.e ** (-(t / H) ** 2)


def w_G_H_hat(xi: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    """
    Fourier transform of the Gaussian window under the same convention.
    """
    xi = mp.mpf(xi)
    H = mp.mpf(H)
    return H * mp.sqrt(mp.pi) * mp.e ** (-(mp.pi * H * xi) ** 2)


def k_G_H_hat(xi: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    """
    Analogue of \hat{k}_H for the Gaussian window.
    """
    xi = mp.mpf(xi)
    H = mp.mpf(H)
    lam = lambda_star(H)
    return ((2 * mp.pi * xi) ** 2 + lam) * w_G_H_hat(xi, H)


def k_G_H(t: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    """
    Time-domain kernel corresponding to w_G via the same construction
    used for w_H, obtained by inverse Fourier transform of k_G_H_hat.

    This is evaluated numerically and used only for illustrative comparison
    (e.g. to show that k_G can become negative at moderate t).
    """
    t = mp.mpf(t)
    H = mp.mpf(H)
    lam = lambda_star(H)

    def integrand(xi: mp.mpf) -> mp.mpf:
        xi = mp.mpf(xi)
        return ((2 * mp.pi * xi) ** 2 + lam) * w_G_H_hat(xi, H) * mp.e ** (2j * mp.pi * xi * t)

    xi_max = mp.mpf(5) / H
    val = mp.quad(lambda y: integrand(y), [-xi_max, xi_max])
    return val


def w_F_H_hat(xi: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    """
    Fejér-type hat: compactly supported in frequency.

    This comparator demonstrates how different window choices interact
    with the k_λ construction; it is not used in the core RH pathway.
    """
    xi = mp.mpf(xi)
    H = mp.mpf(H)
    ax = abs(xi)
    if ax >= H:
        return mp.mpf(0)
    return mp.mpf(1) - ax / H


def w_F_H(t: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    """
    Time-domain Fejér window via inverse Fourier transform of w_F_H_hat.
    """
    t = mp.mpf(t)
    H = mp.mpf(H)

    def integrand(xi: mp.mpf) -> mp.mpf:
        w_hat = w_F_H_hat(xi, H)
        return w_hat * mp.e ** (-2j * mp.pi * xi * t)

    val = mp.quad(lambda y: integrand(y), [-H, H])
    return val


def k_F_H_hat(xi: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    """
    Gaussian-style k-hat built on the Fejér window.
    """
    xi = mp.mpf(xi)
    H = mp.mpf(H)
    lam = lambda_star(H)
    return ((2 * mp.pi * xi) ** 2 + lam) * w_F_H_hat(xi, H)


def k_F_H(t: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    """
    Time-domain Fejér-based kernel via inverse Fourier transform of k_F_H_hat.
    """
    t = mp.mpf(t)
    H = mp.mpf(H)

    def integrand(xi: mp.mpf) -> mp.mpf:
        return k_F_H_hat(xi, H) * mp.e ** (2j * mp.pi * xi * t)

    val = mp.quad(lambda y: integrand(y), [-H, H])
    return val


def fourier_symbol_nonnegative_generic(
    symbol_func,
    H: float = 1.0,
    xi_max: float = 20.0,
    n_grid: int = 4001,
    tol: float = 1e-10,
) -> bool:
    """
    Generic Fourier-symbol nonnegativity check for comparator kernels.
    """
    xis = np.linspace(-xi_max, xi_max, n_grid)
    for xi in xis:
        val = symbol_func(mp.mpf(xi), mp.mpf(H))
        val_re = float(mp.re(val))
        if val_re < -tol:
            return False
    return True


def toeplitz_matrix_from_kernel_generic(
    kernel_func,
    log_ns: np.ndarray,
    H: float,
) -> np.ndarray:
    """
    Generic Toeplitz matrix builder from an arbitrary kernel_func.
    """
    N = len(log_ns)
    K = np.empty((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            t = log_ns[i] - log_ns[j]
            K[i, j] = float(mp.re(kernel_func(mp.mpf(t), mp.mpf(H))))
    return K


def bochner_psd_check_generic(
    kernel_func,
    N: int = 20,
    H: float = 1.0,
    tol: float = 1e-9,
) -> bool:
    """
    Generic PSD check using a kernel_func, used only for comparators.
    """
    ns = np.arange(1, N + 1, dtype=float)
    log_ns = np.log(ns)
    K = toeplitz_matrix_from_kernel_generic(kernel_func, log_ns, H)
    K = 0.5 * (K + K.T)
    return is_psd_matrix(K, tol=tol)


# ===========================================================================
# 9. Interface theorems to Volume I (reduction) and Volume IV (spectral form)
# ===========================================================================

def volume_ii_interface_summary(H: float = 1.0) -> dict:
    """
    Programmatic summary of the "interface theorems" of Volume II.

    This function does not perform proof; it encodes, in a structured way,
    the key properties that the analytic manuscript establishes and that
    later volumes rely on:

      (IF-1) Kernel definition and closed form.
      (IF-2) Admissibility as a test function for Weil's explicit formula:
             even, smooth, rapidly decaying.
      (IF-3) Dual positivity: k_H(t) ≥ 0, \hat{k}_H(ξ) ≥ 0.
      (IF-4) Exact L^1 / L^2 norms.
      (IF-5) Stabilisation constant λ_*.

    The validation suite uses this to cross-check numerical behaviour
    against the claimed analytic properties.
    """
    H_mp = mp.mpf(H)

    properties = {
        "kernel_definition": "k_H(t) = -w_H''(t) + (4/H^2) w_H(t) = (6/H^2) sech^4(t/H)",
        "admissibility": {
            "even": True,
            "smooth": True,
            "rapid_decay": True,
        },
        "positivity": {
            # Sample points only illustrate positivity; the proof is analytic.
            "time_domain_sample": [float(k_H(t, H_mp)) for t in [0, H_mp, 2 * H_mp]],
            "fourier_symbol_nonnegative": fourier_symbol_nonnegative(H=H),
        },
        "l1_norm": float(k_H_L1(H_mp)),
        "l2_norm_squared": float(k_H_L2_squared(H_mp)),
        "lambda_star": float(lambda_star(H_mp)),
    }

    return properties


def spectral_quadratic_form(
    a: np.ndarray,
    xis: np.ndarray,
    H: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Volume II → Volume IV spectral interface.

    For a coefficient vector a_n and frequency grid {ξ_j}, define the
    Dirichlet wave

        S(ξ) = Σ_{n=1}^N a_n e^{-i ξ log n}.

    For the kernel k_H we have the spectral representation

        Q_H[a] = ∫ \hat{k}_H(ξ) |S(ξ)|^2 dξ,

    which is the starting point for the Parseval-type manipulations
    in Volume IV and the positivity operator arguments in later volumes.

    This helper returns:
      • S_vals: sampled S(ξ_j),
      • weights: the corresponding nonnegative weights \hat{k}_H(ξ_j),

    leaving the actual quadrature scheme to the calling volume.
    """
    H_mp = mp.mpf(H)
    ns = np.arange(1, len(a) + 1, dtype=float)
    log_ns = np.log(ns)

    S_vals = []
    weights = []
    for xi in xis:
        xi_mp = mp.mpf(xi)
        S_xi = mp.mpf("0")
        for coeff, ln_n in zip(a, log_ns):
            S_xi += mp.mpf(coeff) * mp.e ** (-1j * xi_mp * mp.mpf(ln_n))
        S_vals.append(complex(S_xi))
        weights.append(complex(k_H_hat(xi_mp, H_mp)))

    return np.array(S_vals, dtype=complex), np.array(weights, dtype=complex)


# ===========================================================================
# 10. Diagnostics (not used by proof logic, but essential for validation)
# ===========================================================================

def demo_volume_ii_diagnostics():
    """
    Run a battery of diagnostics illustrating the main identities and
    properties of Volume II.

    This function is intended for manual inspection and regression tests.
    It does not participate in any formal proof step; its role is to
    certify that the implementation of the analytic identities is faithful.
    """
    H = 1.0
    print("=== Volume II: Kernel Stabilisation Diagnostics (H=1) ===")

    # Identity check: curvature-based construction vs sech^4 closed form.
    ts = np.linspace(-5, 5, 41)
    max_err = 0.0
    for t in ts:
        v1 = float(k_H(t, H))
        v2 = float(k_H_sech4_closed_form(t, H))
        max_err = max(max_err, abs(v1 - v2))
    print(f"[sech] Max identity error k_H vs sech^4 closed form: {max_err:.3e}")

    # Curvature sign structure.
    info = w_double_prime_sign_info(H)
    print("[sech] Transition u =", info["transition_u"])
    print("[sech] Transition t =", info["transition_t"])
    a, b = curvature_negative_interval(H)
    print("[sech] Curvature negative interval (t):", (a, b))

    # Numerical λ* diagnostic.
    lam_num = minimal_lambda_numeric(H=H, t_max=5.0, n_grid=8001)
    print(f"[sech] Numeric minimal λ on [-5,5]: {lam_num:.6f}, λ* = {float(lambda_star(H)):.6f}")

    # Fourier symbol nonnegativity.
    ok_hat = fourier_symbol_nonnegative(H=H, xi_max=10.0, n_grid=4001)
    print("[sech] Fourier symbol nonnegative on grid?", ok_hat)

    # Toeplitz PSD check.
    ok_psd = bochner_psd_check(N=20, H=H)
    print("[sech] Toeplitz Gram PSD (N=20)?", ok_psd)

    # Norms and decay samples.
    print("[sech] ∫ k_H =", k_H_L1(H))
    print("[sech] ∫ k_H^2 =", k_H_L2_squared(H))
    print("[sech] Decay samples:")
    for t, val in k_H_decay_sample(H=H):
        print(f"  t={t:5.2f}, k_H(t)={val:.3e}")

    # λ-sharpness illustration.
    lam_star_val, t_neg, val_neg = lambda_sharpness_verbose(H=H)
    print(f"[sech] Lambda sharpness (H=1): λ*={lam_star_val:.6f}, t_neg={t_neg}, k_λ(t_neg)={val_neg}")

    # Gaussian comparator.
    print("\n=== Classical Gaussian Comparison ===")
    ok_hat_gauss = fourier_symbol_nonnegative_generic(
        k_G_H_hat, H=H, xi_max=10.0, n_grid=4001
    )
    print("[gauss] Fourier symbol nonnegative on grid?", ok_hat_gauss)

    ok_psd_gauss = bochner_psd_check_generic(
        k_G_H, N=20, H=H, tol=1e-7
    )
    print("[gauss] Toeplitz Gram PSD (N=20)?", ok_psd_gauss)

    print("[gauss] Decay samples (numerical k_G_H):")
    for t in [0.0, 1.0 * H, 2.0 * H, 3.0 * H]:
        val = k_G_H(t, H)
        print(f"  t={t:5.2f}, Re k_G_H(t)={float(mp.re(val)):.3e}")

    # Fejér comparator.
    print("\n=== Classical Fejér-Type Comparison ===")
    ok_hat_fej = fourier_symbol_nonnegative_generic(
        k_F_H_hat, H=H, xi_max=5.0, n_grid=2001
    )
    print("[fejer] Fourier symbol nonnegative on grid?", ok_hat_fej)

    ok_psd_fej = bochner_psd_check_generic(
        k_F_H, N=15, H=H, tol=1e-6
    )
    print("[fejer] Toeplitz Gram PSD (N=15)?", ok_psd_fej)

    print("[fejer] Decay samples (numerical k_F_H):")
    for t in [0.0, 0.5 * H, 1.0 * H, 2.0 * H]:
        val = k_F_H(t, H)
        print(f"  t={t:5.2f}, Re k_F_H(t)={float(mp.re(val)):.3e}")


if __name__ == "__main__":
    demo_volume_ii_diagnostics()