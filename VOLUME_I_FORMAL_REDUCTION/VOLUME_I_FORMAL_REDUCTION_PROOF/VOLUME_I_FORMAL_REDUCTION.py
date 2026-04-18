#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FORMAL_REDUCTION_QH_ANALYST_PROBLEM.py
#
# Volume I — Formal Reduction (Production Version)
#
# This module encodes the *reduced* finite-dimensional inequality
# (“The Analyst’s Problem”) associated to the Riemann Hypothesis (RH),
# in a minimal, dependency‑free Python form.
#
# It is aligned with the reframed kernel framework of Volume II
# (the dual‑positive sech⁴ kernel k_H) and with the LRM audit.
#
# --------------------------------------------------------------------
# FORMAL ROLE OF THIS MODULE (VOLUME I INTERFACE)
# --------------------------------------------------------------------
#
# (V1-1) Admissible test functions:
#
#   Volume I defines an admissible class 𝒦 of even, smooth, rapidly
#   decaying test functions k: ℝ → ℝ, suitable for the Weil explicit
#   formula and the Parseval-type bridge used in the reduction.
#
# (V1-2) Kernel-agnostic reduction:
#
#   For any k ∈ 𝒦, one can form a Toeplitz quadratic form
#
#       Q_H^{(k)}(N, T0)
#         = ∑_{m,n ≤ N} x_m x_n (m/n)^(-i T0) k̂(log m − log n),
#
#   with x_n = n^{-σ}, σ = 1/2, and k̂ the Fourier transform (on log‑scale)
#   determined by the explicit formula. The Volume I reduction theorem
#   states:
#
#       RH  ⇔  Q_H^{(k)}(N, T0) ≥ 0
#
#   for all admissible parameters (H, N, T0) in the regime described in
#   the manuscript, uniformly, provided k ∈ 𝒦.
#
# (V1-3) Choice of canonical kernel:
#
#   Volume II proves that
#
#       k_H(t) = (6/H²) sech⁴(t/H)
#
#   is admissible (even, C^∞, rapid decay) and *dual‑positive*:
#   k_H(t) ≥ 0 for all t, and its Fourier symbol k̂_H(ω) ≥ 0 for all ω.
#   In particular, k_H ∈ 𝒦. We therefore *fix* this kernel as the
#   canonical choice for the Analyst’s Problem:
#
#       Q_H(N, T0) := Q_H^{(k_H)}(N, T0).
#
# (V1-4) Time-domain representation:
#
#   Using the Dirichlet polynomial
#
#       D_N(σ, T) = ∑_{n=1}^N n^{-σ} e^{-i T log n},
#
#   the same quadratic form can be written as a convolution:
#
#       Q_H(N, T0)
#         = ∫_{ℝ} k_H(t) |D_N(σ, T0 + t)|² dt
#         =: F̄₂(T0, H; N).
#
#   The “Parseval bridge” (proved analytically in Volume I, validated
#   numerically here) identifies the Toeplitz sum and the integral.
#
# (V1-5) Analyst’s Problem (final statement):
#
#   The Analyst’s Problem is:
#
#       Prove that Q_H(N, T0) ≥ 0
#
#   for all admissible H > 0, N ∈ ℕ, and T0 ∈ ℝ, using the canonical
#   kernel k_H(t) = (6/H²) sech⁴(t/H).
#
#   Volume I shows:
#
#       RH  ⇔  (Analyst’s Problem with k_H).
#
# --------------------------------------------------------------------
# SCOPE OF THIS FILE
# --------------------------------------------------------------------
#
# • Implements k_H and k̂_H in the simplified Fourier convention used
#   in the “Tier‑32 engine” (ω instead of ξ, and w_H hat as in the
#   original code).
#
# • Implements the Dirichlet polynomial D_N, the time‑domain integral
#   F̄₂(T0, H; N), and the Toeplitz form in ω.
#
# • Provides a decomposition Q_H = M1 + Cross, where M1 is the diagonal
#   term and Cross is the off‑diagonal interference term.
#
# • Exposes a “FormalReduction” API used by the validation suite and
#   later volumes.
#
# This module does NOT itself prove RH; it encodes the finite object
# to which RH has been reduced and provides a numerically verified
# Parseval bridge between its equivalent representations.
#

import math
import cmath
from typing import List, Tuple, Sequence


##############################
# 1. BASIC SPECIAL FUNCTIONS #
##############################

def sech(x: float) -> float:
    """
    sech(x) = 1 / cosh(x).

    Hyperbolic base function used in the definition of w_H and k_H.
    """
    return 1.0 / math.cosh(x)


def sech2(x: float) -> float:
    """
    sech²(x), used as the base window w_H(t) = sech²(t/H).
    """
    c = math.cosh(x)
    return 1.0 / (c * c)


def sech4(x: float) -> float:
    """
    sech⁴(x), giving the closed form of the stabilised kernel k_H in t/H.
    """
    s2 = sech2(x)
    return s2 * s2


def tanh(x: float) -> float:
    """
    tanh(x), used only in legacy curvature formulas.
    """
    return math.tanh(x)


def k_H_time(t: float, H: float) -> float:
    """
    Time-domain kernel alias for the canonical k_H:

        k_H(t) = g_H_sech4(t, H) = (6/H²) sech⁴(t/H).

    Exposed as a named alias for clarity where the Volume I manuscript
    refers to “k_H(t) in the time domain”.
    """
    return g_H_sech4(t, H)


########################################
# 1b. LEGACY Λ_H API (FOR TDD IMPORT)  #
########################################

def Lambda_H_tau(tau: float, H: float) -> float:
    """
    Legacy curvature window Λ_H used in earlier drafts:

        Λ_H(τ) = 2π sech²(τ/H).

    Kept ONLY for API compatibility with the historical validation suite.
    It is not used in the final second-moment bridge, which is built on
    the stabilised kernel

        g_{λ*}(t) = (6/H²) sech⁴(t/H) = k_H(t).
    """
    u = tau / H
    return 2.0 * math.pi * sech2(u)


def Lambda_H_dd_tau(tau: float, H: float) -> float:
    r"""
    Second derivative of the legacy Λ_H with respect to τ:

        Λ_H''(τ) = (2π/H²) sech²(τ/H) [4 − 6 sech²(τ/H)].

    This was part of the earlier curvature-based framing and is preserved
    solely for regression and comparison. It plays no role in the final
    reduction, which uses the dual-positive k_H.
    """
    u = tau / H
    s2 = sech2(u)
    return (2.0 * math.pi / (H * H)) * s2 * (4.0 - 6.0 * s2)


##############################
# 2. CORE KERNELS (GROUND TRUTH)
##############################

def w_H_time(t: float, H: float) -> float:
    """
    Base window:

        w_H(t) = sech²(t/H).

    This is the same w_H that appears in the Volume II decomposition
    of k_H as -w_H'' + λ* w_H.
    """
    return sech2(t / H)


def g_H_sech4(t: float, H: float) -> float:
    r"""
    Stabilised kernel in the time domain:

        g_{λ*}(t) = k_H(t) = (6/H²) sech⁴(t/H).

    This is the canonical choice of test function for the Analyst’s
    Problem, proven admissible and dual-positive in Volume II.
    """
    return (6.0 / (H * H)) * sech4(t / H)


def fourier_w_H(omega: float, H: float) -> float:
    r"""
    Fourier transform of w_H(t) = sech²(t/H) in the *engine convention*:

        ŵ_H(ω) = π H² ω / sinh(π H ω / 2),

    with the convention that ŵ_H(0) = 2H.

    Note: this is a scaled version of the analytic convention used in
    Volume II. The mapping is carefully tracked so that

        k̂_H(ω) = (ω² + λ*) ŵ_H(ω)

    still holds in this ω-variable normalisation.
    """
    if abs(omega) < 1e-15:
        return 2.0 * H
    arg = math.pi * H * omega / 2.0
    if abs(arg) > 700.0:
        # Exponential suppression for large |arg|.
        return 0.0
    return (math.pi * H * H * omega) / math.sinh(arg)


def lambda_star(H: float) -> float:
    """
    Stabilisation constant:

        λ* = 4 / H².

    Volume II proves this is the minimal λ in the family -w_H'' + λ w_H
    that yields a globally nonnegative kernel. Here we use it as a
    fixed constant in the definition of k̂_H.
    """
    return 4.0 / (H * H)


def k_H_hat(omega: float, H: float) -> float:
    r"""
    Frequency-side kernel symbol:

        k̂_H(ω) = (ω² + λ*) · ŵ_H(ω),

    where ŵ_H is the w_H transform in this engine convention.

    In this normalisation k̂_H(ω) ≥ 0 for all real ω, implementing the
    Bochner positivity needed for the Toeplitz PSD property.

    This matches the Volume II identity up to the change of variable
    between ξ (analytic) and ω (engine).
    """
    lam = lambda_star(H)
    wh = fourier_w_H(omega, H)
    return (omega * omega + lam) * wh


##############################
# 3. DIRICHLET POLYNOMIAL    #
##############################

def dirichlet_S_N(T: float, N: int, sigma: float = 0.5) -> complex:
    r"""
    Dirichlet polynomial:

        S_N(σ, T) = ∑_{n=1}^N n^{-σ} e^{-i T ln n}.

    In the Analyst’s Problem, σ = 1/2 is the primary case, but σ is
    kept as a parameter for generality and testing.
    """
    s = 0.0 + 0.0j
    for n in range(1, N + 1):
        ln_n = math.log(n)
        s += n ** (-sigma) * cmath.exp(-1j * T * ln_n)
    return s


def dirichlet_S_N_derivatives(T: float, N: int, sigma: float = 0.5):
    r"""
    (S_N, S_N', S_N'') with respect to σ.

    These derivatives are used in the F₂(σ, T) functional:

        F₂(σ, T) = 2 |S_N'|² + 2 Re(S_N'' conj(S_N)),

    which appears in legacy tests. The formal Volume I reduction is
    stated in terms of |S_N|² rather than σ-derivatives; this function
    is maintained to keep the validation suite intact.
    """
    S0 = 0.0 + 0.0j
    S1 = 0.0 + 0.0j
    S2 = 0.0 + 0.0j
    for n in range(1, N + 1):
        ln_n = math.log(n)
        z = n ** (-sigma) * cmath.exp(-1j * T * ln_n)
        S0 += z
        S1 -= ln_n * z
        S2 += (ln_n ** 2) * z
    return S0, S1, S2


def F2_sigma_T(T: float, N: int, sigma: float = 0.5) -> float:
    r"""
    Legacy σ‑second derivative functional:

        F₂(σ, T) = 2|S_N'|² + 2 Re(S_N'' conj(S_N)).

    Used only in tests that explicitly reference it. The main reduction
    works with the second moment |D_N|², but this auxiliary quantity is
    preserved for backwards compatibility.
    """
    S0, S1, S2 = dirichlet_S_N_derivatives(T, N, sigma=sigma)
    return 2.0 * (abs(S1) ** 2) + 2.0 * (S2 * S0.conjugate()).real


####################################
# 4. SECOND-MOMENT BRIDGE OBJECT  #
####################################

def D_N(T: float, N: int, sigma: float = 0.5) -> complex:
    """
    Alias:

        D_N(T) = S_N(σ, T),

    with σ = 1/2 by default. This notation matches the Volume I
    manuscript where the Dirichlet polynomial is denoted D_N.
    """
    return dirichlet_S_N(T, N, sigma=sigma)


def second_moment_integrand(t: float,
                            T0: float,
                            H: float,
                            N: int,
                            sigma: float = 0.5) -> float:
    r"""
    Second-moment integrand:

        k_H(t) |D_N(T0 + t)|²
      = g_{λ*}(t, H) |D_N(T0 + t)|².

    This is the time-domain integrand whose integral is F̄₂(T0, H; N).
    """
    g = g_H_sech4(t, H)
    D = D_N(T0 + t, N, sigma=sigma)
    return g * (D.real * D.real + D.imag * D.imag)


def curvature_F2_bar(T0: float,
                     H: float,
                     N: int,
                     sigma: float = 0.5,
                     tau_min: float = None,
                     tau_max: float = None,
                     num_steps: int = 8001) -> float:
    r"""
    Time-domain second moment:

        F̄₂(T0, H; N) = ∫ k_H(t) |D_N(T0 + t)|² dt,

    implemented numerically via Simpson’s rule.

    This is the same object that the Toeplitz form computes via the
    Parseval bridge. The integral limits [tau_min, tau_max] are chosen
    wide enough (typically ±15H) that truncation error is negligible
    compared to the target tolerance in the validation suite.
    """
    if tau_min is None:
        tau_min = -15.0 * H
    if tau_max is None:
        tau_max = 15.0 * H
    if num_steps % 2 == 0:
        num_steps += 1

    dt = (tau_max - tau_min) / (num_steps - 1)
    s = 0.0
    for k in range(num_steps):
        t = tau_min + k * dt
        if k == 0 or k == num_steps - 1:
            w = 1.0
        elif k % 2 == 1:
            w = 4.0
        else:
            w = 2.0
        s += w * second_moment_integrand(t, T0, H, N, sigma=sigma)
    return s * dt / 3.0


########################################
# 5. TOEPLITZ FORM FOR SECOND MOMENT  #
########################################

def physical_vector_x(N: int, sigma: float = 0.5) -> List[float]:
    """
    Physical coefficient vector:

        x_n = n^{-σ},  1 ≤ n ≤ N.

    This is the coefficient vector used in the Toeplitz quadratic form
    representation Q_H(N, T0).
    """
    return [n ** (-sigma) for n in range(1, N + 1)]


def phased_quadratic_form(N: int,
                          H: float,
                          T0: float,
                          sigma: float = 0.5) -> float:
    r"""
    Toeplitz form (frequency-side representation):

        F̃₂(T0; H, N) =
          ∑_{m,n ≤ N} (mn)^{-σ} (m/n)^{-i T0} k̂_H(log m − log n),

    where k̂_H is defined in the engine ω-convention.

    This is the finite quadratic form Q_H(N, T0) appearing in the
    Analyst’s Problem, expressed via k̂_H and the x_n = n^{-σ}.
    """
    T0 = float(T0)
    s_total = 0.0
    for m in range(1, N + 1):
        ln_m = math.log(m)
        xm = m ** (-sigma)
        for n in range(1, N + 1):
            ln_n = math.log(n)
            xn = n ** (-sigma)
            diff = ln_m - ln_n
            g_hat = k_H_hat(diff, H)
            # (m/n)^(-i T0) = exp(-i T0 (ln m - ln n))
            phase = cmath.exp(-1j * T0 * diff)
            s_total += xm * xn * g_hat * phase
    return s_total.real


##############################
# 6. PARSEVAL BRIDGE         #
##############################

def parseval_bridge_certificate(T0: float,
                                H: float,
                                N: int,
                                sigma: float = 0.5,
                                tau_min: float = None,
                                tau_max: float = None,
                                num_steps: int = 8001) -> float:
    r"""
    Parseval bridge discrepancy:

        Δ(T0, H; N) = F̄₂(T0, H; N) − F̃₂(T0; H, N).

    In the Volume I manuscript, the equality F̄₂ = F̃₂ is proved
    analytically; here we compute Δ numerically as a *certificate*
    that the code faithfully implements that identity.
    """
    F2_int = curvature_F2_bar(T0, H, N, sigma=sigma,
                              tau_min=tau_min, tau_max=tau_max,
                              num_steps=num_steps)
    F2_toeplitz = phased_quadratic_form(N, H, T0, sigma=sigma)
    return F2_int - F2_toeplitz


def parseval_identity_residual(T0: float,
                               H: float,
                               N: int,
                               sigma: float = 0.5,
                               tau_min: float = None,
                               tau_max: float = None,
                               num_steps: int = 8001) -> float:
    """
    Absolute Parseval residual:

        |F̄₂(T0, H; N) − F̃₂(T0; H, N)|.

    The validation suite expects this to be small (e.g. < 1e‑2)
    over its parameter grid, confirming the numerical implementation
    of the analytic Parseval bridge.
    """
    return abs(parseval_bridge_certificate(T0, H, N, sigma=sigma,
                                           tau_min=tau_min,
                                           tau_max=tau_max,
                                           num_steps=num_steps))


##############################
# 7. M1 / CROSS STRUCTURE    #
##############################

def M1_diagonal_term(N: int, H: float, sigma: float = 0.5) -> float:
    r"""
    Diagonal part of the Toeplitz form:

        M1 = ∑_{n ≤ N} n^{-2σ} k̂_H(0).

    In this ω-convention, ŵ_H(0) = 2H and λ* = 4/H², so

        k̂_H(0) = λ* · ŵ_H(0) = (4/H²) · (2H) = 8/H.

    Hence

        M1 = (8/H) ∑_{n ≤ N} n^{-2σ}.
    """
    lam = lambda_star(H)
    g0 = lam * fourier_w_H(0.0, H)   # = 8/H
    acc = 0.0
    for n in range(1, N + 1):
        x = n ** (-sigma)
        acc += x * x
    return g0 * acc


def cross_offdiagonal_term(N: int,
                           H: float,
                           T0: float,
                           sigma: float = 0.5) -> float:
    r"""
    Off-diagonal contribution to the Toeplitz form:

        Cross(T0) = ∑_{m≠n} (mn)^{-σ} (m/n)^{-i T0} k̂_H(log m − log n).

    Together with M1, this reconstructs the full Toeplitz quadratic form:

        Q_H(N, T0) = M1 + Cross(T0).
    """
    T0 = float(T0)
    total = 0.0 + 0.0j
    for m in range(1, N + 1):
        ln_m = math.log(m)
        xm = m ** (-sigma)
        for n in range(1, N + 1):
            if m == n:
                continue
            ln_n = math.log(n)
            xn = n ** (-sigma)
            diff = ln_m - ln_n
            g_hat = k_H_hat(diff, H)
            phase = cmath.exp(-1j * T0 * diff)
            total += xm * xn * g_hat * phase
    return total.real


def QH_from_M1_and_cross(N: int,
                         H: float,
                         T0: float,
                         sigma: float = 0.5) -> float:
    """
    Full Toeplitz form reconstructed as:

        Q_H(N, T0) = M1 + Cross(T0).

    This is the quantity that the Analyst’s Problem requires to be
    nonnegative for all admissible parameters.
    """
    return M1_diagonal_term(N, H, sigma=sigma) + \
        cross_offdiagonal_term(N, H, T0, sigma=sigma)


def absolute_cross_term(N: int,
                        H: float,
                        sigma: float = 0.5) -> float:
    r"""
    Absolute off-diagonal bound:

        AbsoluteCross = ∑_{m≠n} |(mn)^{-σ}| · |k̂_H(log m − log n)|.

    This provides a T0‑uniform bound on |Cross(T0)|. It is crude
    (it ignores phase cancellation) but useful in diagnostics and
    in bounding worst-case leakage.
    """
    total = 0.0
    for m in range(1, N + 1):
        ln_m = math.log(m)
        xm = abs(m ** (-sigma))
        for n in range(1, N + 1):
            if m == n:
                continue
            ln_n = math.log(n)
            xn = abs(n ** (-sigma))
            diff = ln_m - ln_n
            g_hat = abs(k_H_hat(diff, H))
            total += xm * xn * g_hat
    return total


def C_ratio(N: int, H: float, sigma: float = 0.5) -> float:
    r"""
    Relative off-diagonal magnitude:

        C = AbsoluteCross / M1.

    This dimensionless ratio is one diagnostic for how large the
    off-diagonal sector can be relative to the main diagonal mass.
    """
    M1 = M1_diagonal_term(N, H, sigma=sigma)
    if M1 <= 0.0:
        return float('inf')
    AC = absolute_cross_term(N, H, sigma=sigma)
    return AC / M1


##############################
# 8. PSD KERNEL CHECK        #
##############################

def build_toeplitz_matrix(N: int, H: float) -> List[List[float]]:
    r"""
    Phase‑free Toeplitz matrix with entries

        M_{mn} = k̂_H(log m − log n),

    i.e. the Gram matrix associated with the kernel in the frequency
    picture, evaluated on the log-integer grid.

    This matrix is real symmetric by construction and, analytically,
    is positive semi-definite by Bochner’s theorem and the non-negativity
    of k̂_H.
    """
    M = [[0.0 for _ in range(N)] for _ in range(N)]
    for m in range(1, N + 1):
        ln_m = math.log(m)
        for n in range(1, N + 1):
            ln_n = math.log(n)
            diff = ln_m - ln_n
            M[m - 1][n - 1] = k_H_hat(diff, H)
    # Symmetrize numerically to remove roundoff asymmetry.
    for i in range(N):
        for j in range(i + 1, N):
            v = 0.5 * (M[i][j] + M[j][i])
            M[i][j] = v
            M[j][i] = v
    return M


def toeplitz_quadratic_form(M: Sequence[Sequence[float]],
                            x: Sequence[float]) -> float:
    """
    x^T M x for a real symmetric matrix M.

    Included as a low-level helper to test PSD behaviour directly
    with arbitrary vectors x.
    """
    N = len(x)
    total = 0.0
    for i in range(N):
        row = M[i]
        xi = x[i]
        for j in range(N):
            total += xi * row[j] * x[j]
    return total


def check_kernel_positive_definite(N: int,
                                   H: float) -> Tuple[float, float]:
    r"""
    Approximate (λ_min, λ_max) for the phase-free Toeplitz matrix via:

      • Power iteration to estimate λ_max,
      • Gershgorin-type bound for λ_min.

    This is a numerical diagnostic only; the analytic PSD property
    follows from the Fourier nonnegativity of k̂_H. The bounds here
    are used by the validation suite to detect gross implementation
    errors.
    """
    M = build_toeplitz_matrix(N, H)
    v = [1.0] * N
    for _ in range(20):
        w = [0.0] * N
        for i in range(N):
            s = 0.0
            row = M[i]
            for j in range(N):
                s += row[j] * v[j]
            w[i] = s
        norm_w = math.sqrt(sum(z * z for z in w))
        if norm_w == 0.0:
            break
        v = [z / norm_w for z in w]

    lambda_max = 0.0
    for i in range(N):
        s = 0.0
        row = M[i]
        for j in range(N):
            s += row[j] * v[j]
        lambda_max += v[i] * s

    lambda_min = float('inf')
    for i in range(N):
        center = M[i][i]
        radius = 0.0
        row = M[i]
        for j in range(N):
            if j == i:
                continue
            radius += abs(row[j])
        lambda_min = min(lambda_min, center - radius)
    if lambda_min is float('inf'):
        lambda_min = 0.0
    return float(lambda_min), float(lambda_max)


##############################
# 9. PUBLIC API CLASS        #
##############################

class FormalReduction:
    """
    Volume I — Formal Reduction public API.

    This class provides the minimal surface required by the validation
    suite and later volumes. It is conceptually tied to the following
    Volume I interface theorem:

      • Fix H > 0 and N ≥ 1. Let k_H(t) = (6/H²) sech⁴(t/H) and
        k̂_H(ω) = (ω² + λ*) ŵ_H(ω) with λ* = 4/H² and ŵ_H as above.

      • Define
            Q_H(N, T0) = ∑_{m,n ≤ N} (mn)^{-σ} (m/n)^{-i T0}
                           k̂_H(log m − log n),

        with σ = 1/2. Then Q_H(N, T0) also equals the time-domain
        convolution
            ∫ k_H(t) |D_N(σ, T0 + t)|² dt.

      • The Analyst’s Problem is: prove Q_H(N, T0) ≥ 0 for all
        admissible H, N, T0. Volume I shows RH is equivalent to this
        positivity statement for the canonical kernel k_H.
    """

    # --- Kernel accessors ---

    @staticmethod
    def k_H_time(t: float, H: float) -> float:
        """
        Time-domain kernel k_H(t) = (6/H²) sech⁴(t/H).
        """
        return g_H_sech4(t, H)

    @staticmethod
    def k_H_hat(omega: float, H: float) -> float:
        """
        Frequency-side kernel symbol k̂_H(ω).
        """
        return k_H_hat(omega, H)

    # --- Toeplitz matrices and vectors ---

    @staticmethod
    def toeplitz_matrix(N: int, H: float) -> List[List[float]]:
        """
        Phase-free Toeplitz matrix with entries k̂_H(log m − log n).
        """
        return build_toeplitz_matrix(N, H)

    @staticmethod
    def physical_vector(N: int, sigma: float = 0.5) -> List[float]:
        """
        Physical coefficient vector x_n = n^{-σ}.
        """
        return physical_vector_x(N, sigma=sigma)

    # --- Core quadratic form and decompositions ---

    @staticmethod
    def Q_H(N: int, H: float, T0: float = 0.0, sigma: float = 0.5) -> float:
        """
        Full Toeplitz quadratic form Q_H(N, T0).
        """
        return QH_from_M1_and_cross(N, H, T0, sigma=sigma)

    @staticmethod
    def F2_bar(T0: float,
               H: float,
               N: int,
               sigma: float = 0.5,
               tau_min: float = None,
               tau_max: float = None,
               num_steps: int = 8001) -> float:
        """
        Time-domain second moment F̄₂(T0, H; N).
        """
        return curvature_F2_bar(T0, H, N, sigma=sigma,
                                tau_min=tau_min,
                                tau_max=tau_max,
                                num_steps=num_steps)

    @staticmethod
    def M1(N: int, H: float, sigma: float = 0.5) -> float:
        """
        Diagonal mass term M1.
        """
        return M1_diagonal_term(N, H, sigma=sigma)

    @staticmethod
    def Cross(T0: float, N: int, H: float, sigma: float = 0.5) -> float:
        """
        Off-diagonal interaction Cross(T0).
        """
        return cross_offdiagonal_term(N, H, T0, sigma=sigma)

    @staticmethod
    def AbsoluteCross(N: int, H: float, sigma: float = 0.5) -> float:
        """
        AbsoluteCross = ∑_{m≠n} |(mn)^{-σ}| |k̂_H(log m − log n)|.
        """
        return absolute_cross_term(N, H, sigma=sigma)

    @staticmethod
    def C_ratio(N: int, H: float, sigma: float = 0.5) -> float:
        """
        Relative off-diagonal magnitude C = AbsoluteCross / M1.
        """
        return C_ratio(N, H, sigma=sigma)

    # --- Parseval bridge diagnostics ---

    @staticmethod
    def parseval_residual(T0: float,
                          H: float,
                          N: int,
                          sigma: float = 0.5,
                          tau_min: float = None,
                          tau_max: float = None,
                          num_steps: int = 8001) -> float:
        """
        Absolute Parseval residual |F̄₂ − F̃₂|.
        """
        return parseval_identity_residual(T0, H, N, sigma=sigma,
                                          tau_min=tau_min,
                                          tau_max=tau_max,
                                          num_steps=num_steps)

    # --- Kernel PSD diagnostics ---

    @staticmethod
    def kernel_psd_bounds(N: int, H: float) -> Tuple[float, float]:
        """
        Approximate (λ_min, λ_max) for the kernel Toeplitz matrix.
        """
        return check_kernel_positive_definite(N, H)