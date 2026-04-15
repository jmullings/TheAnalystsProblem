#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# FORMAL_REDUCTION_QH_ANALYST_PROBLEM.py
#
# Formal Reduction – Q_H Analyst’s Problem core.
# Implements the proven sech² second‑moment Parseval bridge in a
# minimal, dependency‑free form adapted from Tier‑32 engine code.

import math
import cmath
from typing import List, Tuple, Sequence


##############################
# 1. BASIC SPECIAL FUNCTIONS #
##############################
def k_H_time(t: float, H: float) -> float:
    """Time-domain kernel alias for g_H_sech4."""
    return g_H_sech4(t, H)
    
def sech(x: float) -> float:
    return 1.0 / math.cosh(x)


def sech2(x: float) -> float:
    c = math.cosh(x)
    return 1.0 / (c * c)


def sech4(x: float) -> float:
    s2 = sech2(x)
    return s2 * s2


def tanh(x: float) -> float:
    return math.tanh(x)


########################################
# 1b. LEGACY Λ_H API (FOR TDD IMPORT)  #
########################################

def Lambda_H_tau(tau: float, H: float) -> float:
    """
    Legacy curvature window used by earlier drafts.

    Kept for API compatibility with the validation suite.
    Not used in the final second‑moment bridge, which is
    built on g_{λ*}(t) = (6/H²) sech⁴(t/H).
    """
    u = tau / H
    return 2.0 * math.pi * sech2(u)


def Lambda_H_dd_tau(tau: float, H: float) -> float:
    """
    Second derivative of the legacy Λ_H:

        Λ_H''(τ) = (2π/H²) sech²(τ/H) [4 − 6 sech²(τ/H)].
    """
    u = tau / H
    s2 = sech2(u)
    return (2.0 * math.pi / (H * H)) * s2 * (4.0 - 6.0 * s2)


##############################
# 2. CORE KERNELS (GROUND TRUTH)
##############################

def w_H_time(t: float, H: float) -> float:
    """w_H(t) = sech²(t/H)."""
    return sech2(t / H)


def g_H_sech4(t: float, H: float) -> float:
    """
    g_{λ*}(t) = (6/H²) sech⁴(t/H).

    This is the corrected Bochner‑positive kernel used in the
    second‑moment bridge.
    """
    return (6.0 / (H * H)) * sech4(t / H)


def fourier_w_H(omega: float, H: float) -> float:
    """
    Fourier transform of w_H(t) = sech²(t/H):

        ŵ_H(ω) = π H² ω / sinh(π H ω / 2),

    with the convention that ŵ_H(0) = 2H.
    """
    if abs(omega) < 1e-15:
        return 2.0 * H
    arg = math.pi * H * omega / 2.0
    if abs(arg) > 700.0:
        # Exponential suppression
        return 0.0
    return (math.pi * H * H * omega) / math.sinh(arg)


def lambda_star(H: float) -> float:
    """λ* = 4 / H² in the corrected kernel ĝ_{λ*}."""
    return 4.0 / (H * H)


def k_H_hat(omega: float, H: float) -> float:
    """
    ĝ_{λ*}(ω) = (ω² + λ*) · ŵ_H(ω), strictly ≥ 0 by Bochner.
    """
    lam = lambda_star(H)
    wh = fourier_w_H(omega, H)
    return (omega * omega + lam) * wh


##############################
# 3. DIRICHLET POLYNOMIAL    #
##############################

def dirichlet_S_N(T: float, N: int, sigma: float = 0.5) -> complex:
    """
    S_N(σ,T) = Σ n^{-σ} e^{-i T ln n}.
    """
    s = 0.0 + 0.0j
    for n in range(1, N + 1):
        ln_n = math.log(n)
        s += n ** (-sigma) * cmath.exp(-1j * T * ln_n)
    return s


def dirichlet_S_N_derivatives(T: float, N: int, sigma: float = 0.5):
    """
    (S_N, S_N', S_N'') wrt σ, as in your original script.

    Kept to satisfy existing F2_sigma_T tests, although the bridge
    itself is built on |S_N|², not its σ‑derivatives.
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
    """
    Your σ‑second derivative functional:

        F₂(σ,T) = 2|S_N'|² + 2 Re(S_N'' conj(S_N)).

    Used only for TDD tests that explicitly reference it.
    """
    S0, S1, S2 = dirichlet_S_N_derivatives(T, N, sigma=sigma)
    return 2.0 * (abs(S1) ** 2) + 2.0 * (S2 * S0.conjugate()).real


####################################
# 4. SECOND-MOMENT BRIDGE OBJECT  #
####################################

def D_N(T: float, N: int, sigma: float = 0.5) -> complex:
    """
    D_N(T) = S_N(σ,T) with σ = 1/2 by default.
    """
    return dirichlet_S_N(T, N, sigma=sigma)


def second_moment_integrand(t: float,
                            T0: float,
                            H: float,
                            N: int,
                            sigma: float = 0.5) -> float:
    """
    g_{λ*}(t) |D_N(T₀ + t)|² — the Bochner side integrand for F̃₂.
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
    """
    F̄₂(T₀,H;N) := F̃₂(T₀;H,N) = ∫ g_{λ*}(t) |D_N(T₀+t)|² dt,
    using Simpson’s rule.

    This is the same object the Toeplitz form computes.
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
    """x_n = n^{-σ}, used only for convenience / tests."""
    return [n ** (-sigma) for n in range(1, N + 1)]


def phased_quadratic_form(N: int,
                          H: float,
                          T0: float,
                          sigma: float = 0.5) -> float:
    """
    Toeplitz form for F̃₂, adapted from parseval_toeplitz_F2:

        F̃₂(T₀;H,N) =
          Σ_{m,n} (mn)^{-σ} (m/n)^{-iT₀} ĝ_{λ*}(ln m − ln n),

    with ĝ_{λ*} = k_H_hat above. σ=1/2 in the engine; σ kept general
    here to match your API.
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
            # (m/n)^(-iT0) = exp(-i T0 (ln m - ln n))
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
    """
    Discrepancy F̄₂(T₀,H;N) − F̃₂_toeplitz(T₀;H,N).
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
    |F̄₂ − Toeplitz|; test expects this to be small (< 1e‑2).
    """
    return abs(parseval_bridge_certificate(T0, H, N, sigma=sigma,
                                           tau_min=tau_min,
                                           tau_max=tau_max,
                                           num_steps=num_steps))


##############################
# 7. M1 / CROSS STRUCTURE    #
##############################

def M1_diagonal_term(N: int, H: float, sigma: float = 0.5) -> float:
    """
    Diagonal part of F̃₂ Toeplitz form:

        M1 = Σ_n n^{-2σ} ĝ_{λ*}(0),
        ĝ_{λ*}(0) = λ* · 2H = 8 / H.
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
    """
    Off‑diagonal part of the Toeplitz form:

        Cross = Σ_{m≠n} (mn)^{-σ} (m/n)^{-iT₀} ĝ_{λ*}(ln m − ln n).
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
    M1 + Cross reproduces the full Toeplitz form.
    """
    return M1_diagonal_term(N, H, sigma=sigma) + \
        cross_offdiagonal_term(N, H, T0, sigma=sigma)


def absolute_cross_term(N: int,
                        H: float,
                        sigma: float = 0.5) -> float:
    """
    AbsoluteCross = Σ_{m≠n} |(mn)^{-σ}| |ĝ_{λ*}(ln m − ln n)|.

    This bounds |Cross(T₀)| uniformly in T₀.
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
    """
    C = AbsoluteCross / M1, a relative off‑diagonal bound.
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
    """
    Phase‑free Toeplitz matrix with entries ĝ_{λ*}(ln m − ln n).
    """
    M = [[0.0 for _ in range(N)] for _ in range(N)]
    for m in range(1, N + 1):
        ln_m = math.log(m)
        for n in range(1, N + 1):
            ln_n = math.log(n)
            diff = ln_m - ln_n
            M[m - 1][n - 1] = k_H_hat(diff, H)
    # Symmetrize numerically
    for i in range(N):
        for j in range(i + 1, N):
            v = 0.5 * (M[i][j] + M[j][i])
            M[i][j] = v
            M[j][i] = v
    return M


def toeplitz_quadratic_form(M: Sequence[Sequence[float]],
                            x: Sequence[float]) -> float:
    """x^T M x for real symmetric M."""
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
    """
    Approximate (λ_min, λ_max) for the phase‑free Toeplitz matrix,
    via power iteration and Gershgorin bound.
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
    Minimal API surface required by VALIDATION_SUITE.
    """

    @staticmethod
    def k_H_time(t: float, H: float) -> float:
        # For the API this is the time‑domain kernel used in Q_H;
        # we expose g_{λ*}(t) here.
        return g_H_sech4(t, H)

    @staticmethod
    def k_H_hat(omega: float, H: float) -> float:
        return k_H_hat(omega, H)

    @staticmethod
    def toeplitz_matrix(N: int, H: float) -> List[List[float]]:
        return build_toeplitz_matrix(N, H)

    @staticmethod
    def physical_vector(N: int, sigma: float = 0.5) -> List[float]:
        return physical_vector_x(N, sigma=sigma)

    @staticmethod
    def Q_H(N: int, H: float, T0: float = 0.0, sigma: float = 0.5) -> float:
        return QH_from_M1_and_cross(N, H, T0, sigma=sigma)

    @staticmethod
    def F2_bar(T0: float,
               H: float,
               N: int,
               sigma: float = 0.5,
               tau_min: float = None,
               tau_max: float = None,
               num_steps: int = 8001) -> float:
        return curvature_F2_bar(T0, H, N, sigma=sigma,
                                tau_min=tau_min,
                                tau_max=tau_max,
                                num_steps=num_steps)

    @staticmethod
    def M1(N: int, H: float, sigma: float = 0.5) -> float:
        return M1_diagonal_term(N, H, sigma=sigma)

    @staticmethod
    def Cross(T0: float, N: int, H: float, sigma: float = 0.5) -> float:
        return cross_offdiagonal_term(N, H, T0, sigma=sigma)

    @staticmethod
    def AbsoluteCross(N: int, H: float, sigma: float = 0.5) -> float:
        return absolute_cross_term(N, H, sigma=sigma)

    @staticmethod
    def C_ratio(N: int, H: float, sigma: float = 0.5) -> float:
        return C_ratio(N, H, sigma=sigma)

    @staticmethod
    def parseval_residual(T0: float,
                          H: float,
                          N: int,
                          sigma: float = 0.5,
                          tau_min: float = None,
                          tau_max: float = None,
                          num_steps: int = 8001) -> float:
        return parseval_identity_residual(T0, H, N, sigma=sigma,
                                          tau_min=tau_min,
                                          tau_max=tau_max,
                                          num_steps=num_steps)

    @staticmethod
    def kernel_psd_bounds(N: int, H: float) -> Tuple[float, float]:
        return check_kernel_positive_definite(N, H)