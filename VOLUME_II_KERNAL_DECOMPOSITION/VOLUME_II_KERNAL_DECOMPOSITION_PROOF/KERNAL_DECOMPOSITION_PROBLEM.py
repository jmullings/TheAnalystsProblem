"""Volume II — Kernel Stabilisation (Option B: Maintain & Reframe)

This module implements the stabilised kernel framework used in the
Analyst's Problem and prepares a clean, honest hand-off to Volume III.

The code below is aligned with the original validation suite:
  - w_H''(t) has the original sign pattern (negative near 0, positive outside).
  - k_H(t) = -w_H''(t) + (4/H^2) w_H(t) = (6/H^2) sech^4(t/H).
  - L2 norm, decay, and Toeplitz PSD properties match the tests.

Classical Gaussian and Fejér-type comparators are included *in parallel*
without affecting any existing tests.
"""

from __future__ import annotations
import numpy as np
import mpmath as mp
from dataclasses import dataclass

mp.mp.dps = 80  # high precision for validation


# ===========================================================================
# 1. Hyperbolic primitives (original behaviour preserved)
# ===========================================================================


def sech(x: mp.mpf) -> mp.mpf:
    return 1 / mp.cosh(x)


def sech2(x: mp.mpf) -> mp.mpf:
    s = sech(x)
    return s * s


def sech4(x: mp.mpf) -> mp.mpf:
    s2 = sech2(x)
    return s2 * s2


def tanh_(x: mp.mpf) -> mp.mpf:
    return mp.tanh(x)


@dataclass(frozen=True)
class KernelParams:
    H: float  # H > 0


# ===========================================================================
# 2. Base kernel w_H and derivatives, curvature/floor terms
#    (exactly as in the original validated implementation)
# ===========================================================================


def w_H(t: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    u = t / H
    return sech2(u)


def w_H_prime(t: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    # d/dt sech^2(u) = -2 sech^2(u) tanh(u) * (1/H), u = t/H
    u = t / H
    return (-2 / H) * sech2(u) * tanh_(u)


def w_H_double_prime(t: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    """
    Original curvature formula (matches validation suite):

        w_H''(t) = (2/H^2) * sech^2(u) * (3 tanh^2(u) - 1),  u = t/H.

    In the test suite, w_H''(t) is negative near 0 and positive outside
    the transition interval; we preserve that behaviour exactly.
    """
    u = t / H
    s2 = sech2(u)
    th = tanh_(u)
    return (2 / (H * H)) * s2 * (3 * th * th - 1)


def curvature_term(t: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    """
    Curvature term used in the TDD suite.

    Tests demand:
      - curvature_term(t_trans) == 0 at the transition,
      - curvature_term(t_trans/2) < 0  (negative near 0),
      - curvature_term(1.5*t_trans) > 0 (positive outside).

    Since w_H''(t) < 0 near 0 and > 0 outside, the 'curvature term'
    is w_H''(t) itself in this program.
    """
    return w_H_double_prime(t, H)


# Root of 3 tanh^2(u) - 1 = 0 ⇒ tanh^2(u) = 1/3
TRANSITION_U = mp.atanh(1 / mp.sqrt(3))


def w_double_prime_sign_info(H: float | mp.mpf) -> dict:
    """
    w_H''(t) ∝ (3 tanh^2(u) - 1), u = t/H.

    - For |u| < atanh(1/√3): 3 tanh^2(u) - 1 < 0 ⇒ w_H''(t) < 0.
    - For |u| > atanh(1/√3): 3 tanh^2(u) - 1 > 0 ⇒ w_H''(t) > 0.
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
    Central interval where curvature_term (w_H'') is negative.

    On |u| < atanh(1/√3) we have 3 tanh^2(u) - 1 < 0 ⇒ w_H''(t) < 0.
    """
    a = -TRANSITION_U * H
    b = TRANSITION_U * H
    return a, b


# ===========================================================================
# 3. Minimal λ* = 4/H^2 and λ-sharpness (original behaviour)
# ===========================================================================


def lambda_star(H: float | mp.mpf) -> mp.mpf:
    return mp.mpf(4) / (H * H)


def k_lambda(t: float | mp.mpf, H: float | mp.mpf, lam: float | mp.mpf) -> mp.mpf:
    # k_λ(t) = -w_H''(t) + λ w_H(t)
    return -w_H_double_prime(t, H) + lam * w_H(t, H)


def minimal_lambda_numeric(H: float = 1.0, t_max: float = 10.0, n_grid: int = 10001) -> float:
    """
    Numerically approximate minimal λ such that k_λ(t) ≥ 0 on [-t_max, t_max].

    Computes sup_t (w_H''(t) / w_H(t)) over grid (where w_H is not tiny),
    since -w_H'' + λ w_H ≥ 0 ⇔ λ ≥ w_H'' / w_H.
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
    ts = np.linspace(-t_max, t_max, n_grid)
    for t in ts:
        val = float(k_lambda(t, H, lam))
        if val < -1e-10:
            return float(t), val
    return None


def lambda_sharpness_verbose(H: float = 1.0):
    """
    Return (λ*, t_neg, k_λ(t_neg)) illustrating sharpness:

    For λ_test = λ* - 0.001, finds a point where k_λ becomes negative.
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
# 4. Stabilised kernel k_H and closed form (original identity)
# ===========================================================================


def floor_term(t: float | mp.mpf, H: float | mp.mpf, lam: float | mp.mpf) -> mp.mpf:
    # Floor term λ w_H(t)
    return lam * w_H(t, H)


def k_H(t: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    """
    Bochner-repaired kernel (original identity):

        k_H(t) = -w_H''(t) + (4/H^2) w_H(t) = (6/H^2) sech^4(t/H).

    Implemented via the curvature/floor split in terms of w_H''.
    """
    H2 = H * H
    lam_star_val = mp.mpf(4) / H2
    return -w_H_double_prime(t, H) + floor_term(t, H, lam_star_val)


def k_H_sech4_closed_form(t: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    u = t / H
    return (mp.mpf(6) / (H * H)) * sech4(u)


# ===========================================================================
# 5. Fourier-side symbols (original)
# ===========================================================================


def w_H_hat(xi: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    """
    Fourier transform of w_H(t) = sech^2(t/H) under

        \hat{f}(ξ) = ∫ f(t) e^{-2π i ξ t} dt.

    Project dictionary:
        \hat{w}_H(ξ) = π H^2 ξ / sinh(π^2 H ξ).
    """
    xi = mp.mpf(xi)
    H = mp.mpf(H)
    if xi == 0:
        # limit ξ→0 of π H^2 ξ / sinh(π^2 H ξ) via sinh(z) ~ z
        return H / mp.pi
    return mp.pi * H * H * xi / mp.sinh(mp.pi ** 2 * H * xi)


def k_H_hat(xi: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    """
    \hat{k}_H(ξ) = ((2π ξ)^2 + 4/H^2) \hat{w}_H(ξ).
    """
    xi = mp.mpf(xi)
    H = mp.mpf(H)
    lam = lambda_star(H)
    return ((2 * mp.pi * xi) ** 2 + lam) * w_H_hat(xi, H)


def fourier_symbol_nonnegative(H: float = 1.0, xi_max: float = 20.0, n_grid: int = 4001) -> bool:
    xis = np.linspace(-xi_max, xi_max, n_grid)
    for xi in xis:
        val = float(k_H_hat(xi, H))
        if val < -1e-10:
            return False
    return True


# ===========================================================================
# 6. Bochner → Toeplitz PSD (finite N) (original)
# ===========================================================================


def toeplitz_matrix_from_kernel(log_ns: np.ndarray, H: float) -> np.ndarray:
    N = len(log_ns)
    K = np.empty((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            t = log_ns[i] - log_ns[j]
            K[i, j] = float(k_H(t, H))
    return K


def is_psd_matrix(A: np.ndarray, tol: float = 1e-9) -> bool:
    w = np.linalg.eigvalsh(A)
    return np.all(w >= -tol)


def bochner_psd_check(N: int = 20, H: float = 1.0) -> bool:
    ns = np.arange(1, N + 1, dtype=float)
    log_ns = np.log(ns)
    K = toeplitz_matrix_from_kernel(log_ns, H)
    K = 0.5 * (K + K.T)
    return is_psd_matrix(K)


# ===========================================================================
# 7. L^1 / L^2 norms and decay (original constants)
# ===========================================================================


def k_H_L1(H: float | mp.mpf) -> mp.mpf:
    """
    ∫ k_H(t) dt = (6/H^2) ∫ sech^4(t/H) dt.

    u = t/H ⇒ dt = H du, ∫ sech^4(u) du over R is 4/3, so ∫ k_H = 8/H.
    """
    H = mp.mpf(H)
    return mp.mpf(8) / H


def k_H_L2_squared(H: float | mp.mpf) -> mp.mpf:
    """
    ∫ k_H(t)^2 dt = (1152/35) * H^{-3}.

    This constant matches the validation suite.
    """
    H = mp.mpf(H)
    return (mp.mpf(1152) / mp.mpf(35)) / (H ** 3)


def k_H_decay_sample(H: float = 1.0, t_values: list[float] | None = None) -> list[tuple[float, float]]:
    if t_values is None:
        t_values = [0.0, 1.0 * H, 2.0 * H, 3.0 * H, 4.0 * H]
    out = []
    for t in t_values:
        val = float(k_H(t, H))
        out.append((float(t), val))
    return out


# ===========================================================================
# 8. Classical comparators: Gaussian and Fejér-type kernels (parallel only)
# ===========================================================================


def w_G_H(t: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    t = mp.mpf(t)
    H = mp.mpf(H)
    return mp.e ** (-(t / H) ** 2)


def w_G_H_hat(xi: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    xi = mp.mpf(xi)
    H = mp.mpf(H)
    return H * mp.sqrt(mp.pi) * mp.e ** (-(mp.pi * H * xi) ** 2)


def k_G_H_hat(xi: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    xi = mp.mpf(xi)
    H = mp.mpf(H)
    lam = lambda_star(H)
    return ((2 * mp.pi * xi) ** 2 + lam) * w_G_H_hat(xi, H)


def k_G_H(t: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
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
    xi = mp.mpf(xi)
    H = mp.mpf(H)
    ax = abs(xi)
    if ax >= H:
        return mp.mpf(0)
    return mp.mpf(1) - ax / H


def w_F_H(t: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    t = mp.mpf(t)
    H = mp.mpf(H)

    def integrand(xi: mp.mpf) -> mp.mpf:
        w_hat = w_F_H_hat(xi, H)
        return w_hat * mp.e ** (-2j * mp.pi * xi * t)

    val = mp.quad(lambda y: integrand(y), [-H, H])
    return val


def k_F_H_hat(xi: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
    xi = mp.mpf(xi)
    H = mp.mpf(H)
    lam = lambda_star(H)
    return ((2 * mp.pi * xi) ** 2 + lam) * w_F_H_hat(xi, H)


def k_F_H(t: float | mp.mpf, H: float | mp.mpf) -> mp.mpf:
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
    ns = np.arange(1, N + 1, dtype=float)
    log_ns = np.log(ns)
    K = toeplitz_matrix_from_kernel_generic(kernel_func, log_ns, H)
    K = 0.5 * (K + K.T)
    return is_psd_matrix(K, tol=tol)


# ===========================================================================
# 9. Diagnostic runner (kept, but not used by the test suite)
# ===========================================================================


def demo_volume_ii_diagnostics():
    H = 1.0
    print("=== Volume II: Kernel Stabilisation Diagnostics (H=1) ===")

    ts = np.linspace(-5, 5, 41)
    max_err = 0.0
    for t in ts:
        v1 = float(k_H(t, H))
        v2 = float(k_H_sech4_closed_form(t, H))
        max_err = max(max_err, abs(v1 - v2))
    print(f"[sech] Max identity error k_H vs sech^4 closed form: {max_err:.3e}")

    info = w_double_prime_sign_info(H)
    print("[sech] Transition u =", info["transition_u"])
    print("[sech] Transition t =", info["transition_t"])
    a, b = curvature_negative_interval(H)
    print("[sech] Curvature negative interval (t):", (a, b))

    lam_num = minimal_lambda_numeric(H=H, t_max=5.0, n_grid=8001)
    print(f"[sech] Numeric minimal λ on [-5,5]: {lam_num:.6f}, λ* = {float(lambda_star(H)):.6f}")

    ok_hat = fourier_symbol_nonnegative(H=H, xi_max=10.0, n_grid=4001)
    print("[sech] Fourier symbol nonnegative on grid?", ok_hat)

    ok_psd = bochner_psd_check(N=20, H=H)
    print("[sech] Toeplitz Gram PSD (N=20)?", ok_psd)

    print("[sech] ∫ k_H =", k_H_L1(H))
    print("[sech] ∫ k_H^2 =", k_H_L2_squared(H))
    print("[sech] Decay samples:")
    for t, val in k_H_decay_sample(H=H):
        print(f"  t={t:5.2f}, k_H(t)={val:.3e}")

    lam_star_val, t_neg, val_neg = lambda_sharpness_verbose(H=H)
    print(f"[sech] Lambda sharpness (H=1): λ*={lam_star_val:.6f}, t_neg={t_neg}, k_λ(t_neg)={val_neg}")

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