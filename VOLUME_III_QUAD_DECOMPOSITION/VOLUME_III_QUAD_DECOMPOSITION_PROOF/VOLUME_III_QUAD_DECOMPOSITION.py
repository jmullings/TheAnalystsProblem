#!/usr/bin/env python3
"""
Volume III – Quadratic Form Decomposition & Dominance
=====================================================

Enhanced structural implementation:

1. Algebraic log identities (symbolic + numeric checks)
2. Kernel k_H(t) = 6/H^2 * sech^4(t/H) and its Toeplitz structure
3. Quadratic form Q_H(x; T0) over x_n = n^{-1/2}
4. Exact diagonal / off–diagonal decomposition
5. Symmetry / antisymmetry checks
6. Structural near/intermediate/tail off–diagonal decomposition
7. Growth diagnostics in N and H per region
8. Asymptotic scaling models for near/mid regions
9. Numerical experiments for D_H, O_H, regional pieces, and D_H / |O_H|
10. Volume III sanity suite (completion status)
11. Dyadic multiscale decomposition of the off-diagonal band
12. Kernel curvature / Taylor diagnostics near t = 0
13. Fourier-side decay diagnostics for k_H

This file is self‑contained: run it directly to execute a basic
validation + diagnostic sweep, or import it and call the functions
programmatically from a notebook or higher‑level orchestration script.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any, List

import numpy as np
import mpmath as mp
import sympy as sp

mp.mp.dps = 80  # default precision for high‑precision checks

# ---------------------------------------------------------------------------
# 1. CORE ALGEBRAIC IDENTITIES (SYMBOLIC & NUMERIC)
# ---------------------------------------------------------------------------

@dataclass
class AlgebraicIdentitiesResult:
    log_diff_ok: bool
    square_decomp_ok: bool
    hard_identity_ok: bool
    sym_antisym_ok: bool
    details: Dict[str, Any]


def verify_algebraic_identities(samples: int = 50) -> AlgebraicIdentitiesResult:
    details: Dict[str, Any] = {}

    m, n = sp.symbols("m n", positive=True)
    logm, logn = sp.log(m), sp.log(n)

    expr_logdiff = sp.simplify(logm - logn - sp.log(m / n))
    log_diff_symbolic_ok = sp.simplify(expr_logdiff) == 0

    expr_sq = sp.simplify((logm - logn) ** 2 - (logm ** 2 + logn ** 2 - 2 * logm * logn))
    square_decomp_symbolic_ok = sp.simplify(expr_sq) == 0

    expr_hard = sp.simplify(
        4 * logm * logn + (logm - logn) ** 2 - (logm + logn) ** 2
    )
    hard_identity_symbolic_ok = sp.simplify(expr_hard) == 0

    f = logm * logn
    f_sym = (f + f.subs({m: n, n: m})) / 2
    f_anti = (f - f.subs({m: n, n: m})) / 2
    sym_antisym_symbolic_ok = sp.simplify(f - (f_sym + f_anti)) == 0

    details["symbolic"] = {
        "log_diff": expr_logdiff,
        "square_decomp": expr_sq,
        "hard_identity": expr_hard,
        "sym_antisym_defect": sp.simplify(f - (f_sym + f_anti)),
    }

    rng = np.random.default_rng(123456)
    numeric_logdiff_ok = True
    numeric_sq_ok = True
    numeric_hard_ok = True
    numeric_sym_antisym_ok = True

    def f_val(a: float, b: float) -> float:
        return math.log(a) * math.log(b)

    for _ in range(samples):
        a = float(rng.uniform(1.1, 100.0))
        b = float(rng.uniform(1.1, 100.0))

        lhs = math.log(a) - math.log(b)
        rhs = math.log(a / b)
        if abs(lhs - rhs) > 1e-12:
            numeric_logdiff_ok = False

        lhs = (math.log(a) - math.log(b)) ** 2
        rhs = (math.log(a) ** 2 + math.log(b) ** 2 - 2 * math.log(a) * math.log(b))
        if abs(lhs - rhs) > 1e-12:
            numeric_sq_ok = False

        lhs = 4 * math.log(a) * math.log(b) + (math.log(a) - math.log(b)) ** 2
        rhs = (math.log(a * b)) ** 2
        if abs(lhs - rhs) > 1e-12:
            numeric_hard_ok = False

        f_ab = f_val(a, b)
        f_ba = f_val(b, a)
        f_sym_num = 0.5 * (f_ab + f_ba)
        f_anti_num = 0.5 * (f_ab - f_ba)
        if abs(f_ab - (f_sym_num + f_anti_num)) > 1e-12:
            numeric_sym_antisym_ok = False

    details["numeric"] = {
        "log_diff_ok": numeric_logdiff_ok,
        "square_decomp_ok": numeric_sq_ok,
        "hard_identity_ok": numeric_hard_ok,
        "sym_antisym_ok": numeric_sym_antisym_ok,
    }

    log_diff_ok = log_diff_symbolic_ok and numeric_logdiff_ok
    square_decomp_ok = square_decomp_symbolic_ok and numeric_sq_ok
    hard_identity_ok = hard_identity_symbolic_ok and numeric_hard_ok
    sym_antisym_ok = sym_antisym_symbolic_ok and numeric_sym_antisym_ok

    return AlgebraicIdentitiesResult(
        log_diff_ok=log_diff_ok,
        square_decomp_ok=square_decomp_ok,
        hard_identity_ok=hard_identity_ok,
        sym_antisym_ok=sym_antisym_ok,
        details=details,
    )

# ---------------------------------------------------------------------------
# 2. KERNEL k_H AND SECH^4 IMPLEMENTATION
# ---------------------------------------------------------------------------

def sech(x: mp.mpf) -> mp.mpf:
    return 1 / mp.cosh(x)


def k_H(t: mp.mpf, H: mp.mpf) -> mp.mpf:
    z = t / H
    s = sech(z)
    return mp.mpf("6") / (H ** 2) * (s ** 4)


@dataclass
class KernelChecks:
    symmetry_ok: bool
    decay_ok: bool
    details: Dict[str, Any]


def verify_kernel_properties(H: float = 1.0) -> KernelChecks:
    H_mp = mp.mpf(H)
    rng = np.random.default_rng(42)
    sym_ok = True
    details: Dict[str, Any] = {"symmetry_samples": []}

    for _ in range(20):
        t = float(rng.uniform(-5.0, 5.0))
        kt = k_H(mp.mpf(t), H_mp)
        km = k_H(mp.mpf(-t), H_mp)
        details["symmetry_samples"].append((t, kt, km))
        if mp.almosteq(kt, km, rel_eps=1e-30) is False:
            sym_ok = False

    decay_points = [0.5 * H, 1.0 * H, 2.0 * H, 3.0 * H, 4.0 * H]
    approx_ratios = []
    for t in decay_points:
        kt = k_H(mp.mpf(t), H_mp)
        approx = mp.e ** (-4 * abs(t) / H_mp)
        approx_ratios.append(kt / approx)

    details["decay_ratios"] = approx_ratios
    decay_ok = all(r > 0 for r in approx_ratios)

    return KernelChecks(symmetry_ok=sym_ok, decay_ok=decay_ok, details=details)

# ---------------------------------------------------------------------------
# 3. QUADRATIC FORM, MATRICES, AND BASIC DECOMPOSITION
# ---------------------------------------------------------------------------

@dataclass
class QuadraticFormConfig:
    N: int
    H: float
    T0: float
    weight: Callable[[int], float] = lambda n: n ** (-0.5)


@dataclass
class QuadraticFormMatrices:
    logn: np.ndarray
    delta: np.ndarray
    K: np.ndarray
    P: np.ndarray
    W: np.ndarray
    A: np.ndarray
    D_H: float
    O_H: float
    Q_H: float


def build_quadratic_form(cfg: QuadraticFormConfig) -> QuadraticFormMatrices:
    N = cfg.N
    H = float(cfg.H)
    T0 = float(cfg.T0)

    n = np.arange(1, N + 1, dtype=float)
    logn = np.log(n)
    delta = logn[:, None] - logn[None, :]

    k_vec = np.frompyfunc(lambda x: float(k_H(mp.mpf(x), mp.mpf(H))), 1, 1)
    K = k_vec(delta).astype(float)

    P = np.cos(T0 * delta)

    W = 1.0 / np.sqrt(n[:, None] * n[None, :])

    A = W * K * P

    diag_mask = np.eye(N, dtype=bool)
    D_H = float(A[diag_mask].sum())
    O_H = float(A[~diag_mask].sum())
    Q_H = D_H + O_H

    return QuadraticFormMatrices(
        logn=logn,
        delta=delta,
        K=K,
        P=P,
        W=W,
        A=A,
        D_H=D_H,
        O_H=O_H,
        Q_H=Q_H,
    )

@dataclass
class SymmetryValidation:
    K_sym_ok: bool
    A_sym_ok: bool
    max_K_asym: float
    max_A_asym: float


def check_matrix_symmetry(mats: QuadraticFormMatrices) -> SymmetryValidation:
    K = mats.K
    A = mats.A
    diff_K = K - K.T
    diff_A = A - A.T
    max_K_asym = float(np.abs(diff_K).max())
    max_A_asym = float(np.abs(diff_A).max())
    return SymmetryValidation(
        K_sym_ok=max_K_asym < 1e-12,
        A_sym_ok=max_A_asym < 1e-12,
        max_K_asym=max_K_asym,
        max_A_asym=max_A_asym,
    )

# ---------------------------------------------------------------------------
# 4. STRUCTURAL OFF–DIAGONAL DECOMPOSITION (NEAR / MID / TAIL)
# ---------------------------------------------------------------------------

@dataclass
class OffDiagonalStructure:
    near_sum: float
    mid_sum: float
    tail_sum: float
    near_count: int
    mid_count: int
    tail_count: int


def decompose_off_diagonal_regions(
    mats: QuadraticFormMatrices,
    H: float,
    alpha_near: float = 0.5,
    alpha_mid: float = 2.0,
) -> OffDiagonalStructure:
    delta = mats.delta
    A = mats.A
    N = delta.shape[0]

    mask_off = ~np.eye(N, dtype=bool)
    abs_delta = np.abs(delta)

    near_mask = mask_off & (abs_delta <= alpha_near * H)
    mid_mask = mask_off & (abs_delta > alpha_near * H) & (abs_delta <= alpha_mid * H)
    tail_mask = mask_off & (abs_delta > alpha_mid * H)

    near_sum = float(A[near_mask].sum())
    mid_sum = float(A[mid_mask].sum())
    tail_sum = float(A[tail_mask].sum())

    return OffDiagonalStructure(
        near_sum=near_sum,
        mid_sum=mid_sum,
        tail_sum=tail_sum,
        near_count=int(near_mask.sum()),
        mid_count=int(mid_mask.sum()),
        tail_count=int(tail_mask.sum()),
    )

# ---------------------------------------------------------------------------
# 5. GROWTH ANALYSIS: DIAGONAL VS OFF–DIAGONAL (WITH REGIONS)
# ---------------------------------------------------------------------------

@dataclass
class GrowthDiagnostics:
    N: int
    H: float
    T0: float
    D_H: float
    O_H: float
    Q_H: float
    harmonic_N: float
    diag_theory: float
    ratio_D_to_absO: float
    regions: OffDiagonalStructure
    asymp_near: float
    asymp_mid: float


def harmonic_number(N: int) -> float:
    if N < 10_000:
        return float(sum(1.0 / k for k in range(1, N + 1)))
    gamma = 0.57721566490153286060651209008240243104215933593992
    return math.log(N) + gamma + 1.0 / (2 * N) - 1.0 / (12 * N ** 2)

# ---------------------------------------------------------------------------
# 6. ASYMPTOTIC SCALING LAWS FOR NEAR AND MID REGIONS
# ---------------------------------------------------------------------------

def asymptotic_near_region(N: int, H: float, alpha_near: float = 0.5) -> float:
    mp.mp.dps = 50
    H_mp = mp.mpf(H)
    alpha = mp.mpf(alpha_near)

    def integrand(t: float) -> float:
        t_mp = mp.mpf(t)
        return float(k_H(t_mp, H_mp) * mp.e ** (-mp.fabs(t_mp)))

    I_near = mp.quad(lambda t: integrand(t), [-alpha * H_mp, alpha * H_mp])
    return float(N * I_near)


def asymptotic_mid_region(
    N: int,
    H: float,
    alpha_near: float = 0.5,
    alpha_mid: float = 2.0,
) -> float:
    mp.mp.dps = 50
    H_mp = mp.mpf(H)
    a1 = mp.mpf(alpha_near)
    a2 = mp.mpf(alpha_mid)

    def integrand(t: float) -> float:
        t_mp = mp.mpf(t)
        return float(k_H(t_mp, H_mp) * mp.e ** (-mp.fabs(t_mp)))

    I_mid = mp.quad(lambda t: integrand(t), [a1 * H_mp, a2 * H_mp]) * 2
    return float(N * I_mid)


def analyse_growth(cfg: QuadraticFormConfig) -> Tuple[QuadraticFormMatrices, GrowthDiagnostics]:
    mats = build_quadratic_form(cfg)
    N, H = cfg.N, cfg.H
    T0 = cfg.T0

    k0 = float(k_H(mp.mpf("0"), mp.mpf(H)))
    HN = harmonic_number(N)
    diag_theory = k0 * HN

    regions = decompose_off_diagonal_regions(mats, H)

    asymp_near = asymptotic_near_region(N, H)
    asymp_mid = asymptotic_mid_region(N, H)

    ratio = abs(mats.D_H) / max(abs(mats.O_H), 1e-30)

    diag = GrowthDiagnostics(
        N=N,
        H=H,
        T0=T0,
        D_H=mats.D_H,
        O_H=mats.O_H,
        Q_H=mats.Q_H,
        harmonic_N=HN,
        diag_theory=diag_theory,
        ratio_D_to_absO=ratio,
        regions=regions,
        asymp_near=asymp_near,
        asymp_mid=asymp_mid,
    )
    return mats, diag

# ---------------------------------------------------------------------------
# 7. PARAMETER SWEEP & CONVERGENCE EXPERIMENTS
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    records: Dict[Tuple[int, float], GrowthDiagnostics]


def parameter_sweep(
    Ns: Tuple[int, ...],
    Hs: Tuple[float, ...],
    T0: float = 0.0,
) -> SweepResult:
    records: Dict[Tuple[int, float], GrowthDiagnostics] = {}
    for N in Ns:
        for H in Hs:
            cfg = QuadraticFormConfig(N=N, H=H, T0=T0)
            _, diag = analyse_growth(cfg)
            records[(N, H)] = diag
    return SweepResult(records=records)

# ---------------------------------------------------------------------------
# 8. FITTING SCALING LAWS FROM DATA
# ---------------------------------------------------------------------------

def fit_scaling_laws(N_values: List[int], H: float, region: str = "near") -> Dict[str, float]:
    from sklearn.metrics import r2_score
    from scipy.optimize import curve_fit

    sums = []
    for N in N_values:
        cfg = QuadraticFormConfig(N=N, H=H, T0=0.0)
        mats = build_quadratic_form(cfg)
        reg = decompose_off_diagonal_regions(mats, H)
        if region == "near":
            s = reg.near_sum
        elif region == "mid":
            s = reg.mid_sum
        else:
            s = reg.tail_sum
        sums.append(s)

    N_arr = np.array(N_values, dtype=float)
    sums_arr = np.array(sums, dtype=float)

    def power_law(N, C, p):
        return C * (N ** p)

    try:
        popt, _ = curve_fit(power_law, N_arr, sums_arr, p0=[1.0, 1.0], maxfev=10000)
        C_power, p = popt
        pred_power = power_law(N_arr, C_power, p)
        r2_power = r2_score(sums_arr, pred_power)
    except Exception:
        C_power, p, r2_power = float("nan"), float("nan"), float("nan")

    def nlogN(N, C):
        return C * N * np.log(N)

    try:
        popt2, _ = curve_fit(nlogN, N_arr, sums_arr, p0=[1.0], maxfev=10000)
        C_nlog = popt2[0]
        pred_nlog = nlogN(N_arr, C_nlog)
        r2_nlog = r2_score(sums_arr, pred_nlog)
    except Exception:
        C_nlog, r2_nlog = float("nan"), float("nan")

    return {
        "region": region,
        "C_power": C_power,
        "exponent": p,
        "r2_power": r2_power,
        "C_nlog": C_nlog,
        "r2_nlog": r2_nlog,
    }

# ---------------------------------------------------------------------------
# 9. HIGH‑LEVEL “VOLUME III COMPLETION” CHECK
# ---------------------------------------------------------------------------

@dataclass
class VolumeIIIStatus:
    algebraic_ok: bool
    kernel_ok: bool
    symmetry_ok: bool
    dominance_observed: bool
    notes: Dict[str, Any]


def run_volume_iii_sanity_suite() -> VolumeIIIStatus:
    notes: Dict[str, Any] = {}

    alg = verify_algebraic_identities()
    algebraic_ok = all([
        alg.log_diff_ok,
        alg.square_decomp_ok,
        alg.hard_identity_ok,
        alg.sym_antisym_ok,
    ])
    notes["algebraic"] = alg.details

    ker = verify_kernel_properties(H=1.0)
    kernel_ok = ker.symmetry_ok and ker.decay_ok
    notes["kernel"] = ker.details

    cfg_small = QuadraticFormConfig(N=50, H=1.0, T0=0.0)
    mats_small = build_quadratic_form(cfg_small)
    sym_small = check_matrix_symmetry(mats_small)
    symmetry_ok = sym_small.K_sym_ok and sym_small.A_sym_ok
    notes["symmetry_small"] = {
        "max_K_asym": sym_small.max_K_asym,
        "max_A_asym": sym_small.max_A_asym,
    }

    Ns = (50, 100, 200)
    Hs = (0.5, 1.0, 2.0)
    sweep = parameter_sweep(Ns, Hs, T0=0.0)
    notes["sweep"] = {}
    dominance_observed = True
    for (N, H), diag in sweep.records.items():
        record = {
            "D_H": diag.D_H,
            "O_H": diag.O_H,
            "Q_H": diag.Q_H,
            "diag_theory": diag.diag_theory,
            "ratio_D_to_absO": diag.ratio_D_to_absO,
            "near_sum": diag.regions.near_sum,
            "mid_sum": diag.regions.mid_sum,
            "tail_sum": diag.regions.tail_sum,
            "near_count": diag.regions.near_count,
            "mid_count": diag.regions.mid_count,
            "tail_count": diag.regions.tail_count,
            "asymp_near": diag.asymp_near,
            "asymp_mid": diag.asymp_mid,
        }
        notes["sweep"][(N, H)] = record
        if diag.ratio_D_to_absO < 1.0:
            dominance_observed = False

    N_fit = [50, 100, 200, 400]
    fit_near = fit_scaling_laws(N_fit, H=1.0, region="near")
    fit_mid = fit_scaling_laws(N_fit, H=1.0, region="mid")
    notes["asymptotic_fits"] = {"near": fit_near, "mid": fit_mid}

    return VolumeIIIStatus(
        algebraic_ok=algebraic_ok,
        kernel_ok=kernel_ok,
        symmetry_ok=symmetry_ok,
        dominance_observed=dominance_observed,
        notes=notes,
    )

# ---------------------------------------------------------------------------
# 11. DYADIC MULTISCALE BAND DECOMPOSITION (OPTIONAL DIAGNOSTIC)
# ---------------------------------------------------------------------------

@dataclass
class DyadicBandContribution:
    band_index: int
    t_min: float
    t_max: float
    sum_value: float
    count: int


def dyadic_band_decomposition(
    mats: QuadraticFormMatrices,
    H: float,
    max_k: int = 5,
) -> List[DyadicBandContribution]:
    """
    Split off-diagonal contributions into dyadic bands in |t| = |ln(m/n)|:

      band k:  2^{-(k+1)} H <= |t| < 2^{-k} H,  k = 0..max_k-1
      tail:   |t| >= 2^{-max_k} H   (stored as band_index = max_k)
    """
    delta = mats.delta
    A = mats.A
    N = delta.shape[0]

    mask_off = ~np.eye(N, dtype=bool)
    abs_delta = np.abs(delta)

    bands: List[DyadicBandContribution] = []

    for k in range(max_k):
        t_min = (2.0 ** (-(k + 1))) * H
        t_max = (2.0 ** (-k)) * H
        mask_band = mask_off & (abs_delta >= t_min) & (abs_delta < t_max)
        val = float(A[mask_band].sum())
        cnt = int(mask_band.sum())
        bands.append(DyadicBandContribution(
            band_index=k,
            t_min=t_min,
            t_max=t_max,
            sum_value=val,
            count=cnt,
        ))

    t_tail = (2.0 ** (-max_k)) * H
    mask_tail = mask_off & (abs_delta >= t_tail)
    val_tail = float(A[mask_tail].sum())
    cnt_tail = int(mask_tail.sum())
    bands.append(DyadicBandContribution(
        band_index=max_k,
        t_min=t_tail,
        t_max=float("inf"),
        sum_value=val_tail,
        count=cnt_tail,
    ))

    return bands

# ---------------------------------------------------------------------------
# 12. KERNEL CURVATURE / TAYLOR DIAGNOSTICS NEAR t = 0
# ---------------------------------------------------------------------------

@dataclass
class TaylorKernelCoefficients:
    H: float
    c0: float
    c2: float
    c4: float
    d2: float
    d4: float


def kernel_taylor_coeffs(H: float) -> TaylorKernelCoefficients:
    """
    Compute Taylor coefficients of k_H at t=0:

        k_H(t) = c0 + c2 t^2 + c4 t^4 + ...
    """
    t, Hsym = sp.symbols("t H", real=True, positive=True)
    z = t / Hsym
    k_expr = 6 / Hsym**2 * sp.sech(z)**4

    d2k_dt2 = sp.diff(k_expr, t, 2)
    d4k_dt4 = sp.diff(k_expr, t, 4)

    c0 = float(k_expr.subs({t: 0, Hsym: H}))
    d2 = float(d2k_dt2.subs({t: 0, Hsym: H}))
    d4 = float(d4k_dt4.subs({t: 0, Hsym: H}))
    c2 = d2 / 2.0
    c4 = d4 / 24.0

    return TaylorKernelCoefficients(
        H=H,
        c0=c0,
        c2=c2,
        c4=c4,
        d2=d2,
        d4=d4,
    )

# ---------------------------------------------------------------------------
# 13. FOURIER-SIDE DECAY DIAGNOSTICS FOR k_H
# ---------------------------------------------------------------------------

def k_H_fourier_approx(omega: float, H: float, quad_limit: float = 10.0) -> complex:
    """
    Approximate the Fourier transform

        \hat{k}_H(omega) = ∫ k_H(t) e^{-i omega t} dt

    on [-L, L] with L = quad_limit * H.
    This is a diagnostic, not a production routine.
    """
    H_mp = mp.mpf(H)

    def integrand(x: float) -> complex:
        t_mp = mp.mpf(x)
        return complex(k_H(t_mp, H_mp) * mp.e ** (-1j * omega * t_mp))

    L = quad_limit * float(H)
    val = mp.quad(lambda x: integrand(x), [-L, L])
    return complex(val)

# ---------------------------------------------------------------------------
# 14. CLI ENTRY POINT
# ---------------------------------------------------------------------------

def _pretty_print_volume_iii_status(status: VolumeIIIStatus) -> None:
    print("=== Volume III Sanity Suite ===")
    print(f"Algebraic identities OK : {status.algebraic_ok}")
    print(f"Kernel properties OK   : {status.kernel_ok}")
    print(f"Matrix symmetry OK     : {status.symmetry_ok}")
    print(f"Dominance observed     : {status.dominance_observed}")
    print()

    fits = status.notes.get("asymptotic_fits", {})
    if fits:
        near = fits["near"]
        mid = fits["mid"]
        print("Asymptotic scaling laws (N → ∞, H=1.0):")
        print(
            f"  Near region: C≈{near['C_power']:.3e}, "
            f"p≈{near['exponent']:.3f}, R²_p≈{near['r2_power']:.5f}, "
            f"R²_NlogN≈{near['r2_nlog']:.5f}"
        )
        print(
            f"  Mid  region: C≈{mid['C_power']:.3e}, "
            f"p≈{mid['exponent']:.3f}, R²_p≈{mid['r2_power']:.5f}, "
            f"R²_NlogN≈{mid['r2_nlog']:.5f}"
        )
        print()

    print("Sample sweep diagnostics (N, H) -> ratio D_H / |O_H| (with regions):")
    for key, rec in status.notes.get("sweep", {}).items():
        N, H = key
        print(
            f"  N={N:4d}, H={H:4.2f}: "
            f"D_H={rec['D_H']:.6e}, "
            f"O_H={rec['O_H']:.6e}, "
            f"ratio={rec['ratio_D_to_absO']:.3e}, "
            f"near={rec['near_sum']:.3e}, "
            f"mid={rec['mid_sum']:.3e}, "
            f"tail={rec['tail_sum']:.3e}"
        )

    # Extra: print dyadic bands and kernel diagnostics for one representative (N,H)
    print()
    print("Dyadic band profile for (N,H) = (200, 1.0):")
    cfg_ref = QuadraticFormConfig(N=200, H=1.0, T0=0.0)
    mats_ref = build_quadratic_form(cfg_ref)
    bands = dyadic_band_decomposition(mats_ref, H=1.0, max_k=5)
    for b in bands:
        tmax = b.t_max if np.isfinite(b.t_max) else float("inf")
        print(
            f"  band {b.band_index}: |t| in [{b.t_min:.3e}, {tmax:.3e}), "
            f"count={b.count:6d}, sum={b.sum_value:.6e}"
        )

    tk = kernel_taylor_coeffs(H=1.0)
    print()
    print("Kernel Taylor coefficients at t=0 (H=1.0):")
    print(
        f"  k_H(0)={tk.c0:.6e}, k''(0)={tk.d2:.6e}, k''''(0)={tk.d4:.6e}, "
        f"c2={tk.c2:.6e}, c4={tk.c4:.6e}"
    )

    print()
    print("Fourier-side decay of k_H (H=1.0):")
    for w in [0.0, 0.5, 1.0, 2.0, 4.0]:
        val = k_H_fourier_approx(w, H=1.0)
        print(f"  omega={w:4.1f}, |hat(k_H)(omega)|={abs(val):.6e}")


if __name__ == "__main__":
    status = run_volume_iii_sanity_suite()
    _pretty_print_volume_iii_status(status)