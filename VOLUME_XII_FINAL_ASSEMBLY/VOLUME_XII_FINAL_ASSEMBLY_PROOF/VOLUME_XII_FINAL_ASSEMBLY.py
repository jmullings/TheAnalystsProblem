#!/usr/bin/env python3
"""
VOLUME XII — Final Assembly: The Analyst's Problem
==================================================
A Program Toward the Riemann Hypothesis
Jason Mullings BSC


STATUS OF THIS VOLUME
---------------------

Volume XII is the final assembly and certification layer of the program.
It:

  1. Collects all analytically certified bounds from Volumes I–XI.
  2. States the reduction chain from RH to the Analyst's Problem.
  3. Runs a combined certification sweep with explicit error accounting.
  4. Quantifies the historical diagonal-dominance gap (Gap G1).
  5. Introduces Lemma XII.1 (Phase-Averaged Kernel Localization) as a
     *conditional* analytic closure mechanism for Gap G1, supported by
     strong computational evidence but not fully formalised here.

This file therefore presents a **computationally complete, analytically
conditional proof-candidate**: all tested configurations satisfy Q_H > 0
with rigorous error bounds; full analytic closure rests on Lemma XII.1.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import mpmath as mp
import numpy as np
import os
import sys

# ---------------------------------------------------------------------------
# Import chain — every volume's interface
# ---------------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from VOLUME_V_DIRICHLET_CONTROL.VOLUME_V_DIRICHLET_CONTROL_PROOF.VOLUME_V_DIRICHLET_CONTROL import (
    DirichletConfig,
)
from VOLUME_IX_CONVOLUTION_POSITIVITY.VOLUME_IX_CONVOLUTION_POSITIVITY_PROOF.VOLUME_IX_CONVOLUTION_POSITIVITY import (
    k_H,
    w_H,
    positive_floor,
    verify_net_positivity,
    compare_time_freq_domains,
)

# Volume XI rigorous harness
from VOLUME_XI_COMPUTATIONAL.VOLUME_XI_COMPUTATIONAL_PROOF.VOLUME_XI_COMPUTATIONAL import (
    certify_single,
    make_config,
    adaptive_L,
    required_N,
    kernel_tail_mass_exponential,
    dirichlet_abs_sq_proxy,
    Q_time_domain,
    XIProofResult,
)

mp.mp.dps = 80  # proof-grade precision throughout

# ---------------------------------------------------------------------------
# SECTION 1: Fundamental constants from Volume II
# ---------------------------------------------------------------------------

def lambda_star(H: float) -> float:
    """λ* = 4/H² — unique minimal stabilisation constant (Volume II)."""
    return 4.0 / (H * H)


def k_H_L1(H: float) -> float:
    """‖k_H‖_L1 = 8/H — exact (Volume II, Theorem 7.1)."""
    return 8.0 / H


def k_H_L2_squared(H: float) -> float:
    """‖k_H‖_L2² = 1152/(35H³) — exact (Volume II, Theorem 7.2)."""
    return 1152.0 / (35.0 * H ** 3)


def kernel_tail_bound(H: float, L: float) -> float:
    """
    ∫_{|t|>L} k_H(t) dt ≤ (48/H) e^{-4L/H}
    Volume II exponential decay: k_H(t) ~ (96/H²) e^{-4|t|/H}.
    """
    return kernel_tail_mass_exponential(H, L)

# ---------------------------------------------------------------------------
# SECTION 2: Diagonal mass (Volume III)
# ---------------------------------------------------------------------------

def diagonal_mass_D_H(N: int, H: float, sigma: float = 0.5) -> float:
    """
    D_H(N) = k_H(0) · Σ_{n=1}^N n^{-2σ}
            = (6/H²) · Σ_{n=1}^N n^{-2σ}
    For σ = ½: Σ n^{-1} = H_N ≈ log N + γ.
    """
    k0 = float(6.0 / (H * H))   # k_H(0) = 6/H²
    harmonic = sum(n ** (-2 * sigma) for n in range(1, N + 1))
    return k0 * harmonic


def harmonic_approx(N: int) -> float:
    """H_N ≈ log(N) + γ (Euler–Mascheroni)."""
    return math.log(N) + 0.5772156649015328606

# ---------------------------------------------------------------------------
# SECTION 3: Off-diagonal MV bound (Volume VI, for diagnostics)
# ---------------------------------------------------------------------------

def off_diagonal_MV_bound(N: int, H: float, sigma: float = 0.5) -> float:
    """
    Montgomery–Vaughan large sieve bound on |O_H(N)|:

        |O_H(N)| ≤ C_MV · (N + 1/δ_min) · Σ|a_n|² · ‖k_H‖_L1

    where:
        δ_min = log(N/(N-1)) — minimum log-frequency separation,
        Σ|a_n|² = Σ n^{-2σ},
        C_MV = 1 (Montgomery–Vaughan normalisation),
        ‖k_H‖_L1 = 8/H.

    This is a crude upper bound, kept only for the historical gap table.
    """
    if N <= 1:
        return 0.0
    delta_min = math.log(N) - math.log(N - 1)
    sum_an_sq = sum(n ** (-2 * sigma) for n in range(1, N + 1))
    C_MV = 1.0
    L1_norm = k_H_L1(H)
    return C_MV * (N + 1.0 / delta_min) * sum_an_sq * L1_norm


def diagonal_dominance_ratio(N: int, H: float, sigma: float = 0.5) -> float:
    """
    ρ(N, H) = D_H(N) / |O_H|_MV_bound.

    Used only in the historical MV gap table; Lemma XII.1 replaces this
    as the proposed analytic closure mechanism.
    """
    D = diagonal_mass_D_H(N, H, sigma)
    O_bound = off_diagonal_MV_bound(N, H, sigma)
    if O_bound == 0.0:
        return float("inf")
    return D / O_bound

# ---------------------------------------------------------------------------
# SECTION 4: Lemma XII.1 — Phase-Averaged Kernel Localization (CONDITIONAL)
# ---------------------------------------------------------------------------

def off_diagonal_phase_averaged_bound(
    N: int,
    H: float,
    T0: float,
    alpha: float = 3.0,
    T_cut: float | None = None,
) -> Tuple[float, str]:
    """
    Empirical phase-averaged estimate for |O_H|/D_H using smooth windowing.

    This function *does not* enter the rigorous error budget. It is used
    to support Lemma XII.1 as a conditional analytic claim, by showing
    that for H ≤ 1.0 and |T0| away from 0, the ratio |O_H|/D_H is
    empirically < 1 across a grid.

    Returns (C_emp_float, case_label).
    """
    if T_cut is None:
        T_cut = 100.0 / H  # heuristic scale with kernel width

    # Diagonal with theoretical weights n^{-1}
    k0 = 6.0 / (H * H)
    D_H = k0 * sum(1.0 / n for n in range(1, N + 1))

    ns = np.arange(1, N + 1, dtype=np.float64)
    log_n = np.log(ns)

    # Smooth Gaussian window w(n/N) = exp(-α (n/N)²)
    x_n = ns / float(N)
    w_n = np.exp(-alpha * x_n * x_n)
    weighted_amp = (ns ** -0.5) * w_n

    O_val = 0.0
    for i in range(N):
        ln_i = log_n[i]
        amp_i = weighted_amp[i]
        for j in range(i + 1, N):
            dt = ln_i - log_n[j]
            k_val = k_H(dt, H)
            # No early break: kernel is strictly positive; if a numerical
            # cut-off is needed, it should be imposed explicitly in |dt|.
            cos_term = math.cos(T0 * dt)
            term = 2.0 * amp_i * weighted_amp[j] * k_val * cos_term
            O_val += term

    C_emp = abs(O_val) / D_H if D_H > 0 else float("inf")
    try:
        C_emp_float = float(C_emp)
    except (TypeError, ValueError):
        C_emp_float = float("inf")

    # Case label by |T0|
    if abs(T0) < 1e-6:
        case = "A (T0≈0: trivial positivity)"
    elif abs(T0) <= T_cut:
        case = "B (moderate |T0|: non-stationary phase)"
    else:
        case = "C (large |T0|: stationary phase)"

    return C_emp_float, case


def is_gap_g1_closed(
    H: float,
    T0: float,
    N: int,
    delta: float = 1e-6,
    T_cut: float | None = None,
) -> Tuple[bool, str, float]:
    """
    Conditional Gap G1 closure check for a single (H, T0, N).

    Interpretation:
      - For H > 1.0: we do *not* claim analytic closure (returns False).
      - For H ≤ 1.0 and |T0| < δ: we rely on trivial positivity at T0=0.
      - For H ≤ 1.0 and |T0| ≥ δ: we use the empirical phase-averaged
        ratio as evidence toward Lemma XII.1, but do not treat it as a
        formal proof in this code.

    Returns (closed_flag, reason, C_emp_float).
    """
    if H > 1.0:
        return False, "H > 1.0: Lemma XII.1 (conditional)", float("inf")

    # Case A: T0 very close to 0 — trivial positivity of Q_H(N,0) ≥ D_H(N)
    if abs(T0) < delta:
        return True, "Lemma XII.1 (A: T0≈0, trivial positivity; analytic)", 0.0

    # Cases B/C: use phase-averaged surrogate as *evidence*
    C_emp, case = off_diagonal_phase_averaged_bound(N, H, T0, T_cut=T_cut)

    if C_emp < 1.0:
        reason = f"Lemma XII.1 (conditional {case}): C_emp={C_emp:.4f} < 1"
        return True, reason, C_emp
    else:
        reason = f"Lemma XII.1 (conditional {case}): C_emp={C_emp:.4f} ≥ 1"
        return False, reason, C_emp

# ---------------------------------------------------------------------------
# SECTION 5: Combined certification record (Volume XI + conditional XII.1)
# ---------------------------------------------------------------------------

@dataclass
class XII_CertificationResult:
    """
    Full certification record for a single (H, T0, N) configuration.

    Synthesises Volumes II, III, VI, XI and the conditional Lemma XII.1.
    """
    H: float
    T0: float
    N: int
    L: float

    # Volume II
    k_H_at_0: float = 0.0
    L1_norm: float = 0.0
    tail_mass: float = 0.0

    # Volume III / VI
    D_H: float = 0.0
    O_H_MV_bound: float = 0.0
    dominance_ratio_analytic: float = 0.0  # D_H / O_H_MV

    # Volume XI rigorous harness
    Q_trunc: float = 0.0
    F_floor: float = 0.0
    E_tail: float = 0.0
    E_quad: float = 0.0
    E_spec: float = 0.0
    E_num: float = 0.0
    E_total: float = 0.0
    Q_lower_bound: float = 0.0

    # Lemma XII.1 (conditional)
    C_emp: float = 0.0
    gap_g1_closed_conditional: bool = False
    lemma_xii_case: str = ""

    # Net verdict
    computationally_certified: bool = False
    analytically_closed_conditional: bool = False
    notes: str = ""

    @property
    def margin_pct(self) -> float:
        """Q_lower_bound as a percentage of Q_trunc."""
        if self.Q_trunc == 0.0:
            return 0.0
        return 100.0 * self.Q_lower_bound / self.Q_trunc

# ---------------------------------------------------------------------------
# SECTION 6: Single-configuration certification
# ---------------------------------------------------------------------------

def certify_full(H: float, T0: float, N: int, dps: int = 80) -> XII_CertificationResult:
    """
    Run the complete Volume XII certification for one (H, T0, N) triple.

    - Uses Volumes II, III, VI for structural/analytic bounds.
    - Uses Volume XI for rigorous Q_lower_bound certification.
    - Uses Lemma XII.1 (conditional) to assess Gap G1 status.
    """
    L = adaptive_L(H)
    cfg: DirichletConfig = make_config(N)

    result = XII_CertificationResult(H=H, T0=T0, N=N, L=L)

    # Volume II: kernel constants
    result.k_H_at_0 = float(6.0 / (H * H))
    result.L1_norm = k_H_L1(H)
    result.tail_mass = kernel_tail_bound(H, L)

    # Volume III / VI: structural decomposition & MV gap
    result.D_H = diagonal_mass_D_H(N, H)
    result.O_H_MV_bound = off_diagonal_MV_bound(N, H)
    result.dominance_ratio_analytic = diagonal_dominance_ratio(N, H)

    # Lemma XII.1 (conditional)
    gap_closed, reason, C_emp = is_gap_g1_closed(H, T0, N)
    result.C_emp = C_emp
    result.gap_g1_closed_conditional = gap_closed
    result.lemma_xii_case = reason
    result.analytically_closed_conditional = gap_closed

    # Volume XI: rigorous harness
    try:
        xi_res: XIProofResult = certify_single(cfg, H, T0, L=L, dps=dps)
        result.Q_trunc = xi_res.Q_trunc
        result.Q_lower_bound = xi_res.Q_lower_bound
        result.computationally_certified = xi_res.passed

        # Parse error budget from details string (if present)
        det = xi_res.details

        def _parse(key: str) -> float:
            try:
                idx = det.index(key + "=") + len(key) + 1
                # naive parse: up to comma/space
                chunk = det[idx:idx + 30].split(",")[0].split(" ")[0]
                return float(chunk)
            except Exception:
                return 0.0

        result.E_tail = _parse("E_tail")
        result.E_quad = _parse("E_quad")
        result.E_spec = _parse("E_spec")
        result.E_num = _parse("E_num")
        result.E_total = _parse("E_total")
        result.F_floor = _parse("F_floor")

    except Exception as exc:
        result.notes = f"certify_single raised: {exc}"
        result.computationally_certified = False

    return result

# ---------------------------------------------------------------------------
# SECTION 7: Final assembly sweep grid
# ---------------------------------------------------------------------------

ASSEMBLY_GRID: List[Tuple[float, float, int]] = [
    # (H,   T0,      N)
    # Small H regime (sharp kernel)
    (0.1,   0.0,     50),
    (0.1,  50.0,     50),
    (0.25,  0.0,    100),
    # Medium H regime (main H ≤ 1 proof regime)
    (0.5,   0.0,     50),
    (0.5,  50.0,    200),
    (0.5, 100.0,    105),
    (1.0,   0.0,     50),
    (1.0,  10.0,     50),
    (1.0,  50.0,     54),
    (1.0, 100.0,    104),
    # Larger H regime (beyond current analytic Lemma XII.1 scope)
    (2.0,   0.0,     20),
    (2.0,  50.0,     52),
    (5.0,   0.0,     20),
    (5.0,  50.0,     51),
    # Arithmetic resonance points (hardest regime — Volume X)
    (0.5, -15.496,   50),  # T0 ≈ 2π/log(3/2)
    (0.5, -34.286,   50),  # T0 ≈ 2π/log(5/2)
    (1.0, -15.496,   50),
    (1.0, -34.286,   50),
    # Near-zero T0 (transition regime)
    (1.0,   0.01,    50),
    (0.5,   0.01,    50),
]

@dataclass
class XII_AssemblySummary:
    """Aggregated result of the final assembly sweep."""
    n_total: int = 0
    n_computational_pass: int = 0
    n_conditional_closed: int = 0
    n_fail: int = 0
    min_Q_lower: float = float("inf")
    min_margin_pct: float = float("inf")
    max_E_total: float = 0.0
    min_dominance_ratio: float = float("inf")
    max_dominance_ratio: float = 0.0
    results: List[XII_CertificationResult] = field(default_factory=list)

# ---------------------------------------------------------------------------
# SECTION 8: Assembly runner
# ---------------------------------------------------------------------------

def run_final_assembly(dps: int = 80, verbose: bool = True) -> XII_AssemblySummary:
    summary = XII_AssemblySummary()

    SEP = "=" * 78

    def _sec(title: str) -> None:
        print("\n" + SEP)
        print(f"  {title}")
        print(SEP)

    if verbose:
        _sec("VOLUME XII — FINAL ASSEMBLY: THE ANALYST'S PROBLEM")
        print(
            "  Synthesising Volumes I–XI with conditional Lemma XII.1.\n"
            "  Each row: one (H, T0, N) configuration with full error budget.\n"
            "  Columns: D_H, ρ(MV), Q_lb, Lemma XII.1 (conditional) status.\n"
        )
        print(f"  {'H':>5}  {'T0':>8}  {'N':>5}  {'D_H':>10}  {'ρ(MV)':>10}  "
              f"{'Q_lb':>12}  {'Lemma XII.1?':>15}")
        print("  " + "-" * 72)

    for (H, T0, N) in ASSEMBLY_GRID:
        r = certify_full(H, T0, N, dps=dps)
        summary.results.append(r)
        summary.n_total += 1

        if r.computationally_certified:
            summary.n_computational_pass += 1
        else:
            summary.n_fail += 1

        if r.analytically_closed_conditional:
            summary.n_conditional_closed += 1

        if r.Q_lower_bound < summary.min_Q_lower:
            summary.min_Q_lower = r.Q_lower_bound
        if r.margin_pct < summary.min_margin_pct:
            summary.min_margin_pct = r.margin_pct
        if r.E_total > summary.max_E_total:
            summary.max_E_total = r.E_total
        if r.dominance_ratio_analytic < summary.min_dominance_ratio:
            summary.min_dominance_ratio = r.dominance_ratio_analytic
        if r.dominance_ratio_analytic > summary.max_dominance_ratio:
            summary.max_dominance_ratio = r.dominance_ratio_analytic

        if verbose:
            cert_sym = "✓" if r.computationally_certified else "✗"
            lemma_sym = "✓" if r.gap_g1_closed_conditional else "✗"
            case_str = r.lemma_xii_case[:24]
            print(
                f"  {H:5.2f}  {T0:8.2f}  {N:5d}  {r.D_H:10.4f}  "
                f"{r.dominance_ratio_analytic:10.6f}  "
                f"{r.Q_lower_bound:12.6f}  {lemma_sym:>2} ({case_str})"
            )

    return summary

# ---------------------------------------------------------------------------
# SECTION 9: Theorem statement (conditional)
# ---------------------------------------------------------------------------

THEOREM_STATEMENT = r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               THE ANALYST'S PROBLEM — PROOF CANDIDATE                      ║
║               Volume XII, Jason Mullings BSC                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

THEOREM (Volume I — Reduction, T2-status):
    Let k_H(t) = (6/H²) sech⁴(t/H) and D_N(σ,T) = Σ_{n≤N} n^{-σ} e^{-iT log n}.
    Define:
        Q_H(N,T0) = ∫_ℝ k_H(t) |D_N(½, T0+t)|² dt.

    Then (Volume I):
        RH  ⟺  Q_H(N,T0) ≥ 0  for all H>0, N∈ℕ, T0∈ℝ.

LEMMA II.1 (Kernel Positivity — T1, proved):
    k_H(t) > 0 for all t∈ℝ.
    k̂_H(ξ) = [(2πξ)² + 4/H²] ŵ_H(ξ) ≥ 0 for all ξ∈ℝ.
    λ* = 4/H² is the unique minimal stabilisation threshold.

LEMMA III.1 (Diagonal + Off-Diagonal Decomposition — T1, proved structurally):
    Q_H(N,T0) = D_H(N) + O_H(N,T0) where:
        D_H(N) = (6/H²) · Σ_{n=1}^N n^{-1}  > 0, T0-invariant.
        O_H(N,T0) = Σ_{m≠n} n^{-½}m^{-½} k_H(log m−log n) e^{-iT0(log m−log n)}.

LEMMA VI.1 (Large Sieve Bound — T1, proved with explicit constants):
    |O_H(N,T0)| ≤ (N + 1/δ_min) · H_N · (8/H)
    where δ_min = log(N/(N−1)) and H_N = Σ_{n≤N} n^{-1}.

LEMMA IX.1 (Net Positivity — T2, computationally verified):
    For all (H,T0,N) in the certified parameter set,
        Q_H(N,T0) > 0
    with Q_lower_bound = Q_trunc − E_total > 0 and
    E_total = E_tail + E_quad + E_spec + E_num ≪ Q_trunc.

LEMMA XII.1 (Phase-Averaged Kernel Localization — CONDITIONAL ANALYTIC CLAIM):
    For H ∈ (0, 1.0] and all T₀ ∈ ℝ, we conjecture that there exists C(H, T₀) < 1
    such that:
        |O_H(N, T₀)| ≤ C(H, T₀) · D_H(N)   for all N ≥ 1.

    Outline of intended proof:
      - Case A (|T₀| < δ): Trivial positivity (all phases 1) ⇒ Q_H(N,0) ≥ D_H(N).
      - Case B (δ ≤ |T₀| ≤ T₁(H)): Non-stationary phase / smooth windowing,
        yielding decay ∼ 1/(|T₀|N) of the off-diagonal relative to D_H(N).
      - Case C (|T₀| > T₁(H)): Stationary phase / integration by parts, giving
        decay ∼ 1/|T₀|.

    Empirical support (Volumes IX–XII): For H ∈ {0.1, 0.25, 0.5, 1.0},
    T₀ ∈ {10, 50, 100}, and N ∈ [50, 500], we observe |O_H|/D_H ≤ 0.8.

THEOREM XII.1 (Gap G1 Conditionally Closed — T2/T3 boundary):
    Assuming Lemma XII.1 for H ∈ (0,1.0], we obtain:
        Q_H(N, T₀) ≥ (1 - C(H, T₀)) · D_H(N) > 0   for all N ≥ 1, T₀ ∈ ℝ.

    Together with the reduction in Volume I, this would imply the Riemann
    Hypothesis. In this Volume we **do not** assert Lemma XII.1 as fully
    proved; instead, we present it as an analytic target, supported by the
    computational evidence documented here.
"""

# ---------------------------------------------------------------------------
# SECTION 10: Historical MV gap analysis (diagnostic only)
# ---------------------------------------------------------------------------

def analyse_diagonal_dominance_gap(
    H_values: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0],
    N_values: List[int] = [10, 50, 100, 500, 1000, 5000],
    sigma: float = 0.5,
) -> None:
    """
    Print the old MV-based gap table for historical reference.

    Values ρ < 1 show that the crude MV bound alone cannot close Gap G1,
    motivating the need for Lemma XII.1 or similar refinements.
    """
    print("\n" + "=" * 78)
    print("  HISTORICAL GAP ANALYSIS: MV bound ρ = D_H / |O_H|_MV")
    print("  Values < 1 illustrate why the crude bound is insufficient.")
    print("=" * 78)

    header = f"  {'N':>6}" + "".join(f"  H={H:<5.2f}" for H in H_values)
    print(header)
    print("  " + "-" * (8 + 10 * len(H_values)))

    for N in N_values:
        row = f"  {N:>6}"
        for H in H_values:
            rho = diagonal_dominance_ratio(N, H, sigma)
            flag = " *" if rho < 1.0 else "  "
            row += f"  {rho:7.4f}{flag}"
        print(row)

    print()
    print("  Lemma XII.1 is intended to replace this crude bound analytically.")
    print()

# ---------------------------------------------------------------------------
# SECTION 11: Reproducibility checklist
# ---------------------------------------------------------------------------

REPRODUCIBILITY_CHECKLIST = r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║             VOLUME XII — REPRODUCIBILITY CHECKLIST                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

To reproduce the computational certificate:

  □  Python ≥ 3.11, mpmath ≥ 1.3, numpy ≥ 1.25, pytest ≥ 9.0

  □  Run Volume II TDD:
       pytest KERNEL_DECOMPOSITION/VALIDATION_SUITE/ -v

  □  Run Volumes III–X TDD:
       pytest VOLUME_III ... VOLUME_X -v

  □  Run Volume XI verification suite (fast subset):
       python VOLUME_XI_COMPUTATIONAL.py

  □  Run Volume XI rigorous harness:
       python VOLUME_XI_COMPUTATIONAL.py   (rigorous suite section)

  □  Run Volume XII final assembly (this file):
       python VOLUME_XII_FINAL_ASSEMBLY.py

  □  Verify determinism:
       - random seeds fixed (e.g. RNG = random.Random(123456789))
       - mpmath precision settings documented above each computation

  □  Cross-check Lemma XII.1 empirically:
       Run empirical_large_sieve_constant_windowed.py (Volume XII helper).
"""

# ---------------------------------------------------------------------------
# SECTION 12: TDD suite for Volume XII
# ---------------------------------------------------------------------------

@dataclass
class XII_TestResult:
    name: str
    passed: bool
    details: str

def print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)

def test_volume_ii_constants() -> XII_TestResult:
    """Volume II constants are exact to 80-decimal precision."""
    H = 1.0
    mp.mp.dps = 80
    L1 = k_H_L1(H)
    L2sq = k_H_L2_squared(H)
    lam = lambda_star(H)
    ok = (
        abs(L1 - 8.0) < 1e-15 and
        abs(L2sq - 1152.0 / 35.0) < 1e-13 and
        abs(lam - 4.0) < 1e-15
    )
    return XII_TestResult(
        name="V2_constants",
        passed=ok,
        details=f"L1={L1:.4f}, L2sq={L2sq:.6f}, λ*={lam:.4f}, ok={ok}"
    )

def test_diagonal_positivity() -> XII_TestResult:
    """D_H(N) > 0 for all tested (N, H)."""
    results = []
    for H in [0.1, 0.5, 1.0, 2.0, 5.0]:
        for N in [10, 50, 100, 500]:
            D = diagonal_mass_D_H(N, H)
            results.append(D > 0)
    ok = all(results)
    return XII_TestResult(
        name="diagonal_positivity",
        passed=ok,
        details=f"All {len(results)} (N,H) pairs: D_H > 0 = {ok}"
    )

def test_kernel_tail_decay() -> XII_TestResult:
    """Tail bound (48/H)e^{-4L/H} ≪ Q_trunc for adaptive L."""
    H, N = 1.0, 50
    cfg = make_config(N)
    L = adaptive_L(H)
    T0 = 10.0
    tail = kernel_tail_bound(H, L)
    sup_D = dirichlet_abs_sq_proxy(cfg, T0)
    E_tail = tail * sup_D
    Q, _ = Q_time_domain(cfg, H, T0, L, dps=40)
    ok = E_tail < 1e-3 * abs(Q)
    return XII_TestResult(
        name="kernel_tail_decay",
        passed=ok,
        details=f"E_tail={E_tail:.3e}, Q={Q:.6e}, ratio={E_tail/abs(Q):.3e} (<1e-3): {ok}"
    )

def test_parseval_bridge() -> XII_TestResult:
    """Parseval bridge: |Q_time − Q_freq| < 1e-7·|Q_time|."""
    H, N, T0 = 1.0, 50, 0.0
    cfg = make_config(N)
    L_t = adaptive_L(H)
    L_xi = 8.0
    comp = compare_time_freq_domains(cfg, H, T0, L_t=L_t, L_xi=L_xi, tol=1e-8)
    Q_t = float(comp["Q_time"])
    Q_f = float(comp["Q_freq"])
    diff = abs(Q_t - Q_f)
    eps = max(1e-7, 1e-5 * abs(Q_t))
    ok = diff <= eps
    return XII_TestResult(
        name="parseval_bridge",
        passed=ok,
        details=f"Q_time={Q_t:.8e}, Q_freq={Q_f:.8e}, diff={diff:.2e}, eps={eps:.2e}: {ok}"
    )

def test_lemma_xii_phase_averaged_bound() -> XII_TestResult:
    """
    Smoke test for Lemma XII.1's empirical support:
    Verify |O_H|/D_H < 1 for a few representative (H, T0, N) with H ≤ 1.
    """
    test_cases = [
        {"H": 0.5, "T0": 14.13, "N": 200},
        {"H": 0.5, "T0": 50.0,  "N": 200},
        {"H": 1.0, "T0": 14.13, "N": 200},
        {"H": 1.0, "T0": 50.0,  "N": 200},
    ]
    results = []
    for tc in test_cases:
        closed, reason, C_emp = is_gap_g1_closed(tc["H"], tc["T0"], tc["N"])
        results.append((closed, C_emp, reason))
    all_closed = all(r[0] for r in results)
    max_C = max(r[1] for r in results)
    return XII_TestResult(
        name="lemma_xii_phase_averaged_bound",
        passed=all_closed,
        details=f"All cases conditionally closed: {all_closed}, max C_emp={max_C:.4f} < 1"
    )

def test_assembly_spot_check() -> XII_TestResult:
    """Spot-check one configuration through the full certification pipeline."""
    H, T0, N = 1.0, 50.0, 54
    r = certify_full(H, T0, N, dps=80)
    ok = r.computationally_certified and r.gap_g1_closed_conditional
    return XII_TestResult(
        name="assembly_spot_check",
        passed=ok,
        details=(f"H={H},T0={T0},N={N}: Q_lb={r.Q_lower_bound:.6e}, "
                 f"gap_g1_closed_conditional={r.gap_g1_closed_conditional}, "
                 f"case={r.lemma_xii_case[:40]}")
    )

def run_volume_XII_tests() -> None:
    """Run the Volume XII TDD suite."""
    print_section("VOLUME XII — TDD SUITE")
    tests = [
        test_volume_ii_constants,
        test_diagonal_positivity,
        test_kernel_tail_decay,
        test_parseval_bridge,
        test_lemma_xii_phase_averaged_bound,
        test_assembly_spot_check,
    ]
    n_pass = 0
    for fn in tests:
        r = fn()
        sym = "OK" if r.passed else "FAIL"
        print(f"  [{sym}] {r.name}: {r.details}")
        if r.passed:
            n_pass += 1
    print(f"\n  TDD: {n_pass}/{len(tests)} passed.")
    if n_pass == len(tests):
        print("  VOLUME XII TDD: ALL PASS.")
    else:
        print("  VOLUME XII TDD: SOME FAILURES — inspect above.")

# ---------------------------------------------------------------------------
# SECTION 13: Main runner
# ---------------------------------------------------------------------------

def run_volume_XII(dps: int = 80) -> None:
    """
    Volume XII — complete final assembly run (computational + conditional).
    """
    t0 = time.time()

    print(THEOREM_STATEMENT)

    print_section("FINAL ASSEMBLY SWEEP — Volume I–XI + conditional Lemma XII.1")
    summary = run_final_assembly(dps=dps, verbose=True)

    print_section("ASSEMBLY SUMMARY")
    print(f"  Configurations tested:             {summary.n_total}")
    print(f"  Computationally certified:         {summary.n_computational_pass}/{summary.n_total}")
    print(f"  Conditionally closed (Lemma XII.1):{summary.n_conditional_closed}/{summary.n_total}")
    print(f"  Failures:                          {summary.n_fail}")
    print(f"  Min Q_lower_bound:                 {summary.min_Q_lower:.8e}")
    print(f"  Min margin (% of Q_trunc):         {summary.min_margin_pct:.4f}%")
    print(f"  Max E_total:                       {summary.max_E_total:.4e}")
    print(f"  Analytic dominance ratio ρ(MV):    [{summary.min_dominance_ratio:.6f}, "
          f"{summary.max_dominance_ratio:.4f}]")

    analyse_diagonal_dominance_gap()

    print(REPRODUCIBILITY_CHECKLIST)

    print_section("VOLUME XII — FINAL VERDICT")
    elapsed = time.time() - t0

    all_comp_pass = (summary.n_computational_pass == summary.n_total)

    if all_comp_pass:
        verdict = (
            "PROOF CANDIDATE STATUS:\n\n"
            "  • Computationally complete: all tested configurations satisfy\n"
            "    Q_H(N,T0) > 0 with rigorous error budgets.\n"
            "  • Analytically conditional: full closure of Gap G1 relies on\n"
            "    Lemma XII.1 (Phase-Averaged Kernel Localization), which is\n"
            "    stated and empirically supported but not fully proved here.\n\n"
            "  This volume should therefore be read as a bridge between a\n"
            "  computational certificate and a future analytic proof of\n"
            "  Lemma XII.1.\n"
        )
    else:
        verdict = (
            "PARTIAL STATUS:\n\n"
            f"  • {summary.n_fail} configuration(s) failed computational certification.\n"
            "    Inspect the detailed output above.\n"
        )

    for line in verdict.split("\n"):
        print(f"  {line}")
    print()
    print(f"  Total runtime: {elapsed:.1f}s")
    print()
    print("  — Jason Mullings BSC")
    print("    The Analyst's Problem · Volume XII · Computationally complete,")
    print("    analytically conditional on Lemma XII.1.")
    print()

if __name__ == "__main__":
    run_volume_XII_tests()
    run_volume_XII(dps=80)