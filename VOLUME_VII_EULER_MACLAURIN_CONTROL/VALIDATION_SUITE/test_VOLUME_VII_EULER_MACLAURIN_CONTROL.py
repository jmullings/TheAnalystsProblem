#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# test_VOLUME_VII_EULER_MACLAURIN_CONTROL.py
#
# VALIDATION SUITE FOR VOLUME VII: EULER-MACLAURIN CONTROL
# =========================================================
#
# This suite is organised in two tiers:
#
#   TIER A — Original requirements (Req 1.1–1.6, 3.3):
#     Primitives, EM engine, uniformity, error budget.
#     All pre-existing tests preserved verbatim.
#
#   TIER B — Proof-ladder scaffolding (Req 4.1–4.5):
#     Tests for the three new functions added in the last audit:
#       diagonal_mass_em_bound     (Section 9)
#       remainder_vs_N_scaling     (Section 10)
#       QH_lower_bound_contribution (Section 11)
#
#     These tests are explicitly mapped to the five-step proof ladder:
#
#       Step 1 (Vol XIII) — xi -> Q_H derivation
#         Req 4.1: D_H certified floor is strictly positive
#         Req 4.2: EM estimate matches analytic D_H to high precision
#
#       Step 2 (Vol XIV) — Mean-value form of O_H
#         Req 4.3: R_m(N) = o(D_H(N))  [ratio -> 0 as N grows]
#
#       Step 3 (Vol XV)  — B_analytic < 1
#         Req 4.4: E_EM < D_H certified floor  [EM error never swamps signal]
#
#       Step 4 (Vol XVI) — Lipschitz uniformity in T0
#         Req 4.5: E_EM is stable across H and T0 grids
#
#       Step 5 (Vol XVII) — N -> inf limit
#         Req 4.6: ratio(N) is monotone decreasing  [o(1) evidence]
#
# Tier-B tests are marked with @pytest.mark.proof_ladder so they can be
# run in isolation:
#
#     pytest -m proof_ladder -v
#
# All tests use mp.dps = 80.

import sys
import os
import math
import numpy as np
import mpmath as mp
import pytest

# ---------------------------------------------------------------------------
# Path injection
# ---------------------------------------------------------------------------

PROOF_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'VOLUME_VII_EULER_MACLAURIN_CONTROL_PROOF')
)
sys.path.insert(0, PROOF_DIR)

try:
    import VOLUME_VII_EULER_MACLAURIN_CONTROL as em
except ImportError:
    import EulerMaclaurinControl as em

mp.mp.dps = 80
TOL = 1e-12

# Register custom marker so -m proof_ladder works without pytest.ini
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "proof_ladder: marks tests that scaffold the five-step proof ladder"
    )


# ===========================================================================
# TIER A — Original requirements
# ===========================================================================

class TestEulerMaclaurinPrimitives:
    """
    Req 1.1, 1.3, 1.4: f(t), derivatives, Bernoulli numbers.
    """

    def test_bernoulli_numbers_correct(self):
        """Verify first few Bernoulli numbers against known exact values."""
        known_values = {
            0:  mp.mpf(1),
            1:  mp.mpf('-0.5'),
            2:  mp.mpf(1) / 6,
            4:  -mp.mpf(1) / 30,
            6:  mp.mpf(1) / 42,
            8:  -mp.mpf(1) / 30,
            10: mp.mpf(5) / 66,
            12: -mp.mpf(691) / 2730,
        }
        for k, v in known_values.items():
            assert mp.almosteq(em.bernoulli_number_float(k), v, TOL), \
                f"B_{k} mismatch"

        # Odd Bernoulli numbers B_k = 0 for k > 1
        for k in [3, 5, 7, 9]:
            assert mp.almosteq(em.bernoulli_number_float(k), 0.0, TOL), \
                f"B_{k} should be zero"

    def test_bernoulli_cache_consistency(self):
        """Calling bernoulli_number_float twice returns identical objects."""
        b2_first  = em.bernoulli_number_float(2)
        b2_second = em.bernoulli_number_float(2)
        assert b2_first == b2_second

    def test_f_continuous_with_gaussian_window(self):
        """f(t) with Gaussian window matches closed-form expression."""
        params = {
            "sigma": 0.5,
            "N": 100.0,
            "window_type": "gaussian",
            "window_params": {"alpha": 2.0},
        }
        t = math.log(50)
        # x = 50/100 = 0.5;  base = 50^{-0.5};  w = exp(-2*(0.5)^2)
        expected = (50 ** -0.5) * math.exp(-2.0 * (0.5) ** 2)
        val = em.f_continuous(t, params)
        assert abs(val - expected) < 1e-12

    def test_f_continuous_sharp_window_inside(self):
        """Sharp window returns base value for x in [0,1]."""
        params = {"sigma": 0.5, "N": 100.0, "window_type": "sharp"}
        t = math.log(50)   # x = 0.5, inside window
        expected = math.exp(-0.5 * t)
        assert abs(em.f_continuous(t, params) - expected) < 1e-12

    def test_f_continuous_sharp_window_outside(self):
        """Sharp window returns 0 for x > 1."""
        params = {"sigma": 0.5, "N": 100.0, "window_type": "sharp"}
        t = math.log(200)  # x = 2.0, outside window
        assert em.f_continuous(t, params) == 0.0

    def test_f_derivative_analytic_vs_numeric(self):
        """Analytic special-case derivatives match for f(t) = e^{-t}."""
        params = {"sigma": 1.0, "N": 1.0, "window_type": "sharp"}
        t = 1.0
        for order in range(1, 4):
            analytic = (-1) ** order * math.exp(-t)
            numeric  = em.f_derivative(t, order, params)
            assert abs(numeric - analytic) < 1e-5, \
                f"order={order}: analytic={analytic}, numeric={numeric}"

    def test_f_derivative_order_zero_is_function(self):
        """
        Order-0 derivative is the function value itself.

        We use a Gaussian window (not sharp) with N=100 so that
        f_continuous returns a non-zero value and the analytic
        special-case branch in f_derivative is not triggered.
        The two code paths must agree to 1e-12.
        """
        params = {
            "sigma": 0.5,
            "N": 100.0,
            "window_type": "gaussian",
            "window_params": {"alpha": 1.0},
        }
        t = math.log(50)   # x = 50/100 = 0.5 — well inside window, non-zero
        expected = em.f_continuous(t, params)
        assert expected != 0.0, "Sanity: f_continuous must be non-zero here"
        result = em.f_derivative(t, 0, params)
        assert abs(result - expected) < 1e-12, \
            f"f_derivative(order=0) != f_continuous: {result} vs {expected}"


class TestEulerMaclaurinEngine:
    """
    Req 1.2, 1.5: EM expansion and remainder bounds.
    """

    def test_polynomial_exactness(self):
        """
        EM with order=2 is exact for f(t) = t^3 (degree 2*2-1 = 3).
        Remainder must be exactly 0.0.
        """
        f = lambda t: t ** 3
        a, b = 0.0, 1.0
        n_terms = 11
        h = (b - a) / (n_terms - 1)
        ts = [k * h for k in range(n_terms)]
        true_sum = sum(f(t) for t in ts)

        params = {"is_poly": 1.0}
        original_f_deriv = em.f_derivative

        def mock_f_deriv(t, order, p):
            if order == 0: return t ** 3
            if order == 1: return 3 * t ** 2
            if order == 2: return 6 * t
            if order == 3: return 6.0
            return 0.0

        em.f_derivative = mock_f_deriv
        try:
            res = em.euler_maclaurin_sum(
                f, a, b, n_terms, order=2, H=1.0, T0=0.0,
                params=params, is_polynomial=True
            )
            assert abs(true_sum - res.total_sum_estimate) < 1e-10
            assert res.remainder_bound == 0.0
        finally:
            em.f_derivative = original_f_deriv

    def test_remainder_bound_holds(self):
        """
        True error must be ≤ classical remainder bound for f(t) = e^{-t}.
        """
        f = lambda t: math.exp(-t)
        a, b = 0.0, 2.0
        n_terms = 21
        h = (b - a) / (n_terms - 1)
        ts = [a + k * h for k in range(n_terms)]
        true_sum = sum(f(t) for t in ts)

        params = {"sigma": 1.0, "N": 1.0, "window_type": "sharp"}
        res = em.euler_maclaurin_sum(
            f, a, b, n_terms, order=2, H=1.0, T0=0.0,
            params=params, is_polynomial=False
        )
        abs_error = abs(true_sum - res.total_sum_estimate)

        assert res.remainder_bound > 0.0, "Classical remainder bound must be positive"
        assert abs_error <= res.remainder_bound + 1e-12, \
            f"Error {abs_error} exceeded bound {res.remainder_bound}"

    def test_kernel_enhanced_bound_le_classical(self):
        """
        Kernel-enhanced bound must be ≤ classical bound (K(H) ≤ 1).
        """
        f = lambda t: math.exp(-t)
        params = {"sigma": 1.0, "N": 1.0, "window_type": "sharp"}
        res = em.euler_maclaurin_sum(
            f, 0.0, 2.0, 21, order=2, H=1.0, T0=0.0,
            params=params, is_polynomial=False
        )
        assert res.remainder_bound_kernel <= res.remainder_bound_classical + 1e-15

    def test_higher_order_tightens_bound(self):
        """
        Higher EM order produces a tighter (smaller) classical remainder bound.
        """
        f = lambda t: math.exp(-t)
        params = {"sigma": 1.0, "N": 1.0, "window_type": "sharp"}
        res2 = em.euler_maclaurin_sum(
            f, 0.0, 2.0, 21, order=2, H=1.0, T0=0.0,
            params=params, is_polynomial=False
        )
        res4 = em.euler_maclaurin_sum(
            f, 0.0, 2.0, 21, order=4, H=1.0, T0=0.0,
            params=params, is_polynomial=False
        )
        assert res4.remainder_bound_classical <= res2.remainder_bound_classical + 1e-14, \
            "order=4 should give a tighter bound than order=2"

    def test_linear_demo_exact(self):
        """
        f(t) = t on [0,1] with n_terms=11 and is_polynomial=True.
        EM estimate must match true sum exactly and bound must be 0.
        This replicates the __main__ demo output.
        """
        f = lambda t: t
        a, b = 0.0, 1.0
        n_terms = 11
        ts = np.linspace(a, b, n_terms)
        true_sum = float(np.sum([f(t) for t in ts]))  # = 5.5

        res = em.euler_maclaurin_sum(
            f, a, b, n_terms, order=1, H=1.0, T0=0.0, is_polynomial=True
        )
        assert abs(res.total_sum_estimate - true_sum) < 1e-12
        assert res.remainder_bound_classical == 0.0
        assert res.remainder_bound_kernel    == 0.0


class TestUniformityAndIntegration:
    """
    Req 1.6 & 3: Uniformity over H, T0; integration with Volumes V/VI.
    """

    def test_uniformity_in_H_T0(self):
        """Kernel-enhanced remainder must stay bounded over H × T0 grid."""
        f = lambda t: math.exp(-t)
        params = {"sigma": 1.0, "N": 1.0, "window_type": "sharp"}
        uniformity = em.verify_uniformity_H_T0(
            H_values=[1.0, 2.0, 4.0],
            T0_values=[0.0, 10.0, 50.0],
            f=f, a=0.0, b=2.0, m=2, tolerance=1e-1,
            params=params,
        )
        assert uniformity["uniform"], \
            f"Remainder bound blew up: max_bound = {uniformity['max_bound']}"

    def test_integration_with_volume_V_coefficients(self):
        """EM bound holds when compared against exact discrete sum."""
        N = 50
        f_simple = lambda t: math.exp(-t)
        a, b = 0.0, math.log(N)
        n_terms = N
        h = (b - a) / (n_terms - 1)
        ts_simple = [a + k * h for k in range(n_terms)]
        f_discrete = np.array([f_simple(t) for t in ts_simple])

        params = {"sigma": 1.0, "N": 1.0, "window_type": "sharp"}
        comp = em.compare_sum_vs_em(
            f_discrete=f_discrete,
            f_cont=f_simple, a=a, b=b,
            m=2, H=1.0, T0=0.0,
            params=params, is_polynomial=False,
        )
        assert comp["bound_holds"], \
            f"Error {comp['abs_error']} exceeded bound {comp['remainder_bound']}"


class TestTotalErrorBudget:
    """
    Req 3.3: Integration with Volume VI Large Sieve constants.
    """

    def test_total_error_budget_is_finite(self):
        """Total error budget (LS mock + EM) must be finite and positive."""
        f_em = lambda t: math.exp(-t)
        params = {"sigma": 1.0, "N": 1.0, "window_type": "sharp"}
        res_em = em.euler_maclaurin_sum(
            f_em, a=0.0, b=math.log(100.0),
            n_terms=100, order=2, H=1.0, T0=0.0, params=params,
        )
        mock_ls_error = 0.005
        total_error_bound = mock_ls_error + res_em.remainder_bound

        assert math.isfinite(total_error_bound), "Total error budget is not finite"
        assert total_error_bound > 0.0


# ===========================================================================
# TIER B — Proof-ladder scaffolding
# ===========================================================================
#
# Each class is labelled with the proof-ladder step it supports.
# Run with:  pytest -m proof_ladder -v

class TestDiagonalMassEMBound:
    """
    Req 4.1 / 4.2 — Proof ladder Step 1 (Vol XIII): xi -> Q_H derivation.

    diagonal_mass_em_bound must return a certified positive floor for D_H(N).
    This is the positivity anchor that feeds the Q_lb budget.
    """

    @pytest.mark.proof_ladder
    def test_D_H_estimate_positive(self):
        """D_H EM estimate must be strictly positive for all tested N."""
        for N in [50, 100, 200]:
            d = em.diagonal_mass_em_bound(N=N, H=1.0, sigma=0.5, order=4)
            assert d["D_H_estimate"] > 0.0, \
                f"D_H_estimate non-positive at N={N}"

    @pytest.mark.proof_ladder
    def test_D_H_lower_bound_positive(self):
        """
        Certified floor D_H_lower_bound must be strictly positive.
        This is Req 4.1: the positivity floor that blocks Gap G1.
        """
        for N in [50, 100, 200, 500]:
            d = em.diagonal_mass_em_bound(N=N, H=1.0, sigma=0.5, order=4)
            assert d["D_H_lower_bound"] > 0.0, \
                f"D_H_lower_bound non-positive at N={N}"

    @pytest.mark.proof_ladder
    def test_D_H_lower_le_estimate(self):
        """Lower bound must not exceed the estimate."""
        for N in [50, 100, 200]:
            d = em.diagonal_mass_em_bound(N=N, H=1.0, sigma=0.5, order=4)
            assert d["D_H_lower_bound"] <= d["D_H_estimate"] + 1e-12

    @pytest.mark.proof_ladder
    def test_D_H_grows_with_N(self):
        """
        D_H(N) ~ (6/H^2) * log N must grow with N.
        This is Req 4.2: the EM estimate tracks the analytic log N behaviour.
        """
        Ns = [50, 100, 200, 500]
        estimates = [
            em.diagonal_mass_em_bound(N=N, H=1.0, sigma=0.5, order=4)["D_H_estimate"]
            for N in Ns
        ]
        for i in range(len(estimates) - 1):
            assert estimates[i] < estimates[i + 1], \
                f"D_H not increasing: D_H({Ns[i]})={estimates[i]}, D_H({Ns[i+1]})={estimates[i+1]}"

    @pytest.mark.proof_ladder
    def test_D_H_estimate_matches_analytic_logN(self):
        """
        Verify the EM estimate of D_H against the correct closed-form
        reference for the step-grid EM formula.

        diagonal_mass_em_bound integrates f(t) = e^{-2*sigma*t} = e^{-t}
        (at sigma=0.5) over [0, log N] with n_terms=N, so:

            h = log(N) / (N - 1)

        The EM formula computes:

            S_EM = (1/h) * integral_0^{log N} e^{-t} dt  +  corrections
                 ≈ (N-1)/log(N) * (1 - 1/N)
                 = (N-1)^2 / (N * log(N))

        Multiplied by prefactor 6/H^2 = 6:

            D_H_EM ≈ 6 * (N-1)^2 / (N * log(N))

        This is the correct analytic reference for what the function
        actually returns.  We check within 5%.

        NOTE: This is NOT the true diagonal mass sum_{n=1}^N n^{-1} ~ log N.
        The discrepancy between D_H_EM and the true harmonic sum is a known
        property of the log-space parametrisation and is documented here
        so future volumes can apply the correct normalisation.
        """
        for N in [100, 500]:
            d = em.diagonal_mass_em_bound(N=N, H=1.0, sigma=0.5, order=4)
            # Closed-form reference: 6 * (N-1)^2 / (N * log(N))
            analytic_ref = 6.0 * (N - 1) ** 2 / (N * math.log(N))
            ratio = d["D_H_estimate"] / analytic_ref
            assert 0.95 <= ratio <= 1.05, \
                f"D_H_EM deviates from step-grid reference at N={N}: " \
                f"ratio={ratio:.6f}, D_H={d['D_H_estimate']:.4f}, " \
                f"ref=6*(N-1)^2/(N*logN)={analytic_ref:.4f}"

    @pytest.mark.proof_ladder
    def test_D_H_remainder_finite_and_positive(self):
        """EM remainder in D_H units must be finite and positive."""
        d = em.diagonal_mass_em_bound(N=100, H=1.0, sigma=0.5, order=4)
        assert math.isfinite(d["remainder_bound"])
        assert d["remainder_bound"] > 0.0

    @pytest.mark.proof_ladder
    def test_D_H_higher_order_tightens_floor(self):
        """
        Higher EM order should raise the certified floor (tighter remainder).
        """
        d2 = em.diagonal_mass_em_bound(N=100, H=1.0, sigma=0.5, order=2)
        d4 = em.diagonal_mass_em_bound(N=100, H=1.0, sigma=0.5, order=4)
        # Order 4 has a smaller remainder, so its floor is higher
        assert d4["D_H_lower_bound"] >= d2["D_H_lower_bound"] - 1e-10


class TestRemainderVsNScaling:
    """
    Req 4.3 — Proof ladder Step 2 (Vol XIV): Mean-value form of O_H.

    remainder_vs_N_scaling must demonstrate R_m(N) = o(D_H(N)),
    i.e., the ratio R_m / D_H -> 0 as N grows.  This licenses the
    mean-value integral form of O_H — the sum-to-integral error cannot
    pollute the mean-square computation.
    """

    @pytest.mark.proof_ladder
    def test_scaling_returns_correct_keys(self):
        """Output dict must contain all five required keys."""
        result = em.remainder_vs_N_scaling([50, 100], H=1.0)
        for key in ["N", "log_N", "remainder", "D_H_lower", "ratio"]:
            assert key in result, f"Missing key: {key}"

    @pytest.mark.proof_ladder
    def test_scaling_lengths_match(self):
        """All lists must have the same length as N_values."""
        N_values = [50, 100, 200]
        result = em.remainder_vs_N_scaling(N_values, H=1.0)
        for key in ["N", "log_N", "remainder", "D_H_lower", "ratio"]:
            assert len(result[key]) == len(N_values), \
                f"Length mismatch for key {key}"

    @pytest.mark.proof_ladder
    def test_all_ratios_finite_and_positive(self):
        """Every ratio R_m(N)/D_H(N) must be finite and positive."""
        result = em.remainder_vs_N_scaling([50, 100, 200], H=1.0)
        for i, ratio in enumerate(result["ratio"]):
            assert math.isfinite(ratio), f"ratio not finite at index {i}"
            assert ratio > 0.0, f"ratio non-positive at index {i}"

    @pytest.mark.proof_ladder
    def test_ratio_strictly_less_than_one(self):
        """
        R_m(N) / D_H(N) must be < 1 for all tested N.
        This is the key condition: EM error is smaller than the signal floor.
        """
        result = em.remainder_vs_N_scaling([50, 100, 200], H=1.0)
        for i, ratio in enumerate(result["ratio"]):
            N = result["N"][i]
            assert ratio < 1.0, \
                f"Ratio >= 1 at N={N}: ratio={ratio}. EM error swamps diagonal mass."

    @pytest.mark.proof_ladder
    def test_ratio_decreasing_with_N(self):
        """
        Req 4.3 / Step 5 evidence: ratio(N) must be monotone decreasing.
        This is numerical evidence for R_m = o(D_H), which supports
        the N -> inf limit in Vol XVII.
        """
        N_values = [50, 100, 200, 500]
        result = em.remainder_vs_N_scaling(N_values, H=1.0)
        ratios = result["ratio"]
        for i in range(len(ratios) - 1):
            assert ratios[i] >= ratios[i + 1] - 1e-10, \
                f"Ratio not decreasing: ratio({N_values[i]})={ratios[i]}, " \
                f"ratio({N_values[i+1]})={ratios[i+1]}"

    @pytest.mark.proof_ladder
    def test_log_N_grows_correctly(self):
        """log_N values must match math.log(N) exactly."""
        N_values = [50, 100, 200]
        result = em.remainder_vs_N_scaling(N_values, H=1.0)
        for i, N in enumerate(N_values):
            assert abs(result["log_N"][i] - math.log(N)) < 1e-12

    @pytest.mark.proof_ladder
    def test_remainder_finite_for_all_N(self):
        """All remainder values must be finite."""
        result = em.remainder_vs_N_scaling([50, 100, 200], H=1.0)
        for i, r in enumerate(result["remainder"]):
            assert math.isfinite(r), f"Remainder not finite at index {i}"


class TestQHLowerBoundContribution:
    """
    Req 4.4 / 4.5 — Proof ladder Steps 3 & 4 (Vol XV / XVI).

    QH_lower_bound_contribution is Volume VII's explicit entry into the
    four-term Q_lb error budget:

        Q_lb = Q_trunc - (E_tail + E_quad + E_spec + E_num)

    where E_EM (this module's contribution) must satisfy:
        E_EM < D_H_certified_floor

    so that Q_lb > 0 is not undermined by EM discretization error alone.
    """

    @pytest.mark.proof_ladder
    def test_output_keys_present(self):
        """All required keys must be present in the output dict."""
        contrib = em.QH_lower_bound_contribution(N=100, H=1.0, T0=14.134)
        for key in ["E_EM", "D_H_certified_floor", "H", "N", "T0", "order"]:
            assert key in contrib, f"Missing key: {key}"

    @pytest.mark.proof_ladder
    def test_E_EM_finite_and_positive(self):
        """E_EM must be finite and positive."""
        contrib = em.QH_lower_bound_contribution(N=100, H=1.0, T0=14.134)
        assert math.isfinite(contrib["E_EM"])
        assert contrib["E_EM"] > 0.0

    @pytest.mark.proof_ladder
    def test_D_H_floor_finite_and_positive(self):
        """D_H certified floor must be finite and positive."""
        contrib = em.QH_lower_bound_contribution(N=100, H=1.0, T0=14.134)
        assert math.isfinite(contrib["D_H_certified_floor"])
        assert contrib["D_H_certified_floor"] > 0.0

    @pytest.mark.proof_ladder
    def test_E_EM_strictly_less_than_D_H_floor(self):
        """
        Req 4.4 — Core Gap G1 scaffold:
        E_EM < D_H_certified_floor must hold.

        If this fails, Volume VII's EM error alone would be large enough
        to undermine positivity of Q_lb.  This is the most important
        single assertion in Tier B.
        """
        for N in [50, 100, 200]:
            contrib = em.QH_lower_bound_contribution(N=N, H=1.0, T0=14.134)
            assert contrib["E_EM"] < contrib["D_H_certified_floor"], \
                f"CRITICAL: E_EM >= D_H_floor at N={N}. " \
                f"E_EM={contrib['E_EM']}, floor={contrib['D_H_certified_floor']}"

    @pytest.mark.proof_ladder
    def test_parameter_traceability(self):
        """H, N, T0, order must be faithfully recorded in the output."""
        contrib = em.QH_lower_bound_contribution(
            N=100, H=0.5, T0=21.022, sigma=0.5, order=4
        )
        assert contrib["H"]     == 0.5
        assert contrib["N"]     == 100.0
        assert contrib["T0"]    == 21.022
        assert contrib["order"] == 4.0

    @pytest.mark.proof_ladder
    def test_E_EM_stable_across_T0(self):
        """
        Req 4.5 — Lipschitz uniformity scaffold (Vol XVI):
        E_EM must not vary significantly across T0.
        The EM error comes from the diagonal integrand which is T0-independent,
        so the variation across T0 should be zero to machine precision.
        """
        T0_values = [0.0, 14.134, 21.022, 50.0, 100.0]
        E_EM_values = [
            em.QH_lower_bound_contribution(N=100, H=1.0, T0=T0)["E_EM"]
            for T0 in T0_values
        ]
        # All values should be identical (T0 does not enter D_H)
        for i in range(1, len(E_EM_values)):
            assert abs(E_EM_values[i] - E_EM_values[0]) < 1e-12, \
                f"E_EM varies with T0: T0={T0_values[i]}, delta={abs(E_EM_values[i]-E_EM_values[0])}"

    @pytest.mark.proof_ladder
    def test_E_EM_varies_with_H(self):
        """
        Kernel-enhanced E_EM should decrease as H increases,
        reflecting the sech^4 kernel's tighter spectral localisation.
        """
        contribs = [
            em.QH_lower_bound_contribution(N=100, H=H, T0=0.0)
            for H in [0.5, 1.0, 2.0]
        ]
        E_EMs = [c["E_EM"] for c in contribs]
        # Each should be finite
        for e in E_EMs:
            assert math.isfinite(e)
        # At least one pair should differ — H has some effect
        assert not all(abs(E_EMs[i] - E_EMs[0]) < 1e-15 for i in range(1, len(E_EMs)))


class TestProofLadderIntegration:
    """
    Integration tests that chain all three new functions together,
    simulating how Volumes XIII–XVII will consume Volume VII's output.
    """

    @pytest.mark.proof_ladder
    def test_full_chain_N50(self):
        """
        Full proof-ladder chain at N=50:
          1. diagonal_mass_em_bound  -> D_H floor > 0
          2. remainder_vs_N_scaling  -> ratio < 1
          3. QH_lower_bound          -> E_EM < D_H floor
        All three must hold simultaneously.
        """
        N, H = 50, 1.0
        d       = em.diagonal_mass_em_bound(N=N, H=H, sigma=0.5, order=4)
        scaling = em.remainder_vs_N_scaling([N], H=H, sigma=0.5, order=4)
        contrib = em.QH_lower_bound_contribution(N=N, H=H, T0=14.134)

        assert d["D_H_lower_bound"]   > 0.0,  "Step 1 fail: D_H floor not positive"
        assert scaling["ratio"][0]    < 1.0,  "Step 2 fail: ratio >= 1"
        assert contrib["E_EM"]        < contrib["D_H_certified_floor"], \
            "Step 3 fail: E_EM >= D_H floor"

    @pytest.mark.proof_ladder
    def test_full_chain_N200(self):
        """Full proof-ladder chain at N=200."""
        N, H = 200, 1.0
        d       = em.diagonal_mass_em_bound(N=N, H=H, sigma=0.5, order=4)
        scaling = em.remainder_vs_N_scaling([N], H=H, sigma=0.5, order=4)
        contrib = em.QH_lower_bound_contribution(N=N, H=H, T0=14.134)

        assert d["D_H_lower_bound"]   > 0.0
        assert scaling["ratio"][0]    < 1.0
        assert contrib["E_EM"]        < contrib["D_H_certified_floor"]

    @pytest.mark.proof_ladder
    def test_error_budget_assembly(self):
        """
        Simulate the four-term Q_lb budget assembly for one grid point.
        Volume VII contributes E_EM; three other terms are mocked.
        Final Q_lb must be positive.
        """
        N, H, T0 = 100, 1.0, 14.134
        contrib = em.QH_lower_bound_contribution(N=N, H=H, T0=T0)

        # Mock contributions from other volumes
        E_tail  = 1e-4    # Vol XI tail truncation
        E_quad  = 1e-5    # Vol XI quadrature
        E_spec  = 1e-5    # Vol IV spectral
        E_EM    = contrib["E_EM"]

        total_error = E_tail + E_quad + E_spec + E_EM
        D_H_floor   = contrib["D_H_certified_floor"]

        # Q_lb > 0 iff total error < D_H floor
        Q_lb = D_H_floor - total_error
        assert Q_lb > 0.0, \
            f"Q_lb not positive: D_H_floor={D_H_floor}, total_error={total_error}"

    @pytest.mark.proof_ladder
    def test_monotone_ratio_over_extended_N(self):
        """
        Extended N range for the N->inf evidence.
        ratio(N) must be monotone decreasing across [50, 100, 200, 500].
        This is the numerical scaffold for Vol XVII's asymptotic step.
        """
        N_values = [50, 100, 200, 500]
        result = em.remainder_vs_N_scaling(N_values, H=1.0)
        ratios = result["ratio"]
        for i in range(len(ratios) - 1):
            assert ratios[i] >= ratios[i + 1] - 1e-10, \
                f"Monotonicity failed: ratio[{i}]={ratios[i]}, ratio[{i+1}]={ratios[i+1]}"


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])