#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# test_VOLUME_XI_COMPUTATIONAL.py
#
# VALIDATION SUITE FOR VOLUME XI: COMPUTATIONAL VERIFICATION
# Tests the high-precision ground truth, large N scaling, extreme grid evaluations,
# numerical stability (regime-aware + gradient-consistent), convergence rates,
# time-frequency consistency at scale, tail control, resonance handling,
# scalability, and statistical robustness.
#
# CHANGELOG:
#   - Added pytest.mark.slow to all time-intensive tests. Run with `pytest -m "not slow"`
#     for a quick sanity check, or `pytest` for the full suite.
#   - Registered 'slow' mark in pytest.ini to suppress warnings.


import sys
import os
import pytest
import math
import numpy as np


# Inject the proof directory into sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROOF_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'VOLUME_XI_COMPUTATIONAL_PROOF'))
if PROOF_DIR not in sys.path:
    sys.path.insert(0, PROOF_DIR)


try:
    import VOLUME_XI_COMPUTATIONAL as xi
except ImportError:
    # Fallback for naming variations
    import COMPUTATIONAL_VERIFICATION as xi


def assert_all_passed(results):
    """Helper to cleanly assert that all XIResults in a list passed."""
    failed = [r for r in results if not r.passed]
    if failed:
        error_msgs = "\n".join([f"FAIL [{r.name}]: {r.details}" for r in failed])
        pytest.fail(f"Module failed {len(failed)}/{len(results)} checks:\n{error_msgs}")


class TestModule1HighPrecision:
    """
    Requirement 1: Establish a trusted baseline using arbitrary precision.
    """
    def test_high_precision_ground_truth(self):
        """
        Verify that Q(H, T0, N) computed at 50, 100, and 200 dps agree
        to extreme precision (rel_err < 1e-10 between 50→100 dps,
        rel_err < 1e-12 between 100→200 dps).
        """
        results = xi.module1_high_precision_ground_truth()
        assert_all_passed(results)


class TestModule2LargeNScaling:
    """
    Requirement 2: Verify behavior as N -> infinity.
    """
    @pytest.mark.slow
    def test_large_N_scaling_smoothness(self):
        """
        Verify that Q(N) grows smoothly without oscillatory blow-ups
        and follows a predictable scaling law for large N.
        Ratios Q(N+1)/Q(N) must lie in [0.8, 1.5] and log-log curvature < 0.5.
        """
        results = xi.module2_large_N_scaling()
        assert_all_passed(results)


class TestModule3ExtremeGrid:
    """
    Requirement 3: Uniform positivity everywhere on an extreme parameter grid.
    """
    @pytest.mark.slow
    def test_extreme_coordinate_grid(self):
        """
        Verify that Q > 0 and floor - leakage > 0 for a randomized sample
        of extreme H, T0, and N values. Automatically skips redundant
        asymptotic or undersampled regimes to save compute.
        """
        results = xi.module3_extreme_grid()
        assert_all_passed(results)


class TestModule4NumericalStability:
    """
    Requirement 4: System is robust under intentional numerical perturbations.

    Two distinct sub-tests:

    (a) Precision-drop stability (dps_50, dps_30):
        Q computed at reduced dps must match the 200-dps baseline to
        relative error < 1e-6.  This tests mpmath's internal consistency.

    (b) Gradient-consistent perturbation (H_perturbed, T0_perturbed,
        H_T0_perturbed):
        The fix: measure the local Lipschitz constant ∂Q/∂p via a two-sided
        finite-difference probe δ = 1e-4, then verify that the observed
        change from the larger perturbation Δ is consistent with that
        gradient within a 30% relative envelope.
    """
    def test_numerical_stability_stress(self):
        results = xi.module4_numerical_stability()
        assert_all_passed(results)


class TestModule5ConvergenceRate:
    """
    Requirement 5: Quantify how fast Q_H(N) -> Q_H(infinity).
    """
    def test_convergence_rate_analysis(self):
        """
        Verify that Δ_N = |Q(N) - Q_max| decays with a log-log slope < -0.05,
        confirming monotone convergence toward the infinite-N limit.
        """
        results = xi.module5_convergence_rate()
        assert_all_passed(results)


class TestModule6TimeFreqConsistency:
    """
    Requirement 6: Plancherel holds at large N and large T0.
    """
    @pytest.mark.slow
    def test_time_frequency_consistency_at_scale(self):
        """
        Verify |Q_time - Q_freq| < max(1e-7, 1e-5 * |Q_time|) globally
        across combinations of H ∈ {0.5, 1.0}, T0 ∈ {0, 100}, N ∈ {50, 500}.
        """
        results = xi.module6_time_frequency_consistency()
        assert_all_passed(results)


class TestModule7TailControl:
    """
    Requirement 7: Ensure truncation bounds remain valid.
    """
    def test_tail_control_at_scale(self):
        """
        Verify that the explicit exponential tail bound
        (48/H) e^{-4L/H} × sup|D_N|² remains strictly subdominant to
        the computed integral Q (less than 1% of |Q|) for all tested
        (H, T0, N) combinations.
        """
        results = xi.module7_tail_control()
        assert_all_passed(results)


class TestModule8Resonance:
    """
    Requirement 8: Re-test arithmetic resonances under large N.
    """
    @pytest.mark.slow
    def test_resonance_handling_at_scale(self):
        """
        Verify that deliberately choosing T0 to hit resonance peaks
        T0 = 2πk / log(n/m) does not break global positivity, even for
        large N.
        """
        results = xi.module8_resonance_large_N()
        assert_all_passed(results)


class TestModule9Scalability:
    """
    Requirement 9: Prove algorithmic scalability.
    """
    def test_scalability_diagnostics(self):
        """
        Verify that runtime scales at worst polynomially (O(N^β) with β < 2.0)
        and does not blow up exponentially.
        """
        results = xi.module9_scalability()
        assert_all_passed(results)


class TestModule10StatisticalRobustness:
    """
    Requirement 10: Randomized validation over the parameter space.
    """
    @pytest.mark.slow
    def test_statistical_robustness(self):
        """
        Evaluate Q on randomly sampled (H, T0, N) configurations drawn from
        H ∈ [0.05, 5], T0 ∈ [-500, 500], N ∈ [10, 2000].
        The failure rate MUST be 0%.
        """
        # 50 samples for test-suite speed; increase to 300 for deeper validation
        results = xi.module10_statistical_robustness(num_samples=50, parallel=False)
        assert_all_passed(results)


class TestModule11AsymptoticRegime:
    """
    Requirement 11: Normalized invariance at massive N.
    """
    @pytest.mark.slow
    def test_asymptotic_N_regime(self):
        """
        Verify that Q(N)/N stabilizes as N ∈ {1000, 5000, 10000, 20000, 50000}:
        consecutive ratios must lie in [0.8, 1.2].
        """
        results = xi.module11_asymptotic_regime()
        assert_all_passed(results)


class TestModule12AdversarialWorstCase:
    """
    Requirement 12: Break the system intentionally with adversarial setups.
    """
    @pytest.mark.slow
    def test_adversarial_worst_case(self):
        """
        Verify positivity under near-log collisions (T0 ≈ 2πk/log(n/m) ± ε)
        and highly clustered frequencies at large H, N, and T0.
        """
        results = xi.module12_adversarial_worst_case()
        assert_all_passed(results)


class TestModule13ErrorBudget:
    """
    Requirement 13: Explicit error-budget tracking.
    """
    def test_error_budget_decomposition(self):
        """
        Verify that the total error (quadrature + tail + implementation-gap)
        is strictly bounded by 0.5 × the positive floor value, confirming
        that the positivity margin absorbs all numerical uncertainties.
        """
        results = xi.module13_error_budget()
        assert_all_passed(results)


if __name__ == '__main__':
    # To run only the fast tests:
    # pytest test_VOLUME_XI_COMPUTATIONAL.py -m "not slow"
    # To run the full suite:
    # pytest test_VOLUME_XI_COMPUTATIONAL.py
    pytest.main([__file__, "-v", "-m", "not slow"])