#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# test_VOLUME_VI_LARGE_SIEVE_BRIDGE.py
#
# VALIDATION SUITE FOR VOLUME VI: LARGE SIEVE BRIDGE
#
# Tests:
#   - Montgomery–Vaughan large sieve bounds (discrete Dirichlet side). [web:326][web:330]
#   - Kernel-decay off-diagonal bounds using SECH^2 / SECH^6 spectral weights. [web:297][web:329]
#   - Discrete-to-continuous transition error control.
#   - Explicit constant tracking (no hidden big-O).
#   - “Forced by primes” narrative: only decaying SECH kernels are compatible
#     with the quadratic forms built from Λ-like weights and log-frequencies;
#     artificially exploding spectral weights destroy the inequality scale. [web:292][web:327]

import sys
import os
import math
import numpy as np
import mpmath as mp
import pytest

# Inject the proof directory into sys.path
PROOF_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "VOLUME_VI_LARGE_SIEVE_BRIDGE_PROOF",
    )
)
sys.path.insert(0, PROOF_DIR)

try:
    import VOLUME_VI_LARGE_SIEVE_BRIDGE as ls
except ImportError:
    # Fallback in case of naming variation
    import LARGE_SIEVE_BRIDGE as ls

mp.mp.dps = 80
TOL = 1e-12


class TestLargeSieveConstants:
    """
    Requirement 2 & 5: Explicit constant tracking and No Hidden Big-O.

    Every bound-producing routine must return finite, explicit floats.
    """

    def test_explicit_constants_are_finite(self):
        """All bound functions must return finite floats (no NaNs, no infs)."""
        cfg = ls.DirichletConfig(N=50, sigma=0.5, window_type="sharp")
        xi_values = [0.0, 1.0, 5.0]

        constants, comps = ls.validate_large_sieve_bounds(
            cfg,
            H=1.0,
            xi_values=xi_values,
        )

        # Verify constants object fields are finite floats
        assert isinstance(constants.min_separation, float) and math.isfinite(
            constants.min_separation
        )
        assert isinstance(constants.sum_abs_sq, float) and math.isfinite(
            constants.sum_abs_sq
        )
        assert isinstance(constants.MV_constant, float) and math.isfinite(
            constants.MV_constant
        )
        assert isinstance(constants.MV_bound, float) and math.isfinite(
            constants.MV_bound
        )
        assert isinstance(constants.kernel_bound_constant, float) and math.isfinite(
            constants.kernel_bound_constant
        )
        assert isinstance(constants.kernel_bound, float) and math.isfinite(
            constants.kernel_bound
        )
        assert isinstance(constants.discrete_to_cont_error, float) and math.isfinite(
            constants.discrete_to_cont_error
        )

        # Verify bound comparison fields are finite floats
        for c in comps:
            assert isinstance(c.off_diag_exact, float) and math.isfinite(
                c.off_diag_exact
            )
            assert isinstance(c.ratio_off_to_MV, float) and math.isfinite(
                c.ratio_off_to_MV
            )
            assert isinstance(c.ratio_off_to_kernel, float) and math.isfinite(
                c.ratio_off_to_kernel
            )

    def test_min_separation_computation(self):
        """Ensure δ is correctly computed for log frequencies."""
        N = 10
        # γ_n = log n
        gammas = ls.log_frequencies(N)

        # For γ_n = log n, the minimum separation occurs at the largest n
        # δ = log(N) - log(N-1) = log(N/(N-1))
        expected_delta = math.log(10.0 / 9.0)
        computed_delta = ls.min_separation(gammas)

        assert abs(computed_delta - expected_delta) < 1e-14


class TestMontgomeryVaughanInequality:
    """
    Requirement 1: Montgomery–Vaughan large sieve bound implementation.

    This encodes the classical inequality
      sup_ξ |∑ a_n e^{2π i ξ γ_n}|^2 ≤ (N + 1/δ) ∑ |a_n|^2,
    for γ_n = log n. [web:326][web:330]
    """

    def test_montgomery_vaughan_inequality(self):
        """Verify |off_diag(ξ)| ≤ MV_bound for several ξ."""
        cfg = ls.DirichletConfig(N=30, sigma=0.5, window_type="sharp")
        xi_values = np.linspace(-5.0, 5.0, 20)

        constants, comps = ls.validate_large_sieve_bounds(
            cfg,
            H=1.0,
            xi_values=xi_values,
        )

        for c in comps:
            # The MV inequality states:
            # |S(ξ)|^2 <= (N + 1/δ) sum |a_n|^2.
            #
            # Our off-diagonal extractor returns:
            #   off = | |S(ξ)|^2 - diag |,   diag = sum |a_n|^2.
            #
            # Hence |off| <= MV_bound in a clean numerical sense.
            assert (
                c.ratio_off_to_MV <= 1.0 + TOL
            ), f"MV bound violated at ξ={c.xi}: ratio={c.ratio_off_to_MV}"


class TestKernelDecayBound:
    """
    Requirement 1.3 & 3: Kernel-decay bound for off-diagonal interference.

    This checks that SECH^2 / SECH^6 spectral kernels give a genuine
    upper bound on the off-diagonal quadratic form and that, when
    localized, they can improve on the coarse MV inequality.
    """

    def test_kernel_bound_dominates_off_diagonal(self):
        """
        Show that the kernel-decay bound strictly bounds the actual off-diagonal
        computation for all ξ (numerical safety check).
        """
        cfg = ls.DirichletConfig(N=40, sigma=0.5, window_type="sharp")
        xi_values = np.linspace(-2.0, 2.0, 10)

        constants, comps = ls.validate_large_sieve_bounds(
            cfg,
            H=1.0,
            xi_values=xi_values,
        )

        for c in comps:
            # The ratio |off| / kernel_bound must be ≤ 1.0 in this design.
            assert (
                c.ratio_off_to_kernel <= 1.0 + TOL
            ), f"Kernel bound violated at ξ={c.xi}: ratio={c.ratio_off_to_kernel}"

    def test_kernel_bound_tighter_than_MV_for_large_H(self):
        """
        For a highly localized kernel (larger H in this model), the kernel
        bound should be tighter (smaller) than the general MV bound.
        """
        cfg = ls.DirichletConfig(N=50, sigma=0.5, window_type="sharp")

        # Test with H=2.0 (stronger spectral localization under this model)
        constants, _ = ls.validate_large_sieve_bounds(
            cfg,
            H=2.0,
            xi_values=[0.0],
        )

        assert (
            constants.kernel_bound < constants.MV_bound
        ), "Kernel bound failed to improve upon the loose MV bound"


class TestDiscreteToContinuousTransition:
    """
    Requirement 1.4: Discrete-to-continuous quadrature error bounds.
    """

    def test_discrete_to_continuous_error(self):
        """Compare discrete sum to integral surrogate and verify error bound."""
        # For a(n) = n^{-0.5}, |a_n|^2 = 1/n.
        # S_N = sum_{n=1}^N 1/n (harmonic number) with known behaviour. [web:327]
        N = 100
        cfg = ls.DirichletConfig(N=N, sigma=0.5, window_type="sharp")
        raw_a, _ = ls.build_coefficients(cfg)

        I, error_bound = ls.discrete_to_continuous_quadrature(raw_a)

        # The sum of 1/n for N=100 is ~ 5.187 (sanity check).
        expected_sum = sum(1.0 / n for n in range(1, N + 1))
        assert abs(I - expected_sum) < 1e-12

        # For 1/n, |a_n|^2 is monotonically decreasing, so the total variation of
        # |a_n|^2 telescopes to a_1^2 - a_N^2 = 1 - 1/N.
        expected_error = 0.5 * (1.0 + 1.0 / N) + (1.0 - 1.0 / N)
        assert abs(error_bound - expected_error) < 1e-12


class TestWindowScalingEffects:
    """
    Requirement 4.3 & 5: Verify smooth windows improve large sieve constants.

    A smoother window knocks down ∑|a_n|^2 and thus the MV bound.
    """

    def test_window_improves_large_sieve_constant(self):
        """
        Demonstrate that a smooth window reduces the sum|a_n|^2, thereby
        lowering the overall MV bound compared to a sharp cutoff.
        """
        N = 50
        H = 1.0

        # Sharp window
        cfg_sharp = ls.DirichletConfig(N=N, sigma=0.5, window_type="sharp")
        const_sharp, _ = ls.validate_large_sieve_bounds(cfg_sharp, H, [0.0])

        # Smooth Gaussian window
        cfg_smooth = ls.DirichletConfig(
            N=N,
            sigma=0.5,
            window_type="gaussian",
            window_params={"alpha": 2.0},
        )
        const_smooth, _ = ls.validate_large_sieve_bounds(
            cfg_smooth,
            H,
            [0.0],
        )

        # The smooth window suppresses the tail coefficients, resulting in
        # a strictly smaller MV bound.
        assert (
            const_smooth.MV_bound < const_sharp.MV_bound
        ), "Smoothing failed to reduce the MV large sieve bound"


class TestForcedByPrimesNarrative:
    """
    “Forced by the arithmetic of primes” validation.

    This test class does not alter the production kernels. Instead, it
    compares:

      - The behaviour of the decaying SECH^2 / SECH^6 kernel bounds,
      - Against a hypothetical “exploding” spectral kernel of sinh-type
        growth, inspired by the v4 explicit–formula POC where a bad
        spectral weight produced zero-side contributions of size 10^106–10^107. [web:292][web:327]

    The logic is:

      1. The MV large sieve bound, coming from actual Λ-like weights and
         log n spacing, constrains |S(ξ)|^2 to a finite, moderate scale. [web:326][web:330]
      2. Any admissible spectral kernel used to bound off-diagonal
         interference must live on the same scale.
      3. SECH^2 / SECH^6 kernels satisfy this: their bounds are of the
         same order as MV.
      4. A fake exploding kernel, if taken seriously as a spectral weight,
         would give kernel bounds exceeding MV by many orders of magnitude.
         Such a kernel is therefore ruled out by the arithmetic constraints.
    """

    @staticmethod
    def _exploding_kernel_vals(logn: np.ndarray, H: float) -> np.ndarray:
        """
        Hypothetical "bad" kernel:

            k_bad(Δ) = exp(c * |Δ| / H),

        mimicking the exponentially growing sinh-style spectral weight
        used as a counterexample in the explicit–formula POCs.

        This is test-local and **not** part of the production library.
        """
        N = len(logn)
        logn_col = logn.reshape(-1, 1)
        Δ = logn_col - logn_col.T
        mask_off = ~np.eye(N, dtype=bool)
        Δ_off = Δ[mask_off]
        c = 4.0  # moderate growth rate; enough to explode vs MV for N~50
        return np.exp(c * np.abs(Δ_off) / H)

    def test_sech_kernel_vs_mv_scale(self):
        """
        Show that for SECH^2 / SECH^6 kernels, the kernel bound lives on the
        same scale as the MV bound (the regime seen in run_volume_vi_demo).
        """
        cfg = ls.DirichletConfig(N=80, sigma=0.5, window_type="sharp")
        H = 1.0
        constants_s2, _ = ls.validate_large_sieve_bounds(
            cfg,
            H=H,
            xi_values=[0.0],
            use_sech_basis="sech2",
        )
        constants_s6, _ = ls.validate_large_sieve_bounds(
            cfg,
            H=H,
            xi_values=[0.0],
            use_sech_basis="sech6",
        )

        # In both cases, kernel_bound and MV_bound should be the same order
        # of magnitude (ratios near 1, not 10^10 or 10^-10).
        ratio_s2 = constants_s2.kernel_bound / constants_s2.MV_bound
        ratio_s6 = constants_s6.kernel_bound / constants_s6.MV_bound

        assert 1e-2 <= ratio_s2 <= 1e2, f"SECH^2 kernel bound scale abnormal: ratio={ratio_s2}"
        assert 1e-2 <= ratio_s6 <= 1e2, f"SECH^6 kernel bound scale abnormal: ratio={ratio_s6}"

    def test_exploding_kernel_obliterates_mv_scale(self):
        """
        Construct a hypothetical exploding kernel and show that, if it were
        used to bound off-diagonal terms, its bound would vastly exceed the
        MV bound—mirroring the v4 explicit–formula behaviour where a sinh
        spectral weight produced enormous zero-side sums. [web:292][web:327]
        """
        N = 50
        cfg = ls.DirichletConfig(N=N, sigma=0.5, window_type="sharp")
        # Use plain coefficients with sharp window to mimic "Λ-like" structure
        raw_a, logn = ls.build_coefficients(cfg)
        a = ls.apply_window(cfg, raw_a)

        # Arithmetic side: MV bound
        δ, MV_const, MV_bound = ls.montgomery_vaughan_bound(logn, a)

        # Spectral side (good): SECH^2 kernel bound
        kernel_const_sech, kernel_bound_sech = ls.kernel_decay_off_diagonal_bound(
            a,
            logn,
            H=1.0,
            kernel_type="sech2",
        )

        # Spectral side (bad): exploding kernel
        k_bad_vals = self._exploding_kernel_vals(logn, H=1.0)
        a_abs = np.abs(a)
        prod = (a_abs.reshape(-1, 1) * a_abs.reshape(1, -1))[~np.eye(N, dtype=bool)]
        bad_bound = float(np.sum(prod * k_bad_vals))

        # Sanity: SECH kernel bound is comparable to MV, not ridiculously small or large.
        assert kernel_bound_sech > 0.0
        assert MV_bound > 0.0
        ratio_good = kernel_bound_sech / MV_bound
        assert 1e-4 <= ratio_good <= 1e2, f"SECH kernel/MV ratio out of range: {ratio_good}"

        # Exploding kernel must produce a bound many orders of magnitude larger
        # than MV_bound; this is precisely the behaviour seen in the explicit
        # formula POC when using a sinh-based spectral weight.
        ratio_bad = bad_bound / MV_bound
        assert (
            ratio_bad > 1e4
        ), f"Exploding kernel did not significantly exceed MV scale: ratio={ratio_bad}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])