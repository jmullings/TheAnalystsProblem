#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# test_VOLUME_VIII_POSITIVITY_TRANSFORMATION.py
#
# VALIDATION SUITE FOR VOLUME VIII: TAP HO POSITIVITY TRANSFORMATION
# Aligned with the updated Log-Free Protocol and Spectral Consistency Checks,
# updated to achieve 100% code coverage.

import sys
import os
import numpy as np
import pytest

# Inject the proof directory into sys.path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROOF_DIR = os.path.abspath(os.path.join(THIS_DIR, '..', 'VOLUME_VIII_POSITIVITY_TRANSFORMATION_PROOF'))
sys.path.insert(0, PROOF_DIR)

try:
    import VOLUME_VIII_POSITIVITY_TRANSFORMATION as v8
except ImportError:
    pytest.skip("VOLUME_VIII_POSITIVITY_TRANSFORMATION module not found", allow_module_level=True)


class TestFeatureMap:
    """
    Verify the geometric properties and error-handling of the TAP HO feature map.
    """
    def test_feature_map_dimensions(self):
        N = 100
        branches = 20
        t_vals = np.linspace(0.1, 10, branches // 2)
        embedding = np.log(np.arange(1, N + 1))
        
        Gamma = v8.build_tap_feature_map(N, branches, t_vals, embedding)
        assert Gamma.shape == (N, branches), "Feature map shape is incorrect."

    def test_feature_map_errors(self):
        N = 10
        t_vals = np.array([1.0, 2.0])
        embedding = np.log(np.arange(1, N + 1))
        
        with pytest.raises(ValueError, match="num_branches must be even"):
            v8.build_tap_feature_map(N, 5, t_vals, embedding)
            
        with pytest.raises(ValueError, match="Embedding vector length must match"):
            v8.build_tap_feature_map(N, 4, t_vals, np.array([1.0]))
            
        with pytest.raises(ValueError, match="t_vals length must be num_branches // 2"):
            v8.build_tap_feature_map(N, 4, np.array([1.0]), embedding)


class TestSpectralWeights:
    """
    Verify the properties of the spectral Gaussian weights.
    """
    def test_gaussian_spectral_weights_positive(self):
        branches = 200
        t_vals, w_diag, dt = v8.gaussian_spectral_weights_tap(branches, T_max=10.0, H=5.0)
        assert len(t_vals) == branches // 2
        assert len(w_diag) == branches
        assert np.all(w_diag > 0), "All spectral weights must be strictly positive."
        assert dt > 0

    def test_gaussian_spectral_weights_error(self):
        with pytest.raises(ValueError, match="Need at least 2 frequency points"):
            v8.gaussian_spectral_weights_tap(2)  # half = 1


class TestPositiveGramOperator:
    """
    Verify construction, symmetry, correctness, and boundary behaviors
    of the finite-dimensional Gram Surrogate K_N.
    """
    def test_operator_exact_symmetry(self):
        N = 100
        embedding = np.log(np.arange(1, N + 1))
        op = v8.PositiveGramOperator(N, embedding=embedding, num_branches=10)
        K = op.K_dense
        # Check exact symmetry
        max_asymmetry = np.max(np.abs(K - K.T))
        assert max_asymmetry < 1e-12, f"Matrix is not perfectly symmetric. Max diff: {max_asymmetry}"

    def test_operator_errors(self):
        embedding = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Invalid dimension N"):
            v8.PositiveGramOperator(0, embedding)
            
        with pytest.raises(ValueError, match="Invalid dimension N"):
            v8.PositiveGramOperator(v8._MAX_DIM + 1, embedding)
            
        with pytest.raises(ValueError, match="Embedding length must equal N"):
            v8.PositiveGramOperator(5, embedding)

    def test_evaluate_errors(self):
        N = 5
        embedding = np.log(np.arange(1, N + 1))
        op = v8.PositiveGramOperator(N, embedding)
        bad_a = np.ones(N + 1)
        
        with pytest.raises(ValueError, match="Coefficient vector length must equal N"):
            op.evaluate_dense(bad_a)
            
        with pytest.raises(ValueError, match="Coefficient vector length must equal N"):
            op.evaluate_factorized(bad_a)

    def test_factorization_equivalence_random_coeffs(self):
        np.random.seed(123)
        for N in [50, 100]:
            embedding = np.log(np.arange(1, N + 1))
            op = v8.PositiveGramOperator(N, embedding=embedding, num_branches=40)
            a_coeffs = np.random.randn(N)
            
            dense_val = op.evaluate_dense(a_coeffs)
            factorized_val = op.evaluate_factorized(a_coeffs)
            error = abs(dense_val - factorized_val)
            # The dense and factorized evaluations must mathematically coincide
            assert error < 1e-11, f"Factorization failed. Error: {error}"


class TestDirichletCoefficients:
    """
    Ensure the Dirichlet configuration generates coefficients properly 
    and handles fallbacks seamlessly.
    """
    def test_custom_coeffs(self):
        cfg = v8.DirichletConfig(N=5, custom_coeffs=np.array([1, 2, 3, 4, 5]))
        a, logn = v8.build_dirichlet_coefficients(cfg)
        assert np.array_equal(a, [1, 2, 3, 4, 5])
        assert np.allclose(logn, np.log([1, 2, 3, 4, 5]))

    def test_weight_and_window_types(self):
        # We test normal configs to cover standard implementations and fallbacks
        cfg1 = v8.DirichletConfig(N=5, weight_type="plain", window_type="sharp", sigma=0.5)
        a1, _ = v8.build_dirichlet_coefficients(cfg1)
        assert len(a1) == 5
        
        cfg2 = v8.DirichletConfig(N=5, weight_type="log", window_type="gaussian", sigma=0.5, window_params={"alpha": 1.0})
        a2, _ = v8.build_dirichlet_coefficients(cfg2)
        assert len(a2) == 5


class TestSpectralConsistency:
    """
    Verify the log-free matrix multiplication reliably approximations
    the underlying continuous spectral integral.
    """
    def test_evaluate_spectral_on_grid(self):
        N = 10
        a = np.ones(N)
        logn = np.log(np.arange(1, N + 1))
        t_vals = np.array([1.0, 2.0])
        val = v8.evaluate_spectral_on_grid(a, logn, t_vals, H=5.0, dt=0.5)
        assert val > 0.0

    def test_positivity_transformation_and_spectral_error(self):
        """
        Proof that the Gram surrogate is an exact finite-dimensional embedding 
        of the sampled spectral integral, yielding an error practically zero.
        """
        cfg = v8.DirichletConfig(N=50, sigma=0.5, window_type="sharp")
        res = v8.positivity_transformation(cfg, T_max=10.0, H=5.0, num_branches=100)
        
        assert res.N == 50
        assert res.minimum_weight > 0
        assert res.is_positive_definite is True
        assert res.factorization_error < 1e-11
        assert res.spectral_integral > 0
        assert res.spectral_error is not None
        assert res.spectral_error < 1e-12, f"Spectral Error blew up: {res.spectral_error}"


class TestDiagnostics:
    """
    Test the built-in diagnostic and demo suite to ensure proper execution.
    """
    def test_demo_default(self, capsys):
        v8._demo()
        captured = capsys.readouterr()
        assert "=== Volume VIII: TAP HO Positivity Transformation Demo ===" in captured.out
        assert "Dimension (N)" in captured.out
        assert "Minimum spectral weight" in captured.out
        assert "Factorized" in captured.out

    def test_demo_custom(self, capsys):
        cfg = v8.DirichletConfig(N=10, sigma=0.5, window_type="sharp")
        v8._demo(cfg)
        captured = capsys.readouterr()
        assert "Dimension (N)                : 10" in captured.out


if __name__ == '__main__':
    pytest.main([__file__, "-v"])