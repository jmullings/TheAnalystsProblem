#!/usr/bin/env python3
# KERNEL_SECH4_BOCHNER_MERCER_CERTIFICATION.py
#
# Enhanced certification script for
#   k_H(t) = (6/H^2) * sech^4(t/H)
#
# This script:
#   1. Uses a numerical Fourier transform to evaluate the true Fourier
#      symbol of k_H and check nonnegativity (Bochner side).
#   2. Performs a Gram-matrix PSD test on a large random cloud in R.
#   3. Constructs the integral operator on a compact domain [0, log N],
#      checks PSD, trace-class behavior, and Mercer reconstruction.
#
# Interpretation:
#   - Bochner: strong numerical evidence for positive definiteness
#     via spectral (Fourier) and Gram tests.
#   - Mercer: numerical behavior fully consistent with the theorem.
#
# NOTE: This is numerical certification, not a formal analytic proof.

from __future__ import annotations

import math
import os
import sys
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------
# 0. Project-root path setup (align with QED_HILBERT_POLYA_RH_PROOF.py)
# ---------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# (Optional) You can import shared utilities here if you later create them, e.g.:
# from VOLUME_II_KERNAL_DECOMPOSITION.VOLUME_II_KERNAL_DECOMPOSITION_PROOF.KERNAL_DECOMPOSITION_PROBLEM import k_H as vol2_k_H
# For now this script is self-contained and only depends on NumPy.


# ---------------------------------------------------------------------
# 1. Kernel definition
# ---------------------------------------------------------------------


def k_H(t: np.ndarray | float, H: float = 1.0) -> np.ndarray:
    """
    Translation-invariant kernel:
        k_H(t) = (6/H^2) * sech^4(t/H).

    Implemented in a numerically stable way.
    """
    t_arr = np.asarray(t, dtype=float)
    z = np.abs(t_arr / H)
    # Clip to avoid overflow in cosh for large |t/H|
    z_clip = np.clip(z, 0.0, 40.0)  # cosh(40) ~ 1e17, safe in float64
    val = (6.0 / (H**2)) * (1.0 / np.cosh(z_clip)) ** 4
    # For very large |t|, kernel is effectively zero
    val[z > 40.0] = 0.0
    return val


# ---------------------------------------------------------------------
# 2. Numerical Fourier transform (Bochner spectral check)
# ---------------------------------------------------------------------


def fourier_sech4_numeric(
    omega: np.ndarray | float,
    H: float = 1.0,
    T_max_factor: float = 50.0,
    n_grid: int = 20000,
) -> np.ndarray:
    r"""
    Numerical Fourier transform of k_H:

        \hat{k}_H(ω) = ∫_{R} k_H(t) e^{-2π i ω t} dt.

    Since k_H is even and real, we use:

        \hat{k}_H(ω) = 2 ∫_{0}^{∞} k_H(t) cos(2π ω t) dt

    but in practice we integrate over a large symmetric truncation
    [-T_max, T_max] with fine grid.
    """
    omega_arr = np.asarray(omega, dtype=float)
    T_max = T_max_factor * H
    ts = np.linspace(-T_max, T_max, n_grid)
    dt = ts[1] - ts[0]

    k_vals = k_H(ts, H=H)

    results = []
    for w in omega_arr:
        # Because k_H is even, imaginary part vanishes; we only need cos term.
        vals = k_vals * np.cos(2.0 * math.pi * w * ts)
        val = np.trapz(vals, ts)
        results.append(val)

    return np.array(results, dtype=float)


def certify_bochner(H: float = 1.0) -> Tuple[float, float, float, float]:
    """
    Run Bochner-style checks:
      - numeric Fourier transform sampled on a frequency grid
      - random Gram-matrix PSD test on R

    Returns:
      (min_hat_k, max_hat_k, min_eig_gram, max_eig_gram)
    """
    print("==============================================")
    print(" BOCHNER NUMERICAL CERTIFICATION")
    print(" k_H(t) = (6/H^2) sech^4(t/H)")
    print("==============================================\n")

    # 2.1 Spectral check: numerical Fourier transform
    omegas = np.linspace(-10.0, 10.0, 401)  # moderate band; kernel is localized
    print(
        "Computing numerical Fourier transform on ω ∈ [{:.1f}, {:.1f}] ...".format(
            omegas[0], omegas[-1]
        )
    )
    ft_vals = fourier_sech4_numeric(omegas, H=H, T_max_factor=50.0, n_grid=20000)

    min_ft = float(ft_vals.min())
    max_ft = float(ft_vals.max())

    print(f"  min(\\hat{{k}}_H(ω)) ≈ {min_ft:.6e}")
    print(f"  max(\\hat{{k}}_H(ω)) ≈ {max_ft:.6e}")
    if min_ft >= -1e-6:
        print("Bochner spectral status : PASS (nonnegative on sampled grid)\n")
    else:
        print("Bochner spectral status : WARNING (negative values observed)\n")

    # 2.2 Gram-matrix PSD check on random points in R
    np.random.seed(123)
    N = 500
    t_rand = np.random.uniform(-1000.0, 1000.0, N)
    K = np.empty((N, N), dtype=float)
    for i in range(N):
        diff = t_rand[i] - t_rand
        K[i, :] = k_H(diff, H=H)

    # Enforce symmetry explicitly
    K = 0.5 * (K + K.T)

    eigvals = np.linalg.eigvalsh(K)
    min_eig = float(eigvals.min())
    max_eig = float(eigvals.max())

    print(f"Random Gram test on R with N = {N}")
    print(f"  min eigenvalue ≈ {min_eig:.6e}")
    print(f"  max eigenvalue ≈ {max_eig:.6e}")
    if min_eig >= -1e-10:
        print("Bochner Gram status     : PASS (PSD on random cloud)\n")
    else:
        print("Bochner Gram status     : WARNING (negative eigenvalues)\n")

    return min_ft, max_ft, min_eig, max_eig


# ---------------------------------------------------------------------
# 3. Mercer numerical verification on [0, log N]
# ---------------------------------------------------------------------


def certify_mercer(
    H: float = 1.0,
    N_val: float = 100.0,
    grid_size: int = 200,
) -> Tuple[float, float, float, float, float]:
    """
    Mercer-style certification of k_H on [0, log N]:

      - build integral operator K_ij = k_H(t_i - t_j) * dt
      - check PSD / trace-class-like behaviour
      - test Mercer expansion accuracy

    Returns:
      (min_eig, max_eig, trace_approx, max_err_recon, rel_err_recon)
    """
    print("==============================================")
    print(" MERCER NUMERICAL CERTIFICATION")
    print(" Domain: [0, log N]")
    print("==============================================\n")

    a = 0.0
    b = math.log(N_val)
    t_grid = np.linspace(a, b, grid_size)
    dt = (b - a) / (grid_size - 1)

    # Build integral operator matrix: K_ij = k_H(t_i - t_j) * dt
    K = np.empty((grid_size, grid_size), dtype=float)
    for i in range(grid_size):
        diff = t_grid[i] - t_grid
        K[i, :] = k_H(diff, H=H) * dt

    # Enforce symmetry
    K = 0.5 * (K + K.T)

    eigvals, eigvecs = np.linalg.eigh(K)
    # Sort eigenvalues and eigenvectors descending
    idx = np.argsort(eigvals)[::-1]
    eigvals_sorted = eigvals[idx]
    eigvecs_sorted = eigvecs[:, idx]

    min_eig = float(eigvals_sorted[-1])
    max_eig = float(eigvals_sorted[0])
    trace_approx = float(eigvals_sorted.sum())

    print(f"Domain [a, b]          : [0, log({N_val})] ≈ [{a:.2f}, {b:.4f}]")
    print(f"Grid size              : {grid_size}")
    print(f"min eigenvalue         : {min_eig:.6e}")
    print(f"max eigenvalue         : {max_eig:.6e}")
    print(f"trace (∑ λ_n)          : {trace_approx:.6e}")

    if min_eig >= -1e-12:
        print("Mercer positivity      : PASS (PSD integral operator)\n")
    else:
        print("Mercer positivity      : WARNING (negative eigenvalues)\n")

    # Check trace-class-like behavior: eigenvalue decay (informative, not rigorous)
    top_k = min(10, grid_size)
    print(f"Top eigenvalues (λ_1,...,λ_{top_k}) ≈")
    print("  ", ", ".join(f"{eigvals_sorted[i]:.3e}" for i in range(top_k)))
    print()

    # Mercer reconstruction test using top n_keep eigenpairs
    n_keep = min(50, grid_size)
    print(f"Mercer reconstruction with top {n_keep} eigenpairs...")

    K_recon = np.zeros_like(K)
    for n in range(n_keep):
        lam = eigvals_sorted[n]
        phi = eigvecs_sorted[:, n]
        K_recon += lam * np.outer(phi, phi)

    # Undo dt for pointwise kernel comparison
    K_original = K / dt
    K_recon_kernel = K_recon / dt

    max_err = float(np.max(np.abs(K_original - K_recon_kernel)))
    rel_err = max_err / max(1.0, float(np.max(np.abs(K_original))))
    print(f"Max pointwise error    : {max_err:.6e}")
    print(f"Relative max error     : {rel_err:.6e}")
    if max_err < 1e-10:
        print("Mercer reconstruction  : PASS (high-accuracy eigen expansion)\n")
    else:
        print("Mercer reconstruction  : WARNING (non-negligible error)\n")

    return min_eig, max_eig, trace_approx, max_err, rel_err


# ---------------------------------------------------------------------
# 4. Public API for QED / Volumes
# ---------------------------------------------------------------------


def run_full_certification(
    H: float = 1.0,
    N_val: float = 100.0,
    grid_size: int = 200,
) -> dict:
    """
    Unified entrypoint so QED_HILBERT_POLYA_RH_PROOF.py or a new Volume
    can import and call this module programmatically, e.g.:

        from VOLUME_II_KERNAL_DECOMPOSITION.KERNEL_SECH4_BOCHNER_MERCER_CERTIFICATION import (
            run_full_certification
        )
        result = run_full_certification(H=0.5, N_val=100.0, grid_size=200)

    Returns:
      dict with summary statistics from Bochner and Mercer checks.
    """
    min_ft, max_ft, min_eig_gram, max_eig_gram = certify_bochner(H=H)
    (
        min_eig_int,
        max_eig_int,
        trace_int,
        max_err_recon,
        rel_err_recon,
    ) = certify_mercer(H=H, N_val=N_val, grid_size=grid_size)

    return {
        "H": H,
        "N_val": N_val,
        "grid_size": grid_size,
        "fourier_min": min_ft,
        "fourier_max": max_ft,
        "gram_min_eig": min_eig_gram,
        "gram_max_eig": max_eig_gram,
        "integral_min_eig": min_eig_int,
        "integral_max_eig": max_eig_int,
        "integral_trace": trace_int,
        "mercer_max_err": max_err_recon,
        "mercer_rel_err": rel_err_recon,
    }


# ---------------------------------------------------------------------
# 5. Script entrypoint
# ---------------------------------------------------------------------


if __name__ == "__main__":
    # Default parameters chosen to match other volume test scales
    run_full_certification(H=1.0, N_val=100.0, grid_size=200)