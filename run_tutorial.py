#!/usr/bin/env python3
"""
Shear-Velocity Correlation Tutorial

Demonstrates:
1. Generate κ, γ, α fields from power-law potential spectrum
2. Measure correlation functions with TreeCorr
3. Fit power-law parameters (A, n) using MCMC
4. Recover the blind parameters

Physics:
    P_Φ(ℓ) = A × ℓ^n  (potential power spectrum)

    κ = Bℓ²Φ          →  C^κκ = B²ℓ⁴ P_Φ
    α = Cℓe^{iφ}Φ     →  C^αα = C²ℓ² P_Φ
    γ = Dℓ²e^{2iφ}Φ   →  C^γγ = D²ℓ⁴ P_Φ

Usage:
    python run_tutorial.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from generate_fields_lib import (
    generate_phi_from_power_spectrum,
    generate_spin2_field,
    generate_scalar_field,
    generate_vector_field,
)
from measure_correlations import measure_gg_correlation, measure_kk_correlation, measure_gv_correlation
from theory_correlations import get_theory_correlations
from fit_power_law import fit_power_law_cl, run_mcmc_cl, plot_corner, plot_fit_comparison_cl
from power_spectrum import shear_power_spectrum

# =============================================================================
# Configuration
# =============================================================================

NX, NY = 2048, 2048
BOX_SIZE_DEG = 25.0

# Amplitude factors (fixed, known to fitter)
B, C, D = 0.5, 1.0, 0.5

# IR cutoff scale for broken power law (fixed, not fitted)
# Below ell_0, the spectrum flattens to regularize IR divergence
ELL_0 = 10.0

# BLIND parameters (to be recovered)
# Note: index must be < -4 so that C_gg ~ ell^(n+4) decreases with ell
# This ensures the Hankel transform converges numerically
TRUE_AMPLITUDE = 1.0
TRUE_INDEX = -5.0

# MCMC settings
NWALKERS = 32
NSTEPS = 300
BURN_IN = 100

OUTPUT_DIR = Path("output_tutorial")


def main():
    """Run the full tutorial pipeline."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("=" * 60)
    print("Shear-Velocity Correlation Tutorial")
    print("=" * 60)

    # =========================================================================
    # Step 1: Generate fields from blind power-law spectrum
    # =========================================================================
    print("\n[1/5] Generating fields from power-law spectrum...")
    print(f"  True parameters: A = {TRUE_AMPLITUDE:.2e}, n = {TRUE_INDEX}")

    power_config = {
        "type": "power_law",
        "amplitude": TRUE_AMPLITUDE,
        "index": TRUE_INDEX,
        "ell_0": ELL_0,
    }

    phi_real, phi_fourier = generate_phi_from_power_spectrum(
        NX, NY, BOX_SIZE_DEG, power_config, seed=42
    )

    g1, g2 = generate_spin2_field(phi_fourier, D, NX, NY, BOX_SIZE_DEG)
    kappa = generate_scalar_field(phi_fourier, B, NX, NY, BOX_SIZE_DEG)
    v1, v2 = generate_vector_field(phi_fourier, C, NX, NY, BOX_SIZE_DEG)

    print(f"  κ: mean={kappa.mean():.2e}, std={kappa.std():.4f}")
    print(f"  γ₁: mean={g1.mean():.2e}, std={g1.std():.4f}")
    print(f"  α₁: mean={v1.mean():.2e}, std={v1.std():.4f}")

    # =========================================================================
    # Step 2: Measure correlation functions
    # =========================================================================
    print("\n[2/5] Measuring correlation functions with TreeCorr...")

    gg_result = measure_gg_correlation(g1, g2, BOX_SIZE_DEG, nbins=20, min_sep=1.0, max_sep=50.0)
    kk_result = measure_kk_correlation(kappa, BOX_SIZE_DEG, nbins=20, min_sep=1.0, max_sep=50.0)
    gv_result = measure_gv_correlation(g1, g2, v1, v2, BOX_SIZE_DEG, nbins=20, min_sep=1.0, max_sep=50.0)

    # Filter out bins with no pairs (TreeCorr returns zero variance)
    valid = gg_result["npairs"] > 0
    theta = gg_result["theta"][valid]
    xi_plus = gg_result["xi_plus"][valid]
    xi_minus = gg_result["xi_minus"][valid]
    sigma_plus = gg_result["sigma_plus"][valid]
    sigma_minus = gg_result["sigma_minus"][valid]

    print(f"  θ range: {theta.min():.1f}' to {theta.max():.1f}' ({valid.sum()}/{len(valid)} bins valid)")
    print(f"  ξ₊ range: {xi_plus.min():.2e} to {xi_plus.max():.2e}")

    # =========================================================================
    # Step 3: Compare with theory
    # =========================================================================
    print("\n[3/5] Computing theory predictions...")

    theory = get_theory_correlations(TRUE_AMPLITUDE, TRUE_INDEX, theta, B, C, D, ell_0=ELL_0)

    # =========================================================================
    # Step 4: Fit power-law parameters from C_ℓ
    # =========================================================================
    print("\n[4/5] Fitting power-law parameters from C_ℓ...")

    # Measure power spectrum directly from the field
    from fit_power_law import measure_cl_from_field

    box_rad = np.deg2rad(BOX_SIZE_DEG)
    ell_min = 2 * np.pi / box_rad
    ell_max = np.pi * NX / box_rad * 0.8  # Stay below Nyquist
    cl_result = measure_cl_from_field(g1, g2, BOX_SIZE_DEG, n_bins=15,
                                      ell_min=max(20, ell_min), ell_max=ell_max)
    ell_meas = cl_result['ell']
    cl_meas = cl_result['cl']
    cl_err = cl_result['cl_err']

    # Quick optimizer fit
    quick_fit = fit_power_law_cl(ell_meas, cl_meas, cl_err,
                                 initial_guess=(TRUE_AMPLITUDE, TRUE_INDEX), D=D, ell_0=ELL_0)
    print(f"  Quick fit: A = {quick_fit['amplitude']:.2e}, n = {quick_fit['index']:.2f}")
    print(f"  χ²/ndof = {quick_fit['chi2']:.1f}/{quick_fit['ndof']}")

    # MCMC
    print(f"\n  Running MCMC ({NWALKERS} walkers × {NSTEPS} steps)...")
    sampler = run_mcmc_cl(ell_meas, cl_meas, cl_err,
                          nwalkers=NWALKERS, nsteps=NSTEPS,
                          initial_guess=(quick_fit['amplitude'], quick_fit['index']),
                          D=D, ell_0=ELL_0, progress=True)

    flat_samples = sampler.get_chain(discard=BURN_IN, flat=True)
    amp_pct = np.percentile(flat_samples[:, 0], [16, 50, 84])
    idx_pct = np.percentile(flat_samples[:, 1], [16, 50, 84])

    print("\n  MCMC Results:")
    print(f"  A = {amp_pct[1]:.2e} (+{amp_pct[2]-amp_pct[1]:.2e} / -{amp_pct[1]-amp_pct[0]:.2e})")
    print(f"  n = {idx_pct[1]:.2f} (+{idx_pct[2]-idx_pct[1]:.2f} / -{idx_pct[1]-idx_pct[0]:.2f})")
    print(f"  Truth: A = {TRUE_AMPLITUDE:.2e}, n = {TRUE_INDEX}")

    amp_diff = abs(amp_pct[1] - TRUE_AMPLITUDE) / ((amp_pct[2] - amp_pct[0]) / 2)
    idx_diff = abs(idx_pct[1] - TRUE_INDEX) / ((idx_pct[2] - idx_pct[0]) / 2)
    print(f"\n  Recovery: A at {amp_diff:.1f}σ, n at {idx_diff:.1f}σ from truth")

    # =========================================================================
    # Step 5: Generate plots
    # =========================================================================
    print("\n[5/5] Generating plots...")

    # Corner plot
    fig_corner = plot_corner(sampler, truths=[TRUE_AMPLITUDE, TRUE_INDEX], discard=BURN_IN,
                             output_path=OUTPUT_DIR / "corner_plot.png")
    plt.close(fig_corner)
    print(f"  Saved: {OUTPUT_DIR}/corner_plot.png")

    # Fit comparison (C_ℓ based)
    fig_fit = plot_fit_comparison_cl(ell_meas, cl_meas, cl_err, sampler,
                                     truths=[TRUE_AMPLITUDE, TRUE_INDEX], discard=BURN_IN, D=D, ell_0=ELL_0,
                                     output_path=OUTPUT_DIR / "fit_comparison.png")
    plt.close(fig_fit)
    print(f"  Saved: {OUTPUT_DIR}/fit_comparison.png")

    # Power spectrum plot (theory)
    fig_ps, ax_ps = plt.subplots(figsize=(8, 5))
    ell = theory["ell"]
    ax_ps.loglog(ell, theory["C_gg"], label=r"$C_\ell^{\gamma\gamma}$ (shear)")
    ax_ps.loglog(ell, theory["C_kk"], label=r"$C_\ell^{\kappa\kappa}$ (convergence)")
    ax_ps.loglog(ell, theory["C_vv"], label=r"$C_\ell^{\alpha\alpha}$ (velocity)")
    ax_ps.loglog(ell, theory["C_gv"], label=r"$C_\ell^{\alpha\gamma}$ (cross)")
    ax_ps.set_xlabel(r"Multipole $\ell$")
    ax_ps.set_ylabel(r"$C_\ell$")
    ax_ps.set_title(f"Power Spectra (A={TRUE_AMPLITUDE:.0e}, n={TRUE_INDEX})")
    ax_ps.legend()
    ax_ps.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / "power_spectra.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/power_spectra.png")


    # Correlation function comparison - use fine grid for smooth theory curves
    theta_fine = np.geomspace(theta.min(), theta.max(), 100)
    theory_fine = get_theory_correlations(TRUE_AMPLITUDE, TRUE_INDEX, theta_fine, B, C, D, ell_0=ELL_0)

    fig_xi, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.errorbar(theta, xi_plus, yerr=sigma_plus, fmt='o', ms=4, label='Measured')
    ax.plot(theta_fine, theory_fine["xi_plus"], 'C1-', label='Theory')
    ax.set_xscale('log')
    ax.set_xlabel(r"$\theta$ [arcmin]")
    ax.set_ylabel(r"$\xi_+(\theta)$")
    ax.set_title("Shear ξ₊")
    ax.legend()

    ax = axes[1]
    ax.errorbar(theta, xi_minus, yerr=sigma_minus, fmt='o', ms=4, label='Measured ξ₋')
    ax.plot(theta_fine, theory_fine["xi_minus"], 'C1-', label='Theory ξ₋')
    ax.set_xscale('log')
    ax.set_xlabel(r"$\theta$ [arcmin]")
    ax.set_ylabel(r"$\xi_-(\theta)$")
    ax.set_title(r"Shear ξ₋ (J₄ transform)")
    ax.legend()

    ax = axes[2]
    gv_valid = gv_result["npairs"] > 0
    gv_theta = gv_result["theta"][gv_valid]
    gv_xi_plus = gv_result["xi_plus"][gv_valid]
    gv_sigma_plus = gv_result["sigma_plus"][gv_valid]
    ax.errorbar(gv_theta, gv_xi_plus, yerr=gv_sigma_plus, fmt='o', ms=4, label=r'Measured $\xi_+$')
    ax.plot(theta_fine, theory_fine["xi_gv_plus"], 'C1-', label=r'Theory $\xi_+$ (J₁)')
    ax.set_xscale('log')
    ax.set_xlabel(r"$\theta$ [arcmin]")
    ax.set_ylabel(r"$\xi_{GV}(\theta)$")
    ax.set_title("Shear-Velocity Cross (GV)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_functions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/correlation_functions.png")

    # Field visualization
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, (field, title) in zip(axes, [(kappa, "κ"), (g1, "γ₁"), (v1, "α₁"), (phi_real, "Φ")]):
        vmax = 3 * field.std()
        im = ax.imshow(field, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        ax.set_title(title)
        ax.set_xlabel("x [pixels]")
        ax.set_ylabel("y [pixels]")
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fields.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/fields.png")

    print("\n" + "=" * 60)
    print("Tutorial complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
