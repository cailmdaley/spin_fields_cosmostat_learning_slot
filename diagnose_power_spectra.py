#!/usr/bin/env python3
"""
Diagnostic: Compare measured vs theoretical power spectra.

This isolates whether any offset is in:
- Field generation (ratio ≠ 1 in C_ℓ space)
- Hankel transform (ratio = 1 in C_ℓ but ≠ 1 in ξ)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from generate_fields_lib import (
    generate_phi_from_power_spectrum,
    generate_scalar_field,
    generate_vector_field,
    generate_spin2_field,
)
from measure_power_spectra import compute_all_power_spectra
from power_spectrum import (
    power_law_spectrum,
    convergence_power_spectrum,
    velocity_power_spectrum,
    shear_power_spectrum,
)

# Configuration
NX, NY = 512, 512
BOX_SIZE_DEG = 50.0
SEED = 42

# Amplitude factors (matching run_tutorial.py convention: B=κ, C=α, D=γ)
B_KAPPA = 0.5   # convergence
C_ALPHA = 1.0   # velocity
D_GAMMA = 0.5   # shear

# Potential spectrum
AMPLITUDE = 1.0
INDEX = -5.0  # steep spectrum

# NaMaster config
NAMASTER_CONFIG = {
    'n_bins': 20,
    'ell_min': 50,
    'ell_max': 2000,
    'use_log_bins': True,
}


def main():
    print("=" * 60)
    print("Power Spectrum Diagnostic")
    print("=" * 60)
    print(f"Grid: {NX}x{NY}, Box: {BOX_SIZE_DEG}°")
    print(f"Spectrum: A={AMPLITUDE}, n={INDEX}")
    print(f"Amplitudes: B(κ)={B_KAPPA}, C(α)={C_ALPHA}, D(γ)={D_GAMMA}")

    # Generate fields
    print("\nGenerating fields...")
    power_config = {"type": "power_law", "amplitude": AMPLITUDE, "index": INDEX}
    phi_real, phi_fourier = generate_phi_from_power_spectrum(
        NX, NY, BOX_SIZE_DEG, power_config, seed=SEED
    )

    kappa = generate_scalar_field(phi_fourier, B_KAPPA, NX, NY, BOX_SIZE_DEG)
    v1, v2 = generate_vector_field(phi_fourier, C_ALPHA, NX, NY, BOX_SIZE_DEG)
    g1, g2 = generate_spin2_field(phi_fourier, D_GAMMA, NX, NY, BOX_SIZE_DEG)

    print(f"  Φ: std={phi_real.std():.4e}")
    print(f"  κ: std={kappa.std():.4e}")
    print(f"  α: std={np.sqrt(v1.var() + v2.var()):.4e}")
    print(f"  γ: std={np.sqrt(g1.var() + g2.var()):.4e}")

    # Measure power spectra
    print("\nMeasuring power spectra with NaMaster...")
    results = compute_all_power_spectra(
        phi_real, kappa, (v1, v2), (g1, g2),
        BOX_SIZE_DEG, NAMASTER_CONFIG
    )

    ell = results['ell']

    # Compute theory
    print("\nComputing theoretical power spectra...")
    P_phi_theory = power_law_spectrum(ell, AMPLITUDE, INDEX)
    C_kk_theory = convergence_power_spectrum(ell, AMPLITUDE, INDEX, B_KAPPA)
    C_vv_theory = velocity_power_spectrum(ell, AMPLITUDE, INDEX, C_ALPHA)
    C_gg_theory = shear_power_spectrum(ell, AMPLITUDE, INDEX, D_GAMMA)

    # Compare
    print("\n" + "=" * 60)
    print("RATIOS: Measured / Theory")
    print("=" * 60)

    # Use measured Phi as baseline (accounts for any FFT normalization)
    phi_ratio = results['phi_auto'] / P_phi_theory
    print(f"\nΦ (potential):")
    print(f"  Median ratio: {np.median(phi_ratio):.4f}")
    print(f"  Range: {phi_ratio.min():.4f} to {phi_ratio.max():.4f}")

    # For derived fields, use measured Phi to remove FFT normalization effects
    P_phi_measured = results['phi_auto']

    # Convergence (spin-0)
    C_kk_from_phi = B_KAPPA**2 * ell**4 * P_phi_measured
    kk_ratio = results['scalar_auto'] / C_kk_from_phi
    print(f"\nκ (convergence, spin-0):")
    print(f"  Median ratio vs Φ-based theory: {np.median(kk_ratio):.4f}")
    print(f"  Range: {kk_ratio.min():.4f} to {kk_ratio.max():.4f}")

    # Velocity (spin-1)
    C_vv_from_phi = C_ALPHA**2 * ell**2 * P_phi_measured
    vv_ee_ratio = results['vector_auto_EE'] / C_vv_from_phi
    vv_bb_ratio = results['vector_auto_BB'] / C_vv_from_phi
    print(f"\nα (velocity, spin-1):")
    print(f"  EE median ratio: {np.median(vv_ee_ratio):.4f}")
    print(f"  BB median ratio: {np.median(vv_bb_ratio):.4f}")
    print(f"  EE+BB median ratio: {np.median((results['vector_auto_EE'] + results['vector_auto_BB']) / C_vv_from_phi):.4f}")
    print(f"  BB range: {vv_bb_ratio.min():.4f} to {vv_bb_ratio.max():.4f}")
    vv_ratio = vv_bb_ratio  # Use BB since that's where the power is

    # Shear (spin-2)
    C_gg_from_phi = D_GAMMA**2 * ell**4 * P_phi_measured
    gg_ratio = results['spin2_auto_EE'] / C_gg_from_phi
    print(f"\nγ (shear, spin-2 E-mode):")
    print(f"  Median ratio vs Φ-based theory: {np.median(gg_ratio):.4f}")
    print(f"  Range: {gg_ratio.min():.4f} to {gg_ratio.max():.4f}")

    # Check B-mode (should be ~0)
    gg_bb_ratio = np.median(np.abs(results['spin2_auto_BB']) / results['spin2_auto_EE'])
    print(f"  B-mode / E-mode ratio: {gg_bb_ratio:.4e} (should be ~0)")

    # Plot
    print("\nGenerating diagnostic plot...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Φ
    ax = axes[0, 0]
    ax.loglog(ell, results['phi_auto'], 'o', ms=5, label='Measured')
    ax.loglog(ell, P_phi_theory, '-', lw=2, label='Theory')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$P_\Phi(\ell)$')
    ax.set_title(f'Potential Φ (median ratio: {np.median(phi_ratio):.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # κ
    ax = axes[0, 1]
    ax.loglog(ell, results['scalar_auto'], 'o', ms=5, label='Measured')
    ax.loglog(ell, C_kk_from_phi, '-', lw=2, label='Theory (from Φ)')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$C_\ell^{\kappa\kappa}$')
    ax.set_title(f'Convergence κ (median ratio: {np.median(kk_ratio):.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # α (VV)
    ax = axes[1, 0]
    ax.loglog(ell, results['vector_auto_EE'], 'o', ms=5, label='Measured EE')
    ax.loglog(ell, C_vv_from_phi, '-', lw=2, label='Theory (from Φ)')
    ax.loglog(ell, np.abs(results['vector_auto_BB']), 's', ms=3, alpha=0.5, label='Measured BB (abs)')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$C_\ell^{\alpha\alpha}$')
    ax.set_title(f'Velocity α (median ratio: {np.median(vv_ratio):.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # γ (GG)
    ax = axes[1, 1]
    ax.loglog(ell, results['spin2_auto_EE'], 'o', ms=5, label='Measured EE')
    ax.loglog(ell, C_gg_from_phi, '-', lw=2, label='Theory (from Φ)')
    ax.loglog(ell, np.abs(results['spin2_auto_BB']), 's', ms=3, alpha=0.5, label='Measured BB (abs)')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$C_\ell^{\gamma\gamma}$')
    ax.set_title(f'Shear γ (median ratio: {np.median(gg_ratio):.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Power Spectrum Diagnostic (n={INDEX}, seed={SEED})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = Path("output_tutorial") / "power_spectrum_diagnostic.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    # Ratio plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogx(ell, phi_ratio, 'o-', ms=4, label=f'Φ (median={np.median(phi_ratio):.3f})')
    ax.semilogx(ell, kk_ratio, 's-', ms=4, label=f'κ (median={np.median(kk_ratio):.3f})')
    ax.semilogx(ell, vv_ratio, '^-', ms=4, label=f'α (median={np.median(vv_ratio):.3f})')
    ax.semilogx(ell, gg_ratio, 'v-', ms=4, label=f'γ (median={np.median(gg_ratio):.3f})')
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Expected')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel('Measured / Theory')
    ax.set_title(f'Power Spectrum Ratios (n={INDEX}, seed={SEED})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 2.0)

    output_path = Path("output_tutorial") / "power_spectrum_ratios.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"If VV ratio ≈ 1.0: offset is in Hankel transform, not field generation")
    print(f"If VV ratio ≠ 1.0: offset is in field generation or power spectrum measurement")


if __name__ == "__main__":
    main()
