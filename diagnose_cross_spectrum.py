#!/usr/bin/env python3
"""
Diagnose the shear-velocity cross-correlation discrepancy.

The issue: theory doesn't match measured GV correlation.

Key insight: The gradient ∇Φ introduces a factor of i in Fourier space:
    α̃ = i C ℓ e^{iφ} Φ̃

So the cross-spectrum ⟨γ̃ α̃*⟩ is NOT simply CD ℓ³ P_Φ.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import jv

from generate_fields_lib import (
    generate_phi_from_power_spectrum,
    generate_spin2_field,
    generate_vector_field,
)
from measure_correlations import (
    measure_gg_correlation,
    measure_vv_correlation,
    measure_gv_correlation,
)

sns.set_theme(style="ticks", context="paper", font_scale=1.1)

# Configuration
NX, NY = 512, 512
BOX_SIZE_DEG = 50.0
B, C, D = 0.5, 1.0, 0.5
TRUE_AMPLITUDE = 1.0
TRUE_INDEX = -5.0

def measure_cross_spectrum_directly(g1, g2, v1, v2, box_size_deg):
    """
    Measure the cross-spectrum ⟨γ̃ ṽ*⟩ directly in Fourier space.
    """
    nx, ny = g1.shape
    box_rad = np.deg2rad(box_size_deg)

    # FFT the fields
    gamma = g1 + 1j * g2
    vel = v1 + 1j * v2

    gamma_fft = np.fft.fft2(gamma)
    vel_fft = np.fft.fft2(vel)

    # Cross-spectrum: γ̃ · ṽ*
    cross_2d = gamma_fft * np.conj(vel_fft) * box_rad**2 / (nx * ny)**2

    # Create ℓ grid
    kx = np.fft.fftfreq(nx, d=box_rad / nx)
    ky = np.fft.fftfreq(ny, d=box_rad / ny)
    kxx, kyy = np.meshgrid(kx, ky, indexing='ij')
    ell = 2 * np.pi * np.sqrt(kxx**2 + kyy**2)
    phi_ell = np.arctan2(kyy, kxx)

    # Bin by ℓ
    ell_bins = np.geomspace(20, 1000, 16)
    ell_centers = np.sqrt(ell_bins[:-1] * ell_bins[1:])

    cross_real, cross_imag = [], []
    cross_phase = []
    for i in range(len(ell_bins) - 1):
        mask = (ell >= ell_bins[i]) & (ell < ell_bins[i + 1])
        if mask.sum() > 0:
            cross_vals = cross_2d[mask]
            cross_real.append(np.mean(cross_vals.real))
            cross_imag.append(np.mean(cross_vals.imag))
            # Check phase relative to e^{iφ}
            exp_iphi = np.exp(1j * phi_ell[mask])
            phase_factor = cross_vals / exp_iphi
            cross_phase.append(np.mean(phase_factor))

    return {
        'ell': ell_centers[:len(cross_real)],
        'cross_real': np.array(cross_real),
        'cross_imag': np.array(cross_imag),
        'cross_phase': np.array(cross_phase),
    }


def theory_cross_spectrum(ell, amplitude, index, C_coeff, D_coeff):
    """
    Theory cross-spectrum including the i factor from gradient.

    γ̃ = D ℓ² e^{2iφ} Φ̃
    ṽ = i C ℓ e^{iφ} Φ̃

    ⟨γ̃ ṽ*⟩ = D ℓ² e^{2iφ} · (-i C ℓ e^{-iφ}) · P_Φ
           = -i CD ℓ³ e^{iφ} P_Φ

    After averaging over φ in a ring, only the radial part survives in correlation.
    But the i factor remains!
    """
    P_phi = amplitude * ell**index
    # The cross-spectrum is purely imaginary!
    return -1j * C_coeff * D_coeff * ell**3 * P_phi


def main():
    print("Generating fields...")
    power_config = {
        "type": "power_law",
        "amplitude": TRUE_AMPLITUDE,
        "index": TRUE_INDEX,
    }

    phi_real, phi_fourier = generate_phi_from_power_spectrum(
        NX, NY, BOX_SIZE_DEG, power_config, seed=42
    )

    g1, g2 = generate_spin2_field(phi_fourier, D, NX, NY, BOX_SIZE_DEG)
    v1, v2 = generate_vector_field(phi_fourier, C, NX, NY, BOX_SIZE_DEG)

    print("Measuring cross-spectrum directly...")
    cross_result = measure_cross_spectrum_directly(g1, g2, v1, v2, BOX_SIZE_DEG)

    # Theory
    ell = cross_result['ell']
    theory = theory_cross_spectrum(ell, TRUE_AMPLITUDE, TRUE_INDEX, C, D)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Real part
    ax = axes[0, 0]
    ax.plot(ell, cross_result['cross_real'], 'o-', label='Measured')
    ax.plot(ell, theory.real, '--', label='Theory (real)')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'Re[$C_{\gamma v}$]')
    ax.set_title('Cross-spectrum real part')
    ax.legend()
    ax.axhline(0, color='gray', ls=':')

    # Imaginary part
    ax = axes[0, 1]
    ax.plot(ell, cross_result['cross_imag'], 'o-', label='Measured')
    ax.plot(ell, theory.imag, '--', label='Theory (imag)')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'Im[$C_{\gamma v}$]')
    ax.set_title('Cross-spectrum imaginary part')
    ax.legend()
    ax.axhline(0, color='gray', ls=':')

    # Phase factor (after dividing by e^{iφ})
    ax = axes[1, 0]
    phase_real = cross_result['cross_phase'].real
    phase_imag = cross_result['cross_phase'].imag
    ax.plot(ell, phase_real, 'o-', label='Real part')
    ax.plot(ell, phase_imag, 's-', label='Imag part')
    # Theory: after dividing by e^{iφ}, should get -i CD ℓ³ P_Φ
    theory_phase = -1j * C * D * ell**3 * TRUE_AMPLITUDE * ell**TRUE_INDEX
    ax.plot(ell, theory_phase.real, '--', label='Theory real')
    ax.plot(ell, theory_phase.imag, '--', label='Theory imag')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$C_{\gamma v} / e^{i\phi}$')
    ax.set_title(r'Cross-spectrum $\times e^{-i\phi}$')
    ax.legend()

    # Measure correlations
    print("Measuring correlations with TreeCorr...")
    gv_result = measure_gv_correlation(g1, g2, v1, v2, BOX_SIZE_DEG, nbins=15, min_sep=2.0, max_sep=200.0)

    ax = axes[1, 1]
    valid = gv_result['npairs'] > 0
    ax.errorbar(gv_result['theta'][valid], gv_result['xi_plus'][valid],
                yerr=gv_result['sigma_plus'][valid], fmt='o', label=r'$\xi_+$ (measured)')
    ax.errorbar(gv_result['theta'][valid], gv_result['xi_minus'][valid],
                yerr=gv_result['sigma_minus'][valid], fmt='s', label=r'$\xi_-$ (measured)')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\theta$ [arcmin]')
    ax.set_ylabel(r'$\xi(\theta)$')
    ax.set_title('GV correlation (TreeCorr)')
    ax.legend()
    ax.axhline(0, color='gray', ls=':')

    plt.tight_layout()
    plt.savefig('output_tutorial/diagnose_cross_spectrum.png', dpi=150)
    print("Saved: output_tutorial/diagnose_cross_spectrum.png")
    plt.show()


if __name__ == "__main__":
    main()
