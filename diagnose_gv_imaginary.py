#!/usr/bin/env python3
"""
Check if GV correlation has significant imaginary parts.

For E-mode shear (γ) × B-mode velocity (gradient ∇Φ):
⟨γ v*⟩ = -i CD ℓ³ P_Φ × e^{iφ}

The -i factor means the correlation should be imaginary!
TreeCorr outputs xip_im for this.
"""

import numpy as np
import treecorr

from generate_fields_lib import (
    generate_phi_from_power_spectrum,
    generate_spin2_field,
    generate_vector_field,
)

# Configuration
NX, NY = 512, 512
BOX_SIZE_DEG = 50.0
B, C, D = 0.5, 1.0, 0.5
TRUE_AMPLITUDE = 1.0
TRUE_INDEX = -5.0

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

    # Create catalogs
    x = np.linspace(0, BOX_SIZE_DEG, NX, endpoint=False)
    y = np.linspace(0, BOX_SIZE_DEG, NY, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    cat_g = treecorr.Catalog(
        x=xx.flatten(), y=yy.flatten(),
        g1=g1.flatten(), g2=g2.flatten(),
        x_units='deg', y_units='deg'
    )
    cat_v = treecorr.Catalog(
        x=xx.flatten(), y=yy.flatten(),
        v1=v1.flatten(), v2=v2.flatten(),
        x_units='deg', y_units='deg'
    )

    # Measure GV correlation
    print("Measuring GV correlation...")
    gv = treecorr.GVCorrelation(
        nbins=15, min_sep=2.0, max_sep=200.0,
        sep_units='arcmin', bin_slop=0.1
    )
    gv.process(cat_g, cat_v)

    theta = np.exp(gv.meanlogr)

    print("\n" + "="*60)
    print("GV Correlation Results")
    print("="*60)
    print(f"{'theta':>10} {'xip':>12} {'xip_im':>12} {'xim':>12} {'xim_im':>12}")
    print("-"*60)
    for i in range(len(theta)):
        print(f"{theta[i]:10.2f} {gv.xip[i]:12.6f} {gv.xip_im[i]:12.6f} {gv.xim[i]:12.6f} {gv.xim_im[i]:12.6f}")

    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Mean |xip|:    {np.mean(np.abs(gv.xip)):12.6f}")
    print(f"Mean |xip_im|: {np.mean(np.abs(gv.xip_im)):12.6f}")
    print(f"Mean |xim|:    {np.mean(np.abs(gv.xim)):12.6f}")
    print(f"Mean |xim_im|: {np.mean(np.abs(gv.xim_im)):12.6f}")

    # Check ratio
    print("\n" + "="*60)
    print("Key insight: For E-shear × B-velocity, expect xip_im >> xip")
    print("="*60)
    ratio = np.mean(np.abs(gv.xip_im)) / np.mean(np.abs(gv.xip)) if np.mean(np.abs(gv.xip)) > 0 else float('inf')
    print(f"|xip_im| / |xip| ratio: {ratio:.2f}")

    # Also check VV correlation for comparison
    print("\n" + "="*60)
    print("VV Correlation (for comparison)")
    print("="*60)
    vv = treecorr.VVCorrelation(
        nbins=15, min_sep=2.0, max_sep=200.0,
        sep_units='arcmin', bin_slop=0.1
    )
    vv.process(cat_v)
    print(f"Mean |vv.xip|:    {np.mean(np.abs(vv.xip)):12.6f}")
    print(f"Mean |vv.xip_im|: {np.mean(np.abs(vv.xip_im)):12.6f}")
    print(f"(VV should have xip_im ≈ 0 for auto-correlation)")


if __name__ == "__main__":
    main()
