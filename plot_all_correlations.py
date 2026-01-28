#!/usr/bin/env python3
"""
Plot all 6 correlation functions: GG, VV, GV (each with ξ+ and ξ-)
Compares TreeCorr measurements against theory from numerical Hankel transforms.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
from theory_correlations import get_theory_correlations

# Configuration
NX, NY = 2048, 2048  # Match tutorial settings
BOX_SIZE_DEG = 25.0
B, C, D = 0.5, 1.0, 0.5
ELL_0 = 10.0
AMPLITUDE = 1.0
INDEX = -5.0

# Measurement settings
NBINS = 15
MIN_SEP = 2.0   # arcmin
MAX_SEP = 40.0  # arcmin (stay within box)


def main():
    print("Generating fields...")
    power_config = {
        "type": "power_law",
        "amplitude": AMPLITUDE,
        "index": INDEX,
        "ell_0": ELL_0,
    }
    phi_real, phi_fourier = generate_phi_from_power_spectrum(
        NX, NY, BOX_SIZE_DEG, power_config, seed=42
    )
    g1, g2 = generate_spin2_field(phi_fourier, D, NX, NY, BOX_SIZE_DEG)
    v1, v2 = generate_vector_field(phi_fourier, C, NX, NY, BOX_SIZE_DEG)

    print("Measuring correlations with TreeCorr...")
    gg = measure_gg_correlation(g1, g2, BOX_SIZE_DEG, NBINS, MIN_SEP, MAX_SEP)
    vv = measure_vv_correlation(v1, v2, BOX_SIZE_DEG, NBINS, MIN_SEP, MAX_SEP)
    gv = measure_gv_correlation(g1, g2, v1, v2, BOX_SIZE_DEG, NBINS, MIN_SEP, MAX_SEP)

    # Filter valid bins
    gg_valid = gg["npairs"] > 0
    vv_valid = vv["npairs"] > 0
    gv_valid = gv["npairs"] > 0

    print("Computing theory predictions...")
    theta_fine = np.geomspace(MIN_SEP, MAX_SEP, 100)
    theory = get_theory_correlations(AMPLITUDE, INDEX, theta_fine, B, C, D, ell_0=ELL_0)

    # Set up figure: 2 rows (ξ+, ξ-) × 3 columns (GG, VV, GV)
    sns.set_theme(style="ticks", context="paper", font_scale=1.1)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)

    # Color scheme
    data_color = "C0"
    theory_color = "C1"

    # --- Row 0: ξ+ ---

    # GG ξ+
    ax = axes[0, 0]
    theta = gg["theta"][gg_valid]
    ax.errorbar(theta, gg["xi_plus"][gg_valid], yerr=gg["sigma_plus"][gg_valid],
                fmt='o', ms=4, color=data_color, label="Measured")
    ax.plot(theta_fine, theory["xi_plus"], '-', color=theory_color, lw=1.5, label="Theory")
    ax.set_ylabel(r"$\xi_+$")
    ax.set_title(r"GG (shear $\times$ shear)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")

    # VV ξ+
    ax = axes[0, 1]
    theta = vv["theta"][vv_valid]
    ax.errorbar(theta, vv["xi_plus"][vv_valid], yerr=vv["sigma_plus"][vv_valid],
                fmt='o', ms=4, color=data_color)
    ax.plot(theta_fine, theory["xi_vv_plus"], '-', color=theory_color, lw=1.5)
    ax.set_title(r"VV (velocity $\times$ velocity)", fontsize=11)

    # GV ξ+
    ax = axes[0, 2]
    theta = gv["theta"][gv_valid]
    ax.errorbar(theta, gv["xi_plus"][gv_valid], yerr=gv["sigma_plus"][gv_valid],
                fmt='o', ms=4, color=data_color)
    ax.plot(theta_fine, theory["xi_gv_plus"], '-', color=theory_color, lw=1.5)
    ax.set_title(r"GV (shear $\times$ velocity)", fontsize=11)

    # --- Row 1: ξ- ---

    # GG ξ-
    ax = axes[1, 0]
    theta = gg["theta"][gg_valid]
    ax.errorbar(theta, gg["xi_minus"][gg_valid], yerr=gg["sigma_minus"][gg_valid],
                fmt='o', ms=4, color=data_color)
    ax.plot(theta_fine, theory["xi_minus"], '-', color=theory_color, lw=1.5)
    ax.set_ylabel(r"$\xi_-$")
    ax.set_xlabel(r"$\theta$ [arcmin]")

    # VV ξ-
    ax = axes[1, 1]
    theta = vv["theta"][vv_valid]
    ax.errorbar(theta, vv["xi_minus"][vv_valid], yerr=vv["sigma_minus"][vv_valid],
                fmt='o', ms=4, color=data_color)
    ax.plot(theta_fine, theory["xi_vv_minus"], '-', color=theory_color, lw=1.5)
    ax.set_xlabel(r"$\theta$ [arcmin]")

    # GV ξ-
    ax = axes[1, 2]
    theta = gv["theta"][gv_valid]
    ax.errorbar(theta, gv["xi_minus"][gv_valid], yerr=gv["sigma_minus"][gv_valid],
                fmt='o', ms=4, color=data_color)
    ax.plot(theta_fine, theory["xi_gv_minus"], '-', color=theory_color, lw=1.5)
    ax.set_xlabel(r"$\theta$ [arcmin]")

    # Formatting
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.tick_params(which="both", direction="in", top=True, right=True)
        sns.despine(ax=ax)

    # Row labels on the right
    axes[0, 2].annotate(r"$\xi_+$", xy=(1.08, 0.5), xycoords="axes fraction",
                        fontsize=12, va="center", rotation=-90)
    axes[1, 2].annotate(r"$\xi_-$", xy=(1.08, 0.5), xycoords="axes fraction",
                        fontsize=12, va="center", rotation=-90)

    fig.suptitle(f"Correlation Functions: $P_\\Phi(\\ell) = A\\ell^n / (1 + (\\ell_0/\\ell)^2)$\n"
                 f"$A={AMPLITUDE}$, $n={INDEX}$, $\\ell_0={ELL_0}$",
                 fontsize=12, y=0.98)

    plt.tight_layout(rect=[0, 0, 0.97, 0.94])

    outpath = "output_tutorial/all_correlations.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    main()
