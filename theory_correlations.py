"""
Theory correlation functions via smooth numerical Hankel transforms.

The Hankel transform relates power spectra to real-space correlations:
    ξ(θ) = ∫ (ℓ dℓ)/(2π) C_ℓ J_n(ℓθ)

Bessel function orders for different correlations:
- GG (spin-2×spin-2): ξ₊ uses J_0, ξ₋ uses J_4
- VV (spin-1×spin-1): ξ₊ uses J_0, ξ₋ uses J_2
- GV (spin-2×spin-1): ξ₊ uses J_1, ξ₋ uses J_3
- KK (spin-0×spin-0): ξ uses J_0

IR regularization via broken power law:
    The potential spectrum uses P_Φ(ℓ) = A × ℓ^n / (1 + (ℓ₀/ℓ)²), which
    flattens at low ℓ. This ensures all Hankel integrals converge naturally:

    - At ℓ >> ℓ₀: P_Φ → A ℓ^n (unchanged high-ℓ behavior)
    - At ℓ << ℓ₀: P_Φ → A ℓ₀^{-2} ℓ^{n+2} (2 powers shallower)

    For n=-5, the low-ℓ behavior becomes ℓ^{-3}, making even VV ξ₊ converge:
    - VV (α=-3→-1 at low ℓ): ξ₊ (J_0) now converges (α+n+2 = +1 > 0)

    No box_size_deg parameter is needed - the broken power law handles IR.
"""

import numpy as np
from scipy.special import jv
from scipy.integrate import trapezoid


def hankel_xi(C_ell_func, theta_rad, order, ell_min=2, ell_max=1e5, n_ell=100000):
    """
    Compute ξ(θ) via numerical Hankel transform over dense linear-spaced ℓ.

    ξ(θ) = ∫ (ℓ dℓ)/(2π) C_ℓ J_n(ℓθ)

    Uses trapezoid integration with dense sampling to properly capture
    Bessel function oscillations across all angular scales.

    Parameters
    ----------
    C_ell_func : callable
        Power spectrum function C(ℓ) that takes ℓ array and returns C_ℓ
    theta_rad : array
        Angles in radians
    order : int
        Bessel function order (0, 1, 2, 3, or 4)
    ell_min : float
        Minimum ℓ for integration. Acts as IR cutoff; should match
        the fundamental mode 2π/L for a box of size L radians.
    ell_max : float
        Maximum ℓ for integration (default: 1e5)
    n_ell : int
        Number of ℓ points (default: 100000)

    Returns
    -------
    xi : array
        Correlation function at theta_rad
    """
    ell = np.linspace(ell_min, ell_max, n_ell)
    C_ell = C_ell_func(ell)

    theta_rad = np.atleast_1d(theta_rad)
    xi = np.zeros(len(theta_rad))

    for i, th in enumerate(theta_rad):
        integrand = ell / (2 * np.pi) * C_ell * jv(order, ell * th)
        xi[i] = trapezoid(integrand, ell)

    return xi


def get_theory_correlations(amplitude, index, theta_arcmin, B=0.5, C=1.0, D=0.5,
                            ell_0=10.0):
    """
    Compute all theory correlation functions for broken power-law spectrum.

    Uses smooth numerical Hankel transforms. The broken power law
    P_Φ(ℓ) = A × ℓ^n / (1 + (ℓ₀/ℓ)²) naturally regularizes IR divergences,
    so no box_size parameter is needed.

    Parameters
    ----------
    amplitude : float
        Potential amplitude A
    index : float
        Potential spectral index n
    theta_arcmin : array
        Angles in arcminutes
    B, C, D : float
        Amplitude factors for κ, α, γ
    ell_0 : float
        IR cutoff scale (default: 10). Below this, spectrum flattens.

    Returns
    -------
    dict with keys:
        'theta_arcmin': input angles
        'xi_plus': ξ₊(θ) for shear
        'xi_minus': ξ₋(θ) for shear
        'xi_kk': κ auto-correlation
        'xi_vv_plus', 'xi_vv_minus': velocity auto-correlation
        'xi_gv_plus', 'xi_gv_minus': shear-velocity cross (J₁ and J₃)
        'ell_0': the IR cutoff scale used
    """
    from power_spectrum import (shear_power_spectrum, convergence_power_spectrum,
                                velocity_power_spectrum, shear_velocity_cross_spectrum)

    theta_rad = np.deg2rad(theta_arcmin / 60.0)

    # With broken power law, integrals converge naturally - use small ell_min
    ell_min = 0.5  # Can go low since broken power law regularizes IR

    # Build C_ell functions that close over the parameters (including ell_0)
    def C_gg_func(ell):
        return shear_power_spectrum(ell, amplitude, index, D, ell_0)

    def C_kk_func(ell):
        return convergence_power_spectrum(ell, amplitude, index, B, ell_0)

    def C_vv_func(ell):
        return velocity_power_spectrum(ell, amplitude, index, C, ell_0)

    def C_gv_func(ell):
        return shear_velocity_cross_spectrum(ell, amplitude, index, C, D, ell_0)

    # Compute auto-correlations via Hankel transforms
    xi_plus = hankel_xi(C_gg_func, theta_rad, order=0, ell_min=ell_min)
    xi_minus = hankel_xi(C_gg_func, theta_rad, order=4, ell_min=ell_min)
    xi_kk = hankel_xi(C_kk_func, theta_rad, order=0, ell_min=ell_min)
    xi_vv_plus = hankel_xi(C_vv_func, theta_rad, order=0, ell_min=ell_min)
    xi_vv_minus = hankel_xi(C_vv_func, theta_rad, order=2, ell_min=ell_min)

    # VG cross-correlation
    xi_gv_plus = hankel_xi(C_gv_func, theta_rad, order=1, ell_min=ell_min)
    xi_gv_minus = hankel_xi(C_gv_func, theta_rad, order=3, ell_min=ell_min)

    # TreeCorr sign conventions:
    # After fixing shear sign to main.tex (γ̃ = -ℓ₊²Φ̃), we apply empirical
    # sign corrections to match TreeCorr's internal phase conventions:
    # 1. VV xi_minus: TreeCorr VVCorrelation uses opposite sign for ⟨v v⟩ spin-2
    # 2. GV xi_minus: TreeCorr GVCorrelation uses opposite sign for J_3 term
    xi_vv_minus = -xi_vv_minus
    xi_gv_minus = -xi_gv_minus

    # 1D power spectra for reference (log-spaced for smooth curves)
    ell = np.geomspace(10, 1e4, 200)

    return {
        "theta_arcmin": theta_arcmin,
        "ell": ell,
        "ell_0": ell_0,
        "C_gg": shear_power_spectrum(ell, amplitude, index, D, ell_0),
        "C_kk": convergence_power_spectrum(ell, amplitude, index, B, ell_0),
        "C_vv": velocity_power_spectrum(ell, amplitude, index, C, ell_0),
        "C_gv": shear_velocity_cross_spectrum(ell, amplitude, index, C, D, ell_0),
        "xi_plus": xi_plus,
        "xi_minus": xi_minus,
        "xi_kk": xi_kk,
        "xi_vv_plus": xi_vv_plus,
        "xi_vv_minus": xi_vv_minus,
        "xi_gv_plus": xi_gv_plus,
        "xi_gv_minus": xi_gv_minus,
        # Backwards compatibility aliases
        "xi_gv_t": xi_gv_plus,
        "xi_gv_x": xi_gv_minus,
    }
