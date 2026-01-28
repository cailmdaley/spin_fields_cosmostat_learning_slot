"""
Power spectrum module for broken power-law potential spectra.

P_Φ(ℓ) = A × ℓ^n / (1 + (ℓ₀/ℓ)²)

This broken power law flattens at low ℓ to regularize IR divergences:
- ℓ >> ℓ₀: P_Φ → A ℓ^n (unchanged high-ℓ behavior)
- ℓ << ℓ₀: P_Φ → A ℓ₀^{-2} ℓ^{n+2} (2 powers shallower)

For n=-5, the low-ℓ behavior becomes ℓ^{-3}, ensuring convergence of all
Hankel integrals including VV ξ₊ (which diverges for pure power law).

Derived power spectra:
    C^κκ_ℓ = B² ℓ⁴ P_Φ(ℓ)
    C^αα_ℓ = C² ℓ² P_Φ(ℓ)
    C^γγ_ℓ = D² ℓ⁴ P_Φ(ℓ)
    C^αγ_ℓ = CD ℓ³ P_Φ(ℓ)
"""

import numpy as np

# Default IR cutoff scale (multipole where spectrum flattens)
DEFAULT_ELL_0 = 10.0


def power_law_spectrum(ell, amplitude, index, ell_0=DEFAULT_ELL_0):
    """
    Broken power-law potential spectrum: P_Φ(ℓ) = A × ℓ^n / (1 + (ℓ₀/ℓ)²)

    At high ℓ >> ℓ₀, this reduces to pure power law A × ℓ^n.
    At low ℓ << ℓ₀, the spectrum becomes A × ℓ₀^{-2} × ℓ^{n+2}, two powers
    shallower, ensuring convergence of Hankel integrals for VV ξ₊.

    Parameters
    ----------
    ell : array
        Multipole values
    amplitude : float
        Amplitude A
    index : float
        Spectral index n (typically negative, e.g. -5)
    ell_0 : float
        IR cutoff scale (default: 10). Below this, spectrum flattens.

    Returns
    -------
    P_phi : array
        Potential power spectrum
    """
    ell = np.asarray(ell)
    ell_safe = np.where(ell > 0, ell, 1.0)
    # Broken power law: flattens at low ℓ
    P_phi = amplitude * ell_safe ** index / (1 + (ell_0 / ell_safe) ** 2)
    return np.where(ell > 0, P_phi, 0.0)


def shear_power_spectrum(ell, amplitude, index, D=0.5, ell_0=DEFAULT_ELL_0):
    """
    Shear E-mode power spectrum: C^γγ_ℓ = D² ℓ⁴ P_Φ(ℓ)

    Parameters
    ----------
    ell : array
        Multipole values
    amplitude : float
        Potential amplitude A
    index : float
        Potential spectral index n
    D : float
        Shear amplitude factor (default 0.5)
    ell_0 : float
        IR cutoff scale (default: 10)

    Returns
    -------
    C_gg : array
        Shear power spectrum
    """
    return D**2 * np.asarray(ell)**4 * power_law_spectrum(ell, amplitude, index, ell_0)


def convergence_power_spectrum(ell, amplitude, index, B=0.5, ell_0=DEFAULT_ELL_0):
    """
    Convergence power spectrum: C^κκ_ℓ = B² ℓ⁴ P_Φ(ℓ)

    Parameters
    ----------
    ell : array
        Multipole values
    amplitude : float
        Potential amplitude A
    index : float
        Potential spectral index n
    B : float
        Convergence amplitude factor (default 0.5)
    ell_0 : float
        IR cutoff scale (default: 10)

    Returns
    -------
    C_kk : array
        Convergence power spectrum
    """
    return B**2 * np.asarray(ell)**4 * power_law_spectrum(ell, amplitude, index, ell_0)


def velocity_power_spectrum(ell, amplitude, index, C=1.0, ell_0=DEFAULT_ELL_0):
    """
    Velocity power spectrum: C^αα_ℓ = C² ℓ² P_Φ(ℓ)

    Parameters
    ----------
    ell : array
        Multipole values
    amplitude : float
        Potential amplitude A
    index : float
        Potential spectral index n
    C : float
        Velocity amplitude factor (default 1.0)
    ell_0 : float
        IR cutoff scale (default: 10)

    Returns
    -------
    C_vv : array
        Velocity power spectrum
    """
    return C**2 * np.asarray(ell)**2 * power_law_spectrum(ell, amplitude, index, ell_0)


def shear_velocity_cross_spectrum(ell, amplitude, index, C=1.0, D=0.5, ell_0=DEFAULT_ELL_0):
    """
    Shear-velocity cross power spectrum: C^αγ_ℓ = CD ℓ³ P_Φ(ℓ)

    Parameters
    ----------
    ell : array
        Multipole values
    amplitude : float
        Potential amplitude A
    index : float
        Potential spectral index n
    C : float
        Velocity amplitude factor (default 1.0)
    D : float
        Shear amplitude factor (default 0.5)
    ell_0 : float
        IR cutoff scale (default: 10)

    Returns
    -------
    C_gv : array
        Shear-velocity cross power spectrum
    """
    return C * D * np.asarray(ell)**3 * power_law_spectrum(ell, amplitude, index, ell_0)
