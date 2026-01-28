"""
Fit power-law parameters (A, n) from measured power spectra.

Uses scipy.optimize.minimize for quick fits, with optional MCMC via emcee.
"""

import numpy as np
from scipy.optimize import minimize
from power_spectrum import shear_power_spectrum


# =============================================================================
# C_ℓ-based fitting (preferred approach)
# =============================================================================


def measure_cl_from_field(g1, g2, box_size_deg, n_bins=15, ell_min=20, ell_max=1000):
    """
    Measure shear power spectrum C_ℓ from field.

    Parameters
    ----------
    g1, g2 : array
        Shear components on 2D grid
    box_size_deg : float
        Box size in degrees
    n_bins : int
        Number of ℓ bins
    ell_min, ell_max : float
        ℓ range for binning

    Returns
    -------
    dict with keys:
        'ell': bin centers
        'cl': measured C_ℓ
        'cl_err': error estimate (from scatter in bin)
        'nmodes': number of modes per bin
    """
    nx, ny = g1.shape
    box_rad = np.deg2rad(box_size_deg)

    # FFT the shear field
    gamma = g1 + 1j * g2
    gamma_fft = np.fft.fft2(gamma)

    # Power spectrum with correct normalization
    # E[|gamma_fft[k]|^2] = C_gg[k] * N^4 / L^2
    power_2d = np.abs(gamma_fft) ** 2 * box_rad ** 2 / (nx * ny) ** 2

    # Create ℓ grid
    kx = np.fft.fftfreq(nx, d=box_rad / nx)
    ky = np.fft.fftfreq(ny, d=box_rad / ny)
    kxx, kyy = np.meshgrid(kx, ky, indexing='ij')
    ell = 2 * np.pi * np.sqrt(kxx ** 2 + kyy ** 2)

    # Bin by ℓ
    ell_bins = np.geomspace(ell_min, ell_max, n_bins + 1)
    ell_centers = np.sqrt(ell_bins[:-1] * ell_bins[1:])

    cl_binned, cl_err, nmodes = [], [], []
    for i in range(n_bins):
        mask = (ell >= ell_bins[i]) & (ell < ell_bins[i + 1])
        n = mask.sum()
        if n > 0:
            values = power_2d[mask]
            cl_binned.append(values.mean())
            cl_err.append(values.std() / np.sqrt(n))  # Standard error
            nmodes.append(n)
        else:
            cl_binned.append(np.nan)
            cl_err.append(np.nan)
            nmodes.append(0)

    return {
        'ell': ell_centers,
        'cl': np.array(cl_binned),
        'cl_err': np.array(cl_err),
        'nmodes': np.array(nmodes),
    }


def measure_eb_power_spectra(g1, g2, box_size_deg, n_bins=15, ell_min=20, ell_max=1000):
    """
    Measure E and B mode power spectra from shear field.

    For shear γ = γ₁ + iγ₂, the E/B decomposition in Fourier space is:
        γ̃(ℓ) = (Ẽ + iB̃) e^{2iφ_ℓ}

    where Ẽ and B̃ are the E and B mode amplitudes. For a pure E-mode field
    derived from a scalar potential, B̃ should be zero (noise-dominated).

    Parameters
    ----------
    g1, g2 : array
        Shear components on 2D grid
    box_size_deg : float
        Box size in degrees
    n_bins : int
        Number of ℓ bins
    ell_min, ell_max : float
        ℓ range for binning

    Returns
    -------
    dict with keys:
        'ell': bin centers
        'cl_ee': E-mode power spectrum
        'cl_bb': B-mode power spectrum
        'cl_ee_err', 'cl_bb_err': error estimates
        'nmodes': number of modes per bin
    """
    nx, ny = g1.shape
    box_rad = np.deg2rad(box_size_deg)

    # Create ℓ grid
    kx = np.fft.fftfreq(nx, d=box_rad / nx)
    ky = np.fft.fftfreq(ny, d=box_rad / ny)
    kxx, kyy = np.meshgrid(kx, ky, indexing='ij')
    ell = 2 * np.pi * np.sqrt(kxx ** 2 + kyy ** 2)
    phi_ell = np.arctan2(kyy, kxx)

    # FFT the shear field
    gamma = g1 + 1j * g2
    gamma_fft = np.fft.fft2(gamma)

    # E/B decomposition: γ̃ = (Ẽ + iB̃) e^{2iφ}
    # So (Ẽ + iB̃) = γ̃ e^{-2iφ}
    eb_fft = gamma_fft * np.exp(-2j * phi_ell)
    E_fft = eb_fft.real
    B_fft = eb_fft.imag

    # Power spectra with correct normalization
    norm = box_rad ** 2 / (nx * ny) ** 2
    power_ee = E_fft ** 2 * norm
    power_bb = B_fft ** 2 * norm

    # Bin by ℓ
    ell_bins = np.geomspace(ell_min, ell_max, n_bins + 1)
    ell_centers = np.sqrt(ell_bins[:-1] * ell_bins[1:])

    cl_ee, cl_bb, cl_ee_err, cl_bb_err, nmodes = [], [], [], [], []
    for i in range(n_bins):
        mask = (ell >= ell_bins[i]) & (ell < ell_bins[i + 1])
        n = mask.sum()
        if n > 0:
            ee_vals, bb_vals = power_ee[mask], power_bb[mask]
            cl_ee.append(ee_vals.mean())
            cl_bb.append(bb_vals.mean())
            cl_ee_err.append(ee_vals.std() / np.sqrt(n))
            cl_bb_err.append(bb_vals.std() / np.sqrt(n))
            nmodes.append(n)
        else:
            cl_ee.append(np.nan)
            cl_bb.append(np.nan)
            cl_ee_err.append(np.nan)
            cl_bb_err.append(np.nan)
            nmodes.append(0)

    return {
        'ell': ell_centers,
        'cl_ee': np.array(cl_ee),
        'cl_bb': np.array(cl_bb),
        'cl_ee_err': np.array(cl_ee_err),
        'cl_bb_err': np.array(cl_bb_err),
        'nmodes': np.array(nmodes),
    }


def chi2_cl(params, ell, cl_measured, cl_err, D=0.5, ell_0=10.0):
    """Chi-squared for C_ℓ fitting."""
    amplitude, index = params
    if amplitude <= 0:
        return 1e10
    cl_theory = shear_power_spectrum(ell, amplitude, index, D, ell_0)
    return np.nansum(((cl_measured - cl_theory) / cl_err) ** 2)


def fit_power_law_cl(ell, cl_measured, cl_err, initial_guess=(1.0, -5.0), D=0.5, ell_0=10.0):
    """
    Fit power-law parameters from measured C_ℓ.

    Parameters
    ----------
    ell : array
        Multipole bin centers
    cl_measured : array
        Measured C_ℓ values
    cl_err : array
        Uncertainties on C_ℓ
    initial_guess : tuple
        Initial (amplitude, index)
    D : float
        Shear amplitude factor
    ell_0 : float
        IR cutoff scale (fixed, not fitted)

    Returns
    -------
    dict with fit results
    """
    result = minimize(
        chi2_cl,
        initial_guess,
        args=(ell, cl_measured, cl_err, D, ell_0),
        method='Nelder-Mead',
        options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8},
    )
    amplitude, index = result.x
    valid = ~np.isnan(cl_measured)
    return {
        "amplitude": amplitude,
        "index": index,
        "chi2": result.fun,
        "ndof": valid.sum() - 2,
        "success": result.success,
    }


def log_likelihood_cl(params, ell, cl_measured, cl_err, D=0.5, ell_0=10.0):
    """Log-likelihood for C_ℓ MCMC sampling."""
    amplitude, index = params
    if amplitude <= 0:
        return -np.inf
    cl_theory = shear_power_spectrum(ell, amplitude, index, D, ell_0)
    chi2_val = np.nansum(((cl_measured - cl_theory) / cl_err) ** 2)
    return -0.5 * chi2_val


def log_prior_cl(params):
    """Flat priors on amplitude and index."""
    amplitude, index = params
    if 1e-10 < amplitude < 1e10 and -10 < index < 2:
        return 0.0
    return -np.inf


def log_posterior_cl(params, ell, cl_measured, cl_err, D=0.5, ell_0=10.0):
    """Log-posterior for C_ℓ MCMC."""
    lp = log_prior_cl(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_cl(params, ell, cl_measured, cl_err, D, ell_0)


def run_mcmc_cl(ell, cl_measured, cl_err, nwalkers=32, nsteps=200,
                initial_guess=(1.0, -5.0), D=0.5, ell_0=10.0, progress=True):
    """
    Run MCMC to sample power-law parameters from C_ℓ.

    Much faster than ξ(θ)-based fitting since no Hankel transform needed.
    """
    import emcee

    ndim = 2
    pos = np.array(initial_guess) + 0.01 * np.random.randn(nwalkers, ndim)
    pos[:, 0] = np.abs(pos[:, 0])

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior_cl,
        args=(ell, cl_measured, cl_err, D, ell_0),
    )
    sampler.run_mcmc(pos, nsteps, progress=progress)
    return sampler


def plot_fit_comparison_cl(ell, cl_measured, cl_err, sampler, truths=None,
                           discard=50, D=0.5, ell_0=10.0, output_path=None):
    """
    Plot measured vs theory C_ℓ with fit.
    """
    import matplotlib.pyplot as plt

    flat_samples = sampler.get_chain(discard=discard, flat=True)
    amp_fit, idx_fit = np.median(flat_samples, axis=0)

    ell_fine = np.geomspace(ell.min(), ell.max(), 100)
    cl_fit = shear_power_spectrum(ell_fine, amp_fit, idx_fit, D, ell_0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[3, 1], sharex=True)

    ax1.errorbar(ell, cl_measured, yerr=cl_err, fmt='o', label='Measured', ms=5)
    ax1.plot(ell_fine, cl_fit, 'C1-', label=f'Fit: A={amp_fit:.2e}, n={idx_fit:.2f}')

    if truths:
        cl_true = shear_power_spectrum(ell_fine, truths[0], truths[1], D, ell_0)
        ax1.plot(ell_fine, cl_true, 'C2--', alpha=0.7, label='Truth')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_ylabel(r"$C_\ell^{\gamma\gamma}$")
    ax1.legend()
    ax1.set_title("Shear Power Spectrum")

    cl_fit_at_data = shear_power_spectrum(ell, amp_fit, idx_fit, D, ell_0)
    residuals = (cl_measured - cl_fit_at_data) / cl_err
    ax2.errorbar(ell, residuals, yerr=1, fmt='o', ms=5)
    ax2.axhline(0, color='k', ls='--', lw=0.8)
    ax2.set_xlabel(r"Multipole $\ell$")
    ax2.set_ylabel(r"$(data - fit)/\sigma$")
    ax2.set_ylim(-4, 4)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_corner(sampler, truths=None, discard=100, labels=None, output_path=None):
    """
    Create corner plot from MCMC samples.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
    truths : list, optional
        True values [amplitude, index]
    discard : int
        Burn-in steps to discard
    labels : list, optional
        Parameter labels
    output_path : str or Path, optional
        Save path

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import corner

    flat_samples = sampler.get_chain(discard=discard, flat=True)
    labels = labels or ["$A$", "$n$"]

    fig = corner.corner(
        flat_samples,
        labels=labels,
        truths=truths,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3e",
    )
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
