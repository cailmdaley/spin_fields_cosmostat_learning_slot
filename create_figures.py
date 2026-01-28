#!/usr/bin/env python3
"""
Publication-quality figures for the spin fields tutorial.

Figures:
1. fields.png — 4-panel field visualization (phi, alpha, kappa, gamma)
2. phi_with_velocity.png — Potential with rebinned velocity quiver overlay
3. correlation_functions.png — 3x2 grid of all correlations with theory
4. mcmc_fit.png — Best-fit comparison with residuals
5. mcmc_corner.png — Parameter posterior corner plot

Uses seaborn style, mako colormap, 150 DPI minimum.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from generate_fields_lib import (
    generate_phi_from_power_spectrum,
    generate_spin2_field,
    generate_scalar_field,
    generate_vector_field,
)
from measure_correlations import (
    measure_gg_correlation,
    measure_vv_correlation,
    measure_gv_correlation,
)
from theory_correlations import get_theory_correlations
from fit_power_law import (
    measure_cl_from_field,
    fit_power_law_cl,
    run_mcmc_cl,
)
from power_spectrum import shear_power_spectrum, velocity_power_spectrum, shear_velocity_cross_spectrum
from scipy.special import jv


# =============================================================================
# Configuration
# =============================================================================

NX, NY = 512, 512
BOX_SIZE_DEG = 50.0

# Amplitude factors (fixed, known to fitter)
B, C, D = 0.5, 1.0, 0.5

# True parameters
TRUE_AMPLITUDE = 1.0
TRUE_INDEX = -5.0

# MCMC settings
NWALKERS = 32
NSTEPS = 300
BURN_IN = 100

OUTPUT_DIR = Path("output_tutorial")

# Plotting style
sns.set_theme(style="ticks", context="paper", font_scale=1.1)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.family'] = 'serif'


# =============================================================================
# Cached theory calculator for fast MCMC
# =============================================================================

class CachedTheoryCalculator:
    """
    Pre-computes ell grid and Bessel functions for fast MCMC evaluation.

    The discrete 2D sum ξ(θ) = (1/L²) Σ_k C_ℓ(k) J_n(ℓ_k θ) is expensive
    because we must build the full ell grid each call. This class caches
    the grid and precomputes Bessel function values for fixed theta.
    """

    def __init__(self, theta_arcmin, box_size_deg, nx):
        self.theta_arcmin = np.asarray(theta_arcmin)
        self.theta_rad = np.deg2rad(self.theta_arcmin / 60.0)
        self.box_rad = np.deg2rad(box_size_deg)
        self.nx = nx

        # Build 2D k-space grid (cached)
        kx = 2 * np.pi * np.fft.fftfreq(nx, d=self.box_rad / nx)
        ky = 2 * np.pi * np.fft.fftfreq(nx, d=self.box_rad / nx)
        kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
        self.ell_grid = np.sqrt(kx_grid**2 + ky_grid**2)

        # Precompute Bessel functions for each theta and order
        self._bessel_cache = {}
        for order in [0, 2, 4]:  # Orders we need: 0 (ξ+), 4 (ξ-), 2 (VV ξ-)
            self._bessel_cache[order] = {}
            for i, th in enumerate(self.theta_rad):
                self._bessel_cache[order][i] = jv(order, self.ell_grid * th)

        self.norm = 1.0 / self.box_rad**2

    def compute_correlations(self, amplitude, index, C_coeff, D_coeff):
        """
        Compute GG and VV correlations for given power-law parameters.

        Returns dict with xi_plus, xi_minus, xi_vv_plus, xi_vv_minus.
        """
        # Compute power spectra on 2D grid
        # C_gg(ℓ) = D² ℓ⁴ C_φ(ℓ) = D² ℓ⁴ A ℓⁿ = D² A ℓ^(n+4)
        # C_vv(ℓ) = C² ℓ² C_φ(ℓ) = C² ℓ² A ℓⁿ = C² A ℓ^(n+2)
        C_gg = D_coeff**2 * amplitude * np.power(self.ell_grid, index + 4)
        C_vv = C_coeff**2 * amplitude * np.power(self.ell_grid, index + 2)

        # Zero DC mode
        C_gg[0, 0] = 0
        C_vv[0, 0] = 0

        n_theta = len(self.theta_rad)
        xi_plus = np.zeros(n_theta)
        xi_minus = np.zeros(n_theta)
        xi_vv_plus = np.zeros(n_theta)
        xi_vv_minus = np.zeros(n_theta)

        for i in range(n_theta):
            xi_plus[i] = np.sum(C_gg * self._bessel_cache[0][i]) * self.norm
            xi_minus[i] = np.sum(C_gg * self._bessel_cache[4][i]) * self.norm
            xi_vv_plus[i] = np.sum(C_vv * self._bessel_cache[0][i]) * self.norm
            xi_vv_minus[i] = np.sum(C_vv * self._bessel_cache[2][i]) * self.norm

        # Sign convention to match TreeCorr
        xi_vv_minus = -xi_vv_minus

        return {
            'xi_plus': xi_plus,
            'xi_minus': xi_minus,
            'xi_vv_plus': xi_vv_plus,
            'xi_vv_minus': xi_vv_minus,
        }


# Global cached calculator (initialized in main)
_theory_cache = None


def rebin_field(field, factor):
    """
    Rebin a 2D field by averaging over NxN blocks.

    Parameters
    ----------
    field : array
        2D array to rebin
    factor : int
        Rebinning factor (NxN blocks)

    Returns
    -------
    rebinned : array
        Rebinned array with shape (nx//factor, ny//factor)
    """
    nx, ny = field.shape
    new_nx, new_ny = nx // factor, ny // factor
    rebinned = field[:new_nx * factor, :new_ny * factor].reshape(
        new_nx, factor, new_ny, factor
    ).mean(axis=(1, 3))
    return rebinned


def create_fields_figure(phi, v1, v2, kappa, g1, g2, output_path):
    """
    Figure 1: 4-panel field visualization.

    Panel 1: phi (gravitational potential) - diverging colormap
    Panel 2: alpha (velocity) - colormap of |alpha|
    Panel 3: kappa (convergence) - diverging colormap
    Panel 4: gamma_1 (shear component) - diverging colormap
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    # Common extent in degrees
    extent = [0, BOX_SIZE_DEG, 0, BOX_SIZE_DEG]

    # Panel 1: Phi (potential) - diverging colormap
    ax = axes[0, 0]
    vmax = 3 * phi.std()
    im = ax.imshow(phi.T, origin='lower', extent=extent, cmap='vlag',
                   vmin=-vmax, vmax=vmax, aspect='equal')
    ax.set_xlabel('x [deg]')
    ax.set_ylabel('y [deg]')
    ax.set_title(r'$\Phi$ (gravitational potential)')
    plt.colorbar(im, ax=ax, label=r'$\Phi$', shrink=0.85)

    # Panel 2: |alpha| (velocity magnitude) - sequential
    ax = axes[0, 1]
    alpha_mag = np.sqrt(v1**2 + v2**2)
    im = ax.imshow(alpha_mag.T, origin='lower', extent=extent, cmap='mako',
                   aspect='equal')
    ax.set_xlabel('x [deg]')
    ax.set_ylabel('y [deg]')
    ax.set_title(r'$|\alpha|$ (velocity magnitude)')
    plt.colorbar(im, ax=ax, label=r'$|\alpha|$', shrink=0.85)

    # Panel 3: Kappa (convergence) - diverging colormap
    ax = axes[1, 0]
    vmax = 3 * kappa.std()
    im = ax.imshow(kappa.T, origin='lower', extent=extent, cmap='vlag',
                   vmin=-vmax, vmax=vmax, aspect='equal')
    ax.set_xlabel('x [deg]')
    ax.set_ylabel('y [deg]')
    ax.set_title(r'$\kappa$ (convergence)')
    plt.colorbar(im, ax=ax, label=r'$\kappa$', shrink=0.85)

    # Panel 4: |gamma| (shear magnitude) - sequential colormap
    ax = axes[1, 1]
    gamma_mag = np.sqrt(g1**2 + g2**2)
    im = ax.imshow(gamma_mag.T, origin='lower', extent=extent, cmap='mako',
                   aspect='equal')
    ax.set_xlabel('x [deg]')
    ax.set_ylabel('y [deg]')
    ax.set_title(r'$|\gamma|$ (shear magnitude)')
    plt.colorbar(im, ax=ax, label=r'$|\gamma|$', shrink=0.85)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def create_phi_velocity_figure(phi, v1, v2, output_path):
    """
    Figure 2: Potential with velocity vectors overlaid.

    Background: phi colormap
    Overlay: rebinned velocity quiver field
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    extent = [0, BOX_SIZE_DEG, 0, BOX_SIZE_DEG]

    # Background: phi - diverging colormap
    vmax = 3 * phi.std()
    im = ax.imshow(phi.T, origin='lower', extent=extent, cmap='vlag',
                   vmin=-vmax, vmax=vmax, aspect='equal')

    # Overlay: velocity vectors (rebinned)
    # Flip sign: v = -∇Φ (Newtonian convention, flow toward low potential)
    rebin_factor = 16
    v1_binned = -rebin_field(v1, rebin_factor)  # Flip sign
    v2_binned = -rebin_field(v2, rebin_factor)  # Flip sign

    # Coordinate grid for rebinned field
    nx_bin = NX // rebin_factor
    ny_bin = NY // rebin_factor
    x = (np.arange(nx_bin) + 0.5) * BOX_SIZE_DEG / nx_bin
    y = (np.arange(ny_bin) + 0.5) * BOX_SIZE_DEG / ny_bin
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # Normalize arrows
    v_mag = np.sqrt(v1_binned**2 + v2_binned**2)
    scale = v_mag.max() * 15

    ax.quiver(xx, yy, v1_binned, v2_binned, color='black', alpha=0.7,
              scale=scale, width=0.003, headwidth=3, headlength=4)

    ax.set_xlabel('x [deg]')
    ax.set_ylabel('y [deg]')
    ax.set_title(r'$\Phi$ (potential) with $\alpha$ (velocity) vectors')
    plt.colorbar(im, ax=ax, label=r'$\Phi$', shrink=0.85)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def create_correlation_figure(gg_result, vv_result, gv_result, theory, output_path):
    """
    Figure 3: All correlations in 3x2 grid.

    Layout:
            xi_+         xi_-
    GG   [measured]  [measured]
         + theory    + theory
    VV   [measured]  [measured]
         + theory    + theory
    GV   xi_t        xi_x
         + theory    + theory
    """
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))

    theta = theory['theta_arcmin']

    # Row 1: GG (shear-shear)
    ax = axes[0, 0]
    valid = gg_result['npairs'] > 0
    ax.errorbar(gg_result['theta'][valid], gg_result['xi_plus'][valid],
                yerr=gg_result['sigma_plus'][valid], fmt='o', ms=4, capsize=2,
                label='Measured', color='C0')
    ax.plot(theta, theory['xi_plus'], '-', color='C1', lw=1.5, label='Theory')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\theta$ [arcmin]')
    ax.set_ylabel(r'$\xi_+(\theta)$')
    ax.set_title(r'GG: $\xi_+$ (shear)')
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.errorbar(gg_result['theta'][valid], gg_result['xi_minus'][valid],
                yerr=gg_result['sigma_minus'][valid], fmt='o', ms=4, capsize=2,
                label='Measured', color='C0')
    ax.plot(theta, theory['xi_minus'], '-', color='C1', lw=1.5, label='Theory')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\theta$ [arcmin]')
    ax.set_ylabel(r'$\xi_-(\theta)$')
    ax.set_title(r'GG: $\xi_-$ (shear)')
    ax.legend(frameon=False)

    # Row 2: VV (velocity-velocity)
    ax = axes[1, 0]
    valid = vv_result['npairs'] > 0
    ax.errorbar(vv_result['theta'][valid], vv_result['xi_plus'][valid],
                yerr=vv_result['sigma_plus'][valid], fmt='o', ms=4, capsize=2,
                label='Measured', color='C0')
    ax.plot(theta, theory['xi_vv_plus'], '-', color='C1', lw=1.5, label='Theory')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\theta$ [arcmin]')
    ax.set_ylabel(r'$\xi_+(\theta)$')
    ax.set_title(r'VV: $\xi_+$ (velocity)')
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.errorbar(vv_result['theta'][valid], vv_result['xi_minus'][valid],
                yerr=vv_result['sigma_minus'][valid], fmt='o', ms=4, capsize=2,
                label='Measured', color='C0')
    ax.plot(theta, theory['xi_vv_minus'], '-', color='C1', lw=1.5, label='Theory')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\theta$ [arcmin]')
    ax.set_ylabel(r'$\xi_-(\theta)$')
    ax.set_title(r'VV: $\xi_-$ (velocity)')
    ax.legend(frameon=False)

    # Row 3: GV (shear-velocity cross)
    ax = axes[2, 0]
    valid = gv_result['npairs'] > 0
    ax.errorbar(gv_result['theta'][valid], gv_result['xi_plus'][valid],
                yerr=gv_result['sigma_plus'][valid], fmt='o', ms=4, capsize=2,
                label='Measured', color='C0')
    ax.plot(theta, theory['xi_gv_t'], '-', color='C1', lw=1.5, label='Theory')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\theta$ [arcmin]')
    ax.set_ylabel(r'$\xi_t(\theta)$')
    ax.set_title(r'GV: $\xi_t$ (tangential)')
    ax.legend(frameon=False)

    ax = axes[2, 1]
    ax.errorbar(gv_result['theta'][valid], gv_result['xi_minus'][valid],
                yerr=gv_result['sigma_minus'][valid], fmt='o', ms=4, capsize=2,
                label='Measured', color='C0')
    ax.plot(theta, theory['xi_gv_x'], '-', color='C1', lw=1.5, label='Theory')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\theta$ [arcmin]')
    ax.set_ylabel(r'$\xi_\times(\theta)$')
    ax.set_title(r'GV: $\xi_\times$ (cross)')
    ax.legend(frameon=False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def create_mcmc_fit_figure(gg_result, vv_result, gv_result, sampler, output_path):
    """
    Figure 4: Best-fit comparison with residuals.

    Shows data with error bars and best-fit theory curves.
    3x2 layout matching correlation_functions.png with residual panels.
    """
    flat_samples = sampler.get_chain(discard=BURN_IN, flat=True)
    amp_fit, idx_fit = np.median(flat_samples, axis=0)

    # Get theory at best-fit parameters
    theta_fine = np.geomspace(2, 200, 100)
    theory_fit = get_theory_correlations(amp_fit, idx_fit, theta_fine, BOX_SIZE_DEG, NX, B, C, D)

    fig = plt.figure(figsize=(12, 14))

    # Create gridspec for main + residual panels
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(6, 2, figure=fig, height_ratios=[3, 1, 3, 1, 3, 1], hspace=0.08, wspace=0.25)

    correlations = [
        ('GG', 'xi_plus', 'xi_minus', gg_result, 'xi_plus', 'xi_minus', 'sigma_plus', 'sigma_minus'),
        ('VV', 'xi_vv_plus', 'xi_vv_minus', vv_result, 'xi_plus', 'xi_minus', 'sigma_plus', 'sigma_minus'),
        ('GV', 'xi_gv_t', 'xi_gv_x', gv_result, 'xi_plus', 'xi_minus', 'sigma_plus', 'sigma_minus'),
    ]

    labels_plus = [r'$\xi_+$', r'$\xi_+$', r'$\xi_t$']
    labels_minus = [r'$\xi_-$', r'$\xi_-$', r'$\xi_\times$']

    for row, (name, th_key_plus, th_key_minus, result, data_key_plus, data_key_minus,
              err_key_plus, err_key_minus) in enumerate(correlations):

        valid = result['npairs'] > 0
        theta_data = result['theta'][valid]

        # Theory at data points for residuals
        theory_at_data = get_theory_correlations(amp_fit, idx_fit, theta_data, BOX_SIZE_DEG, NX, B, C, D)

        # Left column: xi_+
        ax_main = fig.add_subplot(gs[row*2, 0])
        ax_res = fig.add_subplot(gs[row*2 + 1, 0], sharex=ax_main)

        data = result[data_key_plus][valid]
        err = result[err_key_plus][valid]

        ax_main.errorbar(theta_data, data, yerr=err, fmt='o', ms=4, capsize=2, color='C0')
        ax_main.plot(theta_fine, theory_fit[th_key_plus], '-', color='C1', lw=1.5)
        ax_main.set_xscale('log')
        ax_main.set_ylabel(labels_plus[row])
        ax_main.set_title(f'{name}: {labels_plus[row]}')
        plt.setp(ax_main.get_xticklabels(), visible=False)

        # Residuals
        residuals = (data - theory_at_data[th_key_plus]) / err
        ax_res.errorbar(theta_data, residuals, yerr=1, fmt='o', ms=3, color='C0')
        ax_res.axhline(0, color='gray', ls='--', lw=0.8)
        ax_res.set_xscale('log')
        ax_res.set_ylabel(r'$\Delta/\sigma$')
        ax_res.set_ylim(-4, 4)
        if row == 2:
            ax_res.set_xlabel(r'$\theta$ [arcmin]')

        # Right column: xi_-
        ax_main = fig.add_subplot(gs[row*2, 1])
        ax_res = fig.add_subplot(gs[row*2 + 1, 1], sharex=ax_main)

        data = result[data_key_minus][valid]
        err = result[err_key_minus][valid]

        ax_main.errorbar(theta_data, data, yerr=err, fmt='o', ms=4, capsize=2, color='C0')
        ax_main.plot(theta_fine, theory_fit[th_key_minus], '-', color='C1', lw=1.5)
        ax_main.set_xscale('log')
        ax_main.set_ylabel(labels_minus[row])
        ax_main.set_title(f'{name}: {labels_minus[row]}')
        plt.setp(ax_main.get_xticklabels(), visible=False)

        # Residuals
        residuals = (data - theory_at_data[th_key_minus]) / err
        ax_res.errorbar(theta_data, residuals, yerr=1, fmt='o', ms=3, color='C0')
        ax_res.axhline(0, color='gray', ls='--', lw=0.8)
        ax_res.set_xscale('log')
        ax_res.set_ylabel(r'$\Delta/\sigma$')
        ax_res.set_ylim(-4, 4)
        if row == 2:
            ax_res.set_xlabel(r'$\theta$ [arcmin]')

    # Add best-fit annotation
    fig.suptitle(f'Best fit: $A = {amp_fit:.2e}$, $n = {idx_fit:.2f}$', fontsize=11, y=1.01)

    fig.subplots_adjust(top=0.95, hspace=0.08, wspace=0.25)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def create_corner_figure(sampler, output_path):
    """
    Figure 5: Corner plot of MCMC posteriors.

    Shows 1sigma/2sigma contours for amplitude A and index n.
    Marks true values.
    """
    import corner

    flat_samples = sampler.get_chain(discard=BURN_IN, flat=True)

    fig = corner.corner(
        flat_samples,
        labels=[r'$A$', r'$n$'],
        truths=[TRUE_AMPLITUDE, TRUE_INDEX],
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt='.3e',
        levels=(1 - np.exp(-0.5), 1 - np.exp(-2)),  # 1sigma, 2sigma
        plot_datapoints=True,
        plot_density=True,
        fill_contours=True,
        color='C0',
        truth_color='C3',
    )

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def get_theory_correlations_mcmc(amplitude, index, theta_arcmin):
    """
    Theory correlations using discrete 2D sum (for MCMC).

    Uses the same discrete 2D sum as get_theory_correlations() for consistency.
    This is slower but matches the measured correlations properly.
    """
    return get_theory_correlations(amplitude, index, theta_arcmin, BOX_SIZE_DEG, NX, B, C, D)


def log_likelihood_joint(params, gg_result, vv_result, gv_result):
    """
    Joint log-likelihood for GG + VV correlations.

    Uses discrete 2D sum for theory (consistent with get_theory_correlations).
    GV is excluded because it is noise-dominated for single realizations.
    """
    amplitude, index = params
    if amplitude <= 0:
        return -np.inf

    chi2_total = 0.0

    # Get all valid theta values (assume same binning)
    valid = gg_result['npairs'] > 0
    theta = gg_result['theta'][valid]

    # Theory using discrete 2D sum (consistent with measurements)
    theory = get_theory_correlations_mcmc(amplitude, index, theta)

    # GG correlation
    for data_key, th_key, err_key in [
        ('xi_plus', 'xi_plus', 'sigma_plus'),
        ('xi_minus', 'xi_minus', 'sigma_minus'),
    ]:
        data = gg_result[data_key][valid]
        err = gg_result[err_key][valid]
        chi2_total += np.sum(((data - theory[th_key]) / err) ** 2)

    # VV correlation
    valid_vv = vv_result['npairs'] > 0
    theta_vv = vv_result['theta'][valid_vv]
    theory_vv = get_theory_correlations_mcmc(amplitude, index, theta_vv)

    for data_key, th_key, err_key in [
        ('xi_plus', 'xi_vv_plus', 'sigma_plus'),
        ('xi_minus', 'xi_vv_minus', 'sigma_minus'),
    ]:
        data = vv_result[data_key][valid_vv]
        err = vv_result[err_key][valid_vv]
        chi2_total += np.sum(((data - theory_vv[th_key]) / err) ** 2)

    # GV excluded - noise-dominated for single realizations

    return -0.5 * chi2_total


def log_prior_joint(params):
    """Flat priors on amplitude and index."""
    amplitude, index = params
    if 1e-10 < amplitude < 1e10 and -10 < index < 2:
        return 0.0
    return -np.inf


def log_posterior_joint(params, gg_result, vv_result, gv_result):
    """Log-posterior for joint MCMC."""
    lp = log_prior_joint(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_joint(params, gg_result, vv_result, gv_result)


def run_mcmc_joint(gg_result, vv_result, gv_result, nwalkers=32, nsteps=300,
                   initial_guess=(1.0, -5.0), progress=True):
    """
    Run joint MCMC fitting GG + VV + GV correlations.
    """
    import emcee

    ndim = 2
    pos = np.array(initial_guess) + 0.01 * np.random.randn(nwalkers, ndim)
    pos[:, 0] = np.abs(pos[:, 0])

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior_joint,
        args=(gg_result, vv_result, gv_result),
    )
    sampler.run_mcmc(pos, nsteps, progress=progress)
    return sampler


def main():
    """Generate all tutorial figures."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Creating Tutorial Figures")
    print("=" * 60)

    # =========================================================================
    # Generate fields
    # =========================================================================
    print("\n[1/6] Generating fields...")

    power_config = {
        "type": "power_law",
        "amplitude": TRUE_AMPLITUDE,
        "index": TRUE_INDEX,
    }

    phi_real, phi_fourier = generate_phi_from_power_spectrum(
        NX, NY, BOX_SIZE_DEG, power_config, seed=42
    )

    g1, g2 = generate_spin2_field(phi_fourier, D, NX, NY, BOX_SIZE_DEG)
    kappa = generate_scalar_field(phi_fourier, B, NX, NY, BOX_SIZE_DEG)
    v1, v2 = generate_vector_field(phi_fourier, C, NX, NY, BOX_SIZE_DEG)

    print(f"  Fields generated: phi, kappa, gamma, alpha")

    # =========================================================================
    # Measure correlations
    # =========================================================================
    print("\n[2/6] Measuring correlations...")

    gg_result = measure_gg_correlation(g1, g2, BOX_SIZE_DEG, nbins=15, min_sep=2.0, max_sep=200.0)
    vv_result = measure_vv_correlation(v1, v2, BOX_SIZE_DEG, nbins=15, min_sep=2.0, max_sep=200.0)
    gv_result = measure_gv_correlation(g1, g2, v1, v2, BOX_SIZE_DEG, nbins=15, min_sep=2.0, max_sep=200.0)

    print(f"  Measured: GG, VV, GV correlations")

    # =========================================================================
    # Compute theory
    # =========================================================================
    print("\n[3/4] Computing theory predictions...")

    valid = gg_result['npairs'] > 0
    theta = gg_result['theta'][valid]
    theory = get_theory_correlations(TRUE_AMPLITUDE, TRUE_INDEX, theta, BOX_SIZE_DEG, NX, B, C, D)

    # =========================================================================
    # Run MCMC
    # =========================================================================
    print("\n[4/6] Running MCMC (GG + VV joint fit)...")

    sampler = run_mcmc_joint(gg_result, vv_result, gv_result, nwalkers=NWALKERS,
                             nsteps=NSTEPS, initial_guess=(TRUE_AMPLITUDE, TRUE_INDEX),
                             progress=True)

    flat_samples = sampler.get_chain(discard=BURN_IN, flat=True)
    amp_fit, idx_fit = np.median(flat_samples, axis=0)
    print(f"  Best fit: A = {amp_fit:.3e}, n = {idx_fit:.2f}")

    # =========================================================================
    # Create figures
    # =========================================================================
    print("\n[5/6] Creating field figures...")

    create_fields_figure(phi_real, v1, v2, kappa, g1, g2, OUTPUT_DIR / "fields.png")
    create_phi_velocity_figure(phi_real, v1, v2, OUTPUT_DIR / "phi_with_velocity.png")

    print("\n[6/6] Creating correlation and MCMC figures...")

    create_correlation_figure(gg_result, vv_result, gv_result, theory, OUTPUT_DIR / "correlation_functions.png")
    create_mcmc_fit_figure(gg_result, vv_result, gv_result, sampler, OUTPUT_DIR / "mcmc_fit.png")
    create_corner_figure(sampler, OUTPUT_DIR / "mcmc_corner.png")

    print("\nComplete!")
    print("=" * 60)
    print(f"All figures saved to {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
