"""
Cosmology module for computing LCDM lensing power spectra using CAMB.

The key parameter is S8 = σ₈ × √(Ωₘ/0.3), which controls the amplitude
of the matter power spectrum relevant for weak lensing.
"""

import numpy as np

try:
    import camb
    from camb import model
    HAS_CAMB = True
except ImportError:
    HAS_CAMB = False
    print("Warning: CAMB not installed. Install with: pip install camb")


def get_lcdm_lensing_power_spectrum(ell_grid, config):
    """
    Compute the lensing convergence power spectrum C_ℓ^κκ using CAMB.

    Parameters:
    -----------
    ell_grid : numpy.ndarray
        2D grid of ℓ values (multipoles)
    config : dict
        Cosmology configuration with keys:
        - S8: σ₈ × √(Ωₘ/0.3), controls amplitude
        - Omega_m: matter density parameter (default: 0.3)
        - Omega_b: baryon density parameter (default: 0.05)
        - h: Hubble parameter H₀/100 (default: 0.7)
        - n_s: scalar spectral index (default: 0.96)
        - z_source: source redshift for lensing (default: 1.0)

    Returns:
    --------
    C_ell : numpy.ndarray
        Lensing convergence power spectrum on the ell_grid
    """
    if not HAS_CAMB:
        raise ImportError("CAMB is required for LCDM power spectra")

    # Extract cosmological parameters with defaults
    S8 = config.get('S8', 0.8)
    Omega_m = config.get('Omega_m', 0.3)
    Omega_b = config.get('Omega_b', 0.05)
    h = config.get('h', 0.7)
    n_s = config.get('n_s', 0.96)
    z_source = config.get('z_source', 1.0)

    # Compute σ₈ from S8: S8 = σ₈ × √(Ωₘ/0.3)
    sigma8 = S8 / np.sqrt(Omega_m / 0.3)

    # Set up CAMB parameters
    # We'll compute with a reference σ₈ and then rescale
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=h * 100,
        ombh2=Omega_b * h**2,
        omch2=(Omega_m - Omega_b) * h**2,
        mnu=0.06,  # Minimal neutrino mass
        omk=0,
        tau=0.06
    )
    pars.InitPower.set_params(As=2e-9, ns=n_s)  # Initial guess for As

    # First run to get reference σ₈
    pars.set_matter_power(redshifts=[0.0], kmax=10.0)
    results = camb.get_results(pars)
    sigma8_ref = results.get_sigma8_0()

    # Rescale As to get desired σ₈
    # σ₈² ∝ As, so As_new = As_old × (σ₈_target / σ₈_ref)²
    As_rescaled = 2e-9 * (sigma8 / sigma8_ref)**2
    pars.InitPower.set_params(As=As_rescaled, ns=n_s)

    # Set up for lensing power spectrum
    # Use source distribution peaked at z_source
    pars.set_matter_power(redshifts=[0.0, z_source], kmax=10.0)
    pars.SourceWindows = [camb.sources.GaussianSourceWindow(redshift=z_source, source_type='lensing', sigma=0.05)]
    pars.Want_CMB = False

    # Get results
    results = camb.get_results(pars)

    # Get the lensing potential power spectrum
    # CAMB returns C_ℓ^ϕϕ for the lensing potential
    # The convergence κ = -∇²ϕ/2, so C_ℓ^κκ = [ℓ(ℓ+1)]²/4 × C_ℓ^ϕϕ
    cls = results.get_source_cls_dict()

    # Get ℓ range from CAMB
    ell_camb = np.arange(2, len(cls['W1xW1']) + 2)
    cl_kappa_camb = cls['W1xW1']  # This is already C_ℓ^κκ for the source window

    # Interpolate to our ℓ grid
    from scipy.interpolate import interp1d

    # Flatten ell_grid for interpolation
    ell_flat = ell_grid.flatten()

    # Find valid range where C_ℓ > 0
    valid = cl_kappa_camb > 1e-20
    ell_valid = ell_camb[valid]
    cl_valid = cl_kappa_camb[valid]

    if len(ell_valid) == 0:
        raise ValueError("CAMB returned no valid C_ℓ values")

    # Interpolate in log-log space for better accuracy
    log_interp = interp1d(
        np.log(ell_valid),
        np.log(cl_valid),
        kind='linear',
        bounds_error=False,
        fill_value=(np.log(cl_valid[0]), np.log(cl_valid[-1]))
    )

    # Clip ell values to valid range for interpolation
    ell_for_interp = np.clip(ell_flat, ell_valid.min(), ell_valid.max())
    cl_interp = np.exp(log_interp(np.log(ell_for_interp)))

    # For ℓ outside CAMB range, use power-law extrapolation
    # High ℓ: C_ℓ ~ ℓ^(-2) approximately
    high_ell_mask = ell_flat > ell_valid.max()
    if np.any(high_ell_mask):
        cl_interp[high_ell_mask] = cl_valid[-1] * (ell_flat[high_ell_mask] / ell_valid.max())**(-2)

    # Low ℓ: use constant extrapolation (already handled by fill_value)
    low_ell_mask = ell_flat < ell_valid.min()
    cl_interp[low_ell_mask] = cl_valid[0]

    # Reshape back to grid
    C_ell = cl_interp.reshape(ell_grid.shape)

    # Set ℓ=0 to 0 (no monopole)
    C_ell[ell_grid < 2] = 0

    return C_ell


def get_matter_power_spectrum(k_grid, config, z=0.0):
    """
    Compute the matter power spectrum P(k) using CAMB.

    Parameters:
    -----------
    k_grid : numpy.ndarray
        Grid of k values [h/Mpc]
    config : dict
        Cosmology configuration (same as get_lcdm_lensing_power_spectrum)
    z : float
        Redshift at which to evaluate P(k)

    Returns:
    --------
    P_k : numpy.ndarray
        Matter power spectrum [(Mpc/h)³]
    """
    if not HAS_CAMB:
        raise ImportError("CAMB is required for LCDM power spectra")

    # Extract parameters
    S8 = config.get('S8', 0.8)
    Omega_m = config.get('Omega_m', 0.3)
    Omega_b = config.get('Omega_b', 0.05)
    h = config.get('h', 0.7)
    n_s = config.get('n_s', 0.96)

    sigma8 = S8 / np.sqrt(Omega_m / 0.3)

    # Set up CAMB
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=h * 100,
        ombh2=Omega_b * h**2,
        omch2=(Omega_m - Omega_b) * h**2,
        mnu=0.06,
        omk=0,
        tau=0.06
    )
    pars.InitPower.set_params(As=2e-9, ns=n_s)

    # Get reference σ₈ and rescale
    pars.set_matter_power(redshifts=[z], kmax=k_grid.max() * 1.1)
    results = camb.get_results(pars)
    sigma8_ref = results.get_sigma8_0()

    As_rescaled = 2e-9 * (sigma8 / sigma8_ref)**2
    pars.InitPower.set_params(As=As_rescaled, ns=n_s)

    # Recompute with correct amplitude
    results = camb.get_results(pars)

    # Get P(k) interpolator
    kh, z_out, pk = results.get_matter_power_spectrum(
        minkh=k_grid.min() * 0.9,
        maxkh=k_grid.max() * 1.1,
        npoints=500
    )

    # Interpolate to our k grid
    from scipy.interpolate import interp1d
    pk_interp = interp1d(kh, pk[0], kind='cubic', fill_value='extrapolate')

    return pk_interp(k_grid)
