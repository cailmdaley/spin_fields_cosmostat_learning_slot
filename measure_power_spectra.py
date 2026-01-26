"""
Measure power spectra from fields using FFT-based methods.

Key concepts:
- E/B decomposition for spin-s fields uses Hermitian symmetry
- For field f̃(ℓ) = [Ẽ(ℓ) + iB̃(ℓ)] × e^{isφ_ℓ}
- Extract: D(ℓ) = f̃(ℓ) × e^{-isφ_ℓ}
- Then: Ẽ(ℓ) = [D(ℓ) + D*(-ℓ)] / 2
         B̃(ℓ) = [D(ℓ) - D*(-ℓ)] / (2i)
"""

import numpy as np


def hermitian_conjugate_flip(arr):
    """
    Get D*(-ℓ) from D(ℓ) for proper E/B decomposition.

    For FFT arrays with DC at (0,0), the mode at -ℓ = (-kx, -ky)
    for mode at index (i, j) is at index (-i mod nx, -j mod ny).
    """
    nx, ny = arr.shape
    # Create index arrays for -ℓ mapping
    i_flip = np.arange(nx)
    j_flip = np.arange(ny)
    i_minus = (-i_flip) % nx
    j_minus = (-j_flip) % ny
    # Advanced indexing to get D(-ℓ) and then conjugate
    return np.conj(arr[i_minus[:, None], j_minus[None, :]])


def compute_power_spectrum_2d(field1, field2, box_size_deg, config):
    """
    Compute 2D power spectrum for scalar fields using FFT.

    Parameters:
    -----------
    field1, field2 : numpy.ndarray
        Fields in real space
    box_size_deg : float
        Box size in degrees
    config : dict
        Configuration with binning parameters

    Returns:
    --------
    ell_centers : numpy.ndarray
        Centers of ell bins
    power_spectrum : numpy.ndarray
        Binned power spectrum
    """
    ny, nx = field1.shape

    # FFT of fields
    fft1 = np.fft.fft2(field1)
    fft2 = np.fft.fft2(field2)

    # Cross power spectrum in 2D
    power_2d = (fft1 * np.conj(fft2)).real

    # Normalize
    box_size_rad = box_size_deg * np.pi / 180.0
    dx = box_size_rad / nx
    dy = box_size_rad / ny
    power_2d *= (dx * dy)**2 / (nx * ny)

    # Create 2D ell grid
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
    ell_2d = np.sqrt(kx_grid**2 + ky_grid**2)

    # Binning
    ell_min = config.get('ell_min', 10)
    ell_max = config.get('ell_max', None)
    if ell_max is None:
        ell_max = np.sqrt(2) * np.pi * min(nx, ny) / (2 * box_size_rad)
    n_bins = config.get('n_bins', 20)
    use_log_bins = config.get('use_log_bins', True)

    if use_log_bins:
        bin_edges = np.geomspace(max(ell_min, 1), ell_max, n_bins + 1)
    else:
        bin_edges = np.linspace(ell_min, ell_max, n_bins + 1)

    # Bin the power spectrum
    power_binned = np.zeros(n_bins)
    ell_centers = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (ell_2d >= bin_edges[i]) & (ell_2d < bin_edges[i+1])
        if np.sum(mask) > 0:
            power_binned[i] = np.mean(power_2d[mask])
            ell_centers[i] = np.mean(ell_2d[mask])
        else:
            ell_centers[i] = (bin_edges[i] + bin_edges[i+1]) / 2

    return ell_centers, power_binned


def compute_spin_power_auto(field_real, box_size_deg, config, spin):
    """
    Compute auto power spectrum for spin-weighted fields with E/B decomposition.

    Uses proper Hermitian symmetry for E/B separation:
    - D(ℓ) = f̃(ℓ) × e^{-isφ_ℓ}
    - Ẽ(ℓ) = [D(ℓ) + D*(-ℓ)] / 2
    - B̃(ℓ) = [D(ℓ) - D*(-ℓ)] / (2i)

    Parameters:
    -----------
    field_real : tuple
        Real-space components (c1, c2) where field = c1 + i*c2
    box_size_deg : float
        Box size in degrees
    config : dict
        Configuration
    spin : int
        Spin weight (1 for vector, 2 for spin-2)

    Returns:
    --------
    ell_centers, power_EE, power_BB, power_EB : numpy.ndarray
        E and B mode auto power spectra, and E×B cross power
    """
    c1, c2 = field_real
    nx, ny = c1.shape

    # Create k-space grid
    box_size_rad = box_size_deg * np.pi / 180.0
    dx = box_size_rad / nx
    dy = box_size_rad / ny
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
    ell_2d = np.sqrt(kx_grid**2 + ky_grid**2)
    phi_k = np.arctan2(ky_grid, kx_grid)

    # FFT of complex field f = c1 + i*c2
    field_complex = c1 + 1j * c2
    field_fft = np.fft.fft2(field_complex)

    # E/B decomposition using Hermitian symmetry
    # D(ℓ) = f̃(ℓ) × e^{-isφ_ℓ}
    D = field_fft * np.exp(-1j * spin * phi_k)

    # Use Hermitian conjugate flip to get D*(-ℓ)
    D_conj_flip = hermitian_conjugate_flip(D)

    # Proper E/B separation
    E_fft = (D + D_conj_flip) / 2  # Complex field with Hermitian symmetry
    B_fft = (D - D_conj_flip) / (2j)  # Complex field with Hermitian symmetry

    # Power spectra: |Ẽ|² and |B̃|²
    power_E_2d = np.abs(E_fft)**2
    power_B_2d = np.abs(B_fft)**2
    power_EB_2d = (E_fft * np.conj(B_fft)).real

    # Normalize
    power_E_2d *= (dx * dy)**2 / (nx * ny)
    power_B_2d *= (dx * dy)**2 / (nx * ny)
    power_EB_2d *= (dx * dy)**2 / (nx * ny)

    # Binning
    ell_min = config.get('ell_min', 10)
    ell_max = config.get('ell_max', None)
    if ell_max is None:
        ell_max = np.sqrt(2) * np.pi * min(nx, ny) / (2 * box_size_rad)
    n_bins = config.get('n_bins', 20)
    use_log_bins = config.get('use_log_bins', True)

    if use_log_bins:
        bin_edges = np.geomspace(max(ell_min, 1), ell_max, n_bins + 1)
    else:
        bin_edges = np.linspace(ell_min, ell_max, n_bins + 1)

    power_E_binned = np.zeros(n_bins)
    power_B_binned = np.zeros(n_bins)
    power_EB_binned = np.zeros(n_bins)
    ell_centers = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (ell_2d >= bin_edges[i]) & (ell_2d < bin_edges[i+1])
        if np.sum(mask) > 0:
            power_E_binned[i] = np.mean(power_E_2d[mask])
            power_B_binned[i] = np.mean(power_B_2d[mask])
            power_EB_binned[i] = np.mean(power_EB_2d[mask])
            ell_centers[i] = np.mean(ell_2d[mask])
        else:
            ell_centers[i] = (bin_edges[i] + bin_edges[i+1]) / 2

    return ell_centers, power_E_binned, power_B_binned, power_EB_binned


def compute_spin_cross(field1_scalar, field2_spin, box_size_deg, config, spin):
    """
    Compute cross power spectrum between scalar (spin-0) and spin field.

    Uses proper Hermitian E/B decomposition for the spin field.

    Parameters:
    -----------
    field1_scalar : numpy.ndarray
        Scalar field in real space
    field2_spin : tuple
        Spin field components (c1, c2)
    box_size_deg : float
        Box size in degrees
    config : dict
        Configuration
    spin : int
        Spin weight of second field

    Returns:
    --------
    ell_centers, power_E, power_B : numpy.ndarray
        E and B mode cross power spectra
    """
    nx, ny = field1_scalar.shape

    # Create k-space grid
    box_size_rad = box_size_deg * np.pi / 180.0
    dx = box_size_rad / nx
    dy = box_size_rad / ny
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
    ell_2d = np.sqrt(kx_grid**2 + ky_grid**2)
    phi_k = np.arctan2(ky_grid, kx_grid)

    # FFT of scalar field
    fft_scalar = np.fft.fft2(field1_scalar)

    # FFT of spin field
    c1, c2 = field2_spin
    field_spin_complex = c1 + 1j * c2
    field_spin_fft = np.fft.fft2(field_spin_complex)

    # E/B decomposition of spin field using Hermitian symmetry
    D = field_spin_fft * np.exp(-1j * spin * phi_k)
    D_conj_flip = hermitian_conjugate_flip(D)

    E_fft = (D + D_conj_flip) / 2
    B_fft = (D - D_conj_flip) / (2j)

    # Cross power: scalar × E and scalar × B
    # C_ℓ = ⟨S̃(ℓ) × Ẽ*(ℓ)⟩
    power_E_2d = (fft_scalar * np.conj(E_fft)).real
    power_B_2d = (fft_scalar * np.conj(B_fft)).real

    # Normalize
    power_E_2d *= (dx * dy)**2 / (nx * ny)
    power_B_2d *= (dx * dy)**2 / (nx * ny)

    # Binning
    ell_min = config.get('ell_min', 10)
    ell_max = config.get('ell_max', None)
    if ell_max is None:
        ell_max = np.sqrt(2) * np.pi * min(nx, ny) / (2 * box_size_rad)
    n_bins = config.get('n_bins', 20)
    use_log_bins = config.get('use_log_bins', True)

    if use_log_bins:
        bin_edges = np.geomspace(max(ell_min, 1), ell_max, n_bins + 1)
    else:
        bin_edges = np.linspace(ell_min, ell_max, n_bins + 1)

    power_E_binned = np.zeros(n_bins)
    power_B_binned = np.zeros(n_bins)
    ell_centers = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (ell_2d >= bin_edges[i]) & (ell_2d < bin_edges[i+1])
        if np.sum(mask) > 0:
            power_E_binned[i] = np.mean(power_E_2d[mask])
            power_B_binned[i] = np.mean(power_B_2d[mask])
            ell_centers[i] = np.mean(ell_2d[mask])
        else:
            ell_centers[i] = (bin_edges[i] + bin_edges[i+1]) / 2

    return ell_centers, power_E_binned, power_B_binned


def compute_spin_spin_cross(field1_spin, field2_spin, box_size_deg, config, spin1, spin2):
    """
    Compute cross power spectrum between two spin fields.

    Uses proper Hermitian E/B decomposition for both fields.

    Parameters:
    -----------
    field1_spin : tuple
        First spin field components (c1, c2)
    field2_spin : tuple
        Second spin field components (c1, c2)
    box_size_deg : float
        Box size in degrees
    config : dict
        Configuration
    spin1, spin2 : int
        Spin weights of the two fields

    Returns:
    --------
    ell_centers, power_E, power_B : numpy.ndarray
        E and B mode cross power spectra (EE and BB)
    """
    c1_1, c2_1 = field1_spin
    c1_2, c2_2 = field2_spin
    nx, ny = c1_1.shape

    # Create k-space grid
    box_size_rad = box_size_deg * np.pi / 180.0
    dx = box_size_rad / nx
    dy = box_size_rad / ny
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
    ell_2d = np.sqrt(kx_grid**2 + ky_grid**2)
    phi_k = np.arctan2(ky_grid, kx_grid)

    # FFT both fields
    field1_fft = np.fft.fft2(c1_1 + 1j * c2_1)
    field2_fft = np.fft.fft2(c1_2 + 1j * c2_2)

    # E/B decomposition for field 1
    D1 = field1_fft * np.exp(-1j * spin1 * phi_k)
    D1_conj_flip = hermitian_conjugate_flip(D1)
    E1_fft = (D1 + D1_conj_flip) / 2
    B1_fft = (D1 - D1_conj_flip) / (2j)

    # E/B decomposition for field 2
    D2 = field2_fft * np.exp(-1j * spin2 * phi_k)
    D2_conj_flip = hermitian_conjugate_flip(D2)
    E2_fft = (D2 + D2_conj_flip) / 2
    B2_fft = (D2 - D2_conj_flip) / (2j)

    # Cross power spectra: E1×E2* and B1×B2*
    power_E_2d = (E1_fft * np.conj(E2_fft)).real
    power_B_2d = (B1_fft * np.conj(B2_fft)).real

    # Normalize
    power_E_2d *= (dx * dy)**2 / (nx * ny)
    power_B_2d *= (dx * dy)**2 / (nx * ny)

    # Binning
    ell_min = config.get('ell_min', 10)
    ell_max = config.get('ell_max', None)
    if ell_max is None:
        ell_max = np.sqrt(2) * np.pi * min(nx, ny) / (2 * box_size_rad)
    n_bins = config.get('n_bins', 20)
    use_log_bins = config.get('use_log_bins', True)

    if use_log_bins:
        bin_edges = np.geomspace(max(ell_min, 1), ell_max, n_bins + 1)
    else:
        bin_edges = np.linspace(ell_min, ell_max, n_bins + 1)

    power_E_binned = np.zeros(n_bins)
    power_B_binned = np.zeros(n_bins)
    ell_centers = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (ell_2d >= bin_edges[i]) & (ell_2d < bin_edges[i+1])
        if np.sum(mask) > 0:
            power_E_binned[i] = np.mean(power_E_2d[mask])
            power_B_binned[i] = np.mean(power_B_2d[mask])
            ell_centers[i] = np.mean(ell_2d[mask])
        else:
            ell_centers[i] = (bin_edges[i] + bin_edges[i+1]) / 2

    return ell_centers, power_E_binned, power_B_binned


def compute_all_power_spectra(phi_real, scalar_field, vector_fields, spin2_fields,
                               box_size_deg, config):
    """
    Compute all auto and cross power spectra with proper E/B decomposition.

    Parameters:
    -----------
    phi_real : numpy.ndarray
        Gravitational potential (spin-0)
    scalar_field : numpy.ndarray
        Scalar field (spin-0)
    vector_fields : tuple
        Vector field components (vx, vy)
    spin2_fields : tuple
        Spin-2 field components (gamma1, gamma2)
    box_size_deg : float
        Box size in degrees
    config : dict
        Configuration for binning

    Returns:
    --------
    results : dict
        Dictionary containing all power spectra
    """
    print("\n  Computing power spectra with proper E/B decomposition...")

    results = {}

    # Scalar auto-correlations
    print("    - phi x phi")
    ell, cl = compute_power_spectrum_2d(phi_real, phi_real, box_size_deg, config)
    results['ell'] = ell
    results['phi_auto'] = cl

    print("    - scalar x scalar")
    _, cl = compute_power_spectrum_2d(scalar_field, scalar_field, box_size_deg, config)
    results['scalar_auto'] = cl

    print("    - phi x scalar")
    _, cl = compute_power_spectrum_2d(phi_real, scalar_field, box_size_deg, config)
    results['phi_scalar_cross'] = cl

    # Vector field (spin-1) auto-correlation
    print("    - vector x vector")
    _, power_EE, power_BB, power_EB = compute_spin_power_auto(vector_fields, box_size_deg, config, spin=1)
    results['vector_auto_EE'] = power_EE
    results['vector_auto_BB'] = power_BB
    results['vector_auto_EB'] = power_EB
    results['vector_auto_BE'] = power_EB  # EB = BE for auto-correlation

    # Spin-2 field auto-correlation
    print("    - spin2 x spin2")
    _, power_EE, power_BB, power_EB = compute_spin_power_auto(spin2_fields, box_size_deg, config, spin=2)
    results['spin2_auto_EE'] = power_EE
    results['spin2_auto_BB'] = power_BB
    results['spin2_auto_EB'] = power_EB
    results['spin2_auto_BE'] = power_EB  # EB = BE for auto-correlation

    # Scalar x Vector cross-correlations
    print("    - phi x vector")
    _, power_E, power_B = compute_spin_cross(phi_real, vector_fields, box_size_deg, config, spin=1)
    results['phi_vector_E'] = power_E
    results['phi_vector_B'] = power_B

    print("    - scalar x vector")
    _, power_E, power_B = compute_spin_cross(scalar_field, vector_fields, box_size_deg, config, spin=1)
    results['scalar_vector_E'] = power_E
    results['scalar_vector_B'] = power_B

    # Scalar x Spin-2 cross-correlations
    print("    - phi x spin2")
    _, power_E, power_B = compute_spin_cross(phi_real, spin2_fields, box_size_deg, config, spin=2)
    results['phi_spin2_E'] = power_E
    results['phi_spin2_B'] = power_B

    print("    - scalar x spin2")
    _, power_E, power_B = compute_spin_cross(scalar_field, spin2_fields, box_size_deg, config, spin=2)
    results['scalar_spin2_E'] = power_E
    results['scalar_spin2_B'] = power_B

    # Vector x Spin-2 cross (both are spin fields)
    print("    - vector x spin2")
    _, power_EE, power_BB = compute_spin_spin_cross(vector_fields, spin2_fields,
                                                     box_size_deg, config, spin1=1, spin2=2)
    results['vector_spin2_EE'] = power_EE
    results['vector_spin2_BB'] = power_BB
    results['vector_spin2_EB'] = np.zeros_like(power_EE)
    results['vector_spin2_BE'] = np.zeros_like(power_EE)

    return results
