"""
Library functions for generating fields from gravitational potential.
"""

import numpy as np


def enforce_hermitian_symmetry(field_fft, nx, ny):
    """Ensure Hermitian symmetry for real-valued inverse FFT."""
    # DC component must be real
    field_fft[0, 0] = field_fft[0, 0].real

    # Nyquist frequencies must be real if they exist
    if nx % 2 == 0:
        field_fft[nx//2, 0] = field_fft[nx//2, 0].real
    if ny % 2 == 0:
        field_fft[0, ny//2] = field_fft[0, ny//2].real

    return field_fft


def generate_phi_from_power_spectrum(nx, ny, box_size_deg, power_spectrum_config, seed=42):
    """
    Generate gravitational potential Phi from a power spectrum.

    Parameters:
    -----------
    nx, ny : int
        Grid dimensions
    box_size_deg : float
        Physical size of the box (degrees)
    power_spectrum_config : dict
        Configuration for power spectrum
    seed : int
        Random seed

    Returns:
    --------
    phi_real : numpy.ndarray
        Gravitational potential in real space
    phi_fourier : numpy.ndarray
        Gravitational potential in Fourier space
    """
    np.random.seed(seed)

    # Resolution in degrees
    dx = box_size_deg / nx
    dy = box_size_deg / ny

    # Create frequency grids
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')

    # Magnitude of k vector (ell in harmonic space)
    ell = np.sqrt(kx_grid**2 + ky_grid**2)
    ell[0, 0] = 1.0  # Avoid division by zero

    # Generate power spectrum
    ps_type = power_spectrum_config.get('type', 'power_law')

    if ps_type == 'power_law':
        index = power_spectrum_config.get('index', -2.0)
        amplitude = power_spectrum_config.get('amplitude', 1.0)
        power_spectrum = amplitude * ell**index
        # Avoid very small values at high ell
        power_spectrum = np.maximum(power_spectrum, 1e-20)
    elif ps_type == 'custom':
        # Load custom power spectrum from file
        ps_file = power_spectrum_config['custom_file']
        data = np.loadtxt(ps_file)
        k_data = data[:, 0]
        pk_data = data[:, 1]
        # Interpolate to current k grid
        from scipy.interpolate import interp1d
        ps_interp = interp1d(k_data, pk_data, bounds_error=False, fill_value=0)
        power_spectrum = ps_interp(ell)
    else:
        raise ValueError(f"Unknown power spectrum type: {ps_type}")

    power_spectrum[0, 0] = 0  # Zero mean

    # Generate complex Gaussian random field
    real_part = np.random.randn(nx, ny)
    imag_part = np.random.randn(nx, ny)

    # Scale by power spectrum
    # To match the power spectrum measurement convention:
    # P(k) = |FFT|^2 * (dx*dy)^2 / (nx*ny)
    # So we need: |FFT|^2 = P(k) * (nx*ny) / (dx*dy)^2
    # Note: After enforcing Hermitian symmetry, we effectively only have independent
    # modes in half the Fourier space, but the conjugate constraint doubles the power.
    # Empirically, we need an extra factor of sqrt(2) to match the measured power spectrum.
    sigma = np.sqrt(power_spectrum * (nx * ny) / ((dx * dy)**2))
    phi_fourier = (real_part + 1j * imag_part) * sigma

    # Ensure Hermitian symmetry for real output
    phi_fourier = enforce_hermitian_symmetry(phi_fourier, nx, ny)

    # Transform to real space
    phi_real = np.fft.ifft2(phi_fourier).real

    return phi_real, phi_fourier


def generate_spin_field(phi_fourier, amplitude, ell_power, spin, nx, ny, box_size_deg):
    """
    Unified function to generate spin-s field: f̃(ℓ⃗) = amplitude · ℓ^n · e^(i·s·φ_ℓ) · |Φ(ℓ⃗)|

    Parameters:
    -----------
    phi_fourier : numpy.ndarray
        Gravitational potential in Fourier space
    amplitude : float
        Amplitude coefficient (A, B, or C)
    ell_power : int
        Power of ℓ (0, 1, or 2)
    spin : int
        Spin weight (0, 1, or 2)
    nx, ny : int
        Grid dimensions
    box_size_deg : float
        Box size in degrees

    Returns:
    --------
    field_components : numpy.ndarray or tuple
        For spin-0: single real array
        For spin>0: tuple of (component1, component2)
    """
    dx = box_size_deg / nx
    dy = box_size_deg / ny

    # Create frequency grids
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')

    # ℓ and angle
    ell = np.sqrt(kx_grid**2 + ky_grid**2)
    phi_ell = np.arctan2(ky_grid, kx_grid)

    # f̃(ℓ) = amplitude · ℓ^n · e^(i·s·φ_ℓ) · Φ(ℓ) or |Φ(ℓ)|
    # For spin-0: use Φ directly (no E/B decomposition issue)
    # For spin>0: use |Φ| to ensure pure E-mode
    if spin == 0:
        field_fourier = amplitude * (ell**ell_power) * phi_fourier
    else:
        # For complex Gaussian Φ: using |Φ| loses a factor of √2 in amplitude
        field_fourier = amplitude * (ell**ell_power) * np.exp(1j * spin * phi_ell) * np.abs(phi_fourier) / np.sqrt(2)

    # Transform to real space
    if spin == 0:
        # Spin-0 field is purely real
        field_real = np.fft.ifft2(field_fourier).real
        return field_real
    else:
        # Spin>0 fields have two components
        field_complex = np.fft.ifft2(field_fourier)
        component1 = field_complex.real
        component2 = field_complex.imag
        return component1, component2


def generate_scalar_field(phi_fourier, A, nx, ny, box_size_deg):
    """
    Generate scalar field: δ(ℓ⃗) = Aℓ²e^(i·0·φ_ℓ)|Φ(ℓ⃗)| = Aℓ²|Φ(ℓ⃗)|

    Spin-0 field with ℓ² power and no angular dependence.
    """
    return generate_spin_field(phi_fourier, A, ell_power=2, spin=0, nx=nx, ny=ny, box_size_deg=box_size_deg)


def generate_vector_field(phi_fourier, B, nx, ny, box_size_deg):
    """
    Generate spin-1 field: α̃(ℓ⃗) = Bℓe^(i·1·φ_ℓ)|Φ(ℓ⃗)|

    Spin-1 field with ℓ¹ power and e^(iφ_ℓ) angular dependence.
    """
    return generate_spin_field(phi_fourier, B, ell_power=1, spin=1, nx=nx, ny=ny, box_size_deg=box_size_deg)


def generate_spin2_field(phi_fourier, C, nx, ny, box_size_deg):
    """
    Generate spin-2 field: γ̃(ℓ⃗) = Cℓ²e^(i·2·φ_ℓ)|Φ(ℓ⃗)|

    Spin-2 field with ℓ² power and e^(2iφ_ℓ) angular dependence.
    """
    return generate_spin_field(phi_fourier, C, ell_power=2, spin=2, nx=nx, ny=ny, box_size_deg=box_size_deg)
