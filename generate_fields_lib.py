"""
Library functions for generating spin-weighted fields from gravitational potential.

Fields derived from a scalar potential Φ:
    κ (convergence) = Bℓ²Φ           (spin-0)
    α (velocity)    = Cℓe^{iφ}Φ      (spin-1)
    γ (shear)       = Dℓ²e^{2iφ}Φ    (spin-2)

All fields derived from a scalar potential are pure E-mode.
"""

import numpy as np


def generate_hermitian_random_field(nx, ny, power_spectrum, rng=None):
    """
    Generate a complex Fourier field with proper Hermitian symmetry.

    For a real-valued field in position space, its FFT must satisfy:
    Φ̃(-ℓ) = Φ̃*(ℓ)

    This means:
    - DC mode (0,0) is real
    - Nyquist modes are real
    - Other modes come in conjugate pairs

    Parameters:
    -----------
    nx, ny : int
        Grid dimensions
    power_spectrum : numpy.ndarray
        2D power spectrum P(ℓ) with shape (nx, ny)
    rng : numpy.random.Generator, optional
        Random number generator

    Returns:
    --------
    field_fft : numpy.ndarray
        Complex field satisfying Hermitian symmetry
    """
    if rng is None:
        rng = np.random.default_rng()

    field_fft = np.zeros((nx, ny), dtype=complex)

    # Standard deviation from power spectrum
    # For proper normalization: ⟨|Φ̃|²⟩ = P(ℓ)
    sigma = np.sqrt(power_spectrum / 2)  # Factor of 2 for real+imag parts

    # DC mode (must be real)
    field_fft[0, 0] = rng.normal() * np.sqrt(power_spectrum[0, 0])

    # Fill modes with proper Hermitian pairing
    # For mode (i, j), the conjugate mode is at (-i mod nx, -j mod ny)
    for i in range(nx):
        for j in range(ny):
            if i == 0 and j == 0:
                continue  # Already set DC

            mi = (-i) % nx
            mj = (-j) % ny

            # Skip if we already set this mode via its conjugate pair
            if (mi, mj) < (i, j):
                continue

            # Check if this is a self-conjugate mode (i == mi and j == mj)
            # This happens at Nyquist when nx or ny is even
            if i == mi and j == mj:
                # Self-conjugate: must be real
                field_fft[i, j] = rng.normal() * np.sqrt(power_spectrum[i, j])
            else:
                # Regular mode: set both (i,j) and (mi, mj) as conjugate pair
                real_part = rng.normal() * sigma[i, j]
                imag_part = rng.normal() * sigma[i, j]
                field_fft[i, j] = real_part + 1j * imag_part
                field_fft[mi, mj] = real_part - 1j * imag_part  # Conjugate

    return field_fft


def generate_phi_from_power_spectrum(nx, ny, box_size_deg, power_spectrum_config, seed=42):
    """
    Generate gravitational potential Phi from a power spectrum.

    Creates a properly Hermitian-symmetric field so that phi_real is truly real
    and E/B decomposition of derived spin fields works correctly.

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
        Gravitational potential in Fourier space (Hermitian-symmetric)
    """
    rng = np.random.default_rng(seed)

    # Convert box size to radians for proper ℓ calculation
    box_size_rad = box_size_deg * np.pi / 180.0
    dx = box_size_rad / nx
    dy = box_size_rad / ny

    # Create frequency grids (ℓ = 2π × spatial_frequency in radians)
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')

    # Magnitude of k vector (ell in harmonic space)
    ell = np.sqrt(kx_grid**2 + ky_grid**2)
    ell[0, 0] = 1.0  # Avoid division by zero

    # Generate power spectrum: P_Φ(ℓ) = A × ℓ^n / (1 + (ℓ₀/ℓ)²)
    # Broken power law to regularize IR divergence
    ps_type = power_spectrum_config.get('type', 'power_law')
    if ps_type != 'power_law':
        raise ValueError(f"Only 'power_law' type supported, got: {ps_type}")

    index = power_spectrum_config.get('index', -2.0)
    amplitude = power_spectrum_config.get('amplitude', 1.0)
    ell_0 = power_spectrum_config.get('ell_0', 10.0)  # IR cutoff scale

    # Broken power law: flattens at low ℓ for IR convergence
    power_spectrum = amplitude * ell**index / (1 + (ell_0 / ell)**2)
    power_spectrum = np.maximum(power_spectrum, 1e-20)  # Numerical floor

    power_spectrum[0, 0] = 0  # Zero mean

    # Scale power spectrum for FFT normalization:
    # Physical convention: Var(field) = ∫ C_ℓ d²ℓ / (2π)² ≈ (1/L²) Σ C_ℓ
    # With numpy's ifft2 (divides by N²): Var(field) = Σ Var(FFT) / N⁴
    # So we need: Var(FFT) = C_ℓ × N⁴ / L²
    scaled_power = power_spectrum * (nx * ny)**2 / (box_size_rad**2)

    # Generate Hermitian-symmetric random field
    phi_fourier = generate_hermitian_random_field(nx, ny, scaled_power, rng)

    # Transform to real space (should be purely real due to Hermitian symmetry)
    phi_real = np.fft.ifft2(phi_fourier).real

    return phi_real, phi_fourier


def generate_spin_field(phi_fourier, amplitude, ell_power, spin, nx, ny, box_size_deg):
    """
    Generate spin-s field: f̃(ℓ⃗) = amplitude · ℓ^n · e^(i·s·φ_ℓ) · Φ(ℓ⃗)

    For fields derived from a scalar potential, this produces pure E-mode fields.
    The E/B decomposition should use Hermitian symmetry (see measure_power_spectra.py).

    Parameters:
    -----------
    phi_fourier : numpy.ndarray
        Gravitational potential in Fourier space (Hermitian-symmetric for real Φ)
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
    # Convert to radians for proper ℓ values
    box_size_rad = box_size_deg * np.pi / 180.0
    dx = box_size_rad / nx
    dy = box_size_rad / ny

    # Create frequency grids
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')

    # ℓ and angle
    ell = np.sqrt(kx_grid**2 + ky_grid**2)
    phi_ell = np.arctan2(ky_grid, kx_grid)

    # f̃(ℓ) = amplitude · ℓ^n · e^(i·s·φ_ℓ) · Φ(ℓ)
    # Use full complex Φ for all fields to preserve correlations
    field_fourier = amplitude * (ell**ell_power) * np.exp(1j * spin * phi_ell) * phi_fourier

    # Transform to real space
    if spin == 0:
        # Spin-0 field is purely real
        field_real = np.fft.ifft2(field_fourier).real
        return field_real
    else:
        # Spin>0 fields have two components (γ₁, γ₂) or (αₓ, αᵧ)
        field_complex = np.fft.ifft2(field_fourier)
        component1 = field_complex.real
        component2 = field_complex.imag
        return component1, component2


def generate_scalar_field(phi_fourier, A, nx, ny, box_size_deg):
    """
    Generate scalar field: δ̃(ℓ⃗) = Aℓ²Φ̃(ℓ⃗)

    Spin-0 field with ℓ² power and no angular dependence.
    """
    return generate_spin_field(phi_fourier, A, ell_power=2, spin=0, nx=nx, ny=ny, box_size_deg=box_size_deg)


def generate_vector_field(phi_fourier, B, nx, ny, box_size_deg):
    """
    Generate spin-1 velocity field as gradient of potential.

    The gradient ∇Φ in Fourier space has components:
        α̃_x = i k_x Φ̃
        α̃_y = i k_y Φ̃

    For NaMaster E/B decomposition to work correctly with the gradient as E-mode,
    we generate the Cartesian components directly.

    Returns (αₓ, αᵧ) components in real space.
    """
    box_size_rad = box_size_deg * np.pi / 180.0
    dx = box_size_rad / nx
    dy = box_size_rad / ny

    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')

    # Gradient components: ∂Φ/∂x = i k_x Φ, ∂Φ/∂y = i k_y Φ
    alpha_x_fourier = B * 1j * kx_grid * phi_fourier
    alpha_y_fourier = B * 1j * ky_grid * phi_fourier

    # Transform to real space (should be real due to Hermitian symmetry)
    alpha_x = np.fft.ifft2(alpha_x_fourier).real
    alpha_y = np.fft.ifft2(alpha_y_fourier).real

    return alpha_x, alpha_y


def generate_spin2_field(phi_fourier, C, nx, ny, box_size_deg):
    """
    Generate spin-2 shear field: γ̃(ℓ⃗) = -Cℓ²e^(2iφ_ℓ)Φ̃(ℓ⃗)

    Following main.tex convention: γ̃ = -ℓ₊²Φ̃ where ℓ₊ = ℓe^{iφ_ℓ}.
    The minus sign comes from the second derivative of the lensing potential.
    Returns (γ₁, γ₂) components in real space.
    """
    # Use negative amplitude to match main.tex: γ̃ = -ℓ₊²Φ̃
    return generate_spin_field(phi_fourier, -C, ell_power=2, spin=2, nx=nx, ny=ny, box_size_deg=box_size_deg)
