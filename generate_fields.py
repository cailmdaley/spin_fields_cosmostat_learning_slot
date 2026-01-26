"""
Generate three fields from a gravitational potential and compute correlations.

This script generates:
1. Gravitational potential Phi from a power spectrum
2. Scalar field: δ(ℓ⃗) = Aℓ²Φ(ℓ⃗)
3. Vector field: α⃗(ℓ⃗) = Bℓe^(iφ_ℓ)Φ(ℓ⃗) = Biℓ⃗Φ(ℓ⃗)
4. Spin-2 field: γ = Cℓ²e^(2iφ_ℓ)Φ(ℓ⃗)

Then computes all auto and cross-correlations using NaMaster.
"""

import numpy as np
import matplotlib.pyplot as plt
import pymaster as nmt
import yaml
import os
from pathlib import Path


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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
    # Factor accounts for 2D FFT normalization
    phi_fourier = (real_part + 1j * imag_part) * np.sqrt(power_spectrum / 2.0) / (nx * ny)

    # Ensure Hermitian symmetry for real output
    phi_fourier = enforce_hermitian_symmetry(phi_fourier, nx, ny)

    # Transform to real space
    phi_real = np.fft.ifft2(phi_fourier).real

    return phi_real, phi_fourier


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


def generate_scalar_field(phi_fourier, A, nx, ny, box_size_deg):
    """
    Generate scalar field: δ(ℓ⃗) = Aℓ²Φ(ℓ⃗)

    Parameters:
    -----------
    phi_fourier : numpy.ndarray
        Gravitational potential in Fourier space
    A : float
        Amplitude coefficient
    nx, ny : int
        Grid dimensions
    box_size_deg : float
        Box size in degrees

    Returns:
    --------
    delta_real : numpy.ndarray
        Scalar field in real space
    """
    dx = box_size_deg / nx
    dy = box_size_deg / ny

    # Create frequency grids
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')

    # ℓ² = kx² + ky²
    ell_squared = kx_grid**2 + ky_grid**2

    # δ(ℓ) = Aℓ²Φ(ℓ)
    delta_fourier = A * ell_squared * phi_fourier

    # Transform to real space
    delta_real = np.fft.ifft2(delta_fourier).real

    return delta_real


def generate_vector_field(phi_fourier, B, nx, ny, box_size_deg):
    """
    Generate vector field: α⃗(ℓ⃗) = Bℓe^(iφ_ℓ)Φ(ℓ⃗) = Biℓ⃗Φ(ℓ⃗)

    In Cartesian components: iℓ⃗ = i(ℓₓ, ℓᵧ)

    Parameters:
    -----------
    phi_fourier : numpy.ndarray
        Gravitational potential in Fourier space
    B : float
        Amplitude coefficient
    nx, ny : int
        Grid dimensions
    box_size_deg : float
        Box size in degrees

    Returns:
    --------
    alpha_x_real, alpha_y_real : numpy.ndarray
        Vector field components in real space
    """
    dx = box_size_deg / nx
    dy = box_size_deg / ny

    # Create frequency grids
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')

    # α⃗(ℓ⃗) = Biℓ⃗Φ(ℓ⃗)
    # This is the gradient operator
    alpha_x_fourier = B * 1j * kx_grid * phi_fourier
    alpha_y_fourier = B * 1j * ky_grid * phi_fourier

    # Transform to real space
    alpha_x_real = np.fft.ifft2(alpha_x_fourier).real
    alpha_y_real = np.fft.ifft2(alpha_y_fourier).real

    return alpha_x_real, alpha_y_real


def generate_spin2_field(phi_fourier, C, nx, ny, box_size_deg):
    """
    Generate spin-2 field: γ = Cℓ²e^(2iφ_ℓ)Φ(ℓ⃗)
    where φ_ℓ = arctan(ℓ_y/ℓ_x)

    Parameters:
    -----------
    phi_fourier : numpy.ndarray
        Gravitational potential in Fourier space
    C : float
        Amplitude coefficient
    nx, ny : int
        Grid dimensions
    box_size_deg : float
        Box size in degrees

    Returns:
    --------
    gamma1_real, gamma2_real : numpy.ndarray
        Spin-2 field components in real space
    """
    dx = box_size_deg / nx
    dy = box_size_deg / ny

    # Create frequency grids
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')

    # φ_ℓ = arctan(ℓ_y/ℓ_x)
    phi_ell = np.arctan2(ky_grid, kx_grid)

    # ℓ² = kx² + ky²
    ell_squared = kx_grid**2 + ky_grid**2

    # γ = Cℓ²e^(2iφ_ℓ)Φ(ℓ⃗)
    gamma_fourier = C * ell_squared * np.exp(2j * phi_ell) * phi_fourier

    # Transform to real space - this gives a complex field
    gamma_complex = np.fft.ifft2(gamma_fourier)

    # Split into real and imaginary parts (γ1 and γ2, like shear components)
    gamma1_real = gamma_complex.real
    gamma2_real = gamma_complex.imag

    return gamma1_real, gamma2_real


def plot_field(field, title, output_path, cmap='RdBu_r', dpi=150):
    """
    Plot a single field and save as PNG.

    Parameters:
    -----------
    field : numpy.ndarray or tuple
        Field to plot (2D array or tuple of arrays for vector fields)
    title : str
        Plot title
    output_path : str
        Path to save PNG
    cmap : str
        Colormap name
    dpi : int
        DPI for output figure
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    if isinstance(field, tuple):
        # Vector field - plot magnitude
        vx, vy = field
        magnitude = np.sqrt(vx**2 + vy**2)
        im = ax.imshow(magnitude, cmap=cmap, origin='lower', aspect='auto')

        # Add arrow overlay (subsample for clarity)
        step = max(field[0].shape[0] // 20, 1)
        x = np.arange(0, field[0].shape[1], step)
        y = np.arange(0, field[0].shape[0], step)
        X, Y = np.meshgrid(x, y)
        ax.quiver(X, Y, vx[::step, ::step], vy[::step, ::step],
                 color='white', alpha=0.6, scale=None, width=0.003)
    else:
        # Scalar field
        im = ax.imshow(field, cmap=cmap, origin='lower', aspect='auto')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('x [pixels]', fontsize=12)
    ax.set_ylabel('y [pixels]', fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def compute_all_power_spectra(phi_real, scalar_field, vector_fields, spin2_fields,
                               box_size_deg, namaster_config, mask=None):
    """
    Compute all auto and cross power spectra using NaMaster.

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
    namaster_config : dict
        NaMaster configuration
    mask : numpy.ndarray, optional
        Binary mask

    Returns:
    --------
    results : dict
        Dictionary containing all power spectra
    """
    ny, nx = phi_real.shape

    # Create mask if not provided
    if mask is None:
        mask = np.ones((ny, nx))

    # Convert box size from degrees to radians
    box_size_rad = box_size_deg * np.pi / 180.0

    print("\n  Creating NaMaster flat-sky fields...")
    # Create NaMaster flat-sky fields
    # Lx, Ly are the dimensions of the box in radians
    field_phi = nmt.NmtFieldFlat(box_size_rad, box_size_rad, mask, [phi_real])
    field_scalar = nmt.NmtFieldFlat(box_size_rad, box_size_rad, mask, [scalar_field])

    vx, vy = vector_fields
    field_vector = nmt.NmtFieldFlat(box_size_rad, box_size_rad, mask, [vx, vy],
                                    purify_b=False)

    gamma1, gamma2 = spin2_fields
    field_spin2 = nmt.NmtFieldFlat(box_size_rad, box_size_rad, mask, [gamma1, gamma2])

    # Set up binning for flat-sky
    print("  Setting up ell bins...")
    ell_min = namaster_config.get('ell_min', 2)
    ell_max = namaster_config.get('ell_max', None)
    if ell_max is None:
        # For flat-sky, maximum ell is related to pixel scale
        ell_max = int(np.pi * min(nx, ny) / box_size_rad)
    n_bins = namaster_config.get('n_bins', 20)
    use_log_bins = namaster_config.get('use_log_bins', True)

    if use_log_bins:
        bin_edges = np.geomspace(ell_min, ell_max, n_bins + 1)
    else:
        bin_edges = np.linspace(ell_min, ell_max, n_bins + 1)

    bins = nmt.NmtBinFlat(bin_edges[:-1], bin_edges[1:])
    ell_eff = bins.get_effective_ells()

    results = {'ell': ell_eff}

    # List of all field pairs to compute
    field_pairs = [
        ('phi', 'phi', field_phi, field_phi),
        ('phi', 'scalar', field_phi, field_scalar),
        ('phi', 'vector', field_phi, field_vector),
        ('phi', 'spin2', field_phi, field_spin2),
        ('scalar', 'scalar', field_scalar, field_scalar),
        ('scalar', 'vector', field_scalar, field_vector),
        ('scalar', 'spin2', field_scalar, field_spin2),
        ('vector', 'vector', field_vector, field_vector),
        ('vector', 'spin2', field_vector, field_spin2),
        ('spin2', 'spin2', field_spin2, field_spin2),
    ]

    print("  Computing power spectra...")
    for name1, name2, f1, f2 in field_pairs:
        print(f"    - {name1} x {name2}")

        # Compute workspace for flat-sky
        w = nmt.NmtWorkspaceFlat()
        w.compute_coupling_matrix(f1, f2, bins)

        # Compute power spectrum
        cl = nmt.compute_full_master_flat(f1, f2, bins, workspace=w)

        # Store results with appropriate names
        if name1 == name2 and name1 in ['phi', 'scalar']:
            # Spin-0 x Spin-0
            results[f'{name1}_auto'] = cl[0]
        elif name1 in ['phi', 'scalar'] and name2 in ['phi', 'scalar']:
            # Spin-0 x Spin-0 cross
            results[f'{name1}_{name2}_cross'] = cl[0]
        elif name1 in ['phi', 'scalar'] and name2 in ['vector', 'spin2']:
            # Spin-0 x Spin-1/2
            results[f'{name1}_{name2}_E'] = cl[0]
            results[f'{name1}_{name2}_B'] = cl[1]
        elif name1 == name2 and name1 in ['vector', 'spin2']:
            # Spin-1/2 x Spin-1/2
            results[f'{name1}_auto_EE'] = cl[0]
            results[f'{name1}_auto_EB'] = cl[1]
            results[f'{name1}_auto_BE'] = cl[2]
            results[f'{name1}_auto_BB'] = cl[3]
        else:
            # Spin-1 x Spin-2 or other combinations
            results[f'{name1}_{name2}_EE'] = cl[0]
            results[f'{name1}_{name2}_EB'] = cl[1]
            results[f'{name1}_{name2}_BE'] = cl[2]
            results[f'{name1}_{name2}_BB'] = cl[3]

    return results


def plot_power_spectra(results, output_path, dpi=150):
    """
    Plot all power spectra.

    Parameters:
    -----------
    results : dict
        Power spectra results
    output_path : str
        Path to save PNG
    dpi : int
        DPI for output
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    ell = results['ell']

    # Row 1: Auto-correlations
    # Phi auto
    ax = axes[0, 0]
    ax.loglog(ell, results['phi_auto'], 'o-', label='Φ auto', color='C0')
    ax.set_xlabel(r'$\ell$', fontsize=11)
    ax.set_ylabel(r'$C_\ell$', fontsize=11)
    ax.set_title('Φ Auto-Correlation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Scalar auto
    ax = axes[0, 1]
    ax.loglog(ell, results['scalar_auto'], 'o-', label='δ auto', color='C1')
    ax.set_xlabel(r'$\ell$', fontsize=11)
    ax.set_ylabel(r'$C_\ell$', fontsize=11)
    ax.set_title('Scalar (δ) Auto-Correlation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Vector auto
    ax = axes[0, 2]
    ax.loglog(ell, results['vector_auto_EE'], 'o-', label='EE', color='C2')
    ax.loglog(ell, np.abs(results['vector_auto_BB']), 's-', label='BB (abs)', color='C3')
    ax.set_xlabel(r'$\ell$', fontsize=11)
    ax.set_ylabel(r'$C_\ell$', fontsize=11)
    ax.set_title('Vector (α) Auto-Correlation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Row 2: More autos and cross-correlations
    # Spin-2 auto
    ax = axes[1, 0]
    ax.loglog(ell, results['spin2_auto_EE'], 'o-', label='EE', color='C4')
    ax.loglog(ell, np.abs(results['spin2_auto_BB']), 's-', label='BB (abs)', color='C5')
    ax.set_xlabel(r'$\ell$', fontsize=11)
    ax.set_ylabel(r'$C_\ell$', fontsize=11)
    ax.set_title('Spin-2 (γ) Auto-Correlation', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Phi-Scalar cross
    ax = axes[1, 1]
    ax.loglog(ell, np.abs(results['phi_scalar_cross']), 'o-', label='Φ-δ cross', color='C6')
    ax.set_xlabel(r'$\ell$', fontsize=11)
    ax.set_ylabel(r'$|C_\ell|$', fontsize=11)
    ax.set_title('Φ-Scalar Cross', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Phi-Vector cross
    ax = axes[1, 2]
    ax.semilogx(ell, results['phi_vector_E'], 'o-', label='E-mode', color='C7')
    ax.semilogx(ell, results['phi_vector_B'], 's-', label='B-mode', color='C8')
    ax.set_xlabel(r'$\ell$', fontsize=11)
    ax.set_ylabel(r'$C_\ell$', fontsize=11)
    ax.set_title('Φ-Vector Cross', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Row 3: More cross-correlations
    # Phi-Spin2 cross
    ax = axes[2, 0]
    ax.semilogx(ell, results['phi_spin2_E'], 'o-', label='E-mode', color='C9')
    ax.semilogx(ell, results['phi_spin2_B'], 's-', label='B-mode', color='C0')
    ax.set_xlabel(r'$\ell$', fontsize=11)
    ax.set_ylabel(r'$C_\ell$', fontsize=11)
    ax.set_title('Φ-Spin2 Cross', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Scalar-Vector cross
    ax = axes[2, 1]
    ax.semilogx(ell, results['scalar_vector_E'], 'o-', label='E-mode', color='C1')
    ax.semilogx(ell, results['scalar_vector_B'], 's-', label='B-mode', color='C2')
    ax.set_xlabel(r'$\ell$', fontsize=11)
    ax.set_ylabel(r'$C_\ell$', fontsize=11)
    ax.set_title('Scalar-Vector Cross', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Scalar-Spin2 cross
    ax = axes[2, 2]
    ax.semilogx(ell, results['scalar_spin2_E'], 'o-', label='E-mode', color='C3')
    ax.semilogx(ell, results['scalar_spin2_B'], 's-', label='B-mode', color='C4')
    ax.set_xlabel(r'$\ell$', fontsize=11)
    ax.set_ylabel(r'$C_\ell$', fontsize=11)
    ax.set_title('Scalar-Spin2 Cross', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")


def main(config_path="config.yaml"):
    """
    Main function to generate fields and compute correlations.

    Parameters:
    -----------
    config_path : str
        Path to configuration YAML file
    """
    print("="*60)
    print("SPIN FIELD GENERATION FROM GRAVITATIONAL POTENTIAL")
    print("="*60)

    # Load configuration
    print(f"\nLoading configuration from {config_path}...")
    config = load_config(config_path)

    # Extract parameters
    nx = config['grid']['nx']
    ny = config['grid']['ny']
    box_size_deg = config['grid']['box_size_deg']
    seed = config['random_seed']

    A = config['amplitudes']['A']
    B = config['amplitudes']['B']
    C = config['amplitudes']['C']

    # Create output directory
    output_dir = config['output']['output_dir']
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    dpi = config['output']['dpi']
    save_fields = config['output']['save_fields']

    print(f"  Grid: {nx} x {ny}")
    print(f"  Box size: {box_size_deg} degrees")
    print(f"  Amplitudes: A={A}, B={B}, C={C}")
    print(f"  Output directory: {output_dir}")

    # Step 1: Generate Phi from power spectrum
    print("\n" + "="*60)
    print("STEP 1: Generate Phi from power spectrum")
    print("="*60)
    phi_real, phi_fourier = generate_phi_from_power_spectrum(
        nx, ny, box_size_deg, config['power_spectrum'], seed
    )
    print(f"  Phi statistics: mean={phi_real.mean():.2e}, std={phi_real.std():.2e}")

    # Step 2: Generate three derived fields
    print("\n" + "="*60)
    print("STEP 2: Generate derived fields")
    print("="*60)

    print("  Generating scalar field δ(ℓ) = Aℓ²Φ(ℓ)...")
    scalar_field = generate_scalar_field(phi_fourier, A, nx, ny, box_size_deg)
    print(f"    Statistics: mean={scalar_field.mean():.2e}, std={scalar_field.std():.2e}")

    print("  Generating vector field α(ℓ) = Biℓ Φ(ℓ)...")
    vector_fields = generate_vector_field(phi_fourier, B, nx, ny, box_size_deg)
    vx, vy = vector_fields
    print(f"    Statistics: vx mean={vx.mean():.2e}, std={vx.std():.2e}")
    print(f"                vy mean={vy.mean():.2e}, std={vy.std():.2e}")

    print("  Generating spin-2 field γ = Cℓ²e^(2iφ_ℓ)Φ(ℓ)...")
    spin2_fields = generate_spin2_field(phi_fourier, C, nx, ny, box_size_deg)
    gamma1, gamma2 = spin2_fields
    print(f"    Statistics: γ1 mean={gamma1.mean():.2e}, std={gamma1.std():.2e}")
    print(f"                γ2 mean={gamma2.mean():.2e}, std={gamma2.std():.2e}")

    # Step 3: Create PNGs of all four fields
    print("\n" + "="*60)
    print("STEP 3: Create field visualizations")
    print("="*60)

    plot_field(phi_real, 'Gravitational Potential Φ',
              f'{output_dir}/phi_field.png', cmap='RdBu_r', dpi=dpi)

    plot_field(scalar_field, r'Scalar Field δ = $A\ell^2\Phi$',
              f'{output_dir}/scalar_field.png', cmap='RdBu_r', dpi=dpi)

    plot_field(vector_fields, r'Vector Field α = $Bi\ell\Phi$ (magnitude + arrows)',
              f'{output_dir}/vector_field.png', cmap='viridis', dpi=dpi)

    # For spin-2, create two plots (one for each component)
    plot_field(gamma1, r'Spin-2 Field $\gamma_1$',
              f'{output_dir}/spin2_field_gamma1.png', cmap='RdBu_r', dpi=dpi)

    plot_field(gamma2, r'Spin-2 Field $\gamma_2$',
              f'{output_dir}/spin2_field_gamma2.png', cmap='RdBu_r', dpi=dpi)

    # Save field arrays if requested
    if save_fields:
        print("\n  Saving field arrays...")
        np.save(f'{output_dir}/phi_field.npy', phi_real)
        np.save(f'{output_dir}/scalar_field.npy', scalar_field)
        np.save(f'{output_dir}/vector_field_x.npy', vx)
        np.save(f'{output_dir}/vector_field_y.npy', vy)
        np.save(f'{output_dir}/spin2_field_gamma1.npy', gamma1)
        np.save(f'{output_dir}/spin2_field_gamma2.npy', gamma2)
        print(f"    Saved to {output_dir}/*.npy")

    # Step 4: Calculate all power and cross power spectra
    print("\n" + "="*60)
    print("STEP 4: Calculate power spectra with NaMaster")
    print("="*60)

    results = compute_all_power_spectra(
        phi_real, scalar_field, vector_fields, spin2_fields,
        box_size_deg, config['namaster']
    )

    # Save power spectra
    print("\n  Saving power spectra...")
    np.savez(f'{output_dir}/power_spectra.npz', **results)
    print(f"    Saved to {output_dir}/power_spectra.npz")

    # Plot power spectra
    print("\n  Plotting power spectra...")
    plot_power_spectra(results, f'{output_dir}/power_spectra.png', dpi=dpi)

    # Summary
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - phi_field.png")
    print("  - scalar_field.png")
    print("  - vector_field.png")
    print("  - spin2_field_gamma1.png")
    print("  - spin2_field_gamma2.png")
    print("  - power_spectra.png")
    print("  - power_spectra.npz")
    if save_fields:
        print("  - *.npy (field arrays)")


if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_file)
