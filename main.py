"""
Main script to generate fields and compute power spectra.
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

from generate_fields_lib import (
    generate_phi_from_power_spectrum,
    generate_scalar_field,
    generate_vector_field,
    generate_spin2_field
)
from measure_power_spectra import compute_all_power_spectra


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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


def compute_theoretical_power_spectra(ell, P_phi, A, B, C):
    """
    Compute theoretical power spectra from the relationships.

    Given:
    - C_ℓ^{ΦΦ} = P_Φ(ℓ) (input power spectrum)
    - δ = Aℓ²Φ  =>  C_ℓ^{δδ} = A² ℓ⁴ P_Φ(ℓ)
    - α = Biℓ Φ  =>  C_ℓ^{αα} = B² ℓ² P_Φ(ℓ)
    - γ = Cℓ²Φ  =>  C_ℓ^{γγ} = C² ℓ⁴ P_Φ(ℓ)

    Cross-correlations:
    - C_ℓ^{Φδ} = A ℓ² P_Φ(ℓ)
    - C_ℓ^{Φα} = B ℓ P_Φ(ℓ)
    - C_ℓ^{Φγ} = C ℓ² P_Φ(ℓ)
    - C_ℓ^{δα} = AB ℓ³ P_Φ(ℓ)
    - C_ℓ^{δγ} = AC ℓ⁴ P_Φ(ℓ)
    - C_ℓ^{αγ} = BC ℓ³ P_Φ(ℓ)
    """
    theory = {}

    # Auto-correlations
    theory['phi_auto'] = P_phi
    theory['scalar_auto'] = A**2 * ell**4 * P_phi
    theory['vector_auto'] = B**2 * ell**2 * P_phi
    theory['spin2_auto'] = C**2 * ell**4 * P_phi

    # Cross-correlations
    theory['phi_scalar'] = A * ell**2 * P_phi
    theory['phi_vector'] = B * ell * P_phi
    theory['phi_spin2'] = C * ell**2 * P_phi
    theory['scalar_vector'] = A * B * ell**3 * P_phi
    theory['scalar_spin2'] = A * C * ell**4 * P_phi
    theory['vector_spin2'] = B * C * ell**3 * P_phi

    return theory


def plot_power_spectra(results, output_path, config, mode='E', dpi=150):
    """
    Plot all power spectra in 4x4 grid with theory curves.

    Layout: Auto-correlations on diagonal, cross-correlations on upper triangle.

    Parameters:
    -----------
    results : dict
        Power spectra results from measurements
    mode : str
        'E' for E-mode power spectra, 'B' for B-mode power spectra
    output_path : str
        Path to save PNG
    config : dict
        Configuration with A, B, C parameters
    dpi : int
        DPI for output
    """
    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    ell = results['ell']

    # Set title based on mode
    if mode == 'E':
        mode_suffix = 'EE'
        mode_label = 'E-mode'
    elif mode == 'B':
        mode_suffix = 'BB'
        mode_label = 'B-mode'
    else:  # mode == 'EB'
        mode_suffix = 'EB'
        mode_label = 'E×B Cross'
    fig.suptitle(f'{mode_label} Power Spectra', fontsize=16, fontweight='bold', y=0.995)

    # Get amplitudes for theory
    A = config['amplitudes']['A']
    B = config['amplitudes']['B']
    C = config['amplitudes']['C']

    # For comparison: what did we actually measure for Phi?
    P_phi_measured = results['phi_auto']

    # Calculate theoretical P_Phi from input power spectrum parameters
    ps_type = config['power_spectrum'].get('type', 'power_law')
    if ps_type == 'power_law':
        ps_index = config['power_spectrum']['index']
        ps_amplitude = config['power_spectrum']['amplitude']
        P_phi_theory = ps_amplitude * ell**ps_index
        print(f"\n  Normalization check:")
        print(f"    Theory/Measured ratio for Phi: {np.median(P_phi_theory / P_phi_measured):.2e}")
    elif ps_type == 'lcdm':
        # Compute LCDM theory curve from CAMB at binned ell values
        # CAMB gives C_ℓ^κκ, convert to C_ℓ^ψψ = 4 C_ℓ^κκ / ℓ⁴
        from cosmology import get_lcdm_lensing_power_spectrum
        ell_2d = ell.reshape(-1, 1)
        C_ell_kappa = get_lcdm_lensing_power_spectrum(ell_2d, config['power_spectrum']).flatten()
        P_phi_theory = 4 * C_ell_kappa / ell**4  # C_ℓ^ψψ (potential)
        print(f"\n  LCDM theory from CAMB (C_ℓ^ψψ = 4 C_ℓ^κκ / ℓ⁴):")
        print(f"    Theory/Measured ratio for Phi (ψ): {np.median(P_phi_theory / P_phi_measured):.2e}")
    else:
        # For other types, use measured as fallback
        P_phi_theory = P_phi_measured
        print(f"\n  Using measured Phi as reference (unknown PS type):")

    # Check the other fields too
    scalar_theory = A**2 * ell**4 * P_phi_measured  # Use measured Phi
    vector_theory = B**2 * ell**2 * P_phi_measured
    spin2_theory = C**2 * ell**4 * P_phi_measured

    print(f"    Theory/Measured ratio for Scalar (δ): {np.median(scalar_theory / results['scalar_auto']):.2e}")
    print(f"    Theory/Measured ratio for Vector (α̇): {np.median(vector_theory / results['vector_auto_EE']):.2e}")
    print(f"    Vector B-mode / E-mode ratio: {np.median(results['vector_auto_BB'] / results['vector_auto_EE']):.2e}")
    print(f"    Vector EB / E-mode ratio: {np.median(np.abs(results['vector_auto_EB']) / results['vector_auto_EE']):.2e}")
    print(f"    Vector B-mode (median): {np.median(results['vector_auto_BB']):.2e}, E-mode (median): {np.median(results['vector_auto_EE']):.2e}")
    print(f"    Theory/Measured ratio for Spin-2 (γ): {np.median(spin2_theory / results['spin2_auto_EE']):.2e}")
    print(f"    Spin-2 B-mode / E-mode ratio: {np.median(results['spin2_auto_BB'] / results['spin2_auto_EE']):.2e}")
    print(f"    Spin-2 EB / E-mode ratio: {np.median(np.abs(results['spin2_auto_EB']) / results['spin2_auto_EE']):.2e}")
    print(f"    Spin-2 B-mode (median): {np.median(results['spin2_auto_BB']):.2e}, E-mode (median): {np.median(results['spin2_auto_EE']):.2e}")

    # Compute theoretical predictions
    theory = compute_theoretical_power_spectra(ell, P_phi_theory, A, B, C)

    # Define field names
    fields = ['ψ', 'κ', 'α̇', 'γ']

    # Row 0, Col 0: Φ auto (scalar, only in E-mode plot)
    ax = axes[0, 0]
    if mode == 'E':
        ax.loglog(ell, results['phi_auto'], 'o', ms=6, label='Measured', color='C0', alpha=0.7)
        ax.loglog(ell, theory['phi_auto'], '-', lw=2, label='Theory', color='black', alpha=0.8)
        ax.set_ylabel(r'$C_\ell^{\psi\psi}$', fontsize=11)
        ax.set_title(r'$C_\ell^{\psi\psi}$', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=9)
        ax.set_xticklabels([])
    else:
        ax.axis('off')

    # Row 0, Col 1: Φ-δ cross (scalar, only in E-mode plot)
    ax = axes[0, 1]
    if mode == 'E':
        ax.loglog(ell, np.abs(results['phi_scalar_cross']), 'o', ms=6, label='Measured', color='C1', alpha=0.7)
        ax.loglog(ell, theory['phi_scalar'], '-', lw=2, label='Theory', color='black', alpha=0.8)
        ax.set_title(r'$C_\ell^{\psi\kappa} = A\ell^2 P_\psi(\ell)$', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=9)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        ax.axis('off')

    # Row 0, Col 2: Φ-α cross
    ax = axes[0, 2]
    if mode == 'EB':
        # No EB mode for scalar × spin cross (EB only exists for spin × spin)
        ax.axis('off')
    else:
        data_key = f'phi_vector_{mode[0]}'  # 'E' -> 'phi_vector_E', 'B' -> 'phi_vector_B'
        ax.loglog(ell, np.abs(results[data_key]), 'o', ms=6, label='Measured', color='C2', alpha=0.7)
        if mode == 'E':
            ax.loglog(ell, theory['phi_vector'], '-', lw=2, label='Theory', color='black', alpha=0.8)
            title = r'$C_\ell^{\psi\dot{\alpha}} = B\ell P_\psi(\ell)$'
        else:
            title = r'$C_\ell^{\psi\dot{\alpha}}$ (B-mode, should be $\approx 0$)'
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=9)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Row 0, Col 3: Φ-γ cross
    ax = axes[0, 3]
    if mode == 'EB':
        ax.axis('off')
    else:
        data_key = f'phi_spin2_{mode[0]}'
        ax.loglog(ell, np.abs(results[data_key]), 'o', ms=6, label='Measured', color='C3', alpha=0.7)
        if mode == 'E':
            ax.loglog(ell, theory['phi_spin2'], '-', lw=2, label='Theory', color='black', alpha=0.8)
            title = r'$C_\ell^{\psi\gamma} = C\ell^2 P_\psi(\ell)$'
        else:
            title = r'$C_\ell^{\psi\gamma}$ (B-mode, should be $\approx 0$)'
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=9)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Row 1, Col 0: Empty (lower triangle)
    axes[1, 0].axis('off')

    # Row 1, Col 1: δ auto (scalar, only in E-mode plot)
    ax = axes[1, 1]
    if mode == 'E':
        ax.loglog(ell, results['scalar_auto'], 'o', ms=6, label='Measured', color='C4', alpha=0.7)
        ax.loglog(ell, theory['scalar_auto'], '-', lw=2, label='Theory', color='black', alpha=0.8)
        ax.set_ylabel(r'$C_\ell^{\kappa\kappa}$', fontsize=11)
        ax.set_title(r'$C_\ell^{\kappa\kappa} = A^2\ell^4 P_\psi(\ell)$', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=9)
        ax.set_xticklabels([])
    else:
        ax.axis('off')

    # Row 1, Col 2: δ-α cross
    ax = axes[1, 2]
    if mode == 'EB':
        ax.axis('off')
    else:
        data_key = f'scalar_vector_{mode[0]}'
        ax.loglog(ell, np.abs(results[data_key]), 'o', ms=6, label='Measured', color='C5', alpha=0.7)
        if mode == 'E':
            ax.loglog(ell, theory['scalar_vector'], '-', lw=2, label='Theory', color='black', alpha=0.8)
            title = r'$C_\ell^{\kappa\dot{\alpha}} = AB\ell^3 P_\psi(\ell)$'
        else:
            title = r'$C_\ell^{\kappa\dot{\alpha}}$ (B-mode, should be $\approx 0$)'
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=9)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Row 1, Col 3: δ-γ cross
    ax = axes[1, 3]
    if mode == 'EB':
        ax.axis('off')
    else:
        data_key = f'scalar_spin2_{mode[0]}'
        ax.loglog(ell, np.abs(results[data_key]), 'o', ms=6, label='Measured', color='C6', alpha=0.7)
        if mode == 'E':
            ax.loglog(ell, theory['scalar_spin2'], '-', lw=2, label='Theory', color='black', alpha=0.8)
            title = r'$C_\ell^{\kappa\gamma} = AC\ell^4 P_\psi(\ell)$'
        else:
            title = r'$C_\ell^{\kappa\gamma}$ (B-mode, should be $\approx 0$)'
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=9)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Row 2, Col 0: Empty (lower triangle)
    axes[2, 0].axis('off')

    # Row 2, Col 1: Empty (lower triangle)
    axes[2, 1].axis('off')

    # Row 2, Col 2: α auto
    ax = axes[2, 2]
    ax.loglog(ell, np.abs(results[f'vector_auto_{mode_suffix}']), 'o', ms=6, label=f'Measured ({mode_suffix})', color='C7', alpha=0.7)
    if mode == 'E':
        ax.loglog(ell, theory['vector_auto'], '-', lw=2, label='Theory', color='black', alpha=0.8)
        title = r'$C_\ell^{\dot{\alpha}\dot{\alpha}} = B^2\ell^2 P_\psi(\ell)$'
    elif mode == 'B':
        title = r'$C_\ell^{\dot{\alpha}\dot{\alpha}}$ (BB, should be $\approx 0$)'
    else:  # mode == 'EB'
        title = r'$C_\ell^{\dot{\alpha}\dot{\alpha}}$ (EB, should be $\approx 0$)'
    ax.set_ylabel(r'$C_\ell^{\dot{\alpha}\dot{\alpha}}$', fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=9)
    ax.set_xticklabels([])

    # Row 2, Col 3: α-γ cross
    ax = axes[2, 3]
    data_key = f'vector_spin2_{mode_suffix}'
    ax.loglog(ell, np.abs(results[data_key]), 'o', ms=6, label='Measured', color='C8', alpha=0.7)
    if mode == 'E':
        ax.loglog(ell, theory['vector_spin2'], '-', lw=2, label='Theory', color='black', alpha=0.8)
        title = r'$C_\ell^{\dot{\alpha}\gamma} = BC\ell^3 P_\psi(\ell)$'
    elif mode == 'B':
        title = r'$C_\ell^{\dot{\alpha}\gamma}$ (BB, should be $\approx 0$)'
    else:  # mode == 'EB'
        title = r'$C_\ell^{\dot{\alpha}\gamma}$ (EB, should be $\approx 0$)'
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=9)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Row 3, Col 0: Empty (lower triangle)
    axes[3, 0].axis('off')

    # Row 3, Col 1: Empty (lower triangle)
    axes[3, 1].axis('off')

    # Row 3, Col 2: Empty (lower triangle)
    axes[3, 2].axis('off')

    # Row 3, Col 3: γ auto
    ax = axes[3, 3]
    ax.loglog(ell, np.abs(results[f'spin2_auto_{mode_suffix}']), 'o', ms=6, label=f'Measured ({mode_suffix})', color='C9', alpha=0.7)
    if mode == 'E':
        ax.loglog(ell, theory['spin2_auto'], '-', lw=2, label='Theory', color='black', alpha=0.8)
        title = r'$C_\ell^{\gamma\gamma} = C^2\ell^4 P_\psi(\ell)$'
    elif mode == 'B':
        title = r'$C_\ell^{\gamma\gamma}$ (BB, should be $\approx 0$)'
    else:  # mode == 'EB'
        title = r'$C_\ell^{\gamma\gamma}$ (EB, should be $\approx 0$)'
    ax.set_xlabel(r'$\ell$', fontsize=11)
    ax.set_ylabel(r'$C_\ell^{\gamma\gamma}$', fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=9)

    # Set x-labels for bottom row
    axes[3, 3].set_xlabel(r'$\ell$', fontsize=11)

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

    print("  Generating deflection rate field α̇(ℓ) = Bℓe^{iφ}ψ(ℓ)...")
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

    plot_field(scalar_field, r'Convergence $\kappa = A\ell^2\psi$',
              f'{output_dir}/scalar_field.png', cmap='RdBu_r', dpi=dpi)

    plot_field(vector_fields, r'Deflection Rate $\dot{\alpha} = B\ell e^{i\phi}\psi$ (magnitude + arrows)',
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
    print("STEP 4: Calculate power spectra")
    print("="*60)

    results = compute_all_power_spectra(
        phi_real, scalar_field, vector_fields, spin2_fields,
        box_size_deg, config['namaster']
    )

    # Save power spectra
    print("\n  Saving power spectra...")
    np.savez(f'{output_dir}/power_spectra.npz', **results)
    print(f"    Saved to {output_dir}/power_spectra.npz")

    # Plot power spectra - E, B, and EB modes separately
    print("\n  Plotting power spectra...")
    plot_power_spectra(results, f'{output_dir}/power_spectra_E.png', config, mode='E', dpi=dpi)
    plot_power_spectra(results, f'{output_dir}/power_spectra_B.png', config, mode='B', dpi=dpi)
    plot_power_spectra(results, f'{output_dir}/power_spectra_EB.png', config, mode='EB', dpi=dpi)

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
    print("  - power_spectra_E.png")
    print("  - power_spectra_B.png")
    print("  - power_spectra_EB.png")
    print("  - power_spectra.npz")
    if save_fields:
        print("  - *.npy (field arrays)")


if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_file)
