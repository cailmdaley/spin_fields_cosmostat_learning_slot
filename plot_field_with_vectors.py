#!/usr/bin/env python
"""
Plot scalar fields (ψ, κ) with deflection rate α̇ arrows overlaid.

Usage:
    python plot_field_with_vectors.py [output_dir]
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path


def plot_field_with_vectors(scalar_field, vx, vy, title, output_path,
                            cmap='RdBu_r', arrow_step=25, dpi=150):
    """
    Plot a scalar field with vector arrows overlaid.

    Parameters
    ----------
    scalar_field : ndarray
        2D scalar field to plot as background
    vx, vy : ndarray
        Vector field components for arrows
    title : str
        Plot title
    output_path : str
        Path to save the figure
    cmap : str
        Colormap for scalar field
    arrow_step : int
        Subsample arrows every N pixels
    dpi : int
        Output resolution
    """
    fig, ax = plt.subplots(figsize=(10, 9))

    # Plot scalar field
    vmax = np.percentile(np.abs(scalar_field), 99)
    im = ax.imshow(scalar_field, cmap=cmap, origin='lower', aspect='equal',
                   vmin=-vmax, vmax=vmax)

    # Add vector arrows (subsampled)
    ny, nx = scalar_field.shape
    x = np.arange(arrow_step//2, nx, arrow_step)
    y = np.arange(arrow_step//2, ny, arrow_step)
    X, Y = np.meshgrid(x, y)

    # Subsample vector field
    vx_sub = vx[arrow_step//2::arrow_step, arrow_step//2::arrow_step]
    vy_sub = vy[arrow_step//2::arrow_step, arrow_step//2::arrow_step]

    # Normalize arrows for visibility
    mag = np.sqrt(vx_sub**2 + vy_sub**2)
    mag_max = np.percentile(mag, 95)
    scale = arrow_step * 0.8 / mag_max if mag_max > 0 else 1

    ax.quiver(X, Y, vx_sub * scale, vy_sub * scale,
              color='black', alpha=0.7, scale=1, scale_units='xy',
              width=0.003, headwidth=4, headlength=5)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('x [pixels]', fontsize=12)
    ax.set_ylabel('y [pixels]', fontsize=12)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'output_lcdm'
    output_dir = Path(output_dir)

    # Load fields
    print(f"Loading fields from {output_dir}/...")
    psi = np.load(output_dir / 'phi_field.npy')
    kappa = np.load(output_dir / 'scalar_field.npy')
    vx = np.load(output_dir / 'vector_field_x.npy')
    vy = np.load(output_dir / 'vector_field_y.npy')

    print(f"  ψ shape: {psi.shape}, range: [{psi.min():.2e}, {psi.max():.2e}]")
    print(f"  κ shape: {kappa.shape}, range: [{kappa.min():.2e}, {kappa.max():.2e}]")
    print(f"  α̇ magnitude range: [{np.sqrt(vx**2+vy**2).min():.2e}, {np.sqrt(vx**2+vy**2).max():.2e}]")

    # Plot ψ with α̇ arrows
    plot_field_with_vectors(
        psi, vx, vy,
        title=r'Lensing Potential $\psi$ with Deflection Rate $\dot{\alpha}$',
        output_path=output_dir / 'psi_with_alpha_dot.png',
        cmap='RdBu_r',
        arrow_step=20
    )

    # Plot κ with α̇ arrows
    plot_field_with_vectors(
        kappa, vx, vy,
        title=r'Convergence $\kappa$ with Deflection Rate $\dot{\alpha}$',
        output_path=output_dir / 'kappa_with_alpha_dot.png',
        cmap='RdBu_r',
        arrow_step=20
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
