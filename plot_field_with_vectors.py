#!/usr/bin/env python
"""
Plot scalar fields (ψ, κ) with deflection rate α̇ arrows overlaid.

Usage:
    python plot_field_with_vectors.py [output_dir]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import sys
from pathlib import Path


def find_peak_region(field, size=128):
    """Find center of strongest peak for zoom."""
    # Smooth to find large-scale peak
    smoothed = gaussian_filter(np.abs(field), sigma=size//4)
    iy, ix = np.unravel_index(np.argmax(smoothed), smoothed.shape)
    # Keep within bounds
    half = size // 2
    ny, nx = field.shape
    ix = np.clip(ix, half, nx - half)
    iy = np.clip(iy, half, ny - half)
    return ix, iy


def plot_field_with_vectors(scalar_field, vx, vy, title, output_path,
                            cmap='RdBu_r', arrow_step=25, dpi=150,
                            zoom_center=None, zoom_size=None):
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
    zoom_center : tuple
        (x, y) center for zoom, or None for full field
    zoom_size : int
        Size of zoom region in pixels
    """
    fig, ax = plt.subplots(figsize=(10, 9))

    # Extract zoom region if specified
    if zoom_center is not None and zoom_size is not None:
        cx, cy = zoom_center
        half = zoom_size // 2
        slc = (slice(cy - half, cy + half), slice(cx - half, cx + half))
        scalar_field = scalar_field[slc]
        vx = vx[slc]
        vy = vy[slc]

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

    # Normalize arrows for visibility (scale by arrow spacing)
    mag = np.sqrt(vx_sub**2 + vy_sub**2)
    mag_max = np.percentile(mag, 90)
    arrow_scale = arrow_step * 0.7 / mag_max if mag_max > 0 else 1

    ax.quiver(X, Y, vx_sub * arrow_scale, vy_sub * arrow_scale,
              color='black', alpha=0.8, scale=1, scale_units='xy',
              width=0.004, headwidth=3.5, headlength=4)

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

    # Find interesting region (near strong κ peak)
    zoom_size = 128
    cx, cy = find_peak_region(kappa, size=zoom_size)
    print(f"\n  Zoom center: ({cx}, {cy}) - near strongest κ feature")

    # Full field views
    print("\nGenerating full-field plots...")
    plot_field_with_vectors(
        psi, vx, vy,
        title=r'Lensing Potential $\psi$ with Deflection Rate $\dot{\alpha}$',
        output_path=output_dir / 'psi_with_alpha_dot.png',
        cmap='RdBu_r',
        arrow_step=20
    )

    plot_field_with_vectors(
        kappa, vx, vy,
        title=r'Convergence $\kappa$ with Deflection Rate $\dot{\alpha}$',
        output_path=output_dir / 'kappa_with_alpha_dot.png',
        cmap='RdBu_r',
        arrow_step=20
    )

    # Zoomed views - arrows should clearly point toward κ > 0
    print("\nGenerating zoomed plots...")
    plot_field_with_vectors(
        psi.copy(), vx.copy(), vy.copy(),
        title=r'$\psi$ with $\dot{\alpha}$ (zoomed)',
        output_path=output_dir / 'psi_with_alpha_dot_zoom.png',
        cmap='RdBu_r',
        arrow_step=8,
        zoom_center=(cx, cy),
        zoom_size=zoom_size
    )

    plot_field_with_vectors(
        kappa.copy(), vx.copy(), vy.copy(),
        title=r'$\kappa$ with $\dot{\alpha}$ (zoomed)',
        output_path=output_dir / 'kappa_with_alpha_dot_zoom.png',
        cmap='RdBu_r',
        arrow_step=8,
        zoom_center=(cx, cy),
        zoom_size=zoom_size
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
