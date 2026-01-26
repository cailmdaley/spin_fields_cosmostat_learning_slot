# Spin Fields from Gravitational Potential

Generate four fields from a gravitational potential and compute their auto and cross power spectra using NaMaster.

## Physics Background

This code implements three derived fields from a gravitational potential Φ in harmonic (Fourier) space:

1. **Gravitational Potential**: Φ(ℓ⃗) - generated from a power spectrum
2. **Scalar field**: δ(ℓ⃗) = Aℓ²Φ(ℓ⃗)
3. **Vector field**: α⃗(ℓ⃗) = Bℓe^(iφ_ℓ)Φ(ℓ⃗) = Biℓ⃗Φ(ℓ⃗)
4. **Spin-2 field**: γ = Cℓ²e^(2iφ_ℓ)Φ(ℓ⃗) where φ_ℓ = tan⁻¹(ℓ₂/ℓ₁)

## Installation

Required packages:
```bash
pip install numpy matplotlib scipy pymaster pyyaml
```

Or with conda:
```bash
conda install numpy matplotlib scipy pyyaml
conda install -c conda-forge namaster
```

## Quick Start

1. Edit the configuration file [config.yaml](config.yaml) to set your parameters
2. Run the script:
```bash
python generate_fields.py
```

3. Check the `output/` directory for results

## Configuration

All parameters are set in [config.yaml](config.yaml):

### Grid Parameters
```yaml
grid:
  nx: 512                    # Grid size in x direction
  ny: 512                    # Grid size in y direction
  box_size_deg: 10.0         # Physical box size in degrees
```

### Field Amplitudes (A, B, C)
```yaml
amplitudes:
  A: 1.0                     # Scalar field amplitude
  B: 1.0                     # Vector field amplitude
  C: 1.0                     # Spin-2 field amplitude
```

**These are the key parameters you'll want to adjust!** They control the relative amplitudes of the three derived fields.

### Power Spectrum for Φ

You can use a power-law spectrum or provide a custom one:

**Option 1: Power Law**
```yaml
power_spectrum:
  type: "power_law"
  index: -2.0                # P(k) ~ k^index
```

**Option 2: Custom Power Spectrum**
```yaml
power_spectrum:
  type: "custom"
  custom_file: "my_power_spectrum.txt"  # Two columns: k, P(k)
```

### NaMaster Settings
```yaml
namaster:
  n_bins: 20                 # Number of ell bins
  ell_min: 2                 # Minimum ell
  ell_max: null              # Maximum ell (null = auto)
  use_log_bins: true         # Use logarithmic binning
```

### Output Settings
```yaml
output:
  output_dir: "./output"     # Directory for output files
  save_fields: true          # Save field arrays as .npy files
  dpi: 150                   # DPI for PNG figures
```

## Usage

### Basic Usage

```bash
python generate_fields.py
```

This uses the default [config.yaml](config.yaml) file.

### Use a Different Config File

```bash
python generate_fields.py my_config.yaml
```

### Workflow

The script follows these steps:

1. **Generate Phi from power spectrum** - Creates the gravitational potential as a Gaussian random field
2. **Generate three derived fields** - Applies the transformations in Fourier space
3. **Create PNGs** - Visualizes all four fields
4. **Calculate power spectra** - Computes all auto and cross-correlations with NaMaster

## Output Files

All output files are saved to the directory specified in the config (default: `./output/`):

### Field Visualizations (PNG)
- `phi_field.png` - Gravitational potential Φ
- `scalar_field.png` - Scalar field δ
- `vector_field.png` - Vector field α (magnitude with arrows)
- `spin2_field_gamma1.png` - Spin-2 field component γ₁
- `spin2_field_gamma2.png` - Spin-2 field component γ₂

### Power Spectra
- `power_spectra.png` - All auto and cross power spectra
- `power_spectra.npz` - Power spectra data (can be loaded with `np.load()`)

### Field Arrays (if save_fields: true)
- `phi_field.npy`
- `scalar_field.npy`
- `vector_field_x.npy`, `vector_field_y.npy`
- `spin2_field_gamma1.npy`, `spin2_field_gamma2.npy`

## Power Spectra Computed

The code computes all possible auto and cross power spectra:

### Auto-correlations
- Φ × Φ
- δ × δ (scalar)
- α × α (vector, EE/EB/BE/BB modes)
- γ × γ (spin-2, EE/EB/BE/BB modes)

### Cross-correlations
- Φ × δ
- Φ × α (E and B modes)
- Φ × γ (E and B modes)
- δ × α (E and B modes)
- δ × γ (E and B modes)
- α × γ (EE/EB/BE/BB modes)

## Implementation Details

### Field Generation

All fields are generated in Fourier space and then transformed to real space:

1. **Φ**: Generated from a power spectrum P(ℓ) with random Gaussian phases
2. **δ**: Multiply Φ(ℓ) by ℓ² in Fourier space
3. **α**: Apply gradient operator iℓ⃗ to Φ(ℓ)
4. **γ**: Multiply Φ(ℓ) by ℓ²e^(2iφ_ℓ), which creates spin-2 structure

### NaMaster Analysis

NaMaster computes pseudo-C_ℓ power spectra with:
- Proper treatment of mode coupling
- E/B mode decomposition for spin-1 and spin-2 fields
- Configurable binning schemes (linear or logarithmic)

### Coordinate Conventions

- ℓ⃗ = (ℓₓ, ℓᵧ): 2D wavevector in Fourier space
- ℓ = |ℓ⃗| = √(ℓₓ² + ℓᵧ²): magnitude
- φ_ℓ = arctan(ℓᵧ/ℓₓ): angle in Fourier space

## Example: Loading and Analyzing Results

```python
import numpy as np
import matplotlib.pyplot as plt

# Load power spectra
data = np.load('output/power_spectra.npz')
ell = data['ell']
phi_auto = data['phi_auto']
scalar_auto = data['scalar_auto']

# Plot
plt.loglog(ell, phi_auto, label='Φ auto')
plt.loglog(ell, scalar_auto, label='δ auto')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell$')
plt.legend()
plt.show()

# Load field arrays
phi = np.load('output/phi_field.npy')
scalar = np.load('output/scalar_field.npy')

# Do further analysis...
```

## Tips

- Start with the default config and adjust A, B, C to explore different field amplitude ratios
- Use `power_spectrum_index: -2.0` for approximately white noise in real space
- For larger grids, you may want to reduce `n_bins` or increase `ell_min` for faster computation
- The vector field visualization shows both magnitude (colormap) and direction (arrows)

## References

- **NaMaster**: https://namaster.readthedocs.io/
- **Spin-weighted fields**: Kamionkowski et al. (1997), Phys. Rev. D 55, 7368

## Troubleshooting

**Import error for pymaster**: Install with `conda install -c conda-forge namaster`

**Out of memory**: Reduce `nx` and `ny` in the config file

**Plots look empty**: Check that A, B, C are not too small relative to the power spectrum normalization
