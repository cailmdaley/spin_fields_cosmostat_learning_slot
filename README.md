# Spin Fields from Gravitational Potential

Generate correlated spin-0, spin-1, and spin-2 fields from a gravitational lensing potential using ΛCDM cosmology.

## Physics

Four fields derived from gravitational potential Φ in harmonic space:

| Field | Spin | Definition | Physical Analog |
|-------|------|------------|-----------------|
| Φ | 0 | Lensing potential | Gravitational potential |
| δ | 0 | Aℓ²Φ̃ | Convergence (mass density) |
| α | 1 | Bℓe^(iφ)Φ̃ | Deflection angle derivative |
| γ | 2 | Cℓ²e^(2iφ)Φ̃ | Shear |

All fields are correlated through their common origin in Φ.

## Installation

```bash
pip install numpy matplotlib scipy pymaster pyyaml camb
```

## Usage

```bash
python main.py config_lcdm.yaml
```

Outputs go to `output_lcdm/`.

---

## Blind Fitting Challenge

The fields in `output_lcdm/` were generated for a cosmological parameter fitting exercise.

### Fixed Parameters (known)

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Hubble parameter | h | 0.70 |
| Baryon density | Ω_b | 0.05 |
| Spectral index | n_s | 0.96 |
| Source redshift | z_source | 1.0 |
| Grid size | N | 512 × 512 |
| Box size | θ | 10° |
| Amplitudes | A, B, C | 1.0, 1.0, 1.0 |

### Blind Parameters (to fit)

| Parameter | Symbol | True Value | Planck 2018 |
|-----------|--------|------------|-------------|
| Matter density | Ω_m | ? | 0.315 |
| S8 | S8 | ? | 0.832 |

where S8 = σ₈√(Ω_m/0.3).

### Data Files

```
output_lcdm/
├── power_spectra.npz    # Measured C_ℓ (use this for fitting)
├── phi_field.npy        # Lensing potential
├── scalar_field.npy     # Convergence δ
├── vector_field_x.npy   # Deflection α_x
├── vector_field_y.npy   # Deflection α_y
├── spin2_field_gamma1.npy  # Shear γ₁
└── spin2_field_gamma2.npy  # Shear γ₂
```

### Power Spectra in `power_spectra.npz`

```python
data = np.load('output_lcdm/power_spectra.npz')
ell = data['ell']           # Multipole bins
C_phi = data['phi_auto']    # C_ℓ^ΦΦ - fit this!
C_delta = data['scalar_auto']
C_gamma_EE = data['spin2_auto_EE']
# ... and all cross-spectra
```

### Theory Prediction

For a given (Ω_m, S8), compute the lensing convergence power spectrum C_ℓ^κκ using CAMB:

```python
from cosmology import get_lcdm_lensing_power_spectrum

config = {'S8': 0.8, 'Omega_m': 0.3, 'h': 0.7, 'n_s': 0.96, 'z_source': 1.0}
C_ell_theory = get_lcdm_lensing_power_spectrum(ell_grid, config)
```

The derived fields scale as:
- C_ℓ^δδ = A²ℓ⁴ C_ℓ^ΦΦ
- C_ℓ^αα = B²ℓ² C_ℓ^ΦΦ
- C_ℓ^γγ = C²ℓ⁴ C_ℓ^ΦΦ
