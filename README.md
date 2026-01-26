# Spin Fields from Gravitational Potential

Generate correlated spin-0, spin-1, and spin-2 fields from a gravitational lensing potential using ΛCDM cosmology.

## Physics

Four fields derived from lensing potential ψ in harmonic space:

| Field | Spin | Definition | Physical Meaning |
|-------|------|------------|------------------|
| ψ | 0 | Lensing potential | Gravitational potential (line-of-sight integrated) |
| κ | 0 | ½ℓ²ψ̃ | Convergence (projected mass density) |
| α̇ | 1 | ℓe^(iφ)ψ̃ | Deflection rate dα/dt (time derivative of deflection angle) |
| γ | 2 | ½ℓ²e^(2iφ)ψ̃ | Shear (tidal distortion) |

All fields are correlated through their common origin in ψ.

### Why α̇ (deflection rate)?

As structure grows, the lensing deflection angle evolves: dα/dt ≠ 0. This spin-1 field would be observable via quasar proper motions with μas/yr precision — futuristic but physically meaningful. It scales as ℓ¹ (same as static deflection α = ∇ψ), with amplitude proportional to the growth rate f×H.

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
| Box size | θ | 50° (2500 sq deg) |
| Amplitudes | A, B, C | 0.5, 1.0, 0.5 |

### Blind Parameters (to fit)

| Parameter | Symbol | True Value | Planck 2018 |
|-----------|--------|------------|-------------|
| Matter density | Ω_m | ? | 0.315 |
| S8 | S8 | ? | 0.832 |

where S8 = σ₈√(Ω_m/0.3).

### Data Files

```
output_lcdm/
├── power_spectra.npz       # Measured C_ℓ (use this for fitting)
├── phi_field.npy           # Lensing potential ψ
├── scalar_field.npy        # Convergence κ
├── vector_field_x.npy      # Deflection rate α̇_x
├── vector_field_y.npy      # Deflection rate α̇_y
├── spin2_field_gamma1.npy  # Shear γ₁
└── spin2_field_gamma2.npy  # Shear γ₂
```

### Power Spectra in `power_spectra.npz`

```python
data = np.load('output_lcdm/power_spectra.npz')
ell = data['ell']              # Multipole bins
C_psi = data['phi_auto']       # C_ℓ^ψψ (potential)
C_kappa = data['scalar_auto']  # C_ℓ^κκ (convergence) - fit this!
C_gamma_EE = data['spin2_auto_EE']
# ... and all cross-spectra
```

### Theory Prediction

CAMB gives C_ℓ^κκ directly. For the potential: C_ℓ^ψψ = 4 C_ℓ^κκ / ℓ⁴

```python
from cosmology import get_lcdm_lensing_power_spectrum

config = {'S8': 0.8, 'Omega_m': 0.3, 'h': 0.7, 'n_s': 0.96, 'z_source': 1.0}
C_ell_kappa = get_lcdm_lensing_power_spectrum(ell_grid, config)
```

The derived fields scale as:
- C_ℓ^κκ = A²ℓ⁴ C_ℓ^ψψ = C_ℓ^κκ (when A=0.5)
- C_ℓ^α̇α̇ = B²ℓ² C_ℓ^ψψ
- C_ℓ^γγ = C²ℓ⁴ C_ℓ^ψψ
