#!/usr/bin/env python3
"""
Quick check: compare field variance to integral of power spectrum.

Var(f) = ∫ d²ℓ/(2π)² C_ℓ = ∫ ℓ dℓ/(2π) C_ℓ  (for isotropic C_ℓ)
"""

import numpy as np
from scipy.integrate import quad

from generate_fields_lib import (
    generate_phi_from_power_spectrum,
    generate_vector_field,
)
from power_spectrum import velocity_power_spectrum

NX, NY = 512, 512
BOX_SIZE_DEG = 50.0
AMPLITUDE = 1.0
INDEX = -5.0
C_ALPHA = 1.0

# Generate
power_config = {"type": "power_law", "amplitude": AMPLITUDE, "index": INDEX}
phi_real, phi_fourier = generate_phi_from_power_spectrum(NX, NY, BOX_SIZE_DEG, power_config, seed=42)
v1, v2 = generate_vector_field(phi_fourier, C_ALPHA, NX, NY, BOX_SIZE_DEG)

# Measured variance
var_v = v1.var() + v2.var()  # Total variance |v|² = v1² + v2²
print(f"Measured Var(v): {var_v:.6e}")

# Theory variance from power spectrum
box_rad = np.deg2rad(BOX_SIZE_DEG)
ell_min = 2 * np.pi / box_rad
ell_max = np.pi * NX / box_rad

def integrand(ell):
    C_vv = velocity_power_spectrum(ell, AMPLITUDE, INDEX, C_ALPHA)
    return ell / (2 * np.pi) * C_vv

var_theory, _ = quad(integrand, ell_min, ell_max)
print(f"Theory Var(v):   {var_theory:.6e}")
print(f"Ratio meas/theory: {var_v / var_theory:.4f}")
print(f"4/π = {4/np.pi:.4f}")

# Also check power spectrum directly
from measure_power_spectra import compute_spin_power_auto
config = {'n_bins': 20, 'ell_min': 50, 'ell_max': 2000, 'use_log_bins': True}
ell, vv_EE, vv_BB, _ = compute_spin_power_auto((v1, v2), BOX_SIZE_DEG, config, spin=1)

# Compare to pure theory (not Φ-based)
C_vv_theory = velocity_power_spectrum(ell, AMPLITUDE, INDEX, C_ALPHA)

print(f"\nPower spectrum comparison (BB mode):")
print(f"  Mean C_vv_meas / C_vv_theory: {np.mean(vv_BB / C_vv_theory):.4f}")

print(f"\nIntegration limits:")
print(f"  ell_min = {ell_min:.2f}")
print(f"  ell_max = {ell_max:.2f}")

# Try integrating the MEASURED power spectrum
from scipy.interpolate import interp1d
C_interp = interp1d(ell, vv_BB, bounds_error=False, fill_value=0)
def integrand_meas(l):
    return l / (2 * np.pi) * C_interp(l)

var_from_measured_Cl, _ = quad(integrand_meas, ell.min(), ell.max())
print(f"\nVar from measured C_ℓ: {var_from_measured_Cl:.6e}")
print(f"Actual field variance:  {var_v:.6e}")
print(f"Ratio: {var_v / var_from_measured_Cl:.4f}")

# Breakdown by ℓ range
print(f"\nVariance breakdown by ℓ range (theory C_vv):")
ranges = [(7, 50), (50, 200), (200, 1000), (1000, 1843)]
for l_lo, l_hi in ranges:
    var_part, _ = quad(integrand, l_lo, l_hi)
    print(f"  ℓ = {l_lo:4d} - {l_hi:4d}: {var_part:.4e} ({100*var_part/var_theory:.1f}%)")

# Check: is measured variance consistent with sum over FFT modes?
# Var = (1/N^4) × Σ|FFT|² × (L/N)^2 for proper normalization
v_complex = v1 + 1j * v2
v_fft = np.fft.fft2(v_complex)
# Power per mode = |FFT|² / N² (for each component of complex field)
var_from_fft = np.sum(np.abs(v_fft)**2) / (NX * NY)**2
# Convert to physical units
var_from_fft *= (box_rad / NX)**2 * (box_rad / NY)**2
print(f"\nVar from FFT sum: {np.sum(np.abs(v_fft)**2) / (NX*NY)**2:.6e} (raw)")
print(f"Actual variance:  {var_v:.6e}")

# Check: compare |α̃|² directly to theory C_vv at each mode
print(f"\n--- Direct FFT vs Theory comparison ---")
kx = 2 * np.pi * np.fft.fftfreq(NX, d=box_rad/NX)
ky = 2 * np.pi * np.fft.fftfreq(NY, d=box_rad/NY)
kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
ell_grid = np.sqrt(kx_grid**2 + ky_grid**2)

# Measured power: |FFT|² normalized to give C_ℓ
# For numpy FFT: ⟨|FFT|²⟩ = N⁴ × (Δx)² × C_ℓ where Δx = L/N
# So C_ℓ = |FFT|² / (N⁴ × (L/N)²) = |FFT|² × N² / (N⁴ × L²) = |FFT|² / (N² × L²)
alpha_x_fft = np.fft.fft2(v1)
alpha_y_fft = np.fft.fft2(v2)
power_measured_2d = (np.abs(alpha_x_fft)**2 + np.abs(alpha_y_fft)**2) * box_rad**2 / (NX * NY)**2

C_vv_theory_2d = velocity_power_spectrum(ell_grid, AMPLITUDE, INDEX, C_ALPHA)
C_vv_theory_2d[0, 0] = 0  # avoid DC

# Compare in a few ℓ bins
for l_lo, l_hi in [(50, 100), (100, 200), (500, 1000)]:
    mask = (ell_grid >= l_lo) & (ell_grid < l_hi)
    if mask.sum() > 0:
        ratio = np.mean(power_measured_2d[mask]) / np.mean(C_vv_theory_2d[mask])
        print(f"  ℓ = {l_lo}-{l_hi}: measured/theory = {ratio:.4f}")

# Also check Φ variance
print(f"\n--- Φ (potential) variance check ---")
var_phi_measured = phi_real.var()
# Theory: Var(Φ) = ∫ ℓ dℓ/(2π) P_Φ(ℓ) where P_Φ = A ℓ^n
def integrand_phi(ell):
    return ell / (2 * np.pi) * AMPLITUDE * ell**INDEX
var_phi_theory, _ = quad(integrand_phi, ell_min, ell_max)
print(f"Measured Var(Φ): {var_phi_measured:.6e}")
print(f"Theory Var(Φ):   {var_phi_theory:.6e}")
print(f"Ratio: {var_phi_measured / var_phi_theory:.4f}")

# Check: variance from FFT power spectrum
phi_fft = np.fft.fft2(phi_real)
# Parseval: Var = (1/N⁴) Σ|FFT|² for numpy convention
var_phi_from_fft = np.sum(np.abs(phi_fft)**2) / (NX * NY)**2
print(f"Var(Φ) from FFT Parseval: {var_phi_from_fft:.6e}")

# Correct formula: Var = (1/L²) Σ_{modes} C_ℓ = (1/L²) Σ C_ℓ
# where the sum is over all N² modes (each with spacing Δk = 2π/L)
P_phi_2d = AMPLITUDE * ell_grid**INDEX
P_phi_2d[0, 0] = 0
var_from_theory_sum = np.sum(P_phi_2d) / box_rad**2
print(f"Var from theory mode sum: {var_from_theory_sum:.6e}")
print(f"Ratio field/theory: {var_phi_measured / var_from_theory_sum:.4f}")

# Key test: measured FFT power should sum to give field variance
# The power spectrum C_ℓ is defined as ⟨|f̃|²⟩ but needs proper normalization
# For numpy: |FFT|² / N⁴ × L² = C_ℓ in physical units
# Then: Var = ∫ ℓ dℓ/(2π) C_ℓ (continuous) ≈ (1/2π) Σ ℓ_k C_ℓ_k × Δℓ (discrete)
# But for a 2D box: Var = ∫ d²ℓ/(2π)² C_ℓ = (1/L²) Σ_k C_ℓ_k (sum over modes)

# Measured C_ℓ on 2D grid
C_phi_measured_2d = np.abs(phi_fft)**2 * box_rad**2 / (NX * NY)**2
C_phi_measured_2d[0, 0] = 0  # exclude DC
var_from_measured_Cl_sum = np.sum(C_phi_measured_2d) / box_rad**2
print(f"Var from measured C_ℓ sum: {var_from_measured_Cl_sum:.6e}")
print(f"Should equal field variance: {var_phi_measured:.6e}")
print(f"Ratio: {var_from_measured_Cl_sum / var_phi_measured:.4f}")

# What theory predicts for the mode sum (not integral):
# Var = (1/L²) Σ_k C_ℓ_k where sum is over N² discrete modes
# For C_ℓ = A ℓ^n, the sum is Σ A ℓ_k^n
P_phi_theory_2d = AMPLITUDE * ell_grid**INDEX
P_phi_theory_2d[0, 0] = 0
var_theory_discrete = np.sum(P_phi_theory_2d) / box_rad**2
print(f"\nTheory discrete sum: {var_theory_discrete:.6e}")
print(f"Measured variance:   {var_phi_measured:.6e}")
print(f"Ratio meas/theory (discrete): {var_phi_measured / var_theory_discrete:.4f}")

# But this depends on the ell grid - let me compare at same ell values
print(f"\nMode-by-mode comparison at a few ell values:")
for target_ell in [50, 100, 500]:
    mask = (ell_grid > target_ell - 5) & (ell_grid < target_ell + 5)
    if mask.sum() > 0:
        meas_mean = np.mean(C_phi_measured_2d[mask])
        theory_mean = np.mean(P_phi_theory_2d[mask])
        print(f"  ℓ≈{target_ell}: meas/theory = {meas_mean/theory_mean:.4f}")

# Check mode counts at different ℓ
print(f"\nMode counts per ℓ ring:")
for l_lo, l_hi in [(7, 15), (15, 30), (30, 50), (50, 100)]:
    mask = (ell_grid >= l_lo) & (ell_grid < l_hi)
    n_modes = mask.sum()
    expected = np.pi * (l_hi**2 - l_lo**2) / (2*np.pi/box_rad)**2  # area of ring / mode spacing²
    power_sum = np.sum(C_phi_measured_2d[mask])
    print(f"  ℓ={l_lo}-{l_hi}: {n_modes} modes (expected ~{expected:.0f}), power sum = {power_sum:.4e}")
