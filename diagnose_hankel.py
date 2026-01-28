#!/usr/bin/env python3
"""
Test: Compute ξ from MEASURED C_ℓ and compare to TreeCorr.
This isolates Hankel transform issues from power spectrum issues.
"""
import numpy as np
from scipy.integrate import quad
from scipy.special import jv
from scipy.interpolate import interp1d

from generate_fields_lib import generate_phi_from_power_spectrum, generate_vector_field
from measure_correlations import measure_vv_correlation
from measure_power_spectra import compute_spin_power_auto

NX, NY = 512, 512
BOX_SIZE_DEG = 50.0
AMPLITUDE = 1.0
INDEX = -5.0
C_ALPHA = 1.0

# Generate fields
power_config = {"type": "power_law", "amplitude": AMPLITUDE, "index": INDEX}
phi_real, phi_fourier = generate_phi_from_power_spectrum(NX, NY, BOX_SIZE_DEG, power_config, seed=42)
v1, v2 = generate_vector_field(phi_fourier, C_ALPHA, NX, NY, BOX_SIZE_DEG)

# Measure power spectrum - use many more bins for accurate integration
config = {'n_bins': 200, 'ell_min': 5, 'ell_max': 1800, 'use_log_bins': True}
ell, vv_EE, vv_BB, _ = compute_spin_power_auto((v1, v2), BOX_SIZE_DEG, config, spin=1)
C_vv_measured = vv_BB  # velocity is B-mode in our convention

# Measure correlation with TreeCorr
vv_result = measure_vv_correlation(v1, v2, BOX_SIZE_DEG, nbins=15, min_sep=2.0, max_sep=200.0)
valid = vv_result['npairs'] > 0
theta = vv_result['theta'][valid]
xi_treecorr = vv_result['xi_plus'][valid]

# Compute ξ from measured C_ℓ using Hankel transform
theta_rad = np.deg2rad(theta / 60.0)

# Use Riemann sum instead of quad (more robust for discrete data)
def xi_from_Cl_sum(th_rad, ell_arr, Cl_arr):
    """Riemann sum: ξ = Σ (ℓ Δℓ)/(2π) C_ℓ J_0(ℓθ)"""
    # Use trapezoid rule with log-spaced ell
    d_ell = np.diff(ell_arr)
    ell_mid = (ell_arr[:-1] + ell_arr[1:]) / 2
    Cl_mid = (Cl_arr[:-1] + Cl_arr[1:]) / 2
    integrand = ell_mid / (2 * np.pi) * Cl_mid * jv(0, ell_mid * th_rad)
    return np.sum(integrand * d_ell)

xi_from_measured_sum = np.array([xi_from_Cl_sum(th, ell, C_vv_measured) for th in theta_rad])

# Also try quad for comparison
C_interp = interp1d(ell, C_vv_measured, bounds_error=False, fill_value=0)
def xi_from_Cl_quad(th_rad):
    def integrand(l):
        return l / (2 * np.pi) * C_interp(l) * jv(0, l * th_rad)
    result, _ = quad(integrand, ell.min(), ell.max(), limit=500)
    return result
xi_from_measured_quad = np.array([xi_from_Cl_quad(th) for th in theta_rad])

# Compare
ratio_sum = xi_treecorr / xi_from_measured_sum
ratio_quad = xi_treecorr / xi_from_measured_quad
print("TreeCorr / Hankel(measured C_ℓ):")
print(f"  Riemann sum method: mean ratio = {np.mean(ratio_sum):.4f} ± {np.std(ratio_sum):.4f}")
print(f"  Quad method:        mean ratio = {np.mean(ratio_quad):.4f} ± {np.std(ratio_quad):.4f}")
print(f"\nIf ratio ≈ 1: Hankel is correct")
print(f"If ratio ≈ 2: Factor of 2 issue in formula or convention")

# Direct 2D approach: ξ(r) = ifft2(|FFT(v)|²) / N² (correlation via FFT)
print("\n--- Direct FFT correlation ---")
v_complex = v1 + 1j * v2
v_fft = np.fft.fft2(v_complex)
power_2d = np.abs(v_fft)**2  # |FFT|²
xi_2d = np.fft.ifft2(power_2d).real / (NX * NY)  # inverse FFT gives correlation

# Sample at specific separations
box_rad = np.deg2rad(BOX_SIZE_DEG)
pix_size_arcmin = BOX_SIZE_DEG * 60 / NX  # arcmin per pixel
print(f"Pixel size: {pix_size_arcmin:.2f} arcmin")

# Compare at a few θ values
for th_arcmin in [5, 20, 50, 100]:
    pix_sep = th_arcmin / pix_size_arcmin
    if pix_sep < NX // 2:
        xi_direct = xi_2d[int(pix_sep), 0]  # x-direction
        # Find TreeCorr value at similar θ
        idx = np.argmin(np.abs(theta - th_arcmin))
        xi_tc = xi_treecorr[idx]
        print(f"  θ={th_arcmin}': direct={xi_direct:.4e}, TreeCorr={xi_tc:.4e}, ratio={xi_tc/xi_direct:.3f}")

# Check: ∫ C_ℓ d²ℓ should equal Σ|FFT|² × normalization
# The direct approach: variance = xi(0) = (1/N²) Σ|FFT|² = (1/N⁴) Σ|FFT|² × N²
print(f"\n--- Variance consistency check ---")
var_direct = xi_2d[0, 0]  # ξ(0) = variance
var_field = v1.var() + v2.var()
print(f"Var from ξ(0): {var_direct:.6e}")
print(f"Var from field: {var_field:.6e}")

# Now check if Hankel integral gives same variance
# ξ(0) = ∫ ℓ dℓ/(2π) C_ℓ J_0(0) = ∫ ℓ dℓ/(2π) C_ℓ
from scipy.integrate import trapezoid
var_hankel = trapezoid(ell * C_vv_measured, ell) / (2 * np.pi)
print(f"Var from Hankel (J_0(0)=1): {var_hankel:.6e}")
print(f"Ratio Hankel/field: {var_hankel / var_field:.4f}")

# Check: compute C_ℓ directly from 2D FFT and compare to binned
print(f"\n--- C_ℓ normalization check ---")
kx = 2 * np.pi * np.fft.fftfreq(NX, d=box_rad/NX)
ky = 2 * np.pi * np.fft.fftfreq(NY, d=box_rad/NY)
kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
ell_grid = np.sqrt(kx_grid**2 + ky_grid**2)

# Raw 2D power: |FFT|² normalized to physical C_ℓ
# For 2D: Var = (1/L²) Σ C_ℓ, so C_ℓ = |FFT|² × L² / N⁴
power_2d_norm = power_2d * box_rad**2 / (NX * NY)**2

# Bin this and compare to measured C_ℓ
ell_edges = np.geomspace(5, 1800, 41)
ell_bin_centers = []
Cl_from_2d = []
for i in range(len(ell_edges)-1):
    mask = (ell_grid >= ell_edges[i]) & (ell_grid < ell_edges[i+1])
    if mask.sum() > 0:
        ell_bin_centers.append(np.mean(ell_grid[mask]))
        Cl_from_2d.append(np.mean(power_2d_norm[mask]))
ell_bin_centers = np.array(ell_bin_centers)
Cl_from_2d = np.array(Cl_from_2d)

# Interpolate measured C_ℓ to same ell values
C_interp_meas = interp1d(ell, C_vv_measured, bounds_error=False, fill_value=0)
Cl_meas_at_bins = C_interp_meas(ell_bin_centers)

ratio_Cl = Cl_from_2d / Cl_meas_at_bins
print(f"Mean ratio (2D FFT / binned C_ℓ): {np.mean(ratio_Cl[Cl_meas_at_bins > 0]):.4f}")
print(f"(Should be 1.0 if normalizations match)")

# Check: is EE+BB = total power?
print(f"\n--- E/B mode check ---")
print(f"Mean EE: {np.mean(vv_EE):.4e}")
print(f"Mean BB: {np.mean(vv_BB):.4e}")
print(f"Mean EE+BB: {np.mean(vv_EE + vv_BB):.4e}")

# Variance from EE+BB
var_from_EE_BB = trapezoid(ell * (vv_EE + vv_BB), ell) / (2 * np.pi)
print(f"Var from Hankel(EE+BB): {var_from_EE_BB:.6e}")
print(f"Var from field:         {var_field:.6e}")
print(f"Ratio: {var_from_EE_BB / var_field:.4f}")

# The discrete sum formula: Var = (1/L²) Σ_k C_ℓ
# Sum over all N² modes with their C_ℓ values
var_discrete_sum = np.sum(power_2d_norm) / box_rad**2
print(f"\n--- Discrete vs continuous integration ---")
print(f"Var from discrete sum (1/L²)ΣC_ℓ: {var_discrete_sum:.6e}")
print(f"Var from field:                    {var_field:.6e}")
print(f"Ratio: {var_discrete_sum / var_field:.4f}")

# Compute 1D Hankel integral from full 2D power (not binned)
# ∫ ℓ dℓ C_ℓ where C_ℓ is the azimuthal average at each ℓ
# Use the 2D power_2d_norm directly
print(f"\n--- 1D average from 2D power ---")

# Create ℓ-bins for averaging
n_ell_bins = 500
ell_bin_edges = np.linspace(0, ell_grid.max(), n_ell_bins + 1)
ell_1d = 0.5 * (ell_bin_edges[:-1] + ell_bin_edges[1:])
Cl_1d = np.zeros(n_ell_bins)
for i in range(n_ell_bins):
    mask = (ell_grid >= ell_bin_edges[i]) & (ell_grid < ell_bin_edges[i+1])
    if mask.sum() > 0:
        Cl_1d[i] = np.mean(power_2d_norm[mask])

# Integrate: ∫ ℓ dℓ/(2π) C_ℓ
var_from_1d_integral = trapezoid(ell_1d * Cl_1d, ell_1d) / (2 * np.pi)
print(f"Var from 1D integral: {var_from_1d_integral:.6e}")
print(f"Var from field:       {var_field:.6e}")
print(f"Ratio: {var_from_1d_integral / var_field:.4f}")

# The issue: 1D integral weights by ℓ, but 2D sum doesn't
# Actually, 2D sum = Σ C_ℓ_k = Σ_ℓ (n_modes at ℓ) × C_ℓ ≈ Σ_ℓ (2πℓ/Δℓ) × C_ℓ
# where Δℓ = 2π/L is mode spacing
# So: Σ C_ℓ_k ≈ (L/2π) × 2π × Σ ℓ C_ℓ × Δℓ = L × ∫ ℓ dℓ C_ℓ
# Therefore: (1/L²) Σ C_ℓ_k = (1/L) ∫ ℓ dℓ C_ℓ

# But we're computing (1/2π) ∫ ℓ dℓ C_ℓ, which differs by L/2π factor?
# Let me check: for our box L = box_rad ≈ 0.87 rad
print(f"\nL = {box_rad:.4f} rad")
print(f"L/(2π) = {box_rad/(2*np.pi):.4f}")
print(f"Ratio × L/(2π) = {var_from_1d_integral / var_field * box_rad/(2*np.pi):.4f}")

# Now compute ξ(θ) using the fine 1D power spectrum
print(f"\n--- ξ(θ) from fine 1D power ---")
xi_from_fine_1d = np.array([xi_from_Cl_sum(th, ell_1d, Cl_1d) for th in theta_rad])
ratio_fine = xi_treecorr / xi_from_fine_1d
print(f"TreeCorr / Hankel(fine 1D): mean = {np.mean(ratio_fine):.4f} ± {np.std(ratio_fine):.4f}")
