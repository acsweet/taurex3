"""
Quick test to verify what the forward model outputs and if it matches observations.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from taurex.cache import OpacityCache, CIACache
from taurex.constants import RJUP, RSOL

from .adc_jax_utils import (
    load_planet_spectrum,
    load_auxiliary_data,
    create_taurex_model_from_adc_planet,
    create_adc_jax_forward_model
)
from .jax_backend import extract_fitting_params

# Setup
OpacityCache().clear_cache()
OpacityCache().set_opacity_path('test_files/xsec/xsec_sampled_R15000_0.3-50')
CIACache().set_cia_path('test_files/cia/HITRAN/data')

# Load ADC planet 1000
planet_id = 1000
print("="*80)
print(f"TESTING FORWARD MODEL OUTPUT FOR PLANET {planet_id}")
print("="*80)

# Load data
aux_data = load_auxiliary_data('test_files/adc_2023/TrainingData/AuxillaryTable.csv', planet_id)
spectrum_dict = load_planet_spectrum('test_files/adc_2023/TrainingData/SpectralData.hdf5', planet_id)

# Use FM parameters as ground truth
import pandas as pd
fm_df = pd.read_csv('test_files/adc_2023/TrainingData/Ground Truth Package/FM_Parameter_Table.csv')
planet_row = fm_df[fm_df['planet_ID'] == f'train{planet_id}'].iloc[0]

gt = {
    'planet_radius': planet_row['planet_radius'],
    'planet_temp': planet_row['planet_temp'],
    'H2O': 10**planet_row['log_H2O'],
    'CO2': 10**planet_row['log_CO2'],
    'CO': 10**planet_row['log_CO'],
    'CH4': 10**planet_row['log_CH4'],
    'NH3': 10**planet_row['log_NH3'],
}

print(f"\n1. GROUND TRUTH PARAMETERS:")
print(f"   Planet radius: {gt['planet_radius']:.6f} R_jup")
print(f"   Temperature: {gt['planet_temp']:.1f} K")
print(f"   H2O: {gt['H2O']:.3e}")
print(f"   Star radius: {aux_data['star_radius_m']/RSOL:.3f} R_sol")

# Expected baseline transit depth
R_p = gt['planet_radius'] * RJUP
R_s = aux_data['star_radius_m']
baseline_depth = (R_p / R_s)**2

print(f"\n2. EXPECTED TRANSIT DEPTH:")
print(f"   Baseline (R_p/R_s)²: {baseline_depth:.6e}")

# Observed spectrum
print(f"\n3. OBSERVED SPECTRUM:")
print(f"   Mean: {spectrum_dict['spectrum'].mean():.6e}")
print(f"   Min: {spectrum_dict['spectrum'].min():.6e}")
print(f"   Max: {spectrum_dict['spectrum'].max():.6e}")
print(f"   Feature amplitude: {(spectrum_dict['spectrum'].max() - spectrum_dict['spectrum'].min()):.3e}")
print(f"   Ratio to baseline: {spectrum_dict['spectrum'].mean() / baseline_depth:.3f}")

# Create TauREx model
print(f"\n4. CREATING TAUREX MODEL...")
tm = create_taurex_model_from_adc_planet(aux_data, gt, nlayers=30)
tm.initialize_profiles()

print(f"   Model built with {tm.nLayers} layers")
print(f"   Planet radius: {tm.planet.fullRadius/RJUP:.6f} R_jup")
print(f"   Star radius: {tm.star.radius/RSOL:.3f} R_sol")
print(f"   Pressure range: {tm.pressureProfile.min():.2e} to {tm.pressureProfile.max():.2e} Pa")
print(f"   Max pressure: {tm.pressureProfile.max()/1e5:.1f} bar")

# Compute TauREx spectrum at observed wavelengths
wl_obs = spectrum_dict['wl_grid']
wn_obs = 10000.0 / wl_obs

print(f"\n5. TAUREX NATIVE SPECTRUM (at observed wavelengths):")
taurex_result = tm.model(wn_obs)
taurex_spec = taurex_result[1] if isinstance(taurex_result, tuple) else taurex_result

print(f"   Mean: {taurex_spec.mean():.6e}")
print(f"   Min: {taurex_spec.min():.6e}")
print(f"   Max: {taurex_spec.max():.6e}")
print(f"   Feature amplitude: {(taurex_spec.max() - taurex_spec.min()):.3e}")
print(f"   Ratio to baseline: {taurex_spec.mean() / baseline_depth:.3f}")
print(f"   Ratio to observed: {taurex_spec.mean() / spectrum_dict['spectrum'].mean():.3f}")

# Create JAX forward model
print(f"\n6. JAX FORWARD MODEL:")
params, _ = extract_fitting_params(tm, dtype=jnp.float64)
forward_model, obs_spectrum = create_adc_jax_forward_model(tm, spectrum_dict, dtype=jnp.float64)

# Test forward model
print(f"   Testing forward model...")
jax_spec = forward_model(params)
print(f"   Output shape: {jax_spec.shape}")
print(f"   Mean: {float(jax_spec.mean()):.6e}")
print(f"   Min: {float(jax_spec.min()):.6e}")
print(f"   Max: {float(jax_spec.max()):.6e}")
print(f"   Feature amplitude: {float(jax_spec.max() - jax_spec.min()):.3e}")

jax_spec_binned = jax_spec  # Use this name for compatibility below

# Compare
print(f"\n7. COMPARISON:")
print(f"   {'Source':<20} {'Mean':<15} {'Feature Amp':<15} {'Match to Obs':<15}")
print(f"   {'-'*65}")
obs_mean = spectrum_dict['spectrum'].mean()
obs_amp = spectrum_dict['spectrum'].max() - spectrum_dict['spectrum'].min()
taurex_match = taurex_spec.mean() / obs_mean * 100
jax_match = float(jax_spec_binned.mean()) / obs_mean * 100
print(f"   {'Baseline':<20} {baseline_depth:<15.6e} {'N/A':<15} {'N/A':<15}")
print(f"   {'Observed':<20} {obs_mean:<15.6e} {obs_amp:<15.3e} {'100%':<15}")
print(f"   {'TauREx':<20} {taurex_spec.mean():<15.6e} {(taurex_spec.max() - taurex_spec.min()):<15.3e} {f'{taurex_match:.1f}%':<15}")
print(f"   {'JAX (binned)':<20} {float(jax_spec_binned.mean()):<15.6e} {float(jax_spec_binned.max() - jax_spec_binned.min()):<15.3e} {f'{jax_match:.1f}%':<15}")

# Calculate residuals (only JAX is binned)
residuals_jax = (spectrum_dict['spectrum'] - np.array(jax_spec_binned)) / spectrum_dict['noise']

print(f"\n8. RESIDUALS (in sigma):")
print(f"   JAX RMS: {np.sqrt(np.mean(residuals_jax**2)):.2f}σ")
print(f"   JAX max: {np.max(np.abs(residuals_jax)):.2f}σ")
print(f"   JAX chi-squared: {np.sum(residuals_jax**2):.2f}")

print(f"\n9. CONCLUSION:")
if abs(float(jax_spec_binned.mean()) / spectrum_dict['spectrum'].mean() - 1.0) < 0.05:
    print(f"   ✓ JAX forward model outputs are within 5% of observed spectrum")
    print(f"   ✓ Model is outputting TRANSIT DEPTH correctly")
else:
    print(f"   ✗ JAX forward model differs by >5% from observations")
    print(f"   ✗ Check forward model implementation")

if abs(taurex_spec.mean() / spectrum_dict['spectrum'].mean() - 1.0) < 0.05:
    print(f"   ✓ TauREx matches observations within 5%")
else:
    print(f"   ⚠ TauREx differs from observations by >5%")
    print(f"      This is expected - the ground truth params may not be optimal")
