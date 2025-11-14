"""
Test and debug the ADC 2023 forward model to find the spectrum scale mismatch.

This test systematically checks:
1. Parameter initialization
2. TauREx native spectrum
3. JAX forward model spectrum
4. Binning correctness
5. Unit conversions
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from taurex.cache import OpacityCache, CIACache
from taurex.constants import RJUP, RSOL, MJUP, MSOL

from adc_jax_utils import (
    load_planet_spectrum,
    load_auxiliary_data, 
    load_ground_truth,
    create_taurex_model_from_adc_planet,
    create_adc_jax_forward_model
)
from experiment import extract_fitting_params


def test_parameter_initialization():
    """Test 1: Verify parameters are set correctly."""
    print("="*80)
    print("TEST 1: Parameter Initialization")
    print("="*80)
    
    # Setup caches
    OpacityCache().clear_cache()
    OpacityCache().set_opacity_path('test_files/xsec/xsec_sampled_R15000_0.3-50')
    CIACache().set_cia_path('test_files/cia/HITRAN/data')
    
    # Load data
    planet_id = 1000
    aux_data = load_auxiliary_data('test_files/adc_2023/TrainingData/AuxillaryTable.csv', planet_id)
    gt = load_ground_truth('test_files/adc_2023/TrainingData/Ground Truth Package/FM_Parameter_Table.csv', planet_id)
    
    print("\n1a. Ground Truth Values:")
    print(f"   Planet radius: {gt['planet_radius']:.6f} R_jup")
    print(f"   Planet temp: {gt['planet_temp']:.2f} K")
    print(f"   H2O: {gt['H2O']:.6e}")
    print(f"   CO2: {gt['CO2']:.6e}")
    print(f"   CO: {gt['CO']:.6e}")
    
    print("\n1b. Auxiliary Data:")
    print(f"   Star temp: {aux_data['star_temperature']:.0f} K")
    print(f"   Star radius: {aux_data['star_radius_m']/RSOL:.3f} R_sol")
    print(f"   Star mass: {aux_data['star_mass_kg']/MSOL:.3f} M_sol")
    print(f"   Planet mass: {aux_data['planet_mass_kg']/MJUP:.3f} M_jup")
    
    # Create model
    tm = create_taurex_model_from_adc_planet(aux_data, gt, nlayers=30)
    
    # Initialize profiles
    tm.initialize_profiles()
    
    print("\n1c. TauREx Model After Build:")
    print(f"   Planet radius: {tm.planet.fullRadius/RJUP:.6f} R_jup")
    print(f"   Star radius: {tm.star.radius:.3f} R_sol")
    print(f"   Temperature: {tm.temperatureProfile[0]:.2f} K")
    print(f"   Active gases: {list(tm.chemistry.activeGases)}")
    print(f"   Number of gases: {len(tm.chemistry.gases)}")
    
    # Extract params
    params, _ = extract_fitting_params(tm, dtype=jnp.float64)
    
    print("\n1d. Extracted Parameters:")
    print(f"   planet_radius: {float(params['planet_radius']):.6f}")
    print(f"   T: {float(params['T']):.2f}")
    print(f"   H2O: {float(params['H2O']):.6e}")
    
    # Check consistency
    radius_match = abs(float(params['planet_radius']) - gt['planet_radius']) < 1e-6
    temp_match = abs(float(params['T']) - gt['planet_temp']) < 0.1
    h2o_match = abs(float(params['H2O']) - gt['H2O']) / gt['H2O'] < 1e-6
    
    print("\n1e. Consistency Check:")
    print(f"   ✓ Radius matches" if radius_match else f"   ✗ Radius mismatch!")
    print(f"   ✓ Temperature matches" if temp_match else f"   ✗ Temperature mismatch!")
    print(f"   ✓ H2O matches" if h2o_match else f"   ✗ H2O mismatch!")
    
    return tm, params, gt, aux_data


def test_taurex_native_spectrum(tm, gt, aux_data):
    """Test 2: Check TauREx native spectrum scale."""
    print("\n" + "="*80)
    print("TEST 2: TauREx Native Spectrum")
    print("="*80)
    
    # Load observed spectrum
    spectrum_dict = load_planet_spectrum('test_files/adc_2023/TrainingData/SpectralData.hdf5', 1000)
    
    # Compute expected baseline transit depth
    R_p = gt['planet_radius'] * RJUP
    R_s = aux_data['star_radius_m']
    baseline_depth = (R_p / R_s) ** 2
    
    print("\n2a. Expected Transit Depth:")
    print(f"   Planet radius: {gt['planet_radius']:.6f} R_jup = {R_p:.6e} m")
    print(f"   Star radius: {R_s/RSOL:.6f} R_sol = {R_s:.6e} m")
    print(f"   Baseline (R_p/R_s)²: {baseline_depth:.6e}")
    
    print("\n2b. Observed Spectrum:")
    print(f"   Mean: {spectrum_dict['spectrum'].mean():.6e}")
    print(f"   Min: {spectrum_dict['spectrum'].min():.6e}")
    print(f"   Max: {spectrum_dict['spectrum'].max():.6e}")
    print(f"   Ratio (obs/baseline): {spectrum_dict['spectrum'].mean() / baseline_depth:.3f}")
    
    # Compute TauREx spectrum at observed wavelengths
    wl_obs = spectrum_dict['wl_grid']
    wn_obs = 10000.0 / wl_obs
    
    print("\n2c. Computing TauREx spectrum at observed wavelengths...")
    taurex_result = tm.model(wn_obs)
    # tm.model() returns (wngrid, spectrum, tau, extra)
    # We want element 1 (spectrum/transit depth)
    if isinstance(taurex_result, tuple) and len(taurex_result) >= 2:
        taurex_spec = taurex_result[1]  # Second element is transit depth
    else:
        taurex_spec = taurex_result
    
    print(f"   TauREx spectrum:")
    print(f"   Mean: {taurex_spec.mean():.6e}")
    print(f"   Min: {taurex_spec.min():.6e}")
    print(f"   Max: {taurex_spec.max():.6e}")
    print(f"   Ratio (taurex/obs): {taurex_spec.mean() / spectrum_dict['spectrum'].mean():.3f}")
    print(f"   Ratio (taurex/baseline): {taurex_spec.mean() / baseline_depth:.3f}")
    
    # Check if TauREx is reasonable
    taurex_reasonable = 0.8 < (taurex_spec.mean() / spectrum_dict['spectrum'].mean()) < 1.2
    
    print("\n2d. TauREx Sanity Check:")
    if taurex_reasonable:
        print(f"   ✓ TauREx spectrum is within 20% of observed")
    else:
        print(f"   ✗ TauREx spectrum differs by >20% from observed")
        print(f"      This suggests a fundamental model setup issue!")
    
    return taurex_spec, spectrum_dict, baseline_depth


def test_jax_forward_model(tm, params, spectrum_dict):
    """Test 3: Check JAX forward model spectrum."""
    print("\n" + "="*80)
    print("TEST 3: JAX Forward Model")
    print("="*80)
    
    print("\n3a. Creating JAX forward model...")
    forward_model, obs_spectrum = create_adc_jax_forward_model(
        tm, spectrum_dict, use_full_diff=True
    )
    
    print(f"   Forward model created")
    print(f"   Observed spectrum bins: {len(obs_spectrum.spectrum)}")
    print(f"   Bin edges: {len(obs_spectrum.binEdges)}")
    
    print("\n3b. Computing JAX spectrum...")
    jax_spec = forward_model(params)
    
    print(f"   JAX spectrum:")
    print(f"   Shape: {jax_spec.shape}")
    print(f"   Mean: {float(jax_spec.mean()):.6e}")
    print(f"   Min: {float(jax_spec.min()):.6e}")
    print(f"   Max: {float(jax_spec.max()):.6e}")
    
    obs_mean = spectrum_dict['spectrum'].mean()
    print(f"\n3c. Comparison:")
    print(f"   Observed mean: {obs_mean:.6e}")
    print(f"   JAX mean: {float(jax_spec.mean()):.6e}")
    print(f"   Ratio (jax/obs): {float(jax_spec.mean() / obs_mean):.3f}")
    
    jax_reasonable = 0.8 < (float(jax_spec.mean()) / obs_mean) < 1.2
    
    print("\n3d. JAX Sanity Check:")
    if jax_reasonable:
        print(f"   ✓ JAX spectrum is within 20% of observed")
    else:
        print(f"   ✗ JAX spectrum differs by >20% from observed")
        ratio = float(jax_spec.mean() / obs_mean)
        if ratio < 0.8:
            print(f"      JAX spectrum is too small by {(1-ratio)*100:.1f}%")
        else:
            print(f"      JAX spectrum is too large by {(ratio-1)*100:.1f}%")
    
    return jax_spec, forward_model


def test_binning(tm, spectrum_dict):
    """Test 4: Check if binning is working correctly."""
    print("\n" + "="*80)
    print("TEST 4: Binning Matrix")
    print("="*80)
    
    # Get high-res wavenumber grid
    from taurex.util import clip_native_to_wngrid
    
    wn_obs = 10000.0 / spectrum_dict['wl_grid']
    high_res_wngrid = clip_native_to_wngrid(
        tm.nativeWavenumberGrid,
        wn_obs
    )
    
    if high_res_wngrid[0] > high_res_wngrid[-1]:
        high_res_wngrid = high_res_wngrid[::-1]
    
    print(f"\n4a. Grid Information:")
    print(f"   Native grid: {len(tm.nativeWavenumberGrid)} points")
    print(f"   Clipped grid: {len(high_res_wngrid)} points")
    print(f"   Observed bins: {len(spectrum_dict['spectrum'])}")
    print(f"   Compression: {len(high_res_wngrid) / len(spectrum_dict['spectrum']):.0f}x")
    
    # Check bin edges
    from taurex.data.spectrum.array import ArraySpectrum
    obs_data = np.column_stack([
        spectrum_dict['wl_grid'],
        spectrum_dict['spectrum'],
        spectrum_dict['noise']
    ])
    obs_spectrum = ArraySpectrum(obs_data)
    
    print(f"\n4b. Bin Edges:")
    print(f"   Number of edges: {len(obs_spectrum.binEdges)}")
    print(f"   Edge range (wn): {obs_spectrum.binEdges.min():.1f} - {obs_spectrum.binEdges.max():.1f} cm⁻¹")
    wl_from_edges = 10000.0 / obs_spectrum.binEdges
    print(f"   Edge range (wl): {wl_from_edges.min():.3f} - {wl_from_edges.max():.3f} μm")
    
    # Check for issues
    edges_sorted = np.all(np.diff(obs_spectrum.binEdges) > 0)
    edges_count = len(obs_spectrum.binEdges) == len(spectrum_dict['spectrum']) + 1
    
    print(f"\n4c. Bin Edge Validation:")
    print(f"   ✓ Edges sorted" if edges_sorted else "   ✗ Edges NOT sorted!")
    print(f"   ✓ Correct count (N+1)" if edges_count else f"   ✗ Wrong count: {len(obs_spectrum.binEdges)} vs {len(spectrum_dict['spectrum'])+1}")
    
    return obs_spectrum


def test_spectrum_comparison(taurex_spec, jax_spec, spectrum_dict):
    """Test 5: Detailed comparison of spectra."""
    print("\n" + "="*80)
    print("TEST 5: Detailed Spectrum Comparison")
    print("="*80)
    
    obs_spec = spectrum_dict['spectrum']
    wl_grid = spectrum_dict['wl_grid']
    
    print(f"\n5a. First 5 wavelength bins:")
    print(f"   {'λ (μm)':<10} {'Observed':<12} {'TauREx':<12} {'JAX':<12} {'Ratio T/O':<12} {'Ratio J/O':<12}")
    print(f"   {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    for i in range(min(5, len(wl_grid))):
        t_ratio = taurex_spec[i] / obs_spec[i]
        j_ratio = float(jax_spec[i]) / obs_spec[i]
        print(f"   {wl_grid[i]:<10.3f} {obs_spec[i]:<12.6e} {taurex_spec[i]:<12.6e} "
              f"{float(jax_spec[i]):<12.6e} {t_ratio:<12.3f} {j_ratio:<12.3f}")
    
    print(f"\n5b. Statistics:")
    taurex_ratio = taurex_spec.mean() / obs_spec.mean()
    jax_ratio = float(jax_spec.mean()) / obs_spec.mean()
    taurex_vs_jax = taurex_spec.mean() / float(jax_spec.mean())
    
    print(f"   TauREx/Observed: {taurex_ratio:.4f}")
    print(f"   JAX/Observed: {jax_ratio:.4f}")
    print(f"   TauREx/JAX: {taurex_vs_jax:.4f}")
    
    # Check if TauREx and JAX match each other
    taurex_jax_match = abs(taurex_vs_jax - 1.0) < 0.05  # Within 5%
    
    print(f"\n5c. Diagnosis:")
    if taurex_ratio > 0.9 and taurex_ratio < 1.1:
        print(f"   ✓ TauREx matches observed well")
    else:
        print(f"   ✗ TauREx differs from observed by {abs(1-taurex_ratio)*100:.1f}%")
    
    if jax_ratio > 0.9 and jax_ratio < 1.1:
        print(f"   ✓ JAX matches observed well")
    else:
        print(f"   ✗ JAX differs from observed by {abs(1-jax_ratio)*100:.1f}%")
    
    if taurex_jax_match:
        print(f"   ✓ TauREx and JAX agree (within 5%)")
        print(f"      → Problem is likely in model parameters, not JAX implementation")
    else:
        print(f"   ✗ TauREx and JAX disagree by {abs(1-taurex_vs_jax)*100:.1f}%")
        print(f"      → Problem is in JAX forward model or binning")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("ADC 2023 FORWARD MODEL DEBUG SUITE")
    print("="*80)
    print("\nThis will systematically check:")
    print("  1. Parameter initialization")
    print("  2. TauREx native spectrum")
    print("  3. JAX forward model")
    print("  4. Binning correctness")
    print("  5. Detailed comparison")
    print("\n" + "="*80)
    
    try:
        # Run tests
        tm, params, gt, aux_data = test_parameter_initialization()
        taurex_spec, spectrum_dict, baseline = test_taurex_native_spectrum(tm, gt, aux_data)
        jax_spec, forward_model = test_jax_forward_model(tm, params, spectrum_dict)
        obs_spectrum = test_binning(tm, spectrum_dict)
        test_spectrum_comparison(taurex_spec, jax_spec, spectrum_dict)
        
        # Final summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        obs_mean = spectrum_dict['spectrum'].mean()
        taurex_ratio = taurex_spec.mean() / obs_mean
        jax_ratio = float(jax_spec.mean()) / obs_mean
        
        if abs(taurex_ratio - 1.0) < 0.1 and abs(jax_ratio - 1.0) < 0.1:
            print("\n✓ SUCCESS: Both TauREx and JAX produce reasonable spectra!")
            print(f"  TauREx: {taurex_ratio:.3f}x observed")
            print(f"  JAX: {jax_ratio:.3f}x observed")
        elif abs(taurex_ratio - 1.0) > 0.1 and abs(jax_ratio - 1.0) > 0.1:
            print("\n✗ ISSUE: Both models produce wrong scale")
            print(f"  This suggests a problem in model setup (parameters/units)")
            print(f"  TauREx: {taurex_ratio:.3f}x observed")
            print(f"  JAX: {jax_ratio:.3f}x observed")
        else:
            print("\n✗ ISSUE: Models disagree")
            if abs(taurex_ratio - 1.0) < 0.1:
                print(f"  TauREx is correct: {taurex_ratio:.3f}x observed")
                print(f"  JAX is wrong: {jax_ratio:.3f}x observed")
                print(f"  → Problem is in JAX forward model or binning")
            else:
                print(f"  JAX is correct: {jax_ratio:.3f}x observed")
                print(f"  TauREx is wrong: {taurex_ratio:.3f}x observed")
                print(f"  → Problem is in model setup")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED WITH ERROR:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
