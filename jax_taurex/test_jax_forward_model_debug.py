"""
Debug the JAX forward model 0.74x scale error.

Tests the three suspected causes:
1. Path integral calculation
2. Absorption normalization by star radius
3. Binning flux conservation

Strategy: Compare TauREx native vs JAX at each stage to find where the 0.74x factor appears.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from taurex.cache import OpacityCache, CIACache
from taurex.constants import RJUP, RSOL
from taurex.util import clip_native_to_wngrid

from .adc_jax_utils import (
    load_planet_spectrum,
    load_auxiliary_data,
    load_ground_truth,
    create_taurex_model_from_adc_planet,
    create_adc_jax_forward_model
)
from .jax_backend import (
    extract_fitting_params,
    full_diff_create_forward_model,
    full_diff_create_binned_forward_model
)


def setup_test_model():
    """Create a consistent model for all tests."""
    OpacityCache().clear_cache()
    OpacityCache().set_opacity_path('test_files/xsec/xsec_sampled_R15000_0.3-50')
    CIACache().set_cia_path('test_files/cia/HITRAN/data')
    
    planet_id = 1000
    aux_data = load_auxiliary_data('test_files/adc_2023/TrainingData/AuxillaryTable.csv', planet_id)
    gt = load_ground_truth('test_files/adc_2023/TrainingData/Ground Truth Package/FM_Parameter_Table.csv', planet_id)
    spectrum_dict = load_planet_spectrum('test_files/adc_2023/TrainingData/SpectralData.hdf5', planet_id)
    
    tm = create_taurex_model_from_adc_planet(aux_data, gt, nlayers=30)
    tm.initialize_profiles()
    
    params, _ = extract_fitting_params(tm, dtype=jnp.float64)
    
    return tm, params, spectrum_dict, gt, aux_data


def test_1_high_res_spectra():
    """
    Test 1: Compare TauREx vs JAX high-resolution spectra (before binning).
    
    This isolates whether the problem is in the forward model or binning.
    """
    print("="*80)
    print("TEST 1: High-Resolution Spectra (Before Binning)")
    print("="*80)
    
    tm, params, spectrum_dict, gt, aux_data = setup_test_model()
    
    # Get high-res wavenumber grid
    wn_obs = 10000.0 / spectrum_dict['wl_grid']
    high_res_wngrid = clip_native_to_wngrid(tm.nativeWavenumberGrid, wn_obs)
    if high_res_wngrid[0] > high_res_wngrid[-1]:
        high_res_wngrid = high_res_wngrid[::-1]
    
    print(f"\n1a. Grid Setup:")
    print(f"   High-res points: {len(high_res_wngrid)}")
    print(f"   Wavenumber range: {high_res_wngrid.min():.1f} - {high_res_wngrid.max():.1f} cm⁻¹")
    
    # Compute TauREx high-res spectrum
    print(f"\n1b. Computing TauREx high-res spectrum...")
    taurex_result = tm.model(high_res_wngrid)
    taurex_highres = taurex_result[1]  # Second element is transit depth
    
    print(f"   TauREx high-res:")
    print(f"   Mean: {taurex_highres.mean():.6e}")
    print(f"   Min: {taurex_highres.min():.6e}")
    print(f"   Max: {taurex_highres.max():.6e}")
    
    # Create JAX high-res forward model
    print(f"\n1c. Computing JAX high-res spectrum...")
    jax_forward_highres = full_diff_create_forward_model(
        tm, high_res_wngrid, pre_interpolate_opacity=False
    )
    jax_highres, _ = jax_forward_highres(params)
    
    print(f"   JAX high-res:")
    print(f"   Mean: {float(jax_highres.mean()):.6e}")
    print(f"   Min: {float(jax_highres.min()):.6e}")
    print(f"   Max: {float(jax_highres.max()):.6e}")
    
    # Compare
    ratio = float(jax_highres.mean()) / taurex_highres.mean()
    print(f"\n1d. High-Res Comparison:")
    print(f"   JAX/TauREx ratio: {ratio:.4f}")
    
    if abs(ratio - 1.0) < 0.05:
        print(f"   ✓ High-res spectra match within 5%")
        print(f"      → Problem is likely in BINNING")
        return "binning"
    elif abs(ratio - 0.74) < 0.05:
        print(f"   ✗ High-res has same 0.74x error")
        print(f"      → Problem is in FORWARD MODEL, not binning")
        return "forward_model"
    else:
        print(f"   ⚠ Unexpected ratio: {ratio:.4f}")
        print(f"      → Need deeper investigation")
        return "unknown"


def test_2_path_integrals():
    """
    Test 2: Compare path integral calculations.
    
    Check if path lengths are being computed correctly.
    """
    print("\n" + "="*80)
    print("TEST 2: Path Integral Calculation")
    print("="*80)
    
    tm, params, spectrum_dict, gt, aux_data = setup_test_model()
    
    print(f"\n2a. Model Geometry:")
    print(f"   Planet radius: {tm.planet.fullRadius/RJUP:.6f} R_jup = {tm.planet.fullRadius:.3e} m")
    print(f"   Star radius: {tm.star.radius/RSOL:.6f} R_sol = {tm.star.radius:.3e} m")
    print(f"   Layers: {tm.nLayers}")
    
    # Get atmospheric structure
    altitude = tm.altitudeProfile
    deltaz = tm.deltaz
    
    print(f"\n2b. Atmospheric Structure:")
    print(f"   Altitude range: {altitude.min():.3e} - {altitude.max():.3e} m")
    print(f"   Layer thickness: {deltaz.mean():.3e} m (mean)")
    print(f"   Total thickness: {altitude.max() - altitude.min():.3e} m")
    print(f"   Scale height (approx): {deltaz.mean():.3e} m")
    
    # Check if JAX uses the same structure
    from .jax_backend import full_diff_prepare_model_data
    jax_static, jax_initial = full_diff_prepare_model_data(tm, dtype=jnp.float64)
    
    print(f"\n2c. JAX Model Data:")
    print(f"   Planet radius (base): {jax_static['planet_radius_base']/RJUP:.6f} R_jup")
    print(f"   Star radius: {jax_static['star_radius']/RSOL:.6f} R_sol")
    print(f"   Layers: {jax_static['nlayers']}")
    
    # Compare key values
    radius_match = abs(jax_static['planet_radius_base'] - tm.planet.fullRadius) < 1.0
    star_match = abs(jax_static['star_radius'] - tm.star.radius) < 1.0
    
    print(f"\n2d. Consistency Check:")
    print(f"   ✓ Planet radius matches" if radius_match else f"   ✗ Planet radius mismatch!")
    print(f"   ✓ Star radius matches" if star_match else f"   ✗ Star radius mismatch!")
    
    # Compute expected baseline transit depth
    baseline = (tm.planet.fullRadius / tm.star.radius) ** 2
    print(f"\n2e. Expected Transit Depth:")
    print(f"   Baseline (R_p/R_s)²: {baseline:.6e}")
    print(f"   Observed mean: {spectrum_dict['spectrum'].mean():.6e}")
    print(f"   Ratio (obs/baseline): {spectrum_dict['spectrum'].mean() / baseline:.3f}")
    print(f"      (Should be >1 due to atmospheric extension)")
    
    return radius_match and star_match


def test_3_absorption_calculation():
    """
    Test 3: Check the absorption calculation formula.
    
    Compare how TauREx and JAX compute final transit depth from tau.
    """
    print("\n" + "="*80)
    print("TEST 3: Absorption Calculation")
    print("="*80)
    
    tm, params, spectrum_dict, gt, aux_data = setup_test_model()
    
    print(f"\n3a. TauREx Absorption Formula:")
    print(f"   TransmissionModel computes:")
    print(f"   absorption = (R_p² + integral) / R_s²")
    print(f"   where integral = Σ (R_p + z) * (1 - exp(-tau)) * dz * 2")
    
    print(f"\n3b. JAX Absorption Formula (from experiment.py):")
    print(f"   From compute_absorption():")
    print(f"   integral = sum((planet_radius + ap) * (1.0 - tau_exp) * _dz * 2.0)")
    print(f"   absorption = ((planet_radius**2.0) + integral) / (star_radius**2)")
    
    # Get a simple test case - small wavelength range
    wn_test = np.linspace(5000.0, 5100.0, 10)  # ~2 microns, 10 points
    
    print(f"\n3c. Test at small wavelength range (5000-5100 cm⁻¹ = ~2.0 μm):")
    
    # TauREx computation
    taurex_result = tm.model(wn_test)
    taurex_spectrum = taurex_result[1]  # All wavelengths
    taurex_wn = taurex_result[0]  # Wavenumbers actually used
    
    print(f"   TauREx result (mean): {taurex_spectrum.mean():.6e}")
    print(f"   TauREx result (first): {taurex_spectrum[0]:.6e}")
    
    # JAX computation
    jax_forward = full_diff_create_forward_model(tm, wn_test, pre_interpolate_opacity=False)
    jax_spectrum, jax_tau = jax_forward(params)
    jax_spectrum_mean = float(jax_spectrum.mean())
    jax_spectrum_first = float(jax_spectrum[0])
    
    print(f"   JAX result (mean): {jax_spectrum_mean:.6e}")
    print(f"   JAX result (first): {jax_spectrum_first:.6e}")
    print(f"   Ratio (JAX/TauREx, mean): {jax_spectrum_mean / taurex_spectrum.mean():.4f}")
    print(f"   Ratio (JAX/TauREx, first): {jax_spectrum_first / taurex_spectrum[0]:.4f}")
    
    # Check normalization
    R_p = tm.planet.fullRadius
    R_s = tm.star.radius
    baseline = (R_p / R_s) ** 2
    
    print(f"\n3d. Normalization Check:")
    print(f"   R_p = {R_p:.6e} m")
    print(f"   R_s = {R_s:.6e} m")
    print(f"   (R_p/R_s)² = {baseline:.6e}")
    print(f"   TauREx / baseline = {taurex_spectrum.mean() / baseline:.4f}")
    print(f"   JAX / baseline = {jax_spectrum_mean / baseline:.4f}")
    
    if taurex_spectrum.mean() / baseline > 1.0 and jax_spectrum_mean / baseline < 1.0:
        print(f"\n   ⚠ WARNING: JAX result is LESS than baseline!")
        print(f"      JAX: {jax_spectrum_mean:.6e}")
        print(f"      Baseline: {baseline:.6e}")
        print(f"      This is physically impossible - the atmosphere should")
        print(f"      extend ABOVE the planet surface, not below it!")
        return False
    
    return True


def test_4_binning_flux_conservation():
    """
    Test 4: Check if binning conserves flux properly.
    
    Test the binning matrix by binning a known signal.
    """
    print("\n" + "="*80)
    print("TEST 4: Binning Flux Conservation")
    print("="*80)
    
    tm, params, spectrum_dict, gt, aux_data = setup_test_model()
    
    # Create binned forward model to get the binning matrix
    from taurex.data.spectrum.array import ArraySpectrum
    obs_data = np.column_stack([
        spectrum_dict['wl_grid'],
        spectrum_dict['spectrum'],
        spectrum_dict['noise']
    ])
    obs_spectrum = ArraySpectrum(obs_data)
    
    # Get high-res grid
    wn_obs = 10000.0 / spectrum_dict['wl_grid']
    high_res_wngrid = clip_native_to_wngrid(tm.nativeWavenumberGrid, wn_obs)
    if high_res_wngrid[0] > high_res_wngrid[-1]:
        high_res_wngrid = high_res_wngrid[::-1]
    
    # Get bin edges
    bin_edges = obs_spectrum.binEdges
    if bin_edges[0] > bin_edges[-1]:
        bin_edges = bin_edges[::-1]
    
    print(f"\n4a. Binning Setup:")
    print(f"   High-res points: {len(high_res_wngrid)}")
    print(f"   Bins: {len(spectrum_dict['spectrum'])}")
    print(f"   Bin edges: {len(bin_edges)}")
    
    # Create binning matrix
    from .jax_backend import _create_binning_matrix_np
    W_np = _create_binning_matrix_np(high_res_wngrid, bin_edges, use_lambda_measure=False)
    
    print(f"\n4b. Binning Matrix Properties:")
    print(f"   Shape: {W_np.shape}")
    row_sums = W_np.sum(axis=1)
    print(f"   Row sums (should be ~1.0):")
    print(f"     Mean: {row_sums.mean():.6f}")
    print(f"     Min: {row_sums.min():.6f}")
    print(f"     Max: {row_sums.max():.6f}")
    print(f"     Std: {row_sums.std():.6f}")
    
    # Test with a flat spectrum
    print(f"\n4c. Test with flat spectrum (all 1.0):")
    flat_highres = np.ones(len(high_res_wngrid))
    flat_binned = np.dot(W_np, flat_highres)
    
    print(f"   Input (high-res): all 1.0")
    print(f"   Output (binned):")
    print(f"     Mean: {flat_binned.mean():.6f}")
    print(f"     Min: {flat_binned.min():.6f}")
    print(f"     Max: {flat_binned.max():.6f}")
    
    if abs(flat_binned.mean() - 1.0) < 0.01:
        print(f"   ✓ Binning preserves flux (mean = 1.0)")
    else:
        print(f"   ✗ Binning does NOT preserve flux (mean = {flat_binned.mean():.3f})")
    
    # Test with TauREx spectrum
    print(f"\n4d. Test with TauREx spectrum:")
    taurex_result = tm.model(high_res_wngrid)
    taurex_highres = taurex_result[1]
    taurex_binned_manual = np.dot(W_np, taurex_highres)
    
    print(f"   TauREx high-res mean: {taurex_highres.mean():.6e}")
    print(f"   Manually binned mean: {taurex_binned_manual.mean():.6e}")
    print(f"   Ratio (binned/highres): {taurex_binned_manual.mean() / taurex_highres.mean():.4f}")
    
    # Compare to JAX binning
    forward_binned = full_diff_create_binned_forward_model(
        tm, obs_spectrum, pre_interpolate_opacity=False
    )
    jax_binned = forward_binned(params)
    
    print(f"\n4e. JAX binned result:")
    print(f"   JAX binned mean: {float(jax_binned.mean()):.6e}")
    print(f"   Ratio (JAX_binned / TauREx_binned): {float(jax_binned.mean()) / taurex_binned_manual.mean():.4f}")
    
    return abs(flat_binned.mean() - 1.0) < 0.01


def test_5_step_by_step_comparison():
    """
    Test 5: Step-by-step comparison of a single wavelength.
    
    Track the computation through every stage to find the 0.74x factor.
    """
    print("\n" + "="*80)
    print("TEST 5: Step-by-Step Single Wavelength Trace")
    print("="*80)
    
    tm, params, spectrum_dict, gt, aux_data = setup_test_model()
    
    # Pick small wavelength range in the middle of the observed range
    wn_test = np.linspace(5000.0, 5010.0, 3)  # 2 microns, 3 points
    wl_test = 10000.0 / wn_test[0]
    
    print(f"\n5a. Test wavelength range: {wl_test:.3f} μm ({wn_test[0]:.1f} cm⁻¹), 3 points")
    
    # TauREx full computation
    print(f"\n5b. TauREx Computation:")
    taurex_result = tm.model(wn_test)
    taurex_wn_used = taurex_result[0]  # Actual wavenumbers
    taurex_spectrum = taurex_result[1]  # Transit depth
    taurex_tau = taurex_result[2]  # Optical depth per layer
    
    print(f"   Wavelengths computed: {len(taurex_wn_used)}")
    print(f"   First wn: {taurex_wn_used[0]:.3f} cm⁻¹")
    print(f"   Transit depth: {taurex_spectrum[0]:.6e}")
    print(f"   Tau shape: {taurex_tau.shape}")
    print(f"   Tau range: {taurex_tau.min():.6e} - {taurex_tau.max():.6e}")
    
    # JAX computation
    print(f"\n5c. JAX Computation:")
    jax_forward = full_diff_create_forward_model(tm, wn_test, pre_interpolate_opacity=False)
    jax_spectrum, jax_tau = jax_forward(params)
    
    print(f"   Transit depth: {float(jax_spectrum[0]):.6e}")
    print(f"   Tau shape: {jax_tau.shape}")
    print(f"   Tau range: {float(jax_tau.min()):.6e} - {float(jax_tau.max()):.6e}")
    
    # Direct comparison
    print(f"\n5d. Direct Comparison:")
    ratio_spectrum = float(jax_spectrum[0]) / taurex_spectrum[0]
    print(f"   JAX/TauREx spectrum: {ratio_spectrum:.6f}")
    
    # Compare tau values
    print(f"\n5e. Optical Depth Comparison:")
    print(f"   TauREx tau (first 3 layers): {taurex_tau[:3, 0]}")
    print(f"   JAX tau (first 3 layers): {np.array(jax_tau[:3, 0])}")
    
    if len(taurex_tau) == len(jax_tau):
        tau_ratio = float(jax_tau.mean()) / taurex_tau.mean()
        print(f"   Mean tau ratio (JAX/TauREx): {tau_ratio:.6f}")
    
    # Check intermediate values
    R_p = tm.planet.fullRadius
    R_s = tm.star.radius
    baseline = (R_p / R_s) ** 2
    
    print(f"\n5f. Normalization Analysis:")
    print(f"   Baseline (R_p/R_s)²: {baseline:.6e}")
    print(f"   TauREx / baseline: {taurex_spectrum[0] / baseline:.6f}")
    print(f"   JAX / baseline: {float(jax_spectrum[0]) / baseline:.6f}")
    
    if float(jax_spectrum[0]) < baseline:
        print(f"\n   ✗ CRITICAL: JAX result is below baseline!")
        print(f"      JAX: {float(jax_spectrum[0]):.6e}")
        print(f"      Baseline: {baseline:.6e}")
        print(f"      This means the 'atmosphere' is making the planet SMALLER,")
        print(f"      which is physically impossible!")
    
    return ratio_spectrum


def main():
    """Run all debug tests."""
    print("\n" + "="*80)
    print("JAX FORWARD MODEL DEBUG SUITE")
    print("="*80)
    print("\nGoal: Find the source of the 0.74x scale error")
    print("\nTests:")
    print("  1. High-res spectra (isolate binning vs forward model)")
    print("  2. Path integrals (check geometry)")
    print("  3. Absorption calculation (check formula)")
    print("  4. Binning flux conservation (check matrix)")
    print("  5. Step-by-step trace (find exact location of error)")
    print("\n" + "="*80)
    
    try:
        # Test 1: Where does the error appear?
        error_location = test_1_high_res_spectra()
        
        # Test 2: Check geometry
        geometry_ok = test_2_path_integrals()
        
        # Test 3: Check absorption formula
        absorption_ok = test_3_absorption_calculation()
        
        # Test 4: Check binning
        binning_ok = test_4_binning_flux_conservation()
        
        # Test 5: Detailed trace
        final_ratio = test_5_step_by_step_comparison()
        
        # Summary
        print("\n" + "="*80)
        print("DIAGNOSTIC SUMMARY")
        print("="*80)
        
        print(f"\n1. Error Location: {error_location}")
        print(f"2. Geometry OK: {geometry_ok}")
        print(f"3. Absorption Formula OK: {absorption_ok}")
        print(f"4. Binning OK: {binning_ok}")
        print(f"5. Final Ratio: {final_ratio:.4f}")
        
        print(f"\nConclusion:")
        if error_location == "binning":
            print(f"  → The high-res spectra match, so the problem is in BINNING")
        elif error_location == "forward_model":
            print(f"  → The high-res spectra already have the error, so problem is in")
            print(f"    FORWARD MODEL (path integrals, absorption calculation, or normalization)")
        
        if not absorption_ok:
            print(f"  → JAX result is below baseline - check absorption formula!")
        
        if not binning_ok:
            print(f"  → Binning matrix is not preserving flux!")
        
        if abs(final_ratio - 0.74) < 0.05:
            print(f"  → Error is consistent: {final_ratio:.4f} ≈ 0.74")
        
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
