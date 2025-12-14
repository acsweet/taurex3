"""
Debug the absorption calculation by comparing intermediate values.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from taurex.cache import OpacityCache, CIACache
from taurex.constants import RJUP, RSOL

from .adc_jax_utils import (
    load_auxiliary_data,
    load_ground_truth,
    create_taurex_model_from_adc_planet,
)
from .jax_backend import (
    extract_fitting_params,
    full_diff_create_forward_model,
    compute_absorption,
)


def main():
    """Compare TauREx vs JAX absorption intermediate values."""
    OpacityCache().clear_cache()
    OpacityCache().set_opacity_path('test_files/xsec/xsec_sampled_R15000_0.3-50')
    CIACache().set_cia_path('test_files/cia/HITRAN/data')
    
    planet_id = 1000
    aux_data = load_auxiliary_data('test_files/adc_2023/TrainingData/AuxillaryTable.csv', planet_id)
    gt = load_ground_truth('test_files/adc_2023/TrainingData/Ground Truth Package/FM_Parameter_Table.csv', planet_id)
    
    tm = create_taurex_model_from_adc_planet(aux_data, gt, nlayers=30)
    tm.initialize_profiles()
    
    # Test wavelength
    wn_test = np.linspace(5000.0, 5010.0, 3)
    
    print("="*80)
    print("ABSORPTION CALCULATION DEBUG")
    print("="*80)
    
    # TauREx computation
    print("\n1. TauREx Native Computation:")
    taurex_result = tm.model(wn_test)
    taurex_spectrum = taurex_result[1]
    taurex_tau = taurex_result[2]  # This is already exp(-tau)!
    
    print(f"   Transit depth: {taurex_spectrum.mean():.6e}")
    print(f"   Tau (exp(-tau)) shape: {taurex_tau.shape}")
    print(f"   Tau (exp(-tau)) range: {taurex_tau.min():.6e} - {taurex_tau.max():.6e}")
    
    # Manually recompute TauREx absorption to verify
    print("\n2. Manual TauREx Absorption Recomputation:")
    tau_exp = taurex_tau
    ap = tm.altitudeProfile[:, None]
    _dz = tm.deltaz[:, None]
    pradius = tm.planet.fullRadius
    sradius = tm.star.radius
    
    print(f"   Planet radius: {pradius/RJUP:.6f} R_jup = {pradius:.6e} m")
    print(f"   Star radius: {sradius/RSOL:.6f} R_sol = {sradius:.6e} m")
    print(f"   Altitude profile shape: {ap.shape}, range: {ap.min():.3e} - {ap.max():.3e} m")
    print(f"   dz shape: {_dz.shape}, mean: {_dz.mean():.3e} m")
    
    integral_taurex = np.sum((pradius + ap) * (1.0 - tau_exp) * _dz * 2.0, axis=0)
    absorption_taurex = ((pradius**2.0) + integral_taurex) / (sradius**2)
    
    print(f"\n   Intermediate values:")
    print(f"     pradius^2 = {pradius**2:.6e} m²")
    print(f"     sradius^2 = {sradius**2:.6e} m²")
    print(f"     Baseline (pradius²/sradius²) = {(pradius**2)/(sradius**2):.6e}")
    print(f"     Integral mean: {integral_taurex.mean():.6e} m²")
    print(f"     (1 - tau_exp) mean: {(1.0 - tau_exp).mean():.6e}")
    print(f"     (pradius + ap) mean: {(pradius + ap).mean():.6e} m")
    
    print(f"\n   Recomputed absorption mean: {absorption_taurex.mean():.6e}")
    print(f"   Original spectrum mean: {taurex_spectrum.mean():.6e}")
    print(f"   Match: {np.allclose(absorption_taurex, taurex_spectrum)}")
    
    # JAX computation
    print("\n3. JAX Computation:")
    params, _ = extract_fitting_params(tm, dtype=jnp.float64)
    jax_forward = full_diff_create_forward_model(tm, wn_test, pre_interpolate_opacity=False)
    jax_spectrum, jax_tau = jax_forward(params)
    
    print(f"   Transit depth: {float(jax_spectrum.mean()):.6e}")
    print(f"   Tau (NOT exp) shape: {jax_tau.shape}")
    print(f"   Tau (NOT exp) range: {float(jax_tau.min()):.6e} - {float(jax_tau.max()):.6e}")
    
    # Manually compute JAX absorption using same formula
    print("\n4. Manual JAX Absorption Computation:")
    jax_tau_exp = jnp.exp(-jax_tau)
    jax_ap = jnp.array(tm.altitudeProfile)[:, None]
    jax_dz = jnp.array(tm.deltaz)[:, None]
    jax_pradius = float(tm.planet.fullRadius)
    jax_sradius = float(tm.star.radius)
    
    print(f"   Planet radius: {jax_pradius/RJUP:.6f} R_jup = {jax_pradius:.6e} m")
    print(f"   Star radius: {jax_sradius/RSOL:.6f} R_sol = {jax_sradius:.6e} m")
    print(f"   Altitude profile shape: {jax_ap.shape}")
    print(f"   dz shape: {jax_dz.shape}")
    
    integral_jax = jnp.sum((jax_pradius + jax_ap) * (1.0 - jax_tau_exp) * jax_dz * 2.0, axis=0)
    absorption_jax_manual = ((jax_pradius**2.0) + integral_jax) / (jax_sradius**2)
    
    print(f"\n   Intermediate values:")
    print(f"     pradius^2 = {jax_pradius**2:.6e} m²")
    print(f"     sradius^2 = {jax_sradius**2:.6e} m²")
    print(f"     Baseline (pradius²/sradius²) = {(jax_pradius**2)/(jax_sradius**2):.6e}")
    print(f"     Integral mean: {float(integral_jax.mean()):.6e} m²")
    print(f"     (1 - tau_exp) mean: {float((1.0 - jax_tau_exp).mean()):.6e}")
    print(f"     (pradius + ap) mean: {float((jax_pradius + jax_ap).mean()):.6e} m")
    
    print(f"\n   Manually computed absorption mean: {float(absorption_jax_manual.mean()):.6e}")
    print(f"   JAX forward model result mean: {float(jax_spectrum.mean()):.6e}")
    print(f"   Match: {jnp.allclose(absorption_jax_manual, jax_spectrum)}")
    
    # Compare tau values
    print("\n5. Tau Comparison:")
    print(f"   TauREx tau (already exp(-tau)):")
    print(f"     Mean: {tau_exp.mean():.6e}")
    print(f"     First layer, first wavelength: {tau_exp[0, 0]:.6e}")
    
    print(f"   JAX tau (raw tau, before exp):")
    print(f"     Mean: {float(jax_tau.mean()):.6e}")
    print(f"     First layer, first wavelength: {float(jax_tau[0, 0]):.6e}")
    
    print(f"   JAX tau_exp (after applying exp(-tau)):")
    print(f"     Mean: {float(jax_tau_exp.mean()):.6e}")
    print(f"     First layer, first wavelength: {float(jax_tau_exp[0, 0]):.6e}")
    
    # Check if tau values are comparable
    print(f"\n   Comparing tau_exp values:")
    print(f"     TauREx tau_exp mean: {tau_exp.mean():.6e}")
    print(f"     JAX tau_exp mean: {float(jax_tau_exp.mean()):.6e}")
    print(f"     Ratio (JAX/TauREx): {float(jax_tau_exp.mean()) / tau_exp.mean():.6f}")
    
    # Compare (1 - tau_exp) which is the absorption term
    print(f"\n   Comparing (1 - tau_exp) - the atmospheric absorption:")
    taurex_absorption_term = (1.0 - tau_exp).mean()
    jax_absorption_term = float((1.0 - jax_tau_exp).mean())
    print(f"     TauREx (1 - tau_exp) mean: {taurex_absorption_term:.6e}")
    print(f"     JAX (1 - tau_exp) mean: {jax_absorption_term:.6e}")
    print(f"     Ratio (JAX/TauREx): {jax_absorption_term / taurex_absorption_term:.6f}")
    
    # Final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(f"TauREx spectrum mean: {taurex_spectrum.mean():.6e}")
    print(f"JAX spectrum mean: {float(jax_spectrum.mean()):.6e}")
    print(f"Ratio (JAX/TauREx): {float(jax_spectrum.mean()) / taurex_spectrum.mean():.4f}")
    print(f"Expected ratio: 1.0000")
    print(f"Actual ratio: {float(jax_spectrum.mean()) / taurex_spectrum.mean():.4f}")
    
    if float(jax_spectrum.mean()) / taurex_spectrum.mean() < 0.8:
        print(f"\n✗ JAX is producing spectra that are TOO SMALL by ~{(1.0 - float(jax_spectrum.mean()) / taurex_spectrum.mean()) * 100:.1f}%")
        
        # Check if the problem is in the integral
        ratio_integral = float(integral_jax.mean()) / integral_taurex.mean()
        print(f"\nIntegral ratio (JAX/TauREx): {ratio_integral:.4f}")
        
        if ratio_integral < 0.8:
            print(f"  → Problem is in the INTEGRAL calculation")
            print(f"     JAX integral is {(1.0 - ratio_integral) * 100:.1f}% too small")
        else:
            print(f"  → Integral looks OK, problem might be elsewhere")


if __name__ == "__main__":
    main()
