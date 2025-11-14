#!/usr/bin/env python
"""
Example: Using float32 precision with JAX TauREx implementation

This shows how to properly configure and use float32 for ~2x speedup
while maintaining numerical stability where needed.
"""

import jax
import jax.numpy as jnp
import numpy as np

# ========================================================================
# STEP 1: Configure JAX for float32 (BEFORE importing experiment.py)
# ========================================================================

# This tells JAX to use float32 as default instead of float64
jax.config.update("jax_enable_x64", False)

print("JAX Configuration:")
print(f"  x64 enabled: {jax.config.read('jax_enable_x64')}")
print(f"  Default device: {jax.devices()[0]}")
print(f"  Default dtype: {jnp.array(1.0).dtype}")
print()

# ========================================================================
# STEP 2: Import and setup (after JAX config)
# ========================================================================

from experiment import (
    extract_fitting_params,
    full_diff_prepare_model_data,
    load_opacity_data,
    full_diff_create_binned_forward_model,
)

from taurex.model import TransmissionModel
from taurex.chemistry import TaurexChemistry, ConstantGas
from taurex.planet import Planet
from taurex.stellar import BlackbodyStar
from taurex.temperature import Isothermal
from taurex.contributions import AbsorptionContribution, CIAContribution, RayleighContribution
from taurex.cache import OpacityCache, CIACache
from taurex.data.spectrum.observed import ObservedSpectrum


def example_float32_forward_model():
    """
    Example: Create and run a forward model in float32.
    """
    
    print("="*70)
    print("Example: Float32 Forward Model")
    print("="*70)
    
    # ========================================================================
    # Setup paths (adjust for your system)
    # ========================================================================
    
    xsec_path = "/Users/asweet/Code/research/taurex3/test_files/Input/xsec/xsec_sampled_R15000_0.3-50"
    cia_path = "/Users/asweet/Code/research/taurex3/test_files/Input/cia/HITRAN/data"
    obs_path = "/Users/asweet/Code/research/taurex3/test_files/quickstart.dat"
    
    OpacityCache().clear_cache()
    OpacityCache().set_opacity_path(xsec_path)
    CIACache().set_cia_path(cia_path)
    
    # ========================================================================
    # Build TauREx model
    # ========================================================================
    
    print("\n1. Building TauREx model...")
    
    planet = Planet(planet_radius=1.0, planet_mass=1.0)
    star = BlackbodyStar(temperature=5700.0, radius=1.0)
    
    chemistry = TaurexChemistry(fill_gases=['H2', 'He'], ratio=0.172)
    chemistry.addGas(ConstantGas('H2O', mix_ratio=1.2e-4))
    chemistry.addGas(ConstantGas('N2', mix_ratio=3.00739e-9))
    
    isothermal = Isothermal(T=1500.0)
    
    tm = TransmissionModel(
        planet=planet,
        temperature_profile=isothermal,
        chemistry=chemistry,
        star=star,
        atm_min_pressure=1e-0,
        atm_max_pressure=1e6,
        nlayers=30
    )
    
    tm.add_contribution(AbsorptionContribution())
    tm.add_contribution(CIAContribution(cia_pairs=['H2-H2', 'H2-He']))
    tm.add_contribution(RayleighContribution())
    
    tm.build()
    print(f"   ✓ Model built with {tm.nLayers} layers")
    
    # ========================================================================
    # Load observation
    # ========================================================================
    
    print("\n2. Loading observation data...")
    obs = ObservedSpectrum(obs_path)
    print(f"   ✓ Loaded {len(obs.spectrum)} data points")
    
    # ========================================================================
    # Extract parameters with EXPLICIT float32 conversion
    # ========================================================================
    
    print("\n3. Extracting parameters (float32)...")
    
    # NEW: Pass dtype=jnp.float32 to ensure float32
    params, param_info = extract_fitting_params(tm, dtype=jnp.float32)
    
    print(f"   ✓ Extracted {len(params)} parameters")
    print("   Parameter dtypes:")
    for key, val in params.items():
        print(f"     {key:20s}: {val.dtype}")
    
    # ========================================================================
    # Create forward model with float32
    # ========================================================================
    
    print("\n4. Creating JAX forward model (float32)...")
    
    # The forward model creation will use float32 internally
    # because we've set jax_enable_x64=False
    forward_binned = full_diff_create_binned_forward_model(tm, obs)
    
    print("   ✓ Forward model created")
    
    # ========================================================================
    # Run forward model
    # ========================================================================
    
    print("\n5. Running forward model...")
    
    # First call triggers JIT compilation (slow)
    result = forward_binned(params)
    
    print(f"   ✓ Model executed")
    print(f"   Result dtype: {result.dtype}")
    print(f"   Result shape: {result.shape}")
    print(f"   Result range: {float(jnp.min(result)):.6e} to {float(jnp.max(result)):.6e}")
    
    # Check for numerical issues
    has_nan = jnp.any(jnp.isnan(result))
    has_inf = jnp.any(jnp.isinf(result))
    
    if has_nan or has_inf:
        print("\n   ❌ WARNING: Numerical issues detected!")
        print(f"      NaN values: {has_nan}")
        print(f"      Inf values: {has_inf}")
        print("   → You may need float64 for numerical stability")
    else:
        print("\n   ✅ No numerical issues (NaN/Inf) detected")
        print("   → Float32 appears to work for this model!")
    
    # ========================================================================
    # Benchmark
    # ========================================================================
    
    print("\n6. Benchmarking...")
    
    import time
    
    # Warmup (compilation already done)
    _ = forward_binned(params)
    
    # Time multiple runs
    n_runs = 10
    start = time.time()
    for _ in range(n_runs):
        result = forward_binned(params)
    result.block_until_ready()  # Wait for GPU
    elapsed = time.time() - start
    
    avg_time = elapsed / n_runs
    print(f"   Average time: {avg_time*1000:.2f} ms ({n_runs} runs)")
    print(f"   Device: {jax.devices()[0]}")
    
    return result, params


def compare_float32_vs_float64():
    """
    Compare float32 vs float64 results and performance.
    """
    
    print("\n" + "="*70)
    print("Comparison: Float32 vs Float64")
    print("="*70)
    
    # You would need to run this twice with different configs
    # or dynamically change the config (not recommended in production)
    
    print("\nTo compare:")
    print("1. Run this script with jax_enable_x64=False (float32)")
    print("2. Run again with jax_enable_x64=True (float64)")
    print("3. Compare:")
    print("   - Result values (should be very close)")
    print("   - Execution time (float32 should be ~1.5-2x faster)")
    print("   - Memory usage (float32 uses ~half)")
    print("   - Numerical stability (check for NaN/Inf)")


if __name__ == "__main__":
    try:
        result, params = example_float32_forward_model()
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"✓ Successfully ran forward model in float32")
        print(f"  Result dtype: {result.dtype}")
        print(f"  GPU used: {jax.devices()[0]}")
        print(f"  x64 enabled: {jax.config.read('jax_enable_x64')}")
        
        if result.dtype == jnp.float32:
            print("\n✅ Float32 configuration working correctly!")
            print("   Expected benefits:")
            print("   - ~1.5-2x faster than float64")
            print("   - ~half the memory usage")
            print("   - Still GPU-accelerated")
        else:
            print(f"\n⚠️  Result is {result.dtype}, not float32!")
            print("   Check that:")
            print("   1. jax_enable_x64 = False is set BEFORE imports")
            print("   2. extract_fitting_params called with dtype=jnp.float32")
            print("   3. All data properly converted to float32")
        
        print("\n" + "="*70)
        
        # Additional comparison info
        compare_float32_vs_float64()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting:")
        print("1. Make sure opacity and CIA paths are correct")
        print("2. Check that JAX is properly installed")
        print("3. Verify GPU is available")
        print("4. Try with float64 first (jax_enable_x64=True)")
