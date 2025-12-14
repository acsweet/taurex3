#!/usr/bin/env python
"""
Test script to identify overflow issues with/without x64 precision.
This will help diagnose where the overflow occurs and suggest solutions.
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys

def test_without_x64():
    """Test the forward model WITHOUT x64 to identify overflow."""
    print("\n" + "="*70)
    print("Testing WITHOUT x64 precision (float32)")
    print("="*70)
    
    jax.config.update("jax_enable_x64", False)
    
    try:
        from .jax_backend import full_diff_create_binned_forward_model, extract_fitting_params
        from taurex.model import TransmissionModel
        from taurex.chemistry import TaurexChemistry, ConstantGas
        from taurex.planet import Planet
        from taurex.stellar import BlackbodyStar
        from taurex.temperature import Isothermal
        from taurex.contributions import AbsorptionContribution, CIAContribution, RayleighContribution
        from taurex.cache import OpacityCache, CIACache
        from taurex.data.spectrum.observed import ObservedSpectrum
        
        # Setup minimal test
        xsec_path = "/Users/asweet/Code/research/taurex3/test_files/Input/xsec/xsec_sampled_R15000_0.3-50"
        cia_path = "/Users/asweet/Code/research/taurex3/test_files/Input/cia/HITRAN/data"
        obs_path = "/Users/asweet/Code/research/taurex3/test_files/quickstart.dat"
        
        OpacityCache().clear_cache()
        OpacityCache().set_opacity_path(xsec_path)
        CIACache().set_cia_path(cia_path)
        
        planet = Planet(planet_radius=1.0, planet_mass=1.0)
        star = BlackbodyStar(temperature=5700.0, radius=1.0)
        chemistry = TaurexChemistry(fill_gases=['H2', 'He'], ratio=0.172)
        chemistry.addGas(ConstantGas('H2O', mix_ratio=1.2e-4))
        
        tm = TransmissionModel(
            planet=planet,
            temperature_profile=Isothermal(T=1500.0),
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
        
        obs = ObservedSpectrum(obs_path)
        forward_binned = full_diff_create_binned_forward_model(tm, obs)
        params, _ = extract_fitting_params(tm)
        
        print("Forward model created, running...")
        result = forward_binned(params)
        
        print(f"✅ SUCCESS with float32!")
        print(f"   Result range: {float(jnp.min(result)):.6e} to {float(jnp.max(result)):.6e}")
        print(f"   Contains NaN: {jnp.any(jnp.isnan(result))}")
        print(f"   Contains Inf: {jnp.any(jnp.isinf(result))}")
        return True
        
    except Exception as e:
        print(f"❌ OVERFLOW/ERROR with float32:")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_x64():
    """Test the forward model WITH x64."""
    print("\n" + "="*70)
    print("Testing WITH x64 precision (float64)")
    print("="*70)
    
    jax.config.update("jax_enable_x64", True)
    
    try:
        from .jax_backend import full_diff_create_binned_forward_model, extract_fitting_params
        from taurex.model import TransmissionModel
        from taurex.chemistry import TaurexChemistry, ConstantGas
        from taurex.planet import Planet
        from taurex.stellar import BlackbodyStar
        from taurex.temperature import Isothermal
        from taurex.contributions import AbsorptionContribution, CIAContribution, RayleighContribution
        from taurex.cache import OpacityCache, CIACache
        from taurex.data.spectrum.observed import ObservedSpectrum
        
        # Setup minimal test
        xsec_path = "/Users/asweet/Code/research/taurex3/test_files/Input/xsec/xsec_sampled_R15000_0.3-50"
        cia_path = "/Users/asweet/Code/research/taurex3/test_files/Input/cia/HITRAN/data"
        obs_path = "/Users/asweet/Code/research/taurex3/test_files/quickstart.dat"
        
        OpacityCache().clear_cache()
        OpacityCache().set_opacity_path(xsec_path)
        CIACache().set_cia_path(cia_path)
        
        planet = Planet(planet_radius=1.0, planet_mass=1.0)
        star = BlackbodyStar(temperature=5700.0, radius=1.0)
        chemistry = TaurexChemistry(fill_gases=['H2', 'He'], ratio=0.172)
        chemistry.addGas(ConstantGas('H2O', mix_ratio=1.2e-4))
        
        tm = TransmissionModel(
            planet=planet,
            temperature_profile=Isothermal(T=1500.0),
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
        
        obs = ObservedSpectrum(obs_path)
        forward_binned = full_diff_create_binned_forward_model(tm, obs)
        params, _ = extract_fitting_params(tm)
        
        print("Forward model created, running...")
        result = forward_binned(params)
        
        print(f"✅ SUCCESS with float64!")
        print(f"   Result range: {float(jnp.min(result)):.6e} to {float(jnp.max(result)):.6e}")
        print(f"   Contains NaN: {jnp.any(jnp.isnan(result))}")
        print(f"   Contains Inf: {jnp.any(jnp.isinf(result))}")
        return True
        
    except Exception as e:
        print(f"❌ ERROR with float64:")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_gpu_performance():
    """Check if x64 impacts GPU performance."""
    print("\n" + "="*70)
    print("GPU Performance Test: float32 vs float64")
    print("="*70)
    
    import time
    
    # Create test arrays
    size = 10000
    x = jnp.linspace(0, 100, size)
    
    # Test float32
    jax.config.update("jax_enable_x64", False)
    x32 = jnp.array(x, dtype=jnp.float32)
    
    # Warmup
    _ = jnp.exp(-x32)
    
    start = time.time()
    for _ in range(1000):
        result = jnp.exp(-x32)
    result.block_until_ready()
    time32 = time.time() - start
    
    # Test float64
    jax.config.update("jax_enable_x64", True)
    x64 = jnp.array(x, dtype=jnp.float64)
    
    # Warmup
    _ = jnp.exp(-x64)
    
    start = time.time()
    for _ in range(1000):
        result = jnp.exp(-x64)
    result.block_until_ready()
    time64 = time.time() - start
    
    print(f"float32 time: {time32:.4f} s")
    print(f"float64 time: {time64:.4f} s")
    print(f"Slowdown: {time64/time32:.2f}x")
    print(f"\nDevice: {jax.devices()[0]}")
    print(f"GPU memory usage: ~2x for float64")


if __name__ == "__main__":
    print("="*70)
    print("Precision and Overflow Diagnostic Tool")
    print("="*70)
    
    # Test 1: Check GPU
    print(f"\nGPU Status: {jax.devices()}")
    print(f"Backend: {jax.default_backend()}")
    
    # Test 2: Performance impact
    check_gpu_performance()
    
    # Test 3: Run without x64
    success_32 = test_without_x64()
    
    # Test 4: Run with x64
    success_64 = test_with_x64()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if success_32 and success_64:
        print("✅ Both float32 and float64 work!")
        print("   You may not actually need x64 for numerical stability.")
        print("   Consider using float32 for better performance.")
    elif not success_32 and success_64:
        print("⚠️  float32 causes overflow, but float64 works.")
        print("   You NEED x64 enabled for numerical stability.")
        print("   GPU is still being used - just with 64-bit precision.")
    elif success_32 and not success_64:
        print("❌ Unexpected: float32 works but float64 fails.")
        print("   This is unusual - may indicate a different issue.")
    else:
        print("❌ Both failed - there's a different problem.")
    
    print("\nRECOMMENDATIONS:")
    print("-" * 70)
    print("1. x64 DOES work on GPU - you're not losing GPU acceleration")
    print("2. x64 uses 2x memory and may be ~1.5-2x slower")
    print("3. For atmospheric modeling, x64 is often necessary due to:")
    print("   - Large pressure ranges (1e-4 to 1e6 Pa)")
    print("   - Small cross-sections (1e-30 cm²)")
    print("   - Exponentials in optical depth")
    print("4. If overflow only happens in specific cases, you can:")
    print("   - Use mixed precision (store in float32, compute in float64)")
    print("   - Rescale problematic quantities")
    print("   - Use log-space for extreme ranges")
    print("="*70)
