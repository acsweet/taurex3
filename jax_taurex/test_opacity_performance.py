"""
Test script to compare performance of opacity interpolation strategies.
Run this to see the speed difference between GPU-optimized (full resolution)
and CPU-optimized (pre-interpolated) modes.
"""

import time
import numpy as np
from taurex.cache import OpacityCache, CIACache
from taurex.model import TransmissionModel
from taurex.planet import Planet
from taurex.stellar import BlackbodyStar
from taurex.chemistry import TaurexChemistry, ConstantGas
from taurex.temperature import Isothermal
from taurex.contributions import AbsorptionContribution, CIAContribution, RayleighContribution
from taurex.data.spectrum.observed import ObservedSpectrum

from .jax_backend import full_diff_create_binned_forward_model, extract_fitting_params


def setup_model():
    """Setup test model."""
    # Setup paths
    xsec_path = "/Users/asweet/Code/research/taurex3/test_files/Input/xsec/xsec_sampled_R15000_0.3-50"
    cia_path = "/Users/asweet/Code/research/taurex3/test_files/Input/cia/HITRAN/data"
    
    OpacityCache().clear_cache()
    OpacityCache().set_opacity_path(xsec_path)
    CIACache().set_cia_path(cia_path)
    
    # Build model
    planet = Planet(planet_radius=1.0, planet_mass=1.0)
    star = BlackbodyStar(temperature=5700.0, radius=1.0)
    chemistry = TaurexChemistry(fill_gases=['H2', 'He'], ratio=0.172)
    chemistry.addGas(ConstantGas('H2O', mix_ratio=1.2e-4))
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
    
    return tm


def benchmark_mode(tm, obs, mode_name, pre_interpolate, n_calls=10):
    """Benchmark one mode."""
    print(f"\n{'='*60}")
    print(f"Testing: {mode_name}")
    print(f"{'='*60}")
    
    # Create forward model
    start = time.time()
    forward_binned = full_diff_create_binned_forward_model(
        tm, obs, 
        pre_interpolate_opacity=pre_interpolate,
        debug_coverage=False
    )
    setup_time = time.time() - start
    print(f"Setup time: {setup_time:.2f} seconds")
    
    # Get parameters
    params, _ = extract_fitting_params(tm)
    
    # Warm-up call (JIT compilation)
    print("Warming up (JIT compilation)...")
    _ = forward_binned(params)
    
    # Benchmark forward passes
    print(f"Running {n_calls} forward passes...")
    times = []
    for i in range(n_calls):
        start = time.time()
        result = forward_binned(params)
        elapsed = time.time() - start
        times.append(elapsed)
        if i == 0:
            print(f"  Call 1: {elapsed:.4f} seconds")
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\nResults:")
    print(f"  Mean time per call: {mean_time:.4f} ± {std_time:.4f} seconds")
    print(f"  Total time: {sum(times):.4f} seconds")
    
    return mean_time, result


def main():
    print("="*60)
    print("OPACITY INTERPOLATION PERFORMANCE TEST")
    print("="*60)
    
    # Setup
    print("\nSetting up model...")
    tm = setup_model()
    obs_path = "/Users/asweet/Code/research/taurex3/test_files/quickstart.dat"
    obs = ObservedSpectrum(obs_path)
    
    # Test GPU-optimized mode (full resolution)
    time_full, result_full = benchmark_mode(
        tm, obs, 
        "GPU Mode (Full Resolution)", 
        pre_interpolate=False,
        n_calls=10
    )
    
    # Test CPU-optimized mode (pre-interpolated)
    time_pre, result_pre = benchmark_mode(
        tm, obs,
        "CPU Mode (Pre-Interpolated)",
        pre_interpolate=True,
        n_calls=10
    )
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"GPU mode time: {time_full:.4f} seconds/call")
    print(f"CPU mode time: {time_pre:.4f} seconds/call")
    speedup = time_full / time_pre
    print(f"Speedup: {speedup:.2f}x")
    
    # Check accuracy
    diff = np.abs(result_full - result_pre)
    rel_diff = diff / (np.abs(result_full) + 1e-12)
    print(f"\nAccuracy check:")
    print(f"  Max absolute difference: {diff.max():.2e}")
    print(f"  Mean absolute difference: {diff.mean():.2e}")
    print(f"  Max relative difference: {rel_diff.max():.2e}")
    print(f"  Mean relative difference: {rel_diff.mean():.2e}")
    
    if rel_diff.max() < 1e-6:
        print("  ✅ Results match to machine precision!")
    elif rel_diff.max() < 1e-3:
        print("  ✅ Results match well (< 0.1%)")
    else:
        print("  ⚠️  Warning: Results differ significantly")
    
    print(f"\n{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")
    if speedup > 1.5:
        print(f"💡 For CPU execution, use pre_interpolate_opacity=True")
        print(f"   This gives {speedup:.2f}x speedup with negligible accuracy loss.")
    else:
        print(f"💡 Both modes perform similarly on this hardware.")
    
    print(f"\nFor GPU execution, use pre_interpolate_opacity=False (default)")
    print(f"to maximize GPU memory bandwidth utilization.")


if __name__ == "__main__":
    main()
