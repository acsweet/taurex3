"""
Example: Fit a single ADC 2023 planet using JAX gradients.

This demonstrates the complete pipeline:
1. Load planet data (spectrum, auxiliary info, ground truth)
2. Create TauREx transmission model
3. Create JAX differentiable forward model
4. Run gradient-based optimization
5. Plot results

Usage:
    python example_fit_single_planet.py

Or customize:
    python example_fit_single_planet.py --planet-id 100 --steps 1000 --float32
"""

import sys
import os
import jax
import jax.numpy as jnp

from taurex.cache import OpacityCache, CIACache
from .adc_jax_utils import fit_adc_planet, plot_adc_fit_results


def main(planet_id=1000, steps=500, lr=1e-3, nlayers=30, use_float64=True):
    """Run atmospheric retrieval on a single ADC planet."""
    
    # ========== CONFIGURATION ==========
    PLANET_ID = planet_id
    USE_FLOAT64 = use_float64
    STEPS = steps
    LEARNING_RATE = lr
    NLAYERS = nlayers
    
    # Parameters to fit
    FIT_PARAMS = ['planet_radius', 'T', 'H2O', 'CO2', 'CO', 'CH4', 'NH3']
    
    # Data paths
    DATA_DIR = 'test_files/adc_2023/TrainingData'
    XSEC_PATH = 'test_files/xsec/xsec_sampled_R15000_0.3-50'
    CIA_PATH = 'test_files/cia/HITRAN/data'
    
    # ========== SETUP JAX ==========
    jax.config.update("jax_enable_x64", USE_FLOAT64)
    DTYPE = jnp.float64 if USE_FLOAT64 else jnp.float32
    
    print("=" * 70)
    print("ADC 2023 Atmospheric Retrieval with JAX")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Planet ID: {PLANET_ID}")
    print(f"  Precision: {'float64' if USE_FLOAT64 else 'float32'}")
    print(f"  Device: {jax.devices()[0]}")
    print(f"  Optimization steps: {STEPS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print("=" * 70)
    
    # ========== SETUP OPACITY CACHE ==========
    print("\nSetting up opacity cache...")
    OpacityCache().clear_cache()
    OpacityCache().set_opacity_path(XSEC_PATH)
    CIACache().set_cia_path(CIA_PATH)
    print("  ✓ Opacity cache ready")
    
    # ========== RUN FITTING ==========
    result = fit_adc_planet(
        planet_id=PLANET_ID,
        data_dir=DATA_DIR,
        fit_params=FIT_PARAMS,
        steps=STEPS,
        lr=LEARNING_RATE,
        nlayers=NLAYERS,
        dtype=DTYPE,
        verbose=True
    )
    
    # ========== PLOT RESULTS ==========
    print("\n" + "=" * 70)
    print("Generating plots...")
    print("=" * 70)
    
    plot_adc_fit_results(
        result,
        save_path=f'results_planet_{PLANET_ID}.png'
    )
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 70)
    print("RETRIEVAL COMPLETE!")
    print("=" * 70)
    print(f"Planet: {PLANET_ID}")
    print(f"Final loss: {result['losses'][-1]:.6e}")
    
    if result['ground_truth'] is not None:
        # Calculate overall accuracy
        fit_params = FIT_PARAMS
        errors = []
        for param in fit_params:
            if param in result['ground_truth']:
                final = float(result['final_params'][param])
                gt = float(result['ground_truth'][param])
                if gt != 0:
                    rel_error = abs(final - gt) / abs(gt)
                    errors.append(rel_error)
        
        if errors:
            mean_error = sum(errors) / len(errors) * 100
            print(f"Mean relative error: {mean_error:.2f}%")
    
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    # Parse command line arguments (optional)
    import argparse
    
    parser = argparse.ArgumentParser(description='Fit ADC 2023 planet with JAX')
    parser.add_argument('--planet-id', type=int, default=1000,
                       help='Planet ID to fit (default: 1000)')
    parser.add_argument('--steps', type=int, default=500,
                       help='Optimization steps (default: 500)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--nlayers', type=int, default=30,
                       help='Number of atmospheric layers (default: 30)')
    parser.add_argument('--float32', action='store_true',
                       help='Use float32 instead of float64')
    
    args = parser.parse_args()
    
    # Run main with parsed arguments
    try:
        result = main(
            planet_id=args.planet_id,
            steps=args.steps,
            lr=args.lr,
            nlayers=args.nlayers,
            use_float64=not args.float32
        )
        print("\n✓ Success!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
