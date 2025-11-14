#!/usr/bin/env python3
"""
Test different loss functions for ADC planet 1000 retrieval.

Compares MSE, Student-t, log-normal, and Gaussian losses to see which
handles the ADC retrieval problem better.
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp
from experiment import full_diff_create_forward_model, fit_with_value_and_grad_adam
from adc_jax_utils import (
    load_planet_spectrum,
    create_taurex_model_from_adc_planet,
    create_adc_jax_forward_model,
    extract_fitting_params,
    load_adc_ground_truth
)

jax.config.update("jax_enable_x64", True)

# Planet ID
PLANET_ID = 1000
STEPS = 500
LR = 0.001

# Loss functions to test
LOSS_TYPES = [
    ("mse", {}),
    ("studentt", {"nu": 4.0}),  # Robust to outliers
    ("lognormal", {}),  # Multiplicative noise
    ("gaussian", {}),  # Standard Gaussian NLL
]

def compute_chi_squared(obs_y, pred_y, obs_err):
    """Compute chi-squared goodness of fit."""
    residuals = (obs_y - pred_y) / obs_err
    chi2 = np.sum(residuals**2)
    return chi2, chi2 / len(obs_y)

def test_loss_function(loss_type, loss_kwargs, aux_data, spectrum_dict, ground_truth):
    """Test a single loss function."""
    print(f"\n{'='*60}")
    print(f"Testing loss: {loss_type}")
    if loss_kwargs:
        print(f"Parameters: {loss_kwargs}")
    print('='*60)
    
    # Create model
    tm = create_taurex_model_from_adc_planet(
        aux_data,
        ground_truth=ground_truth,
        nlayers=30
    )
    
    # Create JAX forward model
    forward_model, obs_spectrum = create_adc_jax_forward_model(
        tm,
        spectrum_dict,
        use_full_diff=True,
        dtype=jnp.float64
    )
    
    # Extract parameters
    params, param_info = extract_fitting_params(tm, dtype=jnp.float64)
    
    # Fitting parameters
    fit_params = [
        'planet_radius', 
        'T', 
        'H2O_gas', 
        'CO2_gas', 
        'CO_gas', 
        'CH4_gas', 
        'NH3_gas'
    ]
    
    # Setup observation data
    obs_y = jnp.asarray(spectrum_dict['spectrum'], dtype=jnp.float64)
    obs_err = jnp.asarray(spectrum_dict['noise'], dtype=jnp.float64)
    
    # Store initial values
    initial_params = {k: params[k] for k in fit_params}
    
    # Compute initial fit quality
    initial_pred = forward_model(params)
    initial_chi2, initial_red_chi2 = compute_chi_squared(
        np.array(obs_y), 
        np.array(initial_pred), 
        np.array(obs_err)
    )
    
    print(f"\nInitial fit:")
    print(f"  χ² = {initial_chi2:.2f}")
    print(f"  χ²/dof = {initial_red_chi2:.3f}")
    
    # Run optimization
    final_params, losses = fit_with_value_and_grad_adam(
        forward_binned=forward_model,
        observed_y=obs_y,
        observed_err=obs_err,
        init_params=params,
        param_info=param_info,
        fit_params=fit_params,
        steps=STEPS,
        lr=LR,
        clip_norm=1.0,
        print_every=100,
        nan_guard=True,
        loss=loss_type,
        loss_kwargs=loss_kwargs
    )
    
    # Compute final fit quality
    final_pred = forward_model(final_params)
    final_chi2, final_red_chi2 = compute_chi_squared(
        np.array(obs_y), 
        np.array(final_pred), 
        np.array(obs_err)
    )
    
    print(f"\nFinal fit:")
    print(f"  χ² = {final_chi2:.2f}")
    print(f"  χ²/dof = {final_red_chi2:.3f}")
    print(f"  Loss improvement: {losses[0]:.2f} -> {losses[-1]:.2f}")
    
    # Compare parameters
    print(f"\nParameter changes:")
    for k in fit_params:
        initial = float(initial_params[k])
        final = float(final_params[k])
        change = final - initial
        pct_change = (change / initial) * 100 if initial != 0 else 0
        
        # Check if it's a gas mixing ratio
        if '_gas' in k:
            print(f"  {k:12s}: {initial:.3e} -> {final:.3e} ({pct_change:+.1f}%)")
        else:
            print(f"  {k:12s}: {initial:.6f} -> {final:.6f} ({pct_change:+.1f}%)")
    
    # Return results summary
    return {
        'loss_type': loss_type,
        'loss_kwargs': loss_kwargs,
        'initial_chi2': initial_red_chi2,
        'final_chi2': final_red_chi2,
        'chi2_improvement': initial_red_chi2 - final_red_chi2,
        'final_params': final_params,
        'losses': losses
    }

def main():
    print(f"Testing loss functions for Planet {PLANET_ID}")
    print(f"Steps: {STEPS}, Learning rate: {LR}")
    
    # Load planet data
    print("\nLoading planet data...")
    aux_data, spectrum_dict = load_planet_spectrum(PLANET_ID)
    
    # Load ground truth
    try:
        ground_truth = load_adc_ground_truth(PLANET_ID)
        print(f"Ground truth loaded")
    except:
        ground_truth = None
        print(f"No ground truth available")
    
    # Test each loss function
    results = []
    for loss_type, loss_kwargs in LOSS_TYPES:
        try:
            result = test_loss_function(
                loss_type, 
                loss_kwargs, 
                aux_data, 
                spectrum_dict, 
                ground_truth
            )
            results.append(result)
        except Exception as e:
            print(f"\nERROR testing {loss_type}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print('='*60)
    print(f"{'Loss Type':<15s} {'Initial χ²/dof':>12s} {'Final χ²/dof':>12s} {'Improvement':>12s}")
    print('-'*60)
    
    for result in results:
        loss_name = result['loss_type']
        if result['loss_kwargs']:
            loss_name += f" (nu={result['loss_kwargs'].get('nu', '')})"
        
        print(f"{loss_name:<15s} {result['initial_chi2']:>12.3f} {result['final_chi2']:>12.3f} {result['chi2_improvement']:>+12.3f}")
    
    # Find best result
    best_result = min(results, key=lambda r: r['final_chi2'])
    print(f"\nBest loss function: {best_result['loss_type']}")
    print(f"  Final χ²/dof: {best_result['final_chi2']:.3f}")
    
    return results

if __name__ == '__main__':
    results = main()
