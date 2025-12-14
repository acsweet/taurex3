"""
Diagnose why gradient optimization is diverging from ground truth.

This script checks:
1. Initial spectrum quality (should be close to observed if using posterior GT)
2. Gradient directions and magnitudes
3. Whether parameters are hitting bounds
4. Loss function behavior
"""

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from .adc_jax_utils import load_ground_truth, load_planet_spectrum, load_auxiliary_data
from .adc_jax_utils import create_taurex_model_from_adc_planet, create_adc_jax_forward_model
from taurex.cache import OpacityCache, CIACache
from .jax_backend import extract_fitting_params
import matplotlib.pyplot as plt

# ADC Prior bounds
ADC_PRIORS = {
    'planet_radius': [0.1, 3.0],
    'T': [0.0, 7000.0],
    'H2O': [1e-12, 0.1],
    'CO2': [1e-12, 0.1],
    'CO': [1e-12, 0.1],
    'CH4': [1e-12, 0.1],
    'NH3': [1e-12, 0.1],
}

def check_bounds(params, priors):
    """Check if parameters are within ADC priors."""
    print("\nParameter Bounds Check:")
    print("="*70)
    for param, bounds in priors.items():
        if param in params:
            val = float(params[param])
            lo, hi = bounds
            in_bounds = lo <= val <= hi
            pct = (val - lo) / (hi - lo) * 100 if hi > lo else 0
            status = "✓" if in_bounds else "✗ OUT OF BOUNDS"
            print(f"{param:15s}: {val:12.6e} [{lo:10.2e}, {hi:10.2e}] {pct:6.1f}% {status}")

def compute_loss_and_grad(forward_model, params, obs_y, obs_err):
    """Compute MSE loss and gradients."""
    def loss_fn(p):
        pred = jnp.asarray(forward_model(p))
        residuals = (obs_y - pred) / obs_err
        return jnp.mean(residuals**2)
    
    loss_val = loss_fn(params)
    grads = jax.grad(loss_fn)(params)
    
    return float(loss_val), grads

def main():
    planet_id = 1000
    data_dir = 'test_files/adc_2023/TrainingData'
    
    # Setup
    print("Setting up opacity cache...")
    OpacityCache().clear_cache()
    OpacityCache().set_opacity_path('test_files/xsec/xsec_sampled_R15000_0.3-50')
    CIACache().set_cia_path('test_files/cia/HITRAN/data')
    
    # Load data
    print(f"\nLoading planet {planet_id}...")
    spectral_path = f'{data_dir}/SpectralData.hdf5'
    aux_path = f'{data_dir}/AuxillaryTable.csv'
    tracedata_path = f'{data_dir}/Ground Truth Package/Tracedata.hdf5'
    
    spectrum_dict = load_planet_spectrum(spectral_path, planet_id)
    aux_data = load_auxiliary_data(aux_path, planet_id)
    ground_truth = load_ground_truth(tracedata_path, planet_id, sample_method='weighted_mean')
    
    # Create model
    print("\nCreating model...")
    tm = create_taurex_model_from_adc_planet(aux_data, ground_truth, nlayers=30)
    forward_model, obs_spectrum = create_adc_jax_forward_model(tm, spectrum_dict)
    
    # Extract parameters
    params, param_info = extract_fitting_params(tm, dtype=jnp.float64)
    
    obs_y = jnp.array(spectrum_dict['spectrum'])
    obs_err = jnp.array(spectrum_dict['noise'])
    
    print("\n" + "="*70)
    print("DIAGNOSTIC RESULTS")
    print("="*70)
    
    # 1. Check initial spectrum
    print("\n1. Initial Spectrum Quality:")
    print("-"*70)
    initial_spec = np.array(forward_model(params))
    chi2 = np.sum(((obs_y - initial_spec) / obs_err)**2)
    chi2_dof = chi2 / len(obs_y)
    print(f"   χ²: {chi2:.2f}")
    print(f"   χ²/dof: {chi2_dof:.2f}")
    print(f"   Mean residual: {np.mean((obs_y - initial_spec) / obs_err):.3f}σ")
    print(f"   RMS residual: {np.std((obs_y - initial_spec) / obs_err):.3f}σ")
    
    # 2. Check bounds
    check_bounds(params, ADC_PRIORS)
    
    # 3. Compute gradients
    print("\n3. Gradient Analysis:")
    print("-"*70)
    loss_val, grads = compute_loss_and_grad(forward_model, params, obs_y, obs_err)
    print(f"   Initial loss: {loss_val:.6f}")
    print(f"\n   Gradient magnitudes:")
    fit_params = ['planet_radius', 'T', 'H2O', 'CO2', 'CO', 'CH4', 'NH3']
    for param in fit_params:
        if param in grads:
            grad_val = float(grads[param])
            param_val = float(params[param])
            rel_grad = grad_val * param_val  # grad * param gives sense of relative change
            print(f"      {param:15s}: {grad_val:12.6e} (relative: {rel_grad:12.6e})")
    
    # 4. Test small parameter perturbations
    print("\n4. Loss Landscape (small perturbations):")
    print("-"*70)
    lr = 1e-3
    for param in ['planet_radius', 'T', 'H2O']:
        if param in grads:
            # Perturb in gradient direction
            test_params = params.copy()
            grad_val = grads[param]
            step = -lr * grad_val  # Negative because we want to minimize
            test_params[param] = params[param] + step
            
            # Check new loss
            new_loss = compute_loss_and_grad(forward_model, test_params, obs_y, obs_err)[0]
            
            print(f"   {param:15s}:")
            print(f"      Current: {float(params[param]):.6e}, Loss: {loss_val:.6f}")
            print(f"      Step: {float(step):.6e} ({float(step/params[param])*100:+.3f}%)")
            print(f"      After step: {float(test_params[param]):.6e}, Loss: {new_loss:.6f}")
            print(f"      Loss change: {new_loss - loss_val:+.6f} {'✓ IMPROVING' if new_loss < loss_val else '✗ DIVERGING'}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    
    if chi2_dof < 2.0:
        print("✓ Initial model is EXCELLENT fit to data (χ²/dof < 2)")
        print("  → Ground truth from Tracedata is working correctly")
        print("  → Problem is with optimization, not initialization")
    
    # Check if gradients point away from optimum
    any_diverging = False
    for param in ['planet_radius', 'T', 'H2O']:
        if param in grads:
            test_params = params.copy()
            step = -lr * grads[param]
            test_params[param] = params[param] + step
            new_loss = compute_loss_and_grad(forward_model, test_params, obs_y, obs_err)[0]
            if new_loss > loss_val:
                any_diverging = True
                break
    
    if any_diverging:
        print("\n✗ Gradients are pointing AWAY from optimum")
        print("  Possible causes:")
        print("  1. Learning rate too high (try lr=1e-4 or 1e-5)")
        print("  2. Parameter transforms causing issues")
        print("  3. Numerical precision problems")
        print("\n  Suggested fixes:")
        print("  - Reduce learning rate: --lr 1e-4")
        print("  - Add strong L2 regularization to keep near initial: --l2-reg 1.0")
        print("  - Use log-space prior for gases: --log-prior 1.0")
    else:
        print("\n✓ Gradients point toward improvement")
        print("  But optimization still diverges, so:")
        print("  1. Learning rate might be too high for cumulative effect")
        print("  2. Need regularization to prevent over-stepping")
        print("\n  Suggested fixes:")
        print("  - Strong regularization: --l2-reg 0.1 --log-prior 2.0")
        print("  - Lower learning rate: --lr 1e-4")

if __name__ == "__main__":
    main()
