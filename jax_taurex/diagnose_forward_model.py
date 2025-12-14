"""
Deep dive into forward model behavior to understand why optimization degrades fit.

This checks:
1. Does the forward model produce consistent results?
2. Are gradients pointing in the right direction?
3. Is there a discrepancy between TauREx native and JAX forward model?
4. What happens with tiny parameter changes?
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

def compare_taurex_vs_jax(tm, forward_model, params):
    """Compare TauREx native model vs JAX forward model."""
    # Get TauREx native spectrum
    wngrid = tm.nativeWavenumberGrid
    native_spec_full = tm.model(wngrid=wngrid)
    
    # TauREx returns spectrum for each contribution - get the total
    if hasattr(native_spec_full, '__len__') and len(native_spec_full) > 1:
        # Sum contributions if multiple
        native_spec = native_spec_full[-1] if len(native_spec_full) == 4 else native_spec_full
    else:
        native_spec = native_spec_full
    
    # Get JAX spectrum (already binned)
    jax_spec = np.array(forward_model(params))
    
    print("\nTauREx Native vs JAX Forward Model:")
    print("="*70)
    print(f"TauREx native points: {len(native_spec) if hasattr(native_spec, '__len__') else 'scalar'}")
    print(f"JAX binned points: {len(jax_spec)}")
    
    if hasattr(native_spec, '__len__'):
        print(f"TauREx mean: {np.mean(native_spec):.6e}")
        print(f"JAX mean: {np.mean(jax_spec):.6e}")
        if len(native_spec) > 0:
            print(f"Ratio (JAX/TauREx): {np.mean(jax_spec)/np.mean(native_spec):.6f}")
    
    return native_spec, jax_spec

def test_gradient_directions(forward_model, params, obs_y, obs_err, fit_params):
    """Test if gradients point toward better fits."""
    
    def mse_loss(p):
        pred = jnp.asarray(forward_model(p))
        residuals = (obs_y - pred) / obs_err
        return jnp.mean(residuals**2)
    
    initial_loss = float(mse_loss(params))
    grads = jax.grad(mse_loss)(params)
    
    print("\nGradient Direction Test:")
    print("="*70)
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"\nTesting small steps in OPPOSITE direction of gradient:")
    print("(Negative gradient direction should DECREASE loss)")
    print("-"*70)
    
    learning_rates = [1e-6, 1e-5, 1e-4, 1e-3]
    
    for param in fit_params[:3]:  # Test first 3 params
        print(f"\n{param}:")
        print(f"  Value: {float(params[param]):.6e}")
        print(f"  Gradient: {float(grads[param]):.6e}")
        
        for lr in learning_rates:
            test_params = params.copy()
            step = -lr * grads[param]
            test_params[param] = params[param] + step
            
            new_loss = float(mse_loss(test_params))
            delta_loss = new_loss - initial_loss
            
            status = "✓" if new_loss < initial_loss else "✗"
            print(f"  lr={lr:.0e}: step={float(step):.3e}, new_loss={new_loss:.6f}, Δ={delta_loss:+.6f} {status}")

def test_parameter_sensitivity(forward_model, params, obs_y, obs_err, param_name, perturbations):
    """Test how loss changes with parameter perturbations."""
    
    def mse_loss(p):
        pred = jnp.asarray(forward_model(p))
        residuals = (obs_y - pred) / obs_err
        return jnp.mean(residuals**2)
    
    initial_val = float(params[param_name])
    initial_loss = float(mse_loss(params))
    
    print(f"\nParameter Sensitivity: {param_name}")
    print("="*70)
    print(f"Initial value: {initial_val:.6e}")
    print(f"Initial loss: {initial_loss:.6f}")
    print("-"*70)
    
    losses = []
    values = []
    
    for pct_change in perturbations:
        test_params = params.copy()
        new_val = initial_val * (1 + pct_change/100)
        test_params[param_name] = jnp.array(new_val, dtype=params[param_name].dtype)
        
        new_loss = float(mse_loss(test_params))
        losses.append(new_loss)
        values.append(new_val)
        
        delta = new_loss - initial_loss
        direction = "better" if delta < 0 else "worse"
        print(f"  {pct_change:+6.1f}%: value={new_val:.6e}, loss={new_loss:.6f}, Δ={delta:+.6f} ({direction})")
    
    return values, losses

def check_spectrum_consistency(forward_model, params, n_runs=5):
    """Check if forward model gives consistent results."""
    print("\nForward Model Consistency Check:")
    print("="*70)
    print(f"Running forward model {n_runs} times with same parameters...")
    
    spectra = []
    for i in range(n_runs):
        spec = np.array(forward_model(params))
        spectra.append(spec)
    
    spectra = np.array(spectra)
    
    # Check if all runs give same result
    max_diff = np.max(np.abs(spectra - spectra[0]))
    mean_val = np.mean(spectra)
    
    print(f"Mean spectrum value: {mean_val:.6e}")
    print(f"Max difference between runs: {max_diff:.6e}")
    print(f"Relative difference: {max_diff/mean_val*100:.6f}%")
    
    if max_diff < 1e-10:
        print("✓ Forward model is deterministic")
    else:
        print("✗ Forward model has inconsistencies!")
    
    return spectra

def plot_loss_landscape(forward_model, params, obs_y, obs_err, param1, param2):
    """Plot 2D loss landscape around current parameters."""
    
    def mse_loss(p):
        pred = jnp.asarray(forward_model(p))
        residuals = (obs_y - pred) / obs_err
        return jnp.mean(residuals**2)
    
    # Create grid around current values
    val1 = float(params[param1])
    val2 = float(params[param2])
    
    range1 = np.linspace(val1 * 0.95, val1 * 1.05, 20)
    range2 = np.linspace(val2 * 0.95, val2 * 1.05, 20)
    
    loss_grid = np.zeros((len(range1), len(range2)))
    
    print(f"\nComputing loss landscape for {param1} vs {param2}...")
    
    for i, v1 in enumerate(range1):
        for j, v2 in enumerate(range2):
            test_params = params.copy()
            test_params[param1] = jnp.array(v1, dtype=params[param1].dtype)
            test_params[param2] = jnp.array(v2, dtype=params[param2].dtype)
            
            loss_grid[i, j] = float(mse_loss(test_params))
    
    # Find minimum
    min_idx = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)
    min_val1 = range1[min_idx[0]]
    min_val2 = range2[min_idx[1]]
    min_loss = loss_grid[min_idx]
    
    print(f"Minimum loss in grid: {min_loss:.6f}")
    print(f"  {param1}: {min_val1:.6e} (current: {val1:.6e})")
    print(f"  {param2}: {min_val2:.6e} (current: {val2:.6e})")
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(range2, range1, loss_grid, levels=20, cmap='viridis')
    plt.colorbar(label='Loss')
    plt.plot(val2, val1, 'r*', markersize=15, label='Current')
    plt.plot(min_val2, min_val1, 'w*', markersize=15, label='Local minimum')
    plt.xlabel(f'{param2}')
    plt.ylabel(f'{param1}')
    plt.title(f'Loss Landscape: {param1} vs {param2}')
    plt.legend()
    plt.savefig(f'loss_landscape_{param1}_{param2}.png', dpi=150, bbox_inches='tight')
    print(f"Saved: loss_landscape_{param1}_{param2}.png")
    plt.close()
    
    return loss_grid, range1, range2

def main():
    planet_id = 1000
    data_dir = 'test_files/adc_2023/TrainingData'
    
    # Setup
    print("Setting up...")
    OpacityCache().clear_cache()
    OpacityCache().set_opacity_path('test_files/xsec/xsec_sampled_R15000_0.3-50')
    CIACache().set_cia_path('test_files/cia/HITRAN/data')
    
    # Load data
    print(f"Loading planet {planet_id}...")
    spectral_path = f'{data_dir}/SpectralData.hdf5'
    aux_path = f'{data_dir}/AuxillaryTable.csv'
    tracedata_path = f'{data_dir}/Ground Truth Package/Tracedata.hdf5'
    
    spectrum_dict = load_planet_spectrum(spectral_path, planet_id)
    aux_data = load_auxiliary_data(aux_path, planet_id)
    ground_truth = load_ground_truth(tracedata_path, planet_id, sample_method='weighted_mean')
    
    # Create model
    print("Creating model...")
    tm = create_taurex_model_from_adc_planet(aux_data, ground_truth, nlayers=30)
    forward_model, obs_spectrum = create_adc_jax_forward_model(tm, spectrum_dict)
    
    # Extract parameters
    params, param_info = extract_fitting_params(tm, dtype=jnp.float64)
    
    obs_y = jnp.array(spectrum_dict['spectrum'])
    obs_err = jnp.array(spectrum_dict['noise'])
    
    fit_params = ['planet_radius', 'T', 'H2O', 'CO2', 'CO', 'CH4', 'NH3']
    
    print("\n" + "="*70)
    print("FORWARD MODEL DIAGNOSTICS")
    print("="*70)
    
    # Test 1: Consistency
    spectra = check_spectrum_consistency(forward_model, params, n_runs=5)
    
    # Test 2: TauREx vs JAX
    native_spec, jax_spec = compare_taurex_vs_jax(tm, forward_model, params)
    
    # Test 3: Gradient directions
    test_gradient_directions(forward_model, params, obs_y, obs_err, fit_params)
    
    # Test 4: Parameter sensitivity
    print("\n" + "="*70)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*70)
    
    # Test planet_radius
    values, losses = test_parameter_sensitivity(
        forward_model, params, obs_y, obs_err, 
        'planet_radius', 
        perturbations=[-2, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2]
    )
    
    # Test H2O
    values, losses = test_parameter_sensitivity(
        forward_model, params, obs_y, obs_err,
        'H2O',
        perturbations=[-50, -20, -10, -5, 0, 5, 10, 20, 50]
    )
    
    # Test 5: Loss landscape
    print("\n" + "="*70)
    print("LOSS LANDSCAPE ANALYSIS")
    print("="*70)
    
    plot_loss_landscape(forward_model, params, obs_y, obs_err, 'planet_radius', 'T')
    plot_loss_landscape(forward_model, params, obs_y, obs_err, 'H2O', 'CO2')
    
    print("\n" + "="*70)
    print("DIAGNOSTICS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
