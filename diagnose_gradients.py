"""
Investigate why gradients are pointing in the wrong direction.

This checks:
1. Parameter transforms (constrained <-> unconstrained space)
2. Analytical gradients vs numerical gradients
3. Chain rule through transforms
4. Individual parameter gradient components
"""

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from adc_jax_utils import load_ground_truth, load_planet_spectrum, load_auxiliary_data
from adc_jax_utils import create_taurex_model_from_adc_planet, create_adc_jax_forward_model
from taurex.cache import OpacityCache, CIACache
from experiment import extract_fitting_params, make_transforms, make_packer

def numerical_gradient(loss_fn, params, param_name, epsilon=1e-6):
    """Compute numerical gradient using finite differences."""
    params_plus = params.copy()
    params_minus = params.copy()
    
    params_plus[param_name] = params[param_name] + epsilon
    params_minus[param_name] = params[param_name] - epsilon
    
    loss_plus = loss_fn(params_plus)
    loss_minus = loss_fn(params_minus)
    
    return (loss_plus - loss_minus) / (2 * epsilon)

def test_transforms(param_info, fit_params, params):
    """Test parameter transforms for correctness."""
    print("\nParameter Transform Test:")
    print("="*70)
    
    to_c, to_u = make_transforms(param_info, fit_params)
    
    # Test constrained params
    c_dict = {k: params[k] for k in fit_params}
    
    # Transform to unconstrained
    u_dict = to_u(c_dict)
    
    # Transform back to constrained
    c_dict_recovered = to_c(u_dict)
    
    print("\nRound-trip test (constrained -> unconstrained -> constrained):")
    print("-"*70)
    for param in fit_params[:5]:  # Test first 5
        original = float(c_dict[param])
        unconstrained = float(u_dict[param])
        recovered = float(c_dict_recovered[param])
        error = abs(recovered - original)
        
        lo, hi = param_info[param]['range']
        scale = param_info[param].get('scale', 'linear')
        
        print(f"{param:15s}:")
        print(f"  Range: [{lo:.2e}, {hi:.2e}], scale: {scale}")
        print(f"  Original:        {original:.6e}")
        print(f"  Unconstrained:   {unconstrained:.6e}")
        print(f"  Recovered:       {recovered:.6e}")
        print(f"  Error:           {error:.6e} {'✓' if error < 1e-10 else '✗ PROBLEM!'}")

def test_transform_gradients(param_info, fit_params, params):
    """Test if transforms preserve gradient signs correctly."""
    print("\nTransform Gradient Test:")
    print("="*70)
    print("Checking if gradients through transforms have correct signs...")
    print("-"*70)
    
    to_c, to_u = make_transforms(param_info, fit_params)
    pack, unpack, keys = make_packer(to_u({k: params[k] for k in fit_params}))
    
    # Test gradient flow through transforms
    for param in fit_params[:3]:
        print(f"\n{param}:")
        
        c_val = params[param]
        u_dict = to_u({param: c_val})
        u_val = u_dict[param]
        
        print(f"  Constrained value: {float(c_val):.6e}")
        print(f"  Unconstrained value: {float(u_val):.6e}")
        
        # Test forward transform gradient
        def forward(c):
            return to_u({param: c})[param]
        
        grad_forward = jax.grad(lambda c: forward(c))(c_val)
        print(f"  d(unconstrained)/d(constrained): {float(grad_forward):.6e}")
        
        # Test inverse transform gradient  
        def inverse(u):
            return to_c({param: u})[param]
        
        grad_inverse = jax.grad(lambda u: inverse(u))(u_val)
        print(f"  d(constrained)/d(unconstrained): {float(grad_inverse):.6e}")
        
        # Chain rule test: product should be ~1
        product = float(grad_forward * grad_inverse)
        print(f"  Product (should be ~1): {product:.6f} {'✓' if abs(product - 1) < 0.01 else '✗'}")

def compare_analytical_vs_numerical(forward_model, params, param_info, obs_y, obs_err, fit_params):
    """Compare analytical gradients from JAX vs numerical finite differences."""
    print("\nAnalytical vs Numerical Gradient Comparison:")
    print("="*70)
    
    # Define loss in CONSTRAINED space (directly on parameters)
    def loss_constrained(p):
        pred = jnp.asarray(forward_model(p))
        residuals = (obs_y - pred) / obs_err
        return jnp.mean(residuals**2)
    
    # Get analytical gradients
    analytical_grads = jax.grad(loss_constrained)(params)
    
    print("\nDirect gradients in constrained space:")
    print("-"*70)
    print(f"{'Parameter':<15} {'Analytical':<15} {'Numerical':<15} {'Ratio':<10} {'Sign Match'}")
    print("-"*70)
    
    for param in fit_params:
        analytical = float(analytical_grads[param])
        numerical = float(numerical_gradient(loss_constrained, params, param, epsilon=1e-7))
        
        ratio = analytical / numerical if abs(numerical) > 1e-30 else float('nan')
        sign_match = "✓" if np.sign(analytical) == np.sign(numerical) else "✗ OPPOSITE!"
        
        print(f"{param:<15} {analytical:>14.6e} {numerical:>14.6e} {ratio:>9.3f} {sign_match}")

def test_unconstrained_space_gradients(forward_model, params, param_info, obs_y, obs_err, fit_params):
    """Test gradients in unconstrained space (how optimizer sees them)."""
    print("\nGradients in Unconstrained Space (Optimizer's View):")
    print("="*70)
    
    to_c, to_u = make_transforms(param_info, fit_params)
    pack, unpack, keys = make_packer(to_u({k: params[k] for k in fit_params}))
    
    fixed = {k: v for k, v in params.items() if k not in fit_params}
    
    def loss_unconstrained(uvec):
        u_dict = unpack(uvec)
        c_dict = to_c(u_dict)
        full_params = {**fixed, **c_dict}
        pred = jnp.asarray(forward_model(full_params))
        residuals = (obs_y - pred) / obs_err
        return jnp.mean(residuals**2)
    
    train0 = {k: params[k] for k in fit_params}
    u0 = to_u(train0)
    uvec = pack(u0)
    
    # Get gradients in unconstrained space
    loss_val = float(loss_unconstrained(uvec))
    grads_u = jax.grad(loss_unconstrained)(uvec)
    
    print(f"\nLoss at current point: {loss_val:.6f}")
    print("\nGradient in unconstrained space:")
    print("-"*70)
    
    for i, param in enumerate(keys):
        grad_u = float(grads_u[i])
        u_val = float(uvec[i])
        c_val = float(params[param])
        
        print(f"{param:<15}:")
        print(f"  Constrained value:     {c_val:.6e}")
        print(f"  Unconstrained value:   {u_val:.6e}")
        print(f"  Gradient (uncon):      {grad_u:.6e}")
        
        # Test tiny step
        test_uvec = uvec.at[i].add(-1e-5 * grad_u)  # Small step opposite gradient
        new_loss = float(loss_unconstrained(test_uvec))
        delta_loss = new_loss - loss_val
        
        improving = "✓ IMPROVING" if delta_loss < 0 else "✗ DIVERGING"
        print(f"  Small step test:       Δloss = {delta_loss:+.6e} {improving}")

def trace_forward_model_components(forward_model, params, fit_params):
    """Trace how each parameter affects the forward model output."""
    print("\nForward Model Component Analysis:")
    print("="*70)
    
    # Get baseline spectrum
    baseline = np.array(forward_model(params))
    
    print("\nTesting individual parameter perturbations:")
    print("-"*70)
    
    for param in fit_params[:3]:
        val = float(params[param])
        
        # Small increase
        test_params = params.copy()
        test_params[param] = params[param] * 1.001  # 0.1% increase
        
        perturbed = np.array(forward_model(test_params))
        
        diff = perturbed - baseline
        mean_change = np.mean(diff)
        max_change = np.max(np.abs(diff))
        
        print(f"\n{param}: {val:.6e} → {float(test_params[param]):.6e} (+0.1%)")
        print(f"  Mean spectrum change: {mean_change:.6e}")
        print(f"  Max spectrum change:  {max_change:.6e}")
        print(f"  Direction: {'↑ increases' if mean_change > 0 else '↓ decreases'} spectrum")

def main():
    planet_id = 1000
    data_dir = 'test_files/adc_2023/TrainingData'
    
    # ADC 2023 competition priors (CRITICAL - must match actual parameter ranges!)
    adc_priors = {
        'planet_radius': [0.1, 3.0],
        'T': [0.0, 7000.0],
        'H2O': [1e-12, 0.1],
        'CO2': [1e-12, 0.1],
        'CO': [1e-12, 0.1],
        'CH4': [1e-12, 0.1],
        'NH3': [1e-12, 0.1]
    }
    
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
    
    # Extract parameters and override bounds with ADC priors
    params, param_info = extract_fitting_params(tm, dtype=jnp.float64)
    
    print("\nOverriding parameter bounds with ADC priors...")
    for param_name, bounds in adc_priors.items():
        if param_name in param_info:
            old_bounds = param_info[param_name]['range']
            param_info[param_name]['range'] = bounds
            print(f"  {param_name}: {old_bounds} -> {bounds}")
    
    obs_y = jnp.array(spectrum_dict['spectrum'])
    obs_err = jnp.array(spectrum_dict['noise'])
    
    fit_params = ['planet_radius', 'T', 'H2O', 'CO2', 'CO', 'CH4', 'NH3']
    
    print("\n" + "="*70)
    print("GRADIENT DIAGNOSTICS")
    print("="*70)
    
    # Test 1: Transforms
    test_transforms(param_info, fit_params, params)
    
    # Test 2: Transform gradients
    test_transform_gradients(param_info, fit_params, params)
    
    # Test 3: Analytical vs numerical
    compare_analytical_vs_numerical(forward_model, params, param_info, obs_y, obs_err, fit_params)
    
    # Test 4: Unconstrained space
    test_unconstrained_space_gradients(forward_model, params, param_info, obs_y, obs_err, fit_params)
    
    # Test 5: Forward model components
    trace_forward_model_components(forward_model, params, fit_params)
    
    print("\n" + "="*70)
    print("DIAGNOSTICS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
