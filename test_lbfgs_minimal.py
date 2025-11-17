"""
Minimal test to debug L-BFGS-B memory issues.
"""

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from jaxopt import LBFGSB

print("Testing L-BFGS-B with simple problem...")

# Simple quadratic problem: minimize ||x - target||^2
target = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

def loss_fn(x):
    return jnp.sum((x - target)**2)

# Test 1: Unconstrained
print("\n1. Testing unconstrained optimization...")
x0 = jnp.zeros(5)
solver = LBFGSB(fun=loss_fn, maxiter=100)
bounds = (jnp.full_like(x0, -10.0), jnp.full_like(x0, 10.0))

try:
    result = solver.run(x0, bounds=bounds)
    print(f"   Success! Final loss: {loss_fn(result.params):.6e}")
    print(f"   Converged: {result.state.converged}")
    print(f"   Iterations: {result.state.iter_num}")
except Exception as e:
    print(f"   FAILED: {e}")

# Test 2: Try without JIT
print("\n2. Testing with explicit iteration loop...")
try:
    solver = LBFGSB(fun=loss_fn, maxiter=100, jit=False)
    result = solver.run(x0, bounds=bounds)
    print(f"   Success! Final loss: {loss_fn(result.params):.6e}")
except Exception as e:
    print(f"   FAILED: {e}")

# Test 3: Smaller history size
print("\n3. Testing with smaller history size...")
try:
    solver = LBFGSB(fun=loss_fn, maxiter=100, history_size=5)
    result = solver.run(x0, bounds=bounds)
    print(f"   Success! Final loss: {loss_fn(result.params):.6e}")
except Exception as e:
    print(f"   FAILED: {e}")

print("\n4. Now testing with atmospheric retrieval problem...")

# Load minimal atmospheric problem
from taurex.cache import OpacityCache, CIACache
from adc_jax_utils import load_planet_spectrum, load_auxiliary_data, load_ground_truth
from adc_jax_utils import create_taurex_model_from_adc_planet, create_adc_jax_forward_model
from experiment import extract_fitting_params, make_transforms, make_packer

OpacityCache().set_opacity_path('test_files/xsec/xsec_sampled_R15000_0.3-50')
CIACache().set_cia_path('test_files/cia/HITRAN/data')

# Load data
print("   Loading planet 1000...")
planet_id = 1000
data_dir = 'test_files/adc_2023/TrainingData'
spectral_path = f'{data_dir}/SpectralData.hdf5'
aux_path = f'{data_dir}/AuxillaryTable.csv'
tracedata_path = f'{data_dir}/Ground Truth Package/Tracedata.hdf5'

spectrum_dict = load_planet_spectrum(spectral_path, planet_id)
aux_data = load_auxiliary_data(aux_path, planet_id)
ground_truth = load_ground_truth(tracedata_path, planet_id, sample_method='weighted_mean')

print("   Creating model...")
tm = create_taurex_model_from_adc_planet(aux_data, ground_truth, nlayers=30)
forward_model, obs_spectrum = create_adc_jax_forward_model(tm, spectrum_dict)

print("   Extracting parameters...")
params, param_info = extract_fitting_params(tm, dtype=jnp.float64)

# Override bounds
adc_priors = {
    'planet_radius': [0.1, 3.0],
    'T': [0.0, 7000.0],
    'H2O': [1e-12, 0.1],
    'CO2': [1e-12, 0.1],
    'CO': [1e-12, 0.1],
    'CH4': [1e-12, 0.1],
    'NH3': [1e-12, 0.1]
}
for param_name, bounds_val in adc_priors.items():
    if param_name in param_info:
        param_info[param_name]['range'] = bounds_val

# Setup observation data
obs_y = jnp.array(spectrum_dict['spectrum'])
obs_err = jnp.array(spectrum_dict['noise'])

fit_params = ['planet_radius', 'T', 'H2O', 'CO2', 'CO', 'CH4', 'NH3']
fixed = {k: v for k, v in params.items() if k not in fit_params}
train0 = {k: params[k] for k in fit_params}

to_c, to_u = make_transforms(param_info, fit_params)
u0 = to_u(train0)
pack, unpack, keys = make_packer(u0)
uvec0 = pack(u0)

print(f"   Parameter vector size: {len(uvec0)}")

# Define loss
def atm_loss_fn(uvec):
    udict = unpack(uvec)
    c_fit = to_c(udict)
    params_full = {**fixed, **c_fit}
    pred = jnp.asarray(forward_model(params_full))
    pred = jnp.nan_to_num(pred, nan=0.0, posinf=1e300, neginf=-1e300)
    residuals = (obs_y - pred) / obs_err
    chi_sq = jnp.sum(residuals**2)
    return chi_sq / len(obs_y)

# Test initial evaluation
print("   Testing loss evaluation...")
try:
    initial_loss = atm_loss_fn(uvec0)
    print(f"   Initial loss: {initial_loss:.6e}")
except Exception as e:
    print(f"   FAILED at loss evaluation: {e}")
    import sys
    sys.exit(1)

# Test gradient
print("   Testing gradient computation...")
try:
    grad_fn = jax.grad(atm_loss_fn)
    initial_grad = grad_fn(uvec0)
    grad_norm = float(jnp.linalg.norm(initial_grad))
    print(f"   Gradient norm: {grad_norm:.3e}")
except Exception as e:
    print(f"   FAILED at gradient: {e}")
    import sys
    sys.exit(1)

# Test L-BFGS with very few iterations
print("\n5. Testing L-BFGS with 10 iterations only...")
try:
    bounds = (jnp.full_like(uvec0, -1e6), jnp.full_like(uvec0, 1e6))
    solver = LBFGSB(
        fun=atm_loss_fn,
        maxiter=10,  # Very small
        history_size=5,  # Small history
        tol=1e-6
    )
    print("   Running optimization...")
    result = solver.run(uvec0, bounds=bounds)
    print(f"   Success! Final loss: {result.state.value:.6e}")
    print(f"   Iterations: {result.state.iter_num}")
    print(f"   Converged: {result.state.converged}")
except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test with jit=False
print("\n6. Testing L-BFGS with jit=False...")
try:
    solver = LBFGSB(
        fun=atm_loss_fn,
        maxiter=10,
        history_size=5,
        jit=False  # Disable JIT
    )
    print("   Running optimization...")
    result = solver.run(uvec0, bounds=bounds)
    print(f"   Success! Final loss: {result.state.value:.6e}")
except Exception as e:
    print(f"   FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\nDiagnostic complete!")
