"""
Test L-BFGS with pre-compiled loss function to avoid recompilation.
"""

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jaxopt import LBFGSB
import gc

print("Testing L-BFGS with pre-compiled functions...")

from taurex.cache import OpacityCache, CIACache
from .adc_jax_utils import load_planet_spectrum, load_auxiliary_data, load_ground_truth
from .adc_jax_utils import create_taurex_model_from_adc_planet, create_adc_jax_forward_model
from .jax_backend import extract_fitting_params, make_transforms, make_packer

OpacityCache().set_opacity_path('test_files/xsec/xsec_sampled_R15000_0.3-50')
CIACache().set_cia_path('test_files/cia/HITRAN/data')

# Load data
print("Loading planet 1000...")
planet_id = 1000
data_dir = 'test_files/adc_2023/TrainingData'
spectral_path = f'{data_dir}/SpectralData.hdf5'
aux_path = f'{data_dir}/AuxillaryTable.csv'
tracedata_path = f'{data_dir}/Ground Truth Package/Tracedata.hdf5'

spectrum_dict = load_planet_spectrum(spectral_path, planet_id)
aux_data = load_auxiliary_data(aux_path, planet_id)
ground_truth = load_ground_truth(tracedata_path, planet_id, sample_method='weighted_mean')

print("Creating model...")
tm = create_taurex_model_from_adc_planet(aux_data, ground_truth, nlayers=30)
forward_model, obs_spectrum = create_adc_jax_forward_model(tm, spectrum_dict)

print("Extracting parameters...")
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

obs_y = jnp.array(spectrum_dict['spectrum'])
obs_err = jnp.array(spectrum_dict['noise'])

fit_params = ['planet_radius', 'T', 'H2O', 'CO2', 'CO', 'CH4', 'NH3']
fixed = {k: v for k, v in params.items() if k not in fit_params}
train0 = {k: params[k] for k in fit_params}

to_c, to_u = make_transforms(param_info, fit_params)
u0 = to_u(train0)
pack, unpack, keys = make_packer(u0)
uvec0 = pack(u0)

print(f"Parameter vector size: {len(uvec0)}")

# Pre-compile loss function
print("Pre-compiling loss and gradient...")

def atm_loss_fn(uvec):
    udict = unpack(uvec)
    c_fit = to_c(udict)
    params_full = {**fixed, **c_fit}
    pred = jnp.asarray(forward_model(params_full))
    pred = jnp.nan_to_num(pred, nan=0.0, posinf=1e300, neginf=-1e300)
    residuals = (obs_y - pred) / obs_err
    chi_sq = jnp.sum(residuals**2)
    return chi_sq / len(obs_y)

# JIT compile
loss_jit = jax.jit(atm_loss_fn)
loss_and_grad_jit = jax.jit(jax.value_and_grad(atm_loss_fn))

# Warm up compilation
print("Warming up JIT compilation...")
_ = loss_jit(uvec0)
_ = loss_and_grad_jit(uvec0)
print("JIT compilation complete!")

# Force garbage collection
gc.collect()

# Try L-BFGS with different settings
print("\nAttempt 1: Default L-BFGS-B...")
try:
    bounds = (jnp.full_like(uvec0, -1e6), jnp.full_like(uvec0, 1e6))
    solver = LBFGSB(
        fun=atm_loss_fn,  # Non-JIT version (jaxopt will JIT internally)
        maxiter=50,
        history_size=10,
        tol=1e-8
    )
    result = solver.run(uvec0, bounds=bounds)
    print(f"SUCCESS! Final loss: {result.state.value:.6e}")
    print(f"Iterations: {result.state.iter_num}")
except KeyboardInterrupt:
    print("Interrupted by user")
    raise
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\nAttempt 2: Disable implicit differentiation...")
try:
    solver = LBFGSB(
        fun=atm_loss_fn,
        maxiter=50,
        history_size=10,
        implicit_diff=False  # Don't use implicit differentiation
    )
    result = solver.run(uvec0, bounds=bounds)
    print(f"SUCCESS! Final loss: {result.state.value:.6e}")
except Exception as e:
    print(f"FAILED: {e}")

print("\nAttempt 3: Very small maxiter and tiny history...")
try:
    solver = LBFGSB(
        fun=atm_loss_fn,
        maxiter=5,
        history_size=3
    )
    result = solver.run(uvec0, bounds=bounds)
    print(f"SUCCESS! Final loss: {result.state.value:.6e}")
except Exception as e:
    print(f"FAILED: {e}")

print("\nAttempt 4: Use LBFGS (without bounds) instead...")
try:
    from jaxopt import LBFGS
    solver = LBFGS(
        fun=atm_loss_fn,
        maxiter=50,
        history_size=10
    )
    result = solver.run(uvec0)  # No bounds
    print(f"SUCCESS! Final loss: {result.state.value:.6e}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\nDiagnostic complete!")
