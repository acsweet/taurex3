# Converting TauREx to Float32 in JAX

## The Problem

TauREx returns everything as `float64` (NumPy arrays and Python floats). When you disable x64 in JAX:

```python
jax.config.update("jax_enable_x64", False)
```

You might expect everything to become float32, but **JAX respects the input dtype** when converting, so float64 arrays stay float64!

## The Solution

You need to **explicitly cast** TauREx data to float32. Here's how:

---

## Option 1: Modify `extract_fitting_params` (RECOMMENDED)

Add explicit float32 casting in the parameter extraction:

```python
def extract_fitting_params(taurex_model, use_float32=False) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Any]]:
    """
    Extract fitting parameters from taurex model.
    
    Args:
        taurex_model: TauREx model
        use_float32: If True, cast parameters to float32 (default: False)
    """
    fitting_params = {}
    param_info = {}
    
    for key, val in taurex_model._fitting_parameters.items():
        get_val = val[2]()
        val_range = val[-1]
        fit_scale = val[4]
        fitting_params[key] = float(get_val)
        param_info[key] = {
            'range': val_range,
            'scale': fit_scale,
            'latex': val[1]
        }
    
    # Convert to JAX arrays with explicit dtype
    if use_float32:
        params = {k: jnp.array(v, dtype=jnp.float32) for k, v in fitting_params.items()}
    else:
        params = {k: jnp.array(v) for k, v in fitting_params.items()}
    
    return params, param_info
```

Usage:
```python
# For float32
params, param_info = extract_fitting_params(tm, use_float32=True)

# For float64 (default)
params, param_info = extract_fitting_params(tm)
```

---

## Option 2: Modify `prepare_model_data` and `full_diff_prepare_model_data`

Add dtype parameter to these functions:

```python
def prepare_model_data(taurex_model: TransmissionModel, dtype=None):
    """
    Prepare static model data for JAX computation.
    
    Args:
        taurex_model: TauREx model
        dtype: Target dtype (e.g., jnp.float32). If None, uses JAX default.
    """
    if not taurex_model.built:
        taurex_model.build()
    
    taurex_model.initialize_profiles()
    
    # Helper function for conversion
    def to_jax(arr):
        return jnp.array(arr, dtype=dtype) if dtype is not None else jnp.array(arr)
    
    model_data = {
        'deltaz': to_jax(taurex_model.deltaz),
        'total_layers': int(taurex_model.nLayers),
        'altitude_profile': to_jax(taurex_model.altitudeProfile),
        'pressure_profile': to_jax(taurex_model.pressureProfile),
        'planet_radius_base': float(taurex_model._planet.fullRadius),
        'star_radius': float(taurex_model._star.radius),
    }
    
    return model_data


def full_diff_prepare_model_data(taurex_model, dtype=None):
    """
    Extract static and initial data from TauREx model.
    
    Args:
        taurex_model: TauREx model
        dtype: Target dtype for arrays (e.g., jnp.float32)
    """
    if not taurex_model.built:
        taurex_model.build()
    
    taurex_model.initialize_profiles()
    
    # Helper function
    def to_jax(arr):
        return jnp.array(arr, dtype=dtype) if dtype is not None else jnp.array(arr)
    
    static_data = {
        'nlayers': int(taurex_model.nLayers),
        'pressure_profile': to_jax(taurex_model.pressureProfile),
        'pressure_levels': to_jax(taurex_model.pressure.pressure_profile_levels),
        'planet_mass': float(taurex_model._planet.fullMass),
        'planet_radius_base': float(taurex_model._planet.fullRadius),
        'star_radius': float(taurex_model._star.radius),
        'active_gases': tuple(taurex_model.chemistry.activeGases),
        'fill_gases': tuple(taurex_model.chemistry._fill_gases),
        'He_H2_ratio': float(taurex_model.chemistry._fill_ratio[0]),
    }
    
    initial_state = {
        'temperature_profile': to_jax(taurex_model.temperatureProfile),
        'altitude_profile': to_jax(taurex_model.altitudeProfile),
        'deltaz': to_jax(taurex_model.deltaz),
        'mu_profile': to_jax(taurex_model.chemistry.muProfile),
    }
    
    return static_data, initial_state
```

---

## Option 3: Global Conversion Wrapper

Create a utility function for consistent conversion:

```python
def convert_to_float32(data):
    """
    Recursively convert data structures to float32.
    
    Handles:
    - JAX/NumPy arrays → jnp.float32
    - Python floats → jnp.float32
    - Dicts → recursive conversion
    - Tuples/lists → recursive conversion
    - Ints/strings → unchanged
    """
    if isinstance(data, (jnp.ndarray, np.ndarray)):
        return jnp.array(data, dtype=jnp.float32)
    elif isinstance(data, float):
        return jnp.array(data, dtype=jnp.float32)
    elif isinstance(data, dict):
        return {k: convert_to_float32(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        converted = [convert_to_float32(v) for v in data]
        return tuple(converted) if isinstance(data, tuple) else converted
    else:
        return data  # int, str, etc.


# Usage:
params, param_info = extract_fitting_params(tm)
params = convert_to_float32(params)

model_data = prepare_model_data(tm)
model_data = convert_to_float32(model_data)
```

---

## Option 4: Cast Inside JIT Function (SIMPLEST)

Just cast at the entry point of your JIT-compiled function:

```python
@partial(jax.jit, static_argnames=['static_data'])
def full_diff_path_integral(wngrid, params, opacity_data, dynamic_data, static_data):
    """Forward model - cast inputs to float32 at entry."""
    
    # Cast parameters to float32
    params_f32 = {k: v.astype(jnp.float32) for k, v in params.items()}
    
    # Cast dynamic data to float32
    dz, altitude, pressure, cia_sigma, other_sigma = dynamic_data
    dynamic_f32 = (
        dz.astype(jnp.float32),
        altitude.astype(jnp.float32),
        pressure.astype(jnp.float32),
        cia_sigma.astype(jnp.float32) if cia_sigma is not None else None,
        other_sigma.astype(jnp.float32) if other_sigma is not None else None,
    )
    
    # ... rest of computation uses float32 ...
    
    return result
```

**Pros:** Simple, localized change  
**Cons:** Redundant casting on every call (though JIT might optimize it away)

---

## Recommended Approach

**For quick testing:**
```python
# At top of experiment.py
jax.config.update("jax_enable_x64", False)

# When creating forward model
params, _ = extract_fitting_params(tm)
params = {k: v.astype(jnp.float32) for k, v in params.items()}  # ← Add this line
```

**For production:**
Modify `extract_fitting_params` to accept `use_float32=True` parameter (Option 1).

---

## Testing Float32 Conversion

```python
import jax
import jax.numpy as jnp

# Disable x64
jax.config.update("jax_enable_x64", False)

from experiment import extract_fitting_params

# ... setup TauREx model tm ...

# Extract and convert
params, _ = extract_fitting_params(tm)
params_f32 = {k: v.astype(jnp.float32) for k, v in params.items()}

# Verify
for key, val in params_f32.items():
    print(f"{key}: {val.dtype}")  # Should all be float32

# Check size difference
import sys
print(f"\nMemory per param:")
print(f"  float64: {sys.getsizeof(params['T'])} bytes")
print(f"  float32: {sys.getsizeof(params_f32['T'])} bytes")
```

---

## Complete Example: Float32 Forward Model

```python
import jax
import jax.numpy as jnp

# Configure JAX for float32
jax.config.update("jax_enable_x64", False)

from experiment import (
    full_diff_create_binned_forward_model,
    extract_fitting_params
)
from taurex.model import TransmissionModel
# ... other imports ...

# Build TauREx model
tm = TransmissionModel(...)
tm.build()

# Load observation
obs = ObservedSpectrum(obs_path)

# Create forward model (will use float32 internally if properly converted)
forward_binned = full_diff_create_binned_forward_model(tm, obs)

# Extract params and convert to float32
params, param_info = extract_fitting_params(tm)
params = {k: v.astype(jnp.float32) for k, v in params.items()}

# Run forward model
result = forward_binned(params)

# Verify dtype
print(f"Result dtype: {result.dtype}")  # Should be float32
print(f"Result shape: {result.shape}")
```

---

## Important Notes

1. **Consistency is key**: All arrays in the computation should be the same dtype (either all float32 or all float64)

2. **Watch for mixing**: If you mix float32 and float64, JAX will promote to float64, defeating the purpose

3. **Check opacity data**: Opacity grids loaded in `load_opacity_data` should also be float32:
   ```python
   def load_opacity_data(active_gases, wngrid_target, dtype=jnp.float32):
       # ...
       'sigma_grid': jnp.array(sigma_grid, dtype=dtype),
       # ...
   ```

4. **JIT compilation**: First call will be slow (compilation), subsequent calls fast

5. **Numerical stability**: Test carefully! If you see NaN/Inf with float32, you need float64 or mixed precision

---

## Summary

**To use float32:**

1. Set `jax.config.update("jax_enable_x64", False)` at the top
2. **Explicitly cast** all TauREx data to float32:
   - Parameters: `params = {k: v.astype(jnp.float32) for k, v in params.items()}`
   - Model data: Add `dtype=jnp.float32` to conversion functions
   - Opacities: Cast in `load_opacity_data`
3. Verify with `print(array.dtype)` checks
4. Test for numerical stability

**Default behavior (with x64 disabled):**
- `jnp.array(numpy_float64_array)` → **float32** ✅
- `jnp.array(python_float)` → **float32** ✅
- BUT: Explicit is better than implicit!

Use `.astype(jnp.float32)` to be sure!
