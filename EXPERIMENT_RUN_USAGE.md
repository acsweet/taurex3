# Using experiment_run.py with Different Precisions

## Quick Start

### Option 1: Float64 (Default, Recommended)
```bash
# Edit experiment_run.py
# Set: USE_FLOAT64 = True

python experiment_run.py
```

**When to use:**
- First time running
- Want guaranteed numerical stability
- Memory is not a constraint (you have 12GB GPU)
- Accuracy is more important than speed

### Option 2: Float32 (Faster)
```bash
# Edit experiment_run.py
# Set: USE_FLOAT64 = False

python experiment_run.py
```

**When to use:**
- After verifying no NaN/Inf with float64
- Need maximum speed (~2x faster)
- Working with large grids (memory constrained)
- GPU memory is limited

## How It Works

The script now automatically:

1. **Configures JAX precision** at startup
   ```python
   USE_FLOAT64 = True  # or False
   jax.config.update("jax_enable_x64", USE_FLOAT64)
   DTYPE = jnp.float64 if USE_FLOAT64 else jnp.float32
   ```

2. **Applies dtype consistently**
   ```python
   # Parameters
   params, _ = extract_fitting_params(tm, dtype=DTYPE)
   
   # Observations
   obs_y = jnp.asarray(obs.spectrum, dtype=DTYPE)
   obs_err = jnp.asarray(obs.errorBar, dtype=DTYPE)
   ```

3. **Shows configuration in output**
   ```
   JAX Configuration:
     Precision: float64
     Device: cuda:0
     Default dtype: float64
   ```

## What Changed

### Before (implicit precision)
```python
# Old code - dtype determined by JAX global config
params, _ = extract_fitting_params(tm)
obs_y = jnp.asarray(obs.spectrum)
```

### After (explicit precision)
```python
# New code - explicit dtype control
DTYPE = jnp.float64 if USE_FLOAT64 else jnp.float32

params, _ = extract_fitting_params(tm, dtype=DTYPE)
obs_y = jnp.asarray(obs.spectrum, dtype=DTYPE)
```

## Checking Your Results

The script now shows dtype information:

```
Available fitting parameters:
  planet_radius: 1.000000 [float64] (linear scale, range: (0, 1000))
  T: 1.500e+03 [float64] (log scale, range: (400, 3000))
  H2O: 1.200e-04 [float64] (log scale, range: (1e-12, 0.1))

Loading observations...
  Spectral bins: 30
  Wavelength range: 0.50 - 5.00 μm
  Data dtype: float64
```

If you see `float32` when you wanted `float64`, check that:
1. `USE_FLOAT64 = True` is set correctly
2. It's set BEFORE the imports
3. You restarted your Python session if testing interactively

## Performance Comparison

Expected performance on your RTX 3080 Ti:

| Precision | Forward Pass | Memory | Numerical Stability |
|-----------|-------------|--------|---------------------|
| float64   | ~150 ms     | ~8 GB  | ✅ Excellent        |
| float32   | ~80 ms      | ~4 GB  | ⚠️ Test carefully   |

Speedup: ~1.9x for float32

## Testing Float32 Safety

Before using float32 for production runs:

```bash
# 1. Run with float64 (default)
# Set: USE_FLOAT64 = True
python experiment_run.py

# Note final parameters and loss

# 2. Run with float32
# Set: USE_FLOAT64 = False  
python experiment_run.py

# 3. Compare:
#    - Do final parameters match? (should be within ~1%)
#    - Any NaN/Inf warnings?
#    - Does post-fit validation pass?
```

If float32 results differ significantly or show NaN/Inf, stick with float64.

## Troubleshooting

### "Result is float64 but I set USE_FLOAT64 = False"

**Cause:** JAX config was set after imports, or Python session wasn't restarted.

**Fix:**
```python
# MUST be in this order:
import jax
jax.config.update("jax_enable_x64", False)  # ← BEFORE other imports
import jax.numpy as jnp

from experiment import ...  # ← AFTER config
```

### "Getting NaN with float32"

**Cause:** Numerical overflow in atmospheric calculations.

**Fix:** Use float64 instead:
```python
USE_FLOAT64 = True
```

### "Memory error on GPU"

**Cause:** Model + data doesn't fit in 12GB VRAM.

**Fix:** Try float32 to halve memory usage:
```python
USE_FLOAT64 = False
```

## Advanced: Mixed Precision

For maximum performance with stability, see:
- `mixed_precision_solution.py` - Example implementations
- `float32_conversion_guide.md` - Detailed guide

This keeps critical operations in float64 (tau accumulation, exp) while using float32 elsewhere.

## Example Run Output

### Float64
```
======================================================================
JAX Configuration:
  Precision: float64
  Device: cuda:0
  Default dtype: float64
======================================================================

...

======================================================================
RUN COMPLETE!
======================================================================
Configuration used:
  Precision: float64
  Device: cuda:0
  Total time: 75.23 seconds
  Average time per step: 0.1505 seconds
======================================================================
```

### Float32
```
======================================================================
JAX Configuration:
  Precision: float32
  Device: cuda:0
  Default dtype: float32
======================================================================

...

======================================================================
RUN COMPLETE!
======================================================================
Configuration used:
  Precision: float32
  Device: cuda:0
  Total time: 42.15 seconds
  Average time per step: 0.0843 seconds
======================================================================
```

Notice: ~1.8x speedup with float32!

## Summary

**To switch precision:**
1. Open `experiment_run.py`
2. Change `USE_FLOAT64 = True` (or `False`)
3. Run: `python experiment_run.py`
4. Check output for dtype confirmation

**Recommendation:**
- **Production/Science**: Use `USE_FLOAT64 = True` (default)
- **Testing/Development**: Try `USE_FLOAT64 = False` after validation

Both modes use your GPU! The difference is just precision.
