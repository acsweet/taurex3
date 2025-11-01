# Summary: Fully Differentiable JAX Implementation Setup

## What We've Done

### 1. Created Documentation
- **`TAUREX_MODEL_EXECUTION_FLOW.md`**: Deep dive into how TauREx works
  - Traces every step of `model.model()` execution
  - Documents parameter dependencies
  - Explains why frozen state causes post-fit divergence
  - Provides testing strategy

### 2. Restructured Code
- **Renamed**: `experiment_v2_5_3_1_1.py` → `experiment.py`
- **Created**: `test_experiment.py` for systematic testing
- **Created**: `FULLY_DIFFERENTIABLE_PROGRESS.md` to track progress

### 3. Implemented Core Components

All in `experiment.py` with `full_diff_` prefix:

#### ✅ Temperature Profile
```python
def full_diff_compute_temperature_profile(T, nlayers):
    return jnp.ones(nlayers) * T  # Isothermal
```

#### ✅ Mixing Ratio Profiles  
```python
def full_diff_compute_mixing_profiles(params, active_gases, nlayers):
    # Returns dict of {gas_name: profile} from current parameter values
```

#### ✅ Mean Molecular Weight
```python
def full_diff_compute_mu_profile(mixing_profiles, active_gases, ...):
    # μ = Σ(mix_ratio × molecular_weight) for all gases
```

#### ✅ Altitude/Gravity/Scale Height
```python
def full_diff_compute_altitude_profile(T_profile, P_levels, μ_profile, M, R):
    # Solves hydrostatic equilibrium
    # Returns: (altitude_boundaries, scale_height, gravity, deltaz)
```

#### ✅ Density Profile
```python
def full_diff_compute_density_profile(P_profile, T_profile):
    return P_profile / (KBOLTZ * T_profile)  # Ideal gas law
```

#### ✅ Model Data Preparation
```python
def full_diff_prepare_model_data(taurex_model):
    # Extracts static config + initial state for comparison
```

### 4. Created Comprehensive Tests

In `test_experiment.py`:

- `TestProfileInitialization`: Validates TauREx profile creation
- `TestModelData`: Validates data extraction
- `TestFullDifferentiableComponents`: Validates each JAX component against TauREx
  - `test_temperature_profile`
  - `test_mixing_profiles`
  - `test_mu_profile`
  - `test_altitude_profile`
  - `test_density_profile`
- `TestParameterExtraction`: Validates parameter extraction

Each test compares JAX output to TauREx with same inputs.

## What's Next (Critical Path)

### Immediate: Environment Setup
You need JAX installed in your Python environment. The test file failed with:
```
ModuleNotFoundError: No module named 'jax'
```

**To fix**:
```bash
# If using conda
conda install -c conda-forge jax jaxlib

# Or using pip
pip install "jax[cpu]"  # CPU version
# pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html  # GPU version
```

### Next: Test Current Components
Once JAX is installed:
```bash
python test_experiment.py
```

This will validate that our 5 implemented components correctly match TauREx.

### Then: Opacity Interpolation (The Big One)

This is the **critical missing piece**. Need to implement:

```python
def full_diff_interpolate_molecular_opacity(
    temperature_profile,  # Current T for each layer
    pressure_profile,     # Static P for each layer  
    wngrid,              # Wavenumber grid
    opacity_data,        # Pre-loaded opacity grids
):
    """
    For each layer:
    1. Find bracketing T and P in opacity grid
    2. Bilinear interpolation to get sigma at (T_layer, P_layer)
    3. Interpolate spectrally to wngrid
    4. Return (nlayers, nwavenumbers) array
    """
```

**Strategy**:
1. Pre-load all opacity data outside JIT (one-time cost)
2. JIT-compile the interpolation function
3. Use `jax.vmap` to vectorize over layers
4. Test against TauREx opacity fetch

### Finally: Assemble Full Forward Model

```python
@jax.jit
def full_diff_forward_model(params, static_data, opacity_data, wngrid):
    # 1. Compute T profile from params['T']
    # 2. Compute mixing profiles from params[gas_names]
    # 3. Compute μ from mixing profiles
    # 4. Compute altitude/gravity/scale_height from T, μ, params['planet_radius']
    # 5. ⚠️ Interpolate opacities at current (T, P) ← NEW!
    # 6. Compute density from P, T
    # 7. Compute path lengths from altitude, planet_radius
    # 8. Integrate optical depth
    # 9. Compute absorption
    return spectrum
```

## File Guide

```
experiment.py                          # Main implementation
├── [Old] extract_fitting_params       # Extract parameters from TauREx
├── [Old] prepare_model_data           # Extract static data (frozen approach)
├── [Old] create_forward_model         # Old frozen forward model
├── [New] full_diff_* functions        # New dynamic components
└── [TODO] full_diff_forward_model     # Final assembly

test_experiment.py                     # Test suite
├── Simple TauREx model fixture
├── Tests for each component
└── Can run standalone: python test_experiment.py

TAUREX_MODEL_EXECUTION_FLOW.md         # How TauREx actually works
FULLY_DIFFERENTIABLE_PROGRESS.md       # Implementation tracker
```

## Design Philosophy

### Key Principle: Pure Functions
Every `full_diff_*` function:
- Takes current parameter values as input
- Computes output from those values
- No cached/frozen state
- Fully differentiable by JAX

### Why This Matters
The old approach:
```python
# ❌ Computed once at initialization
opacity_data = prepare_contributions(tm, wngrid)  

def forward(params):
    # Uses frozen opacities, only reweights by abundance
```

The new approach:
```python
def forward(params):
    T = params['T']
    # ✅ Recompute opacities at current T!
    opacities = interpolate_opacity(T, P_grid, opacity_grids)
```

### Testing Strategy
1. **Component-level**: Each `full_diff_*` function tested individually
2. **Integration**: Full forward model tested against TauREx
3. **Fitting**: Verify parameters update correctly during optimization

## Quick Start (Next Session)

```bash
# 1. Install JAX (if not already installed)
pip install "jax[cpu]" optax

# 2. Run component tests
cd /Users/asweet/Code/research/taurex3
python test_experiment.py

# 3. If tests pass, implement opacity interpolation
# 4. Add tests for opacity interpolation
# 5. Assemble full forward model
# 6. Test end-to-end before fitting
# 7. Test with fitting to verify no divergence
```

## Questions to Consider

1. **Opacity Grid Size**: How much memory do pre-loaded opacity grids require?
2. **Performance**: Is there a speed vs accuracy tradeoff in grid resolution?
3. **GPU**: Worth using GPU acceleration for this model size?
4. **Batch Evaluation**: Could we evaluate multiple parameter sets in parallel?

## Success Criteria

✅ **Phase 1** (Current): Component tests pass
✅ **Phase 2**: Full forward model matches TauREx pre-fit  
✅ **Phase 3**: Full forward model matches TauREx post-fit (no divergence!)
✅ **Phase 4**: Fitting converges correctly
✅ **Phase 5**: Results match TauREx fitting

---

**Status**: Ready for environment setup and component testing.
**Blocker**: JAX not installed in current Python environment.
**Next Step**: Install JAX and run `python test_experiment.py`.
