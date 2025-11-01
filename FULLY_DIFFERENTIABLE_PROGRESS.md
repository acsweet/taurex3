# Fully Differentiable JAX Implementation - Progress

## Status: Component Implementation Phase

We are implementing a fully differentiable JAX version of the TauREx transmission model that recomputes all dependent quantities on each forward pass, matching the behavior documented in `TAUREX_MODEL_EXECUTION_FLOW.md`.

## Completed Components ✅

### 1. Temperature Profile (`full_diff_compute_temperature_profile`)
- **Status**: Implemented
- **Function**: Computes temperature profile from current T parameter
- **For isothermal**: Returns `T * ones(nlayers)`
- **Test**: `test_temperature_profile` in `test_experiment.py`

### 2. Mixing Ratio Profiles (`full_diff_compute_mixing_profiles`)
- **Status**: Implemented  
- **Function**: Computes mixing ratio profiles for all active gases
- **For constant gas**: Returns `mix_ratio * ones(nlayers)` for each gas
- **Test**: `test_mixing_profiles` in `test_experiment.py`

### 3. Mean Molecular Weight (`full_diff_compute_mu_profile`)
- **Status**: Implemented
- **Function**: Computes μ from current mixing ratios and molecular weights
- **Formula**: `μ = Σ(mix_ratio_i × molecular_weight_i)` for all gases
- **Includes**: Active gases + fill gases (H2/He split)
- **Test**: `test_mu_profile` in `test_experiment.py`

### 4. Altitude/Gravity/Scale Height (`full_diff_compute_altitude_profile`)
- **Status**: Implemented
- **Function**: Solves hydrostatic equilibrium for altitude boundaries
- **Key computations**:
  - Surface gravity: `g₀ = GM/R²`
  - Scale height: `H = kT/(μg)`
  - Altitude integration: `dz = -H × ln(P_next/P_i)`
  - Gravity at height: `g(h) = GM/(R+h)²`
- **Test**: `test_altitude_profile` in `test_experiment.py`

### 5. Density Profile (`full_diff_compute_density_profile`)
- **Status**: Implemented
- **Function**: Computes density from ideal gas law
- **Formula**: `ρ = P / (k_B × T)`
- **Test**: `test_density_profile` in `test_experiment.py`

### 6. Model Data Preparation (`full_diff_prepare_model_data`)
- **Status**: Implemented
- **Function**: Extracts static configuration from TauREx model
- **Returns**: 
  - `static_data`: Things that don't change (nlayers, pressure grid, planet mass, etc.)
  - `initial_state`: Initial profiles for debugging/comparison

## Next Steps 🚧

### 7. Opacity Interpolation (CRITICAL)
- **Status**: Not started
- **Challenge**: This is the most complex piece
- **Requirements**:
  - Pre-load opacity grids from OpacityCache into JAX arrays
  - Implement bilinear interpolation in (T, P) space
  - Handle spectral interpolation to wavenumber grid
  - Make it JIT-compilable
  - Vectorize over layers

**Approach**:
```python
def full_diff_interpolate_molecular_opacity(
    T_layer, P_layer, wngrid, 
    opacity_T_grid, opacity_P_grid, opacity_wn_grid, opacity_sigma_grid
):
    """Interpolate opacity at (T, P) for one layer."""
    # 1. Find bracketing indices in T and P
    # 2. Bilinear interpolation (or log-P linear)
    # 3. Spectral interpolation to wngrid
    # 4. Return sigma for this layer
    pass

# Vectorize over all layers
interpolate_all_layers = jax.vmap(
    full_diff_interpolate_molecular_opacity,
    in_axes=(0, 0, None, None, None, None, None)
)
```

### 8. CIA Opacity Interpolation  
- **Status**: Not started
- **Similar to molecular opacity but**:
  - Only T-dependent (not P)
  - Needs to be weighted by `ρ_i × ρ_j` for gas pairs
  - Applied with `ρ²` in tau integration

### 9. Path Length Computation
- **Status**: Partially implemented (reuse existing `compute_path_length_simple`)
- **Needs**: Update to use dynamically computed altitude profile
- **Current**: Uses static altitude from initialization

### 10. Full Forward Model (`full_diff_forward_model`)
- **Status**: Not started
- **Will integrate**:
  1. Temperature profile computation
  2. Mixing ratio profiles
  3. μ computation
  4. Altitude/gravity/scale height
  5. **Opacity interpolation** ← blocking
  6. Density computation
  7. Path length computation
  8. Optical depth integration
  9. Final absorption calculation

### 11. Testing & Validation
- **Pre-fit validation**: Ensure JAX matches TauREx with initial parameters
- **Post-fit validation**: Ensure JAX updates all quantities correctly
- **Parameter sensitivity tests**: Test each parameter independently

## File Structure

```
/Users/asweet/Code/research/taurex3/
├── experiment.py                       # Main implementation
│   ├── [Existing] extract_fitting_params
│   ├── [Existing] prepare_model_data
│   ├── [Existing] prepare_contributions
│   ├── [Existing] create_forward_model
│   ├── [Existing] path_integral
│   ├── [NEW] full_diff_compute_temperature_profile ✅
│   ├── [NEW] full_diff_compute_mixing_profiles ✅
│   ├── [NEW] full_diff_compute_mu_profile ✅
│   ├── [NEW] full_diff_compute_altitude_profile ✅
│   ├── [NEW] full_diff_compute_density_profile ✅
│   ├── [NEW] full_diff_prepare_model_data ✅
│   ├── [TODO] full_diff_interpolate_opacity 🚧
│   ├── [TODO] full_diff_interpolate_cia 🚧
│   └── [TODO] full_diff_forward_model 🚧
│
├── test_experiment.py                  # Test suite
│   ├── TestProfileInitialization ✅
│   ├── TestModelData ✅
│   ├── TestFullDifferentiableComponents ✅
│   │   ├── test_temperature_profile ✅
│   │   ├── test_mixing_profiles ✅
│   │   ├── test_mu_profile ✅
│   │   ├── test_altitude_profile ✅
│   │   └── test_density_profile ✅
│   ├── TestParameterExtraction ✅
│   ├── [TODO] TestOpacityInterpolation 🚧
│   └── [TODO] TestFullForwardModel 🚧
│
├── TAUREX_MODEL_EXECUTION_FLOW.md      # Documentation ✅
└── FULLY_DIFFERENTIABLE_PROGRESS.md    # This file ✅
```

## Testing Strategy

### Phase 1: Component Testing (Current)
- Test each component individually
- Compare JAX output to TauREx with identical inputs
- Use initial parameter values (no fitting yet)

### Phase 2: Integration Testing  
- Test full forward model (once opacity interpolation is done)
- Compare high-res spectrum to TauREx
- Test with different parameter values

### Phase 3: Fitting Testing
- Test that fitted parameters change outputs correctly
- Compare fitted results to TauREx fitted results
- Test gradient computation

## Key Insights

### Why Full Recomputation?
From `TAUREX_MODEL_EXECUTION_FLOW.md`:

1. **Temperature changes** → new opacities (interpolated at new T)
2. **Gas abundances change** → new μ → new scale height → new altitude
3. **Planet radius changes** → new gravity → new altitude → new path lengths
4. **Everything is connected!**

The old approach froze opacities and geometry at initialization. The new approach recomputes everything from current parameter values.

### Performance Considerations
- **One-time cost**: Load all opacity grids into memory
- **JIT compilation**: First call is slow, subsequent calls are fast
- **Batch processing**: Can evaluate multiple parameter sets in parallel
- **GPU acceleration**: JAX can use GPU for large computations

## Dependencies

Required packages (check your environment):
- `jax[cpu]` or `jax[cuda]` for GPU
- `jaxlib`
- `optax` (for optimization)
- `taurex` (for comparison and data loading)
- `numpy`, `scipy`

## Running Tests

```bash
cd /Users/asweet/Code/research/taurex3

# Ensure you're in the correct environment with JAX installed
# Then run:
python test_experiment.py
```

Expected output (once environment is set up):
```
======================================================================
Testing JAX Implementation vs TauREx
======================================================================
...
✅ ALL TESTS PASSED!
```

## Next Session Goals

1. **Fix environment setup** (install JAX if needed)
2. **Run component tests** to verify implementations so far
3. **Implement opacity interpolation** (the critical missing piece)
4. **Test opacity interpolation** against TauREx
5. **Assemble full forward model**
6. **Validate end-to-end** before fitting

## Notes

- All implemented functions are pure functions (no side effects)
- All computations use JAX arrays for automatic differentiation
- Parameter dependencies are explicitly documented in function signatures
- Tests compare numerical outputs with small tolerances (rtol=1e-6 to 1e-10)
