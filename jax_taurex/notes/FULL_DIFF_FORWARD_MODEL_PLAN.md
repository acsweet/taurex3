# Full Differentiable Forward Model Implementation Plan

## Current Status: Ready to Assemble! 🚀

We now have **ALL** the necessary components implemented and tested. Time to create the fully differentiable forward model that recomputes everything on each forward pass.

## ✅ Available Components (All Implemented & Tested)

### Core Atmospheric Profiles
1. **`full_diff_compute_temperature_profile(T, nlayers)`**
   - Input: T parameter (scalar)
   - Output: Temperature array (nlayers,)
   - Purpose: Recompute T profile from current parameter

2. **`full_diff_compute_mixing_profiles(params, active_gases, nlayers)`**
   - Input: Parameter dict, gas names, nlayers
   - Output: Dict of {gas_name: mixing_profile_array}
   - Purpose: Recompute mixing ratios from current gas parameters

3. **`full_diff_compute_mu_profile(mixing_profiles, active_gases, active_gas_weights, fill_gases, fill_gas_weights, He_H2_ratio)`**
   - Input: Mixing profiles, molecular weights, fill gas info
   - Output: Mean molecular weight array (nlayers,)
   - Purpose: Recompute μ from current mixing ratios

4. **`full_diff_compute_altitude_profile(temperature_profile, pressure_levels, mu_profile, planet_mass, planet_radius)`**
   - Input: T, P, μ profiles + planet properties
   - Output: (altitude_boundaries, scale_height, gravity, deltaz)
   - Purpose: Recompute altitude/gravity/scale height from current T, μ, R

5. **`full_diff_compute_density_profile(pressure_profile, temperature_profile)`**
   - Input: P and T profiles
   - Output: Density array (nlayers,)
   - Purpose: Recompute density from ideal gas law

### Opacity Handling (NEW!)
6. **`load_opacity_data(active_gases, wngrid_target)`**
   - Input: Gas names, target wavenumber grid
   - Output: Dict of {gas_name: opacity_info_dict}
   - Purpose: Pre-load opacity grids (call once, outside JIT)

7. **`full_diff_interpolate_opacity_single_layer(T_layer, P_layer, opacity_info)`**
   - Input: T, P scalars + opacity grid
   - Output: Cross-section array (nwavenumbers,)
   - Purpose: Interpolate opacity at one (T, P) point

8. **`full_diff_interpolate_opacity_all_layers(T_profile, P_profile, opacity_info)`**
   - Input: T, P profiles + opacity grid
   - Output: Cross-section array (nlayers, nwavenumbers)
   - Purpose: Vectorized opacity interpolation for all layers

9. **`full_diff_compute_molecular_opacities(T_profile, P_profile, opacity_data_dict)`**
   - Input: T, P profiles + opacity data for all gases
   - Output: Dict of {gas_name: sigma_array (nlayers, nwavenumbers)}
   - Purpose: Compute all molecular opacities at current (T, P)

### Data Preparation
10. **`full_diff_prepare_model_data(taurex_model)`**
    - Input: TauREx model
    - Output: (static_data dict, initial_state dict)
    - Purpose: Extract static config + initial profiles for debugging

### Existing JAX Functions (Reusable)
11. **`compute_path_length_simple(total_layers, altitude_profile, deltaz, planet_radius)`**
    - Already implemented
    - Needs to use dynamically computed altitude

12. **`contribute_tau(startk, endk, density_offset, sigma, density, path, layer, tau)`**
    - Already implemented
    - Generic cross-section integration

13. **`contribute_tau_cia(startk, endk, density_offset, sigma, density, path, layer, tau)`**
    - Already implemented
    - CIA with ρ² dependence

14. **`compute_absorption(tau, dz, altitude_profile, planet_radius, star_radius)`**
    - Already implemented
    - Final absorption calculation

15. **`create_flux_weighted_binning_matrix(high_res_wngrid, bin_edges)`**
    - Already implemented
    - For spectral binning

16. **`bin_spectrum(high_res_spectrum, binning_matrix)`**
    - Already implemented
    - Apply binning to spectrum

## 🚧 What We Need to Create

### 1. Full Differentiable Forward Model (HIGH PRIORITY)

```python
def full_diff_create_forward_model(taurex_model, wngrid):
    """
    Create a fully differentiable forward model that recomputes
    all parameter-dependent quantities on each forward pass.
    
    This is the NEW version that matches TauREx's dynamic behavior.
    """
    # Extract static configuration (once)
    static_data, _ = full_diff_prepare_model_data(taurex_model)
    
    # Pre-load opacity grids (once, outside JIT)
    opacity_data = load_opacity_data(static_data['active_gases'], wngrid)
    
    # Load CIA data if needed (TODO)
    # cia_data = load_cia_data(...)
    
    # Package static data for JIT
    # ...
    
    @jax.jit
    def forward_model(params):
        """
        Full forward pass - recomputes everything from params.
        
        Flow:
        1. Compute T profile from params['T']
        2. Compute mixing profiles from params[gas_names]
        3. Compute μ from mixing profiles
        4. Compute altitude/gravity/scale height from T, μ, params['planet_radius']
        5. Interpolate molecular opacities at current (T, P)
        6. (TODO) Interpolate CIA opacities at current T
        7. Compute density from P, T
        8. Compute path lengths from altitude, planet_radius
        9. Integrate optical depth with all contributions
        10. Compute final absorption
        
        Returns:
            absorption: (nwavenumbers,) array
            tau: (nlayers, nwavenumbers) array
        """
        # Implementation here
        pass
    
    return forward_model
```

### 2. Full Differentiable Binned Forward Model

```python
def full_diff_create_binned_forward_model(taurex_model, observed_spectrum):
    """
    Create binned version of full differentiable forward model.
    
    Combines high-res forward model with spectral binning.
    """
    # Get high-res wavenumber grid
    from taurex.util import clip_native_to_wngrid
    high_res_wngrid = clip_native_to_wngrid(
        taurex_model.nativeWavenumberGrid,
        observed_spectrum.wavenumberGrid
    )
    
    # Create high-res forward model
    forward_high_res = full_diff_create_forward_model(taurex_model, high_res_wngrid)
    
    # Create binning matrix
    binning_matrix = create_flux_weighted_binning_matrix(
        high_res_wngrid, 
        observed_spectrum.binEdges
    )
    
    @jax.jit
    def forward_binned(params):
        """Binned spectrum from high-res forward model."""
        absorption_high_res, _ = forward_high_res(params)
        absorption_binned = bin_spectrum(absorption_high_res, binning_matrix)
        return absorption_binned
    
    return forward_binned
```

### 3. CIA Opacity Support (OPTIONAL for now)

If TauREx model includes CIA contributions, we need:

```python
def load_cia_data(cia_pairs, wngrid_target):
    """
    Pre-load CIA opacity grids.
    
    Similar to load_opacity_data but for CIA pairs.
    """
    pass

def full_diff_interpolate_cia_opacity(T_profile, cia_data):
    """
    Interpolate CIA opacity at current temperatures.
    
    Note: CIA is T-dependent only (not P-dependent).
    """
    pass
```

## 📋 Implementation Checklist

### Phase 1: Core Forward Model
- [ ] Implement `full_diff_create_forward_model`
  - [ ] Extract static data
  - [ ] Pre-load opacity grids
  - [ ] Define JIT-compiled forward function
  - [ ] Step 1-5: Profiles and opacities
  - [ ] Step 6-7: Density
  - [ ] Step 8: Path lengths (update to use dynamic altitude)
  - [ ] Step 9: Optical depth integration
  - [ ] Step 10: Final absorption

### Phase 2: Binned Forward Model
- [ ] Implement `full_diff_create_binned_forward_model`
  - [ ] Clip wavenumber grid
  - [ ] Create high-res model
  - [ ] Create binning matrix
  - [ ] Define binned forward function

### Phase 3: Testing
- [ ] Test full-res forward model vs TauREx
  - [ ] Pre-fit (should match perfectly)
  - [ ] Post-fit with parameter changes (should still match!)
- [ ] Test binned forward model vs TauREx
  - [ ] Pre-fit
  - [ ] Post-fit
- [ ] Add tests to `test_experiment.py`

### Phase 4: CIA Support (if needed)
- [ ] Check if test model uses CIA
- [ ] If yes, implement CIA loading
- [ ] If yes, implement CIA interpolation
- [ ] If yes, add CIA contribution to tau

## 🔑 Key Design Decisions

### What Goes in `static_data`?
Static (never changes during fitting):
- `nlayers`: Number of layers
- `pressure_profile`: Pressure at layer centers
- `pressure_levels`: Pressure at layer boundaries
- `planet_mass`: Planet mass (usually not fitted)
- `star_radius`: Star radius (usually not fitted)
- `active_gases`: Tuple of gas names
- `active_gas_weights`: Tuple of molecular weights
- `fill_gases`: Tuple of fill gas names
- `fill_gas_weights`: Tuple of fill gas molecular weights
- `He_H2_ratio`: He/H2 ratio for fill gases

### What Goes in `params`?
Dynamic (can be fitted):
- `T`: Temperature parameter
- `planet_radius`: Planet radius (relative to base radius)
- `H2O`, `CH4`, etc.: Gas mixing ratios

### What's Pre-loaded (Outside JIT)?
- Opacity grids for all active gases
- CIA grids for all CIA pairs (if applicable)
- Binning matrix
- Wavenumber grids

### What's Recomputed (Inside JIT)?
Everything that depends on parameters:
- Temperature profile
- Mixing ratio profiles
- Mean molecular weight
- Altitude/gravity/scale height
- Opacity values (interpolated at current T, P)
- Density
- Path lengths
- Optical depth
- Final absorption

## 🎯 Expected Behavior

### Before Fitting
```python
params_initial = extract_fitting_params(tm)
spectrum_jax = forward_model(params_initial)
spectrum_taurex = tm.model(wngrid)[1]

# Should match to machine precision!
assert jnp.allclose(spectrum_jax, spectrum_taurex, rtol=1e-8)
```

### After Fitting (Parameter Change)
```python
params_fitted = params_initial.copy()
params_fitted['T'] = 2000.0  # Changed from 1500
params_fitted['H2O'] = 1e-3  # Changed from 1e-4

spectrum_jax_fitted = forward_model(params_fitted)

# Update TauREx model
tm['T'] = 2000.0
tm['H2O'] = 1e-3

spectrum_taurex_fitted = tm.model(wngrid)[1]

# Should STILL match!
assert jnp.allclose(spectrum_jax_fitted, spectrum_taurex_fitted, rtol=1e-6)
```

This is the critical test that the old `create_forward_model` fails!

## 📊 Comparison: Old vs New

### Old `create_forward_model` (FROZEN STATE)
```python
def create_forward_model(taurex_model, wngrid):
    # ❌ Freezes state at initialization
    model_data = prepare_model_data(taurex_model)
    absorption_sigmas, cia_sigma, other_sigma = prepare_contributions(taurex_model, wngrid)
    
    def forward_model(params):
        # Only updates mixing weights
        # Opacities, altitude, etc. are frozen!
        pass
```

**Problem**: After fitting, uses old opacities interpolated at old T, old altitude computed from old planet_radius, etc.

### New `full_diff_create_forward_model` (DYNAMIC)
```python
def full_diff_create_forward_model(taurex_model, wngrid):
    # ✅ Only extracts truly static config
    static_data, _ = full_diff_prepare_model_data(taurex_model)
    opacity_data = load_opacity_data(...)  # Pre-load grids
    
    def forward_model(params):
        # ✅ Recomputes everything from current params
        T_profile = full_diff_compute_temperature_profile(params['T'], nlayers)
        # ... all other profiles
        opacities = full_diff_compute_molecular_opacities(T_profile, P_profile, opacity_data)
        # ... rest of forward pass
        pass
```

**Solution**: On every call, uses current parameter values to recompute all dependent quantities.

## 🚀 Next Steps

1. **Implement `full_diff_create_forward_model`** (30-50 lines)
2. **Implement `full_diff_create_binned_forward_model`** (10-20 lines)
3. **Add tests to `test_experiment.py`**:
   - `test_full_diff_forward_model_pre_fit`
   - `test_full_diff_forward_model_post_fit`
   - `test_full_diff_binned_forward_model`
4. **Run tests** to validate
5. **Update `compare.ipynb`** to use new forward model
6. **Test fitting** with new forward model

## 📝 Notes

- All arrays should use `jnp` (JAX NumPy) for differentiability
- Use `@jax.jit` only on complete forward pass, not individual components
- Keep static data in tuples (not dicts) for JIT compatibility
- Test each step incrementally

## 🎉 Success Criteria

- [ ] Forward model compiles with `@jax.jit`
- [ ] Pre-fit: Matches TauREx to ~1e-8 relative error
- [ ] Post-fit: Matches TauREx to ~1e-6 relative error (after parameter changes)
- [ ] Gradients can be computed with `jax.grad`
- [ ] Can fit and recover correct parameters

---

**Status**: All components ready, ready to assemble! 🎯

**Estimated Time**: 1-2 hours for implementation + testing

**Blocker**: None! We have everything we need.
