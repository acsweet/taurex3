# ADC 2023 Parameter Usage in JAX Forward Model

## Summary

**Your code uses the JAX differentiable forward model, NOT TauREx directly for optimization!**

The JAX model (`full_diff_create_binned_forward_model` from `experiment.py`) recomputes the entire atmospheric structure from parameters on every forward pass. This is what makes it fully differentiable.

## What Happens During Fitting

### 1. Model Setup (Once)
- TauREx `TransmissionModel` is created with initial parameters
- Opacity grids are loaded and prepared for interpolation
- JAX forward model wraps the TauREx model

### 2. During Each Optimization Step (Repeated ~500 times)
The JAX model recomputes **everything** from the current parameter values:

```python
# From experiment.py: full_diff_path_integral()

# 1. Temperature profile from T parameter
temperature_profile = full_diff_compute_temperature_profile(T, nlayers)

# 2. Mixing ratio profiles from gas parameters
mixing_profiles = full_diff_compute_mixing_profiles(params, active_gases, nlayers)

# 3. Mean molecular weight from mixing ratios
mu_profile = full_diff_compute_mu_profile(mixing_profiles, ...)

# 4. Altitude, gravity, scale height from hydrostatic equilibrium
altitude_profile, scaleheight, gravity, deltaz = full_diff_compute_altitude_profile(
    temperature_profile, pressure_levels, mu_profile, 
    planet_mass, planet_radius  # ← These can be fitted!
)

# 5. Density from ideal gas law
density_profile = full_diff_compute_density_profile(pressure_profile, temperature_profile)

# 6. Molecular opacities interpolated at each (T, P) point
molecular_opacities = full_diff_compute_molecular_opacities(
    temperature_profile, pressure_profile, opacity_data_dict
)

# 7. Path integral through atmosphere
spectrum = path_integral(...)  # Computes tau, absorption, transit depth
```

**All of this is differentiable!** Gradients flow backward through the entire computation.

## ADC Data: What's Used vs What's Available

### From Auxiliary Data (AuxillaryTable.csv)

| Parameter | Units | Used? | How Used | Notes |
|-----------|-------|-------|----------|-------|
| `star_temperature` | K | ✅ YES | BlackbodyStar initial temperature | Used for stellar SED |
| `star_radius_m` | m | ✅ YES | BlackbodyStar radius (→ R_sun) | Used in transit depth calculation |
| `star_mass_kg` | kg | ✅ YES | BlackbodyStar mass (→ M_sun) | Stored but not critical for transmission |
| `star_distance` | pc | ✅ YES | BlackbodyStar distance | System distance from Earth |
| `planet_mass_kg` | kg | ✅ YES | Planet mass (→ M_jup) | **Critical**: Used in hydrostatic eq. |
| `planet_orbital_period` | days | ⚠️ STORED | Planet orbital period | Stored in Planet object but not used in transmission model |
| `planet_distance` | AU | ⚠️ STORED | Semi-major axis | Stored in Planet object but not used in transmission model |
| `planet_surface_gravity` | m/s² | ❌ NO | N/A | **JAX recomputes** from mass/radius/altitude |

**Note:** `planet_radius_m` is NOT in auxiliary data! It comes from ground truth.

### From Ground Truth (FM_Parameter_Table.csv)

| Parameter | Units | Used? | How Used | Fittable? |
|-----------|-------|-------|----------|-----------|
| `planet_radius` | R_jup | ✅ YES | Initial planet radius | **YES** - fitted parameter |
| `planet_temp` | K | ✅ YES | Isothermal temperature | **YES** - fitted as 'T' |
| `log_H2O` | log₁₀ | ✅ YES | H₂O mixing ratio | **YES** - fitted as 'H2O' |
| `log_CO2` | log₁₀ | ✅ YES | CO₂ mixing ratio | **YES** - fitted as 'CO2' |
| `log_CO` | log₁₀ | ✅ YES | CO mixing ratio | **YES** - fitted as 'CO' |
| `log_CH4` | log₁₀ | ✅ YES | CH₄ mixing ratio | **YES** - fitted as 'CH4' |
| `log_NH3` | log₁₀ | ✅ YES | NH₃ mixing ratio | **YES** - fitted as 'NH3' |

## What Gets Fitted

Typical fitting configuration:
```python
fit_params = ['planet_radius', 'T', 'H2O', 'CO2', 'CO', 'CH4', 'NH3']
```

### Fixed Parameters (From ADC aux data, not fitted)
- Star temperature, radius, mass
- Planet mass
- System geometry (orbital period, semi-major axis)

### Fitted Parameters (Start from ground truth, optimized)
- `planet_radius`: Planet radius in Jupiter radii
- `T`: Atmospheric temperature (Kelvin)
- `H2O`, `CO2`, `CO`, `CH4`, `NH3`: Molecular mixing ratios

## Why planet_surface_gravity is NOT Used

The auxiliary data includes `planet_surface_gravity`, but **the JAX model ignores it** because:

1. **Surface gravity depends on fitted parameters:**
   ```
   g_surface = G * planet_mass / planet_radius²
   ```
   If we're fitting `planet_radius`, surface gravity must change!

2. **Gravity varies with altitude:**
   ```
   g(h) = G * planet_mass / (planet_radius + h)²
   ```
   The JAX model computes `g(h)` at each atmospheric layer.

3. **Scale height depends on gravity:**
   ```
   H = k_B * T / (μ * g)
   ```
   This is computed at each layer with the layer-specific gravity.

4. **Hydrostatic equilibrium connects everything:**
   ```
   dP/dz = -ρ * g(z)
   ```
   The altitude profile is computed by integrating this equation.

**All of this happens in JAX during each forward pass!**

## Parameter Flow

```
ADC Auxiliary Data
    ↓
TauREx Model Creation (once)
    ↓ (wraps)
JAX Forward Model Setup
    ↓
Optimization Loop (500 steps):
    ├─ Current Parameters (planet_radius, T, H2O, ...)
    ├─ Fixed Data (star params, planet mass, pressure grid)
    │   ↓
    ├─ JAX Recomputes:
    │   ├─ Temperature profile
    │   ├─ Mixing ratio profiles
    │   ├─ Mean molecular weight μ(z)
    │   ├─ Gravity g(z) ← from planet_mass, planet_radius, altitude
    │   ├─ Scale height H(z) ← from T(z), μ(z), g(z)
    │   ├─ Altitude z(i) ← from hydrostatic eq
    │   ├─ Density ρ(z) ← from P(z), T(z)
    │   ├─ Molecular opacities σ(z) ← interpolated at T(z), P(z)
    │   └─ Transit spectrum ← path integral
    │       ↓
    ├─ Compare to Observed Spectrum
    ├─ Compute Loss (MSE)
    ├─ Compute Gradients ∂Loss/∂params (via autodiff)
    └─ Update Parameters (Adam optimizer)
```

## Comparison: JAX vs Traditional TauREx

### Traditional TauREx Retrieval
1. TauREx computes spectrum for given parameters
2. Optimizer samples parameter space (e.g., nested sampling)
3. No gradients - uses sampling methods
4. Slow (~hours to days)

### Your JAX Approach
1. JAX recomputes entire physics from parameters
2. Gradients flow through all computations
3. Gradient-based optimization (Adam)
4. Fast (~minutes)

## What This Means for ADC Data

✅ **You ARE using all critical parameters:**
- Stellar parameters (temperature, radius, mass)
- Planet mass (critical for gravity)
- Ground truth for initialization

✅ **Surface gravity is correctly handled:**
- Not using the fixed value from auxiliary data
- Recomputing from fitted parameters during optimization
- Varies with altitude as it should

⚠️ **Orbital parameters are stored but not used:**
- `planet_orbital_period` 
- `planet_distance` (semi-major axis)
- These don't affect transmission spectrum calculation
- Would be needed for:
  - Phase curve analysis
  - Equilibrium temperature estimates
  - Transit timing

## Recommendations

### Current Implementation: ✅ CORRECT
Your code correctly:
1. Uses stellar parameters from auxiliary data
2. Uses planet mass from auxiliary data
3. Uses planet radius from ground truth (not aux)
4. Recomputes gravity/scale heights in JAX
5. Fits atmospheric parameters with gradients

### Optional Enhancements
If you want to use more ADC data:

1. **Estimate equilibrium temperature:**
   ```python
   # In create_taurex_model_from_adc_planet()
   T_eq = star_temp * sqrt(star_radius / (2 * planet_distance))
   # Use as prior or initial guess for T
   ```

2. **Add metallicity fitting:**
   ```python
   # Star class accepts metallicity parameter
   star = BlackbodyStar(..., metallicity=1.0)
   # Could fit if needed
   ```

3. **Use star distance for flux scaling:**
   ```python
   # Already included in current implementation
   star = BlackbodyStar(..., distance=star_distance_pc)
   ```

But these are minor - **your current approach is sound!**

## Testing

To verify everything is working:

```python
# Run on a planet with ground truth
result = fit_adc_planet(planet_id=1000, steps=500)

# Check if fitted params are close to ground truth
for param in ['planet_radius', 'T', 'H2O']:
    fitted = result['final_params'][param]
    truth = result['ground_truth'][param]
    error = abs(fitted - truth) / truth * 100
    print(f"{param}: {error:.2f}% error")
```

Expected results:
- planet_radius: <5% error
- T: <10% error  
- Molecular abundances: <1 order of magnitude (these are harder!)

## Summary

**Your implementation is correct!** You're:

1. ✅ Using JAX differentiable forward model (not TauREx directly)
2. ✅ Using all critical parameters from ADC data
3. ✅ Correctly handling gravity (recomputed, not fixed)
4. ✅ Fitting the right atmospheric parameters
5. ✅ Taking full advantage of gradient-based optimization

The only unused parameters (orbital period, semi-major axis) are not needed for transmission spectroscopy anyway.
