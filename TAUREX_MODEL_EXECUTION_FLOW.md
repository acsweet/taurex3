# TauREx3 Model Execution Flow Deep Dive

This document traces the complete execution flow of a TauREx3 transmission model, detailing what happens at each step when `model.model()` is called. This is critical for understanding what needs to be replicated in a JAX implementation.

## Executive Summary

**The Core Problem**: Your JAX implementation freezes state (opacities, profiles, geometry) at initialization, but TauREx recomputes ALL parameter-dependent quantities on EVERY model evaluation. This is why you match pre-fit but diverge post-fit.

**What TauREx Does on Every `model()` Call**:
1. Rebuilds temperature profile from current `T` parameter
2. Rebuilds chemistry mixing ratios from current gas abundance parameters  
3. Recomputes mean molecular weight μ from updated mixing ratios
4. Recomputes altitude/gravity/scale-height from updated T, μ, and planet radius
5. Refetches opacities from cache at current (T, P) for each layer
6. Recomputes path lengths with updated geometry
7. Integrates optical depth with all updated quantities

---

## Complete Execution Flow

### Level 1: `model.model(wngrid, cutoff_grid=True)`

**Location**: `src/taurex/model/simplemodel.py:484-530`

```python
def model(self, wngrid=None, cutoff_grid=True):
    if not self.built:
        self.build()                          # ← Only once
    
    self.initialize_profiles()                 # ← EVERY CALL! ⚠️
    
    # Clip grid
    native_grid = self.nativeWavenumberGrid
    if wngrid is not None and cutoff_grid:
        native_grid = clip_native_to_wngrid(native_grid, wngrid)
    
    self._star.initialize(native_grid)         # ← EVERY CALL! ⚠️
    
    for contrib in self.contribution_list:
        contrib.prepare(self, native_grid)     # ← EVERY CALL! ⚠️
    
    absorp, tau = self.path_integral(native_grid, False)
    
    return native_grid, absorp, tau, None
```

**Key Insight**: Only `build()` happens once. Everything else is recomputed on every call, using current fitted parameter values.

---

### Level 2A: `initialize_profiles()`

**Location**: `src/taurex/model/simplemodel.py:222-258`

This is where the atmosphere state is rebuilt from scratch each time.

```python
def initialize_profiles(self):
    # 1. Compute pressure profile (usually static, but can be fitted)
    self.pressure.compute_pressure_profile()
    
    # 2. Initialize temperature profile with current parameter values
    self._temperature_profile.initialize_profile(
        self._planet, 
        self.pressure.nLayers, 
        self.pressure.profile
    )
    
    # 3. Setup photochemistry (if any)
    self._chemistry.set_star_planet(self.star, self.planet)
    
    # 4. Initialize chemistry with CURRENT T and P
    self._chemistry.initialize_chemistry(
        self.pressure.nLayers,
        self.temperatureProfile,    # ← Uses current T
        self.pressureProfile,
        None
    )
    
    # 5. Recompute altitude, gravity, scale height with updated μ
    self._compute_altitude_gravity_scaleheight_profile()
```

#### Deep Dive: Temperature Profile Initialization

**For Isothermal** (`src/taurex/data/profiles/temperature/isothermal.py`):

```python
def initialize_profile(self, planet, nlayers, pressure_profile):
    self.nlayers = nlayers
    self.pressure_profile = pressure_profile
    self.planet = planet
    # That's it! The profile property dynamically reads self.isoTemperature

@property
def profile(self):
    return np.ones(self.nlayers) * self.isoTemperature  # ← Uses CURRENT T value!
```

**Critical**: The temperature profile is NOT cached. Every access to `model.temperatureProfile` calls `self._temperature_profile.profile`, which reads the **current** fitted `T` parameter value.

#### Deep Dive: Chemistry Initialization

**Location**: `src/taurex/data/profiles/chemistry/taurexchemistry.py:400-430`

```python
def initialize_chemistry(self, nlayers, temperature_profile, pressure_profile, altitude_profile):
    mix_profile = []
    
    # For each gas, compute its mixing ratio profile
    for gas in self._gases:
        gas.initialize_profile(
            nlayers, 
            temperature_profile,  # ← Current T
            pressure_profile, 
            altitude_profile
        )
        mix_profile.append(gas.mixProfile)  # ← Uses CURRENT mix_ratio parameter!
    
    # Validate total < 1.0
    total_mix = sum(mix_profile)
    if np.any(total_mix > 1.0):
        raise InvalidChemistryException
    
    # Fill remainder with H2/He
    mixratio_remainder = 1.0 - total_mix
    mix_profile = self.fill_atmosphere(mixratio_remainder) + mix_profile
    
    self._mix_profile = np.vstack(mix_profile)
    
    # Compute mean molecular weight from CURRENT mixing ratios
    self.compute_mu_profile(nlayers)
```

**For ConstantGas**:

```python
def initialize_profile(self, nlayers, temp_prof, press_prof, alt_prof):
    # Simply fills array with current mix_ratio parameter value
    self._mix_profile = np.ones(nlayers) * self.mixRatio  # ← CURRENT value!
```

**Critical**: Chemistry reads the **current** gas abundance parameters every time. If you fit `H2O` from 1e-5 to 1e-4, the next `initialize_chemistry()` will use 1e-4.

#### Deep Dive: Mean Molecular Weight (μ) Computation

**Location**: `src/taurex/data/profiles/chemistry/chemistry.py:237-260`

```python
def compute_mu_profile(self, nlayers):
    from taurex.util import get_molecular_weight
    
    # Get active gases (those with opacities)
    active_mix = self.activeGasMixProfile    # Shape: (n_gases, n_layers)
    active_gases = self.activeGases
    
    # Get inactive gases (H2, He fill gas)
    inactive_mix = self.inactiveGasMixProfile
    inactive_gases = self.inactiveGases
    
    mu_profile = np.zeros(nlayers)
    
    # Sum weighted by molecular mass
    for gas_idx, gas_name in enumerate(active_gases):
        mass = get_molecular_weight(gas_name)
        mu_profile += active_mix[gas_idx] * mass
    
    for gas_idx, gas_name in enumerate(inactive_gases):
        mass = get_molecular_weight(gas_name)
        mu_profile += inactive_mix[gas_idx] * mass
    
    self._mu_profile = mu_profile  # In kg
```

**Critical**: μ depends on ALL gas mixing ratios. If you update any gas abundance, μ changes, which affects scale height and altitude.

#### Deep Dive: Altitude/Gravity/Scale Height Computation

**Location**: `src/taurex/model/simplemodel.py:328-353` and `src/taurex/data/planet.py:254-330`

```python
def _compute_altitude_gravity_scaleheight_profile(self, mu_profile=None):
    if mu_profile is None:
        mu_profile = self._chemistry.muProfile  # ← Uses CURRENT μ
    
    pressure_levels = self.pressure.pressure_profile_levels
    z, scaleheight, g, deltaz = self.planet.calculate_scale_properties(
        self.temperatureProfile,  # ← CURRENT T
        pressure_levels,
        mu_profile               # ← CURRENT μ
    )
    
    self.altitude_profile = z[:-1]
    self.scaleheight_profile = scaleheight[:-1]
    self.gravity_profile = g[:-1]
    self.altitude_boundaries = z
    self.deltaz = deltaz
```

**In Planet.calculate_scale_properties**:

```python
def calculate_scale_properties(self, temperature, pressure_levels, mu):
    nlayers = temperature.shape[0]
    scaleheight = np.zeros(nlayers)
    g = np.zeros(nlayers)
    z = np.zeros(nlayers + 1)
    deltaz = np.zeros(nlayers + 1)
    
    # Surface (bottom layer)
    g[0] = self.gravity  # ← Uses CURRENT planet_radius and planet_mass!
    scaleheight[0] = (KBOLTZ * temperature[0]) / (mu[0] * g[0])
    
    # Build altitude from bottom up (hydrostatic equilibrium)
    for i in range(1, nlayers + 1):
        # Integrate pressure to get altitude change
        deltaz[i] = (-1.0) * scaleheight[i-1] * np.log(
            pressure_levels[i] / pressure_levels[i-1]
        )
        z[i] = z[i-1] + deltaz[i]
        
        if i < nlayers:
            # Gravity decreases with altitude
            g[i] = self.gravity_at_height(z[i])  # ← G*M/(R+z)²
            # Scale height H = kT / (μg)
            scaleheight[i] = (KBOLTZ * temperature[i]) / (mu[i] * g[i])
    
    return z, scaleheight, g, deltaz[1:]
```

**Critical Dependencies**:
- Scale height H ∝ T / (μ × g)
- Altitude z depends on H through hydrostatic integration
- Gravity g depends on planet_radius and planet_mass
- If you fit T, any gas, planet_radius, or planet_mass → z, H, g all change!

---

### Level 2B: `star.initialize(native_grid)`

**Location**: `src/taurex/data/stellar/star.py:112-120`

```python
def initialize(self, wngrid):
    """Compute blackbody SED on wavenumber grid"""
    self.sed = black_body(wngrid, self.temperature)  # ← Uses CURRENT star temp
```

**For transmission models**, this mainly affects normalization if you're fitting stellar radius or temperature. Usually static, but CAN be fitted.

---

### Level 2C: `contribution.prepare(model, native_grid)`

This is where opacities are fetched from the cache at the **current** (T, P) state.

#### For AbsorptionContribution

**Location**: `src/taurex/contributions/absorption.py:228-320`

```python
def prepare_each(self, model, wngrid):
    sigma_xsec = None
    
    for gas in model.chemistry.activeGases:
        # Get mixing ratio profile for this gas (CURRENT parameter value!)
        gas_mix = model.chemistry.get_gas_mix_profile(gas)
        
        # Get opacity object from cache
        xsec = self._opacity_cache[gas]  # e.g., H2O, CH4
        
        # Initialize cross-section array
        if sigma_xsec is None:
            sigma_xsec = np.zeros(shape=(nlayers, ngrid))
        else:
            sigma_xsec[...] = 0.0
        
        # For each atmospheric layer, interpolate opacity at (T, P)
        for idx_layer, (temperature, pressure) in enumerate(
            zip(model.temperatureProfile, model.pressureProfile)
        ):
            # ⚠️ CRITICAL: Fetches opacity at CURRENT (T, P) for this layer!
            sigma_xsec[idx_layer] += (
                xsec.opacity(temperature, pressure, wngrid) * gas_mix[idx_layer]
            )
        
        yield gas, sigma_xsec

def prepare(self, model, wngrid):
    sigma_xsec = None
    for gas, sigma in self.prepare_each(model, wngrid):
        if sigma_xsec is None:
            sigma_xsec = np.zeros_like(sigma)
        sigma_xsec += sigma
    
    self.sigma_xsec = sigma_xsec  # Stores sum of all molecular opacities
```

**What `xsec.opacity(T, P, wngrid)` Does**:

The opacity cache stores pre-computed cross-sections on a (T, P, ν) grid. For each call:

```python
# Pseudocode for opacity interpolation
def opacity(self, temperature, pressure, wngrid):
    # Find bracketing T and P in cache grid
    T_idx_low, T_idx_high = find_bracket(self.T_grid, temperature)
    P_idx_low, P_idx_high = find_bracket(self.P_grid, pressure)
    
    # Get cross-sections at 4 corner points
    sigma_00 = self.sigma_grid[T_idx_low, P_idx_low, :]
    sigma_01 = self.sigma_grid[T_idx_low, P_idx_high, :]
    sigma_10 = self.sigma_grid[T_idx_high, P_idx_low, :]
    sigma_11 = self.sigma_grid[T_idx_high, P_idx_high, :]
    
    # Bilinear (or log-pressure linear) interpolation
    sigma_interp = interp_2d(
        sigma_00, sigma_01, sigma_10, sigma_11,
        temperature, T_low, T_high,
        pressure, P_low, P_high
    )
    
    # Interpolate to requested wavenumber grid if needed
    sigma_final = np.interp(wngrid, self.wn_grid, sigma_interp)
    
    return sigma_final
```

**Critical**: The interpolation happens at the **current** (T, P) values. If T or P changes (e.g., through fitting), you get different opacities even for the same gas abundance.

#### For CIAContribution

**Location**: `src/taurex/contributions/cia.py:207-252`

```python
def prepare_each(self, model, wngrid):
    sigma_cia = np.zeros(shape=(model.nLayers, wngrid.shape[0]))
    chemistry = model.chemistry
    
    for pair_name in self.ciaPairs:  # e.g., "H2-He", "H2-H2"
        cia = self._cia_cache[pair_name]
        sigma_cia[...] = 0.0
        
        # CIA opacity scales with BOTH gas densities
        cia_factor = (
            chemistry.get_gas_mix_profile(cia.pairOne) *    # ← CURRENT H2
            chemistry.get_gas_mix_profile(cia.pairTwo)      # ← CURRENT He
        )
        
        for idx_layer, temperature in enumerate(model.temperatureProfile):
            # Fetch CIA cross-section at CURRENT temperature
            _cia_xsec = cia.cia(temperature, wngrid)  # ← T-dependent interpolation
            sigma_cia[idx_layer] += _cia_xsec * cia_factor[idx_layer]
        
        yield pair_name, sigma_cia
```

**Critical**: CIA depends on:
1. Temperature (interpolated from T-dependent tables)
2. Product of mixing ratios (ρ_H2 × ρ_He)
3. Contributes as ρ² in the integral (density squared!)

---

### Level 3: `path_integral(native_grid, return_contrib=False)`

**Location (TransmissionModel)**: `src/taurex/model/transmission.py:135-186`

```python
def path_integral(self, wngrid, return_contrib=False):
    dz = self.deltaz          # ← From CURRENT altitude computation
    total_layers = self.nLayers
    density_profile = self.densityProfile  # ← P / (kB * T), uses CURRENT T!
    
    # Compute path lengths with CURRENT geometry
    if self.new_method:
        path_length = self.compute_path_length()
    else:
        path_length = self.compute_path_length_old(dz)
    
    tau = np.zeros(shape=(total_layers, wngrid.shape[0]))
    
    # For each layer, integrate along line of sight
    for layer in range(total_layers):
        dl = path_length[layer]
        end_k = total_layers - layer
        
        for contrib in self.contribution_list:
            if tau[layer].min() > 10:
                break  # Optically thick, stop
            
            # Integrate: τ = Σ σ(z) × ρ(z) × dl(z)
            contrib.contribute(
                self, 0, end_k, layer, layer,
                density_profile,  # ← Uses CURRENT density!
                tau,
                path_length=dl
            )
    
    # Convert optical depth to absorption
    absorption, tau = self.compute_absorption(tau, dz)
    return absorption, tau
```

#### Path Length Computation (Old Method)

**Location**: `src/taurex/model/transmission.py:96-134`

```python
def compute_path_length_old(self, dz):
    dl = []
    planet_radius = self._planet.fullRadius  # ← CURRENT planet_radius!
    total_layers = self.nLayers
    z = self.altitudeProfile  # ← From CURRENT altitude computation!
    
    for layer in range(0, total_layers):
        # Impact parameter at this layer
        p = (planet_radius + dz[0]/2 + z[layer])**2
        
        k = np.zeros(shape=(self.nLayers - layer))
        
        # First segment (layer itself)
        k[0] = np.sqrt(
            (planet_radius + dz[0]/2.0 + z[layer] + dz[layer]/2.0)**2 - p
        )
        
        # Segments through upper layers
        k[1:] = np.sqrt(
            (planet_radius + dz[0]/2 + z[layer+1:] + dz[layer+1:]/2)**2 - p
        )
        k[1:] -= np.sqrt(
            (planet_radius + dz[0]/2 + z[layer:self.nLayers-1] + 
             dz[layer:self.nLayers-1]/2)**2 - p
        )
        
        dl.append(k * 2.0)  # Factor of 2 for both sides
    
    return dl
```

**Critical**: Path lengths depend on:
- `planet_radius` (fitted parameter)
- `altitudeProfile` (which depends on T, μ, planet_radius, planet_mass)
- Geometric calculation assumes spherical shells

#### Optical Depth Integration

**Location**: `src/taurex/contributions/contribution.py:36-104`

```python
def contribute_tau_numpy(startk, endk, density_offset, 
                         sigma, density, path, 
                         nlayers, ngrid, layer, tau):
    """
    Computes: τ[layer, λ] += Σ_{z'=startk}^{endk} σ[z', λ] × ρ[z'] × dl[z']
    """
    _path = path[startk:endk, None]          # Integration path lengths
    _density = density[startk+density_offset:endk+density_offset, None]
    _sigma = sigma[startk+layer:endk+layer, :]
    
    tau[layer, :] += np.sum(_sigma * _path * _density, axis=0)
    return tau
```

For CIA (density squared):

```python
def contribute_cia_numpy(startk, endk, density_offset,
                         sigma, density, path,
                         nlayers, ngrid, layer, tau):
    """
    CIA: τ[layer, λ] += Σ σ[z', λ] × ρ[z']² × dl[z']
    """
    _path = path[startk:endk, None]
    _density = density[startk+density_offset:endk+density_offset, None]
    _sigma = sigma[startk+layer:endk+layer, :]
    
    tau[layer, :] += np.sum(_sigma * _path * _density * _density, axis=0)
```

#### Final Absorption Calculation

**Location**: `src/taurex/model/transmission.py:188-204`

```python
def compute_absorption(self, tau, dz):
    tau_exp = np.exp(-tau)  # Transmission = e^(-τ)
    ap = self.altitudeProfile[:, None]
    pradius = self._planet.fullRadius  # ← CURRENT planet_radius!
    sradius = self._star.radius
    _dz = dz[:, None]
    
    # Integrate (Rp + z) × (1 - e^(-τ)) × dz over all layers
    integral = np.sum((pradius + ap) * (1.0 - tau_exp) * _dz * 2.0, axis=0)
    
    # Transit depth: (Rp² + integral) / Rs²
    absorption = ((pradius**2.0) + integral) / (sradius**2)
    
    return absorption, tau
```

**Critical**: The final transit depth scales with:
- `planet_radius²` directly
- Integral that depends on altitude, optical depth, and layer thickness
- All divided by `star_radius²`

---

## Summary of Parameter Dependencies

### Every Parameter Affects:

| Parameter | Direct Effects | Indirect Effects |
|-----------|---------------|------------------|
| **T** (temperature) | - Temperature profile<br>- Scale height H ∝ T<br>- Density ρ ∝ 1/T<br>- Opacity interpolation | - Altitude (via H)<br>- Path lengths (via altitude)<br>- μ (if chemistry is T-dependent) |
| **Gas abundances** (H2O, CH4, etc.) | - Mixing ratio profiles<br>- Mean molecular weight μ<br>- Absorption opacity weights | - Scale height H ∝ 1/μ<br>- Altitude (via μ)<br>- Path lengths (via altitude)<br>- CIA (pair products) |
| **planet_radius** | - Surface gravity g ∝ 1/R²<br>- Path length geometry<br>- Final absorption ∝ R² | - Altitude (via g)<br>- Scale height H ∝ 1/g |
| **planet_mass** | - Surface gravity g ∝ M | - Altitude (via g)<br>- Scale height H ∝ 1/g |

### Computational Order (Dependencies):

```
Fitted Parameters (T, gases, planet_radius, planet_mass, etc.)
    ↓
1. Temperature Profile
    ↓
2. Chemistry → Mixing Ratios → Mean Molecular Weight μ
    ↓
3. Gravity g (from planet_radius, planet_mass)
    ↓
4. Scale Height H = kT / (μg)
    ↓
5. Altitude z (integrate H with pressure)
    ↓
6. Opacity Interpolation at (T, P) for each layer
    ↓
7. Density ρ = P / (kT)
    ↓
8. Path Lengths (from planet_radius, altitude)
    ↓
9. Optical Depth τ = Σ σ(T,P) × ρ(T) × path(z,R)
    ↓
10. Absorption = f(R², τ, z, Rs²)
```

---

## Implications for JAX Implementation

### What You're Currently Doing Wrong:

```python
# ❌ FREEZING STATE AT INITIALIZATION
model_data = prepare_model_data(taurex_model)  # Once
absorption_sigmas, cia_sigma, other_sigma = prepare_contributions(
    taurex_model, wngrid
)  # Once

def forward_model(params):
    # Only updates gas mixing weights, not T-dependent opacities!
    # Only updates planet_radius, not altitude/scale height!
    # ... 
```

### What You Must Do:

```python
def jax_forward_model(params):
    """FULL RECOMPUTE like TauREx does"""
    
    # 1. Extract parameters
    T = params['T']
    planet_radius = params['planet_radius']
    gas_abundances = {g: params[g] for g in active_gases}
    # ... etc
    
    # 2. Build temperature profile (easy, usually just T * ones(nlayers))
    temperature_profile = build_temperature_profile(T, nlayers)
    
    # 3. Build mixing ratio profiles
    mix_profiles = build_mixing_profiles(gas_abundances, nlayers)
    
    # 4. Compute mean molecular weight
    mu_profile = compute_mu(mix_profiles, molecular_weights)
    
    # 5. Compute gravity
    g = compute_gravity_profile(planet_radius, planet_mass, altitude_old)
    
    # 6. Compute scale height
    H = compute_scale_height(temperature_profile, mu_profile, g)
    
    # 7. Compute altitude from hydrostatic equilibrium
    z, deltaz = compute_altitude(H, pressure_levels)
    
    # 8. ⚠️ RECOMPUTE OPACITIES at current (T, P)
    #    This is the hard part for JAX!
    absorption_sigma = interpolate_opacities_2d(
        opacity_cache_grid,  # Pre-loaded (T_grid, P_grid, wn_grid, sigma_grid)
        temperature_profile,
        pressure_profile,
        wngrid
    )
    
    cia_sigma = interpolate_cia_1d(
        cia_cache_grid,      # Pre-loaded (T_grid, wn_grid, sigma_grid)
        temperature_profile,
        wngrid
    )
    
    # 9. Weight opacities by mixing ratios
    weighted_absorption = weight_opacities(absorption_sigma, mix_profiles)
    weighted_cia = weight_cia(cia_sigma, mix_profiles)  # Uses products!
    
    # 10. Compute density
    density = compute_density(pressure_profile, temperature_profile, mu_profile)
    
    # 11. Compute path lengths
    path_lengths = compute_path_length(planet_radius, z, deltaz, nlayers)
    
    # 12. Integrate optical depth
    tau = integrate_tau(
        weighted_absorption, weighted_cia, 
        density, path_lengths, nlayers
    )
    
    # 13. Compute final absorption
    absorption = compute_final_absorption(tau, z, deltaz, planet_radius, star_radius)
    
    return absorption
```

### The Opacity Interpolation Challenge:

The hardest part for JAX is step 8. You need to:

1. **Pre-load** all opacity grids into JAX arrays (one-time, outside JIT)
2. **JIT-compile** the interpolation function
3. Handle **each gas separately** or use vmap

```python
# Outside JIT: Load all opacity data
opacity_data = {}
for gas in active_gases:
    xsec = opacity_cache[gas]
    opacity_data[gas] = {
        'T_grid': jnp.array(xsec.temperature_grid),
        'P_grid': jnp.array(xsec.pressure_grid),
        'wn_grid': jnp.array(xsec.wavenumber_grid),
        'sigma': jnp.array(xsec.cross_section_grid)  # Shape: (nT, nP, nWN)
    }

# JIT-able function
@jax.jit
def interpolate_opacity_layer(T, P, wngrid, opacity_grid):
    """Interpolate opacity at single (T, P) point"""
    # Bilinear interpolation in (T, P), then spectral interpolation
    sigma_TP = jax.scipy.ndimage.map_coordinates(
        opacity_grid['sigma'],
        [T_index, P_index, wn_indices],
        order=1  # Linear
    )
    return sigma_TP

# Vectorize over layers
interpolate_all_layers = jax.vmap(interpolate_opacity_layer, in_axes=(0, 0, None, None))
```

---

## Recommended Testing Strategy

Test your JAX implementation by **controlling which parameters change**:

### Test 1: Only Fit Gas Abundances (T, planet_radius fixed)

```python
# TauREx will:
# - Keep T, P, planet_radius constant → altitude, path lengths unchanged
# - Update mixing ratios → μ changes slightly → altitude changes slightly
# - Fetch same opacities (same T, P) → only weighting changes

# Your JAX should:
# - Recompute μ from new mixing ratios
# - Recompute altitude (small change)
# - Keep opacity tables same, update weights
```

**Expected**: Close match if you recompute μ and altitude.

### Test 2: Only Fit Temperature (gases, planet_radius fixed)

```python
# TauREx will:
# - Update T → new temperature profile
# - Fetch NEW opacities at new (T, P) for each layer ⚠️
# - Recompute μ (unchanged if gas-independent chemistry)
# - Recompute H, z, g
# - Recompute density ρ ∝ 1/T
# - Recompute path lengths

# Your JAX must:
# - Reinterpolate ALL opacities at new T
# - Recompute altitude, scale height, density
# - Recompute path lengths
```

**Expected**: Large divergence if you don't reinterpolate opacities!

### Test 3: Only Fit Planet Radius (T, gases fixed)

```python
# TauREx will:
# - Update planet_radius → new geometry
# - Recompute g ∝ 1/R²
# - Recompute H ∝ 1/g
# - Recompute z (via H integration)
# - Recompute path lengths (geometric effect + altitude effect)
# - Final absorption ∝ R²

# Your JAX must:
# - Recompute gravity, scale height, altitude
# - Recompute path lengths with new geometry
```

**Expected**: Divergence if you freeze geometry!

---

## Quick Diagnostic

To confirm the opacity freezing hypothesis:

```python
# In your experiment file
import numpy as np

# Before fit
tm.initialize_profiles()
for contrib in tm.contribution_list:
    contrib.prepare(tm, wngrid)
sigma_before = contrib.sigma_xsec.copy()

# After fit (change T parameter)
tm['T'] = 2000.0  # Different from initial

# Rebuild
tm.initialize_profiles()
for contrib in tm.contribution_list:
    contrib.prepare(tm, wngrid)
sigma_after = contrib.sigma_xsec

print("Opacity changed?", not np.allclose(sigma_before, sigma_after))
# Should print: Opacity changed? True
```

If opacities change but your JAX implementation keeps them frozen, you've found the bug!

---

## Additional Notes

### Binning

Your binning concern is secondary. The issue is:

```python
# TauREx typical flow:
native_wn, native_spectrum, tau, _ = model.model(wngrid=None)
binned_wn, binned_spectrum, _, _ = binner.bindown(native_wn, native_spectrum)

# Your flow should match exactly:
native_spectrum_jax = jax_forward_model(params)
binned_spectrum_jax = jax_bindown(native_wn, native_spectrum_jax, bin_edges)
```

Use the **same binner** (FluxBinner or SimpleBinner) and **same edges** as TauREx. The binning math is straightforward; the model is where complexity lies.

### Caching Strategy

To make JAX fast:

1. **Load ALL opacity data once** into GPU/CPU arrays outside JIT
2. **JIT only the interpolation** and forward model
3. Accept that first compile is slow, subsequent calls are fast
4. Consider **lower-resolution opacity grids** for faster interpolation (at cost of accuracy)

---

## Conclusion

Your ChatGPT advisor was exactly right:

> "You're freezing state that Taurex recomputes on every likelihood call"

TauREx is a **fully dynamic** forward model. Every parameter change triggers a cascade of recomputations. Your JAX version must do the same, especially:

1. **Opacity interpolation** at current (T, P)
2. **Altitude/scale height** from current (T, μ, planet_radius, planet_mass)
3. **Path lengths** from current geometry
4. **Density** from current (P, T, μ)

The good news: All of these are differentiable operations, so JAX can handle them! The challenge is preloading the opacity grids and JIT-compiling the interpolation efficiently.
