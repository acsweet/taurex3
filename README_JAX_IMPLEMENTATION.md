# Fully Differentiable JAX Implementation of TauREx Transmission Model

This directory contains a fully differentiable JAX implementation of the TauREx atmospheric transmission model. Unlike the initial implementation which froze opacities and geometry, this version recomputes all parameter-dependent quantities on every forward pass, matching TauREx's behavior.

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| **`IMPLEMENTATION_SUMMARY.md`** | **START HERE** - Overview of what's been implemented and next steps |
| `TAUREX_MODEL_EXECUTION_FLOW.md` | Deep dive into how TauREx works internally |
| `FULLY_DIFFERENTIABLE_PROGRESS.md` | Detailed progress tracker with component status |
| `README_JAX_IMPLEMENTATION.md` | This file |

## 🚀 Quick Start

### 1. Check Your Environment

```bash
python check_environment.py
```

This will tell you if you have all required packages installed.

### 2. Install Missing Packages (if needed)

```bash
# Using pip
pip install jax jaxlib optax numpy scipy taurex

# Or using conda
conda install -c conda-forge jax jaxlib optax numpy scipy
pip install taurex  # TauREx usually via pip
```

### 3. Run Component Tests

```bash
python test_experiment.py
```

This validates that each JAX component matches TauREx output.

### 4. (Later) Run Full Model Tests

Once opacity interpolation is implemented:

```bash
python -m pytest test_experiment.py -v
```

## 📁 File Structure

```
.
├── experiment.py                      # Main implementation
│   ├── [Existing] Old frozen model
│   └── [New] full_diff_* components (dynamic recomputation)
│
├── test_experiment.py                 # Comprehensive test suite
├── check_environment.py               # Environment verification script
│
├── IMPLEMENTATION_SUMMARY.md          # ⭐ Start here
├── TAUREX_MODEL_EXECUTION_FLOW.md     # How TauREx works
├── FULLY_DIFFERENTIABLE_PROGRESS.md   # Progress tracker
└── README_JAX_IMPLEMENTATION.md       # This file
```

## ✅ Implemented Components

### Core Atmospheric Profiles
- [x] **Temperature Profile**: `full_diff_compute_temperature_profile`
- [x] **Mixing Ratios**: `full_diff_compute_mixing_profiles`
- [x] **Mean Molecular Weight**: `full_diff_compute_mu_profile`
- [x] **Altitude/Gravity/Scale Height**: `full_diff_compute_altitude_profile`
- [x] **Density**: `full_diff_compute_density_profile`

### Data Preparation
- [x] **Model Data Extraction**: `full_diff_prepare_model_data`
- [x] **Parameter Extraction**: `extract_fitting_params`

## 🚧 In Progress / TODO

### Critical Path
- [ ] **Opacity Interpolation** ← Blocking everything else
  - [ ] Molecular opacity (T, P) interpolation
  - [ ] CIA opacity (T) interpolation
  - [ ] Pre-load opacity grids into JAX arrays
  - [ ] JIT-compile interpolation functions

### Integration
- [ ] **Path Length Computation** (update to use dynamic altitude)
- [ ] **Full Forward Model** (`full_diff_forward_model`)
- [ ] **End-to-End Testing**

### Validation
- [ ] Pre-fit validation (JAX matches TauREx)
- [ ] Post-fit validation (no divergence!)
- [ ] Parameter sensitivity tests

## 🧪 Testing Strategy

### Phase 1: Component Testing (Current)
Each component is tested individually against TauREx:
```python
# Example: Temperature profile test
jax_T_profile = full_diff_compute_temperature_profile(T, nlayers)
taurex_T_profile = tm.temperatureProfile
assert jnp.allclose(jax_T_profile, taurex_T_profile)
```

### Phase 2: Integration Testing
Full forward model tested against TauREx:
```python
# High-res spectrum comparison
jax_spectrum = full_diff_forward_model(params, ...)
taurex_spectrum = tm.model(wngrid)[0]
assert jnp.allclose(jax_spectrum, taurex_spectrum, rtol=1e-6)
```

### Phase 3: Fitting Testing
Verify parameters update correctly:
```python
# Test temperature change
params_new = params.copy()
params_new['T'] = 2000.0  # Changed from 1500
spectrum_new = full_diff_forward_model(params_new, ...)
# Spectrum should change significantly
assert not jnp.allclose(spectrum_new, spectrum_old)
```

## 🔑 Key Design Principles

### 1. Pure Functions
Every `full_diff_*` function:
- Takes current parameter values as input
- Computes output from scratch
- No cached/frozen state
- Fully differentiable

### 2. Explicit Dependencies
Function signatures make dependencies clear:
```python
def full_diff_compute_altitude_profile(
    temperature_profile,  # Depends on params['T']
    pressure_levels,      # Static
    mu_profile,          # Depends on gas abundances
    planet_mass,         # Static
    planet_radius        # Depends on params['planet_radius']
):
    # Recomputes altitude from current values
```

### 3. Match TauREx Exactly
Each component replicates TauREx's computation:
- Same formulas
- Same integration methods  
- Same numerical precision
- Validated with tests

## 🎯 Why This Approach?

### The Problem with Frozen State

**Old approach** (causes post-fit divergence):
```python
# ❌ Computed once at initialization
opacity_data = prepare_contributions(tm, wngrid)
altitude = compute_altitude_once(tm)

def forward(params):
    # Only reweights frozen opacities
    weighted_sigma = weight_opacities(opacity_data, params['H2O'])
    # Uses frozen altitude
    path_lengths = compute_paths(altitude)
```

**What TauREx actually does**:
```python
def model():
    initialize_profiles()  # ← Rebuilds T, μ, chemistry every time!
    star.initialize(wngrid)
    for contrib in contributions:
        contrib.prepare(self, wngrid)  # ← Refetches opacities every time!
    absorption, tau = path_integral(wngrid)
```

### The Solution: Full Recomputation

**New approach** (matches TauREx):
```python
@jax.jit
def full_diff_forward_model(params, static_data, opacity_grids, wngrid):
    # 1. Recompute T profile from params['T']
    T_profile = full_diff_compute_temperature_profile(params['T'], nlayers)
    
    # 2. Recompute mixing ratios from params[gas_names]
    mixing_profiles = full_diff_compute_mixing_profiles(params, active_gases, nlayers)
    
    # 3. Recompute μ from mixing ratios
    mu_profile = full_diff_compute_mu_profile(mixing_profiles, ...)
    
    # 4. Recompute altitude from current T, μ, planet_radius
    z, H, g, dz = full_diff_compute_altitude_profile(
        T_profile, P_levels, mu_profile, 
        planet_mass, params['planet_radius'] * R_base
    )
    
    # 5. ⚠️ Reinterpolate opacities at current (T, P)
    opacities = full_diff_interpolate_opacity(T_profile, P_grid, opacity_grids, wngrid)
    
    # 6. Integrate and return spectrum
    ...
```

## 📊 Parameter Dependencies

| Parameter | Affects | Ripple Effects |
|-----------|---------|----------------|
| `T` | Temperature profile | → Scale height → Altitude → Path lengths<br>→ Opacity interpolation<br>→ Density |
| `H2O, CH4, ...` | Mixing ratios | → μ → Scale height → Altitude<br>→ Opacity weighting |
| `planet_radius` | Geometry | → Gravity → Scale height → Altitude<br>→ Path lengths<br>→ Final absorption (R²) |
| `planet_mass` | Gravity | → Scale height → Altitude |

**Everything is connected!** That's why we need full recomputation.

## 🔬 Example: Temperature Change

```python
# Initial state (T = 1500 K)
params = {'T': 1500.0, 'H2O': 1e-4, 'planet_radius': 1.0}
spectrum_1500 = full_diff_forward_model(params, ...)

# After fitting (T = 2000 K)
params_fitted = {'T': 2000.0, 'H2O': 1e-4, 'planet_radius': 1.0}
spectrum_2000 = full_diff_forward_model(params_fitted, ...)

# What changes?
# ✅ Temperature profile: 1500 → 2000 everywhere
# ✅ Opacities: Reinterpolated at T=2000 (different values!)
# ✅ Scale height: H ∝ T, so increases by ~33%
# ✅ Altitude: Changes due to new H
# ✅ Density: ρ ∝ 1/T, so decreases by ~25%
# ✅ Path lengths: Change due to new altitude
# ✅ Final spectrum: All effects combined!
```

With the old frozen approach, only density changed. With full recomputation, everything updates correctly!

## 📈 Performance Considerations

### One-Time Costs
- Loading opacity grids into memory
- JIT compilation (first call only)

### Per-Evaluation Costs
- Profile recomputation: ~ms
- Opacity interpolation: ~ms (depends on grid size)
- Path integral: ~ms

### Optimizations
- Use `@jax.jit` for compiled speed
- Use `jax.vmap` for vectorization
- Batch parameter evaluations
- Optional: GPU acceleration

## 🐛 Debugging Tips

### Test Individual Components
```python
# Isolate which component is wrong
python -c "from test_experiment import *; tm = ...; test_altitude_profile(tm)"
```

### Compare Numerical Values
```python
# Check where JAX diverges from TauREx
import numpy as np
diff = jax_output - taurex_output
print(f"Max difference: {np.max(np.abs(diff))}")
print(f"Relative error: {np.max(np.abs(diff / taurex_output))}")
```

### Visualize Profiles
```python
import matplotlib.pyplot as plt
plt.plot(pressure_profile, altitude_jax, label='JAX')
plt.plot(pressure_profile, altitude_taurex, label='TauREx')
plt.xscale('log')
plt.legend()
plt.show()
```

## 🤝 Contributing

When adding new components:

1. **Implement** the function in `experiment.py` with `full_diff_` prefix
2. **Document** parameters and dependencies in docstring
3. **Test** in `test_experiment.py` against TauREx
4. **Update** `FULLY_DIFFERENTIABLE_PROGRESS.md` checklist

## 📝 Notes

- All arrays use `jax.numpy` (not `numpy`) for differentiability
- Use `@jax.jit` only on complete, pure functions
- Static arguments go in `static_argnames` for JIT
- Test both pre-fit and post-fit scenarios

## 🆘 Getting Help

1. Check `IMPLEMENTATION_SUMMARY.md` for current status
2. Read `TAUREX_MODEL_EXECUTION_FLOW.md` for TauREx internals
3. Run `python check_environment.py` for package issues
4. Check test output for specific failures

---

**Current Status**: ✅ Core components implemented, awaiting opacity interpolation

**Blocker**: Opacity interpolation needs to be implemented before assembly

**Next Step**: Install JAX and run `python test_experiment.py`
