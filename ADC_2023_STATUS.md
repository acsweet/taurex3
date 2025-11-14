# ADC 2023 Implementation Status

## ✅ Completed

### 1. Data Loading (`adc_jax_utils.py`)
- ✅ `load_planet_spectrum()` - Loads HDF5 spectral data
- ✅ `load_auxiliary_data()` - Loads stellar/planetary parameters from CSV
- ✅ `load_ground_truth()` - Loads atmospheric parameters with linear mixing ratios
- ✅ Resolution handling: Properly computes bin edges from wavelength centers (midpoint method)

### 2. Model Creation
- ✅ `create_taurex_model_from_adc_planet()` - Creates TauREx TransmissionModel from ADC parameters
  - Correctly uses stellar parameters (temperature, radius, mass)
  - Correctly uses planetary parameters (mass, radius from ground truth)
  - Properly initializes chemistry with 5 molecules (H2O, CO2, CO, CH4, NH3)
  - Added NH3 molecular weight to MOLECULAR_WEIGHTS dict

### 3. Forward Model Integration
- ✅ `create_adc_jax_forward_model()` - Wraps TauREx model with JAX differentiable forward model
  - Uses `full_diff_create_binned_forward_model` from `experiment.py`
  - Properly creates ObservedSpectrum with bin edges
  - Resolution: 76,744 native points → 52 binned observations (1476x compression)
  
### 4. Fitting Pipeline
- ✅ `fit_adc_planet()` - Complete end-to-end fitting function
  - Automatically initializes opacity caches
  - Extracts fitting parameters
  - Runs Adam optimization with gradients
  - Returns full result dictionary

### 5. Experiment Framework
- ✅ `experiment_adc_2023.py` - Standalone experiment script
  - Command-line interface with argparse
  - Comparison to ground truth with statistics
  - 3-panel plots (spectrum, residuals, loss)
  - Results saved to file

### 6. Documentation
- ✅ `ADC_2023_JAX_README.md` - User guide
- ✅ `ADC_PARAMETER_USAGE.md` - Parameter flow documentation
- ✅ `example_fit_single_planet.py` - Simple example script
- ✅ `adc_2023_quickstart.ipynb` - Interactive tutorial

## ✅ RESOLVED - Forward Model Fixed (Nov 9, 2025)

### Problem
The JAX forward model produced spectra that were **0.744x observed** (25% too small).

### Root Cause
**Planet radius unit mismatch**: The `planet_radius` parameter is stored in Jupiter radii (R_jup), but the code was treating it as a scaling factor and multiplying by `planet_radius_base` (already in meters):
- Wrong: `0.8639 * 6.176e7 m = 5.335e7 m` 
- Correct: `0.8639 * RJUP = 6.176e7 m`
- Error: `(5.335/6.176)² = 0.746` ← exactly our 0.74x bug!

### Fix Applied
In `experiment.py` (lines 295, 935):
```python
# OLD: planet_radius = params['planet_radius'] * planet_radius_base
# NEW: planet_radius = params['planet_radius'] * jnp.array(RJUP, dtype=params['planet_radius'].dtype)
```

### Test Results After Fix
- **TauREx native**: 0.999x observed ✓
- **JAX forward model**: 1.000x observed ✓
- **experiment_run.py** (quickstart.dat): Converged perfectly, χ² = 20.34 ✓

---

## ⚠️ Previous Investigation (Fixed)

```
Parameter initialization: ✓ ALL CORRECT
  - Planet radius: 0.863896 R_jup ✓
  - Temperature: 945.92 K ✓  
  - H2O: 6.953937e-09 ✓
  - All parameters match ground truth

TauREx Native: ✓ PERFECT
  - Mean: 8.88e-03 (observed: 8.89e-03)
  - Ratio: 0.999x observed ✓

JAX Forward Model: ✗ 25% TOO SMALL
  - Mean: 6.62e-03 (observed: 8.89e-03)
  - Ratio: 0.744x observed ✗
  - TauREx/JAX disagreement: 34.3%
```

### Diagnosis
- ✅ **Model parameters are correct** - TauREx produces perfect spectra
- ✅ **ADC data loading is correct** - All parameters verified
- ✅ **Binning matrix is correct** - 53 edges for 52 bins, monotonic, full coverage
- ✗ **JAX forward model has systematic 0.74x scale error**

### Where the Bug Is
The issue is in `experiment.py`, specifically in:
- `full_diff_create_forward_model()` (lines 834-973)
- `path_integral()` (lines 271-331)  
- `compute_absorption()` (lines 104-127)

Likely causes:
1. Missing factor in absorption calculation
2. Incorrect normalization by star radius squared
3. Path integral not matching TauREx's implementation

### Evidence
ADC Planet 1000 first 5 bins:

| λ (μm) | Observed   | TauREx     | JAX        | T/O   | J/O   |
|--------|------------|------------|------------|-------|-------|
| 0.550  | 8.853e-03  | 8.842e-03  | 6.649e-03  | 0.999 | 0.751 |
| 0.700  | 8.826e-03  | 8.875e-03  | 6.630e-03  | 1.006 | 0.751 |
| 0.950  | 8.807e-03  | 8.864e-03  | 6.632e-03  | 1.006 | 0.753 |

The 0.74-0.75 ratio is consistent across all wavelengths → systematic scale error, not wavelength-dependent.

### Next Steps to Debug

1. **Compare JAX vs TauREx at same parameters**:
   ```python
   # Compute both at same wavelengths
   jax_spec = jax_forward_model(params)
   taurex_spec = tm.model(wngrid)
   # Find where they diverge
   ```

2. **Check intermediate outputs**:
   - Altitude profile
   - Temperature profile
   - Density profile
   - Optical depth (tau)
   - Path integrals

3. **Test with simpler model**:
   - Single gas (H2O only)
   - Isothermal atmosphere
   - Fewer layers (10 instead of 30)

4. **Validate binning separately**:
   - Create fake high-res spectrum
   - Bin it with the matrix
   - Check if integral is conserved

## Working Test Case

To verify data loading and model creation work:

```bash
python -c "
from adc_jax_utils import load_planet_spectrum, load_ground_truth
spec = load_planet_spectrum('test_files/adc_2023/TrainingData/SpectralData.hdf5', 1000)
gt = load_ground_truth('test_files/adc_2023/TrainingData/Ground Truth Package/FM_Parameter_Table.csv', 1000)
print(f'Loaded {len(spec[\"spectrum\"])} spectral bins')
print(f'Planet radius: {gt[\"planet_radius\"]:.3f} R_jup')
print(f'H2O: {gt[\"H2O\"]:.3e}')
"
```

## File Summary

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `adc_jax_utils.py` | 659 | Core utilities | ✅ Complete |
| `experiment_adc_2023.py` | 305 | Experiment runner | ✅ Complete |
| `example_fit_single_planet.py` | ~50 | Simple example | ✅ Complete |
| `adc_2023_quickstart.ipynb` | - | Tutorial notebook | ✅ Complete |
| `ADC_2023_JAX_README.md` | - | User guide | ✅ Complete |
| `ADC_PARAMETER_USAGE.md` | - | Parameter docs | ✅ Complete |

## Resolution Handling: Verified Correct ✓

The implementation correctly handles the resolution mismatch:

1. **ADC Data**: 52 pre-binned observations (instrument resolution)
2. **TauREx Native**: 76,744 high-resolution points (~0.0006 μm spacing)
3. **Forward Model**: Computes at full native resolution
4. **Binning Matrix**: Properly integrates down to 52 bins
5. **Output**: Matches ADC resolution (52 bins)

This is the **correct** approach for atmospheric retrieval - the forward model must be computed at high resolution to capture spectral features, then binned to match the instrument.

## Conclusion

The infrastructure is complete and robust. The remaining issue is a **~26% systematic offset** in the JAX forward model that prevents successful fitting. This requires detailed debugging of the `experiment.py` forward model implementation, which is outside the scope of data pipeline development.

**Recommendation**: Either:
1. Debug the JAX forward model (`experiment.py` lines 834-1045)
2. Use TauREx's native retrieval instead of JAX (traditional MCMC/nested sampling)
3. Verify the JAX model works on the original `quickstart.dat` test case before applying to ADC data
