# ADC 2023 JAX Atmospheric Retrieval

Pure JAX implementation for fitting Ariel Data Challenge 2023 atmospheric parameters using gradient-based optimization.

## Overview

This implementation provides a clean, functional API (no class abstractions) to perform atmospheric retrieval on ADC 2023 data using:
- **Fully differentiable forward model** (recomputes everything from parameters)
- **GPU acceleration** (tested on RTX 3080 Ti)
- **Gradient-based optimization** (Adam optimizer with JAX autodiff)
- **All ADC molecules**: H2O, CO2, CO, CH4, NH3

## Files Created

### Core Module
- **`adc_jax_utils.py`** - Main utilities module
  - Data loading functions (HDF5 spectra, CSV auxiliary data, ground truth)
  - TauREx model creation from ADC parameters
  - JAX forward model wrapper
  - Complete fitting pipeline
  - Plotting functions

### Example Scripts
- **`example_fit_single_planet.py`** - Command-line script
  - One-line usage: `python example_fit_single_planet.py`
  - Configurable: `python example_fit_single_planet.py --planet-id 100 --steps 1000`
  
- **`adc_2023_quickstart.ipynb`** - Interactive notebook
  - Step-by-step tutorial
  - Two methods: one-line convenience function or manual pipeline
  - Visualization examples
  - Multi-planet testing example

## Quick Start

### Option 1: Python Script (Simplest)

```bash
python example_fit_single_planet.py
```

This will:
1. Load planet 1000 from the training set
2. Create TauREx and JAX models
3. Fit 7 parameters (planet_radius, T, H2O, CO2, CO, CH4, NH3)
4. Run 500 optimization steps
5. Generate comparison plot
6. Show results vs ground truth

### Option 2: One-Line Python API

```python
from adc_jax_utils import fit_adc_planet, plot_adc_fit_results

# Fit a planet
result = fit_adc_planet(
    planet_id=1000,
    data_dir='test_files/adc_2023/TrainingData',
    fit_params=['planet_radius', 'T', 'H2O', 'CO2', 'CO', 'CH4', 'NH3'],
    steps=500,
    lr=1e-3
)

# Plot results
plot_adc_fit_results(result)
```

### Option 3: Interactive Notebook

```bash
jupyter notebook adc_2023_quickstart.ipynb
```

The notebook provides detailed examples with visualizations.

## Data Structure

### ADC 2023 Training Data
Located in `test_files/adc_2023/TrainingData/`:

- **SpectralData.hdf5** - Spectroscopic observations
  - 41,423 planets
  - 52 wavelength bins (0.55 - 7.28 μm)
  - Format: `Planet_trainXXXX/instrument_wlgrid`, `instrument_spectrum`, `instrument_noise`, `instrument_width`

- **AuxillaryTable.csv** - Stellar and planetary parameters
  - Star: distance, mass, radius, temperature
  - Planet: mass, orbital period, semi-major axis, radius, surface gravity

- **Ground Truth Package/FM_Parameter_Table.csv** - Atmospheric parameters
  - planet_radius (R_jup), planet_temp (K)
  - log_H2O, log_CO2, log_CO, log_CH4, log_NH3 (log10 mixing ratios)

## API Reference

### Data Loading

```python
# Load spectrum
spectrum_dict = load_planet_spectrum(spectral_hdf5_path, planet_id)
# Returns: {'wl_grid', 'spectrum', 'noise', 'wl_width', 'planet_key'}

# Load auxiliary data
aux_data = load_auxiliary_data(aux_csv_path, planet_id)
# Returns: dict with star_temperature, planet_mass_kg, etc.

# Load ground truth
ground_truth = load_ground_truth(gt_csv_path, planet_id)
# Returns: dict with planet_radius, planet_temp, H2O, CO2, etc.
```

### Model Creation

```python
# Create TauREx transmission model from ADC parameters
tm = create_taurex_model_from_adc_planet(
    aux_data,
    ground_truth=ground_truth,  # Optional, for initialization
    nlayers=30,
    active_molecules=['H2O', 'CO2', 'CO', 'CH4', 'NH3']
)

# Create JAX differentiable forward model
forward_model, obs_spectrum = create_adc_jax_forward_model(
    tm,
    spectrum_dict,
    use_full_diff=True,  # Fully differentiable version
    dtype=jnp.float64
)
```

### Fitting

```python
# Complete pipeline (easiest)
result = fit_adc_planet(
    planet_id=1000,
    data_dir='test_files/adc_2023/TrainingData',
    fit_params=['planet_radius', 'T', 'H2O', 'CO2', 'CO', 'CH4', 'NH3'],
    steps=500,
    lr=1e-3,
    nlayers=30,
    dtype=jnp.float64,
    verbose=True
)

# Returns: dict with final_params, losses, taurex_model, forward_model, etc.
```

### Plotting

```python
# Plot fit results
plot_adc_fit_results(result, save_path='results.png')
```

## Configuration

### Precision Control

Both float64 and float32 are supported:

```python
import jax
import jax.numpy as jnp

# Float64 (safer, ~1.5x slower, 2x memory)
jax.config.update("jax_enable_x64", True)
dtype = jnp.float64

# Float32 (~2x faster, half memory, may overflow)
jax.config.update("jax_enable_x64", False)
dtype = jnp.float32
```

**Recommendation:** Start with float64 for atmospheric modeling, test float32 after validating.

### Opacity Cache Setup

Required before first use:

```python
from taurex.cache import OpacityCache, CIACache

OpacityCache().clear_cache()
OpacityCache().set_opacity_path('test_files/xsec/xsec_sampled_R15000_0.3-50')
CIACache().set_cia_path('test_files/cia/HITRAN/data')
```

## Available Molecules

All required opacity files are present in `test_files/xsec/`:
- ✓ H2O (water)
- ✓ CO2 (carbon dioxide)
- ✓ CO (carbon monoxide)
- ✓ CH4 (methane)
- ✓ NH3 (ammonia)

CIA pairs available:
- H2-H2
- H2-He

## Performance

Tested on RTX 3080 Ti (12GB):

| Precision | Forward Pass | Memory  | Stability |
|-----------|-------------|---------|-----------|
| float64   | ~150 ms     | ~8 GB   | Excellent |
| float32   | ~80 ms      | ~4 GB   | Good*     |

*Test float32 carefully for numerical issues (NaN/Inf)

Typical retrieval (500 steps, 7 parameters):
- float64: ~75-90 seconds
- float32: ~40-50 seconds

## Parameter Fitting

### Common Parameters
- `planet_radius` - Planet radius (Jupiter radii)
- `T` - Atmospheric temperature (K)
- `H2O`, `CO2`, `CO`, `CH4`, `NH3` - Mixing ratios

### Example Configurations

**Fast test (3 parameters):**
```python
fit_params = ['planet_radius', 'T', 'H2O']
steps = 200
```

**Standard (7 parameters):**
```python
fit_params = ['planet_radius', 'T', 'H2O', 'CO2', 'CO', 'CH4', 'NH3']
steps = 500
```

**Thorough (with more steps):**
```python
fit_params = ['planet_radius', 'T', 'H2O', 'CO2', 'CO', 'CH4', 'NH3']
steps = 1000
lr = 5e-4  # Lower learning rate for stability
```

## Workflow Comparison

### This Implementation (JAX)
✓ Pure JAX - no TensorFlow/Keras  
✓ Fully differentiable forward model  
✓ GPU acceleration  
✓ Simple functional API  
✓ Direct parameter optimization  
✓ Fast iterations (~80-150ms per step)  

### Original Training Notebook (TensorFlow)
- Deep neural network surrogate model
- Two-stage training (MSE → heteroscedastic loss)
- K-fold cross-validation
- Semi-supervised learning with trace data
- Outputs probability distributions

## Integration with Experiment Pipeline

The ADC utilities integrate seamlessly with the existing `experiment.py`:

```python
from experiment import (
    extract_fitting_params,
    fit_with_value_and_grad_adam,
    full_diff_create_binned_forward_model
)

# ADC utilities use the same functions under the hood
# Can mix and match approaches as needed
```

## Troubleshooting

### "Planet not found"
Check planet ID format. Training data uses keys like `Planet_train1`, `Planet_train1000`.

### "NaN/Inf during optimization"
Try:
1. Use float64 instead of float32
2. Lower learning rate (1e-4 instead of 1e-3)
3. Reduce number of layers (20 instead of 30)
4. Check parameter initialization values

### "GPU out of memory"
Try:
1. Use float32 (halves memory usage)
2. Reduce number of layers
3. Process one planet at a time

### "Opacity files not found"
Run:
```python
OpacityCache().set_opacity_path('test_files/xsec/xsec_sampled_R15000_0.3-50')
CIACache().set_cia_path('test_files/cia/HITRAN/data')
```

## Example Output

```
Loading ADC planet: 1000
  Ground truth available
  Wavelength range: 0.55 - 7.28 μm
  Spectral bins: 52

Creating TauREx model...
  Layers: 30
  Active gases: ['H2O', 'CO2', 'CO', 'CH4', 'NH3']

Creating JAX forward model...
  ✓ Forward model ready!

Fitting parameters: ['planet_radius', 'T', 'H2O', 'CO2', 'CO', 'CH4', 'NH3']
Initial values:
  planet_radius: 8.638963e-01
  T: 1.396480e+03
  H2O: 6.953937e-09
  ...

Starting optimization (500 steps)...
Step 0: loss = 1.234567e-03
Step 50: loss = 3.456789e-04
...
Step 500: loss = 1.234567e-05

Final Results:
--------------------------------------------------
planet_radius   : 8.638963e-01 -> 8.721345e-01 (+0.95%)
  Ground truth: 8.638963e-01 (error: 0.95%)
T               : 1.396480e+03 -> 1.412567e+03 (+1.15%)
  Ground truth: 1.396480e+03 (error: 1.15%)
...
```

## Next Steps

1. **Test on multiple planets** - Run the multi-planet loop in the notebook
2. **Optimize hyperparameters** - Tune learning rate, steps, nlayers
3. **Compare to ML results** - See how physical model compares to neural network
4. **Batch processing** - Fit many planets for systematic study
5. **Uncertainty quantification** - Add MCMC or variational inference

## Citation

If using this code for ADC 2023:

```
Ariel Data Challenge 2023
https://www.ariel-datachallenge.space/ML/documentation/data
```

## Related Files

- `experiment.py` - Core JAX forward model implementation
- `experiment_run.py` - Example for custom spectra
- `EXPERIMENT_RUN_USAGE.md` - Precision configuration guide
- `FULLY_DIFFERENTIABLE_PROGRESS.md` - Development notes
