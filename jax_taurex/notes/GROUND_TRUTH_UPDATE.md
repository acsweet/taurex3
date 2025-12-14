# Ground Truth Data Source Update

## Summary

Updated the ground truth data source from **FM_Parameter_Table.csv** to **Tracedata.hdf5** for more accurate atmospheric retrieval validation.

## What Changed

### Previous Approach (Incorrect)
- Used `FM_Parameter_Table.csv` as ground truth
- This file contains the **forward model input parameters** used to generate synthetic spectra
- These are the initial values, not the "true" recovered parameters

### New Approach (Correct)
- Uses `Tracedata.hdf5` as ground truth
- This file contains **posterior distributions** from retrievals
- Each planet has ~4000 samples with weights representing posterior probability
- Provides multiple sampling methods:
  - `weighted_random`: Random sample from posterior (default)
  - `weighted_mean`: Posterior expectation
  - `max_weight`: Maximum a posteriori (MAP) estimate

## Updated Functions

### `adc_jax_utils.py`

**`load_ground_truth(tracedata_hdf5_path, planet_id, sample_method='weighted_random', random_seed=None)`**
- Now loads from Tracedata.hdf5 instead of FM_Parameter_Table.csv
- Supports three sampling methods
- Returns posterior samples count and metadata

**`fit_adc_planet()`**
- Updated to use `Tracedata.hdf5` path
- Shows number of posterior samples in verbose output

## Example Usage

```python
from adc_jax_utils import load_ground_truth

# Weighted random sample (default, recommended)
gt = load_ground_truth(
    'test_files/adc_2023/TrainingData/Ground Truth Package/Tracedata.hdf5',
    planet_id=1000,
    sample_method='weighted_random',
    random_seed=42  # For reproducibility
)

# Weighted mean (posterior expectation)
gt_mean = load_ground_truth(..., sample_method='weighted_mean')

# MAP estimate (maximum posterior)
gt_map = load_ground_truth(..., sample_method='max_weight')
```

## Key Differences: FM Table vs Tracedata

For Planet 1000:

| Parameter | FM Table (Input) | Tracedata Mean (Posterior) | Difference |
|-----------|------------------|----------------------------|------------|
| planet_radius | 0.863896 R_jup | 0.863878 R_jup | ~0.002% |
| planet_temp | 945.92 K | 952.22 K | ~0.7% |
| log_H2O | -8.157769 | -9.202118 | ~1 order magnitude |
| log_CO2 | -8.555389 | -9.648866 | ~1 order magnitude |
| log_CO | -4.282107 | -4.358342 | ~0.08 |
| log_CH4 | -5.736519 | -5.761767 | ~0.03 |
| log_NH3 | -6.001003 | -6.048140 | ~0.05 |

**Notable:** Molecular abundances show significant differences, especially for H2O and CO2. The FM table represents the input, but the posterior (tracedata) represents what was actually recovered from the synthetic observations.

## Impact on Retrieval Validation

Using Tracedata instead of FM table provides:
1. **More realistic validation**: Compare against recovered parameters, not inputs
2. **Uncertainty quantification**: Access to full posterior distribution
3. **Better error metrics**: Know the actual uncertainty in each parameter

## Files Modified

- `adc_jax_utils.py`: Updated `load_ground_truth()` and `fit_adc_planet()`
- Scripts using these functions automatically use the new ground truth source:
  - `example_fit_single_planet.py`
  - `experiment_adc_2023.py`

## Backward Compatibility

If you need the old FM parameter table values for any reason, you can still access them directly:

```python
import pandas as pd
fm_df = pd.read_csv('test_files/adc_2023/TrainingData/Ground Truth Package/FM_Parameter_Table.csv')
```

---

**Date**: November 14, 2025
**Reason**: Correct ground truth should be the retrieved posterior, not the forward model input
