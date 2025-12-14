# JAX TauREx Implementation

This directory contains the JAX-based differentiable implementation of TauREx for atmospheric retrieval.

## Structure

```
jax_taurex/
├── jax_backend.py          # Core JAX differentiable forward model
├── adc_jax_utils.py         # Utilities for ADC 2023 data and model creation
├── experiment_adc_2023.py   # Main script for running ADC retrievals
├── experiment_run.py        # Alternative experiment runner
├── example_*.py             # Example scripts demonstrating usage
├── test_*.py                # Test and debugging scripts
├── debug_*.py               # Gradient debugging utilities
├── diagnose_*.py            # Diagnostic scripts
├── __init__.py              # Package exports
└── notes/                   # Documentation and implementation notes
    ├── ADC_2023_JAX_README.md
    ├── IMPLEMENTATION_SUMMARY.md
    ├── GRADIENT_FIX_SUMMARY.md
    └── ...
```

## Key Modules

### jax_backend.py
Core implementation of the fully differentiable forward model:
- `extract_fitting_params()` - Extract parameters from TauREx model
- `prepare_contributions()` - Setup absorption, CIA, Rayleigh contributions
- `path_integral()` - Compute optical depth with differentiable masking
- `create_forward_model()` - Build complete forward model function
- `fit_with_value_and_grad_adam()` - Adam optimizer implementation
- `fit_with_lbfgsb()` - L-BFGS-B optimizer implementation

### adc_jax_utils.py
Utilities for working with ADC 2023 data:
- `load_planet_spectrum()` - Load spectrum from ADC dataset
- `load_ground_truth()` - Load ground truth parameters
- `create_taurex_model_from_adc_planet()` - Build TauREx model from ADC data
- `fit_adc_planet()` - Run retrieval on ADC planet
- `plot_adc_fit_results()` - Visualize retrieval results

## Usage

### Basic Import
```python
from jax_taurex import jax_backend, adc_jax_utils
```

### Running ADC Experiments
```bash
# Activate the correct virtualenv
pyenv activate taurex-dev-3.12.12

# Run retrieval on ADC planet
python jax_taurex/experiment_adc_2023.py --planet-id 1 --steps 1000 --nlayers 30
```

### Fitting a Single Planet
```python
from jax_taurex.adc_jax_utils import fit_adc_planet

results = fit_adc_planet(
    planet_id=1,
    n_steps=1000,
    nlayers=30,
    optimizer='adam',
    learning_rate=0.01
)
```

## Notes

See the `notes/` directory for detailed documentation on:
- ADC 2023 implementation details
- Gradient computation and debugging
- Precision handling (float32 vs float64)
- Optimizer implementation
- Performance optimization guides
