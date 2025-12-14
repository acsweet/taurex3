"""
JAX-based atmospheric retrieval implementation for TauREx.

This package provides a fully differentiable forward model implementation
using JAX for efficient gradient-based optimization.
"""

from .jax_backend import (
    extract_fitting_params,
    prepare_model_data,
    prepare_contributions,
    create_forward_model,
    create_binned_forward_model,
    full_diff_create_binned_forward_model,
    compute_path_length_simple,
    contribute_tau,
    contribute_tau_cia,
    path_integral,
)

from .adc_jax_utils import (
    load_planet_spectrum,
    load_auxiliary_data,
    load_ground_truth,
    load_fm_ground_truth,
    create_taurex_model_from_adc_planet,
    create_adc_jax_forward_model,
    fit_adc_planet,
)

__all__ = [
    # Core JAX backend functions
    'extract_fitting_params',
    'prepare_model_data',
    'prepare_contributions',
    'create_forward_model',
    'create_binned_forward_model',
    'full_diff_create_binned_forward_model',
    'compute_path_length_simple',
    'contribute_tau',
    'contribute_tau_cia',
    'path_integral',
    # ADC utilities
    'load_planet_spectrum',
    'load_auxiliary_data',
    'load_ground_truth',
    'load_fm_ground_truth',
    'create_taurex_model_from_adc_planet',
    'create_adc_jax_forward_model',
    'fit_adc_planet',
]
