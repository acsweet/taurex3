"""
JAX-based utilities for Ariel Data Challenge 2023 atmospheric retrieval.

This module provides clean, functional utilities to:
- Load ADC 2023 planet data (spectra, auxiliary info, ground truth)
- Create TauREx transmission models from ADC parameters
- Wrap models with JAX differentiable forward models
- Run gradient-based optimization

**IMPORTANT: This uses your JAX differentiable forward model, NOT TauREx directly!**

The fitting process:
1. TauREx model is created with initial parameters (create_taurex_model_from_adc_planet)
2. JAX forward model wraps it (full_diff_create_binned_forward_model from experiment.py)
3. During optimization, the JAX model recomputes EVERYTHING from parameters:
   - Atmospheric structure (altitude profile, gravity, scale heights)
   - Molecular opacities (interpolated at each T, P)
   - Density profiles from ideal gas law
   - Path integrals through atmosphere
4. Gradients flow through the entire computation for efficient optimization

This is different from traditional retrievals where TauREx computes the spectrum
and only certain parameters are varied. Here, the FULL atmospheric physics is
recomputed in JAX at every step, making it fully differentiable.

No class abstractions - simple functions following the pattern in experiment.py.

Data structure:
- Spectral data: HDF5 with instrument_wlgrid, instrument_spectrum, instrument_noise, instrument_width
- Auxiliary data: CSV with stellar and planetary parameters
- Ground truth: HDF5 (Tracedata.hdf5) with posterior distributions from retrievals

Available molecules in test_files/xsec/:
- H2O, CO2, CO, CH4, NH3 (all needed for ADC 2023)
"""

import h5py
import numpy as np
import pandas as pd
import jax.numpy as jnp

from taurex.cache import OpacityCache, CIACache
from taurex.contributions import AbsorptionContribution, CIAContribution, RayleighContribution
from taurex.model import TransmissionModel
from taurex.planet import Planet
from taurex.stellar import BlackbodyStar
from taurex.chemistry import TaurexChemistry, ConstantGas
from taurex.temperature import Isothermal


# ========== DATA LOADING ==========

def load_planet_spectrum(spectral_hdf5_path, planet_id):
    """
    Load spectrum for a single planet from ADC 2023 HDF5 file.
    
    Args:
        spectral_hdf5_path: Path to SpectralData.hdf5
        planet_id: Planet identifier (e.g., 'train1', 'train100', or just 1, 100)
    
    Returns:
        dict with keys:
            - wl_grid: Wavelength grid (microns)
            - spectrum: Observed spectrum (transit depth, unitless)
            - noise: Uncertainty (same units as spectrum)
            - wl_width: Wavelength bin widths (microns)
            - planet_key: Full key used in HDF5 (e.g., 'Planet_train1')
    """
    with h5py.File(spectral_hdf5_path, 'r') as f:
        # Handle different planet_id formats
        if isinstance(planet_id, str):
            if planet_id.startswith('Planet_'):
                planet_key = planet_id
            elif planet_id.startswith('train'):
                planet_key = f'Planet_{planet_id}'
            else:
                planet_key = f'Planet_train{planet_id}'
        else:
            planet_key = f'Planet_train{planet_id}'
        
        if planet_key not in f:
            raise ValueError(f"Planet {planet_key} not found in {spectral_hdf5_path}")
        
        planet_data = f[planet_key]
        
        return {
            'wl_grid': planet_data['instrument_wlgrid'][:],
            'spectrum': planet_data['instrument_spectrum'][:],
            'noise': planet_data['instrument_noise'][:],
            'wl_width': planet_data['instrument_width'][:],
            'planet_key': planet_key
        }


def load_auxiliary_data(aux_csv_path, planet_id):
    """
    Load auxiliary data (stellar/planetary parameters) for a planet.
    
    ADC 2023 provides the following parameters:
        - star_distance: Distance to system (pc)
        - star_mass_kg: Stellar mass (kg)
        - star_radius_m: Stellar radius (meters)
        - star_temperature: Stellar effective temperature (K)
        - planet_mass_kg: Planetary mass (kg)
        - planet_orbital_period: Orbital period (days)
        - planet_distance: Semi-major axis (AU) 
        - planet_surface_gravity: Surface gravity (m/s²)
    
    Note: planet_radius is NOT in auxiliary data. It comes from:
        1. Ground truth (FM_Parameter_Table.csv) - most accurate
        2. Can be derived from transit depth + star radius (less accurate)
    
    Args:
        aux_csv_path: Path to AuxillaryTable.csv
        planet_id: Planet identifier (e.g., 'train1', 'train100', or just 1, 100)
    
    Returns:
        dict with stellar and planetary parameters (keys match CSV column names)
    """
    aux_df = pd.read_csv(aux_csv_path)
    
    # Handle different planet_id formats
    if isinstance(planet_id, str):
        if planet_id.startswith('train'):
            search_id = planet_id
        else:
            search_id = f'train{planet_id}'
    else:
        search_id = f'train{planet_id}'
    
    planet_row = aux_df[aux_df['planet_ID'] == search_id]
    
    if planet_row.empty:
        raise ValueError(f"Planet {search_id} not found in {aux_csv_path}")
    
    return planet_row.iloc[0].to_dict()


def load_ground_truth(tracedata_hdf5_path, planet_id, sample_method='weighted_random', random_seed=None):
    """
    Load ground truth atmospheric parameters for a planet from Tracedata.hdf5.
    
    The Tracedata.hdf5 contains posterior distributions from retrievals, not just
    point estimates. Each planet has a 'tracedata' array (N_samples x 7 parameters)
    and a 'weights' array (N_samples) representing the posterior probability.
    
    Args:
        tracedata_hdf5_path: Path to Tracedata.hdf5 (in Ground Truth Package/)
        planet_id: Planet identifier (e.g., 'train1', 'train100', or just 1, 100)
        sample_method: How to extract ground truth:
            - 'weighted_random': Random sample weighted by posterior (default)
            - 'weighted_mean': Weighted mean of posterior
            - 'max_weight': Sample with maximum weight (MAP estimate)
        random_seed: Random seed for reproducibility (only for weighted_random)
    
    Returns:
        dict with atmospheric parameters:
            - planet_radius: Jupiter radii
            - planet_temp: K
            - log_H2O: log10 mixing ratio
            - log_CO2: log10 mixing ratio
            - log_CO: log10 mixing ratio
            - log_CH4: log10 mixing ratio
            - log_NH3: log10 mixing ratio
            - H2O, CO2, CO, CH4, NH3: linear mixing ratios
            - sample_method: method used
            - n_samples: number of posterior samples available
    
    Tracedata columns:
        0: planet_radius (R_jup)
        1: planet_temp (K)
        2: log_H2O
        3: log_CO2
        4: log_CO
        5: log_CH4
        6: log_NH3
    """
    import h5py
    
    # Handle different planet_id formats
    if isinstance(planet_id, str):
        if planet_id.startswith('Planet_'):
            planet_key = planet_id
        elif planet_id.startswith('train'):
            planet_key = f'Planet_{planet_id}'
        else:
            planet_key = f'Planet_train{planet_id}'
    else:
        planet_key = f'Planet_train{planet_id}'
    
    with h5py.File(tracedata_hdf5_path, 'r') as f:
        if planet_key not in f:
            raise ValueError(f"Planet {planet_key} not found in {tracedata_hdf5_path}")
        
        planet_data = f[planet_key]
        tracedata = planet_data['tracedata'][:]
        weights = planet_data['weights'][:]
    
    # Normalize weights (should already sum to 1, but ensure it)
    weights = weights / weights.sum()
    
    # Extract ground truth based on method
    if sample_method == 'weighted_random':
        # Random sample weighted by posterior probability
        rng = np.random.RandomState(random_seed)
        idx = rng.choice(len(weights), p=weights)
        sample = tracedata[idx]
    elif sample_method == 'weighted_mean':
        # Weighted mean (posterior expectation)
        sample = np.average(tracedata, axis=0, weights=weights)
    elif sample_method == 'max_weight':
        # Maximum a posteriori (MAP) estimate
        idx = np.argmax(weights)
        sample = tracedata[idx]
    else:
        raise ValueError(f"Unknown sample_method: {sample_method}. Use 'weighted_random', 'weighted_mean', or 'max_weight'")
    
    # Extract parameters from sample
    params = {
        'planet_radius': float(sample[0]),
        'planet_temp': float(sample[1]),
        'log_H2O': float(sample[2]),
        'log_CO2': float(sample[3]),
        'log_CO': float(sample[4]),
        'log_CH4': float(sample[5]),
        'log_NH3': float(sample[6]),
    }
    
    # Also provide linear mixing ratios
    params['H2O'] = 10 ** params['log_H2O']
    params['CO2'] = 10 ** params['log_CO2']
    params['CO'] = 10 ** params['log_CO']
    params['CH4'] = 10 ** params['log_CH4']
    params['NH3'] = 10 ** params['log_NH3']
    
    # Metadata
    params['sample_method'] = sample_method
    params['n_samples'] = len(weights)
    if sample_method == 'weighted_random':
        params['random_seed'] = random_seed
    
    return params


def load_fm_ground_truth(fm_csv_path, planet_id):
    """
    Load ground truth parameters from FM_Parameter_Table.csv.
    These are the exact parameters used to generate the synthetic spectrum.
    
    Args:
        fm_csv_path: Path to FM_Parameter_Table.csv
        planet_id: Planet identifier (e.g., 'train1', 'train100', or just 1, 100)
    
    Returns:
        dict with atmospheric parameters
    """
    # Handle different planet_id formats
    if isinstance(planet_id, str):
        if planet_id.startswith('train'):
            search_id = planet_id
        else:
            search_id = f'train{planet_id}'
    else:
        search_id = f'train{planet_id}'
    
    df = pd.read_csv(fm_csv_path)
    planet_row = df[df['planet_ID'] == search_id]
    
    if planet_row.empty:
        raise ValueError(f"Planet {search_id} not found in {fm_csv_path}")
    
    row = planet_row.iloc[0]
    
    params = {
        'planet_radius': float(row['planet_radius']),
        'planet_temp': float(row['planet_temp']),
        'log_H2O': float(row['log_H2O']),
        'log_CO2': float(row['log_CO2']),
        'log_CO': float(row['log_CO']),
        'log_CH4': float(row['log_CH4']),
        'log_NH3': float(row['log_NH3']),
    }
    
    # Add linear mixing ratios
    params['H2O'] = 10 ** params['log_H2O']
    params['CO2'] = 10 ** params['log_CO2']
    params['CO'] = 10 ** params['log_CO']
    params['CH4'] = 10 ** params['log_CH4']
    params['NH3'] = 10 ** params['log_NH3']
    
    params['source'] = 'FM_Parameter_Table'
    
    return params


# ========== TAUREX MODEL CREATION ==========

def create_taurex_model_from_adc_planet(
    aux_data,
    ground_truth=None,
    nlayers=30,
    atm_min_pressure=1e-0,
    atm_max_pressure=1e6,
    chemistry_fill_gases=['H2', 'He'],
    chemistry_fill_ratio=0.172,
    active_molecules=['H2O', 'CO2', 'CO', 'CH4', 'NH3'],
    use_aux_planet_radius=False,
    add_clouds=False,
    add_lee_mie=False
):
    """
    Create a TauREx TransmissionModel from ADC planet parameters.
    
    This creates the TauREx model which is then wrapped by the JAX differentiable
    forward model. The JAX model (full_diff_create_binned_forward_model) will
    recompute atmospheric structure (altitude, gravity, scale height) from the
    fitted parameters during optimization.
    
    ADC Data Used:
    -------------
    FROM AUX DATA (stellar/planetary system):
        - star_temperature (K) → BlackbodyStar temperature
        - star_radius_m (m) → BlackbodyStar radius (converted to R_sun)
        - star_mass_kg (kg) → BlackbodyStar mass (converted to M_sun)
        - planet_mass_kg (kg) → Planet mass (converted to M_jup)
        - planet_radius_m (m) → Planet radius ONLY if use_aux_planet_radius=True
                                (usually we use ground_truth instead)
    
    FROM GROUND TRUTH (atmospheric parameters, if available):
        - planet_radius (R_jup) → Planet radius for initial model
        - planet_temp (K) → Isothermal temperature profile
        - H2O, CO2, CO, CH4, NH3 → Mixing ratios for chemistry
    
    NOT CURRENTLY USED from aux_data (but available):
        - star_distance (pc) - could be used for Star.distance
        - planet_orbital_period (days) - stored but not used in transmission
        - planet_distance (AU) - semi-major axis, stored but not used in transmission
        - planet_surface_gravity (m/s²) - JAX model recomputes from mass/radius
    
    Args:
        aux_data: Dict from load_auxiliary_data() with stellar/planetary parameters
        ground_truth: Optional dict from load_ground_truth() to initialize atmosphere.
                     If None, uses reasonable defaults.
        nlayers: Number of atmospheric layers
        atm_min_pressure: Minimum pressure (Pa)
        atm_max_pressure: Maximum pressure (Pa)
        chemistry_fill_gases: Fill gases (typically ['H2', 'He'])
        chemistry_fill_ratio: He/H2 ratio
        active_molecules: List of molecules to include in chemistry
        use_aux_planet_radius: If True, use aux_data planet_radius instead of ground_truth.
                              Usually False since ground_truth radius is more accurate.
        add_clouds: If True, add SimpleCloudsContribution (opaque cloud deck).
        add_lee_mie: If True, add LeeMieContribution (Mie scattering).
    
    Returns:
        TransmissionModel configured for the planet
    
    Notes:
        - The JAX forward model (experiment.py) recomputes atmospheric structure
          dynamically from planet_mass, planet_radius, and temperature during fitting
        - Gravity and scale heights are computed on-the-fly in JAX
        - Only the stellar parameters and initial atmospheric state need to be set here
    """
    # Convert units for TauREx (expects Jupiter masses/radii, solar units for star)
    from taurex.constants import MJUP, RJUP, RSOL, MSOL
    from taurex.contributions import SimpleCloudsContribution, LeeMieContribution
    
    # Planetary parameters
    planet_mass_mjup = aux_data['planet_mass_kg'] / MJUP
    
    # Choose planet radius source
    if use_aux_planet_radius:
        # Use auxiliary data radius (less accurate, derived from star/spectrum)
        planet_radius_rjup = aux_data.get('planet_radius_m', 1.0 * RJUP) / RJUP
    elif ground_truth is not None and 'planet_radius' in ground_truth:
        # Prefer ground truth radius (more accurate, from forward model input)
        planet_radius_rjup = ground_truth['planet_radius']
    else:
        # Fallback to auxiliary data if available, else default
        planet_radius_rjup = aux_data.get('planet_radius_m', 1.0 * RJUP) / RJUP
    
    # Stellar parameters  
    star_mass_msol = aux_data['star_mass_kg'] / MSOL
    star_radius_rsol = aux_data['star_radius_m'] / RSOL
    star_temp = aux_data['star_temperature']
    star_distance_pc = aux_data.get('star_distance', 10.0)  # Distance to system (pc)
    
    # Atmospheric temperature
    if ground_truth is not None and 'planet_temp' in ground_truth:
        atm_temp = ground_truth['planet_temp']
    else:
        # Estimate equilibrium temperature or use default
        # T_eq ≈ T_star * sqrt(R_star / (2 * a))
        # For now, use a reasonable default
        atm_temp = 1500.0
    
    # Create planet
    # Note: Planet class also accepts planet_sma (semi-major axis), impact_param,
    # orbital_period, albedo, transit_time - but these aren't used in transmission model
    planet = Planet(
        planet_mass=planet_mass_mjup,
        planet_radius=planet_radius_rjup,
        planet_distance=aux_data.get('planet_distance', 1.0),  # semi-major axis in AU
        orbital_period=aux_data.get('planet_orbital_period', 2.0)  # days
    )
    
    # Create star
    # Note: Star class also accepts distance (pc), magnitudeK, metallicity
    star = BlackbodyStar(
        temperature=star_temp,
        radius=star_radius_rsol,
        mass=star_mass_msol,
        distance=star_distance_pc
    )
    
    # Create chemistry
    chemistry = TaurexChemistry(
        fill_gases=chemistry_fill_gases,
        ratio=chemistry_fill_ratio
    )
    
    # Add active molecules
    for molecule in active_molecules:
        if ground_truth is not None and molecule in ground_truth:
            mix_ratio = ground_truth[molecule]
        else:
            # Default very low mixing ratio
            mix_ratio = 1e-8
        
        chemistry.addGas(ConstantGas(molecule, mix_ratio=mix_ratio))
    
    # Create temperature profile
    temperature = Isothermal(T=atm_temp)
    
    # Create transmission model
    tm = TransmissionModel(
        planet=planet,
        temperature_profile=temperature,
        chemistry=chemistry,
        star=star,
        atm_min_pressure=atm_min_pressure,
        atm_max_pressure=atm_max_pressure,
        nlayers=nlayers
    )
    
    # Add contributions
    tm.add_contribution(AbsorptionContribution())
    tm.add_contribution(CIAContribution(cia_pairs=['H2-H2', 'H2-He']))
    tm.add_contribution(RayleighContribution())
    
    if add_clouds:
        # Add opaque cloud deck
        # Default pressure 1e3 Pa (10 mbar), but will be fitted
        tm.add_contribution(SimpleCloudsContribution(clouds_pressure=1e3))
    
    if add_lee_mie:
        # Add Lee Mie scattering
        # Default parameters:
        # radius = 0.01 um
        # Q = 40
        # mix = 1e-6 (increased from 1e-10 to ensure non-zero gradients)
        tm.add_contribution(LeeMieContribution(
            lee_mie_radius=0.01,
            lee_mie_q=40,
            lee_mie_mix_ratio=1e-6,
            lee_mie_bottomP=1e6,
            lee_mie_topP=1e-2
        ))
    
    # Build model
    tm.build()
    
    return tm


# ========== JAX FORWARD MODEL CREATION ==========

def create_adc_jax_forward_model(
    taurex_model,
    adc_spectrum_dict,
    use_full_diff=True,
    dtype=jnp.float64
):
    """
    Create JAX differentiable forward model for ADC planet.
    
    Args:
        taurex_model: Built TauREx TransmissionModel
        adc_spectrum_dict: Dict from load_planet_spectrum() with wl_grid, spectrum, noise
        use_full_diff: If True, uses full_diff version (recomputes everything from params).
                      If False, uses original version (faster but less flexible).
        dtype: JAX dtype (jnp.float64 or jnp.float32)
    
    Returns:
        forward_model: Callable that takes params dict and returns binned spectrum
        obs_spectrum: ObservedSpectrum object (for compatibility with experiment.py functions)
    """
    from taurex.data.spectrum.array import ArraySpectrum
    from jax_backend import create_binned_forward_model, full_diff_create_binned_forward_model
    import numpy as np
    
    # Create spectrum from ADC data
    # NOTE: ADC 'instrument_width' appears to be unreliable (creates negative wavelengths)
    # We'll compute bin edges from wavelength centers using midpoint method
    wl_grid = adc_spectrum_dict['wl_grid']
    spectrum = adc_spectrum_dict['spectrum']
    noise = adc_spectrum_dict['noise']
    
    # For ArraySpectrum without bin widths, provide only 3 columns:
    # [wavelength (μm), spectrum, error]
    # This will trigger automatic bin edge calculation from centers
    obs_data = np.column_stack([wl_grid, spectrum, noise])
    
    # Create ArraySpectrum (parent of ObservedSpectrum)
    obs_spectrum = ArraySpectrum(obs_data)
    
    # ArraySpectrum.manual_binning() computes edges as midpoints between centers
    # This is the standard approach when bin widths aren't reliably provided
    # Edges are computed as: edge[i] = (center[i-1] + center[i]) / 2
    # with extrapolation for first and last edges
    
    # Create JAX forward model
    if use_full_diff:
        forward_model = full_diff_create_binned_forward_model(
            taurex_model,
            obs_spectrum,
            pre_interpolate_opacity=False  # Set True for CPU, False for GPU
        )
    else:
        forward_model = create_binned_forward_model(
            taurex_model,
            obs_spectrum,
            use_lambda_measure=False,
            debug_coverage=False
        )
    
    return forward_model, obs_spectrum


# ========== FITTING CONVENIENCE FUNCTION ==========

def fit_adc_planet(
    planet_id,
    data_dir='test_files/adc_2023/TrainingData',
    fit_params=['planet_radius', 'T', 'H2O', 'CO2', 'CO', 'CH4', 'NH3'],
    steps=1000,
    lr=1e-3,  # Learning rate (for Adam optimizer)
    nlayers=30,
    dtype=jnp.float64,
    verbose=True,
    l2_reg=0.0,  # L2 regularization (0 = no penalty on parameter changes)
    log_prior=None,  # Log-space prior for mixing ratios (None = no prior)
    clip_norm=1.0,  # Gradient clipping norm (for Adam)
    adc_priors=None,  # ADC competition prior bounds (will override TauREx defaults)
    ground_truth_method='weighted_mean',  # How to sample from posterior: weighted_mean, weighted_random, max_weight
    randomize_start=False,  # If True, randomize initial parameters within prior bounds
    randomize_scale=0.5,  # Fraction of prior range to randomize (0.5 = ±50% of range)
    random_seed=None,  # Random seed for reproducibility
    optimizer='adam',  # Optimizer: 'adam' or 'lbfgs'
    multi_start=1,  # Number of random initializations (for L-BFGS multi-start)
    loss='chi_squared',  # Loss function: 'chi_squared' (recommended) or 'mse'
    add_clouds=False,  # If True, include cloud deck in model and fitting
    add_lee_mie=False  # If True, include Lee Mie scattering in model and fitting
):
    """
    Complete pipeline to load ADC planet and fit atmospheric parameters.
    
    This fits the model to the OBSERVED SPECTRUM. The 'ground_truth' parameters
    are from posterior distributions of previous retrievals (using Nestle + TauREx),
    which we use for validation, not as the optimization target.
    
    Args:
        planet_id: Planet identifier (e.g., 1, 100, 'train1')
        data_dir: Directory containing ADC data files
        fit_params: List of parameter names to fit
        steps: Optimization steps (for Adam; ignored for L-BFGS which uses maxiter)
        lr: Learning rate (for Adam optimizer, default 1e-3)
        nlayers: Number of atmospheric layers
        dtype: JAX dtype (jnp.float64 or jnp.float32)
        verbose: Print progress
        l2_reg: L2 regularization strength (0 = none, >0 = penalize changes from initial)
        log_prior: Log-space prior std for mixing ratios (None = no prior, >0 = constrain)
        clip_norm: Gradient clipping norm (for Adam, prevents huge steps)
        adc_priors: Dict of parameter bounds from ADC competition. If None, uses default:
            {
                'planet_radius': [0.1, 3],
                'T': [0, 7000],
                'H2O': [1e-12, 0.1],
                'CO2': [1e-12, 0.1],
                'CO': [1e-12, 0.1],
                'CH4': [1e-12, 0.1],
                'NH3': [1e-12, 0.1],
                'clouds_pressure': [1e-1, 1e6]
            }
        ground_truth_method: How to sample from posterior distribution:
            - 'weighted_mean': Posterior expectation (most stable, default)
            - 'weighted_random': Random sample weighted by posterior
            - 'max_weight': Maximum a posteriori (MAP) estimate
        randomize_start: If True, start from random parameters instead of posterior mean
        randomize_scale: Fraction of prior range for randomization (e.g., 0.5 = ±50%)
        random_seed: Random seed for reproducibility
        optimizer: Optimizer to use:
            - 'adam': First-order adaptive method (good for exploration)
            - 'lbfgs': L-BFGS-B quasi-Newton method (faster convergence, recommended)
        multi_start: Number of random initializations (only for L-BFGS, 1=single run)
        loss: Loss function:
            - 'chi_squared': Chi-squared (weighted by measurement uncertainties, recommended)
            - 'mse': Mean squared error (unweighted)
        add_clouds: If True, adds 'clouds_pressure' to model and fit_params (if not present)
        add_lee_mie: If True, adds Lee Mie parameters to model and fit_params
    
    Returns:
        dict with:
            - final_params: Fitted parameters
            - losses: Loss history
            - initial_params: Starting parameters
            - param_info: Parameter metadata (bounds, scales)
            - taurex_model: TauREx model
            - forward_model: JAX forward model
            - obs_spectrum: Observed spectrum object
            - spectrum_dict: Raw spectrum data
            - ground_truth: Ground truth parameters from posterior (if available)
            - aux_data: Auxiliary stellar/planetary data
            - all_results: (L-BFGS only) All multi-start results
    """
    import os
    from taurex.cache import OpacityCache, CIACache
    from jax_backend import extract_fitting_params, fit_with_value_and_grad_adam
    
    # Default ADC 2023 competition priors
    if adc_priors is None:
        adc_priors = {
            'planet_radius': [0.1, 3.0],
            'T': [0.0, 7000.0],
            'H2O': [1e-12, 0.1],
            'CO2': [1e-12, 0.1],
            'CO': [1e-12, 0.1],
            'CH4': [1e-12, 0.1],
            'NH3': [1e-12, 0.1],
            'clouds_pressure': [1e-1, 1e6],  # 0.1 Pa to 10 bar
            'lee_mie_radius': [0.001, 1.0], # 1 nm to 1 um
            'lee_mie_q': [1.0, 100.0],
            'lee_mie_mix_ratio': [1e-12, 1e-2], # Increased range
            'lee_mie_bottomP': [1e-1, 1e6],
            'lee_mie_topP': [1e-1, 1e6]
        }
    
    # Add clouds_pressure to fit_params if clouds are enabled
    if add_clouds and 'clouds_pressure' not in fit_params:
        fit_params = list(fit_params) + ['clouds_pressure']
        
    # Add Lee Mie parameters if enabled
    if add_lee_mie:
        lee_mie_params = ['lee_mie_radius', 'lee_mie_q', 'lee_mie_mix_ratio', 'lee_mie_bottomP', 'lee_mie_topP']
        for p in lee_mie_params:
            if p not in fit_params:
                fit_params = list(fit_params) + [p]
    
    # Initialize opacity caches if not already set
    if not OpacityCache()._opacity_path:
        OpacityCache().set_opacity_path('test_files/xsec/xsec_sampled_R15000_0.3-50')
    if not CIACache()._cia_path:
        CIACache().set_cia_path('test_files/cia/HITRAN/data')
    
    # Construct paths
    spectral_path = os.path.join(data_dir, 'SpectralData.hdf5')
    aux_path = os.path.join(data_dir, 'AuxillaryTable.csv')
    tracedata_path = os.path.join(data_dir, 'Ground Truth Package', 'Tracedata.hdf5')
    fm_params_path = os.path.join(data_dir, 'Ground Truth Package', 'FM_Parameter_Table.csv')
    
    if verbose:
        print(f"Loading ADC planet: {planet_id}")
    
    # Load data
    spectrum_dict = load_planet_spectrum(spectral_path, planet_id)
    aux_data = load_auxiliary_data(aux_path, planet_id)
    
    try:
        ground_truth = load_ground_truth(tracedata_path, planet_id, sample_method=ground_truth_method)
        has_gt = True
        if verbose:
            print(f"  Ground truth available (method: {ground_truth_method}, {ground_truth['n_samples']} posterior samples)")
    except:
        ground_truth = None
        has_gt = False
        if verbose:
            print(f"  No ground truth available for this planet")
    
    # Load exact FM parameters if available
    try:
        fm_ground_truth = load_fm_ground_truth(fm_params_path, planet_id)
        has_fm_gt = True
        if verbose:
            print(f"  FM ground truth available")
            
        # Use FM ground truth as the primary ground truth for initialization
        # This ensures we start from the exact parameters used to generate the data
        ground_truth = fm_ground_truth
        has_gt = True
        if verbose:
            print(f"  Using FM ground truth as starting values")
            
    except Exception as e:
        fm_ground_truth = None
        has_fm_gt = False
        if verbose:
            print(f"  No FM ground truth available for this planet: {e}")
    
    if verbose:
        print(f"  Wavelength range: {spectrum_dict['wl_grid'].min():.2f} - {spectrum_dict['wl_grid'].max():.2f} μm")
        print(f"  Spectral bins: {len(spectrum_dict['spectrum'])}")
    
    # Create TauREx model
    if verbose:
        print(f"\nCreating TauREx model...")
    
    tm = create_taurex_model_from_adc_planet(
        aux_data,
        ground_truth=ground_truth,
        nlayers=nlayers,
        add_clouds=add_clouds,
        add_lee_mie=add_lee_mie
    )
    
    if verbose:
        print(f"  Layers: {tm.nLayers}")
        print(f"  Active gases: {list(tm.chemistry.activeGases)}")
    
    # Create JAX forward model
    if verbose:
        print(f"\nCreating JAX forward model...")
    
    forward_model, obs_spectrum = create_adc_jax_forward_model(
        tm,
        spectrum_dict,
        use_full_diff=True,
        dtype=dtype
    )
    
    # Extract parameters
    params, param_info = extract_fitting_params(tm, dtype=dtype)
    
    # Override TauREx default bounds with ADC competition priors
    # This is CRITICAL - TauREx uses narrow bounds like [0.9, 1.1] for planet_radius
    # but ADC competition allows [0.1, 3.0]. If ground truth is outside TauREx bounds,
    # the transform will clamp values and produce zero gradients!
    if verbose:
        print(f"\nOverriding parameter bounds with ADC priors...")
    
    for param_name, bounds in adc_priors.items():
        if param_name in param_info:
            old_bounds = param_info[param_name]['range']
            param_info[param_name]['range'] = bounds
            if verbose:
                print(f"  {param_name}: {old_bounds} -> {bounds}")
    
    # Randomize starting parameters if requested
    if randomize_start:
        if verbose:
            print(f"\nRandomizing initial parameters (scale={randomize_scale}, seed={random_seed})...")
        
        rng = np.random.RandomState(random_seed)
        
        for param_name in fit_params:
            if param_name in param_info and param_name in params:
                bounds = param_info[param_name]['range']
                scale = param_info[param_name]['scale']
                
                if scale == 'log':
                    # For log-scale parameters (mixing ratios), randomize in log space
                    log_min, log_max = np.log10(bounds[0]), np.log10(bounds[1])
                    log_center = (log_min + log_max) / 2
                    log_range = (log_max - log_min) * randomize_scale
                    log_val = log_center + rng.uniform(-log_range/2, log_range/2)
                    params[param_name] = jnp.array(10**log_val, dtype=dtype)
                else:
                    # For linear parameters (radius, temp), randomize in linear space
                    center = (bounds[0] + bounds[1]) / 2
                    param_range = (bounds[1] - bounds[0]) * randomize_scale
                    val = center + rng.uniform(-param_range/2, param_range/2)
                    params[param_name] = jnp.array(val, dtype=dtype)
                
                if verbose:
                    print(f"  {param_name}: {float(params[param_name]):.6e}")
    
    # Setup observation data with correct dtype
    # we reverse these here because the spectrum is in the order of wavelengths
    # and taurex model expects in the oder of wavenumbers
    obs_y = jnp.asarray(spectrum_dict['spectrum'][::-1], dtype=dtype)
    obs_err = jnp.asarray(spectrum_dict['noise'][::-1], dtype=dtype)
    
    if verbose:
        print(f"\nFitting parameters: {fit_params}")
        if not randomize_start:
            print("Initial values (from posterior):")
        else:
            print("Initial values (randomized):")
        for k in fit_params:
            print(f"  {k}: {float(params[k]):.6e}")
        if has_gt and ground_truth is not None:
            print("\nPosterior reference (for comparison):")
            for k in fit_params:
                if k in ground_truth:
                    print(f"  {k}: {float(ground_truth[k]):.6e}")
        print(f"\nOptimizing to fit OBSERVED SPECTRUM...")
        print(f"Optimizer: {optimizer.upper()}")
        print(f"Loss function: {loss}")
        if optimizer == 'adam':
            print(f"Steps: {steps}, lr={lr}, clip_norm={clip_norm}, l2_reg={l2_reg}, log_prior={log_prior}")
        else:
            print(f"Max iterations: {steps}, multi_start={multi_start}")
    
    # Run optimization
    from jax_backend import fit_with_value_and_grad_adam, fit_with_lbfgsb
    
    if optimizer == 'adam':
        # Adam optimizer (first-order adaptive method)
        loss_kwargs = {}
        if l2_reg > 0.0:
            loss_kwargs['l2_reg'] = l2_reg
        if log_prior is not None:
            loss_kwargs['log_prior'] = log_prior
        
        final_params, losses = fit_with_value_and_grad_adam(
            forward_binned=forward_model,
            observed_y=obs_y,
            observed_err=obs_err,
            init_params=params,
            param_info=param_info,
            fit_params=fit_params,
            steps=steps,
            lr=lr,
            clip_norm=clip_norm,
            print_every=50 if verbose else steps + 1,
            nan_guard=True,
            loss=loss,
            loss_kwargs=loss_kwargs
        )
        all_results = None
        
    elif optimizer == 'lbfgs':
        # L-BFGS-B optimizer (quasi-Newton with box constraints)
        final_params, final_loss, all_results, losses = fit_with_lbfgsb(
            forward_binned=forward_model,
            observed_y=obs_y,
            observed_err=obs_err,
            init_params=params,
            param_info=param_info,
            fit_params=fit_params,
            maxiter=steps,
            print_every=100 if verbose else 0,
            nan_guard=True,
            loss=loss,
            multi_start=multi_start,
            random_seed=random_seed
        )
        
    else:
        raise ValueError(f"Unknown optimizer '{optimizer}'. Choose 'adam' or 'lbfgs'.")
    
    if verbose:
        print("\nFinal Results:")
        print("-" * 50)
        for k in fit_params:
            initial = float(params[k])
            final = float(final_params[k])
            change = final - initial
            pct_change = (change / initial) * 100 if initial != 0 else 0
            print(f"{k:15s}: {initial:.6e} -> {final:.6e} ({pct_change:+.2f}%)")
            
            if has_gt and ground_truth is not None and k in ground_truth:
                gt_val = float(ground_truth[k])
                error = abs(final - gt_val) / gt_val * 100 if gt_val != 0 else abs(final - gt_val)
                print(f"{'':15s}  Ground truth: {gt_val:.6e} (error: {error:.2f}%)")
    
    return {
        'final_params': final_params,
        'losses': losses,
        'initial_params': params,
        'param_info': param_info,
        'taurex_model': tm,
        'forward_model': forward_model,
        'obs_spectrum': obs_spectrum,
        'spectrum_dict': spectrum_dict,
        'ground_truth': ground_truth if has_gt else None,
        'aux_data': aux_data,
        'all_results': all_results  # L-BFGS multi-start results
    }


# ========== PLOTTING ==========

def plot_adc_fit_results(result_dict, save_path=None):
    """
    Plot ADC fitting results.
    
    Args:
        result_dict: Output from fit_adc_planet()
        save_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    
    spectrum_dict = result_dict['spectrum_dict']
    forward_model = result_dict['forward_model']
    initial_params = result_dict['initial_params']
    final_params = result_dict['final_params']
    ground_truth = result_dict['ground_truth']
    losses = result_dict['losses']
    
    # Get spectra
    obs_y = spectrum_dict['spectrum']
    obs_err = spectrum_dict['noise']
    wl_grid = spectrum_dict['wl_grid']
    
    initial_spectrum = np.array(forward_model(initial_params))
    final_spectrum = np.array(forward_model(final_params))
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot spectra
    ax = axes[0]
    ax.errorbar(wl_grid, obs_y, yerr=obs_err, fmt='o', alpha=0.5, 
                label='Observed', markersize=4)
    ax.plot(wl_grid, initial_spectrum, '--', label='Initial model', linewidth=2)
    ax.plot(wl_grid, final_spectrum, '-', label='Fitted model', linewidth=2)
    
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('Transit Depth')
    ax.set_title('ADC Planet Atmospheric Retrieval')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot residuals
    ax = axes[1]
    initial_residuals = (obs_y - initial_spectrum) / obs_err
    final_residuals = (obs_y - final_spectrum) / obs_err
    
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.scatter(wl_grid, initial_residuals, alpha=0.5, label='Initial residuals')
    ax.scatter(wl_grid, final_residuals, alpha=0.5, label='Final residuals')
    
    ax.set_xlabel('Wavelength (μm)')
    ax.set_ylabel('Residuals (σ)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()
    
    # Also print comparison table if ground truth available
    if ground_truth is not None:
        print("\nParameter Comparison:")
        print("-" * 70)
        print(f"{'Parameter':<15} {'Initial':<15} {'Fitted':<15} {'Ground Truth':<15} {'Error %':<10}")
        print("-" * 70)
        
        fit_params = [k for k in final_params.keys() 
                     if k in ground_truth and initial_params[k] != final_params[k]]
        
        for param in fit_params:
            initial = float(initial_params[param])
            final = float(final_params[param])
            gt = float(ground_truth[param])
            error = abs(final - gt) / abs(gt) * 100 if gt != 0 else abs(final - gt)
            
            print(f"{param:<15} {initial:<15.6e} {final:<15.6e} {gt:<15.6e} {error:<10.2f}")
