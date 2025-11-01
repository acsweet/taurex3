import time
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optax
from functools import partial
from typing import Dict, Tuple, Any

from taurex.constants import KBOLTZ
from taurex.cache import OpacityCache, CIACache
from taurex.contributions import AbsorptionContribution, CIAContribution, RayleighContribution
from taurex.data.spectrum.observed import ObservedSpectrum
from taurex.model import TransmissionModel
from taurex.util import clip_native_to_wngrid

from taurex.planet import Planet
from taurex.stellar import BlackbodyStar
from taurex.chemistry import TaurexChemistry, ConstantGas
from taurex.temperature import Isothermal

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

# ========== PARAMETER EXTRACTION ==========
def extract_fitting_params(taurex_model) -> Tuple[Dict[str, jnp.ndarray], Dict[str, Any]]:
    """Extract fitting parameters from taurex model."""
    fitting_params = {}
    param_info = {}
    
    for key, val in taurex_model._fitting_parameters.items():
        get_val = val[2]()
        val_range = val[-1]
        fit_scale = val[4]
        fitting_params[key] = float(get_val)
        param_info[key] = {
            'range': val_range,
            'scale': fit_scale,
            'latex': val[1]
        }
    
    # Convert to JAX arrays
    params = {k: jnp.array(v) for k, v in fitting_params.items()}
    return params, param_info

# ========== JAX FORWARD MODEL ==========
def compute_path_length_simple(total_layers, altitude_profile, deltaz, planet_radius):
    """Simplified path length computation matching taurex's old method."""
    path_lengths = []
    
    for layer in range(total_layers):
        p = (planet_radius + deltaz[0] / 2 + altitude_profile[layer]) ** 2
        k = jnp.zeros(total_layers - layer)
        
        # First element
        k = k.at[0].set(jnp.sqrt(jnp.maximum(
            (planet_radius + deltaz[0] / 2.0 + altitude_profile[layer] + deltaz[layer] / 2.0) ** 2 - p,
            1e-10
        )))
        
        # Remaining elements if they exist
        if total_layers - layer > 1:
            upper_indices = jnp.arange(layer + 1, total_layers)
            lower_indices = jnp.arange(layer, total_layers - 1)
            
            upper_terms = jnp.sqrt(jnp.maximum(
                (planet_radius + deltaz[0] / 2 + altitude_profile[upper_indices] + deltaz[upper_indices] / 2) ** 2 - p,
                1e-10
            ))
            lower_terms = jnp.sqrt(jnp.maximum(
                (planet_radius + deltaz[0] / 2 + altitude_profile[lower_indices] + deltaz[lower_indices] / 2) ** 2 - p,
                1e-10
            ))
            k = k.at[1:].set(upper_terms - lower_terms)
        
        path_lengths.append(k * 2.0)
    
    return path_lengths

def contribute_tau(startk, endk, density_offset, sigma, density, path, layer, tau):
    """Generic cross-section integration function for tau."""
    _path = path[startk:endk, None]
    _density = density[startk + density_offset:endk + density_offset, None]
    _sigma = sigma[startk + layer:endk + layer, :]
    
    contribution = jnp.sum(_sigma * _path * _density, axis=0)
    return tau.at[layer, :].add(contribution)

def compute_absorption(tau, dz, altitude_profile, planet_radius, star_radius):
    """Compute final absorption and optical depth."""
    # jax.debug.print(f'(jtm - compute_absorption) tau: {jnp.min(tau)} - {jnp.max(tau)}')
    tau_exp = jnp.exp(-tau)
    ap = altitude_profile[:, None]
    _dz = dz[:, None]

    # jax.debug.print(f'(jtm - compute_absorption) planet_radius: {planet_radius}')
    # jax.debug.print(f'(jtm - compute_absorption) star_radius: {star_radius}')
    # jax.debug.print(f'(jtm - compute_absorption) tau_exp: {jnp.min(tau_exp)} - {jnp.max(tau_exp)}')
    # jax.debug.print(f'(jtm - compute_absorption) altitude_profile: {jnp.min(ap)} - {jnp.max(ap)}')
    # jax.debug.print(f'(jtm - compute_absorption) _dz: {jnp.min(_dz)} - {jnp.max(_dz)}')
    
    integral = jnp.sum((planet_radius + ap) * (1.0 - tau_exp) * _dz * 2.0, axis=0)
    absorption = ((planet_radius**2.0) + integral) / (star_radius**2)
    
    # jax.debug.print(f'(jtm - compute_absorption) integral: {jnp.min(integral)} - {jnp.max(integral)}')
    # jax.debug.print(f'(jtm - compute_absorption) absorption: {jnp.min(absorption)} - {jnp.max(absorption)}')

    return absorption, tau_exp

def prepare_model_data(taurex_model: TransmissionModel):
    """Prepare static model data for JAX computation."""
    if not taurex_model.built:
        taurex_model.build()
    
    taurex_model.initialize_profiles()
    
    model_data = {
        'deltaz': jnp.array(taurex_model.deltaz),
        'total_layers': int(taurex_model.nLayers),
        'altitude_profile': jnp.array(taurex_model.altitudeProfile),
        'pressure_profile': jnp.array(taurex_model.pressureProfile),
        'planet_radius_base': float(taurex_model._planet.fullRadius),
        'star_radius': float(taurex_model._star.radius),
    }
    
    return model_data

# Add this dictionary near the top of your file
MOLECULAR_WEIGHTS = {
    'H2': 2.016, 'He': 4.0026, 'H2O': 18.015, 'N2': 28.014,
    'CH4': 16.04, 'CO': 28.01, 'CO2': 44.01
    # Add other molecules as needed
}

# ========== NEW FUNCTION: Add this after your existing contribute_tau ==========
def contribute_tau_cia(startk, endk, density_offset, sigma, density, path, layer, tau):
    """CIA integration function with ρ² dependence."""
    _path = path[startk:endk, None]
    _density = density[startk + density_offset:endk + density_offset, None]
    _sigma = sigma[startk + layer:endk + layer, :]
    
    # CIA requires density squared (collision-induced absorption)
    contribution = jnp.sum(_sigma * _path * _density * _density, axis=0)
    return tau.at[layer, :].add(contribution)


# ========== REPLACEMENT: prepare_contributions ==========
def prepare_contributions(taurex_model, wngrid):
    """
    Revised to separate molecular absorption, CIA, and other contributions.
    Returns molecular opacities dict, CIA sigma array, and other opacities array.
    """
    wngrid_np = np.array(wngrid)
    taurex_model._star.initialize(wngrid_np)

    absorption_sigmas = {}
    cia_sigmas_list = []
    other_sigmas_list = []
    opacity_cache = OpacityCache()

    for contrib in taurex_model.contribution_list:
        if isinstance(contrib, AbsorptionContribution):
            temp_profile = taurex_model.temperatureProfile
            press_profile = taurex_model.pressureProfile
            n_layers = taurex_model.nLayers
            
            for gas in taurex_model.chemistry.activeGases:
                gas_sigma_xsec = np.zeros(shape=(n_layers, len(wngrid_np)))
                xsec = opacity_cache[gas]
                for i in range(n_layers):
                    temperature = temp_profile[i]
                    pressure = press_profile[i]
                    gas_sigma_xsec[i] = xsec.opacity(temperature, pressure, wngrid_np)
                absorption_sigmas[gas] = jnp.array(gas_sigma_xsec)
        
        elif isinstance(contrib, CIAContribution):
            # CIA needs separate handling due to ρ² dependence
            contrib.prepare(taurex_model, wngrid_np)
            cia_sigmas_list.append(jnp.array(contrib.sigma_xsec))
        
        else:
            # Rayleigh and other contributions
            contrib.prepare(taurex_model, wngrid_np)
            other_sigmas_list.append(jnp.array(contrib.sigma_xsec))
    
    # Sum up contributions of each type
    cia_sigma = jnp.sum(jnp.array(cia_sigmas_list), axis=0) if cia_sigmas_list else None
    other_sigma = jnp.sum(jnp.array(other_sigmas_list), axis=0) if other_sigmas_list else None
            
    return absorption_sigmas, cia_sigma, other_sigma


# ========== REPLACEMENT: create_forward_model ==========
def create_forward_model(taurex_model: TransmissionModel, wngrid):
    """
    Create forward model with proper handling of CIA contributions.
    """
    # 1. Prepare all data
    model_data = prepare_model_data(taurex_model)
    absorption_sigmas, cia_sigma, other_sigma = prepare_contributions(taurex_model, wngrid)
    
    # 2. Get the active gases and their molecular weights
    active_gases = tuple(absorption_sigmas.keys())
    active_gas_weights = tuple(MOLECULAR_WEIGHTS[g] for g in active_gases)
    
    # 3. Get fill gas info
    fill_gases = tuple(taurex_model.chemistry._fill_gases)
    fill_gas_weights = tuple(MOLECULAR_WEIGHTS[g] for g in fill_gases)
    He_H2_ratio = float(taurex_model.chemistry._fill_ratio[0])
    
    # 4. Package molecular cross-sections
    abs_sigmas_tuple = tuple(absorption_sigmas[g] for g in active_gases)
    
    # 5. Package static model data (no arrays!)
    static_data = (
        model_data['total_layers'],
        model_data['planet_radius_base'], 
        model_data['star_radius'],
        active_gases,
        active_gas_weights,
        fill_gases, 
        fill_gas_weights,
        He_H2_ratio
    )
    
    # 6. Package dynamic arrays - NOW INCLUDING CIA SEPARATELY
    dynamic_data = (
        model_data['deltaz'], 
        model_data['altitude_profile'],
        model_data['pressure_profile'], 
        cia_sigma,
        other_sigma
    )
    
    def forward_model(params):
        return path_integral(
            jnp.array(wngrid),
            params,
            abs_sigmas_tuple,
            dynamic_data,
            static_data
        )
    
    return forward_model


# ========== REPLACEMENT: path_integral ==========
@partial(jax.jit, static_argnames=['static_data'])
def path_integral(wngrid, params, absorption_sigmas_tuple, dynamic_data, static_data):
    """
    Path integral with proper CIA handling (ρ² dependence).
    """
    # 1. Unpack static data
    (
        total_layers, planet_radius_base, star_radius,
        active_gases, active_gas_weights, fill_gases, fill_gas_weights, He_H2_ratio
    ) = static_data
    
    # 2. Unpack dynamic data - NOW WITH CIA SEPARATE
    dz, altitude_profile, pressure_profile, cia_sigma, other_sigma = dynamic_data
    
    # 3. Get variable parameters
    T = params['T']
    planet_radius = params['planet_radius'] * planet_radius_base
    
    # 4. Calculate current mean molecular weight
    mix_ratios = jnp.array([params.get(g, 0.0) for g in active_gases])
    active_gas_weights_array = jnp.array(active_gas_weights)
    fill_gas_weights_array = jnp.array(fill_gas_weights)
    
    trace_gas_mix = jnp.sum(mix_ratios)
    mu_from_trace = jnp.sum(mix_ratios * active_gas_weights_array)
    
    remainder = 1.0 - trace_gas_mix
    main_fill_mix = remainder / (1.0 + He_H2_ratio)
    second_fill_mix = main_fill_mix * He_H2_ratio
    
    fill_mixes = jnp.array([main_fill_mix, second_fill_mix])
    mu_from_fill = jnp.sum(fill_mixes * fill_gas_weights_array)
    
    mu_current = mu_from_trace + mu_from_fill
    
    # 5. Calculate density
    density_profile = pressure_profile / (KBOLTZ * T)
    
    # 6. Build total cross-section for regular contributions (ρ)
    absorption_sigmas_stacked = jnp.stack(absorption_sigmas_tuple)
    molecular_contribution = jnp.sum(mix_ratios[:, None, None] * absorption_sigmas_stacked, axis=0)
    
    # Regular sigma includes molecular absorption + Rayleigh + other (but NOT CIA)
    regular_sigma = molecular_contribution
    if other_sigma is not None:
        regular_sigma = regular_sigma + other_sigma
    
    # 7. Compute path lengths
    path_lengths = compute_path_length_simple(
        total_layers, altitude_profile, dz, planet_radius
    )
    
    # 8. Integrate optical depth
    tau = jnp.zeros((total_layers, wngrid.shape[0]))
    
    for layer in range(total_layers):
        dl = path_lengths[layer]
        end_k = total_layers - layer
        
        # Regular contributions with ρ dependence
        tau = contribute_tau(
            0, end_k, layer, regular_sigma, density_profile, dl, layer, tau
        )
        
        # CIA contributions with ρ² dependence
        if cia_sigma is not None:
            tau = contribute_tau_cia(
                0, end_k, layer, cia_sigma, density_profile, dl, layer, tau
            )
    
    # 9. Compute final absorption
    absorption, tau_exp = compute_absorption(
        tau, dz, altitude_profile, planet_radius, star_radius
    )
    
    return absorption, tau_exp

# ========== BINNING ==========
def create_flux_weighted_binning_matrix(high_res_wngrid, bin_edges):
    """Create binning matrix for differentiable spectral binning."""
    high_res_wngrid = np.array(high_res_wngrid)
    bin_edges = np.array(bin_edges)
    
    n_bins = len(bin_edges) - 1
    n_high_res = len(high_res_wngrid)
    
    # Estimate high-res bin widths
    high_res_edges = np.zeros(n_high_res + 1)
    high_res_edges[1:-1] = 0.5 * (high_res_wngrid[:-1] + high_res_wngrid[1:])
    high_res_edges[0] = high_res_wngrid[0] + (high_res_wngrid[0] - high_res_edges[1])
    high_res_edges[-1] = high_res_wngrid[-1] + (high_res_wngrid[-1] - high_res_edges[-2])
    
    binning_matrix = np.zeros((n_bins, n_high_res))
    
    for i in range(n_bins):
        bin_start = bin_edges[i+1]  # Wavenumber decreases
        bin_end = bin_edges[i]
        bin_width = bin_end - bin_start
        
        for j in range(n_high_res):
            hr_start = high_res_edges[j+1]
            hr_end = high_res_edges[j]
            
            # Calculate overlap
            overlap_start = max(bin_start, hr_start)
            overlap_end = min(bin_end, hr_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > 0:
                weight = overlap / bin_width
                binning_matrix[i, j] = weight
    
    # Normalize rows
    row_sums = np.sum(binning_matrix, axis=1)
    for i in range(n_bins):
        if row_sums[i] > 0:
            binning_matrix[i, :] /= row_sums[i]
        else:
            # Fallback: nearest neighbor
            bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
            nearest_idx = np.argmin(np.abs(high_res_wngrid - bin_center))
            binning_matrix[i, nearest_idx] = 1.0
    
    return jnp.array(binning_matrix)

@jax.jit
def bin_spectrum(high_res_spectrum, binning_matrix):
    """Apply differentiable binning to a high-resolution spectrum."""
    return jnp.dot(binning_matrix, high_res_spectrum)


# ========================================================================
# FULLY DIFFERENTIABLE MODEL - Recomputes everything from parameters
# ========================================================================

def full_diff_compute_temperature_profile(T, nlayers):
    """
    Compute temperature profile from parameter.
    For isothermal: just T repeated nlayers times.
    """
    return jnp.ones(nlayers) * T


def full_diff_compute_mixing_profiles(params, active_gases, nlayers):
    """
    Compute mixing ratio profiles for all active gases.
    For constant gas profiles, this is just the parameter value repeated.
    
    Returns: dict of {gas_name: mixing_ratio_profile}
    """
    mixing_profiles = {}
    for gas in active_gases:
        mix_ratio = params.get(gas, 0.0)
        mixing_profiles[gas] = jnp.ones(nlayers) * mix_ratio
    return mixing_profiles


def full_diff_compute_mu_profile(mixing_profiles, active_gases, active_gas_weights, 
                                   fill_gases, fill_gas_weights, He_H2_ratio):
    """
    Compute mean molecular weight profile from mixing ratios.
    
    μ = Σ(mix_ratio_i × molecular_weight_i) for all gases
    
    Returns: mean molecular weight in kg (not atomic mass units!)
    """
    from taurex.constants import AMU
    
    # Handle case when there are no active gases
    if not mixing_profiles or len(active_gases) == 0:
        # Return scalar mu from fill gases only
        # For constant profiles, mu is the same at all layers
        main_fill_weight = fill_gas_weights[0]
        second_fill_weight = fill_gas_weights[1]
        mu_u = (main_fill_weight + He_H2_ratio * second_fill_weight) / (1.0 + He_H2_ratio)
        # Convert from atomic mass units to kg
        return mu_u * AMU
    
    nlayers = next(iter(mixing_profiles.values())).shape[0]
    
    # Stack active gas mixing profiles and weights for vectorized computation
    active_mix_array = jnp.stack([mixing_profiles[gas] for gas in active_gases])  # (n_gases, nlayers)
    active_weight_array = jnp.array(active_gas_weights)[:, None]  # (n_gases, 1)
    
    # Contribution from active gases (vectorized)
    mu_from_active = jnp.sum(active_mix_array * active_weight_array, axis=0)  # (nlayers,)
    
    # Total active gas mixing ratio
    total_active = jnp.sum(active_mix_array, axis=0)  # (nlayers,)
    remainder = 1.0 - total_active
    
    # Split remainder between H2 and He
    main_fill_mix = remainder / (1.0 + He_H2_ratio)
    second_fill_mix = main_fill_mix * He_H2_ratio
    
    # Contribution from fill gases
    fill_mixes = jnp.array([main_fill_mix, second_fill_mix])  # (2, nlayers)
    fill_weights_array = jnp.array(fill_gas_weights)[:, None]  # (2, 1)
    
    mu_from_fill = jnp.sum(fill_mixes * fill_weights_array, axis=0)  # (nlayers,)
    
    mu_u = mu_from_active + mu_from_fill
    
    # Convert from atomic mass units to kg
    return mu_u * AMU


def full_diff_compute_gravity_at_height(planet_mass, planet_radius, height):
    """
    Compute gravity at a given height above the surface.
    g(h) = G * M / (R + h)²
    """
    from taurex.constants import G
    return (G * planet_mass) / ((planet_radius + height) ** 2)


def full_diff_compute_scale_height(temperature_profile, mu_profile, gravity_profile):
    """
    Compute atmospheric scale height.
    H = k_B * T / (μ * g)
    """
    return (KBOLTZ * temperature_profile) / (mu_profile * gravity_profile)


def full_diff_compute_altitude_profile(temperature_profile, pressure_levels, 
                                         mu_profile, planet_mass, planet_radius):
    """
    Compute altitude, gravity, and scale height profiles from hydrostatic equilibrium.
    
    This is the JAX-differentiable version of Planet.calculate_scale_properties()
    Uses lax.scan for efficient, JIT-compilable iteration.
    
    Args:
        mu_profile: Can be scalar (for constant mu) or array (nlayers,)
    
    Returns: (altitude_boundaries, scaleheight, gravity, deltaz)
    """
    from jax import lax
    from taurex.constants import G, KBOLTZ
    
    nlayers = temperature_profile.shape[0]
    
    # Handle scalar mu_profile (broadcast to array if needed)
    mu_array = jnp.ones(nlayers) * mu_profile if jnp.ndim(mu_profile) == 0 else mu_profile
    
    # Surface gravity and scale height (bottom layer, index 0)
    g_surface = (G * planet_mass) / (planet_radius ** 2)
    H_surface = (KBOLTZ * temperature_profile[0]) / (mu_array[0] * g_surface)
    
    def altitude_step(carry, i):
        """
        One step of hydrostatic integration.
        
        carry: (z_current, g_current, H_current)
        i: layer index
        returns: (new_carry, outputs_for_this_layer)
        """
        z_current, g_current, H_current = carry
        
        # Integrate altitude: dz = -H * ln(P_next / P_i)
        dz_i = -H_current * jnp.log(pressure_levels[i + 1] / pressure_levels[i])
        z_next = z_current + dz_i
        
        # Compute gravity at new altitude
        g_next = (G * planet_mass) / ((planet_radius + z_next) ** 2)
        
        # Compute scale height at new altitude (using T and mu at i+1)
        # For last layer, we use current values to avoid out-of-bounds
        T_next = jnp.where(i < nlayers - 1, temperature_profile[i + 1], temperature_profile[i])
        mu_next = jnp.where(i < nlayers - 1, mu_array[i + 1], mu_array[i])
        H_next = (KBOLTZ * T_next) / (mu_next * g_next)
        
        # Outputs for this layer
        outputs = (dz_i, z_next, g_current, H_current)
        
        # New carry for next iteration
        new_carry = (z_next, g_next, H_next)
        
        return new_carry, outputs
    
    # Initial state
    init_carry = (jnp.array(0.0), g_surface, H_surface)
    
    # Scan over all layers
    final_carry, outputs = lax.scan(altitude_step, init_carry, jnp.arange(nlayers))
    
    # Unpack outputs
    deltaz, z_boundaries_upper, gravity, scaleheight = outputs
    
    # Build complete altitude boundary array (nlayers + 1)
    # First boundary is at surface (z=0)
    z_boundaries = jnp.concatenate([jnp.array([0.0]), z_boundaries_upper])
    
    # TauREx returns nlayers-1 elements for gravity and scaleheight
    # (they're defined at intermediate boundaries, not at top/bottom)
    return z_boundaries, scaleheight[:-1], gravity[:-1], deltaz


def full_diff_compute_density_profile(pressure_profile, temperature_profile):
    """
    Compute density profile from ideal gas law.
    ρ = P / (k_B * T)
    """
    return pressure_profile / (KBOLTZ * temperature_profile)


def full_diff_prepare_model_data(taurex_model):
    """
    Extract static and initial data from TauREx model for full differentiable version.
    This extracts things that don't change during fitting.
    """
    if not taurex_model.built:
        taurex_model.build()
    
    taurex_model.initialize_profiles()
    
    # Static configuration (doesn't change with parameters)
    static_data = {
        'nlayers': int(taurex_model.nLayers),
        'pressure_profile': jnp.array(taurex_model.pressureProfile),
        'pressure_levels': jnp.array(taurex_model.pressure.pressure_profile_levels),
        'planet_mass': float(taurex_model._planet.fullMass),  # In kg (not Jupiter masses!)
        'planet_radius_base': float(taurex_model._planet.fullRadius),  # In m
        'star_radius': float(taurex_model._star.radius),  # In m
        'active_gases': tuple(taurex_model.chemistry.activeGases),
        'fill_gases': tuple(taurex_model.chemistry._fill_gases),
        'He_H2_ratio': float(taurex_model.chemistry._fill_ratio[0]),
    }
    
    # Initial state (for debugging/comparison)
    initial_state = {
        'temperature_profile': jnp.array(taurex_model.temperatureProfile),
        'altitude_profile': jnp.array(taurex_model.altitudeProfile),
        'deltaz': jnp.array(taurex_model.deltaz),
        'mu_profile': jnp.array(taurex_model.chemistry.muProfile),
    }
    
    return static_data, initial_state


# ========================================================================
# OPACITY INTERPOLATION - Fully differentiable opacity handling
# ========================================================================

def load_opacity_data(active_gases, wngrid_target, pre_interpolate_to_target=False):
    """
    Load opacity grids from OpacityCache into JAX-compatible format.
    This should be called ONCE outside the JIT-compiled forward model.
    
    Args:
        active_gases: List of gas names (e.g., ['H2O', 'CH4'])
        wngrid_target: Target wavenumber grid for interpolation
        pre_interpolate_to_target: If True, pre-interpolate spectral dimension to target grid
            This reduces memory usage and speeds up CPU execution ~2-3x, but is less necessary
            on GPU. Default False to preserve full-resolution behavior for GPU testing.
    
    Returns:
        opacity_data: Dict of {gas_name: opacity_info_dict}
    """
    opacity_cache = OpacityCache()
    opacity_data = {}
    
    for gas in active_gases:
        try:
            xsec = opacity_cache[gas]
        except Exception as e:
            print(f"Warning: {gas} not found in opacity cache ({e}), skipping")
            continue
            
        # Get the opacity grids
        # Note: Different opacity implementations may have different attributes
        # We'll try to handle the common case
        try:
            T_grid = np.array(xsec.temperatureGrid)
            P_grid = np.array(xsec.pressureGrid)
            wn_grid_native = np.array(xsec.wavenumberGrid)
            
            nT, nP = len(T_grid), len(P_grid)
            
            if pre_interpolate_to_target:
                # CPU-optimized path: pre-interpolate to target wavenumber grid
                nWN_target = len(wngrid_target)
                print(f"Loading {gas} opacity: T={nT}, P={nP}, WN={nWN_target} (pre-interpolated to target)")
                
                sigma_grid = np.zeros((nT, nP, nWN_target))
                for i, T in enumerate(T_grid):
                    for j, P in enumerate(P_grid):
                        # Get native grid opacity
                        sigma_native = xsec.opacity(T, P, wn_grid_native)
                        # Interpolate to target grid once
                        sigma_grid[i, j, :] = np.interp(wngrid_target, wn_grid_native, sigma_native)
                
                # Store with target grid (no spectral interpolation needed later)
                opacity_data[gas] = {
                    'T_grid': jnp.array(T_grid),
                    'P_grid': jnp.array(P_grid),
                    'sigma_grid': jnp.array(sigma_grid),  # (nT, nP, nWN_target)
                    'pre_interpolated': True,  # Flag to skip spectral interp
                }
            else:
                # GPU-friendly path: keep full native resolution
                nWN_native = len(wn_grid_native)
                print(f"Loading {gas} opacity: T={nT}, P={nP}, WN={nWN_native} (native resolution)")
                
                # Pre-compute opacity on native grid for all (T, P) points
                sigma_grid = np.zeros((nT, nP, nWN_native))
                for i, T in enumerate(T_grid):
                    for j, P in enumerate(P_grid):
                        sigma_grid[i, j, :] = xsec.opacity(T, P, wn_grid_native)
                
                opacity_data[gas] = {
                    'T_grid': jnp.array(T_grid),
                    'P_grid': jnp.array(P_grid),
                    'wn_grid': jnp.array(wn_grid_native),
                    'sigma_grid': jnp.array(sigma_grid),  # (nT, nP, nWN_native)
                    'wngrid_target': jnp.array(wngrid_target),
                    'pre_interpolated': False,
                }
            
        except AttributeError as e:
            print(f"Warning: Could not load opacity for {gas}: {e}")
            continue
    
    return opacity_data


def full_diff_interpolate_opacity_single_layer(T_layer, P_layer, opacity_info):
    """
    Interpolate opacity at a single (T, P) point for one gas.
    Uses bilinear interpolation in (T, log P) space, then spectral interpolation if needed.
    
    Args:
        T_layer: Temperature for this layer (scalar)
        P_layer: Pressure for this layer (scalar)
        opacity_info: Dict with 'T_grid', 'P_grid', 'sigma_grid', and either:
            - 'wn_grid' + 'wngrid_target' (if pre_interpolated=False), or
            - just sigma_grid already on target grid (if pre_interpolated=True)
    
    Returns:
        sigma: Cross-section on target grid for this layer (1D array)
    """
    T_grid = opacity_info['T_grid']
    P_grid = opacity_info['P_grid']
    sigma_grid = opacity_info['sigma_grid']
    pre_interpolated = opacity_info.get('pre_interpolated', False)
    
    # Find bracketing indices for T
    # Clip to valid range
    T_layer = jnp.clip(T_layer, T_grid[0], T_grid[-1])
    i_T = jnp.searchsorted(T_grid, T_layer) - 1
    i_T = jnp.clip(i_T, 0, len(T_grid) - 2)
    
    # Find bracketing indices for log P (opacity typically linear in log P)
    log_P_grid = jnp.log(P_grid)
    log_P_layer = jnp.log(jnp.maximum(P_layer, 1e-30))  # Avoid log(0)
    log_P_layer = jnp.clip(log_P_layer, log_P_grid[0], log_P_grid[-1])
    i_P = jnp.searchsorted(log_P_grid, log_P_layer) - 1
    i_P = jnp.clip(i_P, 0, len(P_grid) - 2)
    
    # Get corner values
    T_low, T_high = T_grid[i_T], T_grid[i_T + 1]
    log_P_low, log_P_high = log_P_grid[i_P], log_P_grid[i_P + 1]
    
    # Get cross-sections at 4 corners (each is a spectrum)
    sigma_00 = sigma_grid[i_T, i_P, :]      # T_low, P_low
    sigma_01 = sigma_grid[i_T, i_P + 1, :]  # T_low, P_high
    sigma_10 = sigma_grid[i_T + 1, i_P, :]  # T_high, P_low
    sigma_11 = sigma_grid[i_T + 1, i_P + 1, :]  # T_high, P_high
    
    # Bilinear interpolation weights
    w_T = (T_layer - T_low) / (T_high - T_low + 1e-30)
    w_P = (log_P_layer - log_P_low) / (log_P_high - log_P_low + 1e-30)
    
    # Interpolate in (T, log P)
    sigma_interp = (
        (1 - w_T) * (1 - w_P) * sigma_00 +
        (1 - w_T) * w_P * sigma_01 +
        w_T * (1 - w_P) * sigma_10 +
        w_T * w_P * sigma_11
    )
    
    # Spectral interpolation to target wavenumber grid (if not pre-interpolated)
    if pre_interpolated:
        # Already on target grid, return directly
        return sigma_interp
    else:
        # Need to interpolate spectrally
        wn_grid = opacity_info['wn_grid']
        wngrid_target = opacity_info['wngrid_target']
        sigma_target = jnp.interp(wngrid_target, wn_grid, sigma_interp)
        return sigma_target


def full_diff_interpolate_opacity_all_layers(T_profile, P_profile, opacity_info):
    """
    Interpolate opacity at all layers for one gas.
    Vectorizes over layers using vmap.
    
    Args:
        T_profile: Temperature profile (nlayers,)
        P_profile: Pressure profile (nlayers,)
        opacity_info: Dict with opacity grid data
    
    Returns:
        sigma: Cross-sections for all layers (nlayers, nwavenumbers)
    """
    # Vectorize over layers
    interpolate_vmap = jax.vmap(
        full_diff_interpolate_opacity_single_layer,
        in_axes=(0, 0, None)
    )
    
    return interpolate_vmap(T_profile, P_profile, opacity_info)


def full_diff_compute_molecular_opacities(T_profile, P_profile, opacity_data_dict):
    """
    Compute molecular opacities for all active gases at current (T, P).
    
    Args:
        T_profile: Temperature profile (nlayers,)
        P_profile: Pressure profile (nlayers,)
        opacity_data_dict: Dict of {gas_name: opacity_info}
    
    Returns:
        opacities: Dict of {gas_name: sigma_array(nlayers, nwavenumbers)}
    """
    opacities = {}
    
    for gas, opacity_info in opacity_data_dict.items():
        opacities[gas] = full_diff_interpolate_opacity_all_layers(
            T_profile, P_profile, opacity_info
        )
    
    return opacities


# ========================================================================
# FULLY DIFFERENTIABLE FORWARD MODEL - Assembles all components
# ========================================================================

def full_diff_create_forward_model(taurex_model: TransmissionModel, wngrid, 
                                    pre_interpolate_opacity=False):
    """
    Create a fully differentiable forward model that recomputes all
    parameter-dependent quantities on each forward pass.
    
    This NEW version matches TauREx's dynamic behavior - it recomputes
    profiles, opacities, and geometry from current parameter values on
    every call, unlike the old frozen approach.
    
    Strategy: Reuse the working path_integral logic, but feed it dynamically
    computed opacities and profiles instead of frozen ones.
    
    Args:
        taurex_model: Built TauREx model
        wngrid: Wavenumber grid for output spectrum
        pre_interpolate_opacity: If True, pre-interpolate opacities to target grid
            for better CPU performance (~2-3x speedup). Default False for GPU compatibility.
    
    Returns:
        forward_model: Function (params) -> (absorption, tau)
    """
    # 1. Extract static configuration (once)
    static_data, _ = full_diff_prepare_model_data(taurex_model)
    
    # 2. Pre-load opacity grids for molecular absorption (once, outside JIT)
    print(f"Pre-loading opacities for gases: {static_data['active_gases']}")
    opacity_data = load_opacity_data(
        static_data['active_gases'], 
        wngrid, 
        pre_interpolate_to_target=pre_interpolate_opacity
    )
    
    # 3. Pre-compute CIA and Rayleigh (they're weakly T-dependent, treat as static for now)
    # This uses the INITIAL state - for full correctness we'd need to interpolate CIA too
    _, cia_sigma, other_sigma = prepare_contributions(taurex_model, wngrid)
    
    # 4. Package static data for JIT (must be hashable)
    nlayers = static_data['nlayers']
    pressure_profile = static_data['pressure_profile']
    pressure_levels = static_data['pressure_levels']
    planet_mass = static_data['planet_mass']
    planet_radius_base = static_data['planet_radius_base']
    star_radius = static_data['star_radius']
    
    active_gases = static_data['active_gases']
    active_gas_weights = tuple(MOLECULAR_WEIGHTS[g] for g in active_gases)
    fill_gases = static_data['fill_gases']
    fill_gas_weights = tuple(MOLECULAR_WEIGHTS[g] for g in fill_gases)
    He_H2_ratio = static_data['He_H2_ratio']
    
    # 5. Convert opacity data to tuple format for JIT
    opacity_data_tuple = tuple(
        (gas, opacity_data[gas]) for gas in active_gases
    )
    
    # 6. Package static contributions
    static_contributions = (cia_sigma, other_sigma)
    
    # 7. Build static_data tuple for path_integral (matches old signature)
    static_data_tuple = (
        nlayers,
        planet_radius_base, 
        star_radius,
        active_gases,
        active_gas_weights,
        fill_gases, 
        fill_gas_weights,
        He_H2_ratio
    )
    
    @partial(jax.jit, static_argnames=['nlayers', 'active_gases', 'fill_gases'])
    def forward_model(params, nlayers=nlayers, active_gases=active_gases, fill_gases=fill_gases):
        """
        Full forward pass - recomputes everything from params, then uses
        the proven path_integral logic.
        
        Returns:
            absorption: (nwavenumbers,) array
            tau: (nlayers, nwavenumbers) array
        """
        # Step 1: Compute temperature profile from current T parameter
        T_profile = full_diff_compute_temperature_profile(params['T'], nlayers)
        
        # Step 2: Compute mixing ratio profiles from current gas parameters
        mixing_profiles = full_diff_compute_mixing_profiles(params, active_gases, nlayers)
        
        # Step 3: Compute mean molecular weight from current mixing ratios
        mu_profile = full_diff_compute_mu_profile(
            mixing_profiles,
            active_gases,
            active_gas_weights,
            fill_gases,
            fill_gas_weights,
            He_H2_ratio
        )
        
        # Step 4: Compute altitude/gravity/scale height from current T, μ, planet_radius
        planet_radius = params['planet_radius'] * planet_radius_base
        altitude_boundaries, scale_height, gravity, deltaz = full_diff_compute_altitude_profile(
            T_profile,
            pressure_levels,
            mu_profile,
            planet_mass,
            planet_radius
        )
        
        # Compute altitude at layer centers
        altitude_profile = 0.5 * (altitude_boundaries[:-1] + altitude_boundaries[1:])
        
        # Step 5: Interpolate molecular opacities at current (T, P)
        # Build stacked opacity array for all gases
        absorption_sigmas_list = []
        for gas, opacity_info in opacity_data_tuple:
            sigma = full_diff_interpolate_opacity_all_layers(
                T_profile, pressure_profile, opacity_info
            )
            absorption_sigmas_list.append(sigma)
        
        absorption_sigmas_tuple = tuple(absorption_sigmas_list)
        
        # Step 6: Package dynamic data for path_integral
        dynamic_data = (
            deltaz,
            altitude_profile,
            pressure_profile,
            static_contributions[0],  # cia_sigma
            static_contributions[1]   # other_sigma
        )
        
        # Step 7: Call the proven path_integral function
        absorption, tau = path_integral(
            jnp.array(wngrid),
            params,
            absorption_sigmas_tuple,
            dynamic_data,
            static_data_tuple
        )
        
        return absorption, tau
    
    return forward_model


def full_diff_create_binned_forward_model(
    taurex_model: TransmissionModel, 
    observed_spectrum,
    *,
    use_lambda_measure=False,
    debug_coverage=True,
    pre_interpolate_opacity=False
):
    """
    Create binned version of fully differentiable forward model.
    
    Uses the robust binning matrix creation from the working old version.
    
    Args:
        taurex_model: Built TauREx model
        observed_spectrum: ObservedSpectrum with binning info
        use_lambda_measure: If True, average using λ-measure
        debug_coverage: Print info about empty/partial coverage rows
        pre_interpolate_opacity: If True, pre-interpolate opacities to target grid
            for better CPU performance (~2-3x speedup). Default False for GPU compatibility.
    
    Returns:
        forward_binned: Function (params) -> binned_absorption
    """
    # 1) Clip model grid to observed domain (still high-res)
    high_res_wngrid = clip_native_to_wngrid(
        taurex_model.nativeWavenumberGrid,
        observed_spectrum.wavenumberGrid
    )
    
    # Ensure increasing order
    if high_res_wngrid[0] > high_res_wngrid[-1]:
        high_res_wngrid = high_res_wngrid[::-1]
    
    # Get bin edges
    bin_edges = observed_spectrum.binEdges
    if bin_edges[0] > bin_edges[-1]:
        bin_edges = bin_edges[::-1]
    
    # 2) Create high-res forward model (NEW fully differentiable version)
    forward_high_res = full_diff_create_forward_model(
        taurex_model, 
        high_res_wngrid,
        pre_interpolate_opacity=pre_interpolate_opacity
    )
    
    # 3) Build robust binning matrix (same as old working version)
    W_np = _create_binning_matrix_np(
        high_res_wngrid, bin_edges, use_lambda_measure=use_lambda_measure
    )
    
    if debug_coverage:
        row_sums = W_np.sum(axis=1)
        n_empty = int(np.sum(row_sums == 0.0))
        n_partial = int(np.sum((row_sums > 0.0) & (np.abs(row_sums - 1.0) > 1e-6)))
        print(f"[binning] W shape: {W_np.shape}, empty rows: {n_empty}, partial rows: {n_partial}")
        print(f"[grids] high-res N={len(high_res_wngrid)}, "
              f"bins={W_np.shape[0]}, observed points={len(observed_spectrum.spectrum)}")
    
    W = jnp.asarray(W_np)
    
    @jax.jit
    def forward_binned(params):
        """Binned spectrum from high-res forward model."""
        absorption_high_res, _ = forward_high_res(params)
        # Use matrix multiplication (same as old working version)
        absorption_binned = jnp.dot(W, absorption_high_res)
        return absorption_binned
    
    return forward_binned


def _edges_from_centers_np(x):
    """Return edges for a 1D monotonic grid of centers (NumPy)."""
    x = np.asarray(x)
    assert x.ndim == 1 and x.size >= 2
    xe = np.empty(x.size + 1, dtype=x.dtype)
    xe[1:-1] = 0.5 * (x[:-1] + x[1:])
    xe[0]    = x[0] - (xe[1]  - x[0])
    xe[-1]   = x[-1] + (x[-1] - xe[-2])
    return xe

def _create_binning_matrix_np(high_res_wngrid, obs_bin_edges, *, use_lambda_measure=False):
    """
    Robust binning matrix built in NumPy (easier to debug), returned as np.ndarray.
    - Computes exact overlap with high-res *cell edges*.
    - Clips observed edges to model domain.
    - Fills truly empty rows with linear interpolation at the bin center (no NN spikes).
    - If use_lambda_measure=True, applies σ^{-2} Jacobian (λ-averaging).
    """
    x = np.asarray(high_res_wngrid)
    be = np.asarray(obs_bin_edges)

    # Ensure increasing
    if x[0] > x[-1]:
        x = x[::-1]
    if be[0] > be[-1]:
        be = be[::-1]

    xe = _edges_from_centers_np(x)      # high-res cell edges
    lo, hi = xe[0], xe[-1]

    # Clip observed edges to model domain
    be = np.clip(be, lo, hi)

    n_bins = be.size - 1
    n_hr   = x.size

    # Overlap of each obs bin with each high-res cell
    left  = np.maximum(be[:-1, None], xe[:-1][None, :])   # (n_bins, n_hr)
    right = np.minimum(be[1:,  None], xe[1:][None,  :])
    overlap = np.clip(right - left, 0.0, None)

    # Optional λ-measure: dλ/dσ ∝ 1/σ^2 (constant cancels on normalization)
    if use_lambda_measure:
        jac = 1.0 / (x[None, :] ** 2)
        weights = overlap * jac
    else:
        weights = overlap

    row_sums = weights.sum(axis=1, keepdims=True)
    W = np.divide(weights, row_sums, out=np.zeros_like(weights), where=row_sums > 0)

    # Handle zero-coverage rows by linear interpolation at bin centers
    empty = (row_sums[:, 0] == 0)
    if np.any(empty):
        bc = 0.5 * (be[:-1] + be[1:])
        bc_empty = bc[empty]

        j_right = np.searchsorted(x, bc_empty, side="left")
        j_right = np.clip(j_right, 1, n_hr - 1)
        j_left  = j_right - 1

        xL, xR = x[j_left], x[j_right]
        alpha = (bc_empty - xL) / (xR - xL + 1e-12)

        rows = np.where(empty)[0]
        W[rows, j_left]  = 1.0 - alpha
        W[rows, j_right] = alpha

    return W

def create_binned_forward_model(
    taurex_model,
    observed_spectrum,
    *,
    use_lambda_measure=False,   # set True if observed bins came from wavelength
    debug_coverage=True
):
    """Create forward model that outputs binned spectra.

    Args
    ----
    taurex_model: model with .nativeWavenumberGrid and used by create_forward_model
    observed_spectrum: object with .wavenumberGrid (centers) and .binEdges (edges, *in wavenumber*)
    use_lambda_measure: if True, average using λ-measure (adds σ^{-2} Jacobian)
    debug_coverage: print info about empty/partial coverage rows
    """
    # 1) Clip model grid to observed domain (still high-res)
    high_res_wngrid = clip_native_to_wngrid(
        taurex_model.nativeWavenumberGrid,
        observed_spectrum.wavenumberGrid
    )

    # Ensure increasing order for safety
    if high_res_wngrid[0] > high_res_wngrid[-1]:
        high_res_wngrid = high_res_wngrid[::-1]

    bin_edges = observed_spectrum.binEdges
    if bin_edges[0] > bin_edges[-1]:
        bin_edges = bin_edges[::-1]

    # 2) JAX forward model at high resolution
    jax_forward_model = create_forward_model(taurex_model, high_res_wngrid)

    # 3) Build robust binning matrix (NumPy for construction + debug; convert to JAX array)
    W_np = _create_binning_matrix_np(
        high_res_wngrid, bin_edges, use_lambda_measure=use_lambda_measure
    )

    if debug_coverage:
        row_sums = W_np.sum(axis=1)
        n_empty = int(np.sum(row_sums == 0.0))
        n_partial = int(np.sum((row_sums > 0.0) & (np.abs(row_sums - 1.0) > 1e-6)))
        print(f"[binning] W shape: {W_np.shape}, empty rows: {n_empty}, partial rows: {n_partial}")

        # Sanity: number of bins vs observed spectrum length
        print(f"[grids] high-res N={len(high_res_wngrid)}, "
              f"bins={W_np.shape[0]}, observed points={len(observed_spectrum.spectrum)}")

    W = jnp.asarray(W_np)

    @jax.jit
    def binned_forward_model(params):
        high_res_spectrum, _tau = jax_forward_model(params)   # tau unused here
        # Multiply W (bins x high-res) by spectrum (high-res,)
        binned_spectrum = jnp.dot(W, high_res_spectrum)
        return binned_spectrum

    return binned_forward_model


# def create_binned_forward_model(taurex_model, observed_spectrum):
#     """Create forward model using TauREx's FluxBinner for consistency."""
#     # 1) Clip model grid to observed domain
#     high_res_wngrid = clip_native_to_wngrid(
#         taurex_model.nativeWavenumberGrid,
#         observed_spectrum.wavenumberGrid
#     )
    
#     # 2) Create JAX forward model at high resolution
#     jax_forward_model = create_forward_model(taurex_model, high_res_wngrid)
    
#     # 3) Use TauREx's own binner - guaranteed consistency!
#     binner = observed_spectrum.create_binner()
    
#     def binned_forward_model(params):
#         high_res_spectrum, _tau = jax_forward_model(params)
#         # Convert to numpy for binning, then back to JAX
#         high_res_np = np.array(high_res_spectrum)
#         _, binned_np, _, _ = binner.bindown(high_res_wngrid, high_res_np)
#         return jnp.array(binned_np)
    
#     return binned_forward_model

# ========== PARAMETER TRANSFORMS ==========
def _sigmoid_box(u, lo, hi):
    s = jax.nn.sigmoid(u)
    return lo + s * (hi - lo)

def _inv_sigmoid_box(x, lo, hi, eps=1e-8):
    s = jnp.clip((x - lo) / (hi - lo), eps, 1 - eps)
    return jnp.log(s) - jnp.log1p(-s)

def _log_box(u, lo, hi):
    z = jax.nn.sigmoid(u)
    return jnp.exp(jnp.log(lo) + z * (jnp.log(hi) - jnp.log(lo)))

def _inv_log_box(x, lo, hi, eps=1e-12):
    x = jnp.clip(x, lo + eps, hi - eps)
    z = (jnp.log(x) - jnp.log(lo)) / (jnp.log(hi) - jnp.log(lo))
    z = jnp.clip(z, eps, 1 - eps)
    return jnp.log(z) - jnp.log1p(-z)

def make_transforms(param_info, fit_params):
    use_log, bounds = {}, {}
    for k in fit_params:
        lo, hi = param_info[k]['range']
        scale = str(param_info[k].get('scale', 'linear')).lower()
        use_log[k] = (scale == 'log' and lo > 0.0 and hi > lo)
        bounds[k] = (float(lo), float(hi))
    
    def to_constrained(u_dict):
        c = {}
        for k, u in u_dict.items():
            lo, hi = bounds[k]
            c[k] = _log_box(u, lo, hi) if use_log[k] else _sigmoid_box(u, lo, hi)
        return c
    
    def to_unconstrained(c_dict):
        u = {}
        for k, x in c_dict.items():
            lo, hi = bounds[k]
            u[k] = _inv_log_box(x, lo, hi) if use_log[k] else _inv_sigmoid_box(x, lo, hi)
        return u
    
    return to_constrained, to_unconstrained

def make_packer(u0_dict):
    keys = list(u0_dict.keys())
    def pack(d): return jnp.stack([d[k] for k in keys])
    def unpack(v): return {k: v[i] for i, k in enumerate(keys)}
    return pack, unpack, keys

# ========== OPTIMIZER ==========
def fit_with_value_and_grad_adam(
    forward_binned,
    observed_y,
    observed_err,
    init_params,
    param_info,
    fit_params,
    steps=500,
    lr=1e-3,
    clip_norm=1.0,
    print_every=50,
    nan_guard=True,
    loss="mse",                 # "mse" | "gaussian" | "lognormal" | "studentt" | "gp"
    reduction="mean",           # for "mse": "mean" or "sum"
    loss_kwargs=None,           # dict of extras, e.g. {"nu":4.0, "rho":0.02, "ell":50.0, "noise_floor":0.0, "jitter":1e-6}
    x_for_gp=None               # 1D array of bin centers (wavelength or wavenumber) for "gp"
):
    loss_kwargs = {} if loss_kwargs is None else dict(loss_kwargs)
    EPS = 1e-12

    # ---- transforms / packing (unchanged) ----
    fixed = {k: v for k, v in init_params.items() if k not in fit_params}
    train0 = {k: init_params[k] for k in fit_params}
    to_c, to_u = make_transforms(param_info, fit_params)
    u0 = to_u(train0)
    pack, unpack, keys = make_packer(u0)
    uvec = pack(u0)

    # ---- optimizer ----
    opt_chain = []
    if clip_norm is not None:
        opt_chain.append(optax.clip_by_global_norm(clip_norm))
    opt_chain.append(optax.adam(lr))
    opt = optax.chain(*opt_chain)
    opt_state = opt.init(uvec)

    # ---- data tensors ----
    obs_y = jnp.asarray(observed_y)
    obs_err = jnp.maximum(jnp.asarray(observed_err), EPS)

    # ---- loss primitives ----
    def mse_loss(y, mu, sigma, reduction="mean"):
        r = (y - mu) / sigma
        r2 = r * r
        return jnp.mean(r2) if reduction == "mean" else jnp.sum(r2)

    def gaussian_nll(y, mu, sigma):
        r = (y - mu) / (sigma + EPS)
        return 0.5 * jnp.sum(r**2) + jnp.sum(jnp.log(sigma + EPS))

    def lognormal_nll(y, mu, sigma_y):
        # multiplicative noise; guard for positivity
        y = jnp.clip(y, a_min=EPS)
        mu = jnp.clip(mu, a_min=EPS)
        sigma_ln = jnp.sqrt(jnp.log1p((sigma_y / (y + EPS))**2))
        r = (jnp.log(y) - jnp.log(mu)) / (sigma_ln + EPS)
        return 0.5 * jnp.sum(r**2) + jnp.sum(jnp.log(sigma_ln + EPS))

    def studentt_nll(y, mu, sigma, nu=4.0):
        r = (y - mu) / (sigma + EPS)
        c = 0.5 * (nu + 1.0)
        return jnp.sum(c * jnp.log1p((r**2) / nu) + jnp.log(sigma + EPS))

    def gaussian_gp_nll(y, mu, x, sigma_white, rho, ell, jitter=1e-6):
        # SE kernel; O(N^3)
        dx = (x[:, None] - x[None, :]) / (ell + EPS)
        K = (rho**2) * jnp.exp(-0.5 * dx**2)
        K = K + jnp.diag(sigma_white**2 + jitter)
        r = y - mu
        L = jnp.linalg.cholesky(K)
        # Solve K^{-1} r via two triangular solves
        alpha = jsp.linalg.solve_triangular(L.T,
                 jsp.linalg.solve_triangular(L, r, lower=True), lower=False)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L) + EPS))
        N = y.size
        return 0.5 * (r @ alpha) + 0.5 * logdet + 0.5 * N * jnp.log(2.0*jnp.pi)

    # ---- master loss ----
    def loss_fn(uvec_):
        udict = unpack(uvec_)
        c_fit = to_c(udict)
        params = {**fixed, **c_fit}
        pred = jnp.asarray(forward_binned(params))

        if nan_guard:
            pred = jnp.nan_to_num(pred, nan=0.0, posinf=1e300, neginf=-1e300)

        if loss == "mse":
            return mse_loss(obs_y, pred, obs_err, reduction=reduction)

        elif loss == "gaussian":
            return gaussian_nll(obs_y, pred, obs_err)

        elif loss == "lognormal":
            return lognormal_nll(obs_y, pred, obs_err)

        elif loss == "studentt":
            nu = float(loss_kwargs.get("nu", 4.0))
            return studentt_nll(obs_y, pred, obs_err, nu=nu)

        elif loss == "gp":
            if x_for_gp is None:
                raise ValueError("loss='gp' requires x_for_gp (bin centers).")
            x = jnp.asarray(x_for_gp)
            noise_floor = float(loss_kwargs.get("noise_floor", 0.0))
            rho  = float(loss_kwargs.get("rho", 1.0))
            ell  = float(loss_kwargs.get("ell", 10.0))
            jitter = float(loss_kwargs.get("jitter", 1e-6))
            sigma_eff = jnp.sqrt(obs_err**2 + noise_floor**2)
            return gaussian_gp_nll(obs_y, pred, x, sigma_eff, rho, ell, jitter=jitter)

        else:
            raise ValueError(f"Unknown loss '{loss}'. "
                             "Choose from: 'mse', 'gaussian', 'lognormal', 'studentt', 'gp'.")

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    # ---- loop ----
    losses = []
    for step in range(steps):
        val, g = loss_and_grad(uvec)
        updates, opt_state = opt.update(g, opt_state, uvec)
        uvec = optax.apply_updates(uvec, updates)

        if (step % print_every) == 0 or step == steps - 1:
            grad_norm = float(jnp.linalg.norm(jnp.asarray(g)))
            losses.append(float(val))
            print(f"step {step:4d} | loss={float(val):.6g} | ||grad||={grad_norm:.3e}")

        if not jnp.isfinite(val):
            print("Loss is not finite; try lowering lr, tightening bounds, or enabling nan_guard.")
            break

    final_cfit = to_c(unpack(uvec))
    final_params = {**fixed, **final_cfit}
    return final_params, losses

# ========== PLOTTING ==========
def plot_fit_comparison(obs, forward_binned, initial_params, final_params, fit_params):
    """Plot comparison between observed data and fitted model."""
    import matplotlib.pyplot as plt
    
    # Generate predictions
    initial_pred = np.array(forward_binned(initial_params))
    final_pred = np.array(forward_binned(final_params))
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Main comparison plot
    ax1.errorbar(obs.wavelengthGrid, obs.spectrum, obs.errorBar if hasattr(obs, 'errorBar') else None,
                fmt='o', alpha=0.7, label='Observed', markersize=4)
    ax1.plot(obs.wavelengthGrid, initial_pred, '--', alpha=0.8, label='Initial model', linewidth=2)
    ax1.plot(obs.wavelengthGrid, final_pred, 'r-', label='Best fit', linewidth=2)
    ax1.set_xlabel('Wavelength (μm)')
    ax1.set_ylabel('Transit Depth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Model Fit Comparison')
    
    # Residuals plot
    final_residuals = (obs.spectrum - final_pred)
    if hasattr(obs, 'errorBar'):
        final_residuals_norm = final_residuals / obs.errorBar
        ax2.errorbar(obs.wavelengthGrid, final_residuals_norm, yerr=np.ones_like(obs.spectrum),
                    fmt='o', alpha=0.7, markersize=4)
        ax2.set_ylabel('Normalized Residuals (σ)')
    else:
        ax2.plot(obs.wavelengthGrid, final_residuals, 'o', alpha=0.7, markersize=4)
        ax2.set_ylabel('Residuals')
    
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Wavelength (μm)')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Fit Residuals')
    
    plt.tight_layout()
    
    # Print parameter comparison
    print("\nParameter Comparison:")
    print("=" * 50)
    for key in fit_params:
        initial_val = float(initial_params[key])
        final_val = float(final_params[key])
        change = final_val - initial_val
        percent_change = 100 * change / initial_val if initial_val != 0 else float('inf')
        print(f"{key:15s}: {initial_val:10.6f} -> {final_val:10.6f} "
              f"(Δ{change:+.6f}, {percent_change:+.2f}%)")
    
    # Calculate fit statistics
    if hasattr(obs, 'errorBar'):
        chi_squared = np.sum(((obs.spectrum - final_pred) / obs.errorBar)**2)
        reduced_chi_squared = chi_squared / (len(obs.spectrum) - len(fit_params))
        print(f"\nFit Statistics:")
        print(f"Chi-squared: {chi_squared:.2f}")
        print(f"Reduced chi-squared: {reduced_chi_squared:.2f}")
        print(f"RMS residual: {np.sqrt(np.mean(final_residuals**2)):.6f}")
    
    plt.show()
    
    return fig


import numpy as np
import matplotlib.pyplot as plt

def debug_taurex_vs_jax(tm, params, wngrid, forward_binned):
    """
    Systematic comparison between TauREx and JAX implementations.
    This will help isolate where the discrepancy occurs.
    """
    print("=== DEBUGGING TAUREX vs JAX ===\n")
    
    # 1. COMPARE RAW TAUREX OUTPUT
    print("1. Computing TauREx native spectrum...")
    tm.build()  # Ensure model is built
    tm.initialize_profiles()  # Initialize profiles
    
    # Set parameters to match your JAX model
    for key, val in params.items():
        if key in tm.fittingParameters:
            tm[key] = float(val)
    
    # Get TauREx high-res spectrum (returns tuple of spectrum, tau)
    taurex_result = tm.model(wngrid)
    if isinstance(taurex_result, tuple):
        taurex_spectrum_hr = taurex_result[0]  # First element is the spectrum
    else:
        taurex_spectrum_hr = taurex_result
    
    print(f"   TauREx high-res spectrum shape: {taurex_spectrum_hr.shape}")
    print(f"   TauREx spectrum range: {taurex_spectrum_hr.min():.6f} - {taurex_spectrum_hr.max():.6f}")
    
    # 2. COMPARE YOUR JAX HIGH-RES OUTPUT  
    print("\n2. Computing JAX high-res spectrum...")
    jax_forward_model = create_forward_model(tm, wngrid)
    jax_spectrum_hr, _ = jax_forward_model(params)
    jax_spectrum_hr = np.array(jax_spectrum_hr)
    print(f"   JAX high-res spectrum shape: {jax_spectrum_hr.shape}")
    print(f"   JAX spectrum range: {jax_spectrum_hr.min():.6f} - {jax_spectrum_hr.max():.6f}")
    
    # 3. COMPARE HIGH-RES SPECTRA
    print("\n3. Comparing high-resolution spectra...")
    
    # Handle size mismatch by comparing overlapping region
    min_size = min(len(taurex_spectrum_hr), len(jax_spectrum_hr))
    taurex_trimmed = taurex_spectrum_hr[:min_size]
    jax_trimmed = jax_spectrum_hr[:min_size]
    
    # Check if TauREx is in ppm units (values > 100 suggest ppm)
    if np.median(taurex_trimmed) > 100:
        print("   WARNING: TauREx appears to be in ppm units, converting...")
        taurex_trimmed = taurex_trimmed * 1e-6  # Convert ppm to fractional
        print(f"   TauREx converted range: {taurex_trimmed.min():.6f} - {taurex_trimmed.max():.6f}")
    
    hr_diff = np.abs(taurex_trimmed - jax_trimmed)
    hr_rel_diff = hr_diff / (np.abs(taurex_trimmed) + 1e-10)
    print(f"   Max absolute difference: {hr_diff.max():.2e}")
    print(f"   Mean absolute difference: {hr_diff.mean():.2e}")
    print(f"   Max relative difference: {hr_rel_diff.max():.2e}")
    print(f"   Mean relative difference: {hr_rel_diff.mean():.2e}")
    
    # Store converted values for plotting
    taurex_spectrum_hr_converted = taurex_trimmed
    jax_spectrum_hr_trimmed = jax_trimmed
    
    # 4. COMPARE BINNED SPECTRA
    print("\n4. Computing binned spectra...")
    jax_spectrum_binned = np.array(forward_binned(params))
    
    print(f"   JAX binned spectrum shape: {jax_spectrum_binned.shape}")
    print(f"   JAX binned range: {jax_spectrum_binned.min():.6f} - {jax_spectrum_binned.max():.6f}")
    
    # Note: For complete binning comparison, pass obs to this function and compare binned spectra
    
    # 5. DETAILED COMPONENT ANALYSIS
    print("\n5. Analyzing individual components...")
    
    # Compare density profiles
    print("   5a. Density Profiles:")
    taurex_density = tm.densityProfile
    
    # Compute your JAX density
    T = float(params['T'])
    mu_current = compute_current_mu(tm, params)  # You'll need to extract this
    jax_density = np.array(tm.pressureProfile) * mu_current / (KBOLTZ * T)
    
    density_diff = np.abs(taurex_density - jax_density)
    print(f"      TauREx density range: {taurex_density.min():.2e} - {taurex_density.max():.2e}")
    print(f"      JAX density range: {jax_density.min():.2e} - {jax_density.max():.2e}")
    print(f"      Max density difference: {density_diff.max():.2e}")
    print(f"      Relative density difference: {(density_diff/taurex_density).max():.2e}")
    
    # Compare path lengths
    print("   5b. Path Lengths:")
    taurex_paths = tm.compute_path_length_old(tm.deltaz)
    jax_paths = compute_path_length_simple(
        tm.nLayers, 
        np.array(tm.altitudeProfile),
        np.array(tm.deltaz),
        float(params['planet_radius']) * tm._planet.fullRadius
    )
    
    # Compare first few path lengths
    for i in range(min(3, len(taurex_paths))):
        path_diff = np.abs(np.array(taurex_paths[i]) - np.array(jax_paths[i]))
        print(f"      Layer {i} path diff: max={path_diff.max():.2e}, mean={path_diff.mean():.2e}")
    
    # 6. PLOTTING COMPARISON
    print("\n6. Creating comparison plots...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Use trimmed arrays for plotting
    wngrid_trimmed = wngrid[:min_size]
    
    # Plot high-res comparison
    axes[0,0].plot(wngrid_trimmed, taurex_spectrum_hr_converted, 'b-', label='TauREx', alpha=0.7)
    axes[0,0].plot(wngrid_trimmed, jax_spectrum_hr_trimmed, 'r--', label='JAX', alpha=0.7)
    axes[0,0].set_xlabel('Wavenumber')
    axes[0,0].set_ylabel('Transit Depth')
    axes[0,0].set_title('High-Resolution Comparison')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot high-res difference
    axes[0,1].plot(wngrid_trimmed, hr_diff, 'k-')
    axes[0,1].set_xlabel('Wavenumber') 
    axes[0,1].set_ylabel('Absolute Difference')
    axes[0,1].set_title('High-Res Absolute Difference')
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot density comparison
    axes[1,0].plot(taurex_density, tm.pressureProfile, 'b-', label='TauREx')
    axes[1,0].plot(jax_density, tm.pressureProfile, 'r--', label='JAX')
    axes[1,0].set_xlabel('Density')
    axes[1,0].set_ylabel('Pressure')
    axes[1,0].set_yscale('log')
    axes[1,0].set_xscale('log')
    axes[1,0].set_title('Density Profile Comparison')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot relative difference vs wavelength
    axes[1,1].plot(1e4/wngrid, hr_rel_diff * 100, 'k-')  # Convert to wavelength in microns
    axes[1,1].set_xlabel('Wavelength (μm)')
    axes[1,1].set_ylabel('Relative Difference (%)')
    axes[1,1].set_title('Relative Difference vs Wavelength')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 7. SUMMARY AND RECOMMENDATIONS
    print("\n7. DEBUGGING SUMMARY:")
    print("=" * 50)
    
    if hr_rel_diff.max() < 0.01:  # Less than 1% difference
        print("✅ High-res spectra match well (< 1% difference)")
    elif hr_rel_diff.max() < 0.1:  # Less than 10% difference  
        print("⚠️  High-res spectra have moderate differences (1-10%)")
        print("   → Check density profile and molecular weight calculations")
    else:
        print("❌ High-res spectra have large differences (> 10%)")
        print("   → Major issue with physics implementation")
        
        if (density_diff/taurex_density).max() > 0.1:
            print("   → PROBLEM: Density profiles don't match - check mu calculation")
        
        # Check if it's a scaling issue
        scaling_factor = np.median(taurex_spectrum_hr / jax_spectrum_hr)
        if 0.8 < scaling_factor < 1.2:
            print(f"   → Possible scaling issue: factor = {scaling_factor:.3f}")
            print("   → Check units, radius scaling, or normalization")
    
    return {
        'taurex_hr': taurex_spectrum_hr,
        'jax_hr': jax_spectrum_hr, 
        'hr_diff': hr_diff,
        'hr_rel_diff': hr_rel_diff,
        'taurex_density': taurex_density,
        'jax_density': jax_density
    }

def compute_current_mu(tm, params):
    """Helper function to compute current mean molecular weight."""
    # You'll need to implement this based on your JAX path_integral logic
    active_gases = list(tm.chemistry.activeGases)
    active_gas_weights = np.array([MOLECULAR_WEIGHTS[g] for g in active_gases])
    
    mix_ratios = np.array([params.get(g, 0.0) for g in active_gases])
    trace_gas_mix = np.sum(mix_ratios)
    mu_from_trace = np.sum(mix_ratios * active_gas_weights)
    
    # Fill gas contribution
    He_H2_ratio = tm.chemistry._fill_ratio[0]
    remainder = 1.0 - trace_gas_mix
    main_fill_mix = remainder / (1.0 + He_H2_ratio)
    second_fill_mix = main_fill_mix * He_H2_ratio
    
    fill_gas_weights = np.array([MOLECULAR_WEIGHTS[g] for g in tm.chemistry._fill_gases])
    fill_mixes = np.array([main_fill_mix, second_fill_mix])
    mu_from_fill = np.sum(fill_mixes * fill_gas_weights)
    
    return mu_from_trace + mu_from_fill