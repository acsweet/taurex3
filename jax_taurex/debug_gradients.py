import jax
import jax.numpy as jnp
import numpy as np
from taurex.model import TransmissionModel
from taurex.planet import Planet
from taurex.stellar import BlackbodyStar
from taurex.chemistry import TaurexChemistry, ConstantGas
from taurex.temperature import Isothermal
from taurex.contributions import AbsorptionContribution, CIAContribution, RayleighContribution, SimpleCloudsContribution
from taurex.cache import OpacityCache, CIACache

from .jax_backend import (
    full_diff_create_forward_model,
    extract_fitting_params,
    full_diff_compute_altitude_profile,
    full_diff_compute_temperature_profile,
    full_diff_compute_mixing_profiles,
    full_diff_compute_mu_profile,
    full_diff_prepare_model_data,
    RJUP, KBOLTZ
)

# Enable float64
jax.config.update("jax_enable_x64", True)
DTYPE = jnp.float64

def setup_model():
    # Setup paths (assuming running from workspace root)
    xsec_path = "test_files/xsec/xsec_sampled_R15000_0.3-50"
    cia_path = "test_files/cia/HITRAN/data"
    
    OpacityCache().clear_cache()
    OpacityCache().set_opacity_path(xsec_path)
    CIACache().set_cia_path(cia_path)

    planet = Planet(planet_radius=1.0, planet_mass=1.0)
    star = BlackbodyStar(temperature=5700.0, radius=1.0)
    chemistry = TaurexChemistry(fill_gases=['H2', 'He'], ratio=0.172)
    chemistry.addGas(ConstantGas('H2O', mix_ratio=1.2e-4))
    isothermal = Isothermal(T=1500.0)

    tm = TransmissionModel(
        planet=planet,
        temperature_profile=isothermal,
        chemistry=chemistry,
        star=star,
        atm_min_pressure=1e-0,
        atm_max_pressure=1e6,
        nlayers=30
    )
    tm.add_contribution(AbsorptionContribution())
    tm.add_contribution(CIAContribution(cia_pairs=['H2-H2', 'H2-He']))
    tm.add_contribution(RayleighContribution())
    tm.add_contribution(SimpleCloudsContribution(clouds_pressure=1e3))
    tm.build()
    
    return tm

def debug_gradients():
    tm = setup_model()
    wngrid = np.linspace(1000, 10000, 100) # Small grid for debugging
    
    # Create forward model
    forward_model = full_diff_create_forward_model(tm, wngrid)
    
    # Extract params
    params, _ = extract_fitting_params(tm, dtype=DTYPE)
    
    # Define fit params
    fit_keys = ['planet_radius', 'T', 'H2O', 'clouds_pressure']
    
    print("Initial params:")
    for k in fit_keys:
        print(f"  {k}: {params[k]}")

    # 1. Test Gradient of Full Forward Model
    print("\n--- Testing Full Forward Model Gradient ---")
    def loss_fn(p_vals):
        # Reconstruct params dict
        p = params.copy()
        for i, k in enumerate(fit_keys):
            p[k] = p_vals[i]
        
        abs_spec, _ = forward_model(p)
        return jnp.sum(abs_spec)

    p_vals_init = jnp.array([params[k] for k in fit_keys])
    
    val, grad = jax.value_and_grad(loss_fn)(p_vals_init)
    print(f"Loss: {val}")
    print(f"Gradients: {grad}")
    
    if jnp.any(jnp.isnan(grad)):
        print("!!! NAN DETECTED IN FULL GRADIENT !!!")
    else:
        print("Full gradient is fine.")
        return

    # 2. Test Intermediate Components
    print("\n--- Debugging Intermediate Components ---")
    
    static_data, _ = full_diff_prepare_model_data(tm, dtype=DTYPE)
    nlayers = static_data['nlayers']
    pressure_levels = static_data['pressure_levels']
    planet_mass = static_data['planet_mass']
    
    # Test Altitude Profile Gradient
    print("\nChecking Altitude Profile Gradient...")
    def alt_loss(p_vals):
        p = params.copy()
        for i, k in enumerate(fit_keys):
            p[k] = p_vals[i]
            
        T_profile = full_diff_compute_temperature_profile(p['T'], nlayers)
        
        # Mock mu profile for simplicity or compute it
        # mixing_profiles = full_diff_compute_mixing_profiles(p, static_data['active_gases'], nlayers)
        # mu_profile = full_diff_compute_mu_profile(...)
        # Let's just use a constant mu for this test to isolate T and Radius
        mu_profile = jnp.ones(nlayers) * 2.3 * 1.66e-27 
        
        planet_radius = p['planet_radius'] * RJUP
        
        alt, scale_H, g, dz = full_diff_compute_altitude_profile(
            T_profile, pressure_levels, mu_profile, planet_mass, planet_radius
        )
        return jnp.sum(alt) + jnp.sum(scale_H) + jnp.sum(g) + jnp.sum(dz)

    val, grad = jax.value_and_grad(alt_loss)(p_vals_init)
    print(f"Altitude Loss: {val}")
    print(f"Altitude Gradients: {grad}")
    
    # Test Cloud Opacity Gradient
    print("\nChecking Cloud Opacity Gradient...")
    def cloud_loss(p_vals):
        p = params.copy()
        for i, k in enumerate(fit_keys):
            p[k] = p_vals[i]
            
        clouds_pressure = p['clouds_pressure']
        pressure_profile = static_data['pressure_profile']
        
        # Cloud logic from experiment.py
        safe_pressure = jnp.maximum(pressure_profile, 1e-20)
        safe_cloud_pressure = jnp.maximum(clouds_pressure, 1e-20)
        
        log_P = jnp.log(safe_pressure)
        log_P_cloud = jnp.log(safe_cloud_pressure)
        
        k = 10.0 
        cloud_opacity_factor = jax.nn.sigmoid(k * (log_P - log_P_cloud))
        
        return jnp.sum(cloud_opacity_factor)

    val, grad = jax.value_and_grad(cloud_loss)(p_vals_init)
    print(f"Cloud Loss: {val}")
    print(f"Cloud Gradients: {grad}")

    # Test Mu Profile Gradient
    print("\nChecking Mu Profile Gradient...")
    def mu_loss(p_vals):
        p = params.copy()
        for i, k in enumerate(fit_keys):
            p[k] = p_vals[i]
            
        mixing_profiles = full_diff_compute_mixing_profiles(p, static_data['active_gases'], nlayers)
        
        mu_profile = full_diff_compute_mu_profile(
            mixing_profiles,
            static_data['active_gases'],
            (18.015,), # H2O weight
            static_data['fill_gases'],
            (2.016, 4.0026), # H2, He weights
            static_data['He_H2_ratio']
        )
        return jnp.sum(mu_profile)

    val, grad = jax.value_and_grad(mu_loss)(p_vals_init)
    print(f"Mu Loss: {val}")
    print(f"Mu Gradients: {grad}")

    # Test Opacity Interpolation Gradient
    print("\nChecking Opacity Interpolation Gradient...")
    # Load opacity data manually for test
    from .jax_backend import load_opacity_data, full_diff_interpolate_opacity_all_layers
    opacity_data = load_opacity_data(['H2O'], wngrid, dtype=DTYPE)
    
    def opacity_loss(p_vals):
        p = params.copy()
        for i, k in enumerate(fit_keys):
            p[k] = p_vals[i]
            
        T_profile = full_diff_compute_temperature_profile(p['T'], nlayers)
        # Use constant pressure for simplicity to isolate T gradient
        P_profile = static_data['pressure_profile']
        
        sigma = full_diff_interpolate_opacity_all_layers(
            T_profile, P_profile, opacity_data['H2O']
        )
        return jnp.sum(sigma)

    val, grad = jax.value_and_grad(opacity_loss)(p_vals_init)
    print(f"Opacity Loss: {val}")
    print(f"Opacity Gradients: {grad}")

    # Test Path Length Gradient
    print("\nChecking Path Length Gradient...")
    from .jax_backend import compute_path_length_simple
    
    def path_length_loss(p_vals):
        p = params.copy()
        for i, k in enumerate(fit_keys):
            p[k] = p_vals[i]
            
        planet_radius = p['planet_radius'] * RJUP
        
        # Recompute altitude profile as it depends on T and Radius
        T_profile = full_diff_compute_temperature_profile(p['T'], nlayers)
        
        # Mock mu profile for simplicity (constant)
        mu_profile = jnp.ones(nlayers) * 2.3 * 1.66e-27 
        
        altitude_boundaries, _, _, deltaz = full_diff_compute_altitude_profile(
            T_profile, pressure_levels, mu_profile, planet_mass, planet_radius
        )
        
        # Compute altitude at layer centers
        altitude_profile = 0.5 * (altitude_boundaries[:-1] + altitude_boundaries[1:])
        
        path_lengths = compute_path_length_simple(
            nlayers, altitude_profile, deltaz, planet_radius
        )
        # Sum of all path lengths
        total_length = 0.0
        for pl in path_lengths:
            total_length = total_length + jnp.sum(pl)
        return total_length

    val, grad = jax.value_and_grad(path_length_loss)(p_vals_init)
    print(f"Path Length Loss: {val}")
    print(f"Path Length Gradients: {grad}")

    # Test Absorption Gradient
    print("\nChecking Absorption Gradient...")
    from .jax_backend import compute_absorption
    
    def absorption_loss(p_vals):
        p = params.copy()
        for i, k in enumerate(fit_keys):
            p[k] = p_vals[i]
            
        planet_radius = p['planet_radius'] * RJUP
        star_radius = static_data['star_radius']
        
        # Recompute altitude profile
        T_profile = full_diff_compute_temperature_profile(p['T'], nlayers)
        mu_profile = jnp.ones(nlayers) * 2.3 * 1.66e-27 
        
        altitude_boundaries, _, _, deltaz = full_diff_compute_altitude_profile(
            T_profile, pressure_levels, mu_profile, planet_mass, planet_radius
        )
        
        # Compute altitude at layer centers
        altitude_profile = 0.5 * (altitude_boundaries[:-1] + altitude_boundaries[1:])
        
        # Mock tau (finite, positive)
        tau = jnp.ones((nlayers, len(wngrid))) * 0.1
        
        absorption, _ = compute_absorption(
            tau, deltaz, altitude_profile, planet_radius, star_radius
        )
        return jnp.sum(absorption)

    val, grad = jax.value_and_grad(absorption_loss)(p_vals_init)
    print(f"Absorption Loss: {val}")
    print(f"Absorption Gradients: {grad}")

    # Test Tau Accumulation Gradient
    print("\nChecking Tau Accumulation Gradient...")
    from .jax_backend import contribute_tau
    
    def tau_loss(p_vals):
        p = params.copy()
        for i, k in enumerate(fit_keys):
            p[k] = p_vals[i]
            
        planet_radius = p['planet_radius'] * RJUP
        
        # Recompute altitude profile
        T_profile = full_diff_compute_temperature_profile(p['T'], nlayers)
        mu_profile = jnp.ones(nlayers) * 2.3 * 1.66e-27 
        
        altitude_boundaries, _, _, deltaz = full_diff_compute_altitude_profile(
            T_profile, pressure_levels, mu_profile, planet_mass, planet_radius
        )
        altitude_profile = 0.5 * (altitude_boundaries[:-1] + altitude_boundaries[1:])
        
        path_lengths = compute_path_length_simple(
            nlayers, altitude_profile, deltaz, planet_radius
        )
        
        # Density
        pressure_profile = static_data['pressure_profile']
        density_profile = pressure_profile / (KBOLTZ * p['T']) # Isothermal T
        
        # Opacity (mock)
        sigma = jnp.ones((nlayers, len(wngrid))) * 1e-20
        
        tau = jnp.zeros((nlayers, len(wngrid)))
        
        for layer in range(nlayers):
            dl = path_lengths[layer]
            end_k = nlayers - layer
            
            tau = contribute_tau(
                0, end_k, layer, sigma, density_profile, dl, layer, tau
            )
            
        return jnp.sum(tau)

    val, grad = jax.value_and_grad(tau_loss)(p_vals_init)
    print(f"Tau Loss: {val}")
    print(f"Tau Gradients: {grad}")

    # Test Tau Accumulation with Real Opacity
    print("\nChecking Tau Accumulation with Real Opacity Gradient...")
    
    def tau_real_opacity_loss(p_vals):
        p = params.copy()
        for i, k in enumerate(fit_keys):
            p[k] = p_vals[i]
            
        planet_radius = p['planet_radius'] * RJUP
        
        # Recompute altitude profile
        T_profile = full_diff_compute_temperature_profile(p['T'], nlayers)
        mu_profile = jnp.ones(nlayers) * 2.3 * 1.66e-27 
        
        altitude_boundaries, _, _, deltaz = full_diff_compute_altitude_profile(
            T_profile, pressure_levels, mu_profile, planet_mass, planet_radius
        )
        altitude_profile = 0.5 * (altitude_boundaries[:-1] + altitude_boundaries[1:])
        
        path_lengths = compute_path_length_simple(
            nlayers, altitude_profile, deltaz, planet_radius
        )
        
        # Density
        pressure_profile = static_data['pressure_profile']
        density_profile = pressure_profile / (KBOLTZ * p['T'])
        
        # Real Opacity
        sigma = full_diff_interpolate_opacity_all_layers(
            T_profile, pressure_profile, opacity_data['H2O']
        )
        
        # Mixing ratio
        mix_ratio = p['H2O']
        sigma_weighted = sigma * mix_ratio
        
        tau = jnp.zeros((nlayers, len(wngrid)))
        
        for layer in range(nlayers):
            dl = path_lengths[layer]
            end_k = nlayers - layer
            
            tau = contribute_tau(
                0, end_k, layer, sigma_weighted, density_profile, dl, layer, tau
            )
            
        return jnp.sum(tau)

    val, grad = jax.value_and_grad(tau_real_opacity_loss)(p_vals_init)
    print(f"Tau Real Opacity Loss: {val}")
    print(f"Tau Real Opacity Gradients: {grad}")

    # Test CIA Gradient
    print("\nChecking CIA Gradient...")
    from .jax_backend import contribute_tau_cia, prepare_contributions
    
    # Prepare CIA sigma (static)
    _, cia_sigma, _ = prepare_contributions(tm, wngrid)
    
    def tau_cia_loss(p_vals):
        p = params.copy()
        for i, k in enumerate(fit_keys):
            p[k] = p_vals[i]
            
        planet_radius = p['planet_radius'] * RJUP
        
        # Recompute altitude profile
        T_profile = full_diff_compute_temperature_profile(p['T'], nlayers)
        mu_profile = jnp.ones(nlayers) * 2.3 * 1.66e-27 
        
        altitude_boundaries, _, _, deltaz = full_diff_compute_altitude_profile(
            T_profile, pressure_levels, mu_profile, planet_mass, planet_radius
        )
        altitude_profile = 0.5 * (altitude_boundaries[:-1] + altitude_boundaries[1:])
        
        path_lengths = compute_path_length_simple(
            nlayers, altitude_profile, deltaz, planet_radius
        )
        
        # Density
        pressure_profile = static_data['pressure_profile']
        density_profile = pressure_profile / (KBOLTZ * p['T'])
        
        tau = jnp.zeros((nlayers, len(wngrid)))
        
        for layer in range(nlayers):
            dl = path_lengths[layer]
            end_k = nlayers - layer
            
            if cia_sigma is not None:
                tau = contribute_tau_cia(
                    0, end_k, layer, cia_sigma, density_profile, dl, layer, tau
                )
            
        return jnp.sum(tau)

    val, grad = jax.value_and_grad(tau_cia_loss)(p_vals_init)
    print(f"CIA Loss: {val}")
    print(f"CIA Gradients: {grad}")

    # Test Rayleigh Gradient
    print("\nChecking Rayleigh Gradient...")
    
    print(f"Active gases: {tm.chemistry.activeGases}")
    print(f"Inactive gases: {tm.chemistry.inactiveGases}")
    print(f"Fill gases: {tm.chemistry._fill_gases}")
    
    from taurex.util.scattering import rayleigh_sigma_from_name
    for gas in ['H2', 'He', 'H2O']:
        sigma = rayleigh_sigma_from_name(gas, wngrid)
        if sigma is not None:
            print(f"Gas {gas} sigma range: {np.min(sigma)} - {np.max(sigma)}")
            if np.any(np.isinf(sigma)):
                print(f"!!! INF in {gas} !!!")
        
        mix = tm.chemistry.get_gas_mix_profile(gas)
        print(f"Gas {gas} mix range: {np.min(mix)} - {np.max(mix)}")
        if np.any(np.isinf(mix)):
            print(f"!!! INF in {gas} mix !!!")
    
    # Prepare Rayleigh sigma (static)
    _, _, other_sigma = prepare_contributions(tm, wngrid)
    
    if other_sigma is not None:
        print(f"Rayleigh sigma range: {jnp.min(other_sigma)} - {jnp.max(other_sigma)}")
        if jnp.any(jnp.isinf(other_sigma)):
            print("!!! INF DETECTED IN RAYLEIGH SIGMA !!!")
            # Find where it is inf
            inf_indices = jnp.where(jnp.isinf(other_sigma))
            print(f"Inf indices (layer, wn): {inf_indices}")
            
        if jnp.any(jnp.isnan(other_sigma)):
            print("!!! NAN DETECTED IN RAYLEIGH SIGMA !!!")
    
    def tau_rayleigh_loss(p_vals):
        p = params.copy()
        for i, k in enumerate(fit_keys):
            p[k] = p_vals[i]
            
        planet_radius = p['planet_radius'] * RJUP
        
        # Recompute altitude profile
        T_profile = full_diff_compute_temperature_profile(p['T'], nlayers)
        mu_profile = jnp.ones(nlayers) * 2.3 * 1.66e-27 
        
        altitude_boundaries, _, _, deltaz = full_diff_compute_altitude_profile(
            T_profile, pressure_levels, mu_profile, planet_mass, planet_radius
        )
        altitude_profile = 0.5 * (altitude_boundaries[:-1] + altitude_boundaries[1:])
        
        path_lengths = compute_path_length_simple(
            nlayers, altitude_profile, deltaz, planet_radius
        )
        
        # Density
        pressure_profile = static_data['pressure_profile']
        density_profile = pressure_profile / (KBOLTZ * p['T'])
        
        # Debug prints (using jax.debug.print or just checking for infs here if possible, 
        # but inside JIT/grad we can't easily print. 
        # However, since we are running value_and_grad, we can check values if we return them, 
        # but we need scalar output for grad.
        # Let's use a side-effect print if not jitted, or just assume we can print in this test script 
        # since it's not explicitly jitted (jax.value_and_grad jits by default? No, only if we ask).
        # value_and_grad does NOT jit by default.
        
        if jnp.any(jnp.isinf(path_lengths[0])):
             print("!!! INF in path_lengths !!!")
        if jnp.any(jnp.isinf(density_profile)):
             print("!!! INF in density_profile !!!")
        
        tau = jnp.zeros((nlayers, len(wngrid)))
        
        for layer in range(nlayers):
            dl = path_lengths[layer]
            end_k = nlayers - layer
            
            if other_sigma is not None:
                tau = contribute_tau(
                    0, end_k, layer, other_sigma, density_profile, dl, layer, tau
                )
            
        return jnp.sum(tau)

    val, grad = jax.value_and_grad(tau_rayleigh_loss)(p_vals_init)
    print(f"Rayleigh Loss: {val}")
    print(f"Rayleigh Gradients: {grad}")

if __name__ == "__main__":
    debug_gradients()
