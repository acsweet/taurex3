"""
Run script for fitting atmospheric parameters using the fully differentiable JAX model.

This script uses the new full_diff_create_binned_forward_model which recomputes
all parameter-dependent quantities on each forward pass.
"""

import time
import jax.numpy as jnp

from taurex.cache import OpacityCache, CIACache
from taurex.contributions import AbsorptionContribution, CIAContribution, RayleighContribution
from taurex.data.spectrum.observed import ObservedSpectrum
from taurex.model import TransmissionModel
from taurex.planet import Planet
from taurex.stellar import BlackbodyStar
from taurex.chemistry import TaurexChemistry, ConstantGas
from taurex.temperature import Isothermal

from experiment import (
    full_diff_create_binned_forward_model,  # NEW fully differentiable version
    extract_fitting_params,
    fit_with_value_and_grad_adam,
    plot_fit_comparison,
)


def main():
    """Main execution function."""
    
    # ========== SETUP PATHS ==========
    print("Setting up data paths...")
    xsec_path = "/Users/asweet/Code/research/taurex3/test_files/Input/xsec/xsec_sampled_R15000_0.3-50"
    cia_path = "/Users/asweet/Code/research/taurex3/test_files/Input/cia/HITRAN/data"
    obs_path = "/Users/asweet/Code/research/taurex3/test_files/quickstart.dat"

    # Setup caches
    OpacityCache().clear_cache()
    OpacityCache().set_opacity_path(xsec_path)
    CIACache().set_cia_path(cia_path)

    # ========== BUILD TAUREX MODEL ==========
    print("\nBuilding TauREx model...")
    planet = Planet(planet_radius=1.0, planet_mass=1.0)
    star = BlackbodyStar(temperature=5700.0, radius=1.0)

    chemistry = TaurexChemistry(fill_gases=['H2', 'He'], ratio=0.172)
    chemistry.addGas(ConstantGas('H2O', mix_ratio=1.2e-4))
    chemistry.addGas(ConstantGas('N2', mix_ratio=3.00739e-9))

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
    tm.build()
    tm['H2O'] = 1.2e-4

    print(f"  Layers: {tm.nLayers}")
    print(f"  Active gases: {list(tm.chemistry.activeGases)}")
    print(f"  Contributions: {[c.name for c in tm.contribution_list]}")

    # ========== LOAD OBSERVATIONS ==========
    print("\nLoading observations...")
    obs = ObservedSpectrum(obs_path)
    
    # Setup observed data
    obs_y = jnp.asarray(obs.spectrum)
    if hasattr(obs, "errorBar"):
        obs_err = jnp.asarray(obs.errorBar)
    else:
        obs_err = 0.02 * jnp.maximum(jnp.abs(obs_y), 1e-12)

    print(f"  Spectral bins: {len(obs_y)}")
    print(f"  Wavelength range: {obs.wavelengthGrid.min():.2f} - {obs.wavelengthGrid.max():.2f} μm")

    # ========== CREATE FORWARD MODEL ==========
    print("\nCreating fully differentiable JAX forward model...")
    print("  (This pre-loads opacity grids and may take a moment...)")
    
    # Use the NEW fully differentiable binned forward model
    # For CPU: set pre_interpolate_opacity=True for 2-3x speedup
    # For GPU: keep pre_interpolate_opacity=False (default) for best memory bandwidth
    forward_binned = full_diff_create_binned_forward_model(
        tm, obs,
        pre_interpolate_opacity=False  # Set to True for CPU optimization
    )
    
    print("  ✓ Forward model ready!")

    # ========== EXTRACT PARAMETERS ==========
    print("\nExtracting fitting parameters...")
    params, param_info = extract_fitting_params(tm)
    
    print("Available fitting parameters:")
    for key, info in param_info.items():
        if info['scale'] == 'log':
            print(f"  {key}: {float(params[key]):.3e} (log scale, range: {info['range']})")
        else:
            print(f"  {key}: {float(params[key]):.6f} (linear scale, range: {info['range']})")

    # ========== DEFINE FITTING PARAMETERS ==========
    # Choose which parameters to fit
    fit_params = ['planet_radius', 'T', 'H2O']
    
    print(f"\nParameters to fit: {fit_params}")
    print("Initial values:")
    for k in fit_params:
        initial = float(params[k])
        print(f"  {k}: {initial:.6e}")

    # ========== RUN OPTIMIZATION ==========
    print("\n" + "="*70)
    print("Starting optimization...")
    print("="*70)
    
    start_time = time.time()
    final_params, losses = fit_with_value_and_grad_adam(
        forward_binned=forward_binned,
        observed_y=obs_y,
        observed_err=obs_err,
        init_params=params,
        param_info=param_info,
        fit_params=fit_params,
        steps=500,
        lr=1e-3,
        clip_norm=1.0,
        print_every=50,
        nan_guard=True,
        loss="mse",
    )
    
    elapsed_time = time.time() - start_time
    
    # ========== DISPLAY RESULTS ==========
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE!")
    print("="*70)
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Time per step: {elapsed_time/1000:.3f} seconds")
    
    print("\nFinal Results:")
    print("-" * 50)
    for k in fit_params:
        initial = float(params[k])
        final = float(final_params[k])
        change = final - initial
        pct_change = (change / initial) * 100 if initial != 0 else 0
        print(f"{k:20s}: {initial:.6e} -> {final:.6e}")
        print(f"{'':20s}  Change: {change:+.6e} ({pct_change:+.2f}%)")

    # ========== PLOT COMPARISON ==========
    print("\nGenerating comparison plot...")
    plot_fit_comparison(obs, forward_binned, params, final_params, fit_params)
    print("  ✓ Plot displayed!")

    # ========== VALIDATE AGAINST TAUREX ==========
    print("\n" + "="*70)
    print("Validating JAX model against TauREx...")
    print("="*70)
    
    # Update TauREx model with fitted parameters
    for k in fit_params:
        tm[k] = float(final_params[k])
    
    # Compute TauREx spectrum
    obin = obs.create_binner()
    taurex_binned = obin.bin_model(tm.model(obs.wavenumberGrid))[1]
    
    # Compute JAX spectrum
    jax_binned = forward_binned(final_params)
    
    # Compare
    import numpy as np
    abs_diff = np.abs(jax_binned - taurex_binned)
    rel_diff = abs_diff / (taurex_binned + 1e-30)
    
    print(f"\nPost-fit comparison:")
    print(f"  Max absolute difference: {np.max(abs_diff):.3e}")
    print(f"  Mean absolute difference: {np.mean(abs_diff):.3e}")
    print(f"  Max relative difference: {np.max(rel_diff):.3e}")
    print(f"  Mean relative difference: {np.mean(rel_diff):.3e}")
    
    if np.mean(rel_diff) < 0.01:
        print("\n  ✓ JAX model matches TauREx to < 1% after fitting!")
    else:
        print(f"\n  ⚠ Warning: JAX differs from TauREx by {np.mean(rel_diff)*100:.2f}%")

    print("\n" + "="*70)
    print("RUN COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
