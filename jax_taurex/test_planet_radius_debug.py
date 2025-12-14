"""
Debug the planet radius parameter.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from taurex.cache import OpacityCache, CIACache
from taurex.constants import RJUP, RSOL

from .adc_jax_utils import (
    load_auxiliary_data,
    load_ground_truth,
    create_taurex_model_from_adc_planet,
)
from .jax_backend import extract_fitting_params


def main():
    """Check what the planet_radius parameter actually is."""
    OpacityCache().clear_cache()
    OpacityCache().set_opacity_path('test_files/xsec/xsec_sampled_R15000_0.3-50')
    CIACache().set_cia_path('test_files/cia/HITRAN/data')
    
    planet_id = 1000
    aux_data = load_auxiliary_data('test_files/adc_2023/TrainingData/AuxillaryTable.csv', planet_id)
    gt = load_ground_truth('test_files/adc_2023/TrainingData/Ground Truth Package/FM_Parameter_Table.csv', planet_id)
    
    tm = create_taurex_model_from_adc_planet(aux_data, gt, nlayers=30)
    tm.initialize_profiles()
    
    print("="*80)
    print("PLANET RADIUS PARAMETER DEBUG")
    print("="*80)
    
    print(f"\n1. TauREx Model:")
    print(f"   Planet radius (fullRadius): {tm.planet.fullRadius:.6e} m")
    print(f"   Planet radius (fullRadius): {tm.planet.fullRadius/RJUP:.6f} R_jup")
    
    print(f"\n2. Extracted Parameters:")
    params, param_info = extract_fitting_params(tm, dtype=jnp.float64)
    
    print(f"   All parameters: {list(params.keys())}")
    print(f"   planet_radius value: {float(params['planet_radius']):.6f}")
    
    if 'planet_radius' in param_info:
        print(f"   planet_radius range: {param_info['planet_radius']['range']}")
        print(f"   planet_radius scale: {param_info['planet_radius']['scale']}")
    
    print(f"\n3. Interpretation:")
    if float(params['planet_radius']) < 2.0:
        print(f"   ✓ planet_radius = {float(params['planet_radius']):.4f} appears to be a SCALING FACTOR")
        print(f"     (close to 1.0, meaning it scales the base radius)")
    else:
        print(f"   ✗ planet_radius = {float(params['planet_radius']):.4e} appears to be ABSOLUTE")
        print(f"     (large value, meaning it's in meters or Jupiter radii)")
    
    print(f"\n4. Check Fitting Parameters:")
    for key, val in tm._fitting_parameters.items():
        if 'radius' in key.lower():
            print(f"   {key}:")
            print(f"     Current value: {val[2]()}")
            print(f"     Range: {val[-1]}")
            print(f"     Scale: {val[4]}")


if __name__ == "__main__":
    main()
