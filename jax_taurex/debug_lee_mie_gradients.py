
import jax
import jax.numpy as jnp
import numpy as np
from .jax_backend import create_forward_model, extract_fitting_params
from .adc_jax_utils import create_taurex_model_from_adc_planet, load_auxiliary_data

def debug_lee_mie():
    print("Setting up Lee Mie debug...")
    
    # Load data for planet 1000
    planet_id = 1000
    aux_path = 'test_files/adc_2023/TrainingData/AuxillaryTable.csv'
    aux_data = load_auxiliary_data(aux_path, planet_id)
    
    # Create model with Lee Mie
    tm = create_taurex_model_from_adc_planet(
        aux_data,
        nlayers=30,
        add_lee_mie=True
    )
    
    # Create forward model
    wngrid = np.linspace(1000, 10000, 100) # Dummy grid
    forward = create_forward_model(tm, wngrid)
    
    # Get params
    params, param_info = extract_fitting_params(tm, dtype=jnp.float64)
    
    # Add Lee Mie params if not present (extract_fitting_params might miss them if not in fitting_parameters list of taurex model, 
    # but create_taurex_model_from_adc_planet adds them to contribution, so they should be there if we set them as fit params in taurex)
    # Wait, create_taurex_model_from_adc_planet in adc_jax_utils.py adds the contribution but doesn't explicitly set them as fitting parameters in the TauREx object unless we do something.
    # Actually, extract_fitting_params iterates over taurex_model.fittingParameters.
    # In adc_jax_utils.py, we just add the contribution. We rely on TauREx to register them.
    # Let's check if they are in params.
    
    print("Params:", params.keys())
    
    # Define loss function
    def loss_fn(p):
        absorption, tau = forward(p)
        return jnp.sum(absorption)
    
    # Compute gradients
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(params)
    
    print("\nGradients:")
    for k in ['lee_mie_radius', 'lee_mie_q', 'lee_mie_mix_ratio', 'lee_mie_bottomP', 'lee_mie_topP']:
        if k in grads:
            print(f"{k}: {grads[k]}")
        else:
            print(f"{k}: NOT IN PARAMS")

if __name__ == "__main__":
    debug_lee_mie()
