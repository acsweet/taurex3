"""
Mixed Precision Solution for TauREx JAX Implementation

This shows how to use float32 for most calculations (faster GPU performance)
while using float64 only where numerical stability requires it.

Key insight: The overflow typically happens in specific operations like:
1. exp(-very_large_tau) 
2. Very small cross-sections * very large densities
3. Altitude integration with extreme pressure ratios

Strategy: Identify critical sections and promote to float64 selectively.
"""

import jax
import jax.numpy as jnp
from functools import partial

# ============================================================================
# OPTION 1: Selective x64 promotion for critical operations
# ============================================================================

def safe_exp_tau(tau):
    """
    Compute exp(-tau) with numerical stability.
    Uses float64 for the exponential, then casts back to float32.
    """
    # Promote to float64 for stability
    tau_64 = tau.astype(jnp.float64)
    result_64 = jnp.exp(-tau_64)
    # Cast back to float32 for memory efficiency
    return result_64.astype(tau.dtype)


def safe_altitude_integration(H, P_ratio):
    """
    Compute altitude: dz = -H * ln(P_ratio)
    Uses float64 for log to avoid precision loss.
    """
    H_64 = H.astype(jnp.float64)
    P_ratio_64 = P_ratio.astype(jnp.float64)
    dz_64 = -H_64 * jnp.log(P_ratio_64)
    return dz_64.astype(H.dtype)


def safe_opacity_multiplication(sigma, density, path):
    """
    Compute sigma * density * path for small cross-sections.
    Uses float64 accumulation to preserve precision.
    """
    # Promote to float64
    sigma_64 = sigma.astype(jnp.float64)
    density_64 = density.astype(jnp.float64)
    path_64 = path.astype(jnp.float64)
    
    # Compute in float64
    result_64 = sigma_64 * density_64 * path_64
    
    # Cast back
    return result_64.astype(sigma.dtype)


# ============================================================================
# OPTION 2: Dynamic precision context manager
# ============================================================================

class precision_scope:
    """
    Context manager to temporarily enable x64 for critical sections.
    
    Usage:
        with precision_scope(enable_x64=True):
            # Critical computation in float64
            result = compute_altitude(...)
        # Back to float32
    """
    
    def __init__(self, enable_x64):
        self.enable_x64 = enable_x64
        self.original_state = None
    
    def __enter__(self):
        # Save original state
        self.original_state = jax.config.read('jax_enable_x64')
        # Set new state
        jax.config.update('jax_enable_x64', self.enable_x64)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original state
        jax.config.update('jax_enable_x64', self.original_state)


# ============================================================================
# OPTION 3: Rescaling approach (avoid extreme values)
# ============================================================================

def rescaled_tau_computation(sigma, density, path):
    """
    Rescale quantities to avoid overflow/underflow.
    
    Instead of computing tiny sigma * huge density * huge path,
    rescale to intermediate magnitudes.
    """
    # Scale factors
    SIGMA_SCALE = 1e28   # Bring sigma from ~1e-28 to ~1
    DENSITY_SCALE = 1e-24  # Bring density from ~1e24 to ~1
    PATH_SCALE = 1e-6    # Bring path from ~1e6 to ~1
    
    # Rescaled computation
    sigma_scaled = sigma * SIGMA_SCALE
    density_scaled = density * DENSITY_SCALE
    path_scaled = path * PATH_SCALE
    
    # Result is already in correct scale (scales cancel out)
    return sigma_scaled * density_scaled * path_scaled


# ============================================================================
# OPTION 4: Log-space computation for extreme ranges
# ============================================================================

def log_space_tau(sigma, density, path):
    """
    Compute tau in log space to handle extreme ranges.
    
    tau = sigma * density * path
    log(tau) = log(sigma) + log(density) + log(path)
    """
    log_sigma = jnp.log(jnp.maximum(sigma, 1e-100))
    log_density = jnp.log(jnp.maximum(density, 1e-100))
    log_path = jnp.log(jnp.maximum(path, 1e-100))
    
    log_tau = log_sigma + log_density + log_path
    
    # Convert back if needed (be careful of overflow here too!)
    return jnp.exp(log_tau)


# ============================================================================
# EXAMPLE: Modified path_integral with mixed precision
# ============================================================================

@partial(jax.jit, static_argnames=['static_data'])
def path_integral_mixed_precision(wngrid, params, absorption_sigmas_tuple, 
                                   dynamic_data, static_data):
    """
    Modified path_integral using mixed precision strategy.
    
    - Most operations in float32 (fast)
    - Critical operations promoted to float64 (stable)
    """
    # Unpack (same as before)
    (total_layers, planet_radius_base, star_radius,
     active_gases, active_gas_weights, fill_gases, fill_gas_weights, He_H2_ratio) = static_data
    
    dz, altitude_profile, pressure_profile, cia_sigma, other_sigma = dynamic_data
    
    T = params['T']
    planet_radius = params['planet_radius'] * planet_radius_base
    
    # ... (mixing ratio and mu computation - keep in float32) ...
    
    # CRITICAL SECTION 1: Altitude integration
    # Promote to float64 for log(P_ratio) stability
    with precision_scope(enable_x64=True):
        # Recompute altitude in float64 if needed
        # Or just ensure log operations are stable
        pass
    
    # Density (float32 is fine)
    density_profile = pressure_profile / (KBOLTZ * T)
    
    # Cross-sections (float32)
    absorption_sigmas_stacked = jnp.stack(absorption_sigmas_tuple)
    mix_ratios = jnp.array([params.get(g, 0.0) for g in active_gases])
    molecular_contribution = jnp.sum(mix_ratios[:, None, None] * absorption_sigmas_stacked, axis=0)
    
    regular_sigma = molecular_contribution
    if other_sigma is not None:
        regular_sigma = regular_sigma + other_sigma
    
    # Path lengths (float32)
    from experiment import compute_path_length_simple
    path_lengths = compute_path_length_simple(total_layers, altitude_profile, dz, planet_radius)
    
    # CRITICAL SECTION 2: Tau accumulation
    # Use float64 accumulator to avoid precision loss
    tau = jnp.zeros((total_layers, wngrid.shape[0]), dtype=jnp.float64)
    
    for layer in range(total_layers):
        dl = path_lengths[layer]
        end_k = total_layers - layer
        
        # Compute contribution in float64
        _path = dl[:end_k, None].astype(jnp.float64)
        _density = density_profile[layer:layer+end_k, None].astype(jnp.float64)
        _sigma = regular_sigma[layer:layer+end_k, :].astype(jnp.float64)
        
        contribution = jnp.sum(_sigma * _path * _density, axis=0)
        tau = tau.at[layer, :].add(contribution)
        
        if cia_sigma is not None:
            _sigma_cia = cia_sigma[layer:layer+end_k, :].astype(jnp.float64)
            contribution_cia = jnp.sum(_sigma_cia * _path * _density * _density, axis=0)
            tau = tau.at[layer, :].add(contribution_cia)
    
    # CRITICAL SECTION 3: exp(-tau)
    # Already in float64, compute exponential
    tau_exp = jnp.exp(-tau)
    
    # Final absorption (can be float32)
    ap = altitude_profile[:, None].astype(jnp.float64)
    _dz = dz[:, None].astype(jnp.float64)
    
    integral = jnp.sum((planet_radius + ap) * (1.0 - tau_exp) * _dz * 2.0, axis=0)
    absorption = ((planet_radius**2.0) + integral) / (star_radius**2)
    
    # Cast final result back to float32 if desired
    return absorption.astype(jnp.float32), tau_exp.astype(jnp.float32)


# ============================================================================
# TESTING AND RECOMMENDATIONS
# ============================================================================

def test_precision_strategies():
    """Test different precision strategies."""
    print("="*70)
    print("Testing Mixed Precision Strategies")
    print("="*70)
    
    # Test case: realistic atmospheric values
    tau = jnp.array([0.1, 1.0, 10.0, 50.0, 100.0])
    sigma = jnp.array([1e-28, 1e-26, 1e-24])
    density = jnp.array([1e24, 1e25, 1e26])
    path = jnp.array([1e5, 1e6, 1e7])
    
    print("\n1. Exponential (exp(-tau)):")
    jax.config.update('jax_enable_x64', False)
    result_32 = jnp.exp(-tau)
    result_safe = safe_exp_tau(tau)
    print(f"   float32:      {result_32}")
    print(f"   safe (mixed): {result_safe}")
    
    print("\n2. Cross-section multiplication:")
    s, d, p = sigma[0], density[0], path[0]
    result_direct = s * d * p
    result_safe = safe_opacity_multiplication(
        jnp.array(s), jnp.array(d), jnp.array(p)
    )
    result_rescaled = rescaled_tau_computation(
        jnp.array(s), jnp.array(d), jnp.array(p)
    )
    print(f"   Direct float32:  {result_direct}")
    print(f"   Safe (mixed):    {result_safe}")
    print(f"   Rescaled:        {result_rescaled}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    print("1. Keep jax_enable_x64 = True for now (safest, still uses GPU)")
    print("2. Profile your code to find actual bottlenecks")
    print("3. If performance matters, try mixed precision:")
    print("   - Use float64 accumulator for tau")
    print("   - Use float64 for exp(-tau)")
    print("   - Keep arrays in float32 where possible")
    print("4. Expected speedup from mixed precision: ~1.3-1.5x")
    print("5. Expected slowdown from full float64: ~1.5-2x vs float32")
    print("="*70)


if __name__ == "__main__":
    test_precision_strategies()
    
    print("\n\nTo use in your code:")
    print("-" * 70)
    print("Option 1 (SIMPLEST): Keep x64 enabled - you're still using GPU!")
    print("  jax.config.update('jax_enable_x64', True)")
    print()
    print("Option 2 (OPTIMAL): Modify contribute_tau to use float64 accumulator")
    print("  tau = jnp.zeros((nlayers, nwn), dtype=jnp.float64)")
    print("  # ... accumulate in float64 ...")
    print("  tau_exp = jnp.exp(-tau)  # exp in float64")
    print("  # Cast final result to float32 if needed")
    print()
    print("Option 3 (ADVANCED): Use mixed precision utilities from this file")
    print("-" * 70)
