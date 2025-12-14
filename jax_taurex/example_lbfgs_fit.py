"""
Example: Fit ADC planet using L-BFGS-B optimizer with chi-squared loss.

This demonstrates the recommended approach for gradient-based atmospheric retrieval:
1. Chi-squared loss (properly weighted by measurement uncertainties)
2. L-BFGS-B optimizer (quasi-Newton method, faster than Adam)
3. Multi-start strategy for non-convex optimization
4. Comprehensive visualization of results

Usage:
    python example_lbfgs_fit.py --planet-id 1000 --multi-start 5
"""

import sys
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from taurex.cache import OpacityCache, CIACache
from .adc_jax_utils import fit_adc_planet


def plot_comprehensive_results(result, save_path='lbfgs_fit_results.png'):
    """Create comprehensive 3-panel plot of optimization results."""
    
    spectrum_dict = result['spectrum_dict']
    forward_model = result['forward_model']
    initial_params = result['initial_params']
    final_params = result['final_params']
    ground_truth = result['ground_truth']
    losses = result['losses']
    all_results = result.get('all_results', None)
    
    # Get spectra
    obs_y = spectrum_dict['spectrum']
    obs_err = spectrum_dict['noise']
    wl_grid = spectrum_dict['wl_grid']
    
    initial_spectrum = np.array(forward_model(initial_params))
    final_spectrum = np.array(forward_model(final_params))
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # ========== PANEL 1: Spectrum Comparison ==========
    ax1 = fig.add_subplot(gs[0, :])
    
    ax1.errorbar(wl_grid, obs_y, yerr=obs_err, fmt='o', alpha=0.6, 
                 label='Observed', markersize=6, color='black', capsize=3)
    ax1.plot(wl_grid, initial_spectrum, '--', alpha=0.8, 
             label='Initial (posterior mean)', linewidth=2.5, color='blue')
    ax1.plot(wl_grid, final_spectrum, '-', 
             label='Optimized fit', linewidth=2.5, color='red')
    
    ax1.set_xlabel('Wavelength (μm)', fontsize=12)
    ax1.set_ylabel('Transit Depth', fontsize=12)
    ax1.set_title('Atmospheric Transmission Spectrum', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Calculate and display chi-squared
    initial_chi2 = np.sum(((obs_y - initial_spectrum) / obs_err)**2) / len(obs_y)
    final_chi2 = np.sum(((obs_y - final_spectrum) / obs_err)**2) / len(obs_y)
    
    ax1.text(0.02, 0.98, f'Initial χ²/dof = {initial_chi2:.2f}\nFinal χ²/dof = {final_chi2:.2f}',
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ========== PANEL 2: Residuals ==========
    ax2 = fig.add_subplot(gs[1, :])
    
    initial_residuals = (obs_y - initial_spectrum) / obs_err
    final_residuals = (obs_y - final_spectrum) / obs_err
    
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(1, color='gray', linestyle=':', alpha=0.3)
    ax2.axhline(-1, color='gray', linestyle=':', alpha=0.3)
    
    ax2.scatter(wl_grid, initial_residuals, alpha=0.6, s=80, 
                label='Initial residuals', color='blue')
    ax2.scatter(wl_grid, final_residuals, alpha=0.6, s=80, 
                label='Final residuals', color='red')
    
    ax2.set_xlabel('Wavelength (μm)', fontsize=12)
    ax2.set_ylabel('Residuals (σ)', fontsize=12)
    ax2.set_title('Fit Residuals (normalized by uncertainty)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Display RMS residuals
    rms_initial = np.sqrt(np.mean(initial_residuals**2))
    rms_final = np.sqrt(np.mean(final_residuals**2))
    ax2.text(0.02, 0.98, f'RMS initial = {rms_initial:.2f}σ\nRMS final = {rms_final:.2f}σ',
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # ========== PANEL 3: Parameter Comparison ==========
    ax3 = fig.add_subplot(gs[2, 0])
    
    # Get fitted parameters
    fit_params = ['planet_radius', 'T', 'H2O', 'CO2', 'CO', 'CH4', 'NH3']
    param_names = []
    initial_vals = []
    final_vals = []
    posterior_vals = []
    
    for param in fit_params:
        if param in final_params:
            param_names.append(param)
            initial_vals.append(float(initial_params[param]))
            final_vals.append(float(final_params[param]))
            if ground_truth and param in ground_truth:
                posterior_vals.append(float(ground_truth[param]))
            else:
                posterior_vals.append(np.nan)
    
    x = np.arange(len(param_names))
    width = 0.25
    
    ax3.bar(x - width, initial_vals, width, label='Initial', alpha=0.7, color='blue')
    ax3.bar(x, final_vals, width, label='Optimized', alpha=0.7, color='red')
    if not np.all(np.isnan(posterior_vals)):
        ax3.bar(x + width, posterior_vals, width, label='Posterior', alpha=0.7, color='green')
    
    ax3.set_xlabel('Parameter', fontsize=12)
    ax3.set_ylabel('Value', fontsize=12)
    ax3.set_title('Parameter Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(param_names, rotation=45, ha='right')
    ax3.legend(fontsize=10)
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ========== PANEL 4: Multi-start Results or Parameter Errors ==========
    ax4 = fig.add_subplot(gs[2, 1])
    
    if all_results and len(all_results) > 1:
        # Show multi-start results
        losses_multi = [loss for _, loss in all_results]
        ax4.bar(range(len(losses_multi)), losses_multi, alpha=0.7, color='purple')
        ax4.axhline(min(losses_multi), color='red', linestyle='--', 
                    label=f'Best: {min(losses_multi):.2e}')
        ax4.set_xlabel('Run Number', fontsize=12)
        ax4.set_ylabel('Final Loss', fontsize=12)
        ax4.set_title(f'Multi-Start Results ({len(all_results)} runs)', 
                      fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
    elif ground_truth:
        # Show parameter errors relative to posterior
        errors = []
        for param in param_names:
            if param in ground_truth:
                gt_val = float(ground_truth[param])
                final_val = float(final_params[param])
                error = abs(final_val - gt_val) / gt_val * 100 if gt_val != 0 else 0
                errors.append(error)
            else:
                errors.append(0)
        
        colors = ['green' if e < 10 else 'orange' if e < 50 else 'red' for e in errors]
        ax4.bar(range(len(errors)), errors, alpha=0.7, color=colors)
        ax4.set_xlabel('Parameter', fontsize=12)
        ax4.set_ylabel('Relative Error (%)', fontsize=12)
        ax4.set_title('Parameter Errors vs Posterior', fontsize=14, fontweight='bold')
        ax4.set_xticks(range(len(param_names)))
        ax4.set_xticklabels(param_names, rotation=45, ha='right')
        ax4.axhline(10, color='orange', linestyle='--', alpha=0.5, label='10%')
        ax4.axhline(50, color='red', linestyle='--', alpha=0.5, label='50%')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'L-BFGS-B Atmospheric Retrieval Results', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comprehensive plot to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Fit ADC planet with L-BFGS-B')
    parser.add_argument('--planet-id', type=int, default=1000, help='Planet ID')
    parser.add_argument('--steps', type=int, default=1000, help='Max iterations')
    parser.add_argument('--multi-start', type=int, default=1, 
                        help='Number of random initializations (1=single run)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--float32', action='store_true', help='Use float32 precision')
    parser.add_argument('--optimizer', type=str, default='lbfgs', 
                        choices=['adam', 'lbfgs'], help='Optimizer to use')
    args = parser.parse_args()
    
    # ========== CONFIGURATION ==========
    PLANET_ID = args.planet_id
    USE_FLOAT64 = not args.float32
    STEPS = args.steps
    MULTI_START = args.multi_start
    OPTIMIZER = args.optimizer
    
    FIT_PARAMS = ['planet_radius', 'T', 'H2O', 'CO2', 'CO', 'CH4', 'NH3']
    DATA_DIR = 'test_files/adc_2023/TrainingData'
    XSEC_PATH = 'test_files/xsec/xsec_sampled_R15000_0.3-50'
    CIA_PATH = 'test_files/cia/HITRAN/data'
    
    # ========== SETUP JAX ==========
    jax.config.update("jax_enable_x64", USE_FLOAT64)
    DTYPE = jnp.float64 if USE_FLOAT64 else jnp.float32
    
    print("=" * 70)
    print("GRADIENT-BASED ATMOSPHERIC RETRIEVAL")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Planet ID: {PLANET_ID}")
    print(f"  Optimizer: {OPTIMIZER.upper()}")
    print(f"  Loss function: Chi-squared (weighted by uncertainties)")
    print(f"  Precision: {'float64' if USE_FLOAT64 else 'float32'}")
    print(f"  Device: {jax.devices()[0]}")
    if OPTIMIZER == 'lbfgs':
        print(f"  Max iterations: {STEPS}")
        print(f"  Multi-start runs: {MULTI_START}")
    else:
        print(f"  Steps: {STEPS}")
    print("=" * 70)
    
    # ========== SETUP OPACITY CACHE ==========
    print("\nSetting up opacity cache...")
    OpacityCache().clear_cache()
    OpacityCache().set_opacity_path(XSEC_PATH)
    CIACache().set_cia_path(CIA_PATH)
    print("  ✓ Ready")
    
    # ========== RUN FITTING ==========
    result = fit_adc_planet(
        planet_id=PLANET_ID,
        data_dir=DATA_DIR,
        fit_params=FIT_PARAMS,
        steps=STEPS,
        nlayers=30,
        dtype=DTYPE,
        verbose=True,
        optimizer=OPTIMIZER,
        multi_start=MULTI_START,
        random_seed=args.seed,
        loss='chi_squared',
        l2_reg=0.0,  # No regularization
        log_prior=None,  # No prior constraints
        lr=1e-3 if OPTIMIZER == 'adam' else None,
        clip_norm=1.0 if OPTIMIZER == 'adam' else None
    )
    
    # ========== PLOT RESULTS ==========
    print("\n" + "=" * 70)
    print("Generating comprehensive visualization...")
    print("=" * 70)
    
    plot_comprehensive_results(
        result,
        save_path=f'{OPTIMIZER}_planet_{PLANET_ID}_results.png'
    )
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 70)
    print(f"Planet: {PLANET_ID}")
    print(f"Optimizer: {OPTIMIZER.upper()}")
    
    if result['losses']:
        print(f"Initial loss: {result['losses'][0]:.6e}")
        print(f"Final loss: {result['losses'][-1]:.6e}")
        if result['losses'][0] > 0:
            improvement = result['losses'][0] / result['losses'][-1]
            print(f"Improvement: {improvement:.1f}x better")
    
    if result.get('all_results') and len(result['all_results']) > 1:
        best_loss = min(loss for _, loss in result['all_results'])
        worst_loss = max(loss for _, loss in result['all_results'])
        print(f"\nMulti-start summary:")
        print(f"  Best loss: {best_loss:.6e}")
        print(f"  Worst loss: {worst_loss:.6e}")
        print(f"  Range: {worst_loss/best_loss:.2f}x")
    
    if result['ground_truth']:
        print(f"\nParameter comparison to posterior:")
        for param in FIT_PARAMS:
            if param in result['ground_truth']:
                gt_val = float(result['ground_truth'][param])
                final_val = float(result['final_params'][param])
                error = abs(final_val - gt_val) / gt_val * 100 if gt_val != 0 else 0
                status = '✓' if error < 10 else '⚠' if error < 50 else '✗'
                print(f"  {param:15s}: {error:6.2f}% {status}")
    
    print("=" * 70)
    print("\n✓ Success!\n")


if __name__ == "__main__":
    main()
