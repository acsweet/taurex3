"""
ADC 2023 Experiment: Validate JAX-based atmospheric retrieval against ground truth.

This script:
1. Loads ADC 2023 planet data
2. Runs JAX-based retrieval fitting
3. Compares fitted parameters to ground truth
4. Generates plots and statistics

Usage:
    python experiment_adc_2023.py [--planet-id 1000] [--steps 500] [--float32]
"""

import argparse
import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from adc_jax_utils import fit_adc_planet, load_ground_truth


def compare_to_ground_truth(result_dict):
    """
    Compare fitted parameters to ground truth.
    
    Args:
        result_dict: Output from fit_adc_planet()
    
    Returns:
        dict with comparison statistics
    """
    ground_truth = result_dict['ground_truth']
    final_params = result_dict['final_params']
    initial_params = result_dict['initial_params']
    
    if ground_truth is None:
        print("No ground truth available for comparison")
        return None
    
    # Map parameter names to ground truth keys
    param_mapping = {
        'planet_radius': 'planet_radius',
        'T': 'planet_temp',
        'H2O': 'H2O',
        'CO2': 'CO2',
        'CO': 'CO',
        'CH4': 'CH4',
        'NH3': 'NH3'
    }
    
    comparison = {}
    
    print("\n" + "="*80)
    print("PARAMETER COMPARISON: FITTED vs GROUND TRUTH")
    print("="*80)
    print(f"\n{'Parameter':<15} {'Ground Truth':<15} {'Initial':<15} {'Fitted':<15} {'Error %':<10}")
    print("-"*80)
    
    for param, gt_key in param_mapping.items():
        if param in final_params and gt_key in ground_truth:
            gt_val = float(ground_truth[gt_key])
            init_val = float(initial_params[param])
            fit_val = float(final_params[param])
            
            error_pct = abs(fit_val - gt_val) / abs(gt_val) * 100
            initial_error_pct = abs(init_val - gt_val) / abs(gt_val) * 100
            
            comparison[param] = {
                'ground_truth': gt_val,
                'initial': init_val,
                'fitted': fit_val,
                'error_pct': error_pct,
                'initial_error_pct': initial_error_pct,
                'improved': error_pct < initial_error_pct
            }
            
            # Format output based on magnitude
            if abs(gt_val) > 1:
                gt_str = f"{gt_val:>14.6f}"
                init_str = f"{init_val:>14.6f}"
                fit_str = f"{fit_val:>14.6f}"
            else:
                gt_str = f"{gt_val:>14.6e}"
                init_str = f"{init_val:>14.6e}"
                fit_str = f"{fit_val:>14.6e}"
            
            improvement = "✓" if error_pct < initial_error_pct else "✗"
            print(f"{param:<15} {gt_str} {init_str} {fit_str} {error_pct:>8.2f}% {improvement}")
    
    print("-"*80)
    
    # Summary statistics
    errors = [v['error_pct'] for v in comparison.values()]
    improvements = sum(1 for v in comparison.values() if v['improved'])
    
    print(f"\nSummary Statistics:")
    print(f"  Mean absolute error: {np.mean(errors):.2f}%")
    print(f"  Median absolute error: {np.median(errors):.2f}%")
    print(f"  Max error: {np.max(errors):.2f}%")
    print(f"  Parameters improved: {improvements}/{len(comparison)}")
    
    return comparison


def plot_fit_with_ground_truth(result_dict, save_path=None):
    """
    Plot fitted spectrum vs observed with ground truth comparison.
    
    Args:
        result_dict: Output from fit_adc_planet()
        save_path: Optional path to save figure
    """
    spectrum_dict = result_dict['spectrum_dict']
    forward_model = result_dict['forward_model']
    initial_params = result_dict['initial_params']
    final_params = result_dict['final_params']
    ground_truth = result_dict['ground_truth']
    losses = result_dict['losses']
    
    obs_y = spectrum_dict['spectrum']
    obs_err = spectrum_dict['noise']
    wl_grid = spectrum_dict['wl_grid']
    
    initial_spectrum = np.array(forward_model(initial_params))
    final_spectrum = np.array(forward_model(final_params))
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Spectra comparison
    ax = axes[0]
    ax.errorbar(wl_grid, obs_y, yerr=obs_err, fmt='o', alpha=0.6, 
                label='Observed (ADC 2023)', markersize=5, color='black')
    ax.plot(wl_grid, initial_spectrum, '--', label='Initial model', 
            linewidth=2, alpha=0.7)
    ax.plot(wl_grid, final_spectrum, '-', label='Fitted model', 
            linewidth=2.5)
    
    ax.set_xlabel('Wavelength (μm)', fontsize=12)
    ax.set_ylabel('Transit Depth', fontsize=12)
    ax.set_title('ADC 2023 Atmospheric Retrieval - JAX Forward Model', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    ax = axes[1]
    initial_residuals = (obs_y - initial_spectrum) / obs_err
    final_residuals = (obs_y - final_spectrum) / obs_err
    
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.axhline(3, color='red', linestyle=':', alpha=0.3, label='±3σ')
    ax.axhline(-3, color='red', linestyle=':', alpha=0.3)
    
    ax.scatter(wl_grid, initial_residuals, alpha=0.5, label='Initial residuals', s=40)
    ax.scatter(wl_grid, final_residuals, alpha=0.7, label='Final residuals', s=40)
    
    ax.set_xlabel('Wavelength (μm)', fontsize=12)
    ax.set_ylabel('Residuals (σ)', fontsize=12)
    ax.set_title('Fit Residuals', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Calculate chi-squared
    chi2_initial = np.sum(initial_residuals**2)
    chi2_final = np.sum(final_residuals**2)
    dof = len(obs_y)
    ax.text(0.02, 0.98, f'Initial χ²/dof = {chi2_initial/dof:.2f}\nFinal χ²/dof = {chi2_final/dof:.2f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    # Plot 3: Loss history
    ax = axes[2]
    steps = np.arange(len(losses))
    ax.plot(steps, losses, linewidth=2, alpha=0.8)
    ax.set_xlabel('Optimization Step', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Training Loss', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Add text with convergence info
    if len(losses) > 1:
        improvement = losses[0] / losses[-1]
        ax.text(0.98, 0.98, f'Steps: {len(losses)}\nImprovement: {improvement:.1f}x',
                transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                fontsize=10)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = f"adc_planet_{result_dict['planet_id']}_fit.png"

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    
    plt.show()
    
    # Print additional statistics
    print("\nFit Quality Metrics:")
    print(f"  Initial χ²/dof: {chi2_initial/dof:.3f}")
    print(f"  Final χ²/dof: {chi2_final/dof:.3f}")
    print(f"  RMS residuals (initial): {np.std(initial_residuals):.3f}σ")
    print(f"  RMS residuals (final): {np.std(final_residuals):.3f}σ")
    print(f"  Max residual (final): {np.max(np.abs(final_residuals)):.2f}σ")


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description='ADC 2023 Retrieval Experiment')
    parser.add_argument('--planet-id', type=int, default=1000,
                        help='Planet ID to fit (default: 1000)')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Number of optimization steps (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--nlayers', type=int, default=30,
                        help='Number of atmospheric layers (default: 30)')
    parser.add_argument('--float32', action='store_true',
                        help='Use float32 instead of float64')
    parser.add_argument('--fit-params', nargs='+', 
                        default=['planet_radius', 'T', 'H2O', 'CO2', 'CO', 'CH4', 'NH3'],
                        help='Parameters to fit')
    parser.add_argument('--save-plot', type=str, default=None,
                        help='Path to save plot (default: adc_planet_<id>_fit.png)')
    parser.add_argument('--l2-reg', type=float, default=0.0,
                        help='L2 regularization strength (default: 0.0)')
    parser.add_argument('--log-prior', type=float, default=None,
                        help='Log-space prior std for mixing ratios (default: None)')
    parser.add_argument('--clip-norm', type=float, default=1.0,
                        help='Gradient clipping norm (default: 1.0)')
    parser.add_argument('--add-clouds', action='store_true',
                        help='Add opaque cloud deck to model and fit clouds_pressure')
    parser.add_argument('--add-lee-mie', action='store_true',
                        help='Add Lee Mie scattering (sloped clouds) to model')
    
    args = parser.parse_args()
    
    # Configure JAX precision
    use_float64 = not args.float32
    jax.config.update("jax_enable_x64", use_float64)
    dtype = jnp.float64 if use_float64 else jnp.float32
    
    print("="*80)
    print("ADC 2023 ATMOSPHERIC RETRIEVAL EXPERIMENT")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Planet ID: {args.planet_id}")
    print(f"  Optimization steps: {args.steps}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Atmospheric layers: {args.nlayers}")
    print(f"  Precision: {'float64' if use_float64 else 'float32'}")
    print(f"  JAX device: {jax.devices()[0]}")
    print(f"  Parameters to fit: {args.fit_params}")
    if args.l2_reg > 0:
        print(f"  L2 regularization: {args.l2_reg}")
    if args.log_prior is not None:
        print(f"  Log-space prior: {args.log_prior}")
    print(f"  Gradient clipping: {args.clip_norm}")
    print(f"  Add clouds: {args.add_clouds}")
    print(f"  Add Lee Mie: {args.add_lee_mie}")
    print("="*80)
    
    # Run fitting
    print("\nStarting retrieval...")
    start_time = time.time()
    
    result = fit_adc_planet(
        planet_id=args.planet_id,
        fit_params=args.fit_params,
        steps=args.steps,
        lr=args.lr,
        nlayers=args.nlayers,
        dtype=dtype,
        verbose=True,
        l2_reg=args.l2_reg,
        log_prior=args.log_prior,
        clip_norm=args.clip_norm,
        add_clouds=args.add_clouds,
        add_lee_mie=args.add_lee_mie
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"Retrieval completed in {elapsed:.2f} seconds ({elapsed/args.steps:.4f} sec/step)")
    print(f"{'='*80}")
    
    # Compare to ground truth
    if result['ground_truth'] is not None:
        comparison = compare_to_ground_truth(result)
    else:
        print("\nNo ground truth available for this planet")
        comparison = None
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Auto-generate save path if not provided
    if args.save_plot is None:
        args.save_plot = f"adc_planet_{args.planet_id}_fit.png"
    
    plot_fit_with_ground_truth(result, save_path=args.save_plot)
    
    # Save results summary
    if comparison:
        summary_file = f"adc_planet_{args.planet_id}_results.txt"
        with open(summary_file, 'w') as f:
            f.write("ADC 2023 Retrieval Results\n")
            f.write("="*80 + "\n\n")
            f.write(f"Planet ID: {args.planet_id}\n")
            f.write(f"Steps: {args.steps}\n")
            f.write(f"Learning rate: {args.lr}\n")
            f.write(f"Time: {elapsed:.2f} seconds\n\n")
            
            f.write("Parameter Comparison:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Parameter':<15} {'Ground Truth':<15} {'Fitted':<15} {'Error %':<10}\n")
            f.write("-"*80 + "\n")
            
            for param, data in comparison.items():
                f.write(f"{param:<15} {data['ground_truth']:<15.6e} "
                       f"{data['fitted']:<15.6e} {data['error_pct']:<10.2f}%\n")
            
            errors = [v['error_pct'] for v in comparison.values()]
            f.write("-"*80 + "\n")
            f.write(f"\nMean absolute error: {np.mean(errors):.2f}%\n")
            f.write(f"Median absolute error: {np.median(errors):.2f}%\n")
        
        print(f"\nResults summary saved to: {summary_file}")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    
    return result


if __name__ == "__main__":
    main()
