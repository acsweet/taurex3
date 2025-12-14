"""
Test different loss functions to understand their behavior.

Compare:
1. Chi-squared (current implementation)
2. Gaussian NLL (what the papers use)
3. Show why the log(σ²) term matters
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Create synthetic data with varying uncertainties
np.random.seed(42)
n_points = 52  # Number of spectral bins

# Simulate observed spectrum
true_signal = np.linspace(0.014, 0.016, n_points)  # Transit depth ~1.5%
noise_levels = np.logspace(-5, -4, n_points)  # Varying uncertainties (10-100 ppm)
observed = true_signal + np.random.normal(0, noise_levels)

# Two model predictions
model_good = true_signal  # Perfect model
model_bad = true_signal + 5e-5  # Systematically offset by 50 ppm

def chi_squared_loss(obs, pred, err):
    """Current implementation: reduced chi-squared"""
    residuals = (obs - pred) / err
    chi_sq = jnp.sum(residuals**2)
    return chi_sq / len(obs)

def gaussian_nll_loss(obs, pred, err):
    """Papers' implementation: Gaussian negative log-likelihood"""
    # -log P(D|model) = 0.5 * sum[log(2π*σ²) + (D-S)²/σ²]
    log_term = jnp.log(2 * jnp.pi * err**2)
    residual_term = ((obs - pred) / err)**2
    nll = 0.5 * jnp.sum(log_term + residual_term)
    return nll / len(obs)  # Average per data point

def gaussian_nll_no_log(obs, pred, err):
    """NLL without log(σ²) term (to show its importance)"""
    residual_term = ((obs - pred) / err)**2
    return 0.5 * jnp.sum(residual_term) / len(obs)

# Compute losses
chi_sq_good = chi_squared_loss(observed, model_good, noise_levels)
chi_sq_bad = chi_squared_loss(observed, model_bad, noise_levels)

nll_good = gaussian_nll_loss(observed, model_good, noise_levels)
nll_bad = gaussian_nll_loss(observed, model_bad, noise_levels)

nll_no_log_good = gaussian_nll_no_log(observed, model_good, noise_levels)
nll_no_log_bad = gaussian_nll_no_log(observed, model_bad, noise_levels)

print("="*80)
print("LOSS FUNCTION COMPARISON")
print("="*80)
print(f"\n{'Loss Function':<30} {'Good Model':<15} {'Bad Model':<15} {'Ratio':<10}")
print("-"*80)
print(f"{'Chi-squared (current)':<30} {chi_sq_good:<15.6f} {chi_sq_bad:<15.6f} {chi_sq_bad/chi_sq_good:<10.3f}")
print(f"{'Gaussian NLL (papers)':<30} {nll_good:<15.6f} {nll_bad:<15.6f} {nll_bad/nll_good:<10.3f}")
print(f"{'NLL without log(σ²)':<30} {nll_no_log_good:<15.6f} {nll_no_log_bad:<15.6f} {nll_no_log_bad/nll_no_log_good:<10.3f}")

print("\n" + "="*80)
print("KEY INSIGHT:")
print("="*80)
print("The log(σ²) term in Gaussian NLL gives higher loss when uncertainties are large.")
print("This encourages the model to fit high-precision data more carefully.")
print("\nChi-squared treats all wavelength bins equally (after normalization by σ).")
print("Gaussian NLL penalizes fitting low-precision data.")

# Visualize per-bin contributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Data and models
ax = axes[0, 0]
ax.errorbar(range(n_points), observed, yerr=noise_levels, fmt='o', 
            label='Observed', alpha=0.6, capsize=3)
ax.plot(model_good, 'g-', linewidth=2, label='Good model')
ax.plot(model_bad, 'r--', linewidth=2, label='Bad model (+50ppm)')
ax.set_xlabel('Wavelength bin')
ax.set_ylabel('Transit depth')
ax.set_title('Data and Models')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Uncertainty profile
ax = axes[0, 1]
ax.semilogy(range(n_points), noise_levels * 1e6, 'b-', linewidth=2)
ax.set_xlabel('Wavelength bin')
ax.set_ylabel('Uncertainty (ppm)')
ax.set_title('Measurement Uncertainties')
ax.grid(True, alpha=0.3)

# Plot 3: Per-bin chi-squared contributions
ax = axes[1, 0]
chi_sq_per_bin_good = ((observed - model_good) / noise_levels)**2
chi_sq_per_bin_bad = ((observed - model_bad) / noise_levels)**2
ax.plot(chi_sq_per_bin_good, 'g-', linewidth=2, label='Good model')
ax.plot(chi_sq_per_bin_bad, 'r--', linewidth=2, label='Bad model')
ax.set_xlabel('Wavelength bin')
ax.set_ylabel('Chi-squared contribution')
ax.set_title('Chi-squared: Per-bin Contributions')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Per-bin NLL contributions (with log term)
ax = axes[1, 1]
log_term = np.log(2 * np.pi * noise_levels**2)
nll_per_bin_good = 0.5 * (log_term + chi_sq_per_bin_good)
nll_per_bin_bad = 0.5 * (log_term + chi_sq_per_bin_bad)
ax.plot(nll_per_bin_good, 'g-', linewidth=2, label='Good model')
ax.plot(nll_per_bin_bad, 'r--', linewidth=2, label='Bad model')
ax.set_xlabel('Wavelength bin')
ax.set_ylabel('NLL contribution')
ax.set_title('Gaussian NLL: Per-bin Contributions (includes log σ²)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('loss_function_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: loss_function_comparison.png")

# Show gradient behavior
print("\n" + "="*80)
print("GRADIENT ANALYSIS")
print("="*80)

# For a single data point with uncertainty σ
sigma = 1e-5  # 10 ppm uncertainty
obs_val = 0.015
pred_val = 0.015 + 2e-5  # 20 ppm error

# Gradients w.r.t. prediction
def grad_chi_sq(obs, pred, sigma):
    return -2 * (obs - pred) / sigma**2

def grad_nll(obs, pred, sigma):
    return -(obs - pred) / sigma**2  # Same as chi-squared!

print(f"For a data point with σ = {sigma*1e6:.1f} ppm:")
print(f"  Observed: {obs_val}")
print(f"  Predicted: {pred_val}")
print(f"  Error: {(pred_val - obs_val)*1e6:.1f} ppm")
print(f"\nGradient (χ²): {grad_chi_sq(obs_val, pred_val, sigma):.3e}")
print(f"Gradient (NLL): {grad_nll(obs_val, pred_val, sigma):.3e}")
print(f"\nNOTE: Gradients are identical! The log(σ²) term only affects the loss magnitude.")
print("But the magnitude matters for:")
print("  1. Early stopping criteria")
print("  2. Learning rate adaptation")
print("  3. Comparing models with different data quality")
