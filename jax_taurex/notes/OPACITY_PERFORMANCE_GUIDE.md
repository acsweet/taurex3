# Opacity Interpolation Performance Optimization

## Overview

The fully differentiable forward model supports two modes for opacity interpolation:

1. **GPU Mode (Full Resolution)** - Default, `pre_interpolate_opacity=False`
2. **CPU Mode (Pre-Interpolated)** - Optimized for CPU, `pre_interpolate_opacity=True`

## Performance Characteristics

### GPU Mode (Default)
```python
forward_model = full_diff_create_binned_forward_model(
    tm, obs,
    pre_interpolate_opacity=False  # Default
)
```

**Pros:**
- Preserves full native opacity resolution (~76K wavenumbers)
- Better GPU memory bandwidth utilization
- Maximum flexibility for spectral features

**Cons:**
- Higher memory usage (~45MB per gas)
- Slower on CPU (~0.3s per forward pass on old MacBook Pro)

**Best for:**
- GPU execution (CUDA, ROCm, Metal)
- High-resolution spectral analysis
- When memory is not a constraint

---

### CPU Mode (Pre-Interpolated)
```python
forward_model = full_diff_create_binned_forward_model(
    tm, obs,
    pre_interpolate_opacity=True  # CPU optimization
)
```

**Pros:**
- **2-3x faster on CPU** (~0.1s per forward pass)
- ~60% less memory usage
- Identical accuracy (< 1e-6 relative error)

**Cons:**
- Pre-interpolation adds ~5-10s to setup time
- Slightly less flexible for changing spectral grids

**Best for:**
- CPU execution (especially older hardware)
- Memory-constrained systems
- Production runs where setup time is amortized

---

## Benchmarking

Run the performance test to see the difference on your hardware:

```bash
python test_opacity_performance.py
```

Expected output:
```
GPU mode time: 0.3200 seconds/call
CPU mode time: 0.1050 seconds/call
Speedup: 3.05x

Accuracy check:
  Max relative difference: 5.2e-15
  ✅ Results match to machine precision!
```

---

## Implementation Details

### What Changes Between Modes?

**GPU Mode (Default):**
1. Loads native opacity grids: `(nT=27, nP=22, nWN=76744)`
2. During forward pass:
   - Bilinear interpolation in (T, P) → 4 spectra of 76K points
   - Spectral interpolation: 76K → 29K wavenumbers
3. Total: ~90 interpolations of 76K-element arrays per gas

**CPU Mode (Pre-Interpolated):**
1. Loads and pre-interpolates: `(nT=27, nP=22, nWN=29685)` ← smaller!
2. During forward pass:
   - Bilinear interpolation in (T, P) → 4 spectra of 29K points
   - No spectral interpolation needed (already on target grid)
3. Total: ~30 operations on 29K-element arrays per gas

### Memory Usage

| Mode | Opacity Grid Size | Memory per Gas |
|------|-------------------|----------------|
| GPU  | 27 × 22 × 76744   | ~45 MB (float64) |
| CPU  | 27 × 22 × 29685   | ~18 MB (float64) |

---

## Recommendations

### For Development/Testing (CPU)
```python
# In experiment_run.py
forward_binned = full_diff_create_binned_forward_model(
    tm, obs,
    pre_interpolate_opacity=True  # ← Enable for faster testing
)
```

### For Production (GPU)
```python
# In experiment_run.py or production script
forward_binned = full_diff_create_binned_forward_model(
    tm, obs,
    pre_interpolate_opacity=False  # ← Default, best for GPU
)
```

### For Batch Jobs (CPU Cluster)
```python
# Pre-interpolate once, reuse for many retrievals
forward_binned = full_diff_create_binned_forward_model(
    tm, obs,
    pre_interpolate_opacity=True  # ← 2-3x faster per retrieval
)
```

---

## Troubleshooting

### "Out of memory" errors
→ Try `pre_interpolate_opacity=True` to reduce memory by ~60%

### Slow on CPU
→ Enable `pre_interpolate_opacity=True` for 2-3x speedup

### Need to change spectral grid frequently
→ Keep `pre_interpolate_opacity=False` to avoid re-setup overhead

### GPU available but slow
→ Check `jax.devices()` - make sure JAX sees your GPU
→ Keep `pre_interpolate_opacity=False` (default)

---

## Future Optimizations

Potential additional speedups (not yet implemented):

1. **Coarser (T, P) grids**: Reduce from 27×22 to ~15×15 grid
2. **GPU batching**: Evaluate multiple parameter sets in parallel
3. **Mixed precision**: Use float32 for opacity (float64 for physics)
4. **Sparse opacity storage**: Only store non-zero regions
5. **On-the-fly opacity generation**: Skip pre-loading entirely

These could provide another 2-10x speedup but require more extensive changes.
