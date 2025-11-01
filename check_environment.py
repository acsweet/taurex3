#!/usr/bin/env python
"""
Quick environment check script.
Run this to verify your Python environment has all required packages.
"""

import sys

print("=" * 70)
print("Environment Check")
print("=" * 70)
print(f"\nPython executable: {sys.executable}")
print(f"Python version: {sys.version}")

# Check required packages
packages = [
    ('jax', 'JAX'),
    ('jaxlib', 'JAX library'),
    ('optax', 'Optax optimizer'),
    ('numpy', 'NumPy'),
    ('scipy', 'SciPy'),
    ('taurex', 'TauREx'),
]

print("\nPackage Status:")
print("-" * 70)

all_ok = True
for module_name, display_name in packages:
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        status = "✅ FOUND"
        print(f"{status} {display_name:20s} (version: {version})")
    except ImportError:
        status = "❌ MISSING"
        all_ok = False
        print(f"{status} {display_name:20s}")

print("=" * 70)

if all_ok:
    print("✅ All required packages are installed!")
    print("\nYou can now run:")
    print("  python test_experiment.py")
else:
    print("❌ Some packages are missing.")
    print("\nTo install missing packages:")
    print("\n  Using pip:")
    print("    pip install jax jaxlib optax numpy scipy")
    print("\n  Using conda:")
    print("    conda install -c conda-forge jax jaxlib optax numpy scipy")
    print("\n  For TauREx (if missing):")
    print("    pip install taurex")

print("=" * 70)

# If JAX is available, check if it works
try:
    import jax.numpy as jnp
    print("\nJAX Quick Test:")
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.sum(x)
    print(f"  jnp.sum([1, 2, 3]) = {float(y)}")
    print("  ✅ JAX is working!")
except Exception as e:
    print(f"\n❌ JAX test failed: {e}")

print("=" * 70)
