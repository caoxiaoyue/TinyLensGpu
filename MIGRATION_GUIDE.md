# Migration Guide: ModelParser to Caskade

⚠️ **HISTORICAL DOCUMENT**

**Note**: As of 2025-12-17, the migration to Caskade is complete and the legacy ModelParser/Profile/Simulator code has been removed from the codebase. This guide is preserved for historical reference only.

**For new users**: You should use the Caskade-based implementation directly. See [CASKADE_GUIDE.md](CASKADE_GUIDE.md) for getting started.

**For existing users**: All demo scripts and examples have been updated to use `RunCaskadeLensModel`. Your YAML configuration files continue to work without modification.

---

This guide provides step-by-step instructions for migrating from the old ModelParser-based system to the new caskade-based implementation.

## Table of Contents

- [Why Migrate?](#why-migrate)
- [Quick Migration](#quick-migration)
- [Detailed Migration Steps](#detailed-migration-steps)
- [Code Changes](#code-changes)
- [Configuration Changes](#configuration-changes)
- [Testing Your Migration](#testing-your-migration)
- [Common Migration Issues](#common-migration-issues)
- [Performance Comparison](#performance-comparison)
- [Advanced Features](#advanced-features)

---

## Why Migrate?

The caskade-based implementation offers several advantages:

### **Benefits**
✅ **Modular Architecture**: All components are `caskade.Module` objects with clean interfaces
✅ **Automatic Parameter Management**: Parameters support dynamic/static/linear/pointer modes automatically
✅ **Better Batch Processing**: Seamless handling of large batch sizes (800+) for nested sampling
✅ **Enhanced Maintainability**: Cleaner code structure with `@ck.forward` decorators
✅ **Type Safety**: Better handling of parameter types across JAX/NumPy/PyTorch
✅ **Backward Compatibility**: Existing YAML configurations work without modification

### **What Changes**
🔄 **Import Statements**: `RunLensModel` → `RunCaskadeLensModel`
🔄 **Internal Architecture**: Profile modules → CaskadeModels
✔️ **Configuration Files**: No changes required (backward compatible)
✔️ **Inference Methods**: Same samplers and optimizers supported
✔️ **Output Format**: Identical results and file formats

---

## Quick Migration

**Minimal changes required!** The caskade implementation is designed for backward compatibility.

### Old Code (ModelParser)
```python
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from TinyLensGpu.RunModel.RunLensModel import RunLensModel

config_path = 'model_config.yaml'
lens_model = RunLensModel(config_path)
lens_model.run()
```

### New Code (Caskade)
```python
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from TinyLensGpu.CaskadeInference.runner import RunCaskadeLensModel

config_path = 'model_config.yaml'
lens_model = RunCaskadeLensModel(config_path)  # Only this line changes!
lens_model.run()
```

**That's it!** Your existing configuration files and workflow remain the same.

---

## Detailed Migration Steps

### Step 1: Update Dependencies

Ensure `caskade` is installed with JAX backend:

```bash
pip install "caskade[jax]"
```

Verify installation:
```python
import caskade as ck
print(ck.__version__)  # Should be >= 1.0.0
```

### Step 2: Update Import Statements

**Find and replace** the following imports in your Python scripts:

| Old Import | New Import |
|-----------|-----------|
| `from TinyLensGpu.RunModel.RunLensModel import RunLensModel` | `from TinyLensGpu.CaskadeInference.runner import RunCaskadeLensModel` |
| `RunLensModel(...)` | `RunCaskadeLensModel(...)` |

### Step 3: Verify Configuration Files

Your existing YAML configuration files work **without modification**. The new parser (`CaskadeConfigParser`) automatically handles both old and new formats.

**No changes needed!** But you can optionally modernize the format (see [Configuration Changes](#configuration-changes)).

### Step 4: Update Test Scripts

If you have custom test scripts, update them similarly:

**Old**:
```python
from TinyLensGpu.RunModel.RunLensModel import RunLensModel

def test_my_lens():
    runner = RunLensModel('test_config.yaml')
    runner.run()
    assert 'samples' in runner.results
```

**New**:
```python
from TinyLensGpu.CaskadeInference.runner import RunCaskadeLensModel

def test_my_lens():
    runner = RunCaskadeLensModel('test_config.yaml')
    runner.run()
    assert 'samples' in runner.results
```

### Step 5: Run Tests

Test your migration with a quick optimization run before full sampling:

```python
from TinyLensGpu.CaskadeInference.runner import RunCaskadeLensModel

# Load your configuration
runner = RunCaskadeLensModel('model_config.yaml')
runner.load_data()
runner.setup_model()

# Override with quick optimizer for testing
from TinyLensGpu.CaskadeInference.config_parser import CaskadeConfigParser
parser = CaskadeConfigParser('model_config.yaml')
bounds = parser.prior_transform.get_param_bounds()

runner.inference_config = {
    'type': 'optimizer',
    'method': 'differential_evolution',
    'settings': {
        'maxiter': 10,  # Quick test
        'popsize': 5,
        'bounds': bounds,
    }
}

runner.setup_inference()
runner.init_jit_likelihood()
runner.run_inference()

print(f"Test optimization completed: {runner.results}")
```

If this succeeds, proceed with full sampling.

---

## Code Changes

### Example 1: Basic Lens Modeling Script

**Before (Old System)**:
```python
#%%
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from TinyLensGpu.RunModel.RunLensModel import RunLensModel

config_path = 'model_config.yaml'
lens_model = RunLensModel(config_path)
lens_model.run()

# Access results
samples = lens_model.results['samples']
log_evidence = lens_model.results['log_evidence']
```

**After (Caskade System)**:
```python
#%%
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from TinyLensGpu.CaskadeInference.runner import RunCaskadeLensModel

config_path = 'model_config.yaml'
lens_model = RunCaskadeLensModel(config_path)
lens_model.run()

# Access results (same interface)
samples = lens_model.results['samples']
log_evidence = lens_model.results['log_evidence']
```

### Example 2: Step-by-Step Workflow

**Before**:
```python
from TinyLensGpu.RunModel.RunLensModel import RunLensModel

runner = RunLensModel('model_config.yaml')
runner.load_data()
runner.plot_data()
runner.setup_model()
runner.setup_inference()
runner.run_inference()
```

**After**:
```python
from TinyLensGpu.CaskadeInference.runner import RunCaskadeLensModel

runner = RunCaskadeLensModel('model_config.yaml')
runner.load_data()
runner.plot_data()
runner.setup_model()
runner.setup_inference()
runner.init_jit_likelihood()  # New: explicit JIT compilation step
runner.run_inference()
```

**New method**: `init_jit_likelihood()` explicitly triggers JIT compilation (10-15 seconds for large batches). This is optional but recommended to separate compilation from inference timing.

### Example 3: Custom Workflow with Manual Setup

**Before**:
```python
from TinyLensGpu.RunModel.RunLensModel import RunLensModel
from TinyLensGpu.ModelParser.ModelParser import ModelParser

# Manual setup
parser = ModelParser('model_config.yaml')
phys_model = parser.build_physical_model()
# ... more manual setup
```

**After**:
```python
from TinyLensGpu.CaskadeInference.runner import RunCaskadeLensModel
from TinyLensGpu.CaskadeInference.config_parser import CaskadeConfigParser

# Manual setup
parser = CaskadeConfigParser('model_config.yaml')
phys_model = parser.phys_model  # Already built
parser.set_static_params()  # Set fixed parameters
# ... more manual setup
```

---

## Configuration Changes

### Backward Compatibility

**Good news**: Your existing YAML files work without modification!

```yaml
# Old format - still works!
model_components:
  lens_mass_list:
    - type: "SIE"
      params:
        theta_E:
          prior_type: "uniform"
          prior_settings: [0.5, 2.5]
          limits: [0.0, 10.0]
          fixed: false
```

The parser automatically handles:
- `prior_type` / `prior_settings` → internal prior transformation
- `limits` → parameter bounds
- `fixed: true` / `fixed_value` → static mode
- `use_linear: true` → linear solver mode

### Optional Modernization

While not required, you can optionally use more explicit syntax:

**Old format** (still supported):
```yaml
theta_E:
  prior_type: "uniform"
  prior_settings: [0.5, 2.5]
  limits: [0.0, 10.0]
  fixed: false
```

**New format** (optional, more explicit):
```yaml
theta_E:
  mode: "dynamic"  # explicit: dynamic, static, linear, or pointer
  prior:
    type: "uniform"
    range: [0.5, 2.5]
    limits: [0.0, 10.0]
```

Both formats produce identical results.

### New Feature: Parameter Linking

The caskade system makes parameter linking (pointer mode) explicit:

**Multi-Gaussian Expansion (MGE) Example**:
```yaml
lens_light_list:
  - type: "Gaussian"  # Component 0
    params:
      sigma:
        prior_type: "uniform"
        prior_settings: [0.01, 0.5]
        fixed: false
      e1:
        prior_type: "gaussian"
        prior_settings: [0.0, 0.1]
        fixed: false
      center_x:
        fixed: true
        fixed_value: 0.0
      Amp:
        use_linear: true

  - type: "Gaussian"  # Component 1
    params:
      sigma:
        prior_type: "uniform"
        prior_settings: [0.5, 1.0]
        fixed: false
      # e1, e2, center_x, center_y will be linked to component 0
      Amp:
        use_linear: true

  # ... components 2-14 similar
```

The parser automatically links shared parameters in MGE configurations.

---

## Testing Your Migration

### Test Suite

Run the comprehensive test suite to verify your migration:

```bash
# Test all caskade functionality
pytest tests/test_caskade_models.py       # Physical models
pytest tests/test_config_parser.py        # Configuration parsing
pytest tests/test_lens_simulator.py       # Forward simulation
pytest tests/test_caskade_inference.py    # Inference system
pytest tests/test_demo_lens_src.py        # Full workflow

# Run all tests
pytest
```

### Quick Validation Script

Create a validation script to compare old vs new:

```python
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from TinyLensGpu.CaskadeInference.runner import RunCaskadeLensModel

def validate_migration(config_path):
    """
    Quick validation test for migration.

    Runs a short optimization to verify:
    1. Configuration loads correctly
    2. Data loads correctly
    3. Model builds correctly
    4. Inference runs without errors
    """
    print("="*70)
    print(f"Validating migration for: {config_path}")
    print("="*70)

    # Initialize
    runner = RunCaskadeLensModel(config_path)

    # Load data
    print("\n1. Loading data...")
    runner.load_data()
    print(f"   Image shape: {runner.image_map.shape}")
    print(f"   Noise shape: {runner.noise_map.shape}")

    # Setup model
    print("\n2. Setting up model...")
    runner.setup_model()
    print(f"   Dynamic params: {runner.config_parser.ndim}")
    print(f"   Linear params: {runner.config_parser.n_linear_params}")

    # Quick optimization test
    print("\n3. Running quick optimization test...")
    bounds = runner.config_parser.prior_transform.get_param_bounds()
    runner.inference_config = {
        'type': 'optimizer',
        'method': 'differential_evolution',
        'settings': {
            'maxiter': 5,
            'popsize': 3,
            'bounds': bounds,
        }
    }
    runner.setup_inference()
    runner.init_jit_likelihood()
    runner.run_inference()

    print(f"   Optimization result: {runner.results}")

    print("\n" + "="*70)
    print("Migration validation PASSED! ✓")
    print("="*70)

    return runner

# Test your configuration
runner = validate_migration('model_config.yaml')
```

### Numerical Comparison

Compare results between old and new systems:

```python
# Run old system
from TinyLensGpu.RunModel.RunLensModel import RunLensModel
old_runner = RunLensModel('model_config.yaml')
old_runner.run()
old_samples = old_runner.results['samples']

# Run new system
from TinyLensGpu.CaskadeInference.runner import RunCaskadeLensModel
new_runner = RunCaskadeLensModel('model_config.yaml')
new_runner.run()
new_samples = new_runner.results['samples']

# Compare
import numpy as np
for param in old_samples.keys():
    old_median = np.median(old_samples[param])
    new_median = np.median(new_samples[param])
    diff = abs(old_median - new_median) / old_median
    print(f"{param}: old={old_median:.4f}, new={new_median:.4f}, diff={diff:.2%}")
```

Expected: Differences < 5% (due to stochastic sampling).

---

## Common Migration Issues

### Issue 1: Import Errors

**Error**:
```
ModuleNotFoundError: No module named 'caskade'
```

**Solution**:
```bash
pip install "caskade[jax]"
```

### Issue 2: Type Errors with Parameters

**Error**:
```
TypeError: where requires ndarray or scalar arguments, got <class 'torch.Tensor'>
```

**Cause**: Caskade parameters may be torch.Tensor in mixed backend mode.

**Solution**: Already fixed in all caskade models! All `@ck.forward` methods include `jnp.asarray()` conversions. If you encounter this in custom code:

```python
@ck.forward
def my_function(self, x, y, param=None):
    param = jnp.asarray(param)  # Convert to JAX array
    # ... rest of computation
```

### Issue 3: Configuration Path Issues

**Error**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/image.fits'
```

**Cause**: Configuration uses relative paths, but script runs from different directory.

**Solution**:
```python
import os

# Option 1: Change to config directory
config_dir = os.path.dirname(os.path.abspath('model_config.yaml'))
os.chdir(config_dir)
runner = RunCaskadeLensModel('model_config.yaml')

# Option 2: Use absolute paths in config
# Edit YAML file to use absolute paths
```

### Issue 4: JIT Compilation Timeout

**Symptom**: First likelihood call takes 15+ seconds, appears frozen.

**Cause**: JAX is compiling the function (normal behavior).

**Solution**:
1. This is expected! Subsequent calls are fast (<1 second)
2. Reduce batch_size if compilation is too slow:
   ```yaml
   inference:
     settings:
       batch_size: 400  # Reduce from 800
   ```

### Issue 5: GPU Memory Issues

**Error**:
```
ResourceExhaustedError: Out of memory
```

**Solution**:
1. Set environment variable (already recommended):
   ```python
   os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
   ```

2. Reduce batch_size:
   ```yaml
   inference:
     settings:
       batch_size: 400
   ```

3. Reduce image size or subsampling:
   ```yaml
   dataset:
     pixel_scale: 0.1  # Larger pixels = smaller image
   # Or reduce nsub in simulator (if using manual setup)
   ```

### Issue 6: Optimizer Missing Bounds

**Error**:
```
ValueError: bounds must be provided for differential evolution
```

**Cause**: Differential Evolution optimizer requires bounds in settings.

**Solution**:
```python
from TinyLensGpu.CaskadeInference.config_parser import CaskadeConfigParser

parser = CaskadeConfigParser('model_config.yaml')
bounds = parser.prior_transform.get_param_bounds()

runner.inference_config = {
    'type': 'optimizer',
    'method': 'differential_evolution',
    'settings': {
        'maxiter': 100,
        'bounds': bounds,  # Add this
    }
}
```

### Issue 7: Results Key Mismatch

**Error**:
```
KeyError: 'samples'
```

**Cause**: Optimizer returns different keys than sampler.

**Solution**:
```python
# Check what keys are available
print(runner.results.keys())

# Samplers return: 'samples', 'log_like', 'log_evidence', etc.
# Optimizers return: 'x', 'fun', 'success', etc.

if 'samples' in runner.results:
    # Sampler result
    samples = runner.results['samples']
elif 'x' in runner.results:
    # Optimizer result
    best_params = runner.results['x']
    best_merit = runner.results['fun']
```

---

## Performance Comparison

### Benchmark Results

Tested on **lens_src demo** (200×200 image, SIE+Shear+2 Sérsic, 15 dynamic + 2 linear params):

| Metric | Old System | Caskade System | Change |
|--------|-----------|---------------|--------|
| **Optimizer (10 iter)** | 0.25 min | 0.23 min | **-8%** ✓ |
| **JIT Compilation** | 12.5 sec | 14.4 sec | +15% |
| **Likelihood (bs=1)** | N/A | 0.05 sec | - |
| **Likelihood (bs=800)** | N/A | 1.2 sec | - |
| **Memory Usage** | ~4 GB | ~4 GB | Same |

**Summary**: Caskade system has **comparable or slightly better** performance, with cleaner code architecture.

### Batch Processing Efficiency

Caskade excels at batch processing (important for nested sampling):

| Batch Size | Time per Sample (ms) | Efficiency |
|-----------|---------------------|-----------|
| 1 | 50 | Baseline |
| 100 | 1.5 | **33× faster** |
| 800 | 1.5 | **33× faster** |

Batch processing is highly efficient thanks to JAX's `vmap` and caskade's automatic broadcasting.

---

## Advanced Features

### Feature 1: Custom Linear Solvers

The caskade system supports custom linear solvers:

```python
from TinyLensGpu.CaskadeSimulator.lens_simulator import LensSimulator
from TinyLensGpu.CaskadeSimulator.config import SimulatorConfig

# Use NNLS solver (non-negative, more physical)
simulator = LensSimulator(phys_model, sim_config, solver_type='nnls')

# Or use normal least squares (faster, may produce negatives)
simulator = LensSimulator(phys_model, sim_config, solver_type='normal')
```

### Feature 2: Position Likelihood Constraints

For multiple image systems, ensure all images map to the same source:

```yaml
position_likelihood:
  threshold: 0.05  # Max allowed source plane separation (arcsec)
  min_log_like: -1000.0  # Penalty value
  image_positions:
    - [0.5, 0.3]
    - [-0.4, 0.6]
    - [-0.3, -0.5]
    - [0.6, -0.2]
```

This adds a penalty to the likelihood if image positions don't map to a consistent source position.

### Feature 3: Manual Parameter Management

For advanced use cases, you can manually manage caskade parameters:

```python
from TinyLensGpu.CaskadeInference.config_parser import CaskadeConfigParser

parser = CaskadeConfigParser('model_config.yaml')
phys_model = parser.phys_model

# Access specific component
sie = phys_model.lens_mass[0]

# Set parameter to specific value
sie.theta_E.to_static(1.5)

# Get current value
print(sie.theta_E.value)

# Set to dynamic (will be varied during sampling)
# This is handled automatically by inference adapters

# Link parameters (pointer mode)
sie2 = phys_model.lens_mass[1]
sie2.center_x = sie.center_x  # Now sie2.center_x points to sie.center_x
```

### Feature 4: Custom Batch Sizes

Control batch size for different use cases:

```python
# Small batch for debugging
prob_model.likelihood(bs=1)

# Large batch for production sampling
prob_model.likelihood(bs=800)

# Custom batch size
prob_model.likelihood(bs=200)
```

---

## Next Steps

After successful migration:

1. **Run Full Tests**: Execute complete sampling runs on your datasets
2. **Compare Results**: Verify numerical consistency with old system
3. **Optimize Performance**: Tune batch_size and nsub for your hardware
4. **Explore New Features**: Try position likelihood, custom solvers, etc.
5. **Update Documentation**: Document any project-specific migration notes

---

## Getting Help

- **API Reference**: See [CASKADE_API.md](CASKADE_API.md) for detailed API documentation
- **Usage Guide**: See [CASKADE_GUIDE.md](CASKADE_GUIDE.md) for usage examples
- **Test Examples**: Check `tests/test_demo_lens_src.py` for complete workflow examples
- **Issues**: Report problems at [GitHub Issues](https://github.com/caoxiaoyue/TinyLensGpu/issues)

---

## Summary

**Minimal Migration Path**:
1. Install `caskade[jax]`
2. Change import: `RunLensModel` → `RunCaskadeLensModel`
3. Run existing configuration files unchanged
4. Test with quick optimization before full sampling

**That's all!** The caskade system is designed for seamless migration with backward compatibility.

Happy lensing! 🔭✨
