# Caskade-Based Lens Modeling Guide

## Overview

This guide demonstrates how to use the new caskade-based inference system for gravitational lens modeling.

## Quick Start

### Running a Demo with Optimizer (Fast)

```python
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from TinyLensGpu.CaskadeInference.runner import RunCaskadeLensModel

# Initialize runner
runner = RunCaskadeLensModel('model_config.yaml')

# Load data
runner.load_data()
runner.plot_data()  # Visualize dataset

# Setup model
runner.setup_model()

# (Optional) Override to use optimizer for quick test
runner.inference_config = {
    'type': 'optimizer',
    'method': 'differential_evolution',
    'settings': {
        'maxiter': 100,
        'popsize': 15,
        'bounds': runner.config_parser.prior_transform.get_param_bounds(),
    }
}

runner.setup_inference()
runner.init_jit_likelihood()
runner.run_inference()

# Results
print(f"Best parameters: {runner.results['x']}")
print(f"Best merit: {runner.results['fun']}")
```

### Running with Sampler (Full Inference)

```python
from TinyLensGpu.CaskadeInference.runner import RunCaskadeLensModel

# Initialize and setup
runner = RunCaskadeLensModel('model_config.yaml')
runner.load_data()
runner.setup_model()
runner.setup_inference()  # Uses config from YAML
runner.init_jit_likelihood()

# Run nested sampling
runner.run_inference()

# Results will be saved to output directory specified in config
```

## Configuration File Format

The caskade system is **backward compatible** with existing configuration files. No changes needed!

Example configuration (lens_src):

```yaml
dataset:
  data_path: "data/image.fits"
  noise_path: "data/noise.fits"
  psf_path: "data/psf.fits"
  pixel_scale: 0.074

model_components:
  lens_mass_list:
    - type: "SIE"
      params:
        theta_E:
          prior_type: "uniform"
          prior_settings: [0.001, 3.001]
          fixed: false
        center_x:
          fixed: true
          fixed_value: 0.0

  source_light_list:
    - type: "Sersic"
      params:
        R_sersic:
          prior_type: "uniform"
          prior_settings: [0.001, 2.001]
          fixed: false
        Ie:
          use_linear: true

inference:
  type: "sampler"
  method: "nautilus"
  settings:
    nlive: 200
    batch_size: 800

solver_type: "nnls"

output:
  path: "output"
  figures:
    results: "model_results.png"
  tables:
    samples: "result_samples.csv"
```

## Key Features

### 1. Backward Compatibility

- Existing YAML configurations work without modification
- CaskadeConfigParser automatically handles old format

### 2. Parameter Management

- **Dynamic parameters**: Sampled during inference
- **Static parameters**: Fixed values
- **Linear parameters**: Solved via NNLS/normal least squares
- **Pointer parameters**: Linked to other parameters (for MGE)

### 3. Supported Inference Methods

**Samplers:**
- Nautilus (recommended for nested sampling)
- Dynesty

**Optimizers:**
- Differential Evolution
- Basin Hopping
- DIRECT

### 4. Supported Models

**Mass Profiles:**
- SIE (Singular Isothermal Ellipsoid)
- Shear (External shear)

**Light Profiles:**
- SersicEllipse (Sersic profile)
- GaussianEllipse (for MGE)

## Performance Notes

### JIT Compilation

The first likelihood evaluation triggers JIT compilation (~0.7-15 seconds depending on model complexity and batch size).

**Example timings for lens_src demo:**
- Optimizer (bs=1): ~0.7 seconds
- Sampler (bs=800): ~14 seconds

### Batch Processing

The system automatically handles batch processing for samplers:

```python
# Nautilus with batch_size=800
runner.inference_config = {
    'type': 'sampler',
    'method': 'nautilus',
    'settings': {
        'nlive': 200,
        'batch_size': 800,  # Evaluates 800 likelihoods simultaneously
    }
}
```

## Advanced Usage

### Custom Prior Transformation

Access the prior transform directly:

```python
parser = runner.config_parser

# Get parameter bounds
bounds = parser.prior_transform.get_param_bounds()

# Transform unit cube to physical space
unit_cube = np.random.rand(10, parser.ndim)  # 10 samples
physical_params = parser.prior_transform.transform(unit_cube)
```

### Manual Parameter Setting

```python
# Set parameters manually
inference = runner.inference
param_array = np.array([...])  # ndim values
inference.params_array2kargs(param_array)

# Compute likelihood
log_like = runner.prob_model.likelihood(bs=1)
```

### Position Likelihood Constraints

For multiple image systems, add position likelihood constraints:

```yaml
position_likelihood:
  positions: [[x1, y1], [x2, y2], [x3, y3]]
  threshold_arcsec: 0.01
  min_log_like: -100.0
```

## Comparison: Original vs Caskade

| Feature | Original | Caskade |
|---------|----------|---------|
| Config Format | YAML | Same (backward compatible) |
| Parameter System | Manual dict passing | caskade Param objects |
| Models | Profile classes | ck.Module classes |
| Forward Functions | Python methods | @ck.forward decorated |
| Batch Processing | Manual axis handling | Automatic |
| Parameter Linking | Manual | caskade pointers |
| Usage | RunLensModel | RunCaskadeLensModel |

## Testing

Test files are available in `tests/`:

```bash
# Test full inference system
python tests/test_caskade_inference.py

# Test lens_src demo
python tests/test_demo_lens_src.py
```

## Migration Path

**For existing users:**

1. No configuration changes needed
2. Update import:
   ```python
   # Old
   from TinyLensGpu.RunModel.RunLensModel import RunLensModel

   # New
   from TinyLensGpu.CaskadeInference.runner import RunCaskadeLensModel
   ```
3. Rest of the code remains the same!

## Troubleshooting

### NNLS Failures

If you see "NNLS failed to find a solution":
- Check that linear parameters (Ie, flux) are reasonable
- Try switching to 'normal' solver in config:
  ```yaml
  solver_type: "normal"
  ```

### NaN in Likelihood

- Check parameter bounds are reasonable
- Verify data quality (no NaN/Inf in image/noise/PSF)
- Use `debug=True` in likelihood computation

### Memory Issues

For large batch sizes, reduce:
```yaml
inference:
  settings:
    batch_size: 400  # Reduce from 800
```

## Next Steps

- See `paper/demo/` for full working examples
- Check Phase 5 plan for additional demos (MGE, position likelihood)
- Performance benchmarking coming soon

## Support

For issues or questions:
- Check test files for usage examples
- Review configuration files in `paper/demo/`
- See plan document at `.claude/plans/crispy-sprouting-reddy.md`
