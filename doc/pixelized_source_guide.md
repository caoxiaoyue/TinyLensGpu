# Pixelized Source Reconstruction Guide

## Overview

TinyLensGpu now supports **pixelized source reconstruction** in addition to parametric source modeling. This approach represents the source galaxy as discrete pixels in the source plane, with Gaussian Process regularization to ensure smooth reconstructions.

### Key Differences from Parametric Source Models

| Aspect | Parametric Source | Pixelized Source |
|--------|------------------|------------------|
| **Representation** | Analytical profiles (Sersic, Gaussian, etc.) | Discrete pixels in source plane |
| **Parameters** | Few (5-10 per component) | Many (100-10000 pixels) |
| **Flexibility** | Limited by profile shape | Highly flexible |
| **Inference** | Log likelihood | Log evidence (marginalized) |
| **Use Case** | Simple, smooth sources | Complex, irregular sources |
| **Regularization** | None (implicit in profile) | Explicit GP regularization |

## Architecture

The pixelized source implementation consists of several modules:

### 1. Core Utilities (`TinyLensGpu/PixelizedSource/`)

- **`source_inversion.py`**: Linear inversion solver with Bayesian evidence calculation
- **`regularization.py`**: Gaussian Process regularization matrices (exp, gauss, Matern-3/2, Matern-5/2)
- **`lensing.py`**: Lens mapping matrix and PSF convolution operations
- **`source_mesh.py`**: Adaptive source mesh generation
- **`interp_kernel.py`**: Wendland kernel interpolation

### 2. Model Classes (`TinyLensGpu/Models/`)

- **`PixelizedSourceConfig`**: Configuration with hyperparameters (regularization, mesh settings)
- **`PixelizedSourceModel`**: Caskade module for pixelized source

### 3. Probability Model (`TinyLensGpu/ProbModel/Image/`)

- **`PixelizedImageProbModel`**: Computes log evidence for Bayesian inference

## Mathematical Framework

### Linear Inversion

The pixelized source reconstruction solves:

```
s = (F^T N^{-1} F + H)^{-1} F^T N^{-1} d
```

where:
- `s`: Source pixel intensities (N_source,)
- `d`: Observed data (N_data,)
- `F`: Blurred lens mapping matrix (N_data, N_source)
- `N`: Noise covariance matrix (N_data, N_data)
- `H`: Regularization matrix (N_source, N_source)

### Log Evidence

The Bayesian evidence is:

```
log P(d|M) = -0.5 * [chi^2 + s^T H s] + 0.5 * log|H| - 0.5 * log|M| + const
```

where `M = F^T N^{-1} F + H`.

This is **analogous to log likelihood** in parametric models and can be used for:
1. Hyperparameter optimization (regularization scale/coefficient)
2. Mass model parameter inference
3. Model comparison

### Regularization

Gaussian Process regularization with covariance kernels:

- **Exponential**: `K(r) = exp(-r/ℓ)`
- **Gaussian (RBF)**: `K(r) = exp(-r²/(2ℓ²))`
- **Matern-3/2**: `K(r) = (1 + √3 r/ℓ) exp(-√3 r/ℓ)`
- **Matern-5/2**: `K(r) = (1 + √5 r/ℓ + 5r²/(3ℓ²)) exp(-√5 r/ℓ)`

Regularization matrix: `H = λ K^{-1}`

## Usage Guide

### Basic Example

```python
import numpy as np
from TinyLensGpu.Models.mass import SIE
from TinyLensGpu.Models.composite import PhysicalModel
from TinyLensGpu.Models.pixelized_source import PixelizedSourceModel, PixelizedSourceConfig
from TinyLensGpu.ProbModel.Image.pixelized_image_model import PixelizedImageProbModel

# 1. Create mass model
sie = SIE(theta_E=1.5, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
phys_model = PhysicalModel(lens_mass=[sie])

# 2. Configure pixelized source
config = PixelizedSourceConfig(
    reg_scale=0.05,           # Regularization length scale
    reg_coefficient=1.0,      # Regularization strength
    reg_type='exp',           # Kernel type
    n_source_points=1500,     # Number of source pixels
    mesh_alpha=1.5,           # Mesh density bias
    k_neighbors=5,            # Interpolation neighbors
)

# 3. Create pixelized source model
pix_src = PixelizedSourceModel(config=config)

# 4. Create probability model
prob_model = PixelizedImageProbModel(
    image_data=image,
    noise_map=noise,
    psf_kernel=psf,
    dpix=0.05,
    phys_model=phys_model,
    pix_src_model=pix_src,
    mask=mask,
)

# 5. Compute log evidence
log_ev = prob_model.log_evidence()
print(f"Log evidence: {log_ev:.2f}")

# 6. Reconstruct source
source_intensities, source_mesh_beta, model_image = prob_model.reconstruct_source()
```

### Hyperparameter Optimization

Use nested sampling to optimize regularization hyperparameters:

```python
import dynesty
from TinyLensGpu.Inference import build_prior, build_likelihood

# Define priors for regularization parameters
prior_dict = {
    'pix_src_model.config.reg_scale': build_prior.LogUniform(1e-3, 1e2),
    'pix_src_model.config.reg_coefficient': build_prior.LogUniform(1e-3, 1e3),
}

# Create prior transformation
prior = build_prior.make_prior(prob_model, prior_dict)

# Create likelihood function
loglike = build_likelihood.make_likelihood(prob_model, vectorized=True)

# Run nested sampling
sampler = dynesty.NestedSampler(
    loglike, prior, ndim=2,
    nlive=100, bound='multi', sample='rwalk'
)
sampler.run_nested(dlogz=0.01)
results = sampler.results

# Extract best-fit parameters
best_idx = np.argmax(results.logl)
best_params = results.samples[best_idx]
print(f"Best reg_scale: {best_params[0]:.4f}")
print(f"Best reg_coefficient: {best_params[1]:.4f}")
```

### Joint Mass + Hyperparameter Inference

```python
# Define priors for both mass and regularization
prior_dict = {
    # Mass parameters
    'phys_model.lens_mass_0.theta_E': build_prior.Uniform(1.0, 2.0),
    'phys_model.lens_mass_0.e1': build_prior.Uniform(-0.3, 0.3),
    'phys_model.lens_mass_0.e2': build_prior.Uniform(-0.3, 0.3),
    'phys_model.lens_mass_0.center_x': build_prior.Uniform(-0.1, 0.1),
    'phys_model.lens_mass_0.center_y': build_prior.Uniform(-0.1, 0.1),
    # Regularization hyperparameters
    'pix_src_model.config.reg_scale': build_prior.LogUniform(1e-3, 1e2),
    'pix_src_model.config.reg_coefficient': build_prior.LogUniform(1e-3, 1e3),
}

prior = build_prior.make_prior(prob_model, prior_dict)
loglike = build_likelihood.make_likelihood(prob_model, vectorized=True)

# Run nested sampling
sampler = dynesty.NestedSampler(
    loglike, prior, ndim=7,
    nlive=500, bound='multi', sample='rwalk'
)
sampler.run_nested(dlogz=0.01)
```

## Configuration Parameters

### PixelizedSourceConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reg_scale` | float | 0.05 | Regularization length scale (arcsec) |
| `reg_coefficient` | float | 1.0 | Regularization strength |
| `reg_type` | str | 'exp' | Kernel: 'exp', 'gauss', 'matern32', 'matern52' |
| `n_source_points` | int | 1500 | Number of source mesh points |
| `mesh_alpha` | float | 1.5 | Density bias (>1 favors bright regions) |
| `mesh_blur_sigma` | float | 0.0 | Gaussian blur for mesh sampling (pixels) |
| `mesh_method` | str | 'random' | Sampling: 'random' or 'sobol' |
| `mesh_seed` | int | 42 | Random seed for reproducibility |
| `k_neighbors` | int | 5 | Number of interpolation neighbors |
| `interp_kernel` | str | 'wendland_c4' | Interpolation kernel |
| `radius_scale` | float | 1.5 | Kernel support radius scale |

### Tuning Guidelines

**Regularization Scale (`reg_scale`)**:
- Smaller values (0.01-0.05): Fine-scale structure
- Medium values (0.05-0.2): Balanced smoothness
- Larger values (0.2-1.0): Strong smoothing

**Regularization Coefficient (`reg_coefficient`)**:
- Smaller values (0.1-1.0): Weak regularization, more flexibility
- Medium values (1.0-10): Balanced regularization
- Larger values (10-100): Strong regularization, smoother source

**Number of Source Points (`n_source_points`)**:
- Fewer points (500-1000): Faster, less detail
- Medium (1000-2000): Good balance
- More points (2000-5000): Slower, more detail

**Mesh Alpha (`mesh_alpha`)**:
- α = 1.0: Uniform sampling
- α > 1.0: Concentrate points in bright regions
- α < 1.0: More uniform coverage

## Performance Considerations

### Memory Usage

- **Source mesh**: O(N_source)
- **Lens mapping matrix**: O(N_data × N_source) but sparse
- **PSF matrix**: O(N_data²) - can be large!
- **Regularization matrix**: O(N_source²)

For large images (npix > 200), consider:
1. Using larger masks to reduce N_data
2. Using sparse PSF matrices
3. Reducing n_source_points

### Computational Cost

- **Matrix construction**: One-time cost, can be cached
- **Linear solve**: O(N_source³) - dominant cost
- **Log evidence**: Includes determinant calculations

Typical timings (npix=200, n_source=1500):
- First evaluation: ~5-10 seconds (includes matrix construction)
- Subsequent evaluations: ~0.5-1 second (with caching)

### GPU Acceleration

All operations are JAX-based and GPU-compatible:
```python
import jax
jax.config.update('jax_platform_name', 'gpu')
```

## Comparison with Parametric Models

### When to Use Pixelized Source

**Advantages**:
- Highly flexible, can model irregular sources
- No assumptions about source morphology
- Better for complex, multi-component sources
- Provides uncertainty quantification

**Use pixelized source when**:
- Source has complex, irregular morphology
- Multiple source components with unknown structure
- Need model-independent reconstruction
- Exploring source structure without assumptions

### When to Use Parametric Source

**Advantages**:
- Fewer parameters, faster inference
- Physical interpretation of parameters
- Better for simple, smooth sources
- No hyperparameter tuning needed

**Use parametric source when**:
- Source is well-described by simple profiles
- Need physical parameters (size, ellipticity, etc.)
- Limited data quality or coverage
- Computational efficiency is critical

## Advanced Topics

### Custom Regularization Kernels

You can implement custom kernels by adding to `regularization.py`:

```python
@jax.jit
def custom_cov_matrix_from(scale_coefficient, pixel_points):
    # Your custom kernel implementation
    diff = pixel_points[:, None, :] - pixel_points[None, :, :]
    distances = jnp.sqrt(jnp.sum(diff**2, axis=-1))
    # Custom covariance function
    covariance_matrix = your_kernel_function(distances, scale_coefficient)
    return covariance_matrix
```

### Adaptive Mesh Refinement

For iterative refinement:

```python
# Initial reconstruction
results1 = prob_model.reconstruct_source()

# Generate refined mesh based on reconstruction
# (requires custom implementation)
refined_mesh = generate_refined_mesh(results1)

# Create new model with refined mesh
# (requires modifying source_mesh generation)
```

### Multi-plane Lensing

For multiple source planes, create separate `PixelizedImageProbModel` instances for each plane and combine log evidences.

## Troubleshooting

### Common Issues

**1. Negative log evidence**
- Check data quality and noise map
- Verify mask is correct
- Try different regularization parameters

**2. Numerical instability**
- Reduce n_source_points
- Increase regularization coefficient
- Check for NaN/Inf in data

**3. Slow performance**
- Reduce n_source_points
- Use larger mask
- Enable GPU acceleration
- Cache PSF matrix

**4. Poor reconstruction quality**
- Tune regularization parameters
- Increase n_source_points
- Adjust mesh_alpha for better coverage
- Try different regularization kernels

## References

1. **Suyu et al. (2006)**: "Dissecting the Gravitational Lens B1608+656"
   - Original pixelized source reconstruction method

2. **Nightingale & Dye (2015)**: "Adaptive Semi-linear Inversion"
   - Adaptive mesh refinement techniques

3. **Tessore et al. (2016)**: "VKL Inversion"
   - Voronoi tessellation for source reconstruction

4. **Nightingale et al. (2018)**: "PyAutoLens"
   - Modern implementation of pixelized source modeling

## See Also

- [Demo: Pixelized Source Reconstruction](../paper/demo/src_only_pix_src/demo_pix_src.py)
- [API Reference: PixelizedSourceModel](../TinyLensGpu/Models/pixelized_source.py)
- [API Reference: PixelizedImageProbModel](../TinyLensGpu/ProbModel/Image/pixelized_image_model.py)
