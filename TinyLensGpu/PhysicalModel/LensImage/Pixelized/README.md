# PixelizedSource Module

## Overview

The `PixelizedSource` module provides core utilities for pixelized source reconstruction in gravitational lensing. Unlike parametric source models that use analytical profiles (Sersic, Gaussian, etc.), pixelized source reconstruction represents the source galaxy as discrete pixels in the source plane.

## Key Features

- **Linear Inversion**: Efficient solver with Bayesian evidence calculation
- **GP Regularization**: Multiple kernel options (exponential, Gaussian, Matern-3/2, Matern-5/2)
- **Lens Mapping**: Kernel-based interpolation with Wendland kernels
- **PSF Convolution**: Both dense and sparse matrix implementations
- **Adaptive Meshing**: Brightness-weighted source mesh generation
- **JAX-Optimized**: Fully JIT-compiled for GPU acceleration

## Modules

### source_inversion.py

Linear inversion solver implementing:
```
s = (F^T N^{-1} F + H)^{-1} F^T N^{-1} d
```

**Key Class**: `LinearInversion`
- Solves regularized linear inversion
- Computes Bayesian log evidence
- Returns source reconstruction and covariance
- Registered as JAX PyTree for efficient JIT compilation

**Usage**:
```python
from TinyLensGpu.PixelizedSource import LinearInversion

inverter = LinearInversion(
    d=data_vector,              # Observed data
    F=mapping_matrix,           # Lens mapping matrix
    noise_cov=noise_variance,   # Noise covariance
    H=regularization_matrix,    # Regularization matrix
)

# Reconstruct source
source, covariance = inverter.invert()

# Compute log evidence
log_ev = inverter.log_evidence()
```

### regularization.py

Gaussian Process regularization matrix construction.

**Key Function**: `regularization_matrix_gp_from()`

**Supported Kernels**:
- `'exp'`: Exponential kernel - K(r) = exp(-r/ℓ)
- `'gauss'`: Gaussian (RBF) kernel - K(r) = exp(-r²/(2ℓ²))
- `'matern32'`: Matern-3/2 kernel - Once differentiable
- `'matern52'`: Matern-5/2 kernel - Twice differentiable

**Usage**:
```python
from TinyLensGpu.PixelizedSource import regularization_matrix_gp_from

reg_matrix = regularization_matrix_gp_from(
    scale=0.05,              # Length scale
    coefficient=1.0,         # Regularization strength
    points=source_coords,    # Source coordinates
    reg_type='exp',          # Kernel type
)
```

### lensing.py

Lens mapping matrix and PSF convolution operations.

**Key Functions**:
- `lens_mapping_matrix_from()`: Compute lens mapping via interpolation
- `build_psf_matrix_dense()`: Dense PSF convolution matrix
- `build_psf_matrix_sparse()`: Sparse PSF convolution matrix

**Usage**:
```python
from TinyLensGpu.PixelizedSource import (
    lens_mapping_matrix_from,
    build_psf_matrix_dense
)

# Lens mapping matrix
lens_map = lens_mapping_matrix_from(
    source_mesh_beta=source_coords,  # Source plane coords
    data_mesh_beta=data_coords,      # Image plane coords
    k_neighbors=5,                   # Interpolation neighbors
    kernel='wendland_c4',            # Interpolation kernel
)

# PSF convolution matrix
psf_matrix = build_psf_matrix_dense(mask, psf_kernel)

# Combined blurred lens mapping
blurred_map = psf_matrix @ lens_map
```

### source_mesh.py

Adaptive source mesh generation based on image brightness.

**Key Function**: `sample_points_weighted()`

**Features**:
- Brightness-weighted sampling
- Random or Sobol (quasi-Monte Carlo) sampling
- Mask support for restricted regions
- Configurable density bias

**Usage**:
```python
from TinyLensGpu.PixelizedSource import sample_points_weighted

source_mesh, (H, W), _ = sample_points_weighted(
    img=image,              # Input image
    mask=valid_mask,        # Sampling mask
    n_points=1500,          # Number of points
    alpha=1.5,              # Density bias (>1 favors bright)
    method='random',        # Sampling method
    seed=42,                # Random seed
)
```

### interp_kernel.py

Wendland kernel interpolation for smooth, accurate reconstruction.

**Key Function**: `get_interpolation_weights()`

**Supported Kernels**:
- `'wendland_c2'`: C² continuous
- `'wendland_c4'`: C⁴ continuous (recommended)
- `'wendland_c6'`: C⁶ continuous

**Usage**:
```python
from TinyLensGpu.PixelizedSource.interp_kernel import get_interpolation_weights

weights, indices, distances = get_interpolation_weights(
    points=source_points,
    query_points=query_points,
    k_neighbors=5,
    kernel='wendland_c4',
)

# Interpolate values
interpolated = jnp.sum(weights * values[indices], axis=1)
```

## Mathematical Background

### Linear Inversion

The pixelized source reconstruction solves:
```
minimize: ||d - F·s||²_N + s^T·H·s
```

where:
- `d`: Observed data vector (N_data,)
- `F`: Blurred lens mapping matrix (N_data, N_source)
- `s`: Source pixel intensities (N_source,)
- `N`: Noise covariance matrix
- `H`: Regularization matrix

**Solution**:
```
s = (F^T N^{-1} F + H)^{-1} F^T N^{-1} d
```

### Bayesian Evidence

The log evidence (marginal likelihood) is:
```
log P(d|M) = -0.5·[d^T N^{-1} d - s^T (F^T N^{-1} d)]
             + 0.5·log|H| - 0.5·log|F^T N^{-1} F + H|
             - 0.5·N_data·log(2π) - 0.5·log|N|
```

This is used for:
1. Hyperparameter optimization (regularization parameters)
2. Mass model parameter inference
3. Model comparison

### Regularization

Gaussian Process prior with covariance kernel K(r):
```
H = λ·K^{-1}
```

where λ is the regularization coefficient and K is the covariance matrix.

## Performance

### Computational Complexity

- **Lens mapping matrix**: O(N_data × k_neighbors)
- **PSF matrix construction**: O(N_data² × psf_size)
- **Linear solve**: O(N_source³) - dominant cost
- **Log evidence**: O(N_source³) - includes determinants

### Memory Usage

- **Lens mapping matrix**: ~N_data × N_source × 4 bytes (sparse)
- **PSF matrix**: ~N_data² × 4 bytes (dense) or ~N_data × psf_size × 4 bytes (sparse)
- **Regularization matrix**: ~N_source² × 4 bytes

### Optimization Tips

1. **Use GPU**: All operations are JAX-based and GPU-compatible
2. **Cache PSF matrix**: Reuse across evaluations with same PSF
3. **Reduce N_source**: Fewer source points = faster computation
4. **Use masks**: Reduce N_data by masking out irrelevant regions
5. **Sparse PSF**: Use sparse PSF matrix for large images

## Integration with TinyLensGpu

This module is designed to work seamlessly with TinyLensGpu's higher-level interfaces:

```python
from TinyLensGpu.Models import PhysicalModel, PixelizedSourceModel, PixelizedSourceConfig
from TinyLensGpu.ProbModel.Image import PixelizedImageProbModel

# Create models
phys_model = PhysicalModel(lens_mass=[...])
config = PixelizedSourceConfig(reg_scale=0.05, reg_coefficient=1.0)
pix_src = PixelizedSourceModel(config=config)

# Create probability model (uses PixelizedSource utilities internally)
prob_model = PixelizedImageProbModel(
    image_data=image,
    noise_map=noise,
    psf_kernel=psf,
    dpix=0.05,
    phys_model=phys_model,
    pix_src_model=pix_src,
)

# Compute log evidence
log_ev = prob_model.log_evidence()
```

## References

1. **Suyu et al. (2006)**: "Dissecting the Gravitational Lens B1608+656"
   - Original pixelized source reconstruction method

2. **Treu & Koopmans (2004)**: "Massive Dark Matter Halos and Evolution of Early-Type Galaxies"
   - Wendland kernel interpolation for lensing

3. **Nightingale & Dye (2015)**: "Adaptive Semi-linear Inversion"
   - Modern pixelized source reconstruction techniques

## See Also

- [Pixelized Source Guide](../../doc/pixelized_source_guide.md) - Comprehensive user guide
- [Demo](../../paper/demo/src_only_pix_src/demo_pix_src.py) - Example usage
- [Code Review](../../doc/pixelized_source_code_review.md) - Implementation review
