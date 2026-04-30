# Pixelized Source Model - Legacy Implementation Backup

## 1. Overview and Motivation

This document serves as a legacy backup of the **Pixelized Source Model** implementation in TinyLensGpu, which was removed to streamline the codebase. 

The pixelized source approach represented the source galaxy as discrete pixels in the source plane, coupled with Gaussian Process (GP) or sparse regularization to ensure smooth reconstructions. Unlike parametric models (e.g., Sersic profiles) that use analytical shapes with few parameters, the pixelized approach allowed for highly flexible reconstructions of complex, irregular sources using hundreds or thousands of pixels. Inference was driven by maximizing the Bayesian log evidence (marginalizing over the source pixel intensities) rather than standard log-likelihood.

## 2. Architecture and Code Structure

The implementation followed a strict separation of concerns, integrating tightly with TinyLensGpu's Caskade parameter system and observation models.

### 2.1 Core Utilities (`TinyLensGpu/utils/` & `pixelized_core/`)
- **Linear & Operator Solvers**: Contained in `utils/inversion/linear_solver.py` (matrix backend) and `utils/inversion/operator_solver.py` (matrix-free backend). These handled the core semi-linear inversion and Bayesian evidence calculations.
- **Lensing Operations**: `utils/lensing/mapping.py`, `psf.py`, and `regularization.py` provided lens mapping (interpolation), PSF convolution, and GP/sparse regularization matrices.
- **Source Mesh**: `utils/mesh/source_mesh.py` handled adaptive, brightness-weighted source mesh generation (e.g., random and Sobol sampling).
- **Assembly Strategies**: The `ForwardSimulation/LensImage/pixelized_core/` directory used a strategy pattern to assemble grids, mappings, regularizations, and inversion methods.

### 2.2 Model Classes (`TinyLensGpu/PhysicalModel/LensImage/Pixelized/`)
- **`PixelizedSourceModel`**: A Caskade module acting as the drop-in replacement for parametric light profiles.
- **Configuration**: Managed via typed dataclasses in `config.py` (e.g., `PixelizedSourceConfig`, `IrregularGridConfig`, `MappingConfig`).

### 2.3 Probability Model (`TinyLensGpu/ObservationModel/LensImage/`)
- **`PixelizedImageProbModel`**: A subclass of the standard image probability model. It replaced the standard chi-square likelihood with the marginalized Bayesian log evidence, enabling hyperparameter optimization (e.g., regularization strength) and mass model inference.

## 3. Mathematical Framework

The core problem was recovering the source plane pixel coefficients $s$ from the observed unmasked image data $d$.

### 3.1 The Forward Model
$$ d \approx A s + n, \quad n \sim \mathcal{N}(0, N) $$
Where:
- $d \in \mathbb{R}^{n_{\mathrm{data}}}$: 1D vector of unmasked image pixels.
- $s \in \mathbb{R}^{n_{\mathrm{src}}}$: Source plane coefficients.
- $N$: Diagonal noise covariance matrix.
- $A$: The blurred lens mapping matrix, decomposed as $A = P M$, where $M$ is the geometric mapping/interpolation operator and $P$ is the PSF convolution operator.

### 3.2 Bayesian Inversion & Regularization
Introducing a Gaussian prior $p(s) \propto \exp\left(-\frac{1}{2} s^T H s\right)$, the Maximum A Posteriori (MAP) solution is given by the normal equations:
$$ (A^T N^{-1} A + H) s = A^T N^{-1} d $$
Where $H$ is the regularization matrix (e.g., dense GP kernel or sparse finite differences). 

The Bayesian Log Evidence, used for inference, was calculated as:
$$ \log P(d|M) = -\frac{1}{2} [\chi^2 + s^T H s] + \frac{1}{2} \log|H| - \frac{1}{2} \log|F| + \text{const} $$
where $F = A^T N^{-1} A + H$.

## 4. Backend Implementations

To handle the computational complexity, two distinct backends were implemented:

### 4.1 Matrix Backend (`linear_solver.py`)
- **Method**: Explicitly constructs the massive mapping matrix $A$ and the curvature matrix $F$.
- **Solving**: Uses direct linear algebra (e.g., Cholesky/LU decomposition).
- **Pros/Cons**: Exact log-evidence and straightforward implementation, but requires $O(N_{\mathrm{data}} \times N_{\mathrm{src}})$ memory and $O(N_{\mathrm{src}}^3)$ time, making it unscalable for very large images.

### 4.2 Operator Backend (`operator_solver.py`)
- **Method**: Matrix-free approach. The large matrix $A$ is treated as a black-box linear operator providing only forward ($x \mapsto Ax$) and adjoint ($y \mapsto A^T y$) multiplications.
- **Mapping Operator ($M$)**: Implemented via Scatter-Add operations (`jax.numpy.take` and `out.at[indices].add`).
- **PSF Operator ($P$)**: Implemented using fast 2D FFT convolution on padded grids, seamlessly scattering unmasked pixels to a 2D grid and gathering them back.
- **Solving**: Uses iterative Krylov subspace methods (e.g., Conjugate Gradient, FISTA) to solve the normal equations without forming $F$.

## 5. Key Features & Hyperparameters

- **Grids**: Supported both Rectangular (regular) and Irregular (adaptive) source plane grids.
- **Regularization**: 
  - Dense GP (Exponential, Gaussian, Matern-3/2, Matern-5/2).
  - Sparse (Rectangular first-order gradients, kNN).
- **Caskade Integration**: Hyperparameters like `reg_scale` (regularization length scale) and `reg_coefficient` (regularization strength) were exposed as Caskade parameters, allowing them to be inferred alongside mass model parameters using nested sampling (e.g., Dynesty).

## 6. Legacy Usage Example

Below is a snapshot of how the model was previously initialized and used:

```python
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.config import (
    PixelizedSourceConfig, IrregularGridConfig, MappingConfig, RegularizationConfig
)
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.pixelized_source import PixelizedSourceModel
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model import PixelizedImageProbModel

# 1. Configure the pixelized source
config = PixelizedSourceConfig(
    grid=IrregularGridConfig(n_source_points=1500),
    mapping=MappingConfig(k_neighbors=5, interp_kernel="wendland_c4"),
    regularization=RegularizationConfig(mode="dense_gp", gp_kernel="exp"),
)
pix_src = PixelizedSourceModel(config=config, reg_scale=0.05, reg_coefficient=1.0)

# 2. Build physical model
phys_model = PhysicalModel(lens_mass=[sie], source_light=[pix_src])

# 3. Create probability model and compute evidence
prob_model = PixelizedImageProbModel(
    image_data=image, noise_map=noise, sim_config=sim_config, phys_model=phys_model
)
log_evidence = prob_model.log_evidence()

# 4. Reconstruct source image
s_intensities, s_mesh, model_img, _ = prob_model.simulator.reconstruct_source(...)
```

## 7. Restoration Guide
If the pixelized source model needs to be restored in the future:
1. Recover the linear and operator solvers in `TinyLensGpu/utils/inversion/`.
2. Restore the strategy patterns in `TinyLensGpu/ForwardSimulation/LensImage/pixelized_core/`.
3. Bring back `PixelizedSourceModel` and its configuration dataclasses.
4. Restore `PixelizedImageProbModel` to override standard likelihood calculations with Bayesian evidence.
5. Ensure masking rules apply correctly to 1D unmasked data vectors during mapping matrix construction, to avoid fitting masked pixels (as per `03fxf6r3cm6syy5oxq1zdv39w` memory constraints).
