# API Specification

## Public API Surface

### PhysicalModel Layer

```python
from TinyLensGpu.PhysicalModel import (
    PhysicalModel,      # Composite model
    SIE,                # Singular Isothermal Ellipsoid
    Shear,              # External shear
    EPL,                # Elliptical Power Law
    SersicEllipse,      # Elliptical Sérsic profile
    GaussianEllipse,    # Elliptical Gaussian profile
    ConstantBackground,  # Constant background
    ShapeletBasisFunction,      # Shapelet basis
    build_shapelet_set,         # Shapelet set builder
    build_shapelet_basis_matrix, # Shapelet matrix builder
    PixelizedSourceModel,       # Pixelized source
)
```

### ForwardSimulation Layer

```python
from TinyLensGpu.ForwardSimulation import (
    LensSimulator,      # Main simulation engine
    SimulatorConfig,   # Configuration
    SimulationResult,  # Output container
    make_grid_2d,      # Grid generation
)
```

### ObservationModel Layer

```python
from TinyLensGpu.ObservationModel import (
    ImageProbModel,           # Single-band likelihood
    MultiBandImageProbModel,  # Multi-band likelihood
    PixelizedImageProbModel,  # Pixelized likelihood
    PointSourceProbModel,     # Point source likelihood
    BandImageData,            # Band data container
)
```

### Inference Layer

```python
from TinyLensGpu.Inference import (
    ParamU,                    # Parameter with prior
    make_prior_transformation, # Prior builder
    make_likelihood,           # Likelihood builder
    EllipticityConstraint,     # Ellipticity constraint
    GaussianPriorPasser,       # Prior passing
    nautilus_posterior_summary, # Posterior summary
)
```

### Utilities

```python
from TinyLensGpu.utils import (
    load_lens_data,           # FITS data loading
    LinearSolver,             # Linear system solver
    solve_linear_system,      # Solve linear system
    prepare_linear_system,    # Prepare linear system
    build_lens_mapping_matrix, # Lens mapping
    build_source_grid,         # Source grid
    infer_source_bbox,         # Source bounding box
    mag2cps, cps2mag,         # Magnitude conversions
    weighted_quantile,         # Weighted quantile
)
```

### Visualization

```python
from TinyLensGpu.visualizer import (
    plot_model_results,              # 2x3 parametric diagnostics
    plot_pixelized_source_results,  # 1x4 pixelized diagnostics
    overlay_critical_lines,         # Critical line overlay
    overlay_caustics,               # Caustic overlay
    overlay_critical_and_caustics,  # Both overlays
)
```

## Parameter Modes

All `ParamU` parameters support three modes:

1. **Dynamic** (`to_dynamic()`) - Sampled by nested samplers/optimizers
2. **Static** (`to_static(value)`) - Fixed value, not sampled
3. **Linear** (default) - Solved during forward modeling when `use_linear=True`

## Linear Solver Configuration

| solver_type | Behavior | Use Case |
|-------------|----------|----------|
| `"nnls"` | Non-negative least squares, physical | Default, prevents negative fluxes |
| `"normal"` | Normal equations, faster | When speed matters, may produce negatives |

## ImageProbModel Configuration

```python
ImageProbModel(
    image_data=...,      # jnp.array, observed image
    noise_map=...,      # jnp.array, noise standard deviation
    psf_kernel=...,     # jnp.array, PSF kernel
    dpix=0.074,         # float, pixel scale in arcsec
    nsub=4,             # int, sub-pixel integration factor
    phys_model=...,     # PhysicalModel instance
    use_linear=True,    # bool, solve intensities linearly
    solver_type="nnls", # str, linear solver type
    mask=None,          # optional jnp.array, bad pixel mask
    position_likelihood=None,  # optional position constraint
)
```

## Sampler Integration

```python
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.Inference.build_likelihood import make_likelihood

prior, prior_specs = make_prior_transformation(prob_model)
loglike = make_likelihood(prob_model, vectorized=True)

# Nautilus
from nautilus import Sampler
sampler = Sampler(prior, loglike, n_dim=len(prior_specs), n_live=200, vectorized=True, n_batch=200)
sampler.run(verbose=True, n_eff=800)
```
