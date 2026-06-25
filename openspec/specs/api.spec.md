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
    EllipticityConstraint,     # Ellipticity constraint
    GaussianPriorPasser,       # Prior passing
    nautilus_posterior_summary, # Posterior summary
    build_prior,               # module: make_prior_transformation, PriorSpec, extract_prior_specs
    build_likelihood,          # module: make_likelihood
)

# Prior/likelihood factories live on submodules (re-exported as modules, not callables)
from TinyLensGpu.Inference.build_prior import make_prior_transformation, PriorSpec, extract_prior_specs
from TinyLensGpu.Inference.build_likelihood import make_likelihood

# Abstract base shared by all samplers and optimizers (not top-level exported)
from TinyLensGpu.Inference.base import AbstractInference
```

#### NestedSampler subpackage

```python
# Note: NestedSampler/__init__.py is empty; import from the module files directly.
from TinyLensGpu.Inference.NestedSampler.nautilus_sampler import NautilusSampler
from TinyLensGpu.Inference.NestedSampler.dynesty_sampler import DynestySampler
from TinyLensGpu.Inference.NestedSampler.ultranest_sampler import UltraNestSampler
```

All samplers subclass `AbstractInference` and accept `prob_model`/`ndim`/`prior_transform`
(prior is derived lazily from `prob_model` via `_ensure_prior_transform()` on first `run()`).

#### Optimizer subpackage

```python
from TinyLensGpu.Inference.Optimizer import (
    BasinHoppingOptimizer,           # scipy.optimize.basinhopping (L-BFGS-B local)
    DifferentialEvolutionOptimizer,  # scipy.optimize.differential_evolution
    DirectOptimizer,                 # scipy.optimize.direct (DIviding RECTangles)
)
# BaseOptimizer (AbstractInference subclass) is not re-exported by the subpackage __init__.
```

Optimizers minimize the **negative** log-likelihood (SciPy convention). `run()` returns a dict
with keys `x`, `fun`, `nfev`, `nit`, `success`, `message`, `result`.

### Utilities

```python
from TinyLensGpu.utils import (
    load_lens_data,            # FITS data loading (image/noise/psf/mask)
    LinearSolver,              # Linear system solver
    solve_linear_system,       # Solve linear system (nnls/normal dispatch)
    prepare_linear_system,     # Build design matrix + data vector
    build_lens_mapping_matrix, # Lens mapping
    build_source_grid,         # Source grid
    infer_source_bbox,         # Source bounding box
    generate_radial_basis_knots, # Log-spaced radial knots (bspline/mge modes)
    mag2cps, cps2mag,          # Magnitude conversions
    weighted_quantile,         # Weighted quantile (array of quantiles)
)
```

Deeper utils (not re-exported at top level) require direct imports, e.g.
`TinyLensGpu.utils.cg_solver.pcg_solve`, `TinyLensGpu.utils.chebyshev.*`,
`TinyLensGpu.utils.geometry.*`, `TinyLensGpu.utils.interpolation.*`,
`TinyLensGpu.utils.inversion.DenseRegularizationBuilder`,
`TinyLensGpu.utils.lensing.{critical_line,point_source_solver,psf,mapping}`.

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

## Sampler / Optimizer Integration

### Low-level (build prior + likelihood, feed any backend)

```python
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.Inference.build_likelihood import make_likelihood

prior, prior_specs = make_prior_transformation(prob_model)
loglike = make_likelihood(prob_model, vectorized=True)

# Nautilus directly
from nautilus import Sampler
sampler = Sampler(prior, loglike, n_dim=len(prior_specs), n_live=200, vectorized=True, n_batch=200)
sampler.run(verbose=True, n_eff=800)
```

### High-level wrappers (AbstractInference interface)

```python
# Nested samplers — prior/ndim derived lazily from prob_model
from TinyLensGpu.Inference.NestedSampler.nautilus_sampler import NautilusSampler
ns = NautilusSampler(prob_model=prob_model)
ns.run(nlive=1000, vectorized=True)            # stores .samples, .weights, .log_z, .quantiles
# DynestySampler.run(nlive=1000, bound='multi', sample='auto')
# UltraNestSampler.run(log_dir='logs', vectorized=True)

# Optimizers — return dict with x, fun, nfev, nit, success, message, result
from TinyLensGpu.Inference.Optimizer import DifferentialEvolutionOptimizer
opt = DifferentialEvolutionOptimizer(prob_model=prob_model)
result = opt.run(bounds=[(lo, hi), ...], maxiter=1000)
# BasinHoppingOptimizer.run(x0, niter=100, T=1.0, stepsize=0.5, ftol=3e-9)
# DirectOptimizer.run(bounds=[(lo, hi), ...], maxiter=1000)
```
