# TinyLensGpu Main Specification

## Overview

TinyLensGpu is a GPU-accelerated software for galaxy-galaxy strong gravitational lens modeling, built using JAX. It processes lensing data from space telescopes such as Euclid, CSST, and Roman.

## Architecture

The system is organized into 4 layers. Each of `PhysicalModel`, `ForwardSimulation`, and
`ObservationModel` contains a `LensImage/` subpackage (implemented).

### 1. PhysicalModel (`TinyLensGpu/PhysicalModel/`)
Mass and light components implemented as `caskade.Module` subclasses.

**Key classes:**
- `SIE` - Singular Isothermal Ellipsoid mass profile
- `Shear` - External shear
- `EPL` - Elliptical Power Law
- `SersicEllipse` - Elliptical Sérsic light profile
- `GaussianEllipse` - Elliptical Gaussian light profile
- `ConstantBackground` - Constant background intensity
- `ShapeletBasisFunction` - Shapelet basis functions
- `PixelizedSourceModel` - Pixelized source reconstruction
- `PhysicalModel` - Composite model assembling lens_mass, source_light, lens_light

**Import paths:**
- Shallow: `from TinyLensGpu.PhysicalModel import SIE, SersicEllipse, PhysicalModel`
- Deep: `from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import SIE`

### 2. ForwardSimulation (`TinyLensGpu/ForwardSimulation/`)
Ray-tracing, PSF convolution, and image simulation.

**Key classes:**
- `LensSimulator` - Main simulation engine
- `SimulatorConfig` - Configuration for simulation parameters
- `SimulationResult` - Container for simulation outputs
- `make_grid_2d` - Coordinate grid generation

### 3. ObservationModel (`TinyLensGpu/ObservationModel/`)
Likelihood and probability models for comparing predictions with observations.

**Key classes:**
- `ImageProbModel` - Single-band image likelihood
- `MultiBandImageProbModel` - Multi-band image likelihood
- `PixelizedImageProbModel` - Pixelized source likelihood
- `PointSourceProbModel` - Point source position likelihood
- `BandImageData` - Data container for multi-band observations

### 4. Inference (`TinyLensGpu/Inference/`)
Prior definitions, likelihood builders, and sampler/optimizer interfaces.

**Key classes/functions:**
- `ParamU` - Parameter with prior metadata (extends caskade.Param)
- `make_prior_transformation` - Generate prior transformation for samplers (in `build_prior`)
- `make_likelihood` - Build likelihood function for samplers (in `build_likelihood`)
- `PriorSpec` / `extract_prior_specs` - Prior spec dataclass + traversal (in `build_prior`)
- `EllipticityConstraint` - Ellipticity constraints
- `StagePosterior` - Stage-to-stage posterior transfer from likelihood-bound samples
- `nautilus_posterior_summary` - Posterior summary from a finished nautilus sampler
- `AbstractInference` - Abstract base (ABC) shared by all samplers and optimizers (`base.py`)

**NestedSampler subpackage** (wrappers subclass `AbstractInference`; `__init__.py` is empty, use deep imports):
- `NautilusSampler` - Nautilus nested sampling (supports vectorized JAX vmap likelihood)
- `DynestySampler` - Dynesty nested sampling
- `UltraNestSampler` - UltraNest reactive nested sampling (requires `ultranest` extra)

**Optimizer subpackage** (wrappers subclass `AbstractInference` via `BaseOptimizer`; minimize negative log-likelihood):
- `DifferentialEvolutionOptimizer` - scipy `differential_evolution`
- `BasinHoppingOptimizer` - scipy `basinhopping` (L-BFGS-B local minimizer)
- `DirectOptimizer` - scipy `direct` (DIviding RECTangles)

## Data Flow

1. **Load data** - `load_lens_data()` reads FITS image/noise/PSF and optional mask
2. **Build components** - Instantiate `ParamU` parameters in physical models
3. **Select modes** - `.to_dynamic()` (sampled), `.to_static(value)` (fixed), linear (solved)
4. **Assemble** - `PhysicalModel(lens_mass=[...], source_light=[...], lens_light=[...])`
5. **Likelihood** - `ImageProbModel(..., use_linear=True, solver_type="nnls")`
6. **Sample/Optimize** - `make_prior_transformation(prob_model)` → `make_likelihood(prob_model, vectorized=True)`
   → `NautilusSampler`/`DynestySampler`/`UltraNestSampler` or
   `BasinHoppingOptimizer`/`DifferentialEvolutionOptimizer`/`DirectOptimizer`

## Key Parameters

- **dpix** - Pixel scale (arcsec/pixel)
- **nsub** - Sub-pixel integration factor (1=fast, 3+=high accuracy)
- **solver_type** - "nnls" (physical, non-negative) or "normal" (fast, may produce negatives)
- **use_linear** - Whether to solve intensity parameters linearly
- **vectorized** - Enable batched likelihood evaluation for throughput

## Testing

- ~320 tests across 17 modules in `tests/`
- Test markers: `unit`, `integration`, `slow`, `performance`, `boundary` (registered in `pytest.ini`;
  `boundary` is registered but currently unused by any test)
- Run: `pytest`, `pytest -m "not slow"`, `pytest -m integration`
- Fixtures in `tests/conftest.py`: `sample_image_data`, `sample_noise_map`, `sample_psf_kernel`, `coordinate_grids`
- Some test modules (`test_light_profile.py`, `test_mass_profile.py`) are gated on optional `lenstronomy`;
  `test_bspline_multipole.py` is gated on `scipy` (skipped by default in a standard install)

## JAX & GPU Quirks

- Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` for memory-constrained GPUs
- First JIT call takes 10-15s (expected compilation time)
- GPU memory exhaustion → reduce `batch_size` or `n_batch` in samplers
- NaNs in likelihood → check prior bounds, ensure no NaN/Inf in FITS data, use masks

## Dependencies

Core (`install_requires`): `jax[cuda12]`, `caskade[jax]`, `numpy<2.0`, `scipy`, `astropy`,
`matplotlib`, `corner`, `pyyaml`, `numba`, `nautilus-sampler`, `dynesty`, `jaxnnls`
Extras: `ultranest` (UltraNest sampler); `dev` (pytest, mypy, black, flake8, isort, ruff, pre-commit);
`docs` (sphinx); `notebooks` (jupyter); `all` (union)
Optional test-only: `lenstronomy` (migration comparison tests, skipped if absent)

## Entry Points

- Demos: `examples/*/run_model.py` (parametric) or `single_step_inversion.py` (pixelized/shapelet)
- Tests: `tests/`
- Guide: `doc/GUIDE.md`
- Setup: `setup.py`, `requirements.txt`, `requirements-dev.txt`
