# TinyLensGpu Main Specification

## Overview

TinyLensGpu is a GPU-accelerated software for galaxy-galaxy strong gravitational lens modeling, built using JAX. It processes lensing data from space telescopes such as Euclid, CSST, and Roman.

## Architecture

The system is organized into 4 layers:

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
Prior definitions, likelihood builders, and sampler interfaces.

**Key classes/functions:**
- `ParamU` - Parameter with prior metadata (extends caskade.Param)
- `make_prior_transformation` - Generate prior transformation for samplers
- `make_likelihood` - Build likelihood function for samplers
- `EllipticityConstraint` - Ellipticity constraints
- `GaussianPriorPasser` - Gaussian prior passing utility

## Data Flow

1. **Load data** - `load_lens_data()` reads FITS image/noise/PSF and optional mask
2. **Build components** - Instantiate `ParamU` parameters in physical models
3. **Select modes** - `.to_dynamic()` (sampled), `.to_static(value)` (fixed), linear (solved)
4. **Assemble** - `PhysicalModel(lens_mass=[...], source_light=[...], lens_light=[...])`
5. **Likelihood** - `ImageProbModel(..., use_linear=True, solver_type="nnls")`
6. **Sample** - `make_prior_transformation(prob_model)` → `make_likelihood(prob_model, vectorized=True)` → Nautilus/Dynesty

## Key Parameters

- **dpix** - Pixel scale (arcsec/pixel)
- **nsub** - Sub-pixel integration factor (1=fast, 3+=high accuracy)
- **solver_type** - "nnls" (physical, non-negative) or "normal" (fast, may produce negatives)
- **use_linear** - Whether to solve intensity parameters linearly
- **vectorized** - Enable batched likelihood evaluation for throughput

## Testing

- 90+ tests covering all major functionality
- Test markers: `unit`, `integration`, `slow`, `performance`, `boundary`
- Run: `pytest`, `pytest -m "not slow"`, `pytest -m integration`
- Fixtures in `tests/conftest.py`: sample_image_data, sample_noise_map, sample_psf_kernel, coordinate_grids

## JAX & GPU Quirks

- Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` for memory-constrained GPUs
- First JIT call takes 10-15s (expected compilation time)
- GPU memory exhaustion → reduce `batch_size` or `n_batch` in samplers
- NaNs in likelihood → check prior bounds, ensure no NaN/Inf in FITS data, use masks

## Dependencies

Core: `jax[cuda12]`, `caskade[jax]`, `numpy<2.0`, `scipy`, `astropy`, `matplotlib`, `jaxnnls`
Samplers: `nautilus-sampler`, `dynesty`
Dev: `pytest`, `mypy`, `black`, `flake8`, `isort`, `ruff`

## Entry Points

- Demos: `examples/*/run_model.py` (parametric) or `single_step_inversion.py` (pixelized/shapelet)
- Tests: `tests/`
- Guide: `doc/GUIDE.md`
- Setup: `setup.py`, `requirements.txt`, `requirements-dev.txt`
