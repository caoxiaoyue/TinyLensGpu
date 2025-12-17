# TinyLensGpu Caskade API Reference

This document provides a comprehensive API reference for all caskade-based modules in TinyLensGpu.

## Table of Contents

- [CaskadeModels](#caskademodels)
  - [Mass Models](#mass-models)
  - [Light Models](#light-models)
  - [Composite Model](#composite-model)
- [CaskadeSimulator](#caskadesimulator)
- [CaskadeInference](#caskadeinference)
- [ProbModel](#probmodel)

---

## CaskadeModels

The `CaskadeModels` module provides physical models for gravitational lensing, implemented as `caskade.Module` objects.

### Mass Models

#### SIE (Singular Isothermal Ellipsoid)

**Module**: `TinyLensGpu.CaskadeModels.mass.SIE`

Singular Isothermal Ellipsoid mass distribution for modeling dark matter halos and galaxy masses.

**Constructor**:
```python
SIE(
    theta_E=None,      # Einstein radius (arcsec)
    e1=None,           # Ellipticity component 1
    e2=None,           # Ellipticity component 2
    center_x=None,     # Center x-coordinate (arcsec)
    center_y=None      # Center y-coordinate (arcsec)
)
```

**Parameters**:
- `theta_E` (`ck.Param`): Einstein radius in arcseconds. Typical range: [0.5, 3.0]
- `e1` (`ck.Param`): Ellipticity component 1 (e1 = e*cos(2φ)). Range: [-1.0, 1.0]
- `e2` (`ck.Param`): Ellipticity component 2 (e2 = e*sin(2φ)). Range: [-1.0, 1.0]
- `center_x` (`ck.Param`): Center x-coordinate in arcseconds
- `center_y` (`ck.Param`): Center y-coordinate in arcseconds

**Methods**:

```python
@ck.forward
def deriv(self, x, y, theta_E=None, e1=None, e2=None, center_x=None, center_y=None)
```
Compute deflection angles at positions (x, y).

- **Parameters**:
  - `x`, `y`: Image plane coordinates (arcsec), can be arrays
  - `theta_E`, `e1`, `e2`, `center_x`, `center_y`: Override parameter values
- **Returns**:
  - `alpha_x`, `alpha_y`: Deflection angles in x and y directions (same shape as input)

**Example**:
```python
import caskade as ck
from TinyLensGpu.CaskadeModels.mass import SIE

# Create SIE model
sie = SIE(
    theta_E=ck.Param("theta_E", 1.5),
    e1=ck.Param("e1", 0.1),
    e2=ck.Param("e2", -0.05),
    center_x=ck.Param("center_x", 0.0),
    center_y=ck.Param("center_y", 0.0)
)

# Set parameter to static (fixed value)
sie.theta_E.to_static(1.5)
sie.center_x.to_static(0.0)

# Compute deflection
import jax.numpy as jnp
x = jnp.linspace(-2, 2, 100)
y = jnp.linspace(-2, 2, 100)
alpha_x, alpha_y = sie.deriv(x, y)
```

---

#### SHEAR (External Shear)

**Module**: `TinyLensGpu.CaskadeModels.mass.SHEAR`

External shear representing tidal gravitational field from large-scale structure.

**Constructor**:
```python
SHEAR(
    gamma1=None,   # Shear component 1
    gamma2=None    # Shear component 2
)
```

**Parameters**:
- `gamma1` (`ck.Param`): Shear component 1 (γ1 = γ*cos(2φ)). Typical range: [-0.3, 0.3]
- `gamma2` (`ck.Param`): Shear component 2 (γ2 = γ*sin(2φ)). Typical range: [-0.3, 0.3]

**Methods**:

```python
@ck.forward
def deriv(self, x, y, gamma1=None, gamma2=None)
```
Compute deflection angles due to external shear.

- **Parameters**:
  - `x`, `y`: Image plane coordinates (arcsec)
  - `gamma1`, `gamma2`: Override parameter values
- **Returns**:
  - `alpha_x`, `alpha_y`: Deflection angles

**Example**:
```python
from TinyLensGpu.CaskadeModels.mass import SHEAR

shear = SHEAR(
    gamma1=ck.Param("gamma1", 0.05),
    gamma2=ck.Param("gamma2", -0.02)
)

shear.gamma1.to_static(0.05)
shear.gamma2.to_static(-0.02)

alpha_x, alpha_y = shear.deriv(x, y)
```

---

### Light Models

#### SersicEllipse

**Module**: `TinyLensGpu.CaskadeModels.light.SersicEllipse`

Elliptical Sérsic profile for modeling galaxy light distributions.

**Constructor**:
```python
SersicEllipse(
    R_sersic=None,    # Effective radius (arcsec)
    n_sersic=None,    # Sérsic index
    e1=None,          # Ellipticity component 1
    e2=None,          # Ellipticity component 2
    center_x=None,    # Center x-coordinate (arcsec)
    center_y=None,    # Center y-coordinate (arcsec)
    Ie=None           # Intensity at effective radius
)
```

**Parameters**:
- `R_sersic` (`ck.Param`): Effective radius in arcseconds. Typical range: [0.1, 5.0]
- `n_sersic` (`ck.Param`): Sérsic index. n=1 (exponential), n=4 (de Vaucouleurs). Range: [0.3, 6.0]
- `e1` (`ck.Param`): Ellipticity component 1. Range: [-1.0, 1.0]
- `e2` (`ck.Param`): Ellipticity component 2. Range: [-1.0, 1.0]
- `center_x` (`ck.Param`): Center x-coordinate in arcseconds
- `center_y` (`ck.Param`): Center y-coordinate in arcseconds
- `Ie` (`ck.Param`): Intensity at effective radius. Can be set to `mode: linear` for NNLS solving

**Methods**:

```python
@ck.forward
def light(self, x, y, R_sersic=None, n_sersic=None, e1=None, e2=None,
          center_x=None, center_y=None, Ie=None)
```
Compute surface brightness at positions (x, y).

- **Parameters**:
  - `x`, `y`: Image plane coordinates (arcsec)
  - Parameters: Override values if provided
- **Returns**:
  - Surface brightness array (same shape as input)

**Example**:
```python
from TinyLensGpu.CaskadeModels.light import SersicEllipse

sersic = SersicEllipse(
    R_sersic=ck.Param("R_sersic", 1.0),
    n_sersic=ck.Param("n_sersic", 4.0),
    e1=ck.Param("e1", 0.1),
    e2=ck.Param("e2", 0.0),
    center_x=ck.Param("center_x", 0.0),
    center_y=ck.Param("center_y", 0.0),
    Ie=ck.Param("Ie", 1.0)
)

# Set intensity to linear mode for NNLS solving
# This is typically done via configuration file

# Compute light distribution
brightness = sersic.light(x, y)
```

---

#### GaussianEllipse

**Module**: `TinyLensGpu.CaskadeModels.light.GaussianEllipse`

Elliptical Gaussian profile for modeling compact light sources or multi-Gaussian expansion (MGE).

**Constructor**:
```python
GaussianEllipse(
    sigma=None,       # Gaussian width (arcsec)
    e1=None,          # Ellipticity component 1
    e2=None,          # Ellipticity component 2
    center_x=None,    # Center x-coordinate (arcsec)
    center_y=None,    # Center y-coordinate (arcsec)
    Amp=None          # Amplitude
)
```

**Parameters**:
- `sigma` (`ck.Param`): Gaussian width in arcseconds. Typical range: [0.01, 2.0]
- `e1` (`ck.Param`): Ellipticity component 1
- `e2` (`ck.Param`): Ellipticity component 2
- `center_x` (`ck.Param`): Center x-coordinate
- `center_y` (`ck.Param`): Center y-coordinate
- `Amp` (`ck.Param`): Amplitude. Can be set to `mode: linear` for NNLS solving

**Methods**:

```python
@ck.forward
def light(self, x, y, sigma=None, e1=None, e2=None,
          center_x=None, center_y=None, Amp=None)
```
Compute surface brightness at positions (x, y).

- **Returns**: Surface brightness array

**Example (Multi-Gaussian Expansion)**:
```python
from TinyLensGpu.CaskadeModels.light import GaussianEllipse

# Create 15 Gaussian components for MGE
gaussians = []
for i in range(15):
    gauss = GaussianEllipse(
        sigma=ck.Param(f"sigma_{i}", 0.1 * (i + 1)),
        e1=ck.Param("e1_shared", 0.0),  # Shared ellipticity
        e2=ck.Param("e2_shared", 0.0),
        center_x=ck.Param("center_x_shared", 0.0),  # Shared center
        center_y=ck.Param("center_y_shared", 0.0),
        Amp=ck.Param(f"Amp_{i}", 1.0)
    )
    gaussians.append(gauss)

# Link parameters (Gaussians 1-14 share center/ellipticity with Gaussian 0)
for i in range(1, 15):
    gaussians[i].e1 = gaussians[0].e1
    gaussians[i].e2 = gaussians[0].e2
    gaussians[i].center_x = gaussians[0].center_x
    gaussians[i].center_y = gaussians[0].center_y
```

---

### Composite Model

#### PhysicalModel

**Module**: `TinyLensGpu.CaskadeModels.composite.PhysicalModel`

Composite model combining multiple mass and light components.

**Constructor**:
```python
PhysicalModel(
    lens_mass=None,      # List of mass models
    source_light=None,   # List of source light models
    lens_light=None      # List of lens light models
)
```

**Parameters**:
- `lens_mass` (`list`): List of mass models (SIE, SHEAR, etc.)
- `source_light` (`list`): List of source light models (SersicEllipse, GaussianEllipse, etc.)
- `lens_light` (`list`): List of lens light models

**Methods**:

```python
@ck.forward
def deflection(self, x, y)
```
Compute total deflection from all mass components.

- **Parameters**: `x`, `y` - Image plane coordinates
- **Returns**: `alpha_x`, `alpha_y` - Total deflection angles

```python
@ck.forward
def source_surface_brightness(self, x, y)
```
Compute source surface brightness at source plane positions (after ray-tracing).

- **Parameters**: `x`, `y` - Source plane coordinates
- **Returns**: Total source brightness

```python
@ck.forward
def lens_surface_brightness(self, x, y)
```
Compute lens surface brightness at image plane positions.

- **Parameters**: `x`, `y` - Image plane coordinates
- **Returns**: Total lens brightness

**Example**:
```python
from TinyLensGpu.CaskadeModels.composite import PhysicalModel
from TinyLensGpu.CaskadeModels.mass import SIE, SHEAR
from TinyLensGpu.CaskadeModels.light import SersicEllipse

# Create components
sie = SIE(theta_E=ck.Param("theta_E", 1.5), ...)
shear = SHEAR(gamma1=ck.Param("gamma1", 0.05), ...)
source = SersicEllipse(R_sersic=ck.Param("R_src", 0.5), ...)
lens = SersicEllipse(R_sersic=ck.Param("R_lens", 1.0), ...)

# Combine into physical model
phys_model = PhysicalModel(
    lens_mass=[sie, shear],
    source_light=[source],
    lens_light=[lens]
)

# Compute deflection
alpha_x, alpha_y = phys_model.deflection(x, y)

# Compute source plane coordinates
beta_x = x - alpha_x
beta_y = y - alpha_y

# Compute brightness
source_brightness = phys_model.source_surface_brightness(beta_x, beta_y)
lens_brightness = phys_model.lens_surface_brightness(x, y)
total_brightness = source_brightness + lens_brightness
```

---

## CaskadeSimulator

### LensSimulator

**Module**: `TinyLensGpu.CaskadeSimulator.lens_simulator.LensSimulator`

Forward simulator for gravitational lens systems, including ray-tracing, PSF convolution, and linear solving.

**Constructor**:
```python
LensSimulator(
    phys_model,           # PhysicalModel instance
    sim_config,           # SimulatorConfig instance
    solver_type='nnls'    # 'nnls' or 'normal'
)
```

**Parameters**:
- `phys_model` (`PhysicalModel`): Physical model with mass and light components
- `sim_config` (`SimulatorConfig`): Simulation configuration
- `solver_type` (`str`): Linear solver type. 'nnls' for non-negative least squares, 'normal' for standard least squares

**Methods**:

```python
@ck.forward
def simulate(self, bs=1, use_linear=False, return_intensity=False,
             image_map=None, noise_map=None)
```
Simulate lensed image.

- **Parameters**:
  - `bs` (`int`): Batch size for batch processing
  - `use_linear` (`bool`): Whether to use linear solver for intensity parameters
  - `return_intensity` (`bool`): Whether to return intensity coefficients
  - `image_map` (`array`): Observed image for linear solving
  - `noise_map` (`array`): Noise map for linear solving
- **Returns**:
  - If `return_intensity=False`: `image_model` (simulated image)
  - If `return_intensity=True`: `(image_model, intensity_list)` where `intensity_list` contains solved linear parameters

**Example**:
```python
from TinyLensGpu.CaskadeSimulator.lens_simulator import LensSimulator
from TinyLensGpu.CaskadeSimulator.config import SimulatorConfig
import jax.numpy as jnp

# Create simulator configuration
npix = 100
dpix = 0.074  # arcsec/pixel
psf_kernel = jnp.ones((3, 3)) / 9.0  # Simple box PSF

config = SimulatorConfig(
    dpix=dpix,
    npix=npix,
    psf_kernel=psf_kernel,
    nsub=2,  # Subsampling factor
    mask=None
)

# Create simulator
simulator = LensSimulator(
    phys_model=phys_model,
    sim_config=config,
    solver_type='nnls'
)

# Simulate image (non-linear, all parameters fixed)
image = simulator.simulate(bs=1, use_linear=False)

# Simulate with linear solver (given observed data)
image_model, intensities = simulator.simulate(
    bs=1,
    use_linear=True,
    return_intensity=True,
    image_map=observed_image,
    noise_map=noise_map
)
```

---

### SimulatorConfig

**Module**: `TinyLensGpu.CaskadeSimulator.config.SimulatorConfig`

Configuration for lens simulator.

**Constructor**:
```python
SimulatorConfig(
    dpix,          # Pixel scale (arcsec/pixel)
    npix,          # Number of pixels
    psf_kernel,    # PSF kernel array
    nsub=1,        # Subsampling factor
    mask=None      # Optional mask array
)
```

**Attributes**:
- `dpix` (`float`): Pixel scale in arcsec/pixel
- `npix` (`int`): Number of pixels along each axis
- `psf_kernel` (`array`): PSF kernel for convolution
- `nsub` (`int`): Subsampling factor for ray-tracing (higher = more accurate, slower)
- `mask` (`array`, optional): Boolean mask array (True = masked pixels)

**Example**:
```python
from TinyLensGpu.CaskadeSimulator.config import SimulatorConfig
import jax.numpy as jnp

config = SimulatorConfig(
    dpix=0.074,
    npix=200,
    psf_kernel=jnp.array([[0.0, 0.1, 0.0],
                          [0.1, 0.6, 0.1],
                          [0.0, 0.1, 0.0]]),
    nsub=2,
    mask=None
)
```

---

## CaskadeInference

### CaskadeConfigParser

**Module**: `TinyLensGpu.CaskadeInference.config_parser.CaskadeConfigParser`

Parser for YAML configuration files, building caskade models and managing parameter states.

**Constructor**:
```python
CaskadeConfigParser(config_path)
```

**Parameters**:
- `config_path` (`str`): Path to YAML configuration file

**Attributes**:
- `config` (`dict`): Parsed configuration dictionary
- `phys_model` (`PhysicalModel`): Built physical model
- `prior_transform` (`PriorTransform`): Prior transformation module
- `ndim` (`int`): Number of dynamic (sampling) parameters
- `n_linear_params` (`int`): Number of linear parameters

**Methods**:

```python
def set_static_params(self)
```
Set all non-dynamic parameters to static mode with initial values.

```python
def get_param_info(self, comp_type, comp_idx, param_name)
```
Get parameter information from configuration.

- **Returns**: Dictionary with parameter configuration

**Example**:
```python
from TinyLensGpu.CaskadeInference.config_parser import CaskadeConfigParser

# Parse configuration
parser = CaskadeConfigParser('model_config.yaml')

# Build physical model
phys_model = parser.phys_model

# Set static parameters
parser.set_static_params()

# Get prior bounds
bounds = parser.prior_transform.get_param_bounds()

print(f"Number of dynamic params: {parser.ndim}")
print(f"Number of linear params: {parser.n_linear_params}")
```

**Configuration File Format**:
```yaml
model_components:
  lens_mass_list:
    - type: "SIE"
      params:
        theta_E:
          prior_type: "uniform"
          prior_settings: [0.5, 2.5]  # [min, max]
          limits: [0.0, 10.0]
          fixed: false
        center_x:
          fixed: true
          fixed_value: 0.0

  source_light_list:
    - type: "Sersic"
      params:
        Ie:
          use_linear: true  # Linear parameter (NNLS)
        R_sersic:
          prior_type: "gaussian"
          prior_settings: [1.0, 0.3]  # [mean, std]
          limits: [0.0, 5.0]
          fixed: false
```

---

### RunCaskadeLensModel

**Module**: `TinyLensGpu.CaskadeInference.runner.RunCaskadeLensModel`

Main runner for complete lens modeling workflow.

**Constructor**:
```python
RunCaskadeLensModel(config_path)
```

**Parameters**:
- `config_path` (`str`): Path to YAML configuration file

**Attributes**:
- `config_parser` (`CaskadeConfigParser`): Configuration parser
- `phys_model` (`PhysicalModel`): Physical model
- `prob_model` (`CaskadeImageProbModel`): Probability model
- `inference` (`CaskadeModelInference`): Inference adapter
- `results` (`dict`): Inference results

**Methods**:

```python
def run(self)
```
Run complete workflow: load data → setup model → setup inference → run inference.

```python
def load_data(self)
```
Load FITS data files (image, noise, PSF).

```python
def plot_data(self)
```
Generate data visualization plots.

```python
def setup_model(self)
```
Build physical model and probability model from configuration.

```python
def setup_inference(self)
```
Setup inference adapter (sampler or optimizer) based on configuration.

```python
def init_jit_likelihood(self)
```
Initialize JIT compilation for likelihood function.

```python
def run_inference(self)
```
Execute inference (sampling or optimization).

**Example**:
```python
from TinyLensGpu.CaskadeInference.runner import RunCaskadeLensModel
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Create and run
runner = RunCaskadeLensModel('model_config.yaml')
runner.run()

# Or step-by-step
runner = RunCaskadeLensModel('model_config.yaml')
runner.load_data()
runner.plot_data()
runner.setup_model()
runner.setup_inference()
runner.init_jit_likelihood()
runner.run_inference()

# Access results
print(runner.results.keys())
if 'samples' in runner.results:
    samples = runner.results['samples']
    log_like = runner.results['log_like']
```

---

### Inference Adapters

#### NautilusCaskadeModelSampler

**Module**: `TinyLensGpu.CaskadeInference.runner.NautilusCaskadeModelSampler`

Adapter for Nautilus nested sampler.

**Constructor**:
```python
NautilusCaskadeModelSampler(prob_model, config_path)
```

**Configuration**:
```yaml
inference:
  type: "sampler"
  method: "nautilus"
  settings:
    nlive: 200          # Number of live points
    batch_size: 800     # Batch size for likelihood evaluation
```

**Methods**:
```python
def run(self)
```
Run Nautilus sampling.

- **Returns**: Dictionary with 'samples', 'log_like', 'log_evidence', 'log_evidence_err'

---

#### DynestyCaskadeModelSampler

**Module**: `TinyLensGpu.CaskadeInference.runner.DynestyCaskadeModelSampler`

Adapter for Dynesty nested sampler.

**Configuration**:
```yaml
inference:
  type: "sampler"
  method: "dynesty"
  settings:
    nlive: 200
    sample: 'auto'
```

---

#### DifferentialEvolutionCaskadeModelOptimizer

**Module**: `TinyLensGpu.CaskadeInference.runner.DifferentialEvolutionCaskadeModelOptimizer`

Adapter for scipy's Differential Evolution optimizer.

**Configuration**:
```yaml
inference:
  type: "optimizer"
  method: "differential_evolution"
  settings:
    maxiter: 100
    popsize: 15
    tol: 0.01
```

**Methods**:
```python
def run(self)
```
Run optimization.

- **Returns**: scipy.optimize.OptimizeResult with 'x' (best params), 'fun' (best merit), 'success', etc.

---

#### BasinHoppingCaskadeModelOptimizer

**Module**: `TinyLensGpu.CaskadeInference.runner.BasinHoppingCaskadeModelOptimizer`

Adapter for scipy's Basin Hopping optimizer (global optimization with local minimization).

**Configuration**:
```yaml
inference:
  type: "optimizer"
  method: "basin_hopping"
  settings:
    niter: 100
```

---

#### DirectCaskadeModelOptimizer

**Module**: `TinyLensGpu.CaskadeInference.runner.DirectCaskadeModelOptimizer`

Adapter for DIRECT (Dividing Rectangles) optimizer.

**Configuration**:
```yaml
inference:
  type: "optimizer"
  method: "direct"
  settings:
    maxiter: 1000
```

---

## ProbModel

### CaskadeImageProbModel

**Module**: `TinyLensGpu.ProbModel.Image.caskade_model.CaskadeImageProbModel`

Probability model for computing image likelihood with optional position likelihood constraints.

**Constructor**:
```python
CaskadeImageProbModel(
    image_data,           # Observed image array
    noise_map,            # Noise map array
    psf_kernel,           # PSF kernel
    dpix,                 # Pixel scale
    nsub,                 # Subsampling factor
    phys_model,           # PhysicalModel instance
    use_linear=False,     # Use linear solver
    mask=None,            # Optional mask
    solver_type='nnls',   # Solver type
    position_likelihood=None  # Position likelihood config
)
```

**Parameters**:
- `image_data` (`array`): Observed lensed image (npix × npix)
- `noise_map` (`array`): Noise map (standard deviation per pixel)
- `psf_kernel` (`array`): Point spread function kernel
- `dpix` (`float`): Pixel scale in arcsec/pixel
- `nsub` (`int`): Subsampling factor for ray-tracing
- `phys_model` (`PhysicalModel`): Physical model
- `use_linear` (`bool`): Use NNLS/normal solver for intensity parameters
- `mask` (`array`, optional): Boolean mask
- `solver_type` (`str`): 'nnls' or 'normal'
- `position_likelihood` (`dict`, optional): Position likelihood configuration

**Methods**:

```python
def forward_model(self, bs=1)
```
Simulate model image.

- **Parameters**: `bs` - Batch size
- **Returns**: `(image_model, intensity_list)`

```python
def likelihood(self, bs=1, debug=True)
```
Compute log-likelihood.

- **Parameters**:
  - `bs` - Batch size
  - `debug` - If True, return -inf for invalid models
- **Returns**: Log-likelihood value (or array if bs > 1)

**Example**:
```python
from TinyLensGpu.ProbModel.Image.caskade_model import CaskadeImageProbModel
import jax.numpy as jnp

# Observed data
image_data = jnp.array(...)  # 200×200
noise_map = jnp.ones((200, 200)) * 0.1
psf_kernel = jnp.array([[0, 0.1, 0], [0.1, 0.6, 0.1], [0, 0.1, 0]])

# Create probability model
prob_model = CaskadeImageProbModel(
    image_data=image_data,
    noise_map=noise_map,
    psf_kernel=psf_kernel,
    dpix=0.074,
    nsub=2,
    phys_model=phys_model,
    use_linear=True,
    solver_type='nnls'
)

# Compute likelihood (parameters already set in phys_model)
log_like = prob_model.likelihood(bs=1)
print(f"Log-likelihood: {log_like}")
```

**Position Likelihood Configuration**:
```yaml
position_likelihood:
  threshold: 0.05              # Maximum allowed source plane separation (arcsec)
  min_log_like: -1000.0        # Penalty value
  image_positions:             # Multiple image positions
    - [0.5, 0.3]
    - [-0.4, 0.6]
    - [-0.3, -0.5]
    - [0.6, -0.2]
```

When configured, the likelihood includes a penalty term that ensures all image positions map to the same source position within the threshold.

---

## Parameter Modes

Caskade parameters support four modes:

### 1. Dynamic (Sampling)
Parameters varied during inference.

**Configuration**:
```yaml
theta_E:
  prior_type: "uniform"
  prior_settings: [0.5, 2.5]
  limits: [0.0, 10.0]
  fixed: false
```

**Code**:
```python
# Automatically set by inference adapter
inference.params_array2kargs(param_array)
```

### 2. Static (Fixed)
Parameters held constant.

**Configuration**:
```yaml
center_x:
  fixed: true
  fixed_value: 0.0
```

**Code**:
```python
param_obj.to_static(0.0)
```

### 3. Linear (NNLS/Normal Solver)
Parameters solved via linear least squares during forward model.

**Configuration**:
```yaml
Ie:
  use_linear: true
```

**Code**:
```python
# Automatically handled by simulator when use_linear=True
```

### 4. Pointer (Parameter Linking)
Parameters shared between components (e.g., MGE with shared center).

**Code**:
```python
# Link Gaussian components 1-14 to component 0
for i in range(1, 15):
    gaussians[i].center_x = gaussians[0].center_x
    gaussians[i].center_y = gaussians[0].center_y
```

---

## Prior Types

### Uniform Prior
```yaml
prior_type: "uniform"
prior_settings: [min, max]
limits: [hard_min, hard_max]
```

### Gaussian Prior
```yaml
prior_type: "gaussian"
prior_settings: [mean, std]
limits: [hard_min, hard_max]
```

### Log-Uniform Prior
```yaml
prior_type: "log_uniform"
prior_settings: [log_min, log_max]
limits: [hard_min, hard_max]
```

---

## Batch Processing

All caskade models support batch processing for efficient nested sampling:

```python
# Single evaluation
phys_model.theta_E.to_static(1.5)
alpha_x, alpha_y = phys_model.deflection(x, y)

# Batch evaluation (800 parameter sets)
theta_E_batch = jnp.array([...])  # Shape: (800,)
phys_model.theta_E.to_static(theta_E_batch)
alpha_x, alpha_y = phys_model.deflection(x, y)  # Broadcasting handled automatically
```

Batch size is controlled by:
- Sampler: `batch_size` setting in configuration
- Optimizer: Typically uses batch_size=1
- Simulator: `bs` parameter in `simulate()`

---

## Complete Workflow Example

```python
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from TinyLensGpu.CaskadeInference.runner import RunCaskadeLensModel

# Run complete modeling
runner = RunCaskadeLensModel('model_config.yaml')
runner.run()

# Access results
samples = runner.results['samples']
log_evidence = runner.results['log_evidence']

# Save results
import pandas as pd
df = pd.DataFrame(samples)
df.to_csv('output/samples.csv', index=False)
```

---

## Performance Tips

1. **JIT Compilation**: First likelihood call will be slow (~10-15 seconds for batch_size=800). Subsequent calls are fast.

2. **Batch Size**:
   - Nautilus: Use batch_size=800 for best performance
   - Optimizers: batch_size=1 is fine

3. **Subsampling**:
   - `nsub=1`: Fast but less accurate
   - `nsub=2`: Good balance (recommended)
   - `nsub=3+`: More accurate but slower

4. **Linear Solver**:
   - NNLS: Non-negative constraint (physical), slightly slower
   - Normal: Faster but may produce negative values

5. **GPU Memory**:
   - Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` to prevent OOM errors
   - Reduce batch_size if memory issues persist

---

## Troubleshooting

### Issue: NaN in likelihood
**Cause**: Parameters outside valid range or uninitialized
**Solution**: Check parameter bounds and ensure `set_static_params()` was called

### Issue: Slow JIT compilation
**Cause**: First call compiles the function
**Solution**: Normal behavior, subsequent calls are fast

### Issue: Type errors (torch.Tensor vs JAX)
**Cause**: Caskade parameters may be torch.Tensor
**Solution**: All forward methods include `jnp.asarray()` conversions

### Issue: File not found in demos
**Cause**: Demo configs use relative paths
**Solution**: Run from demo directory or use absolute paths

---

## Migration from Old System

**Old code**:
```python
from TinyLensGpu.RunModel.RunLensModel import RunLensModel
lens_model = RunLensModel('model_config.yaml')
lens_model.run()
```

**New code**:
```python
from TinyLensGpu.CaskadeInference.runner import RunCaskadeLensModel
lens_model = RunCaskadeLensModel('model_config.yaml')
lens_model.run()
```

**Configuration files**: Backward compatible! Existing YAML files work without modification.

---

For more examples and usage patterns, see [CASKADE_GUIDE.md](CASKADE_GUIDE.md).
