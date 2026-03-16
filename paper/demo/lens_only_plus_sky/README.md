# Lens Light Plus Sky Demo

This demo shows how to simulate and fit a parametric lens-light image that includes both galaxy light and a constant sky background, using the programmatic interface without YAML configuration.

## Quick Start

```bash
python sim_data.py
python run_model.py
python run_model_nonlinear.py
```

## What This Demo Does

1. **Simulates mock data** with a fixed random seed, a Sersic lens-light component, and a constant sky background of `0.5`
2. **Loads FITS data** for image, noise, PSF, and mask
3. **Builds models** programmatically with `ParamU` parameters
4. **Runs Nautilus** to infer posterior distributions
5. **Outputs results** with recovered parameter estimates and diagnostic plots

## Code Structure

The directory contains three scripts:

```python
# 1. sim_data.py
#    Build a PhysicalModel with:
#    - SersicEllipse(..., Ie=1.0)
#    - ConstantBackground(intensity=0.5)
#    Then generate noisy FITS data.

# 2. run_model.py
#    Fit the mock image with the same two lens-light components.
#    The Sersic shape parameters are sampled, while
#    `Ie` and `sky_intensity` are solved linearly.

# 3. run_model_nonlinear.py
#    Fit the same mock image again, but now sample both
#    `Ie` and `sky_intensity` nonlinearly.
```

## Model Components

- **Lens Light**: Sersic profile with 4 dynamic structural parameters
  - `R_sersic`: Effective radius (uniform prior)
  - `n_sersic`: Sersic index (Gaussian prior)
  - `e1`, `e2`: Ellipticity components (Gaussian priors)
- **Sky Background**: Constant surface-brightness component
  - `sky_intensity`: Constant image-plane background level
- **Linear run**: `Ie` and `sky_intensity` are solved with NNLS
- **Nonlinear run**: `Ie` and `sky_intensity` are sampled directly

## Key Features

✅ **No YAML configuration** - Pure Python code  
✅ **Sky background aware** - Mock generation and fitting share the same physical model  
✅ **Two inference modes** - Compare linear amplitude solving with fully nonlinear sampling  
✅ **Type-safe** - Full IDE support and type hints  
✅ **Flexible** - Easy to modify and extend  
✅ **Clean** - Simple, script-based workflow for experimentation  

## Expected Output

```
============================================================
Loading data...
Building model...

Model has 4 dynamic parameters:
  R_sersic: [0.00, 2.00], limits=(0.0, 5.0)
  n_sersic: N(4.00, 0.50), limits=(0.3, 6.0)
  e1: N(0.00, 0.30), limits=(-1.0, 1.0)
  e2: N(0.00, 0.30), limits=(-1.0, 1.0)

Posterior median with linear-solved light amplitudes:
{'Ie': ~0.99, 'sky_intensity': ~0.51, ...}
============================================================
```

In the nonlinear script, the posterior summary additionally reports sampled `Ie` and `sky_intensity`, which should recover values close to the mock truth.
