# Lens-Only Model Demo

This demo shows how to build and run gravitational lens models using the programmatic interface (no YAML configuration).

## Quick Start

```bash
python run_model.py
```

## What This Demo Does

1. **Loads data** from FITS files (image, noise, PSF)
2. **Builds model** programmatically with ParamU parameters
3. **Runs Nautilus sampler** to infer posterior distributions
4. **Outputs results** with parameter estimates

## Code Structure

The demo follows the `example_v4.py` style:

```python
# 1. Build problem
likelihood = build_problem()

# 2. Extract prior transformation
prior, prior_specs = make_prior_transformation(likelihood)
param_names = [spec.name for spec in prior_specs]

# 3. Create likelihood function
loglike = make_likelihood(likelihood, vectorized=True)

# 4. Run sampler
sampler = Sampler(prior, loglike, n_dim=len(param_names), 
                  n_live=200, vectorized=True, n_batch=200)
sampler.run(verbose=True, n_eff=800)

# 5. Process results
samples, log_w, _ = sampler.posterior()
```

## Model Components

- **Lens Light**: Sersic profile with 4 dynamic parameters
  - `R_sersic`: Effective radius (uniform prior)
  - `n_sersic`: Sersic index (Gaussian prior)
  - `e1`, `e2`: Ellipticity components (Gaussian priors)
  - `Ie`: Intensity (solved linearly via NNLS)

## Key Features

✅ **No YAML configuration** - Pure Python code  
✅ **Type-safe** - Full IDE support and type hints  
✅ **Flexible** - Easy to modify and extend  
✅ **Clean** - Following example_v4.py style  

## Expected Output

```
============================================================
Programmatic Lens Model Inference (No YAML)
============================================================
Loading data...
Building model...

Model has 4 dynamic parameters:
  R_sersic: [0.00, 2.00], limits=(0.0, 5.0)
  n_sersic: N(4.00, 0.50), limits=(0.3, 6.0)
  e1: N(0.00, 0.30), limits=(-1.0, 1.0)
  e2: N(0.00, 0.30), limits=(-1.0, 1.0)

Running Nautilus sampler...
[Sampling progress...]

============================================================
Posterior Summary
============================================================
  R_sersic     = 1.006 (-0.681, +0.672)
  n_sersic     = 4.007 (-0.504, +0.487)
  e1           = -0.001 (-0.297, +0.295)
  e2           = -0.002 (-0.301, +0.301)

============================================================
Inference Complete!
============================================================
```
