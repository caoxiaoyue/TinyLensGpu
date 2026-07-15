# Lens-Only MGE Demo

## Description

This demo uses Multi-Gaussian Expansion (MGE) to model the lens light distribution. MGE uses multiple Gaussian components to fit complex light distributions.

## Migrating from YAML to Programmatic API

Since MGE contains a large number of Gaussian components (usually 10-20), manually creating each component can be cumbersome. It is recommended to use loops or helper functions to create them.

### Example Code

```python
from TinyLensGpu.Inference import ParamU
from TinyLensGpu.ObservationModel.LensImage.parametric_image_model import (
    ImageProbModel,
)
from TinyLensGpu.PhysicalModel import GaussianEllipse, PhysicalModel

# Gaussian widths obtained from an MGE fit
mge_sigmas = [0.01, 0.015, 0.023, 0.035]

# Shared geometric parameters
center_x = ParamU("center_x", 0.0, prior_type="gaussian", 
                  prior_settings=[0.0, 0.1], limits=[-3.0, 3.0])
center_y = ParamU("center_y", 0.0, prior_type="gaussian",
                  prior_settings=[0.0, 0.1], limits=[-3.0, 3.0])
e1 = ParamU("e1", 0.0, prior_type="gaussian",
            prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])
e2 = ParamU("e2", 0.0, prior_type="gaussian",
            prior_settings=[0.0, 0.3], limits=[-1.0, 1.0])

# Create MGE component list
gaussians = []
for i, sigma in enumerate(mge_sigmas):
    gauss = GaussianEllipse(
        sigma=ParamU(f"sigma_{i}", sigma),
        center_x=center_x,
        center_y=center_y,
        e1=e1,
        e2=e2,
        flux=ParamU(f"flux_{i}", 1.0),
    )
    gauss.sigma.to_static(sigma)
    # A unit-amplitude static basis lets ImageProbModel solve its coefficient.
    gauss.flux.to_static(1.0)
    gaussians.append(gauss)

# Set dynamic parameters
center_x.to_dynamic()
center_y.to_dynamic()
e1.to_dynamic()
e2.to_dynamic()

# Build the physical and observation models directly.
phys_model = PhysicalModel(lens_light=gaussians)

prob_model = ImageProbModel(
    image_data=image_data,
    noise_map=noise_map,
    psf_kernel=psf_kernel,
    dpix=0.074,
    nsub=4,
    phys_model=phys_model,
    use_linear=True,
    mask=mask,
    solver_type="nnls",
)
```

## Notes

1. **Parameter Sharing**: All Gaussian components in MGE usually share the same center and ellipticity.
2. **Fixed Width**: Gaussian width (sigma) is usually obtained from MGE fitting and kept fixed.
3. **Linear Amplitude**: With `use_linear=True`, the coefficient of each unit-amplitude Gaussian basis is solved by NNLS.

## Current Status

`run_model.py` uses the current programmatic API. Adjust its Gaussian widths and data paths for a specific dataset.

## References

- Complete runnable script: `run_model.py`
