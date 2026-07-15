# Lens + Source MGE Demo

## Description

This demo uses separate Multi-Gaussian Expansions (MGEs) for the lens and source light distributions, combined with an SIE plus external-shear mass model.

## Migrating from YAML to Programmatic API

Since MGE contains a large number of Gaussian components (usually 10-20), manually creating each component can be cumbersome. It is recommended to use loops or helper functions to create them.

### Example Code

```python
from TinyLensGpu.Inference import ParamU
from TinyLensGpu.ObservationModel.LensImage.parametric_image_model import (
    ImageProbModel,
)
from TinyLensGpu.PhysicalModel import (
    GaussianEllipse,
    PhysicalModel,
    Shear,
    SIE,
)

def make_mge(sigmas, prefix, center_x, center_y, e1, e2):
    components = []
    for i, sigma in enumerate(sigmas):
        gaussian = GaussianEllipse(
            sigma=ParamU(f"sigma_{prefix}_{i}", sigma),
            center_x=center_x,
            center_y=center_y,
            e1=e1,
            e2=e2,
            flux=ParamU(f"flux_{prefix}_{i}", 1.0),
        )
        gaussian.sigma.to_static(sigma)
        gaussian.flux.to_static(1.0)
        components.append(gaussian)
    return components


# In the full script, each MGE has its own shared dynamic geometry.
lens_gaussians = make_mge(
    lens_sigmas, "lens", center_x_lens, center_y_lens, e1_lens, e2_lens
)
source_gaussians = make_mge(
    source_sigmas, "source", center_x_src, center_y_src, e1_src, e2_src
)

phys_model = PhysicalModel(
    lens_mass=[sie, shear],
    source_light=source_gaussians,
    lens_light=lens_gaussians,
)
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
3. **Linear Amplitude**: With `use_linear=True`, NNLS jointly solves the coefficients of the lens and source bases.

## Current Status

`run_model.py` uses the current programmatic API. Adjust its two sets of Gaussian widths and data paths for a specific dataset.

## References

- Complete runnable script: `run_model.py`
