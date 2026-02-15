# Lens + Source MGE Demo

## Description

This demo uses Multi-Gaussian Expansion (MGE) to model the lens light distribution while also modeling the source light distribution. MGE uses multiple Gaussian components to fit complex light distributions.

## Migrating from YAML to Programmatic API

Since MGE contains a large number of Gaussian components (usually 10-20), manually creating each component can be cumbersome. It is recommended to use loops or helper functions to create them.

### Example Code

```python
from TinyLensGpu.Models import ParamU, GaussianEllipse
from TinyLensGpu.Models.builder import build_lens_model, build_likelihood

# MGE parameters (obtained from YAML or MGE fitting)
mge_sigmas = [0.01, 0.015, 0.023, ...]  # Gaussian width
mge_weights = [0.1, 0.15, 0.12, ...]    # Relative weights

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
for i, (sigma, weight) in enumerate(zip(mge_sigmas, mge_weights)):
    gauss = GaussianEllipse(
        sigma=ParamU(f"sigma_{i}", sigma),  # Fixed
        center_x=center_x,  # Shared
        center_y=center_y,  # Shared
        e1=e1,  # Shared
        e2=e2,  # Shared
        flux=ParamU(f"flux_{i}", weight),  # Linear parameter
    )
    gaussians.append(gauss)

# Set dynamic parameters
center_x.to_dynamic()
center_y.to_dynamic()
e1.to_dynamic()
e2.to_dynamic()

# Build model
phys_model = build_lens_model(lens_light=gaussians)

# Subsequent steps are the same as other demos
prob_model = build_likelihood(phys_model, image_data, ...)
```

## Notes

1. **Parameter Sharing**: All Gaussian components in MGE usually share the same center and ellipticity.
2. **Fixed Width**: Gaussian width (sigma) is usually obtained from MGE fitting and kept fixed.
3. **Linear Parameter**: The flux of each Gaussian is solved as a linear parameter.

## Current Status

Due to the complexity of MGE configuration, it is recommended to:
1. Use `lens_only/run_model.py` as a template.
2. Adjust the code according to actual MGE parameters.
3. Or continue to use YAML configuration (requires older version of code).

## References

- Main demo directory: `../lens_src/`
