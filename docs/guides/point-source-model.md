# Point Source Position Modeling Guide

This guide describes how to use `PointSourceProbModel` to model lensed
point-image positions in `TinyLensGpu`.

## Purpose

`PointSourceProbModel` provides a standalone likelihood term based on image
positions only (without image-pixel flux fitting):

- Solves lens equation for a source position
- Matches model image positions to observed image positions
- Computes Gaussian position log-likelihood

## Basic Usage

```python
import numpy as np
from TinyLensGpu.PhysicalModel import PhysicalModel, SIE, Shear
from TinyLensGpu.ObservationModel import PointSourceProbModel

phys_model = PhysicalModel(
    lens_mass=[
        SIE(theta_E=1.2, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0),
        Shear(gamma1=0.03, gamma2=-0.01),
    ],
    source_light=[],
    lens_light=[],
)

observed_positions = np.array([
    [0.11, -1.07],
    [0.93, 1.52],
], dtype=float)

position_sigma = np.array([0.01, 0.01], dtype=float)

model = PointSourceProbModel(
    phys_model=phys_model,
    observed_positions=observed_positions,
    position_sigma=position_sigma,
    solver="optimization",  # or "amr"
)

log_like = model.likelihood()
```

## Key Arguments

- `observed_positions`: shape `(N, 2)`, arcsec
- `position_sigma`: shape `(N,)`, arcsec, per-image uncertainty
- `source_x`, `source_y`: source position parameters (`ParamU` or float)
- `source_position_fixed`: set `True` to keep source position fixed
- `solver`: `"optimization"` or `"amr"`
- `solver_config`: controls search/refinement settings

## Solver Config (defaults)

Common:

- `initial_range=5.0`
- `n_x=100`, `n_y=100`
- `k_keep=20`
- `tolerance=1e-4`
- `cluster_tol=0.05`

Optimization-specific:

- `num_iters=20`
- `jacobian_eps=1e-6`

AMR-specific:

- `subgrid_res=20`
- `depth=10`
- `search_factor=2.0`

## Image-Position Solver APIs

`solve_image_positions()` preserves the original convenience interface and
returns dynamically sized `(images, dists)` arrays containing every valid,
unique root found by the configured solver.

For repeated GPU work, use `solve_image_positions_fixed()` instead:

```python
images, dists, valid_mask, count = model.solve_image_positions_fixed()
valid_images = images[valid_mask]
```

It is outer-JITed and returns static shapes `(k_keep, 2)` and `(k_keep,)`.
Invalid rows are zero padded and identified by `valid_mask`. The dynamic
method trims this fixed result only at its Python-facing boundary, so both APIs
have the same valid roots and residual ordering.

The position likelihood permits unobserved extra images: it requires at least
as many valid predicted roots as observed positions, then matches the observed
positions one-to-one to the best subset of all valid predicted roots. Extra
predicted roots are not penalized, which is appropriate when demagnified or
lens-light-contaminated images may be absent from the catalog.

## Inference Integration

`PointSourceProbModel` is compatible with existing inference utilities:

- `TinyLensGpu.Inference.build_prior.make_prior_transformation`
- `TinyLensGpu.Inference.build_likelihood.make_likelihood`

By default, `source_x/source_y` are dynamic parameters and can be jointly
sampled with lens mass parameters.
