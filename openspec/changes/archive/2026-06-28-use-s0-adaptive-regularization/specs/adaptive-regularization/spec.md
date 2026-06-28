## ADDED Requirements

### Requirement: Source-template adaptive scale maps

The system SHALL support adaptive regularization scale maps derived from a fixed pixelized source reconstruction template instead of image-plane seed rays.

The source-template scale builder SHALL accept `source_pixels` with shape `(ny * nx,)` or `(ny, nx)` and numeric dtype convertible to `jax.float32`. It SHALL compute:

1. `s_pos = max(source_pixels, 0)`
2. `b_norm = s_pos / max(mean(s_pos), eps)`
3. `scale = floor + (1 - floor) / (1 + alpha * b_norm)`

The returned `scale` SHALL be flat with shape `(ny * nx,)`, finite, positive, and compatible with existing finite-difference regularization scale application.

#### Scenario: Positive source template creates lower scale in bright pixels

- **WHEN** a fixed `S0` source template contains bright and dark source pixels and `adaptive_reg_alpha > 0`
- **THEN** the derived scale map SHALL assign smaller scale values to brighter pixels and values near 1.0 to dark pixels

#### Scenario: Negative source pixels are clipped

- **WHEN** `S0` contains negative source-pixel values from an unconstrained linear solve
- **THEN** the scale builder SHALL clip those values to zero before normalization and SHALL NOT produce negative or non-finite scale values

#### Scenario: Uniform regularization alpha

- **WHEN** `adaptive_reg_alpha == 0`
- **THEN** source-template scale construction SHALL return `None` or an equivalent uniform-regularization fast path rather than materializing a non-uniform scale map

### Requirement: No additional smoothing for source-template scale maps

The source-template adaptive scale path SHALL NOT apply Gaussian smoothing to `S0` before normalization by default.

This replaces the earlier image-plane seed-ray smoothing path for adaptive scale construction.

#### Scenario: Structured S0 template

- **WHEN** `S0` contains a compact bright source feature that was already reconstructed under a regularization prior
- **THEN** the source-template scale map SHALL preserve that feature's pixel-scale contrast through clipping, global-mean normalization, and the existing scale formula without an extra smoothing convolution

### Requirement: Fixed source bbox for source-template inference

The operator pixelized image probability model SHALL support a fixed source-plane bbox for source-template adaptive regularization.

The API SHALL expose a configuration equivalent to:

```python
PixelizedImageProbModelOperator(
    ...,
    fixed_source_bbox: tuple[float, float, float, float] | None = None,
    fixed_reg_scale: Array | None = None,
)
```

When `fixed_source_bbox` is provided, `_get_bbox()` SHALL return that bbox for source-grid construction while still computing current mass-model betas for the lensing operator.

#### Scenario: M2 mass proposal with fixed source grid

- **WHEN** stage-m2 evaluates a new EPL+shear mass proposal and `fixed_source_bbox` is configured from stage-m0
- **THEN** the likelihood SHALL use the stage-m0 source bbox for the source grid instead of inferring a new bbox from the proposal's seed-ray betas

#### Scenario: Missing fixed bbox for adaptive S0 path

- **WHEN** adaptive regularization is enabled through the source-template path and no `fixed_source_bbox` is provided
- **THEN** the operator likelihood SHALL fail with a clear error before JIT tracing or evidence evaluation

### Requirement: External fixed scale map for operator likelihoods

The operator pixelized image probability model SHALL accept an externally supplied fixed regularization scale map.

The fixed scale map SHALL have shape `(nx * ny,)`, finite positive values, and dtype convertible to `jax.float32`. When provided and `adaptive_reg_alpha > 0`, the operator likelihood's adaptive-scale access path SHALL return the fixed scale map without performing image-plane brightness accumulation, segment sums, or smoothing.

#### Scenario: Fixed scale bypasses mass-dependent scale construction

- **WHEN** a likelihood is configured with `fixed_reg_scale` from `S0`
- **THEN** repeated likelihood evaluations at different mass parameters SHALL use the same scale values and SHALL NOT rebuild the adaptive scale from current seed-ray betas

#### Scenario: Invalid fixed scale shape

- **WHEN** `fixed_reg_scale` has a shape different from `(nx * ny,)`
- **THEN** the likelihood construction or first scale validation SHALL fail with a clear error describing the expected shape

#### Scenario: Missing fixed scale for adaptive S0 path

- **WHEN** adaptive regularization is enabled and no fixed scale map has been configured from `S0`
- **THEN** the operator likelihood SHALL fail with a clear error rather than falling back to seed-ray scale construction

### Requirement: Retired seed-ray adaptive path is unavailable

The system SHALL NOT expose the retired image-plane seed-ray adaptive regularization path, its configurable brightness modes, its Gaussian smoothing parameter, or its `freeze_scale()` / `unfreeze_scale()` empirical-Bayes cache as active adaptive-regularization APIs.

Backends that do not yet support fixed source-template adaptive inputs SHALL fail clearly when `adaptive_reg_alpha > 0` instead of computing a mass-dependent scale map from current seed-ray betas.

#### Scenario: Dense backend adaptive request

- **WHEN** a dense pixelized image probability model is constructed or evaluated with `adaptive_reg_alpha > 0`
- **THEN** it SHALL raise a clear error explaining that adaptive regularization now requires fixed source-template inputs and is currently supported by the operator backend

#### Scenario: Retired freeze API

- **WHEN** a caller tries to use `freeze_scale()` or `unfreeze_scale()` on pixelized image probability models
- **THEN** those APIs SHALL be absent or raise a clear error rather than caching a mass-dependent scale map

#### Scenario: Retired physical source configuration

- **WHEN** a caller constructs `PixelizedSourceModel`
- **THEN** it SHALL NOT accept `adaptive_reg_mode`, `adaptive_reg_smooth_sigma`, or `adaptive_reg_freeze` as active configuration fields

### Requirement: Stage-m0 source-template artifact

The adaptive regularization demo pipeline SHALL include a stage-m0 before stage-m1.

Stage-m0 SHALL fix SIE+shear mass parameters at the stage-A medians, disable adaptive regularization, optimize the uniform regularization strength with the same grid-search style used by stage-m1, solve the MAP pixelized source at the best regularization strength, and save a reusable `S0` package.

The `S0` package SHALL include at minimum:

- `source_pixels` with shape `(nx * ny,)`
- `source_bbox = (xmin, xmax, ymin, ymax)`
- `source_x_axis` and `source_y_axis`
- `nx` and `ny`
- the best stage-m0 regularization value
- enough metadata to validate that downstream stages use the same grid shape

#### Scenario: Cached stage-m0 output is reused

- **WHEN** the demo is run with `--skip-done` and a valid stage-m0 output exists
- **THEN** the pipeline SHALL load `S0` from cache and SHALL NOT recompute the uniform source reconstruction

#### Scenario: Missing S0 metadata

- **WHEN** a cached stage-m0 output lacks required source grid metadata
- **THEN** the pipeline SHALL treat the cache as invalid and recompute or raise a clear error before stage-m1 starts

### Requirement: Stage-m1 and stage-m2 consume the same S0 grid

Stages m1 and m2 in the adaptive regularization demo SHALL use the same `S0` package, fixed source bbox, and fixed source-template-derived scale map.

Stage-m1 SHALL re-optimize `log_lambda_reg` using the fixed `S0` scale map. Stage-m2 SHALL keep the stage-m1 source regularization value fixed while sampling mass parameters, and SHALL use the same fixed source bbox and scale map as stage-m1.

#### Scenario: Stage-m1 lambda search with S0 scale

- **WHEN** stage-m1 evaluates its regularization grid
- **THEN** every grid point SHALL use the same fixed source bbox and `S0`-derived scale map, varying only `log_lambda_reg`

#### Scenario: Stage-m2 mass inference with S0 scale

- **WHEN** stage-m2 samples mass parameters
- **THEN** the source adaptive prior structure SHALL remain fixed by `S0`, while the mass proposal only changes the lensing operator and position-likelihood terms

### Requirement: JIT-compatible fixed adaptive inputs

Fixed source bbox and fixed regularization scale inputs SHALL be concrete values available before `make_likelihood` traces or compiles the likelihood.

The implementation SHALL avoid Python-side mutation from inside JIT-traced likelihood evaluation when using the `S0` path. Fixed arrays SHALL be captured as constants or model attributes before tracing.

#### Scenario: Compiled likelihood uses fixed arrays

- **WHEN** `make_likelihood(..., vectorized=True)` traces a likelihood configured with `fixed_source_bbox` and `fixed_reg_scale`
- **THEN** the compiled function SHALL evaluate without attempting to write cache state during tracing

#### Scenario: Fixed scale remains stable across vectorized batch

- **WHEN** a vectorized likelihood evaluates a batch of stage-m1 `log_lambda_reg` values
- **THEN** each batch element SHALL share the same fixed source bbox and fixed regularization scale map

## REMOVED Requirements

### Requirement: Brightness-only adaptive regularization mode

**Reason**: The adaptive scale seed is no longer estimated from image-plane seed rays, so normalized convolution over lensed arc pixels is no longer part of the adaptive regularization model.

**Migration**: Use the stage-m0 `S0` source-template scale path. `S0` is reconstructed once on a fixed source grid, and m1/m2 consume the resulting fixed scale map.

### Requirement: Brightness-weighted legacy mode

**Reason**: The magnification-dependent brightness-times-ray-count estimator is the old adaptive regularization path being retired.

**Migration**: Use the fixed `S0` source-template scale path. Magnification effects remain in the lensing operator and curvature terms rather than in the adaptive prior seed.

### Requirement: Inverse-variance weighting in both modes

**Reason**: Inverse-variance weighting was specific to image-plane seed-ray accumulation. The source-template path derives scale from an already reconstructed source map and does not accumulate image pixels.

**Migration**: Noise weighting remains part of the stage-m0 source reconstruction through the pixelized likelihood. The adaptive scale builder consumes the resulting source template directly.

### Requirement: Unified downstream normalization

**Reason**: The old normalization requirement described a smoothed image-plane brightness proxy shared by seed-ray modes. The new source-template scale requirement defines the replacement normalization directly on `S0`.

**Migration**: Use `s_pos = max(S0, 0)` followed by global-mean normalization and the existing adaptive scale formula.

### Requirement: Configurable smoothing scale

**Reason**: The new source-template path deliberately avoids an additional Gaussian smoothing pass because `S0` is already regularized.

**Migration**: Do not configure `adaptive_reg_smooth_sigma` for `S0` scale construction. Control smoothness through the stage-m0 uniform regularization strength.

### Requirement: Empirical-Bayes freeze

**Reason**: The source-template scale map is fixed by construction and does not depend on the mass model during m1/m2 evaluation, so an eager `freeze_scale()` cache is unnecessary.

**Migration**: Build or load the stage-m0 `S0` package before constructing m1/m2 likelihoods, derive the fixed scale map from `S0`, and pass it as a fixed likelihood input before JIT tracing.
