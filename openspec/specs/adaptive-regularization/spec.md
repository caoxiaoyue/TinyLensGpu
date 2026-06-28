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

### Requirement: Fittable source-template adaptive hyperparameters

The system SHALL allow source-template adaptive regularization hyperparameters to be represented as sampler-visible parameters.

`PixelizedSourceModel` SHALL accept `adaptive_reg_alpha` and `adaptive_reg_floor` as either scalar values or `ParamU` instances. When supplied as dynamic `ParamU` instances, they SHALL be returned through the model's Caskade dynamic parameter traversal and SHALL be consumable by `make_prior_transformation`.

The parameter constraints SHALL be:

- `adaptive_reg_alpha >= 0`
- `0 < adaptive_reg_floor <= 1`

#### Scenario: Prior extraction includes adaptive hyperparameters

- **WHEN** a pixelized source model has dynamic `ParamU` values for `log_lambda_reg`, `adaptive_reg_alpha`, and `adaptive_reg_floor`
- **THEN** prior extraction SHALL include all three parameter names and their configured prior metadata

#### Scenario: Scalar adaptive hyperparameters remain static

- **WHEN** a pixelized source model is constructed with scalar `adaptive_reg_alpha` and `adaptive_reg_floor`
- **THEN** those values SHALL remain static configuration and SHALL NOT add sampler dimensions

### Requirement: JAX-compatible source-template scale generation from traced hyperparameters

The source-template scale builder SHALL support JAX-traced scalar values for `alpha` and `floor` when the source template shape is statically valid.

The public scale-builder behavior SHALL remain equivalent to:

```python
source_template_scale_map(
    source_pixels: Array,
    nx: int,
    ny: int,
    alpha: float | Array,
    floor: float | Array,
    *,
    eps: float = 1.0e-10,
) -> Array | None
```

For traced `alpha` or `floor`, the implementation SHALL NOT call Python `float()` on those values inside the JAX computation path.

#### Scenario: Vectorized likelihood varies scale map by sample

- **WHEN** a vectorized operator likelihood evaluates a batch with different `adaptive_reg_alpha` or `adaptive_reg_floor` values
- **THEN** each batch element SHALL compute its regularization scale map from the shared S0 template and that sample's hyperparameter values

#### Scenario: Static zero alpha keeps uniform fast path

- **WHEN** `adaptive_reg_alpha` is a static scalar equal to zero
- **THEN** scale construction SHALL return `None` or an equivalent uniform-regularization fast path

### Requirement: Fixed source bbox for source-template inference

The operator pixelized image probability model SHALL support a fixed source-plane bbox for source-template adaptive regularization.

The API SHALL expose a configuration equivalent to:

```python
PixelizedImageProbModelOperator(
    ...,
    fixed_source_bbox: tuple[float, float, float, float] | None = None,
    fixed_reg_scale: Array | None = None,
    fixed_reg_template: Array | None = None,
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

### Requirement: Operator likelihood can generate scale maps from an S0 template

The operator pixelized image probability model SHALL accept a fixed source-template input for adaptive scale-map generation.

When `fixed_reg_template` is provided and adaptive regularization is enabled, the operator likelihood SHALL compute the per-pixel scale map using the current `adaptive_reg_alpha` and `adaptive_reg_floor` values. When `fixed_reg_scale` is provided without `fixed_reg_template`, the likelihood SHALL preserve the existing fixed-scale behavior.

#### Scenario: Dynamic template scale path

- **WHEN** the operator likelihood is configured with `fixed_reg_template` and dynamic adaptive hyperparameters
- **THEN** `_get_reg_scale()` SHALL return a scale map computed from the S0 template and the current hyperparameter values

#### Scenario: Fixed scale compatibility

- **WHEN** the operator likelihood is configured with `fixed_reg_scale` and no `fixed_reg_template`
- **THEN** `_get_reg_scale()` SHALL return the fixed scale map as before

#### Scenario: Missing adaptive template inputs

- **WHEN** adaptive regularization is enabled through dynamic adaptive hyperparameters and neither `fixed_reg_scale` nor `fixed_reg_template` is provided
- **THEN** likelihood construction or scale access SHALL fail with a clear error before producing evidence values

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

### Requirement: Continuously-differentiable scale formula

The final per-pixel regularization scale SHALL be computed as:

`scale_i = floor + (1 - floor) / (1 + alpha * b_norm_i)`

where `alpha >= 0`, `0 < floor <= 1`, and `b_norm_i >= 0`.

This formula SHALL be continuously differentiable with respect to `b_norm_i` for all finite inputs.

#### Scenario: Darkest pixel

- **WHEN** `b_norm_i = 0`
- **THEN** `scale_i = 1.0` regardless of `alpha` and `floor`

#### Scenario: Brightest pixel

- **WHEN** `b_norm_i` is very large
- **THEN** `scale_i` SHALL approach `floor` asymptotically

#### Scenario: Mean-brightness pixel

- **WHEN** `b_norm_i = 1`, `alpha = 1`, and `floor = 0.1`
- **THEN** `scale_i = 0.1 + 0.9 / 2 = 0.55`

### Requirement: Scale application to regularization matrix

The per-pixel `scale` array SHALL be applied to finite-difference regularization through edge weights, using geometric-mean interpolation of adjacent pixel scales:

`w_edge(i,j) = sqrt(scale_i * scale_j)`

This preserves symmetry and positive semi-definiteness of the regularizer.

#### Scenario: Uniform scale

- **WHEN** all `scale_i = 1`
- **THEN** all edge weights SHALL be 1, recovering the uniform regularizer

#### Scenario: Adjacent bright and dark pixel

- **WHEN** `scale_i = floor` for a bright pixel adjacent to a dark pixel where `scale_j = 1`
- **THEN** the shared edge weight SHALL be `sqrt(floor * 1) = sqrt(floor)`, intermediate between the two extremes

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

### Requirement: Stage-m1 adaptive hyperparameter posterior artifact

The adaptive regularization demo pipeline SHALL save the stage-m1 fitted regularization hyperparameter posterior.

The stage-m1 artifact SHALL include posterior samples, weights, parameter names, log evidence, posterior medians, S0 fingerprint, and elapsed time. The posterior median dictionary SHALL include:

- `log_lambda_reg`
- `adaptive_reg_alpha`
- `adaptive_reg_floor`

#### Scenario: Stage-m1 output contains fitted medians

- **WHEN** stage-m1 completes successfully
- **THEN** `stage_m1.pkl` SHALL contain posterior medians for `log_lambda_reg`, `adaptive_reg_alpha`, and `adaptive_reg_floor`

#### Scenario: Cached stage-m1 output validates S0

- **WHEN** the demo is run with `--skip-done` and cached stage-m1 output exists
- **THEN** the pipeline SHALL reuse it only when the stored S0 fingerprint matches the current S0 package

### Requirement: Stage-m1 and stage-m2 consume the same S0 grid

Stages m1 and m2 in the adaptive regularization demo SHALL use the same `S0` package and fixed source bbox.

Stage-m1 SHALL fit `log_lambda_reg`, `adaptive_reg_alpha`, and `adaptive_reg_floor` using Nautilus with SIE+shear mass parameters fixed at the stage-A medians. For each likelihood evaluation, stage-m1 SHALL generate the adaptive regularization scale map from the fixed S0 source template and the current sampled `adaptive_reg_alpha` and `adaptive_reg_floor` values.

Stage-m2 SHALL keep `log_lambda_reg`, `adaptive_reg_alpha`, and `adaptive_reg_floor` fixed at the stage-m1 posterior median values while sampling EPL+shear mass parameters. Stage-m2 SHALL use the same fixed S0 source bbox and SHALL generate the scale map from the same S0 source template and fixed median hyperparameters.

#### Scenario: Stage-m1 samples adaptive hyperparameters with S0 template

- **WHEN** stage-m1 evaluates a Nautilus proposal
- **THEN** the proposal SHALL vary `log_lambda_reg`, `adaptive_reg_alpha`, and `adaptive_reg_floor` while reusing the fixed S0 source bbox and source template

#### Scenario: Stage-m2 mass inference with M1 median adaptive hyperparameters

- **WHEN** stage-m2 samples mass parameters
- **THEN** the source adaptive prior structure SHALL be fixed by the S0 template and the stage-m1 posterior median values for `log_lambda_reg`, `adaptive_reg_alpha`, and `adaptive_reg_floor`

#### Scenario: Stage-m2 cache validates fixed hyperparameters

- **WHEN** the demo is run with `--skip-done` and cached stage-m2 output exists
- **THEN** the pipeline SHALL reuse it only when the S0 fingerprint and fixed regularization hyperparameter values match the current stage-m1 medians

### Requirement: JIT-compatible fixed adaptive inputs

Fixed source bbox and fixed source-template inputs SHALL be concrete values available before `make_likelihood` traces or compiles the likelihood.

The implementation SHALL avoid Python-side mutation from inside JIT-traced likelihood evaluation when using the `S0` path. Fixed arrays SHALL be captured as constants or model attributes before tracing, while `adaptive_reg_alpha` and `adaptive_reg_floor` MAY be traced scalar parameters supplied by sampler proposals.

#### Scenario: Compiled likelihood uses fixed arrays and traced scalars

- **WHEN** `make_likelihood(..., vectorized=True)` traces a likelihood configured with `fixed_source_bbox`, `fixed_reg_template`, and dynamic adaptive hyperparameters
- **THEN** the compiled function SHALL evaluate without attempting to write cache state during tracing and SHALL use the traced scalar hyperparameters to compute the scale map

#### Scenario: Fixed template remains stable across vectorized batch

- **WHEN** a vectorized likelihood evaluates a batch of stage-m1 proposals
- **THEN** each batch element SHALL share the same fixed source bbox and S0 source template while using its own `log_lambda_reg`, `adaptive_reg_alpha`, and `adaptive_reg_floor`
