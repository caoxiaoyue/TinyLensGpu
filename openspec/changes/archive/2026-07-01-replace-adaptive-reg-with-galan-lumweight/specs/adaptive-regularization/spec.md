## MODIFIED Requirements

### Requirement: Source-template adaptive scale maps

The system SHALL support adaptive regularization scale maps derived from a fixed pixelized source reconstruction template instead of image-plane seed rays.

The source-template scale builder SHALL accept `source_pixels` with shape `(ny * nx,)` or `(ny, nx)` and numeric dtype convertible to `jax.float32`. It SHALL compute a Galan-style luminosity-weighted precision scale:

1. `s_pos = max(source_pixels, 0)`
2. `s_ref = percentile(s_pos, 99.5)`
3. `u = clip(s_pos / max(s_ref, eps), 0, 1)`
4. `scale = exp(rho * (1 - u))`

The returned `scale` SHALL be flat with shape `(ny * nx,)`, finite, positive, and compatible with existing finite-difference regularization scale application.

#### Scenario: Positive source template creates stronger scale in faint pixels

- **WHEN** a fixed `S0` source template contains bright and dark source pixels and `adaptive_reg_rho > 0`
- **THEN** the derived scale map SHALL assign values near `1.0` to reference-bright pixels and larger values to fainter pixels

#### Scenario: Negative source pixels are clipped

- **WHEN** `S0` contains negative source-pixel values from an unconstrained linear solve
- **THEN** the scale builder SHALL clip those values to zero before percentile normalization and SHALL NOT produce negative or non-finite scale values

#### Scenario: Uniform regularization rho

- **WHEN** `adaptive_reg_rho == 0`
- **THEN** source-template scale construction SHALL return `None` or an equivalent all-ones uniform-regularization fast path rather than materializing a non-uniform scale map

#### Scenario: Percentile reference clips outliers

- **WHEN** `S0` contains values above the 99.5 percentile reference brightness
- **THEN** the normalized brightness `u` for those values SHALL be clipped to `1.0` and their precision scale SHALL be `1.0`

### Requirement: No additional smoothing for source-template scale maps

The source-template adaptive scale path SHALL NOT apply Gaussian smoothing to `S0` before percentile normalization by default.

This preserves the empirical-Bayes interpretation that stage-m0 regularization controls the smoothness of the source template used to derive the Galan-style scale map.

#### Scenario: Structured S0 template

- **WHEN** `S0` contains a compact bright source feature that was already reconstructed under a regularization prior
- **THEN** the source-template scale map SHALL preserve that feature's pixel-scale contrast through clipping, 99.5-percentile normalization, and the Galan-style precision-scale formula without an extra smoothing convolution

### Requirement: Fittable source-template adaptive hyperparameters

The system SHALL allow the source-template adaptive regularization hyperparameter to be represented as a sampler-visible parameter.

`PixelizedSourceModel` SHALL accept `adaptive_reg_rho` as either a scalar value or a `ParamU` instance. When supplied as a dynamic `ParamU` instance, it SHALL be returned through the model's Caskade dynamic parameter traversal and SHALL be consumable by `make_prior_transformation`.

The parameter constraint SHALL be:

- `adaptive_reg_rho >= 0`

`adaptive_reg_alpha` and `adaptive_reg_floor` SHALL NOT be active source-template adaptive scale-map hyperparameters for this Galan-style path.

#### Scenario: Prior extraction includes adaptive rho

- **WHEN** a pixelized source model has dynamic `ParamU` values for `log_lambda_reg` and `adaptive_reg_rho`
- **THEN** prior extraction SHALL include both parameter names and their configured prior metadata

#### Scenario: Scalar adaptive rho remains static

- **WHEN** a pixelized source model is constructed with scalar `adaptive_reg_rho`
- **THEN** that value SHALL remain static configuration and SHALL NOT add a sampler dimension

#### Scenario: Negative rho is rejected

- **WHEN** a caller constructs a pixelized source model with `adaptive_reg_rho < 0`
- **THEN** construction SHALL fail with a clear validation error

### Requirement: JAX-compatible source-template scale generation from traced hyperparameters

The source-template scale builder SHALL support JAX-traced scalar values for `rho` when the source template shape is statically valid.

The public scale-builder behavior SHALL remain equivalent to:

```python
source_template_scale_map(
    source_pixels: Array,
    n: int,
    rho: float | Array,
    *,
    ref_percentile: float = 99.5,
    eps: float = 1.0e-10,
) -> Array | None
```

For traced `rho`, the implementation SHALL NOT call Python `float()` on that value inside the JAX computation path. `ref_percentile` SHALL be treated as static configuration.

#### Scenario: Vectorized likelihood varies scale map by sample

- **WHEN** a vectorized operator likelihood evaluates a batch with different `adaptive_reg_rho` values
- **THEN** each batch element SHALL compute its regularization scale map from the shared S0 template and that sample's `rho` value

#### Scenario: Static zero rho keeps uniform fast path

- **WHEN** `adaptive_reg_rho` is a static scalar equal to zero
- **THEN** scale construction SHALL return `None` or an equivalent uniform-regularization fast path

#### Scenario: Traced zero rho produces uniform scale

- **WHEN** `adaptive_reg_rho` is traced and happens to evaluate to zero
- **THEN** scale construction SHALL produce an all-ones scale map without Python-side branching on the traced value

### Requirement: Operator likelihood can generate scale maps from an S0 template

The operator pixelized image probability model SHALL accept a fixed source-template input for adaptive scale-map generation.

When `fixed_reg_template` is provided and adaptive regularization is enabled, the operator likelihood SHALL compute the per-pixel scale map using the current `adaptive_reg_rho` value. When `fixed_reg_scale` is provided without `fixed_reg_template`, the likelihood SHALL preserve the existing fixed-scale behavior.

#### Scenario: Dynamic template scale path

- **WHEN** the operator likelihood is configured with `fixed_reg_template` and dynamic `adaptive_reg_rho`
- **THEN** `_get_reg_scale()` SHALL return a scale map computed from the S0 template and the current `rho` value

#### Scenario: Fixed scale compatibility

- **WHEN** the operator likelihood is configured with `fixed_reg_scale` and no `fixed_reg_template`
- **THEN** `_get_reg_scale()` SHALL return the fixed scale map as before

#### Scenario: Missing adaptive template inputs

- **WHEN** adaptive regularization is enabled through dynamic `adaptive_reg_rho` and neither `fixed_reg_scale` nor `fixed_reg_template` is provided
- **THEN** likelihood construction or scale access SHALL fail with a clear error before producing evidence values

### Requirement: Continuously-differentiable scale formula

The final per-pixel regularization precision scale SHALL be computed as:

`scale_i = exp(rho * (1 - u_i))`

where `rho >= 0` and `u_i = clip(s_pos_i / max(percentile(s_pos, 99.5), eps), 0, 1)`.

This formula SHALL be continuously differentiable with respect to unclipped finite `u_i` values and SHALL remain finite for all finite `rho` values within configured priors.

#### Scenario: Reference-bright pixel

- **WHEN** `u_i = 1`
- **THEN** `scale_i = 1.0` regardless of `rho`

#### Scenario: Darkest pixel

- **WHEN** `u_i = 0`
- **THEN** `scale_i = exp(rho)`

#### Scenario: Half-reference pixel

- **WHEN** `u_i = 0.5` and `rho = 2`
- **THEN** `scale_i = exp(1)`

#### Scenario: Uniform rho

- **WHEN** `rho = 0`
- **THEN** `scale_i = 1.0` for every source pixel

### Requirement: Scale application to regularization matrix

The per-pixel `scale` array SHALL be applied to finite-difference regularization through edge weights, using geometric-mean interpolation of adjacent pixel scales:

`w_edge(i,j) = sqrt(scale_i * scale_j)`

This preserves symmetry and positive semi-definiteness of the regularizer. Larger scale values SHALL mean stronger regularization precision.

#### Scenario: Uniform scale

- **WHEN** all `scale_i = 1`
- **THEN** all edge weights SHALL be 1, recovering the uniform regularizer

#### Scenario: Adjacent bright and dark pixel

- **WHEN** `scale_i = 1` for a reference-bright pixel adjacent to a dark pixel where `scale_j = exp(rho)`
- **THEN** the shared edge weight SHALL be `exp(rho / 2)`, intermediate between the two extremes

### Requirement: Stage-m1 adaptive hyperparameter posterior artifact

The adaptive regularization demo pipeline SHALL save the stage-m1 fitted regularization hyperparameter posterior.

The stage-m1 artifact SHALL include posterior samples, weights, parameter names, log evidence, posterior medians, S0 fingerprint, and elapsed time. The posterior median dictionary SHALL include:

- `log_lambda_reg`
- `adaptive_reg_rho`

#### Scenario: Stage-m1 output contains fitted medians

- **WHEN** stage-m1 completes successfully
- **THEN** `stage_m1.pkl` SHALL contain posterior medians for `log_lambda_reg` and `adaptive_reg_rho`

#### Scenario: Cached stage-m1 output validates S0

- **WHEN** the demo is run with `--skip-done` and cached stage-m1 output exists
- **THEN** the pipeline SHALL reuse it only when the stored S0 fingerprint matches the current S0 package

### Requirement: Stage-m1 and stage-m2 consume the same S0 grid

Stages m1 and m2 in the adaptive regularization demo SHALL use the same `S0` package and fixed source bbox.

Stage-m1 SHALL fit `log_lambda_reg` and `adaptive_reg_rho` using Nautilus with SIE+shear mass parameters fixed at the stage-A medians. For each likelihood evaluation, stage-m1 SHALL generate the adaptive regularization scale map from the fixed S0 source template and the current sampled `adaptive_reg_rho` value.

Stage-m2 SHALL keep `log_lambda_reg` and `adaptive_reg_rho` fixed at the stage-m1 posterior median values while sampling EPL+shear mass parameters. Stage-m2 SHALL use the same fixed S0 source bbox and SHALL generate the scale map from the same S0 source template and fixed median hyperparameters.

#### Scenario: Stage-m1 samples adaptive hyperparameters with S0 template

- **WHEN** stage-m1 evaluates a Nautilus proposal
- **THEN** the proposal SHALL vary `log_lambda_reg` and `adaptive_reg_rho` while reusing the fixed S0 source bbox and source template

#### Scenario: Stage-m2 mass inference with M1 median adaptive hyperparameters

- **WHEN** stage-m2 samples mass parameters
- **THEN** the source adaptive prior structure SHALL be fixed by the S0 template and the stage-m1 posterior median values for `log_lambda_reg` and `adaptive_reg_rho`

#### Scenario: Stage-m2 cache validates fixed hyperparameters

- **WHEN** the demo is run with `--skip-done` and cached stage-m2 output exists
- **THEN** the pipeline SHALL reuse it only when the S0 fingerprint and fixed regularization hyperparameter values match the current stage-m1 medians

### Requirement: JIT-compatible fixed adaptive inputs

Fixed source bbox and fixed source-template inputs SHALL be concrete values available before `make_likelihood` traces or compiles the likelihood.

The implementation SHALL avoid Python-side mutation from inside JIT-traced likelihood evaluation when using the `S0` path. Fixed arrays SHALL be captured as constants or model attributes before tracing, while `adaptive_reg_rho` MAY be a traced scalar parameter supplied by sampler proposals.

#### Scenario: Compiled likelihood uses fixed arrays and traced scalars

- **WHEN** `make_likelihood(..., vectorized=True)` traces a likelihood configured with `fixed_source_bbox`, `fixed_reg_template`, and dynamic `adaptive_reg_rho`
- **THEN** the compiled function SHALL evaluate without attempting to write cache state during tracing and SHALL use the traced scalar hyperparameter to compute the scale map

#### Scenario: Fixed template remains stable across vectorized batch

- **WHEN** a vectorized likelihood evaluates a batch of stage-m1 proposals
- **THEN** each batch element SHALL share the same fixed source bbox and S0 source template while using its own `log_lambda_reg` and `adaptive_reg_rho`
