## ADDED Requirements

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

### Requirement: Operator likelihood can generate scale maps from an S0 template

The operator pixelized image probability model SHALL accept a fixed source-template input for adaptive scale-map generation.

The API SHALL expose a configuration equivalent to:

```python
PixelizedImageProbModelOperator(
    ...,
    fixed_source_bbox: tuple[float, float, float, float] | None = None,
    fixed_reg_scale: Array | None = None,
    fixed_reg_template: Array | None = None,
)
```

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

## MODIFIED Requirements

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
