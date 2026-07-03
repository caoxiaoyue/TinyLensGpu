# stage-posterior-transfer Specification

## Purpose
Define the lightweight stage posterior API used to pass sampled parameter information between modeling stages without depending on hand-written parameter maps or live likelihood objects.
## Requirements
### Requirement: Stage posterior binds samples to likelihood schema

The system SHALL provide an Inference-layer stage posterior object that binds posterior `samples` and `weights` to the dynamic-parameter schema of the likelihood-like module that produced them.

The public constructor SHALL include:

```python
StagePosterior.from_likelihood(
    likelihood,
    samples: np.ndarray,
    weights: np.ndarray,
    *,
    log_z: float | None = None,
    factor_std: float = 5.0,
) -> StagePosterior
```

`samples` MUST be a 2D numeric array with shape `(n_samples, n_params)`. `weights` MUST be a 1D numeric array with shape `(n_samples,)` and SHALL be normalized internally. The parameter schema MUST be derived from the same ordering used by `extract_prior_specs(likelihood)` or `likelihood.get_dynamic_params()`.

#### Scenario: Construct from sampled likelihood

- **WHEN** a likelihood with three dynamic `ParamU` parameters produces posterior samples with shape `(100, 3)` and weights with shape `(100,)`
- **THEN** `StagePosterior.from_likelihood(likelihood, samples, weights)` maps each sample column to the corresponding likelihood dynamic parameter name in extraction order

#### Scenario: Reject mismatched sample columns

- **WHEN** a likelihood exposes three dynamic parameters and posterior samples have shape `(100, 2)`
- **THEN** constructing the stage posterior MUST raise a clear `ValueError`

#### Scenario: Reject duplicate parameter names

- **WHEN** a likelihood exposes duplicate dynamic parameter names
- **THEN** constructing the stage posterior MUST raise a clear `ValueError` rather than silently choosing one column

### Requirement: Stage posterior exposes posterior summaries

The stage posterior object SHALL expose weighted posterior summary methods keyed by parameter name:

```python
stage.median(name: str) -> float
stage.std(name: str) -> float
stage.median_std(name: str) -> tuple[float, float]
stage.medians() -> dict[str, float]
```

All returned scalar values SHALL be Python `float` values suitable for constructing `ParamU` instances and serializable cache payloads.

#### Scenario: Query weighted median

- **WHEN** a caller requests `stage.median("theta_E")`
- **THEN** the method returns the weighted median of the posterior sample column bound to `theta_E`

#### Scenario: Missing parameter summary

- **WHEN** a caller requests a summary for a name that is not present in the stage schema
- **THEN** the method MUST raise a clear `KeyError` listing available parameter names

### Requirement: Stage posterior creates fixed inherited parameters

The stage posterior object SHALL create static inherited `ParamU` parameters fixed at posterior median values.

The public API SHALL include:

```python
stage.fixed(
    name: str,
    *,
    target: str | None = None,
) -> ParamU
```

The returned `ParamU` SHALL use `target` as its name when provided, otherwise `name`. The returned parameter SHALL have value equal to `stage.median(name)` and SHALL already be marked static with `to_static()`.

#### Scenario: Fixed mass parameter inheritance

- **WHEN** a stage posterior contains `center_x_mass` and a caller requests `stage.fixed("center_x_mass", target="center_x")`
- **THEN** the result is a static `ParamU` named `center_x` with value equal to the weighted median of `center_x_mass`

### Requirement: Stage posterior creates Gaussian inherited parameters

The stage posterior object SHALL create dynamic inherited `ParamU` parameters with Gaussian priors centered on posterior medians.

The public API SHALL include:

```python
stage.gaussian(
    name: str,
    *,
    model: str,
    attr: str,
    target: str | None = None,
    limits: Sequence[float] | None = None,
) -> ParamU
```

The returned `ParamU` SHALL use prior type `"gaussian"`, initial value equal to the weighted median, `prior_settings=[median, sigma]`, and supplied hard `limits`. It SHALL already be marked dynamic with `to_dynamic()`. `sigma` SHALL preserve the current conservative rule: `max(factor_std * posterior_std, empirical_width(model, attr))`, where relative empirical widths are evaluated against `abs(median)`.

#### Scenario: Gaussian EPL inheritance

- **WHEN** a stage posterior contains `e1_mass` and a caller requests `stage.gaussian("e1_mass", target="e1", model="EPL", attr="e1", limits=[-1.0, 1.0])`
- **THEN** the result is a dynamic Gaussian-prior `ParamU` named `e1` centered on the weighted median of `e1_mass`

#### Scenario: Unknown empirical width key

- **WHEN** a caller requests Gaussian inheritance with an unknown `(model, attr)` pair
- **THEN** the method MUST raise a clear `KeyError`

### Requirement: Stage posterior supports lightweight cache payloads

The system SHALL support reconstructing a stage posterior from a lightweight serialized schema without requiring the original live likelihood object.

The public API SHALL include a schema-based constructor or equivalent cache-load path that accepts `samples`, `weights`, parameter names or prior specs, and optional `log_z`.

#### Scenario: Rehydrate cached stage

- **WHEN** a cached stage payload contains samples, weights, log evidence, and serialized parameter schema
- **THEN** the system can reconstruct a stage posterior that supports summary and inherited-parameter factory methods without unpickling a likelihood object

### Requirement: GaussianPriorPasser is removed

The system SHALL remove `GaussianPriorPasser` as a public API and migrate internal examples to the stage posterior API.

#### Scenario: Public inference exports

- **WHEN** a caller imports public symbols from `TinyLensGpu.Inference`
- **THEN** `StagePosterior` or the chosen replacement name is exported and `GaussianPriorPasser` is no longer exported
