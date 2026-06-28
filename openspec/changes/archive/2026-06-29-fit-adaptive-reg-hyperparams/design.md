## Context

Adaptive regularization currently uses a stage-m0 `S0` source reconstruction to build a fixed per-pixel regularization scale map. In `model_adpt_reg.py`, stage-m1 then grid-searches only `log_lambda_reg`, while `adaptive_reg_alpha` and `adaptive_reg_floor` remain module-level constants. The operator likelihood can already consume a fixed source bbox and fixed scale map, but the scale map is computed before likelihood tracing and is not parameterized by sampler values.

This change spans the PhysicalModel, ObservationModel, Inference, and example-pipeline layers:

- PhysicalModel: `PixelizedSourceModel` must expose adaptive hyperparameters as Caskade/`ParamU` parameters when requested.
- ObservationModel: `PixelizedImageProbModelOperator` must generate source-template scale maps from fixed S0 data and current parameter values in a JAX-compatible path.
- Inference: existing `make_prior_transformation` and `make_likelihood` should discover and evaluate the new dynamic parameters without a new sampler abstraction.
- Example pipeline: stage-m1 becomes a Nautilus hyperparameter fit; stage-m2 conditions on the stage-m1 posterior medians.

## Goals / Non-Goals

**Goals:**

- Fit `log_lambda_reg`, `adaptive_reg_alpha`, and `adaptive_reg_floor` in stage-m1 by maximizing/marginalizing the pixelized-source evidence with Nautilus.
- Keep stage-m2 efficient by fixing all pixelized-source regularization hyperparameters to stage-m1 posterior medians while sampling EPL+shear mass parameters.
- Preserve vectorized JAX likelihood execution for sampler batches.
- Keep S0 source-template adaptive regularization as the only active adaptive path.

**Non-Goals:**

- No recursive source-solve/scale-map/source-solve adaptive update inside one likelihood evaluation.
- No stage-m2 marginalization over regularization hyperparameters.
- No revival of seed-ray adaptive scale construction or freeze/unfreeze APIs.
- No dense-backend adaptive regularization implementation.

## Decisions

### Decision: Represent adaptive hyperparameters as optional `ParamU` fields

`PixelizedSourceModel` will accept `adaptive_reg_alpha` and `adaptive_reg_floor` as either scalars or `ParamU`, matching `log_lambda_reg` and `kernel_scale`. Scalar values remain static configuration. `ParamU` values can be marked dynamic and discovered by `make_prior_transformation`.

Alternative considered: place the hyperparameters only on `PixelizedImageProbModelOperator`. That would avoid changing the source model, but it would bypass the existing Caskade parameter ownership pattern and require special-case prior extraction.

### Decision: Add a source-template scale path, not a fixed-scale-only path

The operator likelihood will support fixed source-template inputs in addition to the existing fixed scale map:

```python
PixelizedImageProbModelOperator(
    ...,
    fixed_source_bbox=(xmin, xmax, ymin, ymax),
    fixed_reg_scale=scale_or_none,
    fixed_reg_template=source_pixels_or_none,
)
```

When `fixed_reg_template` is provided and adaptive regularization is enabled, `_get_reg_scale()` will compute the scale map using the current `adaptive_reg_alpha` and `adaptive_reg_floor` values. When only `fixed_reg_scale` is provided, the current fixed-scale behavior remains available for compatibility.

Alternative considered: precompute many scale maps on a grid of alpha/floor values. That is incompatible with continuous sampler proposals and would add interpolation error.

### Decision: Make scale-map construction tracer-friendly

`source_template_scale_map` will retain static shape validation but avoid Python `float()` coercion for `alpha` and `floor` in the JAX computation path. The shape of `source_pixels` is static; the scalar values can be JAX tracers. Runtime finite/positive validation of traced scale values should remain in the existing regularization builder path where possible.

Uniform `alpha == 0` is a special fast path for scalar values. For traced values, the implementation should prefer a JAX expression that returns an all-ones-equivalent scale or uses `jnp.where`, rather than Python branching on a tracer.

### Decision: M1 uses Nautilus; M2 fixes M1 posterior medians

Stage-m1 will build a likelihood with fixed SIE+shear from stage-A medians and dynamic:

- `log_lambda_reg`
- `adaptive_reg_alpha`
- `adaptive_reg_floor`

Stage-m2 will build an EPL+shear likelihood with these three source hyperparameters static at stage-m1 posterior medians. This keeps stage-m2 dimension and runtime close to the current pipeline while making the adaptive prior shape evidence-driven.

Alternative considered: jointly sampling mass and regularization hyperparameters in stage-m2. That is statistically broader but more expensive and not the requested staged empirical-Bayes workflow.

### Decision: Cache metadata includes fitted hyperparameters

Stage-m1 output will save posterior medians and S0 fingerprint. Stage-m2 cache validation will compare both the S0 fingerprint and the fixed hyperparameter values used to build the likelihood. This prevents `--skip-done` from reusing stale mass posteriors after M1 hyperparameters change.

## Risks / Trade-offs

- [Risk] Evidence values in stage-m2 are conditional on M1-selected regularization hyperparameters, not marginalized over them. -> Mitigation: store and report the fixed M1 median values in stage-m2 output and summaries.
- [Risk] Very small floors can make regularization blocks ill-conditioned. -> Mitigation: use conservative priors and hard limits, and keep PCG non-convergence penalties.
- [Risk] Traced `alpha == 0` cannot use Python `None` fast paths. -> Mitigation: allow traced adaptive runs to materialize a scale array; retain `None` only for static scalar zero-alpha configurations.
- [Risk] Adding dynamic parameters to `PixelizedSourceModel` may affect parameter ordering. -> Mitigation: add targeted tests for prior names and update pipeline code to consume names by dictionary medians rather than fixed positions.

## Migration Plan

1. Add dynamic adaptive hyperparameter support in core modules with backwards-compatible scalar defaults.
2. Update operator scale generation to support S0 template plus traced alpha/floor.
3. Refactor stage-m1 and stage-m2 in `model_adpt_reg.py`.
4. Update cache payloads and diagnostic plots.
5. Add tests for prior extraction, vectorized likelihood compilation, scale-map variation, and pipeline metadata helpers.

Rollback is straightforward: scalar `adaptive_reg_alpha` and `adaptive_reg_floor` remain supported, and the existing fixed-scale path can continue to serve callers that precompute one scale map.
