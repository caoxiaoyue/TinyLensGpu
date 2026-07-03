## Context

Current multi-stage pipeline examples manually pass `samples`, `weights`, and `param_names` between stages. The parameter names are derived from `make_prior_transformation(likelihood)` / `likelihood.get_dynamic_params()`, but that schema is separated from the likelihood object after sampling. Downstream stages then reconstruct the mapping through ad hoc median dictionaries and `GaussianPriorPasser` calls.

This is an Inference-layer concern. Physical models and likelihood classes already expose the Caskade dynamic-parameter ordering; the new abstraction should reuse that ordering instead of introducing a second naming source.

## Goals / Non-Goals

**Goals:**

- Represent a completed stage as posterior samples and weights bound to the likelihood-derived dynamic-parameter schema.
- Replace `GaussianPriorPasser` with a clearer `StagePosterior` or `StageResult` API.
- Provide factories for inherited `ParamU` objects:
  - fixed at posterior median and already static;
  - Gaussian prior centered on posterior median and already dynamic.
- Keep explicit physical model assembly in pipeline code.
- Preserve current empirical-width behavior for Gaussian inheritance.
- Keep runtime and GPU behavior unchanged; this is Python-side bookkeeping before JIT execution.

**Non-Goals:**

- No full pipeline DSL or automatic `SIE`/`EPL`/`Shear` stage builder.
- No changes to source inversion, likelihood math, samplers, or JAX kernels.
- No requirement to pickle full likelihood objects for `--skip-done`.
- No backward-compatible `GaussianPriorPasser` alias; imports and examples should migrate.

## Decisions

1. Put the new abstraction in the Inference layer.

   `StagePosterior` should live near `prior_passing.py` or replace it with a more accurate module name such as `stage_posterior.py`. It depends on `ParamU`, `extract_prior_specs`, weighted quantiles, and empirical-width rules, all of which are inference concerns.

   Alternative considered: place this in `ObservationModel` because likelihood objects provide schema. Rejected because the object consumes likelihood schema but creates inference parameters and posterior summaries.

2. Use likelihood schema as the authoritative sample-column mapping.

   The primary constructor should accept a likelihood-like module plus samples and weights:

   ```python
   stage = StagePosterior.from_likelihood(likelihood, samples, weights, log_z=log_z)
   ```

   It should derive parameter names and prior specs from `extract_prior_specs(likelihood)` or `likelihood.get_dynamic_params()`. A secondary constructor may accept an already serialized schema for cached stages.

   Alternative considered: keep passing explicit `param_names`. Rejected as the main API because it duplicates data already present in the likelihood and can drift from sample order.

3. Keep target semantics explicit for inherited Gaussian parameters.

   Calls still need `model` and `attr` because a posterior column like `e1_mass` can target an EPL constructor attribute `e1`, and empirical width lookup uses semantic model/attribute keys:

   ```python
   stage.gaussian("e1_mass", target="e1", model="EPL", attr="e1", limits=[-1, 1])
   ```

   The API should not infer physical semantics from names until components carry richer metadata.

4. Return mode-ready `ParamU` instances.

   `stage.fixed(name)` returns a `ParamU` with median value and calls `to_static()`. `stage.gaussian(...)` returns a `ParamU` with Gaussian prior metadata, hard limits, median initial value, and calls `to_dynamic()`.

   This removes repeated `for p in (...): p.to_dynamic()` blocks for inherited parameters.

5. Separate lightweight cache payloads from live likelihood objects.

   A live `StagePosterior` can hold `likelihood` for immediate chaining. Cached stage files should store samples, weights, log evidence, parameter names/specs, medians, and summary metadata, not the full likelihood object. Rehydration should use the serialized schema.

## Risks / Trade-offs

- [Risk] Cached stages may not have the original likelihood object available. → Provide a schema-based constructor and persist parameter names/specs alongside samples.
- [Risk] Removing `GaussianPriorPasser` breaks example imports. → Migrate examples in the same change and add focused tests covering the replacement behavior.
- [Risk] Duplicate parameter names in Caskade dynamic params would make name lookup ambiguous. → Validate uniqueness in `StagePosterior` construction and raise a clear error.
- [Risk] Automatic `to_dynamic()` / `to_static()` may surprise callers that expect raw `ParamU`. → Document method behavior and provide explicit helper names (`fixed`, `gaussian`) rather than a vague generic factory.
- [Risk] Holding likelihood references could increase memory pressure if many stages are retained. → Store only one or a few active stage results in examples, and allow `without_likelihood()` or cache payloads to keep only samples and schema.

## Migration Plan

1. Add the new stage-posterior abstraction and tests.
2. Remove `GaussianPriorPasser` export and migrate imports.
3. Update representative pipelines, starting with `examples/pix_src_demo_operator/pipe/galan24_test/model.py`.
4. Update other examples using `GaussianPriorPasser` in a mechanical pass.
5. Run focused inference tests and fast example-level import/build checks.

Rollback is straightforward before implementation release: restore `prior_passing.py` and revert example migrations.

## Open Questions

- Should `_run_sampler(...)` return `StagePosterior` directly, or a richer `StageResult` that also carries `likelihood`, `log_z`, timing, and plotting metadata?
- Should the API expose `inherit_many(...)` in the first implementation, or start with scalar `fixed(...)` / `gaussian(...)` methods and add batch specs later?
- Should explicit free-parameter helpers such as `free_uniform(...)` and `free_truncated_gaussian(...)` live beside `StagePosterior`, or remain direct `ParamU(...)` calls in pipeline code?
