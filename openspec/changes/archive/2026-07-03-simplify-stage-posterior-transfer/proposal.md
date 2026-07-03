## Why

Multi-stage pixelized-source pipelines currently pass information between stages through hand-maintained `samples`, `weights`, `param_names`, median dictionaries, and repeated `GaussianPriorPasser.gaussian(...)` calls. This makes physically identical parameter inheritance verbose and fragile, especially when a previous stage's likelihood already contains the dynamic-parameter schema needed to interpret posterior sample columns.

## What Changes

- Introduce a stage posterior transfer API that binds posterior samples and weights to the likelihood object that produced them.
- Replace the narrow `GaussianPriorPasser` concept with a clearer stage-result or stage-posterior abstraction that can create inherited `ParamU` parameters from a previous stage.
- Support both fixed median inheritance and dynamic Gaussian-prior inheritance through the same API.
- Allow `_run_sampler(...)` or related pipeline helpers to return a structured stage result carrying the likelihood schema, samples, weights, log evidence, parameter specs, and posterior summary helpers.
- Use the previous likelihood's `get_dynamic_params()` / prior extraction order as the authoritative mapping from posterior sample columns to parameter names.
- Keep pipeline model construction explicit: stage code should still assemble `SIE`, `EPL`, `Shear`, `GaussianEllipse`, `PixelizedSourceModel`, and likelihood objects directly.
- **BREAKING**: Remove `GaussianPriorPasser` rather than keeping it as an alias; example pipelines and imports will migrate to the new abstraction.

## Non-goals

- Do not introduce a full pipeline DSL or automatic stage builder that hides physical model construction.
- Do not change JAX likelihood evaluation, source inversion math, sampler behavior, or GPU execution paths.
- Do not serialize complete likelihood objects as the long-term cache format for `--skip-done`; cached stages should remain lightweight and reconstructable.
- Do not change existing empirical-width rules except where necessary to expose them through the new API.

## Capabilities

### New Capabilities

- `stage-posterior-transfer`: Defines how a completed inference stage exposes posterior samples, likelihood-derived parameter schema, posterior summary values, and inherited `ParamU` factories for subsequent stages.

### Modified Capabilities

- None.

## Impact

- Affects `TinyLensGpu/Inference/prior_passing.py` and likely adjacent inference utilities that extract prior specs or summarize posteriors.
- Affects pipeline examples under `examples/pix_src_demo_operator/pipe/`, especially `galan24_test/model.py`, where stage-A, stage-M1, stage-M2, and stage-M3 currently repeat manual inheritance logic.
- May affect legacy examples importing `GaussianPriorPasser`; these imports and calls should migrate to the new stage-posterior API.
- Runtime performance should be neutral: the change is Python-side pipeline construction and posterior bookkeeping, not JIT-compiled forward-model or solver logic.
