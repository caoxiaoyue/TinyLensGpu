## Why

The current adaptive brightness regularization demo fixes the scale-map hyperparameters `adaptive_reg_alpha` and `adaptive_reg_floor` as constants, so the evidence optimization only selects the global regularization strength. This makes the adaptive prior shape a manual tuning choice even though the pipeline already uses Bayesian evidence to select source-regularization settings.

## What Changes

- Allow source-template adaptive regularization scale maps to be driven by fitted hyperparameters instead of only fixed Python constants.
- Make stage-m1 in `examples/pix_src_demo_operator/pipe/no_lens_light/model_adpt_reg.py` use Nautilus to sample `log_lambda_reg`, `adaptive_reg_alpha`, and `adaptive_reg_floor` with fixed SIE+shear mass parameters and the stage-m0 `S0` template.
- Store stage-m1 posterior samples, weights, parameter names, log evidence, posterior medians, and cache metadata for the fitted regularization hyperparameters.
- Keep stage-m2 focused on EPL+shear mass inference by fixing the pixelized-source regularization hyperparameters to the stage-m1 posterior medians.
- Update adaptive regularization plotting and pipeline summaries to report the fitted `lambda`, `alpha`, and `floor` values used by each stage.
- Preserve GPU/JAX vectorized likelihood behavior for sampler batches by generating scale maps from fixed S0 template data and traced scalar hyperparameters without Python-side mutation during JIT evaluation.

## Non-goals

- Do not implement recursive self-consistent adaptive regularization where each likelihood sample solves a source, rebuilds the scale map from that source, and solves again.
- Do not make stage-m2 marginalize over the regularization hyperparameters; stage-m2 conditions on the stage-m1 posterior median estimates.
- Do not reintroduce the retired image-plane seed-ray adaptive scale construction or freeze/unfreeze cache APIs.
- Do not change dense-backend adaptive regularization support in this change.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `adaptive-regularization`: Extend source-template adaptive regularization so `alpha` and `floor` can be fitted as regularization hyperparameters, with stage-m1 estimating `log_lambda_reg`, `adaptive_reg_alpha`, and `adaptive_reg_floor`, and stage-m2 consuming their posterior median values as fixed settings.

## Impact

- Affects `PixelizedSourceModel` parameter handling for adaptive regularization hyperparameters.
- Affects `source_template_scale_map` and related validation so JAX-traced scalar hyperparameters can generate per-sample scale maps from fixed S0 templates.
- Affects `PixelizedImageProbModelOperator` adaptive scale access, cache validation, and vectorized likelihood paths.
- Affects the no-lens-light adaptive regularization demo pipeline, especially stage-m1/stage-m2 outputs, skip-done cache checks, and diagnostic plots.
- Adds or updates unit/integration tests around prior extraction, JIT/vectorized operator likelihoods, stage artifact metadata, and scale-map variation with sampled `alpha` and `floor`.
