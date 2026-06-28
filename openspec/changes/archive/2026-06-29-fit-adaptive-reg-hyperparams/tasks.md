## 1. Core Parameter Support

- [x] 1.1 Update `PixelizedSourceModel` so `adaptive_reg_alpha` and `adaptive_reg_floor` accept either scalars or `ParamU`, preserve scalar defaults, and expose dynamic values through Caskade traversal when marked dynamic.
- [x] 1.2 Add unit tests proving prior extraction includes dynamic `log_lambda_reg`, `adaptive_reg_alpha`, and `adaptive_reg_floor`, while scalar adaptive values remain static.

## 2. JAX-Compatible Scale Generation

- [x] 2.1 Refactor `source_template_scale_map` so shape validation remains static but the scale formula accepts JAX-traced `alpha` and `floor` without Python `float()` coercion in the traced computation path.
- [x] 2.2 Add tests that scale maps vary with different traced `alpha` and `floor` values, preserve the static zero-alpha uniform fast path, and reject invalid static shapes or floors clearly.

## 3. Operator Likelihood Integration

- [x] 3.1 Extend `PixelizedImageProbModelOperator` with a `fixed_reg_template` input, validate its `(nx * ny,)` shape and dtype, and keep existing `fixed_reg_scale` behavior backwards compatible.
- [x] 3.2 Update `_get_reg_scale()` so template-backed adaptive runs compute scale maps from the fixed S0 template and current adaptive hyperparameter values, while fixed-scale runs still return the supplied scale map.
- [x] 3.3 Add operator tests for dynamic template scale generation, missing adaptive template errors, fixed-scale compatibility, and vectorized `make_likelihood(..., vectorized=True)` compilation. GPU is optional; CPU JAX is sufficient for unit coverage.

## 4. Stage M1 Pipeline Refactor

- [x] 4.1 Refactor `build_stage_m1_likelihood()` in `model_adpt_reg.py` to create dynamic `ParamU` values for `log_lambda_reg`, `adaptive_reg_alpha`, and `adaptive_reg_floor`, and pass the S0 source template plus fixed source bbox to the operator likelihood.
- [x] 4.2 Replace the stage-m1 lambda grid search with Nautilus sampling, using existing sampler helpers and saving samples, weights, parameter names, log evidence, posterior medians, S0 fingerprint, and runtime.
- [x] 4.3 Update stage-m1 plotting and summaries to report posterior median `lambda`, `alpha`, and `floor`, and generate the displayed scale map from those median values.

## 5. Stage M2 Pipeline Refactor

- [x] 5.1 Refactor `build_stage_m2_likelihood()` to accept fixed stage-m1 median `log_lambda_reg`, `adaptive_reg_alpha`, and `adaptive_reg_floor`, keep them static, and use the same S0 template and fixed source bbox.
- [x] 5.2 Update stage-m2 cache validation to require both the S0 fingerprint and fixed regularization hyperparameter values to match the current stage-m1 medians before reusing cached results.
- [x] 5.3 Update final pipeline summaries and stage-m2 artifact metadata to record the fixed M1 median hyperparameters used for mass inference.

## 6. Verification

- [x] 6.1 Run focused unit tests for regularization and operator behavior from the relevant test directory, including `tests/test_regularization.py` and `tests/test_pixelized_operator.py` selections.
- [x] 6.2 Run focused tests covering `PixelizedSourceModel` parameter behavior.
- [x] 6.3 Perform a syntax/import check of `examples/pix_src_demo_operator/pipe/no_lens_light/model_adpt_reg.py`; full demo execution may require GPU and real data runtime.
- [x] 6.4 If GPU resources are available, run a short smoke execution of the adaptive demo pipeline or a reduced sampler configuration to confirm stage-m1 and stage-m2 artifacts are produced.
