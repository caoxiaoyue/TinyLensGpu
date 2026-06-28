## 1. Operator Fixed-Input Support

- [x] 1.1 Add optional `fixed_source_bbox` support to `PixelizedImageProbModelOperator` and make `_get_bbox()` return the fixed bbox while still computing current sub-grid and seed betas.
- [x] 1.2 Add optional `fixed_reg_scale` support to `PixelizedImageProbModelOperator`, including shape `(nx * ny,)`, finite, and positive-value validation.
- [x] 1.3 Update `_compute_reg_scale_from_betas(...)` so fixed scale maps bypass image-plane brightness accumulation and missing fixed scale inputs fail clearly instead of falling back to the old dynamic/freeze path.
- [x] 1.4 Verify fixed bbox and fixed scale are concrete JAX-compatible values before likelihood tracing and do not require mutation during JIT evaluation.

## 2. Source-Template Scale Utilities

- [x] 2.1 Implement a reusable `S0 -> scale` helper that accepts flat or 2D source pixels, clips negative values to zero, normalizes by the global mean, and applies the existing adaptive scale formula.
- [x] 2.2 Ensure the helper returns `None` for `adaptive_reg_alpha == 0` and returns flat `(nx * ny,)` `float32` scale arrays otherwise.
- [x] 2.3 Keep the helper free of Gaussian smoothing and document that stage-m0 regularization, not an extra smoothing pass, controls the smoothness of `S0`.

## 3. Tests

- [x] 3.1 Add unit tests for source-template scale construction, covering bright/dark contrast, negative clipping, all-dark finite behavior, and alpha-zero uniform behavior.
- [x] 3.2 Add operator tests for fixed bbox behavior: configured bbox is used, and missing fixed bbox under adaptive S0 mode fails clearly.
- [x] 3.3 Add operator tests for fixed scale behavior: scale is reused across different mass parameter values and invalid scale shapes fail clearly.
- [x] 3.4 Add a JIT/vectorized likelihood test confirming fixed bbox and fixed scale work under `make_likelihood(..., vectorized=True)`.
- [x] 3.5 Run targeted CPU-compatible tests from their test-file directories; note that full demo/runtime validation may require GPU.

## 4. Demo Pipeline Stage M0

- [x] 4.1 Add stage-m0 to `examples/pix_src_demo_operator/pipe/no_lens_light/model_adpt_reg.py` after stage-B/position setup and before stage-m1.
- [x] 4.2 Build stage-m0 with SIE+shear fixed at stage-A medians, pixelized source, `adaptive_reg_alpha = 0`, and the same regularization grid-search style as stage-m1.
- [x] 4.3 Solve and save the stage-m0 MAP source package with `source_pixels`, 2D source image for diagnostics, source bbox, axes, grid shape, lambda metadata, and stage-A medians.
- [x] 4.4 Add `--skip-done` cache loading/validation for `stage_m0.pkl`, including clear errors or recomputation when required S0 metadata is missing.
- [x] 4.5 Add a stage-m0 diagnostic plot showing data/model/residual/source and enough bbox/grid metadata to inspect the generated `S0`.

## 5. Demo Pipeline M1/M2 Refactor

- [x] 5.1 Update stage-m1 likelihood construction to accept the `S0` package, derive or load its fixed scale map, use the fixed source bbox, and re-optimize `log_lambda_reg`.
- [x] 5.2 Update stage-m2 likelihood construction to accept the same `S0` package, use the fixed source bbox and scale map, and preserve the existing stage-m1 lambda handoff.
- [x] 5.3 Update replot paths for stage-m1 and stage-m2 so missing plots can be regenerated from cached `S0`, fixed bbox, and fixed scale metadata.
- [x] 5.4 Update stage summaries and printed timing to include stage-m0 and distinguish the stage-m0 uniform lambda from the stage-m1 adaptive lambda.
- [x] 5.5 Remove or bypass old seed-ray adaptive-scale and `freeze_scale()` usage in this workflow, and update callers/tests that assumed dynamic adaptive scale construction.

## 6. Verification

- [x] 6.1 Run `pytest tests/test_regularization.py -k "adaptive or scale"` from the `tests/` directory.
- [x] 6.2 Run targeted operator tests from the `tests/` directory, including the new fixed bbox/fixed scale/JIT cases.
- [x] 6.3 Run `pytest -m "not slow"` if targeted tests pass and runtime is acceptable.
- [x] 6.4 Defer the optional full `model_adpt_reg.py` GPU demo run after targeted and fast-suite validation passed.

## 7. Global Retirement of Old Adaptive Path

- [x] 7.1 Remove retired adaptive seed-ray configuration fields from `PixelizedSourceModel` and update callers/tests that still pass `adaptive_reg_mode`, `adaptive_reg_smooth_sigma`, or `adaptive_reg_freeze`.
- [x] 7.2 Remove or disable dense backend seed-ray adaptive scale construction and `freeze_scale()` / `unfreeze_scale()` so `adaptive_reg_alpha > 0` fails clearly unless fixed-template adaptive inputs are supported.
- [x] 7.3 Add regression tests that the dense backend rejects adaptive regularization and that retired `PixelizedSourceModel` kwargs are no longer accepted.
- [x] 7.4 Add cache validation for stage-m1/stage-m2 so stale outputs from the retired workflow cannot be reused with a new `S0` package under `--skip-done`.
- [x] 7.5 Re-run targeted regularization/operator tests and the fast suite.
