## 1. Core Stage Posterior API

- [x] 1.1 Implement `StagePosterior` in the Inference layer with constructors from likelihood schema and from serialized schema; verify with unit tests for shape validation, weight normalization, and duplicate-name rejection.
- [x] 1.2 Move or reuse empirical-width and conservative-sigma logic from `GaussianPriorPasser`; verify Gaussian sigma values match current behavior.
- [x] 1.3 Add summary methods `median`, `std`, `median_std`, and `medians`; verify weighted quantile/std outputs against small deterministic arrays.
- [x] 1.4 Add inherited `ParamU` factories `fixed(...)` and `gaussian(...)`; verify returned parameters have expected names, values, prior metadata, hard limits, and static/dynamic modes.
- [x] 1.5 Export the replacement API from `TinyLensGpu.Inference` and remove `GaussianPriorPasser` from public exports; verify imports reflect the new API.

## 2. Stage Result And Cache Integration

- [x] 2.1 Update sampler helper patterns so pipeline `_run_sampler(...)` can return or construct a stage result/posterior object carrying likelihood-derived schema, samples, weights, log evidence, and names/specs.
- [x] 2.2 Update stage pickle payloads to include lightweight serialized schema needed for cache rehydration; verify `--skip-done` can rebuild a stage posterior without pickling a full likelihood object.
- [x] 2.3 Preserve existing posterior summary output and plotting inputs while sourcing medians from the new object; verify no change to likelihood evaluation inputs.

## 3. Pipeline Migration

- [x] 3.1 Migrate `examples/pix_src_demo_operator/pipe/galan24_test/model.py` from `GaussianPriorPasser`, `names_a`, and manual median dictionaries where appropriate to `StagePosterior`.
- [x] 3.2 Refactor fixed mass/shear inheritance helpers to use `stage.fixed(...)`; verify `SIE`, `EPL`, and `Shear` parameters remain static where intended.
- [x] 3.3 Refactor Gaussian mass/shear inheritance helpers to use `stage.gaussian(...)`; verify EPL/shear prior metadata matches previous `GaussianPriorPasser` results.
- [x] 3.4 Mechanically migrate remaining example pipelines that import `GaussianPriorPasser`; verify no `GaussianPriorPasser` references remain outside archival text.

## 4. Verification

- [x] 4.1 Run focused unit tests for the new Inference-layer API from the relevant test directory using the `tinylens_gpu` environment.
- [x] 4.2 Run fast non-slow test subset covering prior extraction and inference helpers: `pytest -m "not slow"` if runtime is acceptable.
- [x] 4.3 Run lightweight import/build checks for migrated pipeline modules without launching long samplers; note that full sampler/GPU execution is not required for this change unless a migrated path cannot be validated otherwise.
- [x] 4.4 Document any remaining examples intentionally not migrated or any GPU-only verification that was not run. No migrated examples were intentionally left with `GaussianPriorPasser`; full long sampler/GPU pipeline execution was not run.
