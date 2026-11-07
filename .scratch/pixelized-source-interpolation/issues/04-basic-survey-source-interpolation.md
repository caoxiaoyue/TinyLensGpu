# 04: Carry source interpolation through basic and survey workflows

**What to build:** Users of the remaining maintained basic, survey, and lens-light pre-subtraction workflows can select, save, and resume the same source interpolation using the established staged workflow rules.

**Blocked by:** 03 — Save and resume a staged fit with its source interpolation.

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Apply the shared mode and provenance contract to the remaining basic, no-lens-light, adaptive-template, Galan, SLACS, Euclid, and pre-subtraction workflows, including their dense and operator variants where applicable.
- [x] Confirm configured mode reaches the source model before fitting and remains consistent across all pixelized stages and source-template transfers. Persist it in stage settings and exported source packages.
- [x] Verify known older bilinear checkpoints restore correctly, recorded Hermite checkpoints reconstruct the same model, and conflicting explicit choices stop before saved results are reused.
- [x] Exercise representative default and explicit-mode round trips through the existing basic and survey regression entry points. Preserve each workflow's fitted-grid and source-template validation rules.

## Comments

Claimed for implementation on `codex/psi-04-workflows`.

Implemented in `548bf10`, merged as `69c950f`. From `examples/tests/` in
`tinylens_gpu` with CPU, preallocation disabled and `MPLBACKEND=Agg`,
the following command passed 157 tests with one out-of-ticket joint case
deselected.

```bash
pytest -q test_basic_shared_support.py test_survey_shared_support.py \
  test_dense_shared_support.py test_adaptive_demo_s0_package.py \
  test_pixelized_support.py test_pipeline_source_export.py \
  test_source_compatibility.py::test_survey_source_export_preserves_public_products \
  test_source_compatibility.py::test_survey_stage_propagates_invalid_plot_and_retains_posterior \
  -k 'not simult_fit_lens_source'
```

Dense SLACS and the no-lens-light adaptive example exercised default,
legacy and Hermite round trips. The suite checks early rejection of late
incompatible caches, legacy S0 fingerprints, and signed-mode basic exports.
The scoped constructor audit and five representative CLI help checks passed.
