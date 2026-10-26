# 10: Transfer a source template and render it through shared support

**What to build:** The representative pipeline completes reconstruction export, source-template validation, consumption by the next stage, and diagnostic rendering through reusable support while retaining its existing data contracts.

**Blocked by:** 09: Run sampling, persistence, and recovery through shared support

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Complete a representative operator SLACS source-template transition using the explicit configuration and stage-support contract introduced by the preceding ticket.
- [x] Centralize reusable reconstruction export and prepared-template validation while preserving their different required fields and their existing numerical representations.
- [x] Validation returns the same dictionary object and preserves in-place scale-map generation or coercion, square-grid checks, obsolete-key rejection, and optional expected-bounding-box checks.
- [x] Pass grid size, expected bounding box when applicable, and existing case-specific normalization inputs explicitly; preserve the next stage's template and fixed-grid interpretation.
- [x] Extract reusable plotting support that consumes public results and diagnostics, preserving the representative layout and allowing meaningful case-specific render variants.
- [x] Verify the complete export-to-next-stage path, dictionary identity, applicable validation failures, preserved artifact contents, and rendered data with small inputs.
- [x] Reject invalid reconstruction before formal export or plotting without rejecting valid reconstructions merely because optional diagnostics are unavailable.
- [x] Shared helpers are directly importable without added process mutations; keep unmigrated case helpers working and record focused verification.

## Comments

- Claimed/resolved by Sidekick on `pwr/10-shared-source-plot-support`. New
  `example_support/source.py` (`source_axes_from_bbox`, `normalize_source_bbox`,
  `source_bboxes_match`, `require_valid_reconstruction` raising
  `InvalidReconstructionError`, `export_source_package`,
  `validate_source_package` with `normalize_bbox`/`check_axes`/
  `expected_source_bbox`/`require_regularization` options) and `plotting.py`
  (`draw_image_panel`, lazy pyplot, `colorbar_kwargs` for the scale label).
  SLACS pipeline: `_source_axes_from_bbox`/`_validate_source_package`/
  `_solve_pixel_source_for_package` are thin adapters; `_plot_pix_stage` uses
  shared validity + `draw_image_panel` (all layout/masks/overlays preserved).
- Verification (conda tinylens_gpu, XLA_PYTHON_CLIENT_PREALLOCATE=false,
  MPLBACKEND=Agg, examples/tests):
  `pytest test_pixelized_support.py -q` -> 13 passed (new: helper imports,
  raw-vs-prepared validation identity/float32 scale/error boundaries,
  Galan-option bbox/axis checks, require_valid joint-unsupported pass,
  SLACS export->validate->build_stage_m2->plot artifact + PCG-failure abort).
  `pytest test_source_compatibility.py test_pipeline_source_export.py -k slacs -q` -> 11 passed.
- Review corrections: `check_axes` now uses exact Galan tolerances
  (rtol=1e-6, atol=1e-7); expected-bbox error restored to the fixed stage-A
  wording; axis boundary cases (+5e-8 zero-coord accept, +6e-7 reject) and
  float64->float32 supplied-scale coercion covered; transition test plots the
  built m2 stage with its own medians, captures rendered artists vs m2
  result/diagnostics (float32 tolerances), asserts nonuniform
  `regularization_scale == source_template_scale_map(...)`; subprocess
  mutation test also imports source/plotting and asserts no new
  `matplotlib.pyplot` module. To satisfy that contract, sampling/stages/source
  core imports are lazy inside functions. Rerun:
  `pytest test_pixelized_support.py -q` -> 13 passed.
