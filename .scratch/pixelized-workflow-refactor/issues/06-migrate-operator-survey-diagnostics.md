# 06: Migrate operator survey and lens-subtracted diagnostics

**What to build:** SLACS, Euclid, and lens-light pre-subtraction workflows produce their existing plots and source exports from the common public numerical interfaces.

**Blocked by:** 03: Deliver operator diagnostics through a complete example

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Migrate operator SLACS and Euclid pixelized stages and the standard, Sersic, MGE, and split variants of the pre-subtraction workflows.
- [x] Obtain reconstructions, regularization displays, enabled position diagnostics, and the existing source-only effective-parameter approximation through public results rather than private solver assembly.
- [x] Preserve each case's masks, crop or zoom behavior, color choices, overlays, position constraints, and source-package conventions.
- [x] Reject explicitly unconverged or nonfinite reconstruction before formal plotting or export; unavailable optional quantities alone do not indicate failure.
- [x] Preserve stage scientific choices and numerical results, including parameter ordering, source-grid configuration, and regularization settings.
- [x] Extend appropriate source-export and example regression coverage for the materially different display and source-package variants, recording focused verification results.

## Comments

- Claimed by the ticket06 agent on `pwr/06-operator-survey-diagnostics`
  in `/tmp/tinylens-pwr-06`; `git merge-base --is-ancestor 9c9e09b HEAD`
  succeeded before work. Ticket03 is resolved at this integration baseline.
- Invoked the `tdd` skill by reading and following its `SKILL.md`, `tests.md`,
  and `mocking.md`: red→green slices use the spec-approved public caller seam,
  real small source models, and temporary plot/export artifacts. Scope is only
  the assigned survey/pre-subtraction callers, `test_source_compatibility.py`,
  and this ticket. No commit is requested.

### Delivered inventory

Paths below are under `examples/pix_src_demo_operator/pipe/`.

| File | Migrated entry points |
| --- | --- |
| `slacs_examples/pipeline.py` | `_solve_pixel_source_for_package` (line 327), `_plot_pix_stage` (line 1103) |
| `euclid_examples/pipeline.py` | `_solve_pixel_source_for_package` (line 353), `_plot_pix_stage` (line 1132) |
| `pre_subtract_lens/pix_src_pipe/model.py` | `_solve_pixel_source_for_package` (line 220), `_plot_pix_stage` (line 441) |
| `pre_subtract_lens/pix_src_pipe_sersic/model.py` | `_solve_pixel_source_for_package` (line 253), `_plot_pix_stage` (line 506) |
| `pre_subtract_lens/pix_src_pipe_mge/model.py` | `_solve_pixel_source_for_package` (line 296), `_plot_pix_stage` (line 856) |
| `pre_subtract_lens/pix_src_pipe_mge/model_split.py` | `_plot_pix_stage` (line 656); this variant has no raw source-package helper |

- All six plots request effective DOF from one public
  `evaluate_with_diagnostics(theta, include_effective_dof=True)` call.
  All five exports call the same API once with default flags. Invalid results
  are rejected before figure creation or source-package construction.
- Model/source/bbox, residuals/statistics, regularization displays and enabled
  position coordinates/penalty/separation now come from public results.
  Physical overlays run inside the median-theta context. Existing five-panel
  and split four-panel layouts, SLACS/presubtraction padding, Euclid masked
  center crop, split masked edge crop, colors, and markers are retained.
- All 42 plot/replot exception handlers propagate `InvalidReconstructionError`
  while preserving unrelated non-fatal plotting behavior. This `RuntimeError`
  subclass came from integration commit `3887be3`; no core files were edited
  in this ticket. Saved posteriors remain present when diagnostic plotting fails.
- An AST inventory against `9c9e09b` confirmed: every migrated entry point
  contains exactly one public diagnostic call and no private attribute reads;
  all other functions are unchanged after excluding the explicit failure
  propagation guards; module configuration assignments/branches are unchanged.
  No sampling controls, priors, scientific stage setup, tuples, or cache formats
  were changed.

### Verification evidence

All pytest commands ran from `/tmp/tinylens-pwr-06/examples/tests` after
`source ~/anaconda3/bin/activate && conda activate tinylens_gpu`, with
`XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg`.

- TDD tracer: `pytest test_source_compatibility.py -k survey_plot -q`
  first failed because the old plot made zero public diagnostic calls, then
  passed after migration (12 render cases across six scripts and two rho values).
- Export/failure slice:
  `pytest test_source_compatibility.py -k 'source_export or invalid_reconstruction' -q`
  failed on the old export interface, then passed **22 tests** after migration.
- Stage-stop slice:
  `pytest test_source_compatibility.py -k stage_propagates -q` first failed
  **2 tests** because SLACS/Euclid swallowed real invalid reconstructions,
  then passed **2 tests** after propagation was added. Only the external
  sampler is controlled; model assembly, diagnostics, posterior transfer and
  saved stage payload are real.
- Final required check: `pytest test_source_compatibility.py -q` —
  **39 passed in 28.25s** after the final integration merge. Evidence lives in
  `examples/tests/test_source_compatibility.py`: render checks at line 189,
  raw exports at line 299, actual zero-iteration/nonfinite-input failures at
  line 352, and stage failure/persisted-posterior checks at line 410.
  The 3×3 source / 10×10 image fixture includes a nontrivial PSF, mask,
  adaptive template and positions; expected products come from public
  evaluation, not private solver mocks. The tiny fixture alone tightens PCG
  rtol to `1e-7` with highest matmul precision to compare independent float32
  solves at `2e-5` tolerances (production solver configuration is unchanged).
- Final required legacy-reference check:
  `pytest test_pipeline_source_export.py -k matches_legacy_real_operator -q`
  — **18 passed, 18 deselected in 10.43s** after the final integration merge.
  The two final commands ran in
  separate processes; no server or full scientific sampler was launched.
- `git diff --check` passed. `git merge --ff-only
  integration/pixelized-workflow-refactor` fast-forwarded this branch from
  `9c9e09b` to `3887be3`, then to `d99b8b2` (ticket04 integration); the
  integration ancestor check passed afterward. Both required test selections
  were rerun successfully on `d99b8b2`.
- Transitional limitation: the other 18 probes in
  `test_pipeline_source_export.py` still implement only `evaluate()` at this
  tip. Ticket05 owns their dual-interface adaptation; they were deliberately
  not edited or used to force callers back to the old API. Full-suite execution
  remains the coordinator's later gate.

Implementation is complete and the owned diff is **uncommitted**, ready for
lead review. No implementation commit was created.
