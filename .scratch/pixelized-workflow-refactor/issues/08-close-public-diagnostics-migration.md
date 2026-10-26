# 08: Complete and verify the public diagnostics migration

**What to build:** Users can rely on the documented public reconstruction and diagnostic interfaces across affected maintained callers, completing the second P1 deliverable without competing private plotting implementations.

**Blocked by:** 04: Migrate remaining dense plots and source exports; 05: Migrate operator basic and source-reconstruction cases; 06: Migrate operator survey and lens-subtracted diagnostics; 07: Correct joint reconstruction displays and migrate subhalo diagnostics

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Reconcile the affected maintained caller inventory, including actual backend variants, and complete any missed formal plotting or source-export migrations.
- [x] Remove application-level reconstruction and diagnostic copies that no longer have callers; retain legitimate numerical solver internals, existing public interfaces, and necessary compatibility wrappers.
- [x] Document the new single-parameter diagnostic interface, opt-in work, method identification, unavailable operator joint statistics, and approved case-level failure behavior.
- [x] Verify representative numerical equivalence, single-reconstruction behavior, component reconstruction, and unsupported-versus-invalid distinctions through the public interfaces.
- [x] Run the relevant existing gradient, batching, and failure regressions for unchanged numerical entry points, and verify that ordinary likelihood evaluation performs no optional diagnostic work.
- [x] Record focused core and example verification separately and mark the second P1 ready for independent review; reserve the full-suite final gate for the final integration ticket.

## Comments

- Claimed by Sidekick on branch `pwr/08-diagnostics-close` (worktree /tmp/tinylens-pwr-08).
- Resolved by Sidekick on `pwr/08-diagnostics-close`: dead `_source_param_value`
  removed from 10 pipe files (retained in `galan24_test/model_predefine_src_box.py`,
  used for stage-A sigma); AST selection lists updated in
  `test_pipeline_source_export.py` and `test_source_compatibility.py`; probe
  renamed `test_source_package_helper_uses_public_diagnostics` with `evaluate`
  now an always-fail stub and the sole counter under
  `evaluate_with_diagnostics`; guide.md entry-point paragraph replaced with the
  `evaluate_with_diagnostics` contract + example; maintained inventory recorded
  in the feature README (model_matrix = dense backend in the operator tree;
  bench/profile/dump/sim_data excluded).
- Verification (conda tinylens_gpu, XLA_PYTHON_CLIENT_PREALLOCATE=false):
  `pytest test_pixelized_inversion.py test_pixelized_operator.py -k 'diagnostic or evaluate or forward_model or gradient or gradients or vectorized or chunked' -q` -> 78 passed.
  `pytest test_operator_autodiff.py test_solver_autodiff.py -q` -> 31 passed.
  `MPLBACKEND=Agg pytest test_pipeline_source_export.py -k 'uses_public_diagnostics or run_stage_m1' -q` -> 27 passed.
  `MPLBACKEND=Agg pytest test_source_compatibility.py -q` -> 39 passed.
