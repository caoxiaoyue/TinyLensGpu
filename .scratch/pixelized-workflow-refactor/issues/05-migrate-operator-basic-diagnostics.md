# 05: Migrate operator basic and source-reconstruction cases

**What to build:** The remaining basic operator and source-reconstruction cases use public reconstruction diagnostics while retaining their source-grid, template, and scientific display behavior.

**Blocked by:** 03: Deliver operator diagnostics through a complete example

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Migrate affected remaining simple source, compact-source, and EPL-plus-shear fitting demos, the standard and adaptive no-lens-light cases, and the inferred- and predefined-bounding-box Galan cases.
- [x] Replace private reconstruction and diagnostic assembly with public results, requesting existing source-only effective-parameter diagnostics only where previously used.
- [x] Preserve fixed or inferred source grids, adaptive scale-map behavior, position constraints, source-export contents, and each case's plotting conventions.
- [x] Stop formal plotting and source export for explicit operator non-convergence or nonfinite reconstruction, with informative failure context and retained saved artifacts.
- [x] Keep the dense matrix variant assigned to the dense migration ticket rather than treating its directory location as its backend.
- [x] Verify representative source-only, adaptive, and fixed-bounding-box workflows through public results and output artifacts; avoid a second reconstruction solely for display or export.
- [x] Record coverage and focused verification without changing sampling settings, cache formats, or the later shared-support scope.

## Comments

- Claimed and resolved by Sidekick on branch `pwr/05-operator-basic-diagnostics` (worktree /tmp/tinylens-pwr-05).
- Inventory migrated (one private reconstruction block each):
  `simple/pix_compact_src/fit_src_pos_like.py` (kept `dof = N_d - N_eff if N_eff < N_d else 1` display convention),
  `simple/pix_src_epl_sh/fit_lens_src.py`, `simple/pix_src_epl_sh/fit_lens_src_pos_like.py`
  (`dof = N_d - N_eff`); `simple/pix_src/fit_src_pos_like.py` and `simple/pix_src/fit_lens_src_pos_like.py` (added in rework), `simple/pix_src/demo_inversion.py` and `simple/pix_src_epl_sh/demo_inversion.py` (formal plot/summary exports, not profiling); `pipe/no_lens_light/model.py`,
  `pipe/no_lens_light/model_adpt_reg.py`, `pipe/galan24_test/model.py`,
  `pipe/galan24_test/model_predefine_src_box.py`: `_plot_pix_stage` now calls
  `evaluate_with_diagnostics(include_effective_dof=True)` once, raises RuntimeError
  before any figure on `not diagnostics.valid`, and reads lam/rho/scale/position
  fields from the diagnostics object; `_solve_pixel_source_for_package` uses
  `evaluate_with_diagnostics` and keeps the 'failed' RuntimeError (solver-metric
  detail when `solver_info` reports non-convergence, `invalid_reason` otherwise).
  `pipe/no_lens_light/model_matrix.py` left for the dense ticket. Profiling
  internals (bench_inversion, demo_inversion, dump_jaxpr, sim_data) untouched.
- `examples/tests/test_pipeline_source_export.py`: ProbeLikelihood gained a
  transitional `evaluate_with_diagnostics` that delegates to its public
  `evaluate` and returns `SimpleNamespace(valid, invalid_reason)`, so both
  migrated and legacy helpers pass until ticket 08; added real failure tests
  (constructor NaN `image_data`, `pcg_max_iter=0`) for the four migrated helpers.
- Verification (conda tinylens_gpu, `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
  `MPLBACKEND=Agg`, from `examples/tests/`):
  `pytest test_pipeline_source_export.py test_adaptive_demo_s0_package.py -q`
  -> 48 passed.
  Rework: all owned invalid guards raise `InvalidReconstructionError`
  (imported from `inversion_result`); broad `except Exception` handlers wrapping
  `_plot_pix_stage` calls now re-raise it first; simple fit scripts wrap
  `overlay_critical_and_caustics` in `ck.ActiveContext` + `fill_params` so
  overlays see fitted parameters. `test_pipeline_source_export.py` gained
  runtime plot-block coverage for all six migrated simple fits and both
  `demo_inversion` loop bodies (real small operator model, artifact assertions,
  NaN-observation rejection before artifacts) plus a `run_stage_m1`
  propagation test with a controlled sampler and temp storage asserting
  `InvalidReconstructionError` and retained `stage_m1.pkl`.
  Rerun: `pytest test_pipeline_source_export.py -q` -> 61 passed.
  Final rework: `import caskade as ck` moved after TinyLensGpu imports in all six
  simple fit scripts; every `_plot_pix_stage` overlay wrapped in
  `ck.ActiveContext(likelihood)` + `fill_params(q50)`; new tests cover
  import-order backend selection (isolated process, CASKADE_BACKEND absent),
  pipe overlay fitted-theta (6 simple blocks + 4 pipe `_plot_pix_stage` with
  distinct median theta_E). `pytest test_pipeline_source_export.py -k
  'imports_keep_jax or overlay_uses_fitted'` -> 16 passed; `-k 'simple_plot or
  demo_inversion or run_stage_m1'` -> 23 passed.
