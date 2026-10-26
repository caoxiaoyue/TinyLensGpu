# 04: Migrate remaining dense plots and source exports

**What to build:** Maintained dense examples generate reconstructions, diagnostic plots, and source exports through the supported public interfaces with their existing numerical and artifact conventions.

**Blocked by:** 02: Deliver dense diagnostics through a complete example

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Migrate the remaining dense simple fitting examples, dense SLACS and pre-subtraction pipelines, dense joint MGE example, and the matrix variant located among operator examples, according to their actual backend.
- [x] Replace application-level private solving and duplicated diagnostic calculations with public evaluation or public evaluation-with-diagnostics as appropriate; each requested reconstruction is solved once.
- [x] Preserve dense source-only and joint diagnostic conventions, masks, noise normalization, source grids, position constraints, and existing plot layouts, colors, and overlays.
- [x] Reject nonfinite reconstruction before formal plotting or source-package export without inventing a dense convergence-status object.
- [x] Preserve raw source-export keys and numerical representations and distinguish raw exports from prepared template packages.
- [x] Extend appropriate example regressions with representative real numerical and artifact checks; leave sampler, storage, and stage-flow consolidation for the later shared-support tickets.
- [x] Record the migrated caller inventory and focused verification results, retaining compatibility for unrelated callers.

## Comments

- Claimed by Sidekick on branch `pwr/04-dense-callers` (worktree
  `/tmp/tinylens-pwr-04`). Confirmed clean HEAD and integration base `f09af9d`
  before edits; no reset needed. Loaded the `tdd` skill, using the spec-approved
  public reconstruction, plotting-artifact, and raw source-export seams.

- Diagnostic migration implemented, uncommitted for lead review. All owned
  reconstruction/plot consumers now call `evaluate_with_diagnostics` once and
  reject invalid reconstruction before figure or output creation. Source pixels
  come directly from `result.source_pixels`; joint amplitudes and component
  images come from their named public fields. Existing curvature statistics
  use the agreed dense convention; the two simple joint scripts retain their
  original raw parameter-count denominator and do not request effective DOF.
  Components are requested only where drawn.
  The existing joint stage plot had invalid 1-D indexing into its 2-D axes;
  corrected it to the existing data/model/residual/source panels. Fitted
  parameter contexts are retained for critical-curve/caustic overlays.

### Migrated caller inventory

Paths below are relative to `examples/`:

| Caller | Migration |
| --- | --- |
| `pix_src_demo/simple/pix_src/fit_lens_src_pos_like.py` | Reconstruction/statistics and fitted overlay context |
| `pix_src_demo/simple/pix_src_epl_sh/fit_lens_src.py` | Reconstruction/statistics and fitted overlay context |
| `pix_src_demo/simple/pix_src_epl_sh/fit_lens_src_pos_like.py` | Reconstruction/statistics and fitted overlay context |
| `pix_src_demo/simple/pix_src_lens_light/fit_joint.py` | Joint result, fitted components, preserved raw parameter-count denominator |
| `pix_src_demo/simple/pix_src_lens_light_mge/fit_joint.py` | Joint result, fitted components, preserved raw parameter-count denominator |
| `pix_src_demo/simple/pix_src/demo_inversion.py` | Validity and residuals; preserves chi2 per active pixel, explicitly labeled `chi2/N_pix` |
| `pix_src_demo/simple/pix_src_epl_sh/demo_inversion.py` | Same demo migration; existing summary dictionary keys/types retained |
| `pix_src_demo/pipe/pre_subtract_lens/pix_src_pipe/model.py::_plot_pix_stage` | Dense plot and fitted overlay context |
| `pix_src_demo/pipe/pre_subtract_lens/pix_src_pipe_mge/model.py::_plot_pix_stage` | Dense plot and fitted overlay context |
| `pix_src_demo/pipe/pre_subtract_lens/pix_src_pipe_sersic/model.py::_plot_pix_stage` | Dense plot, existing position markers and fitted overlay context |
| `pix_src_demo/pipe/slacs_examples/pipeline.py::_plot_pix_stage` | Dense plot and fitted overlay context |
| `pix_src_demo/pipe/simult_fit_lens_source/mge_lens_light/model.py::_plot_stage_b` | Joint result/components, valid 2-D overlay axes, fitted overlay context |
| `pix_src_demo_operator/pipe/no_lens_light/model_matrix.py::_plot_pix_stage` | Dense backend, public position diagnostics, preserved mask zoom and overlays |

- `pix_src_demo/simple/pix_src/fit_lens_src.py` was already migrated by ticket 02;
  its only ticket-04 change is importing/raising `InvalidReconstructionError`.
  Profiling, benchmarking, jaxpr inspection, simulation, parametric-only calls,
  sampler/configuration/stage assembly are unchanged. Stage exception handlers
  have the separately authorized propagation changes recorded below.
- Inventory found no raw source-package export helper in these owned paths:
  `_solve_pixel_source_for_package` helpers live in other operator callers,
  outside ticket 04 ownership. No raw export dictionary or prepared template
  format was changed or invented. Demo summary keys remain
  `lambda`, `log_ev`, `chi2_nu`, with the same per-pixel numerical convention.

### TDD and focused verification

All pytest runs used `source ~/anaconda3/bin/activate && conda activate tinylens_gpu`,
from `examples/tests/`, with `XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg`.
Only `examples/tests/test_joint_fit_plotting.py` was edited.

- Initial `pytest test_joint_fit_plotting.py -k dense_joint_stage_rejects -q`:
  **red**, real NaN observation reached figure creation; after migration
  **1 passed**.
- `pytest test_joint_fit_plotting.py -k dense_joint_stage_plots -q`:
  **red**, existing joint overlay axes raised `IndexError`; after correction
  `-k dense_joint_stage` yielded **2 passed**. Real displayed fitted components
  sum to the returned total; a nonzero fitted lens-light image is required.
- `pytest test_joint_fit_plotting.py -k 'dense_stage_rejects and pix_src_pipe/model' -q`:
  **red**, NaN reconstruction reached figure creation. Migrated all remaining
  dense stage plots; `-k dense_stage_rejects` yielded **6 passed**.
- `pytest test_joint_fit_plotting.py -k 'dense_simple_fit_rejects and pix_src_lens_light/fit' -q`:
  **red**, NaN reconstruction reached figure creation. An initial test-only
  Caskade import-order error was corrected before confirming this behavioral red.
  After migration `-k dense_simple_fit_rejects` yielded **5 passed**.
- `pytest test_joint_fit_plotting.py -k 'inversion_demo and True and pix_src/demo' -q`:
  **red**, NaN summary/plot output was attempted; after migration
  `-k inversion_demo` yielded **4 passed**, including unchanged summary keys and
  chi2/pixel values against real reconstructions.
- `pytest test_joint_fit_plotting.py -k 'source_stage or tracer_plot_block_uses' -q`:
  **9 passed**. Real artifacts, residual masks, source grids, fitted overlay
  context, and matrix image/source position markers checked.
- Merged integration tip with `git merge --ff-only integration/pixelized-workflow-refactor`:
  **fast-forward `f09af9d` -> `9c9e09b`**, no new commit and no conflicts.
- After integration:
  `pytest test_joint_fit_plotting.py test_source_compatibility.py -q`:
  **31 passed in 15.57s**.
- `git diff --check`: clean. Read-only AST comparison with HEAD confirmed all
  non-diagnostic functions and top-level configuration assignments are unchanged
  in all six edited pipeline modules. `rg` inventory found no private inversion,
  regularization, or position-diagnostic reads in maintained owned plot callers.
  No full suite or scientific sampling was run.

### Review corrections and resolution

- The lead authorized stage propagation and supplied
  `InvalidReconstructionError(RuntimeError)` on integration commit `3887be3`.
  `git merge --ff-only integration/pixelized-workflow-refactor` fast-forwarded
  `9c9e09b` to `3887be3`; repeated final merge checks are up to date.
- All 14 owned public diagnostic consumers now import and raise that exception,
  preserving the original messages. All 13 handlers around pixelized plot calls
  re-raise it before their existing broad handler; unrelated plotting errors
  keep their original non-fatal behavior.
- Current handler locations (relative to the inventory above): joint
  `model.py:692`, pre-subtraction `pix_src_pipe/model.py:738,812`,
  `pix_src_pipe_mge/model.py:752,826`,
  `pix_src_pipe_sersic/model.py:731,806`, SLACS `pipeline.py:785,874`, matrix
  `model_matrix.py:653,838,914,951`.
- The initial migration incorrectly changed the two simple joint scripts from
  raw parameter counts to curvature DOF. Corrected both to use public chi2 with
  `N_active - result.source_pixels.size - len(result.lens_amplitudes)`, including
  the absent-amplitude and nonpositive-DOF cases, with no effective-DOF request.
  The established pipeline joint curvature diagnostic is unchanged.
- Added a stage-level regression at
  `examples/tests/test_joint_fit_plotting.py:541`. It executes real SLACS
  `run_stage_m`, assembly, prior transfer, dense inversion, and pickle IO; only
  the external sampler is controlled. The small test configuration uses a 5x5
  source, 10x10 NaN observation, and the stage's existing Cholesky solver.
  `InvalidReconstructionError` must escape while `stage_m.pkl` remains readable
  with the sampled posterior and no success plot exists.
- During test development, the MGE stage's existing NNLS configuration (40x40
  source, one lens basis, 10x10 NaN observation) returned finite reconstruction
  arrays rather than the required nonfinite reconstruction. The regression
  therefore uses the real source-only Cholesky stage above; no solver or validity
  semantics were changed to manufacture failure.
- Added real Sersic/MGE simple plot-block checks at
  `examples/tests/test_joint_fit_plotting.py:637`: 5x5 and 8x8 source grids give
  raw denominators 38 and -1 with 64 active image pixels and one lens amplitude.
  Tests verify chi2_nu and its display, unrequested effective fields, nonzero
  actual lens images, component sums, and source pixels.

Focused review verification used the same conda environment, working directory,
and XLA/MPL settings documented above:

- `pytest test_joint_fit_plotting.py -k propagates_invalid -q`:
  **red**, the real NaN reconstruction error was swallowed after stage persistence;
  after propagation changes **1 passed in 6.51s**.
- `pytest test_joint_fit_plotting.py -k simple_joint_preserves -q`:
  **4 red** for changed denominators; after correction **4 passed in 12.63s**.
- `pytest test_joint_fit_plotting.py -q` after integration:
  **33 passed in 20.77s**.
- `git diff --check`: clean. Read-only AST audit verified all 13 pixelized
  plotting handlers, unchanged top-level scientific settings, and no
  non-diagnostic function changes except the authorized re-raise handlers.
  All modified Python sources compiled. No broad suite was rerun.
- **Resolved after the review corrections and focused verification.** No
  remaining acceptance blocker; all changes are uncommitted for lead review.
  The lead commits and uses the merger.
