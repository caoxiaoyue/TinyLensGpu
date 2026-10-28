# 07: Operator tuning and single-step backend calibration

**What to build:** A researcher can automatically tune a single-step reconstruction with either backend on small reproducible mocks and inspect a recorded comparison of selected strengths, recovered fields, and sensitivity to the fixed amplitude prior.

**Blocked by:** 02 — Generate Pseudo-Jaffe mocks and truth; 05 — Optimize dense single-step regularization strengths; 06 — Operator single-step reconstruction and block evidence.

**Status:** ready-for-agent
**State:** resolved

**Parent:** [Approved potential-correction spec](../spec.md).

- [x] The operator backend works through the same bounded log-strength optimization workflow as the dense backend.
- [x] Small Pseudo-Jaffe and no-subhalo inputs provide reproducible truth-derived initialization and identical configurations for backend comparisons.
- [x] A comparison scans regularization strengths and records complete versus approximate evidence, selected optima, model residuals, and recovered source/potential/convergence on their valid domains.
- [x] Sensitivity runs vary the independently fixed amplitude-prior value between runs, never optimize or rescale it within a run.
- [x] The adopted demonstration amplitude strength, search bounds, and numerical tolerances are documented with the evidence supporting them.
- [x] The comparison distinguishes numerical solve errors, determinant-approximation effects, linearization error, and the physical/discretization convention of truth products.
- [x] A materially misleading approximate optimum or recovery remains an unmet acceptance item; do not declare success merely because evidence is finite.
- [x] Any necessary refinement stays within the approved block-approximation design; SLQ, a dense cross-field Schur construction, or new scientific scope requires a separate decision.
- [x] Reordered candidate evaluations are reproducible, and optimization failures and search-boundary optima remain visible.
- [x] A runnable small-case calibration workflow and compact retained result record make the comparison independently repeatable.

## Scope and handoff

This ticket is the approved single-step validation milestone. Ticket 08 is
blocked by it to preserve the spec's delivery order, not because dense LM
mathematically requires operator evidence. Full-size plots and persistence
are delivered by ticket 11, which can proceed alongside ticket 08.

## Comments

- 2026-10-28: Claimed by `/root/ticket07` on `codex/pc-07` in
  `/tmp/tinylens-pc-07`, based on integration commit `fb5b057` (tickets 01–06).
  Public numerical/optimizer results and runnable calibration artifacts are
  the already approved TDD seams. Scientific acceptance remains conditional
  on the measured recovery and determinant-approximation comparison.


- Preliminary verification: the new example artifact seam first failed for a
  missing driver, then passed for four tiny physical sensitivity runs; the
  explicit incidence-layout test subsequently failed for absent layout support
  and passed after adding the sparse builder. The tiny grids validate assembly
  and records only. Public operator/DIRECT plus existing operator tests passed
  32 cases. Full physical calibration and prior sensitivity remain running;
  this ticket is not resolved by those structural checks.
- The initial full physical searches did not complete a single run before
  interruption; their configurations/logs and unfinished status are retained
  in `research/interrupted-calibration-searches.json`. They are not acceptance
  evidence. The coordinator measured repeated triangular-vector application
  as the bottleneck and explicitly authorized cached bounded inverse lower
  factors for the same principal preconditioner.
- That optional operator-contract test failed before implementation and then
  passed in both dtypes. Focused curvature/operator/inference: 39 passed;
  dense/rendering/shared solver/autodiff: 62 passed; selected existing
  PCG/Schur regressions: 23 passed. Separate benchmark checks passed the
  production application speed comparison and bounded allocation regression.
  Twelve real physical extreme-strength dense/operator comparisons all passed;
  see `research/inverse-factor-fixed-strengths.json`. Full calibration is
  now rerunning with this verified numerical implementation.

- 2026-10-29: Resolved on `codex/pc-07` after all six physical calibration
  runs and the same-geometry boundary probe completed. Numerical implementation
  is `694812d` (workflow start `c064f04`), with source/macro/gradient/partition
  and inputs checked identical across independent workers. The full report is
  [single-step calibration](../../../docs/validation/potential-correction-single-step-calibration.md).
  Retained `research/single-step-calibration/` contains the compact summary,
  compressed complete candidate records, inputs and raw selected fields.
- Adopted demonstration: lambda0=10000; source bounds `[1e-5,1e-3]`,
  D4 bounds `[1e5,1e11]`; float64; source-incidence tiles 24/cap 1024;
  PCG maxiter 20000, rtol 1e-12, atol 0; actual residual acceptance 1e-8.
  The report explains low-mode sensitivity, attenuated peaks, nonzero controls,
  the finite upper-bound plateau and incomplete DIRECT budget termination.
  All other tested fixed lambda0 values remain recorded. The smaller misleading
  block layouts remain rejected rather than accepted from finite evidence.
- Verification commands, with `tinylens_gpu` activated and
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`:
  - From `tests/`, `pytest test_curvature_operator.py test_potential_correction_operator.py test_potential_correction_inference.py -q`: **39 passed**.
  - From `tests/`, `pytest test_potential_correction_dense.py test_potential_correction_rendering.py test_solver_autodiff.py test_operator_autodiff.py -q`: **62 passed**.
  - From `tests/`, `pytest test_pixelized_operator.py -q -k 'pcg or schur or joint_curvature or ragged_source_blocks or preconditioner_is_spd'`: **23 passed**.
  - From `examples/tests/`, `pytest test_potential_correction_calibration.py test_potential_correction_tuning.py test_potential_correction_operator_example.py test_potential_correction_mock.py -q`: **16 passed**. After the initialization-metadata fix, the two calibration layouts passed again.
  - From `benchmarks/`, the application-speed and bounded-allocation test files each passed separately.
  - The full calibration command in the report completed **2452 successful
    inner DIRECT evaluations**, **90 successful matched scan pairs**, all six
    reordered checks and all six deliberate failure checks. All 12 DIRECT
    optimizers retained budget termination. The independent high-smoothing
    boundary probe also completed without a failed solve.
- The numerical outcome supports localized reconstruction and backend strength
  calibration for this mock, with all raw/whole-core limitations retained.
  It does not establish complete potential recovery, detection significance,
  or mass precision. Tickets 08 and 11 can now start under the approved scope.
