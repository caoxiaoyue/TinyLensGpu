# 04: Dense single-step reconstruction at fixed strengths

**What to build:** A caller can perform a complete dense single-step source–potential reconstruction at fixed regularization strengths and receive the fitted fields, complete conditional Gaussian evidence, and distinct linear and nonlinear model images.

**Blocked by:** 03 — Render supplied potential corrections.

**Status:** ready-for-agent
**State:** resolved

**Parent:** [Approved potential-correction spec](../spec.md).

- [x] A public single-step fit entry point accepts the observation, fixed macro, fixed source geometry, truth-derived initial gradient provider, and explicit strengths.
- [x] The full source brightness and potential coefficients are solved jointly as signed variables; neither field is clipped or passed through an all-variable nonnegative solver.
- [x] The response freezes the initial source gradient and has the correct lens-equation sign, using the same physical potential map and imaging operations as nonlinear rendering.
- [x] Potential regularization enumerates every complete horizontal/vertical five-node window in the support exactly once, without bridging absent nodes, restricting to core-centered windows, or falling back to lower orders.
- [x] The prior includes existing source regularization and the complete fourth-difference-plus-independent-positive-zero-order potential precision on all support coefficients.
- [x] No hard affine gauge is added and the amplitude strength is not silently changed to rescue numerical failure.
- [x] The complete coupled system is solved with Cholesky-based linear algebra in JAX, with an explicit float64 reference path.
- [x] Evidence uses complete prior/posterior determinants and noise normalization; the potential prior determinant includes the independent zero-order term.
- [x] A common structured result exposes fields, geometry and validity, strengths, dtype, solver/failure information, initialization provenance, and evidence-method metadata.
- [x] The result distinguishes the fitted linear image from a nonlinear re-render; only the linear model determines single-step evidence.
- [x] Small real systems verify coupling, linear-system residuals, known regularization behavior, independent Gaussian evidence, and agreement with nonlinear image changes in the small-perturbation limit.
- [x] Failed or nonfinite candidates are distinguishable from successful fits and do not produce fabricated valid evidence.

## Scope and handoff

Deliver the dense fixed-strength numerical reference and callable evaluation
boundary. Hyperparameter optimization is ticket 05, operator execution is
ticket 06, and full scientific plotting/export is ticket 11. A small real
example or regression must demonstrate this ticket without those later tickets.

## Comments

- 2026-10-28: Claimed by ticket04 agent on `codex/pc-04`, worktree
  `/tmp/tinylens-pc-04`, based on integration `02c9239` (tickets 01–03).
  Confirmed test seams are the public fixed-strength fit/result and focused
  numerical operator contracts designated in the approved parent spec.
- 2026-10-28: Ownership continued after the server restart by
  `ticket04_resume` in the same branch/worktree. The interrupted uncommitted
  dense implementation and tests were retained; remaining acceptance checks
  and red→green work resume from that state.
- 2026-10-28: Resolved in implementation commit `60d5bec`. Public API:
  `SingleStepPotentialCorrection(renderer, image_data, noise_map,
  source_gradient, *, lambda_amplitude, ...)`, with fixed-strength
  `fit(lambda_source, lambda_potential, backend="dense")` and callable
  evaluation returning the shared `PotentialCorrectionResult`. Complete
  numerical actions remain available for ticket 06 without eager matrices.
  See [the implementation and verification record](../research/dense-single-step.md)
  for the exact API, configurations, tolerances, failure contract and limits.
- Verification in `tinylens_gpu` on RTX 5080, from `tests/`:
  `XLA_PYTHON_CLIENT_PREALLOCATE=false pytest test_potential_correction_dense.py test_potential_correction_rendering.py -q`
  → **31 passed in 28.19 s** (14 dense + 17 rendering). `git diff --check`
  passed. The scoped float32 precision test first failed with reduced-mantissa
  GPU products; the dense JIT now requests full-precision products locally,
  retaining process defaults. No full existing-suite repetition was needed
  for these new modules and the focused renderer extension.
- Latest integration `e5c34c4` was merged into this branch as `4f2ff24` after
  the implementation commit. The merge imported only the retained supplied-truth
  experiment and its evidence; numerical library/test code was unchanged.
  Hyperparameter optimization, operator execution, scientific recovery and
  performance remain with their downstream tickets.

