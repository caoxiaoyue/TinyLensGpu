# 01: Prepare source regularization and precision

**What to build:** Existing pixelized-source reconstruction can prepare and run operator inversions without eagerly allocating dense source-difference matrices, while explicit precision is preserved through the shared source preparation needed by potential correction. Existing callers retain their numerical and public-interface behavior.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent
**State:** open

**Parent:** [Approved potential-correction spec](../spec.md).

- [ ] Operator preparation for the supported finite-difference source priors does not eagerly construct full dense difference matrices; observable allocation or memory evidence accompanies numerical checks.
- [ ] Dense callers can still request the corresponding matrices, with the same coefficients, boundary rules, regularization penalties, and determinant behavior.
- [ ] Explicit float64 values remain float64 through the relevant shared source-grid and regularization preparation; existing default dtype behavior is preserved.
- [ ] Small real dense and operator source reconstructions retain their model images, recovered source, objective/evidence conventions, and solver behavior within justified tolerances.
- [ ] Existing nonadaptive regularization choices remain usable; this is not a change to their mathematical prior or an adaptive-regularization project.
- [ ] Public imports, existing constructor/call patterns, and supported differentiation behavior remain compatible.
- [ ] Existing source–lens-light Schur preconditioning, lens-light basis validation, PSF behavior, and fixed-source boundary semantics are unchanged.
- [ ] Relevant core regressions run in the documented project environment, with commands, precision, numerical tolerances, and results recorded.

## Scope and handoff

This is the approved prefactoring slice, not a potential-correction implementation.
Keep changes limited to lazy source regularization and the necessary opt-in
precision propagation. Optimizer-specific precision integration belongs to
ticket 05. Do not change process-wide precision defaults or introduce a general
backend redesign.

