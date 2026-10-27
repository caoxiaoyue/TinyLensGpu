# 11: Single-step scientific examples, diagnostics, and exports

**What to build:** A researcher can run an approximately 200-by-200-image single-step Pseudo-Jaffe experiment with either backend and obtain reproducible saved reconstructions and figures that distinguish recovery from linearization error.

**Blocked by:** 07 — Operator tuning and single-step backend calibration.

**Status:** ready-for-agent
**State:** open

**Parent:** [Approved potential-correction spec](../spec.md).

- [ ] A documented example composes the approved mock, fixed true macro, truth-derived gradient initialization, and dense/operator single-step fitting/tuning.
- [ ] Image size, source resolution, potential spacing, integration settings, strengths/search bounds, dtype, and output destination are configurable.
- [ ] The example includes matched subhalo and no-subhalo runs, with adopted weak-prior calibration and numerical settings recorded.
- [ ] Numerical exports preserve source and potential fields, core/support geometry, derivative validity, strengths, method annotations, initialization provenance, configuration, and solver status.
- [ ] Figures separately identify the linear fitted image and nonlinear re-render, with corresponding residual diagnostics; exported single-step evidence remains tied to the linear model.
- [ ] Diagnostic products include source, potential, deflection, and convergence, plus perturbation localization, convergence error, source recovery, and image residual summaries on documented domains.
- [ ] Raw fitted fields are retained; any affine-aligned truth comparison is explicitly identified and does not replace inference outputs.
- [ ] Failure records remain available, but a failed fit cannot be exported or labeled as a successful scientific reconstruction.
- [ ] Plotting and persistence consume the public result rather than independently assembling or solving the inversion.
- [ ] Small example tests verify assembly, saved metadata and numerical products, distinct rendering semantics, valid-domain handling, and failure behavior.
- [ ] The scientific run and its reproduction instructions record actual configurations and results without claiming formal detection significance or mass confidence intervals.

## Scope and handoff

This slice may proceed alongside dense iterative development. Establish
reusable result-based example persistence and plotting where it is genuinely
shared, while keeping the explicit single-step workflow visible.

