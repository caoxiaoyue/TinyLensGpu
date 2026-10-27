# 12: Iterative scientific examples, diagnostics, and exports

**What to build:** A researcher can run the same scientific mock through either iterative backend, inspect nonlinear recovery and convergence history, and compare it with the single-step results using a consistent saved-result and diagnostic workflow.

**Blocked by:** 10 — Optimize strengths around complete iterative reconstructions; 11 — Single-step scientific examples, diagnostics, and exports.

**Status:** ready-for-agent
**State:** open

**Parent:** [Approved potential-correction spec](../spec.md).

- [ ] A documented iterative example supports both fixed-strength and complete outer-optimization workflows on the approved subhalo/no-subhalo mocks.
- [ ] Macro parameters, source geometry, imaging conventions, and truth-comparison definitions match the stated single-step comparison configuration.
- [ ] Initial source-node intensities come from truth and potential starts at zero; exported provenance distinguishes this from the single-step frozen-gradient initialization.
- [ ] Result persistence includes the final nonlinear image and residuals, recovered fields, valid domains, selected strengths, local/block evidence annotations, and termination status.
- [ ] Iteration history and diagnostic plots expose objective progression, accepted/rejected updates, relevant damping information, and the final stopping reason.
- [ ] Single-step/iterative comparison reports use matched configurations and distinguish linearization effects from determinant approximations.
- [ ] Shared exports/plots consume the common public result, with algorithm-specific fields and images represented explicitly rather than filled with misleading placeholders.
- [ ] Unconverged or nonfinite runs cannot appear as successful final scientific artifacts, while diagnostic failure records remain inspectable.
- [ ] Small example tests verify the two backends, nested-workflow assembly, history persistence, common metadata, and failure propagation.
- [ ] Scientific example results and reproduction instructions are retained with actual grid, prior, noise, precision, and integration settings.

## Scope and handoff

Do not add macro-mismatch stress tests, posterior sampling, uncertainty maps,
or formal detection claims. This completes the runnable scientific workflows
that the precision/performance ticket will measure.

