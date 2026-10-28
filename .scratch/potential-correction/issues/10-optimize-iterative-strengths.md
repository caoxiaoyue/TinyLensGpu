# 10: Optimize strengths around complete iterative reconstructions

**What to build:** A researcher can select source and potential smoothing strengths for either iterative backend through an outer DIRECT search whose candidates each perform a complete, fixed-strength inner reconstruction.

**Blocked by:** 09 — Operator iterative reconstruction at fixed strengths.

**Status:** ready-for-agent
**State:** claimed

**Parent:** [Approved potential-correction spec](../spec.md).

- [ ] Both dense and operator iterative fits are callable by the bounded two-dimensional log-strength optimizer.
- [ ] Every candidate resets to the same truth-derived source initialization and zero correction; no source, potential, or damping state leaks from the preceding candidate.
- [ ] Strengths remain fixed throughout each inner iteration and the amplitude-prior strength remains fixed throughout the outer search.
- [ ] Candidate evidence is evaluated only for successfully converged inner results using the proper backend-specific final local curvature.
- [ ] Failed, nonfinite, stalled, or iteration-limited inner runs use the optimizer's invalid-objective convention and retain their termination reasons.
- [ ] Result metadata identifies local-linearization and determinant approximations independently, as well as search bounds, boundary optima, precision, and selected strengths.
- [ ] Repeated and reordered candidate evaluations are reproducible at the declared numerical tolerance.
- [ ] A small controlled real-problem comparison checks the nested optimum against an independent scan and checks the effect of backend approximation on recovery.
- [ ] Tests cover initial-state reset, fixed inner strengths, invalid candidates, explicit precision, and successful-result handoff to diagnostics.
- [ ] A runnable small example demonstrates the complete outer-search/inner-reconstruction workflow without a posterior-sampling requirement.

## Scope and handoff

Reuse the established optimizer and result boundaries. Do not update strengths
inside LM, reinterpret failed terminal iterates as successful fits, or change
the search to a path-dependent warm-start procedure.

## Comments

- Claimed by `/root/iterative_exploration` on `codex/pc-10` in
  `/tmp/tinylens-pc-10`, based exactly on integration
  `2675c9b3107b47d3d2c7ac2af4ef2dad28dcbc72`. Ticket 09 is resolved and merged.
  Implementation follows the approved public fit/evidence/result and artifact
  seams, with independent small nonlinear validation; no scientific-200 work.

