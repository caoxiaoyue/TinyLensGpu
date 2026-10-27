# 07: Operator tuning and single-step backend calibration

**What to build:** A researcher can automatically tune a single-step reconstruction with either backend on small reproducible mocks and inspect a recorded comparison of selected strengths, recovered fields, and sensitivity to the fixed amplitude prior.

**Blocked by:** 02 — Generate Pseudo-Jaffe mocks and truth; 05 — Optimize dense single-step regularization strengths; 06 — Operator single-step reconstruction and block evidence.

**Status:** ready-for-agent
**State:** open

**Parent:** [Approved potential-correction spec](../spec.md).

- [ ] The operator backend works through the same bounded log-strength optimization workflow as the dense backend.
- [ ] Small Pseudo-Jaffe and no-subhalo inputs provide reproducible truth-derived initialization and identical configurations for backend comparisons.
- [ ] A comparison scans regularization strengths and records complete versus approximate evidence, selected optima, model residuals, and recovered source/potential/convergence on their valid domains.
- [ ] Sensitivity runs vary the independently fixed amplitude-prior value between runs, never optimize or rescale it within a run.
- [ ] The adopted demonstration amplitude strength, search bounds, and numerical tolerances are documented with the evidence supporting them.
- [ ] The comparison distinguishes numerical solve errors, determinant-approximation effects, linearization error, and the physical/discretization convention of truth products.
- [ ] A materially misleading approximate optimum or recovery remains an unmet acceptance item; do not declare success merely because evidence is finite.
- [ ] Any necessary refinement stays within the approved block-approximation design; SLQ, a dense cross-field Schur construction, or new scientific scope requires a separate decision.
- [ ] Reordered candidate evaluations are reproducible, and optimization failures and search-boundary optima remain visible.
- [ ] A runnable small-case calibration workflow and compact retained result record make the comparison independently repeatable.

## Scope and handoff

This ticket is the approved single-step validation milestone. Ticket 08 is
blocked by it to preserve the spec's delivery order, not because dense LM
mathematically requires operator evidence. Full-size plots and persistence
are delivered by ticket 11, which can proceed alongside ticket 08.

