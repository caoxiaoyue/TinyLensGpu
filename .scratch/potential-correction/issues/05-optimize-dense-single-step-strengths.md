# 05: Optimize dense single-step regularization strengths

**What to build:** A researcher can automatically select source and potential smoothing strengths for a dense single-step fit using bounded DIRECT optimization, while retaining fixed initialization and an explicit fixed amplitude prior.

**Blocked by:** 04 — Dense single-step reconstruction at fixed strengths.

**Status:** ready-for-agent
**State:** open

**Parent:** [Approved potential-correction spec](../spec.md).

- [ ] DIRECT searches the logarithms of the positive source and fourth-difference strengths through the public fit/evidence boundary.
- [ ] Search bounds and controls are explicit and returned or saved with the result; an optimum at a search boundary is reported.
- [ ] The independent weak zero-order strength remains fixed and does not scale with the searched fourth-difference strength.
- [ ] Fixed source/potential geometry and the same initial source-gradient provider are used for every candidate.
- [ ] Each candidate performs the required single-step joint fit; there is no search-order-dependent reconstruction warm start.
- [ ] Configuration errors and numerical candidate failures remain distinguishable; numerical failures use the optimizer's invalid-objective convention and retain their reasons.
- [ ] Optimizer integration preserves the requested numerical precision instead of forcing a float32 parameter path.
- [ ] A callable remains available for other optimizers or samplers without introducing a required posterior-sampling workflow.
- [ ] A small runnable example returns selected strengths and their reconstruction; tests compare with an independent bounded scan and check repeated/reordered candidate evaluations.
- [ ] Tests cover explicit bounds, boundary reporting, fixed amplitude-prior semantics, and failure propagation.

## Scope and handoff

Use the dense evidence reference. The operator adapter and cross-backend
calibration are ticket 07; this ticket does not select the final scientific
demo's weak-prior value or search limits without the agreed sensitivity record.

