# 14: Complete usage documentation and integration verification

**What to build:** A researcher can discover, configure, run, and interpret the complete potential-correction feature, and a reviewer can trace every approved requirement to implementation and recorded verification.

**Blocked by:** 13 — Validate float32 and measure GPU performance.

**Status:** ready-for-agent
**State:** claimed

**Parent:** [Approved potential-correction spec](../spec.md).

- [ ] User-facing documentation explains both algorithm entry points, backend selection, configuration, truth-derived example initialization, result fields, and reproducible example use.
- [ ] Documentation distinguishes core nodes from support unknowns, physical derivatives from pixel-unit smoothing, full fourth differences from boundary fallback, and the explicit weak amplitude prior from numerical jitter.
- [ ] Evidence documentation distinguishes the single-step conditional Gaussian model, iterative local Gauss–Newton approximation, and operator block determinant approximation.
- [ ] Guidance explains convergence/failure statuses, invalid optimization candidates, boundary optima, valid derivative domains, raw/aligned truth comparisons, and both single-step renderings.
- [ ] A requirement-coverage record accounts for the full parent spec and all 14 tickets, with commands, results, exact configurations, and remaining limitations.
- [ ] Relevant core, example, and benchmark suites run in separate processes using the documented project environment, after their focused per-ticket checks.
- [ ] Integration regressions cover existing pixelized behavior, public interfaces, source boundaries, PSF semantics, supported differentiation in changed shared code, and source–lens-light architectural contracts.
- [ ] End-to-end examples remain usable without the PyAuto runtime; reference-environment work is not mistaken for target-library validation.
- [ ] Scientific recovery and approximation validation are assessed from their recorded results; unresolved failures remain open work rather than being waived by documentation.
- [ ] Usage and performance claims match measured evidence and the approved exclusions, including no calibrated detection, mass intervals, or required posterior sampling.
- [ ] Publication and implementation lifecycle records remain truthful: do not claim unrun checks, silently close unresolved tickets, or rewrite the parent spec to fit an implementation gap.

## Scope and handoff

This is final integration and documentation after every slice has already
carried its own tests. It is not the first testing phase and does not authorize
scope expansion. Record any unresolved issue explicitly rather than declaring
the feature complete.


## Comments

- 2026-10-29: Claimed by the final-integration agent on `codex/pc-14` in `/tmp/tinylens-pc-14`, based on integration `5170da8e5ec6f162ff6b66ab7b75b8a4b6fd2bed` after tickets 01–13 were resolved. This ticket owns discoverable documentation, complete requirement coverage and broad integration verification. The parent spec remains open pending the coordinator’s final independent review/fix gate.
