# 12: Iterative scientific examples, diagnostics, and exports

**What to build:** A researcher can run the same scientific mock through either iterative backend, inspect nonlinear recovery and convergence history, and compare it with the single-step results using a consistent saved-result and diagnostic workflow.

**Blocked by:** 10 — Optimize strengths around complete iterative reconstructions; 11 — Single-step scientific examples, diagnostics, and exports.

**Status:** ready-for-agent
**State:** resolved

**Parent:** [Approved potential-correction spec](../spec.md).

- [x] A documented iterative example supports both fixed-strength and complete outer-optimization workflows on the approved subhalo/no-subhalo mocks.
- [x] Macro parameters, source geometry, imaging conventions, and truth-comparison definitions match the stated single-step comparison configuration.
- [x] Initial source-node intensities come from truth and potential starts at zero; exported provenance distinguishes this from the single-step frozen-gradient initialization.
- [x] Result persistence includes the final nonlinear image and residuals, recovered fields, valid domains, selected strengths, local/block evidence annotations, and termination status.
- [x] Iteration history and diagnostic plots expose objective progression, accepted/rejected updates, relevant damping information, and the final stopping reason.
- [x] Single-step/iterative comparison reports use matched configurations and distinguish linearization effects from determinant approximations.
- [x] Shared exports/plots consume the common public result, with algorithm-specific fields and images represented explicitly rather than filled with misleading placeholders.
- [x] Unconverged or nonfinite runs cannot appear as successful final scientific artifacts, while diagnostic failure records remain inspectable.
- [x] Small example tests verify the two backends, nested-workflow assembly, history persistence, common metadata, and failure propagation.
- [x] Scientific example results and reproduction instructions are retained with actual grid, prior, noise, precision, and integration settings.

## Scope and handoff

Do not add macro-mismatch stress tests, posterior sampling, uncertainty maps,
or formal detection claims. This completes the runnable scientific workflows
that the precision/performance ticket will measure.


## Comments

- 2026-10-29: Claimed on `codex/pc-12` in `/tmp/tinylens-pc-12`, based on integration `af96b07961fa8c27d64bf2b3b576211fe82af61b`; blockers 10 and 11 are resolved. This owner implements the narrow diagnosed iterative-operator residual correction and the scientific workflow, using the already approved public result/operator/export seams.
- 2026-10-29: Resolved. Implementation commits `b65c691` (public red regression then bounded residual correction), `4322be3` (scientific runner/nested assembly/history), `764e4b1` (matched comparisons and explicit measured controls), and `323e00f` (nominal control caption/public-result reproduction). The [validation report](../../../docs/validation/potential-correction-iterative-scientific.md) and [compact retained records](../research/iterative-scientific/) document actual matched 200-pixel fits at clean numerical revision `764e4b1`, all four genuinely converged, exact ticket11 operator partition reuse, separate algorithm/determinant comparisons, final true residuals and aggregate budgets. Actual artifacts remain in the root checkout's ignored `.scratch/potential-correction/output/iterative-scientific/`; 160 current/reference hashes and 33 historical diagnosis hashes were verified. Replotting preserved numerical NPZ/JSON hashes.
- 2026-10-29: Verification in `tinylens_gpu`, preallocation false, separate processes and each suite's directory: core `pytest -q test_potential_correction_residual_refinement.py test_potential_correction_iterative_inference.py` **20 passed**; examples `pytest -q test_potential_correction_iterative_scientific.py test_potential_correction_scientific.py` **9 passed**; benchmarks `pytest -q -s test_potential_correction_memory.py` **2 passed**. The earlier operator compatibility check had 18 passes and the known coupled float32 stopping failure; one unchanged-integration baseline also failed. No test/default was relaxed; this remains ticket13's separate handoff.
- 2026-10-29: Scientific limits remain explicit: localized kappa improves with a 52% suppressed peak, raw low modes are prior-sensitive, and local/block evidence offsets are substantial. The measured lambda0 10000-to-30000 sensitivity and failed 3000 neighbors are both retained; lower-side convergence is unverified and no globally weak-prior claim is made. Production results use fixed smoothing; complete nested workflows are supported and tested without claiming a converged 200-pixel outer search. Latest integration `8ed9e7e` was merged without conflicts before handoff. Precision/performance and final whole-spec verification remain 13/14.
