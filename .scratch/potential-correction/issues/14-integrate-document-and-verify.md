# 14: Complete usage documentation and integration verification

**What to build:** A researcher can discover, configure, run, and interpret the complete potential-correction feature, and a reviewer can trace every approved requirement to implementation and recorded verification.

**Blocked by:** 13 — Validate float32 and measure GPU performance.

**Status:** ready-for-agent
**State:** resolved

**Parent:** [Approved potential-correction spec](../spec.md).

- [x] User-facing documentation explains both algorithm entry points, backend selection, configuration, truth-derived example initialization, result fields, and reproducible example use.
- [x] Documentation distinguishes core nodes from support unknowns, physical derivatives from pixel-unit smoothing, full fourth differences from boundary fallback, and the explicit weak amplitude prior from numerical jitter.
- [x] Evidence documentation distinguishes the single-step conditional Gaussian model, iterative local Gauss–Newton approximation, and operator block determinant approximation.
- [x] Guidance explains convergence/failure statuses, invalid optimization candidates, boundary optima, valid derivative domains, raw/aligned truth comparisons, and both single-step renderings.
- [x] A requirement-coverage record accounts for the full parent spec and all 14 tickets, with commands, results, exact configurations, and remaining limitations.
- [x] Relevant core, example, and benchmark suites run in separate processes using the documented project environment, after their focused per-ticket checks.
- [x] Integration regressions cover existing pixelized behavior, public interfaces, source boundaries, PSF semantics, supported differentiation in changed shared code, and source–lens-light architectural contracts.
- [x] End-to-end examples remain usable without the PyAuto runtime; reference-environment work is not mistaken for target-library validation.
- [x] Scientific recovery and approximation validation are assessed from their recorded results; unresolved failures remain open work rather than being waived by documentation.
- [x] Usage and performance claims match measured evidence and the approved exclusions, including no calibrated detection, mass intervals, or required posterior sampling.
- [x] Publication and implementation lifecycle records remain truthful: do not claim unrun checks, silently close unresolved tickets, or rewrite the parent spec to fit an implementation gap.

## Scope and handoff

This is final integration and documentation after every slice has already
carried its own tests. It is not the first testing phase and does not authorize
scope expansion. Record any unresolved issue explicitly rather than declaring
the feature complete.

## Comments

- 2026-10-29: Claimed by the final-integration agent on `codex/pc-14` in `/tmp/tinylens-pc-14`, based on integration `5170da8e5ec6f162ff6b66ab7b75b8a4b6fd2bed` after tickets 01–13 were resolved. This ticket owns discoverable documentation, complete requirement coverage and broad integration verification. The parent spec remains open pending the coordinator’s final independent review/fix gate.

- 2026-10-29: Implemented in `6cdfbe9`; latest integration `ef80f51` merged without conflicts as `14b2fcf` (progress README only). Added discoverable [user guide](../../../docs/guides/potential-correction.md), README/main-guide/example/codebase-map links, and [complete coverage record](../../../docs/validation/potential-correction-coverage.md) for all 33 implementation decisions, 13 testing decisions, 50 stories and 14 tickets. Scientific controls, prior sensitivity/failed probes, supported small32 versus failed scientific float32, evidence/status/refinement/export meanings and measured-only resource claims remain explicit. Parent spec stays open for mandatory coordinator review/fix.
- A concrete declared-minimum JAX compatibility issue was reproduced through public `SingleStepPotentialCorrection.fit`: a thin external legacy `lax.map(f,xs)` adapter raised the exact unsupported `batch_size` keyword TypeError (one failed/three deselected in 3.31s on CPU). Added feature detection and a bounded legacy map of 16-column panels plus ordered remainder; modern native path and dependency floor are preserved. Four public single-step/iterative/dtype legacy-adapter cases passed on CPU in 6.44s, then on GPU in the post-fix selection. No generic likelihood/solver, prior, stopping defaults or existing numerical assertions changed.
- In `tinylens_gpu`, with preallocation false, Agg and `PYTHONPATH=/tmp/tinylens-pc-14`, full pre-compatibility `pytest -q` from the repository root: **957 passed in 567.50s**, including both critical slow residual-refinement/budget regressions, existing PNPG stress/high-resolution and legacy source/PSF/Schur/AD/public-interface tests. The four new compatibility cases were added after that collection; this is not claimed as a single 961-case post-fix run.
- Separate post-fix process from `tests/`, `pytest -q test_potential_correction_jax_compatibility.py test_potential_correction_dense.py test_potential_correction_rendering.py test_potential_correction_iterative.py test_potential_correction_precision.py`: **67 passed in 60.95s**. Separate full processes from repo root: `pytest -q examples/tests`: **286 passed in 248.88s**; `pytest -q -s benchmarks`: **14 passed in 44.20s**, with no concurrent GPU timing workers. No final failures/skips/assertion relaxation.
- [Versioned command/outcome/source/log hashes](../research/final-integration-verification.json) point to full logs in the root checkout’s ignored `.scratch/potential-correction/output/final-integration/`. No 45-worker or 200×200 scientific campaign was rerun; prior evidence retains its named implementation/configuration hashes. All three guide Python blocks compile; the full minimal public example executed successfully on CPU with scoped x64 inputs. New local links and complete coverage IDs pass. Black/Ruff defaults for changed Python files and `git diff --check` pass.
