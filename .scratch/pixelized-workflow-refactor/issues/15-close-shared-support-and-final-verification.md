# 15: Complete shared support migration and final verification

**What to build:** All affected maintained workflows use the approved public numerical interfaces and reusable example support, with documented compatibility and a complete final regression record for the three P1 deliverables.

**Blocked by:** 11: Migrate dense pipelines to shared run support; 12: Migrate basic operator workflows to shared run support; 13: Migrate operator survey workflows to shared run support; 14: Migrate operator joint and subhalo workflows to shared run support

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Reconcile the affected maintained entry-point and caller inventory and complete any missed migrations before declaring the effort complete.
- [x] Remove repeated case-support implementations that no longer have callers; retain necessary compatibility wrappers, explicit case-stage assembly, and stable public model interfaces.
- [x] Verify that extracted shared support can be imported directly in tests without AST extraction and without additional working-directory, environment-policy, or JAX precision mutations.
- [x] Confirm artifact and launch compatibility, source-package identity and normalization, protected-stage behavior, and the approved distinction between unavailable diagnostics and invalid reconstruction.
- [x] Update relevant usage documentation for configuration, shared support, diagnostic availability, native array truthiness, and the approved behavior corrections without bulk rewriting historical experiments.
- [x] Run the core regression suite once and the examples regression suite once, in separate processes using the documented project environment; include relevant unchanged gradient and batching coverage.
- [x] Run affected existing benchmark checks where relevant without inventing a new timing threshold, and record commands, outcomes, limitations, and retained evidence with the implementation work.
- [x] Keep the three P1 changes independently reviewable and reversible; do not add new operator joint effective-parameter mathematics or require long full-data scientific sampling as an acceptance gate.

## Comments

- Claimed by Codex on `integration/pixelized-workflow-refactor` after tickets 11–14 resolved; completing inventory, compatibility harnesses, documentation, review and final verification.

- Final two-axis review: Spec reported no findings. Standards reported one P2
  test-isolation issue: the joint-support loader imported the simultaneous-MGE
  entry point without restoring its `os.chdir` side effect. Fixed the loader
  with a nested monkeypatch context that records the caller's directory before
  module execution and restores it immediately afterward (including exceptions).
  Verification in conda `tinylens_gpu`, with `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  and `MPLBACKEND=Agg`: direct real MGE-import probe asserted the original cwd
  both immediately after `load_case` and after outer monkeypatch teardown;
  `pytest test_joint_shared_support.py -q` from `examples/tests` -> **7 passed**.
  No scientific model, sampler controls, or saved artifacts changed.

- Final inventory covers 37 maintained shared-support callers (see
  [overview](../README.md#shared-support-inventory-tickets-1115)); no direct
  sampler construction or private tuple sampler remains in either example tree.
  Removed unused bbox/axis helpers and obsolete imports, centralized subhalo
  comparison-cache readers, retained required CSV/gzip and Galan cache-payload
  annotation formats, and documented explicit configuration, launch/resume,
  diagnostics availability and formal failure policy in the guide.
- Updated old isolated script-test namespaces for shared imports and moved
  B1938 source fixture construction to the public axis helper. Corrected dense
  demo title initialization order found by real rendering regressions.
- Two-axis review (`git diff 83f6002...HEAD`): Standards found one P2 test cwd
  leak, fixed in `c917a4e` (merged `5e35992`) and independently rechecked; Spec
  found no actionable omissions or behavior deviations. Physical-model and
  stage-builder constructor arguments remain unchanged from the resume baseline.
- Final verification on 2026-10-27, conda `tinylens_gpu`, JAX CUDA/RTX 5080,
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `MPLBACKEND=Agg`, repository root,
  each suite in a separate process:
  - `pytest -q` → **813 passed** (367.59 s), including unchanged gradients and
    batching coverage.
  - `pytest examples/tests -q` → **256 passed** (154.77 s). The first complete
    run's three B1938 failures were fixture references to the removed private
    helper; corrected in `45c241b`, verified with 9 focused checks, then the
    complete suite was rerun successfully.
  - `pytest benchmarks -q` → **9 passed** (8.37 s), run without other test
    suites concurrently; existing thresholds retained.
  - `git diff 83f6002...HEAD --check` → clean.
- Retained evidence, configuration, commands, review outcomes and limitations:
  [verification record](../verification.md). Small numerical/controlled-sampling
  checks meet the approved acceptance scope; no full-data sampling or new joint
  effective-parameter estimator was introduced. Tickets 01–15 are resolved.
