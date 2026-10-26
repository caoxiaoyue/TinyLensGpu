# 15: Complete shared support migration and final verification

**What to build:** All affected maintained workflows use the approved public numerical interfaces and reusable example support, with documented compatibility and a complete final regression record for the three P1 deliverables.

**Blocked by:** 11: Migrate dense pipelines to shared run support; 12: Migrate basic operator workflows to shared run support; 13: Migrate operator survey workflows to shared run support; 14: Migrate operator joint and subhalo workflows to shared run support

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] Reconcile the affected maintained entry-point and caller inventory and complete any missed migrations before declaring the effort complete.
- [ ] Remove repeated case-support implementations that no longer have callers; retain necessary compatibility wrappers, explicit case-stage assembly, and stable public model interfaces.
- [ ] Verify that extracted shared support can be imported directly in tests without AST extraction and without additional working-directory, environment-policy, or JAX precision mutations.
- [ ] Confirm artifact and launch compatibility, source-package identity and normalization, protected-stage behavior, and the approved distinction between unavailable diagnostics and invalid reconstruction.
- [ ] Update relevant usage documentation for configuration, shared support, diagnostic availability, native array truthiness, and the approved behavior corrections without bulk rewriting historical experiments.
- [ ] Run the core regression suite once and the examples regression suite once, in separate processes using the documented project environment; include relevant unchanged gradient and batching coverage.
- [ ] Run affected existing benchmark checks where relevant without inventing a new timing threshold, and record commands, outcomes, limitations, and retained evidence with the implementation work.
- [ ] Keep the three P1 changes independently reviewable and reversible; do not add new operator joint effective-parameter mathematics or require long full-data scientific sampling as an acceptance gate.
