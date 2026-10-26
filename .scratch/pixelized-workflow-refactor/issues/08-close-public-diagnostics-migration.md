# 08: Complete and verify the public diagnostics migration

**What to build:** Users can rely on the documented public reconstruction and diagnostic interfaces across affected maintained callers, completing the second P1 deliverable without competing private plotting implementations.

**Blocked by:** 04: Migrate remaining dense plots and source exports; 05: Migrate operator basic and source-reconstruction cases; 06: Migrate operator survey and lens-subtracted diagnostics; 07: Correct joint reconstruction displays and migrate subhalo diagnostics

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] Reconcile the affected maintained caller inventory, including actual backend variants, and complete any missed formal plotting or source-export migrations.
- [ ] Remove application-level reconstruction and diagnostic copies that no longer have callers; retain legitimate numerical solver internals, existing public interfaces, and necessary compatibility wrappers.
- [ ] Document the new single-parameter diagnostic interface, opt-in work, method identification, unavailable operator joint statistics, and approved case-level failure behavior.
- [ ] Verify representative numerical equivalence, single-reconstruction behavior, component reconstruction, and unsupported-versus-invalid distinctions through the public interfaces.
- [ ] Run the relevant existing gradient, batching, and failure regressions for unchanged numerical entry points, and verify that ordinary likelihood evaluation performs no optional diagnostic work.
- [ ] Record focused core and example verification separately and mark the second P1 ready for independent review; reserve the full-suite final gate for the final integration ticket.
