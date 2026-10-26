# 05: Migrate operator basic and source-reconstruction cases

**What to build:** The remaining basic operator and source-reconstruction cases use public reconstruction diagnostics while retaining their source-grid, template, and scientific display behavior.

**Blocked by:** 03: Deliver operator diagnostics through a complete example

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] Migrate affected remaining simple source, compact-source, and EPL-plus-shear fitting demos, the standard and adaptive no-lens-light cases, and the inferred- and predefined-bounding-box Galan cases.
- [ ] Replace private reconstruction and diagnostic assembly with public results, requesting existing source-only effective-parameter diagnostics only where previously used.
- [ ] Preserve fixed or inferred source grids, adaptive scale-map behavior, position constraints, source-export contents, and each case's plotting conventions.
- [ ] Stop formal plotting and source export for explicit operator non-convergence or nonfinite reconstruction, with informative failure context and retained saved artifacts.
- [ ] Keep the dense matrix variant assigned to the dense migration ticket rather than treating its directory location as its backend.
- [ ] Verify representative source-only, adaptive, and fixed-bounding-box workflows through public results and output artifacts; avoid a second reconstruction solely for display or export.
- [ ] Record coverage and focused verification without changing sampling settings, cache formats, or the later shared-support scope.
