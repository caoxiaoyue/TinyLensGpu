# 14: Migrate operator joint and subhalo workflows to shared run support

**What to build:** Operator joint semi-linear inversion and subhalo workflows reuse the common support while retaining their specialized sampling, posterior transfer, stage validation, and protection of saved scientific results.

**Blocked by:** 10: Transfer a source template and render it through shared support

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] Migrate affected operator simple joint examples, the operator simultaneous MGE workflow, and the 0946 and B1938 subhalo pipelines; the dense joint workflow remains assigned to the dense-support ticket.
- [ ] Preserve explicit case configuration, vectorized chunking and other sampling controls, parameter ordering, prior semantics, and stage posterior transfer.
- [ ] Preserve existing stage payload formats and restore behavior, including the B1938 M4/M5 schema checks and protection against overwriting the protected M5 artifact.
- [ ] Retain case-specific model and stage assembly while reusing applicable sampling, persistence, source-package, and plotting support.
- [ ] Preserve the corrected actual joint component images, unsupported-joint-statistic behavior, subhalo overlays, and approved failure handling from the diagnostics migration.
- [ ] Verify representative stage transitions, cache restoration and schema rejection, refusal to overwrite protected output, and relevant chunking behavior with controlled sampling and small artifacts.
- [ ] Record focused results without modifying scientific definitions or requiring full-data nested sampling.
