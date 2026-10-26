# 10: Transfer a source template and render it through shared support

**What to build:** The representative pipeline completes reconstruction export, source-template validation, consumption by the next stage, and diagnostic rendering through reusable support while retaining its existing data contracts.

**Blocked by:** 09: Run sampling, persistence, and recovery through shared support

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] Complete a representative operator SLACS source-template transition using the explicit configuration and stage-support contract introduced by the preceding ticket.
- [ ] Centralize reusable reconstruction export and prepared-template validation while preserving their different required fields and their existing numerical representations.
- [ ] Validation returns the same dictionary object and preserves in-place scale-map generation or coercion, square-grid checks, obsolete-key rejection, and optional expected-bounding-box checks.
- [ ] Pass grid size, expected bounding box when applicable, and existing case-specific normalization inputs explicitly; preserve the next stage's template and fixed-grid interpretation.
- [ ] Extract reusable plotting support that consumes public results and diagnostics, preserving the representative layout and allowing meaningful case-specific render variants.
- [ ] Verify the complete export-to-next-stage path, dictionary identity, applicable validation failures, preserved artifact contents, and rendered data with small inputs.
- [ ] Reject invalid reconstruction before formal export or plotting without rejecting valid reconstructions merely because optional diagnostics are unavailable.
- [ ] Shared helpers are directly importable without added process mutations; keep unmigrated case helpers working and record focused verification.
