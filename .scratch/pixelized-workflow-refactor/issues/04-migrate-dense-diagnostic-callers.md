# 04: Migrate remaining dense plots and source exports

**What to build:** Maintained dense examples generate reconstructions, diagnostic plots, and source exports through the supported public interfaces with their existing numerical and artifact conventions.

**Blocked by:** 02: Deliver dense diagnostics through a complete example

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] Migrate the remaining dense simple fitting examples, dense SLACS and pre-subtraction pipelines, dense joint MGE example, and the matrix variant located among operator examples, according to their actual backend.
- [ ] Replace application-level private solving and duplicated diagnostic calculations with public evaluation or public evaluation-with-diagnostics as appropriate; each requested reconstruction is solved once.
- [ ] Preserve dense source-only and joint diagnostic conventions, masks, noise normalization, source grids, position constraints, and existing plot layouts, colors, and overlays.
- [ ] Reject nonfinite reconstruction before formal plotting or source-package export without inventing a dense convergence-status object.
- [ ] Preserve raw source-export keys and numerical representations and distinguish raw exports from prepared template packages.
- [ ] Extend appropriate example regressions with representative real numerical and artifact checks; leave sampler, storage, and stage-flow consolidation for the later shared-support tickets.
- [ ] Record the migrated caller inventory and focused verification results, retaining compatibility for unrelated callers.
