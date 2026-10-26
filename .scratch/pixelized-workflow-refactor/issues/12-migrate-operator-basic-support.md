# 12: Migrate basic operator workflows to shared run support

**What to build:** Basic operator, no-lens-light, adaptive, and Galan workflows reuse the shared support without changing their source templates, grid policies, sampling choices, or scientific outputs.

**Blocked by:** 10: Transfer a source template and render it through shared support

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] Migrate the affected reusable support in basic operator workflows, standard and adaptive no-lens-light pipelines, and inferred- and predefined-bounding-box Galan pipelines.
- [ ] Use explicit configuration and StagePosterior-based shared sampling while preserving each case's actual parameters, batch behavior, and input/output locations.
- [ ] Preserve adaptive template propagation, fixed-grid interpretation, source-package identity and normalization, and optional expected-bounding-box enforcement.
- [ ] Preserve each case's stage payload format, resume behavior, example launch method, and explicit model and stage assembly.
- [ ] Use shared plotting and export support while keeping the public diagnostic results, existing render variants, and approved invalid-reconstruction handling.
- [ ] Verify representative complete stage transitions and saved artifacts for the materially different source-template and grid variants; record focused verification without requiring full scientific sampling.
