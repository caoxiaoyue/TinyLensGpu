# 10: Export new J0946 macro handoffs with the actual source scale

**What to build:** Both documented J0946 macro producers can create a fresh handoff after an adaptive fit. A downstream fixed-scale potential-correction fit can read the selected M3 draw, its actual precision scale and the scoring provenance together.

**Blocked by:** [06 — Run Euclid M1 through M3 with a shared reference arc](06-euclid-adapt-image-stages.md).

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] Both producers use the new reference-image adaptive input and record the reference and scoring configuration.
- [ ] At the selected M3 draw, obtain the fitted fields, actual adaptive scale and score method from the same valid model evaluation. Save a finite positive scale on the handoff's fitted source grid.
- [ ] A focused handoff round trip verifies the existing fixed `source_scale` consumer can read the new product. An archived product remains readable using its recorded scale and configuration.
- [ ] Limit the migration to the two documented macro entry points and their handoff behavior.
