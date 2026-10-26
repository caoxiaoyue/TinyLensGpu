# 13: Migrate operator survey workflows to shared run support

**What to build:** SLACS, Euclid, and lens-light pre-subtraction workflows consistently use the shared example support while remaining recognizable, runnable, and resumable in their existing forms.

**Blocked by:** 10: Transfer a source template and render it through shared support

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] Complete shared-support migration for the remaining operator SLACS stages, Euclid, and standard, Sersic, MGE, and split pre-subtraction variants.
- [ ] Preserve case-specific explicit configuration, stage model construction, priors, parameter transfer, sampling settings, and input/output location behavior.
- [ ] Preserve target-specific launch entry points, stage cache formats, filenames, and recovery behavior while replacing repeated sampling and storage helpers.
- [ ] Reuse source-package validation and transfer without altering mask, bounding-box, template, or regularization semantics.
- [ ] Retain layout, zoom, mask display, colors, and overlays through shared plotting support and case-specific render choices.
- [ ] Verify representative target entry points, stage recovery and transition, artifact compatibility, and output data, including differences between survey families; record focused results.
