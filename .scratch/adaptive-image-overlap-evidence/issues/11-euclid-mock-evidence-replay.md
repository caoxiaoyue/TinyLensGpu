# 11: Reproduce the 80×80 Euclid mock with image-driven adaptation

**What to build:** A researcher can rerun the original Euclid mock with the new library and inspect the determinant approximation separately from reconstruction error. The run supplies a reproducible M3 configuration for optimizer qualification.

**Blocked by:** [05 — Differentiate and batch the full Overlap score](05-overlap-gradients-and-batching.md).

**Status:** ready-for-agent
**State:** claimed

**Spec:** [Parent specification](../spec.md)

- [ ] Use the archived no-noise mock with its RMS weighting, 80×80 source grid, mask, PSF, subpixel integration and recorded SIE M1 reference producer.
- [ ] Run a new M2 calibration and M3 fit with the image-driven prior. Record the actual model, interpolation, adaptive settings, fitted strengths and evidence method.
- [ ] At matched proposals, compare Block, Overlap and complete determinant scores using the same primary solution. Report absolute offsets, centered errors and ranking as separate quantities.
- [ ] Retain a reproducible runner, input identities, settings and small result summaries. Record slope recovery and validity for the actual M3 configuration.

## Comments

- Claimed on `codex/overlap-11-mock-replay`.
