# 13: Qualify Overlap GPU workspace and chunk width

**What to build:** A GPU user can choose a sampler batch size from matched measurements of Overlap and Block on the qualified implementation. The report provides the evidence for a conservative automatic Overlap chunk width.

**Blocked by:** [12 — Qualify full-score multi-start M3 L-BFGS-B](12-m3-overlap-lbfgs-qualification.md).

**Status:** ready-for-agent
**State:** claimed

**Spec:** [Parent specification](../spec.md)

- [ ] Use the same device for the 80×80 source-only mock and a representative joint semi-linear inversion. Match physical inputs, precision and solver settings when comparing methods.
- [ ] Record preparation time, first compilation, warmed score and gradient time, peak device memory and inner chunk effects at matched outer sampler batches.
- [ ] Verify bounded forward and reverse scoring workspace on the tested configurations and equivalent scores for the compared chunk widths.
- [ ] Select and record a conservative automatic width from the results and demonstrate that explicit `vectorized_chunk_size` continues to control batching. Retain commands and measured configurations.

## Comments

- Claimed on `codex/overlap-13-gpu-qualification`.
