# 01: 固定稠密模型的 source bbox

**What to build:** A lens modeler can freeze the pixelized source grid on the dense evidence model with the same square source bounding box the operator evidence model already accepts. Evidence, reconstruction, and the regularization helper all use that grid. A second model can be pointed at the same grid by reading `infer_source_bbox` from a reference model. Omitting the bbox leaves today's seed-ray inference and detached bounds unchanged. This slice fixes the grid only; it does not cache matrices.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [x] The dense evidence model accepts an optional fixed source bounding box and rejects non-finite values, inverted or zero-width bounds, and a non-square box, using the operator evidence model's existing square tolerance.
- [x] With a fixed bbox, evidence, `forward_model`, and the regularization helper use that bbox and do not infer a new one from seed rays.
- [x] Rays that fall outside the fixed grid still contribute zero mapping rows, and a non-finite evidence value still becomes the existing large negative penalty.
- [x] `infer_source_bbox` returns the stored bbox when one is fixed, and the square bbox inferred from the current seed rays when none is fixed, using the padding and outlier fraction the dense model already uses.
- [x] With no fixed bbox, evidence matches the current dense model, including detached inferred bounds and the existing rejection of dynamic or nonzero adaptive source regularization.
- [x] The position-likelihood penalty still applies on top of the solve, and existing pixelized demo pipelines are left unchanged.
