# 01: Fit bilinear pixelized sources with an adapt image

**What to build:** A modeler can use a fixed native image-plane reference arc with either ordinary pixelized image probability model. Each bilinear fit uses the proposed mass to project that reference onto the fixed source grid and reports the scale used in the reconstruction. Introduce the new input alongside the fixed-template input for the migration.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Both constructors accept a finite `adapt_image` aligned with `image_data` and require a valid square `fixed_source_bbox` for active adaptation. Invalid shape, nonfinite values and missing adaptive inputs fail at construction. A disabled fixed rho uses the uniform path, while dynamic rho initialized at zero still requires the adaptive inputs.
- [x] Prepare the fixed reference once using the positive active-pixel 99.5th-percentile normalization in the spec. An independent masked subpixel example matches the bilinear weighted average at coverage above 10⁻⁸ and the zero result at smaller coverage.
- [x] Matched dense and operator fits agree on the applied scale and reconstruction within the declared numerical tolerance. The scale reaches the finite-difference prior, reconstruction and score for the same proposal.
- [x] Fixed mass reuses invariant geometry and projected brightness. Changing mass, rho or their registered dependencies produces the correct new prior under repeated, compiled and batched calls.
- [x] Public diagnostics report the actual scale from one primary solve. The fixed-template path remains available during caller migration, with a clear validation error for ambiguous mixed adaptive inputs.

## Comments

- Claimed on `codex/overlap-01-bilinear`.
- Implemented in `69916b5`. From `tests/` in the `tinylens_gpu` environment, CUDA `pytest -q test_adapt_image.py` passed all 8 cases.
- Broader CUDA regression on the RTX 5080 from `tests/` passed 108 cases, with 259 deselected. Command `pytest -q test_adapt_image.py test_pixelized_inversion.py test_pixelized_operator.py test_regularization.py -k 'adapt or cache or diagnostic or reconstruction_skips'`.
