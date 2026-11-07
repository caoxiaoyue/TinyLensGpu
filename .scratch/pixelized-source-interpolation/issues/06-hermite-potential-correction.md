# 06: Run single-step and iterative potential correction with Hermite

**What to build:** A potential-correction user can explicitly select Hermite on the renderer and use it for supplied-field rendering, a single-step fit, or an iterative fit with either reconstruction backend. Omitting the selection continues to give the existing bilinear model.

**Blocked by:** 01 — Reconstruct an opt-in Hermite source with the dense backend.

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] `PotentialCorrectionRenderer` validates the explicit `source_interpolation` choice and uses the shared stencil for brightness, adjoints, and spatial derivatives. Both modes agree with ordinary dense pixelized imaging at zero potential and the same source geometry.
- [x] The single-step gradient callback retains its explicit frozen macro-ray contract. Its Hermite values are used inside the extended support and zeroed at the outer edge and beyond. The bilinear mask retains the established behavior. Exercise all of these domains with a callback that exposes masking errors.
- [x] The iterative method refreshes derivatives from the current source and corrected rays using the selected interpolant. Check its Jacobian against the nonlinear forward and verify the transpose identity.
- [x] Both methods record the actual source interpolation, boundary, and gradient provenance in the applicable result configuration. The original source bounding box continues to identify the fitted grid.
- [x] For small fixed-strength problems, compare dense and operator objectives, true residuals, and source and potential reconstructions on their valid domains. Preserve legacy bilinear regressions and the existing solve and evidence methods.

## Comments

Claimed for implementation on `codex/psi-06-correction`.

Implemented in `228e725`, merged as `011f173`. From `tests/` in
`tinylens_gpu` with CUDA and preallocation disabled, 160 tests passed.

```bash
pytest -xq -m 'not slow' test_potential_correction_rendering.py \
  test_potential_correction_dense.py test_potential_correction_operator.py \
  test_potential_correction_iterative.py test_potential_correction_iterative_operator.py \
  test_potential_correction_precision.py test_potential_correction_jax_compatibility.py \
  test_potential_correction_inference.py test_potential_correction_iterative_inference.py \
  test_pixelized_source_interpolation.py
```

The Float64 Hermite fixed-strength single-step fits agreed to
`3.74e-12` in potential and `6.25e-14` in the image, with true relative
residuals below `5.7e-14`. The iterative fits agreed to `2.12e-16` in
potential and `4.44e-16` in the image, with maximum accepted true relative
residuals below `8.6e-13`. The suite exercises the frozen callback domain
and the refreshed Jacobian and its transpose.

The follow-up CUDA command `pytest -xq test_potential_correction_rendering.py
test_potential_correction_residual_refinement.py` passed 25 checks,
including both slow bilinear control and solve-budget regressions.
