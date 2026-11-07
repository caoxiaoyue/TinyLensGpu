# 06: Run single-step and iterative potential correction with Hermite

**What to build:** A potential-correction user can explicitly select Hermite on the renderer and use it for supplied-field rendering, a single-step fit, or an iterative fit with either reconstruction backend. Omitting the selection continues to give the existing bilinear model.

**Blocked by:** 01 — Reconstruct an opt-in Hermite source with the dense backend.

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] `PotentialCorrectionRenderer` validates the explicit `source_interpolation` choice and uses the shared stencil for brightness, adjoints, and spatial derivatives. Both modes agree with ordinary dense pixelized imaging at zero potential and the same source geometry.
- [ ] The single-step gradient callback retains its explicit frozen macro-ray contract. Its Hermite values are used inside the extended support and zeroed at the outer edge and beyond. The bilinear mask retains the established behavior. Exercise all of these domains with a callback that exposes masking errors.
- [ ] The iterative method refreshes derivatives from the current source and corrected rays using the selected interpolant. Check its Jacobian against the nonlinear forward and verify the transpose identity.
- [ ] Both methods record the actual source interpolation, boundary, and gradient provenance in the applicable result configuration. The original source bounding box continues to identify the fitted grid.
- [ ] For small fixed-strength problems, compare dense and operator objectives, true residuals, and source and potential reconstructions on their valid domains. Preserve legacy bilinear regressions and the existing solve and evidence methods.

