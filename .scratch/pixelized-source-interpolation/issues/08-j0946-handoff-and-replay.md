# 08: Carry source interpolation through J0946 handoffs and replay an archived state

**What to build:** The active J0946 research handoff and its single-step and iterative consumers agree on the saved source interpolation. A fixed-state replay measures agreement between the experimental and production Hermite implementations on saved J0946 fields and rays.

**Blocked by:** 03 — Save and resume a staged fit with its source interpolation, and 06 — Run single-step and iterative potential correction with Hermite.

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] Active handoff production and consumption use the shared interpolation provenance contract. A known older library handoff with no mode field constructs a bilinear source, and a recorded Hermite choice is carried into the configured renderer.
- [ ] Preserve the recorded hard-box, experimental zero-ring, and Hermite boundary conventions when reading archived inputs. Retain the original source grid and fields.
- [ ] Replay a documented saved cubic reconstruction with its fixed source intensities, potential coefficients, corrected rays, precision, and imaging settings. Evaluate the experimental and library Hermite paths with the same inputs.
- [ ] Compare ray brightness, spatial derivatives, and the prediction after subpixel integration and PSF convolution. Record the archive identifier, input provenance, command, tolerance, and observed differences as a fixed-state comparison.

