# 03: Render supplied potential corrections

**What to build:** Given a fixed macro model, a square pixelized source, an observation configuration, and supplied potential values, a caller can obtain a corrected image and valid-domain deflection/convergence products without cleaning away observed pixels.

**Blocked by:** 01 — Prepare source regularization and precision.

**Status:** ready-for-agent
**State:** open

**Parent:** [Approved potential-correction spec](../spec.md).

- [ ] Core potential nodes are the union of the actual four coarse-grid interpolation corners required by the forward model's image-subpixel queries.
- [ ] Potential unknown support is the unique union of core nodes and their axial offsets of one and two potential-grid steps in each direction, including support beyond the image edge where needed.
- [ ] Observation pixels are retained for irregular masks, narrow arcs, holes, disconnected selections, isolated selected pixels, and image-edge queries.
- [ ] Potential spacing is configurable, defaults to twice the image-pixel spacing, and is independent of source resolution.
- [ ] Centered physical derivatives on core nodes use angular spacings and the established coordinate orientation; support-only nodes are not recursively required to have derivatives.
- [ ] Prepared four-corner interpolation transfers core-node deflections to subpixels without nearest-valid-cell searches, hole-crossing fallback, or implicit zero padding.
- [ ] Corrected source-plane coordinates subtract the correction deflection from the macro ray-traced coordinates, respecting the existing macro interface's return semantics.
- [ ] The resulting image retains the current pixelized PSF, subpixel integration, fixed-source interpolation, and boundary behavior.
- [ ] Returned products identify core/support nodes and valid deflection/convergence coverage; malformed explicit geometry fails informatively rather than discarding observations.
- [ ] Small analytical-potential examples verify derivative accuracy, sign/orientation, unchanged zero-correction rendering, and forward/transpose contracts where applicable.
- [ ] Static geometry can be prepared once and reused by JAX numerical kernels with explicit precision, without adding a testing-only public API.

## Scope and handoff

This is a supplied-field rendering slice, not an inversion. Its prepared
geometry and numerical rendering boundary are reused by the fit entry points.
Separate fixed geometry from macro-dependent mappings so future single-step
macro variation can rebuild the latter correctly.

