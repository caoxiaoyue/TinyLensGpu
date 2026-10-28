# 03: Render supplied potential corrections

**What to build:** Given a fixed macro model, a square pixelized source, an observation configuration, and supplied potential values, a caller can obtain a corrected image and valid-domain deflection/convergence products without cleaning away observed pixels.

**Blocked by:** 01 — Prepare source regularization and precision.

**Status:** ready-for-agent
**State:** resolved

**Parent:** [Approved potential-correction spec](../spec.md).

- [x] Core potential nodes are the union of the actual four coarse-grid interpolation corners required by the forward model's image-subpixel queries.
- [x] Potential unknown support is the unique union of core nodes and their axial offsets of one and two potential-grid steps in each direction, including support beyond the image edge where needed.
- [x] Observation pixels are retained for irregular masks, narrow arcs, holes, disconnected selections, isolated selected pixels, and image-edge queries.
- [x] Potential spacing is configurable, defaults to twice the image-pixel spacing, and is independent of source resolution.
- [x] Centered physical derivatives on core nodes use angular spacings and the established coordinate orientation; support-only nodes are not recursively required to have derivatives.
- [x] Prepared four-corner interpolation transfers core-node deflections to subpixels without nearest-valid-cell searches, hole-crossing fallback, or implicit zero padding.
- [x] Corrected source-plane coordinates subtract the correction deflection from the macro ray-traced coordinates, respecting the existing macro interface's return semantics.
- [x] The resulting image retains the current pixelized PSF, subpixel integration, fixed-source interpolation, and boundary behavior.
- [x] Returned products identify core/support nodes and valid deflection/convergence coverage; malformed explicit geometry fails informatively rather than discarding observations.
- [x] Small analytical-potential examples verify derivative accuracy, sign/orientation, unchanged zero-correction rendering, and forward/transpose contracts where applicable.
- [x] Static geometry can be prepared once and reused by JAX numerical kernels with explicit precision, without adding a testing-only public API.

## Scope and handoff

This is a supplied-field rendering slice, not an inversion. Its prepared
geometry and numerical rendering boundary are reused by the fit entry points.
Separate fixed geometry from macro-dependent mappings so future single-step
macro variation can rebuild the latter correctly.


## Comments

- Claimed by implementer `/root/ticket03` on `codex/pc-03`, worktree `/tmp/tinylens-pc-03`, based on integration `343df374`. The public seams are the supplied-field rendering result and focused numerical support/derivative/adjoint contracts, already approved in the parent spec.

- Implemented in `9cacd23fa0d9216847e19041d688dc1c761bd3e7` on `codex/pc-03`. Public entry point: `TinyLensGpu.ForwardSimulation.PotentialCorrectionRenderer`; static `PotentialGrid` lives in `TinyLensGpu.utils.lensing.potential`. Numerical actions, geometry domains, precision contexts and downstream API are documented in [the rendering handoff](../research/supplied-field-rendering.md).
- Verification in `tinylens_gpu` on the default RTX 5080 CUDA device: from `tests/`, `XLA_PYTHON_CLIENT_PREALLOCATE=false pytest test_potential_correction_rendering.py -q` → **17 passed in 12.54 s** after final formatting and re-exports. Tests cover float64 physical derivatives/adjoints, actual nsub=1/3 existing-imaging parity, independent nonzero lens-equation rendering, irregular/isolated/edge selections, explicit geometry errors, fixed source boundaries and signed coefficients, float32/float64/default preservation, and macro-dependent ray rebuilding. Numerical tolerances and exact small-system configurations are retained in the handoff and maintained tests.
- Public import smoke check passed through both `TinyLensGpu.ForwardSimulation` and its `LensImage` package; process x64 default remained `False`. `git diff --check` passed. No existing numerical implementation was changed; broad compatibility evidence remains the coordinator baseline and ticket 01's recorded regression suites. No inversion, recovery, or performance result is claimed by this slice.
