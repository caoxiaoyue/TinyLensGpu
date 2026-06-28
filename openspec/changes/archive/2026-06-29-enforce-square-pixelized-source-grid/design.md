## Context

Pixelized source reconstruction spans multiple layers:

- PhysicalModel: `PixelizedSourceModel` owns the source-grid shape and regularization parameters as a Caskade module.
- ForwardSimulation: `PixelizedLensSimulator` and `PixelizedLensOperator` infer source bboxes, build source grids, and construct dense/operator lens mappings.
- ObservationModel: `PixelizedImageProbModel` and `PixelizedImageProbModelOperator` evaluate evidence and validate fixed S0 inputs.
- Inference: samplers consume these models through existing Caskade dynamic-parameter traversal and likelihood builders; no sampler changes are needed.

The current implementation separates pixel counts (`nx`, `ny`) from source-plane spans (`xmin`, `xmax`, `ymin`, `ymax`). Even when examples use `nx == ny`, automatic bbox inference can create unequal physical extents, so source pixels are rectangular in physical units. Finite-difference regularization already compensates with separate `scale_x` and `scale_y`, but the requested model contract is simpler and stricter: pixelized source grids should be square.

## Goals / Non-Goals

**Goals:**

- Enforce square pixelized source shapes at the public `PixelizedSourceModel` boundary.
- Ensure pixelized likelihood paths use square source-plane bboxes, including inferred bboxes and fixed S0 bboxes.
- Keep JAX tracing behavior stable by using scalar bbox transformations and preserving static `nx`/`ny` shape arguments.
- Update examples and tests to express source resolution with one square-grid size where practical.

**Non-Goals:**

- Remove every low-level rectangular helper in this change.
- Rewrite finite-difference regularization stencils.
- Change Caskade parameter ownership or sampler prior extraction.
- Add adaptive, irregular, or non-Cartesian source grids.

## Decisions

### Decision: Enforce `nx == ny` in `PixelizedSourceModel`

`PixelizedSourceModel.__init__` will validate that integer `nx` and `ny` are equal before storing them. Existing attributes `nx` and `ny` remain available so downstream code, cached packages, and reshape logic do not need a broad API rewrite. A convenience alias such as `n` or `source_n` can be added, but the implementation should not require callers to migrate immediately.

Alternative considered: replace `nx, ny` with a single required `n`. That is cleaner long term, but it would create unnecessary churn across examples and tests. Validation gives the behavior change with less API breakage.

### Decision: Square bbox construction expands the shorter span

Add a small source-bbox helper in `TinyLensGpu.utils.lensing.mapping`, exposed through `TinyLensGpu.utils`, with behavior equivalent to:

```python
def make_square_bbox(
    xmin: Array,
    xmax: Array,
    ymin: Array,
    ymax: Array,
) -> tuple[Array, Array, Array, Array]:
    ...
```

The helper computes x/y centers, takes `side = max(xmax - xmin, ymax - ymin)`, and returns equal spans around the original centers. It should work with JAX arrays and avoid Python-side conversion in traced likelihood paths.

The existing `infer_source_bbox()` can either accept `square=True` or remain rectangular with a new `infer_square_source_bbox()` wrapper. Pixelized dense/operator simulators should call the square path. Generic rectangular utilities can remain unchanged.

Alternative considered: crop the longer span to the shorter span. That risks excluding ray-traced source coordinates. Expanding the shorter span preserves coverage.

### Decision: Validate fixed source bboxes at the observation-model boundary

`PixelizedImageProbModelOperator._validate_fixed_source_bbox()` will reject non-square bboxes. Fixed S0 packages in demos should perform the same check before constructing likelihoods. This prevents adaptive regularization from reusing a rectangular cached template with a square-only source model.

Alternative considered: silently square fixed bboxes. That would make the bbox no longer match the saved `source_x_axis`, `source_y_axis`, and source template provenance. Failing clearly is safer.

### Decision: Keep rectangular low-level helpers during this change

Functions such as `lens_mapping_operator_bilinear_rectangular_from()` and `build_source_grid(nx, ny, ...)` may stay general. Existing low-level regularization tests intentionally cover non-square shapes and `dx != dy`; those tests still protect helper correctness. Public pixelized source and probability-model paths become square-only.

Alternative considered: rename and simplify all helpers to square-only immediately. That would create a larger migration with little user-visible value and higher regression risk.

### Decision: Preserve GPU/JAX performance characteristics

Square bbox handling is scalar work before mapping/regularization construction. It should not materialize additional arrays or add per-pixel operations. Validation that requires Python booleans should stay in constructors and non-JIT setup paths; traced likelihood paths should use JAX scalar arithmetic.

## Risks / Trade-offs

- [Risk] Existing external callers may pass `nx != ny`. -> Mitigation: raise a clear `ValueError` at construction with migration guidance to choose one square resolution.
- [Risk] Expanding the shorter bbox increases source pixels covering empty regions. -> Mitigation: this preserves ray-traced coverage and keeps pixel count fixed; users can adjust source masks/padding if needed.
- [Risk] Existing S0 caches may have rectangular bboxes. -> Mitigation: validate caches and fail with a message requiring S0 regeneration under square bbox rules.
- [Risk] Tests for low-level rectangular helpers may appear inconsistent with the new public contract. -> Mitigation: document that rectangular helpers are retained as internal/general utilities while public pixelized-source likelihood paths enforce square geometry.

## Migration Plan

1. Add bbox square helper and tests for asymmetric, offset, padded, and point-like inputs.
2. Add `PixelizedSourceModel` square-shape validation while preserving `nx`/`ny` attributes.
3. Route dense and operator pixelized bbox inference through the square helper.
4. Validate fixed operator bboxes and S0 demo packages as square.
5. Update demos from paired `NSRCX`/`NSRCY` constants to a single `NSRC` where practical.
6. Run focused tests from their directories: `tests/test_pixelized_source_utils.py`, `tests/test_pixelized_inversion.py`, `tests/test_pixelized_operator.py`, and `tests/test_regularization.py` selections.

Rollback is straightforward: restore rectangular bbox inference in pixelized simulators and relax `PixelizedSourceModel`/fixed-bbox validation.

## Open Questions

- Should a single `n`/`npix` keyword become the preferred documented constructor in a follow-up change?
- Should archived or generated S0 artifacts include an explicit `square_grid=True` metadata flag for cache diagnostics?
