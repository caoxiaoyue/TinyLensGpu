## Why

Pixelized source reconstruction currently accepts independent `nx`/`ny` grid sizes and independently inferred x/y source-plane spans. This can produce rectangular source grids or `N x N` arrays with non-square physical pixels, which makes finite-difference regularization geometry harder to reason about and can silently diverge from the intended square source-plane discretization.

This change makes the public pixelized-source reconstruction path consistently square: the pixel count is `N x N` and the source-plane bbox has equal x/y extent.

## What Changes

- **BREAKING**: `PixelizedSourceModel` SHALL reject non-square source shapes where `nx != ny`.
- Add a square source-bbox inference path for pixelized source likelihoods that expands the shorter inferred span around its center so `xmax - xmin == ymax - ymin`.
- Require `fixed_source_bbox` inputs used by the operator pixelized likelihood to be square.
- Preserve lower-level rectangular mapping and regularization helpers as implementation details where useful for tests and compatibility.
- Update pixelized-source demos and S0 package validation to use a single square source-grid size and to reject rectangular S0 bboxes.
- Add focused unit tests for square-shape validation, square bbox construction, dense/operator likelihood bbox behavior, and fixed-S0 validation.

## Capabilities

### New Capabilities

- `square-pixelized-source-grid`: Pixelized source reconstruction uses square source grids and square source-plane bboxes in public likelihood paths.

### Modified Capabilities

- None.

## Impact

- Affected physical model API: `TinyLensGpu.PhysicalModel.LensImage.Pixelized.Light.PixelizedSourceModel`.
- Affected forward simulation utilities: source bbox inference and source-grid construction paths used by pixelized dense and operator simulators.
- Affected observation models: `PixelizedImageProbModel` and `PixelizedImageProbModelOperator`, especially fixed `S0` bbox validation.
- Affected demos: pixelized-source operator pipelines that currently define both `NSRCX` and `NSRCY`.
- No new runtime dependencies are expected.
- GPU/JAX performance should remain neutral: square bbox expansion changes scalar bounds only and does not add per-pixel work; forcing `nx == ny` may simplify future JIT/static-shape assumptions.

## Non-goals

- Remove or rewrite all lower-level rectangular mapping utilities in this change.
- Change the mathematical finite-difference regularization stencil beyond ensuring square public geometry.
- Add adaptive mesh refinement or non-Cartesian source grids.
- Recompute existing archived OpenSpec changes or historical pipeline artifacts.
