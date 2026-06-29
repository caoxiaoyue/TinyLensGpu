## Why

Commit `f6a9a93` enforced square (`N x N`) pixelized source grids at the `PixelizedSourceModel` API boundary, but the entire supporting stack — mapping utilities, regularization builders, forward simulators, and observation models — still carries separate `nx`/`ny` parameters, `_rectangular` function names, `scale_x`/`scale_y` dual scaling, and block-partitioning logic that handles `nx != ny`. This legacy code doubles the parameter surface and makes reasoning about the system harder than necessary. Now that rectangular grids are permanently forbidden, we can remove that complexity.

## What Changes

- **BREAKING**: `PixelizedSourceModel(nx, ny)` → `PixelizedSourceModel(n)` — single grid-dimension parameter
- **BREAKING**: `DenseRegularizationBuilder(nx, ny)` → `DenseRegularizationBuilder(n)` — single grid-dimension parameter; `RegData(scale, scale_x, scale_y)` merged to `RegData(scale, scale_factor)`
- **BREAKING**: `build_source_grid(nx, ny, ...)` → `build_source_grid(n, ...)`
- **BREAKING**: `lens_mapping_operator_bilinear_rectangular_from(...)` renamed to `lens_mapping_operator_bilinear_from(...)`, `ny` parameter removed
- **BREAKING**: `infer_source_bbox(..., square=True/False)` — `square` parameter removed; behaviour is always square; `infer_square_source_bbox` merged into `infer_source_bbox`
- **BREAKING**: `source_template_scale_map(nx, ny, ...)` → `source_template_scale_map(n, ...)`
- Forward simulators (`PixelizedLensSimulator`, `PixelizedLensOperator`) lose `source_nx`/`source_ny` in favour of a single `source_n`
- Observation models (`PixelizedImageProbModel`, `PixelizedImageProbModelOperator`) adopt single-`n` grid parameter
- All `scale_x`/`scale_y` dual scaling in regularization matvecs, dense matrices, block-diagonal routines, and `diag_R` collapsed to a single `scale_factor`
- Block-partitioning `is_uniform` check simplified: only `n % block_size == 0` needed
- Tests and example scripts updated to the new API
- Existing spec `square-pixelized-source-grid` updated to reflect the removal of rectangular backward compatibility

### Non-goals

- Changing regularization mathematics or numerical behaviour (only mechanical simplifications where `dx == dy` is now guaranteed)
- Adding new pixelized source features
- Modifying the operator backend's PCG algorithm or convergence criteria
- Touching parametric (non-pixelized) source models

## Capabilities

### Modified Capabilities

- `square-pixelized-source-grid`: Requirements evolve from "enforce square at boundary, keep rectangular internals" to "remove rectangular support throughout the stack; single `n` parameter everywhere"

## Impact

- **18+ source files** across `TinyLensGpu/utils/`, `TinyLensGpu/PhysicalModel/`, `TinyLensGpu/ForwardSimulation/`, `TinyLensGpu/ObservationModel/`
- **5 test files** and **~15 example scripts** need parameter-name updates
- JIT caches will recompile due to changed function signatures (one-time cost)
- Public API breakage: any external code calling `PixelizedSourceModel(nx=..., ny=...)` or `DenseRegularizationBuilder(nx=..., ny=...)` must update to single-`n` form
