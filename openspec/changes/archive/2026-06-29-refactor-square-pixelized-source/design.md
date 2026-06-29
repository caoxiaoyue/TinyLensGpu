## Context

`PixelizedSourceModel` already enforces `nx == ny` at construction. However, the entire internal stack — mapping utilities, regularization builders, forward simulators, observation models — carries dual `nx`/`ny` parameters originally designed to support rectangular (non-square) source grids. These parameters propagate through function signatures, JIT static argnames, `RegData` tuples, block-partitioning, and physical-scale calculations (`scale_x`/`scale_y`). Since rectangular grids are permanently forbidden, this dual-parameter surface is dead weight.

The refactoring collapses `nx`/`ny` into a single `n` everywhere, renames the misleading `_rectangular_` function names, and merges `scale_x`/`scale_y` into a single `scale_factor` in regularization code where `dx ≡ dy` is now guaranteed.

## Goals / Non-Goals

**Goals:**
- Eliminate the `ny` parameter from all pixelized-source APIs
- Rename `lens_mapping_operator_bilinear_rectangular_from` → `lens_mapping_operator_bilinear_from`
- Merge `scale_x`/`scale_y` into a single `scale_factor` throughout `DenseRegularizationBuilder` and all callers
- Collapse `infer_source_bbox` + `infer_square_source_bbox` into a single always-square function
- Update all tests and examples to the new API
- Update the `square-pixelized-source-grid` spec to reflect full removal of rectangular support

**Non-Goals:**
- Changing regularization mathematics (all simplifications are mechanical, exploiting `dx == dy`)
- Modifying PCG solver or convergence criteria
- Adding new features
- Touching parametric source models

## Decisions

### D1: PixelizedSourceModel(n) instead of PixelizedSourceModel(nx)

**Chosen**: Single positional `n`. No `ny` alias, no deprecation period.
**Rationale**: The codebase is a research package, not a public library with stability guarantees. All call sites are internal. A clean break is simpler than maintaining backward compat shims that would themselves become tech debt.
**Rejected**: `PixelizedSourceModel(nx, *, ny=None)` with deprecation — adds code to remove later.

### D2: Single `scale_factor` instead of `(scale_x, scale_y)`

**Chosen**: `RegData(scale: Array|None, scale_factor: Array)` where `scale_factor` is a scalar JAX array computed once by `_get_scale()`.
**Rationale**: With a square grid and square bbox, `dx = (xmax-xmin)/(n-1) = (ymax-ymin)/(n-1) = dy`, so first-order scaling `1/dx^2 = 1/dy^2` and second-order `1/dx^4 = 1/dy^4` are always equal. The matvec expressions `scale_x * out_x + scale_y * out_y` simplify to `scale_factor * (out_x + out_y)`. This reduces `RegData` from 3 fields to 2 and eliminates per-axis branching in many helper methods.
**Rejected**: Keep `scale_x`/`scale_y` separate "for safety" — they are now guaranteed equal; keeping both invites confusion.

### D3: Merge `infer_square_source_bbox` into `infer_source_bbox`

**Chosen**: `infer_source_bbox(beta_x, beta_y, padding=0.0, outlier_frac=0.01)` always returns a square bbox. Drop the `square` parameter and delete the `infer_square_source_bbox` wrapper.
**Rationale**: All callers already pass `square=True` or use `infer_square_source_bbox`. No caller needs non-square bboxes from this function.

### D4: Rename `lens_mapping_operator_bilinear_rectangular_from`

**Chosen**: Rename to `lens_mapping_operator_bilinear_from`. Drop `ny` parameter; use single `n`.
**Rationale**: "Rectangular" is misleading now that the function only operates on square grids. The shorter name still conveys that it's a bilinear interpolation operator.

### D5: Refactoring order — bottom-up

**Chosen**: Start at the utility layer (`mapping.py`), then `regularization.py`, then `pixelized_source.py`, then forward simulators, then observation models. Update tests and examples last.
**Rationale**: Higher layers import from lower layers. Bottom-up avoids broken intermediate states and allows incremental test verification.

## Risks / Trade-offs

- **[JIT recompilation]** All functions with changed static argnames (`nx`/`ny` → `n`) will recompile on first use. → One-time cost; no runtime impact after compilation.
- **[Numerical drift in regularization]** Merging `scale_x`/`scale_y` mathematically produces identical results when `dx == dy`, but float32 accumulation order may change slightly. → Run full test suite; `rtol=1e-4, atol=1e-4` thresholds are already generous enough to absorb order-of-addition differences.
- **[Example breakage]** ~15 example scripts under `examples/pix_src_demo*/` use the old API. → Mechanical search-and-replace; patterns are highly regular (`nx=X, ny=X` → `n=X`).
- **[Block-diagonal legacy path]** The legacy Python-loop preconditioner path (`_build_block_diag_precond_legacy`) iterates `n_bx` × `n_by`; with `nx == ny` this simplifies to a single loop. → Keep both scan and legacy paths initially; simplify only the scan path (the legacy path is rarely hit and exists for non-uniform blocks).
