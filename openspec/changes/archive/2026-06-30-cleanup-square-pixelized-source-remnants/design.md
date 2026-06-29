## Context

The previous square-grid refactor moved the pixelized-source core to a single `n` dimension across `PhysicalModel`, `ForwardSimulation`, `ObservationModel`, and regularization utilities. The remaining drift is outside the hot JAX likelihood path: demo S0 package metadata, source-plane plotting reshapes, validation tests, and public-facing docstrings still use `nx`/`ny` language for source grids.

The most important remaining data model is the S0 source-template package used by the operator pixelized demo pipeline for fixed-bbox adaptive regularization. It is persisted and fingerprinted outside Caskade/JAX tracing, then passed into `PixelizedImageProbModelOperator` as `fixed_source_bbox` and `fixed_reg_template`.

## Goals / Non-Goals

**Goals:**

- Make source-plane persisted metadata use exactly one dimension field: `n`.
- Reject stale S0 packages that only contain legacy `nx`/`ny` metadata.
- Keep image-plane array code free to use standard `ny, nx = image.shape` naming.
- Replace source-plane example reshapes with `reshape(n, n)`.
- Update docstrings and tests so the codebase presents one square-grid model.

**Non-Goals:**

- No changes to Caskade module structure or dynamic parameter handling.
- No changes to JAX-traced interpolation, regularization, evidence, or PCG code.
- No migration loader for old S0 package files.
- No cleanup of image-plane `ny/nx` variables.

## Decisions

### D1: S0 package schema uses `n`, not `nx`/`ny`

S0 packages will write and validate:

```python
{
    "n": NSRC,
    "source_pixels": array(shape=(n * n,)),
    "source_image": array(shape=(n, n)),
    "source_bbox": (xmin, xmax, ymin, ymax),
    "source_x_axis": array(shape=(n,)),
    "source_y_axis": array(shape=(n,)),
}
```

Rationale: this mirrors `PixelizedSourceModel(n)` and removes the last persisted representation that can express rectangular source grids.

Alternative rejected: accept old `nx`/`ny` and normalize to `n`. That preserves stale cache files but leaves compatibility code whose only purpose is to support an invalid source-grid model.

### D2: Reject legacy S0 packages early

Validation will require `n` and fail before building adaptive scale maps or likelihood fixed kwargs when `n` is missing. If legacy `nx`/`ny` keys are present without `n`, the error should explicitly ask the user to regenerate the S0 package under the single-`n` schema.

Rationale: stale source templates are cache artifacts, not user-authored scientific inputs. Recomputing them is cheaper and clearer than carrying a long-lived migration branch.

### D3: Keep image-plane `ny/nx`

Variables derived from image arrays, masks, or FITS data may continue to use `ny, nx`. Only source-plane grid variables should be renamed to `n` or `source_n`.

Rationale: image arrays are naturally rectangular in NumPy conventions. Forcing `n` there would hide a useful distinction and could imply false square-image constraints.

### D4: No JAX path changes

All S0 package validation and example reshaping runs in Python/NumPy code before JIT tracing. `PixelizedImageProbModelOperator` already receives flat templates and square bboxes, so the runtime likelihood path should be unchanged.

Rationale: this change is maintainability-focused and should not affect GPU memory use, compilation cache keys, or numerical behavior.

## Risks / Trade-offs

- **[Stale cache breakage]** Existing S0 package files with only `nx`/`ny` will fail. → Error messages will tell users to regenerate S0 under the single-`n` schema.
- **[Overbroad rename]** Mechanical replacement could rename image-plane `ny/nx` variables. → Scope searches and edits to source-plane package, plotting, and docstring contexts.
- **[Test drift]** Existing tests assert rectangular S0 grid rejection via `nx != ny`. → Replace with stale legacy schema rejection plus invalid source vector and rectangular bbox tests.
- **[Hidden examples]** Some example scripts may still reshape with `ny, nx` after targeted edits. → Run source-focused `rg` checks for source-plane `nx = ...source_light[0].n` and `reshape(ny, nx)`.
