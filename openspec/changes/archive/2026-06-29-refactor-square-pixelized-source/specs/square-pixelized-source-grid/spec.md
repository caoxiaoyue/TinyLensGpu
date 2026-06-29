## ADDED Requirements

### Requirement: Single grid-dimension parameter for pixelized source model

`PixelizedSourceModel` SHALL accept a single positional integer `n` specifying both source-grid dimensions. The model SHALL expose `self.n` as the grid dimension. The previous `nx` and `ny` parameters SHALL be removed.

#### Scenario: Square source model constructed with n

- **WHEN** a caller constructs `PixelizedSourceModel(n=40)`
- **THEN** construction SHALL succeed
- **THEN** the model SHALL expose `n == 40`

#### Scenario: Invalid n is rejected

- **WHEN** a caller constructs `PixelizedSourceModel(n=0)` or `PixelizedSourceModel(n=-5)`
- **THEN** construction SHALL fail with a clear error

### Requirement: Single grid-dimension parameter for regularization builder

`DenseRegularizationBuilder` SHALL accept a single integer `n` specifying the source-grid dimension. All internal methods SHALL use a single physical `scale_factor` instead of separate `scale_x`/`scale_y`.

`RegData` SHALL carry `(scale: Array | None, scale_factor: Array)` where `scale_factor` is a scalar JAX array encoding the grid-spacing factor (`1/dx^2` for first-order, `1/dx^4` for second-order, `1.0` for zero-order).

#### Scenario: Regularization builder constructed with n

- **WHEN** a caller constructs `DenseRegularizationBuilder(n=40, regularization_type="second-order")`
- **THEN** construction SHALL succeed
- **THEN** `make_reg_data()` SHALL return `RegData` with a single `scale_factor`

#### Scenario: Regularization matvec uses single scale_factor

- **WHEN** `matvec_free(s, xmin, xmax, ymin, ymax)` is called on a square grid with square bbox
- **THEN** the result SHALL be mathematically equivalent to the dense matrix-vector product
- **THEN** `scale_factor * (out_x + out_y)` SHALL replace the old `scale_x * out_x + scale_y * out_y`

### Requirement: Mapping utilities use single grid dimension

`build_source_grid` SHALL accept a single integer `n` and produce a square `(n, n)` meshgrid.

`lens_mapping_operator_bilinear_rectangular_from` SHALL be renamed to `lens_mapping_operator_bilinear_from` and SHALL accept a single `n` parameter instead of `(nx, ny)`.

#### Scenario: build_source_grid produces square grid

- **WHEN** a caller invokes `build_source_grid(n=5, xmin=-1.0, xmax=1.0, ymin=-2.0, ymax=2.0)`
- **THEN** the returned x_axis SHALL have length 5 spanning [-1.0, 1.0]
- **THEN** the returned y_axis SHALL have length 5 spanning [-2.0, 2.0]
- **THEN** the returned mesh SHALL have shape (5, 5)

#### Scenario: lens_mapping_operator_bilinear_from produces valid weights

- **WHEN** `lens_mapping_operator_bilinear_from(data_mesh_beta, x_min, x_max, y_min, y_max, n=5)` is called
- **THEN** it SHALL produce bilinear interpolation weights and indices identical to the old `lens_mapping_operator_bilinear_rectangular_from` with `nx=ny=5`

### Requirement: Forward simulators use single source dimension

`PixelizedLensSimulator` and `PixelizedLensOperator` SHALL expose a single `source_n` attribute. Internal references to `source_nx`/`source_ny` SHALL be replaced. The number of source pixels SHALL be `source_n * source_n`.

#### Scenario: Dense simulator source_n

- **WHEN** `PixelizedLensSimulator` is constructed with a `PixelizedSourceModel(n=40)`
- **THEN** `sim.source_n` SHALL be 40
- **THEN** `sim.n_source_pixels` SHALL be 1600

#### Scenario: Operator simulator source_n

- **WHEN** `PixelizedLensOperator` is constructed with a `PixelizedSourceModel(n=40)`
- **THEN** `sim.source_n` SHALL be 40
- **THEN** `sim.n_source_pixels` SHALL be 1600

### Requirement: Observation models use single source dimension

`PixelizedImageProbModel` and `PixelizedImageProbModelOperator` SHALL derive a single `source_n` from their source model. `DenseRegularizationBuilder` construction SHALL use `DenseRegularizationBuilder(n)`.

#### Scenario: Operator prob model with source_n

- **WHEN** `PixelizedImageProbModelOperator` is constructed with a `PixelizedSourceModel(n=30)`
- **THEN** `reg_builder.n` SHALL be 30

## MODIFIED Requirements

### Requirement: Pixelized source shape is square

The system SHALL require pixelized source models to use a single grid dimension `n` specifying a square `n x n` source grid.

`PixelizedSourceModel(n: int, ...)` SHALL accept a single integer `n` and expose `self.n`. The previous `nx` and `ny` parameters SHALL NOT exist.

#### Scenario: Square source shape is accepted

- **WHEN** a caller constructs `PixelizedSourceModel(n=40)`
- **THEN** construction SHALL succeed
- **THEN** the model SHALL expose `n == 40`

#### Scenario: Non-integer n is coerced

- **WHEN** a caller constructs `PixelizedSourceModel(n=40.7)`
- **THEN** construction SHALL succeed with `n == 40`

### Requirement: Pixelized bbox inference returns square source-plane bounds

The system SHALL infer square source-plane bboxes for all pixelized source likelihood paths. `infer_source_bbox` SHALL always return square bounds without requiring a `square` parameter. The helper API SHALL be:

```python
infer_source_bbox(beta_x, beta_y, padding=0.0, outlier_frac=0.01)
```

The separate `infer_square_source_bbox` function SHALL be removed.

#### Scenario: Asymmetric beta extent expands shorter side

- **WHEN** beta coordinates span x in `[0.0, 3.0]` and y in `[-0.2, 0.2]` with zero padding and no outlier trimming
- **THEN** the inferred bbox SHALL have equal x and y spans of `3.0`
- **THEN** the y bounds SHALL remain centered on the original y center

#### Scenario: Offset beta extent keeps center

- **WHEN** beta coordinates are fully offset from the origin and have unequal x/y spans
- **THEN** the square bbox SHALL preserve each axis center while expanding only the shorter span

#### Scenario: Point-like beta extent remains non-degenerate

- **WHEN** all beta coordinates collapse to a single point
- **THEN** the inferred square bbox SHALL have positive equal x/y spans

### Requirement: Dense and operator pixelized likelihoods use square bboxes

The dense and operator pixelized image probability models SHALL use square source-plane bboxes whenever they infer source grids internally. They SHALL call `infer_source_bbox` (always square) instead of the removed `infer_square_source_bbox`.

This applies to source-grid construction for dense design matrices, matrix-free operator precomputation, source reconstruction returned by `forward_model(return_source=True)`, and evidence evaluation. The implementation SHALL keep bbox arithmetic JAX-compatible and SHALL NOT add per-pixel work beyond the existing mapping and regularization operations.

#### Scenario: Dense likelihood inferred bbox is square

- **WHEN** `PixelizedImageProbModel` infers a source bbox from seed-ray betas with unequal x/y extents
- **THEN** the bbox used for design-matrix and regularization construction SHALL be square

#### Scenario: Operator likelihood inferred bbox is square

- **WHEN** `PixelizedImageProbModelOperator` infers a source bbox from seed-ray betas with unequal x/y extents
- **THEN** the bbox used for operator precomputation, regularization data, and preconditioner construction SHALL be square

## REMOVED Requirements

### Requirement: Rectangular helpers remain internal-compatible

**Reason**: The codebase no longer retains rectangular grid support at any level. All `nx`/`ny` parameters have been collapsed to single `n`, `scale_x`/`scale_y` merged to `scale_factor`, and `_rectangular_` function names renamed. This requirement's allowance of rectangular internals is obsolete.

**Migration**: Tests that exercised rectangular paths (e.g., `DenseRegularizationBuilder` with `nx != ny`) must be updated to use square `n`. Low-level rectangular test fixtures (`asymmetric_grid`) must be removed or converted to square variants.
