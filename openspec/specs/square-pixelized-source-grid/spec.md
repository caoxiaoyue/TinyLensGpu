## Purpose

Define the square-grid contract for pixelized source reconstruction. The entire
pixelized-source stack — source model, mapping utilities, regularization
builders, forward simulators, and observation models — uses a single grid
dimension ``n`` (``n x n`` square grid).  All rectangular-grid support has been
removed.

## Requirements

### Requirement: Single grid-dimension parameter for pixelized source model

`PixelizedSourceModel` SHALL accept a single positional integer `n` specifying both source-grid dimensions. The model SHALL expose `self.n` as the grid dimension. The previous `nx` and `ny` parameters SHALL be removed. Construction SHALL fail when ``n < 2``.

#### Scenario: Square source model constructed with n

- **WHEN** a caller constructs `PixelizedSourceModel(n=40)`
- **THEN** construction SHALL succeed
- **THEN** the model SHALL expose `n == 40`

#### Scenario: Invalid n is rejected

- **WHEN** a caller constructs `PixelizedSourceModel(n=0)` or `PixelizedSourceModel(n=-5)`
- **THEN** construction SHALL fail with a clear error

#### Scenario: Non-integer n is coerced

- **WHEN** a caller constructs `PixelizedSourceModel(n=40.7)`
- **THEN** construction SHALL succeed with `n == 40`

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

`infer_source_bbox` SHALL always return square bounds without requiring a `square` parameter. The separate `infer_square_source_bbox` function SHALL be removed.

#### Scenario: build_source_grid produces square grid

- **WHEN** a caller invokes `build_source_grid(n=5, xmin=-1.0, xmax=1.0, ymin=-2.0, ymax=2.0)`
- **THEN** the returned x_axis SHALL have length 5 spanning [-1.0, 1.0]
- **THEN** the returned y_axis SHALL have length 5 spanning [-2.0, 2.0]
- **THEN** the returned mesh SHALL have shape (5, 5)

#### Scenario: lens_mapping_operator_bilinear_from produces valid weights

- **WHEN** `lens_mapping_operator_bilinear_from(data_mesh_beta, x_min, x_max, y_min, y_max, n=5)` is called
- **THEN** it SHALL produce bilinear interpolation weights and indices identical to the old `lens_mapping_operator_bilinear_rectangular_from` with `nx=ny=5`

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

### Requirement: Dense and operator pixelized likelihoods use square bboxes

The dense and operator pixelized image probability models SHALL use square source-plane bboxes whenever they infer source grids internally. They SHALL call `infer_source_bbox` (always square) instead of the removed `infer_square_source_bbox`.

This applies to source-grid construction for dense design matrices, matrix-free operator precomputation, source reconstruction returned by `forward_model(return_source=True)`, and evidence evaluation. The implementation SHALL keep bbox arithmetic JAX-compatible and SHALL NOT add per-pixel work beyond the existing mapping and regularization operations.

#### Scenario: Dense likelihood inferred bbox is square

- **WHEN** `PixelizedImageProbModel` infers a source bbox from seed-ray betas with unequal x/y extents
- **THEN** the bbox used for design-matrix and regularization construction SHALL be square

#### Scenario: Operator likelihood inferred bbox is square

- **WHEN** `PixelizedImageProbModelOperator` infers a source bbox from seed-ray betas with unequal x/y extents
- **THEN** the bbox used for operator precomputation, regularization data, and preconditioner construction SHALL be square

### Requirement: Fixed operator source bboxes are square

The operator pixelized image probability model SHALL accept fixed source bboxes only when they are square.

`fixed_source_bbox` SHALL remain a 4-tuple `(xmin, xmax, ymin, ymax)` with finite values and positive spans. In addition, the x and y spans MUST be equal within a numeric tolerance appropriate for `float32` bbox values. Non-square fixed bboxes SHALL fail before JIT tracing or likelihood evaluation.

#### Scenario: Square fixed source bbox is accepted

- **WHEN** `PixelizedImageProbModelOperator` is constructed with `fixed_source_bbox=(-0.5, 0.5, -0.5, 0.5)`
- **THEN** construction SHALL accept the fixed bbox

#### Scenario: Rectangular fixed source bbox is rejected

- **WHEN** `PixelizedImageProbModelOperator` is constructed with `fixed_source_bbox=(-1.0, 1.0, -0.5, 0.5)`
- **THEN** construction SHALL fail with a clear error explaining that fixed pixelized source bboxes must be square

### Requirement: S0 source-template packages use square grid metadata

Pixelized source demo pipelines that persist or consume S0 source-template packages SHALL validate that the package grid shape and bbox are square.

An S0 package SHALL contain source pixels compatible with `(N * N,)`, grid metadata where `nx == ny == N`, and a square `source_bbox`. Cached or loaded S0 packages with rectangular shape or bbox metadata SHALL be rejected before being used to build adaptive regularization scale maps or fixed source-grid likelihoods.

#### Scenario: Square S0 package is reusable

- **WHEN** a saved S0 package has `nx == ny`, `source_pixels.shape == (nx * ny,)`, and a square `source_bbox`
- **THEN** the pipeline SHALL allow the package to be reused

#### Scenario: Rectangular S0 package is rejected

- **WHEN** a saved S0 package has `nx != ny` or a non-square `source_bbox`
- **THEN** the pipeline SHALL fail with a clear validation error requiring regeneration under square source-grid rules
