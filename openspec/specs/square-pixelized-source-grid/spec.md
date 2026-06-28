## Purpose

Define the square-grid contract for public pixelized source reconstruction.
Pixelized source models use square `N x N` source arrays and square source-plane
bboxes, while lower-level rectangular utilities may remain available for
internal compatibility and tests.

## Requirements

### Requirement: Pixelized source shape is square

The system SHALL require public pixelized source models to use equal source-grid dimensions.

`PixelizedSourceModel(nx: int, ny: int, ...)` SHALL accept construction only when `int(nx) == int(ny)`. The model SHALL preserve the existing `nx` and `ny` attributes for compatibility, and both attributes SHALL contain the validated square dimension.

#### Scenario: Square source shape is accepted

- **WHEN** a caller constructs `PixelizedSourceModel(nx=40, ny=40)`
- **THEN** construction SHALL succeed
- **THEN** the model SHALL expose `nx == 40` and `ny == 40`

#### Scenario: Rectangular source shape is rejected

- **WHEN** a caller constructs `PixelizedSourceModel(nx=40, ny=50)`
- **THEN** construction SHALL fail with a clear error explaining that pixelized source grids must be square

### Requirement: Pixelized bbox inference returns square source-plane bounds

The system SHALL infer square source-plane bboxes for public pixelized source likelihood paths.

The square bbox behavior SHALL first compute finite x/y bounds using the existing outlier trimming, padding, and minimum-span rules, then expand the shorter span around its center so that `xmax - xmin == ymax - ymin`. The helper API SHALL be JAX-compatible and equivalent to one of:

```python
infer_source_bbox(beta_x, beta_y, padding=..., outlier_frac=..., square=True)
```

or:

```python
infer_square_source_bbox(beta_x, beta_y, padding=..., outlier_frac=...)
```

The returned values SHALL be scalar arrays or scalars compatible with downstream JAX source-grid construction.

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

The dense and operator pixelized image probability models SHALL use square source-plane bboxes whenever they infer source grids internally.

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

### Requirement: Rectangular helpers remain internal-compatible

The system MAY retain lower-level rectangular mapping and regularization helpers for internal tests and compatibility, but public pixelized source reconstruction paths SHALL NOT use them to expose rectangular source grids.

Existing helper APIs that accept `nx` and `ny` SHALL continue to document their accepted shapes if they remain public utility exports. Their presence SHALL NOT weaken the square-grid contract for `PixelizedSourceModel`, dense pixelized likelihoods, or operator pixelized likelihoods.

#### Scenario: Low-level rectangular regularization test remains valid

- **WHEN** a low-level test constructs a `DenseRegularizationBuilder` with `nx != ny`
- **THEN** the helper MAY continue to produce a valid matrix or operator for that rectangular shape
- **THEN** this SHALL NOT imply that `PixelizedSourceModel(nx != ny)` is accepted
