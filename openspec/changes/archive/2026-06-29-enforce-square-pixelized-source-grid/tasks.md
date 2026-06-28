## 1. Source Bbox Utilities

- [x] 1.1 Add a JAX-compatible square-bbox helper in `TinyLensGpu.utils.lensing.mapping` that expands the shorter span around its center and preserves finite/minimum-span behavior.
- [x] 1.2 Expose the square bbox path through either `infer_source_bbox(..., square=True)` or a dedicated `infer_square_source_bbox()` API, and update `TinyLensGpu.utils` exports if a new public helper is added.
- [x] 1.3 Add unit tests in `tests/test_pixelized_source_utils.py` for asymmetric extents, offset extents, padding/outlier behavior, and point-like inputs.

## 2. Physical Model Validation

- [x] 2.1 Update `PixelizedSourceModel` to reject `nx != ny` with a clear square-grid error while preserving `nx` and `ny` attributes for square inputs.
- [x] 2.2 Add focused tests that square `PixelizedSourceModel(nx=N, ny=N)` construction succeeds and rectangular construction fails before any likelihood is built.

## 3. Pixelized Likelihood Bbox Routing

- [x] 3.1 Route dense pixelized simulator bbox inference through the square bbox path for design-matrix and forward-model source reconstruction flows.
- [x] 3.2 Route operator pixelized simulator/probability bbox inference through the square bbox path for `_get_bbox()`, operator precomputation, regularization data, and block preconditioner construction.
- [x] 3.3 Add focused dense/operator tests that inferred bboxes used by pixelized likelihood paths are square when seed-ray beta extents are asymmetric.

## 4. Fixed S0 and Operator Validation

- [x] 4.1 Update `PixelizedImageProbModelOperator._validate_fixed_source_bbox()` to reject non-square fixed source bboxes before JIT tracing.
- [x] 4.2 Add tests that square `fixed_source_bbox` values are accepted and rectangular fixed bboxes fail with a clear error.
- [x] 4.3 Update S0 package validation in adaptive pixelized-source demo code to require `nx == ny`, `(N * N,)` source pixels, and square `source_bbox`, with focused S0 package validation tests.

## 5. Demo and Documentation Cleanup

- [x] 5.1 Replace paired pixelized-source demo constants such as `NSRCX`/`NSRCY` with a single square-grid constant where practical, keeping reshape calls compatible with `ny, nx` attributes.
- [x] 5.2 Update nearby comments/docstrings that describe rectangular source grids so public pixelized-source docs say square source grids and square bboxes are required.

## 6. Verification

- [x] 6.1 From the `tests` directory, run focused source utility and physical model tests for square bbox and `PixelizedSourceModel` validation.
- [x] 6.2 From the `tests` directory, run focused dense/operator pixelized likelihood tests covering square inferred bbox, fixed bbox validation, and S0 package validation.
- [x] 6.3 From the relevant example directory, run a syntax/import check for updated pixelized demo modules; full GPU demo execution is optional and should be noted if skipped.
- [x] 6.4 Run `openspec status --change "enforce-square-pixelized-source-grid"` and ensure the change remains apply-ready after implementation tasks are updated.
