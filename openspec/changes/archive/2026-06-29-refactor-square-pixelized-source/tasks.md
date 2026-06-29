## 1. Mapping utilities (`mapping.py`)

- [x] 1.1 Rename `lens_mapping_operator_bilinear_rectangular_from` to `lens_mapping_operator_bilinear_from`, replace `(nx, ny)` with `(n,)`, update internal logic
- [x] 1.2 `build_source_grid(nx, ny, ...)` → `build_source_grid(n, ...)`, update docstring from "rectangular" to "square"
- [x] 1.3 `infer_source_bbox` — remove `square` parameter, always apply square expansion; delete `infer_square_source_bbox`
- [x] 1.4 `build_lens_mapping_matrix` — update call to `lens_mapping_operator_bilinear_from`
- [x] 1.5 Update `__all__` in `mapping.py` — remove `infer_square_source_bbox`, update `lens_mapping_operator_bilinear_rectangular_from` → `lens_mapping_operator_bilinear_from`
- [x] 1.6 Update `utils/lensing/__init__.py` and `utils/__init__.py` exports

## 2. Pixelized source model (`pixelized_source.py`)

- [x] 2.1 `PixelizedSourceModel(nx, ny, ...)` → `PixelizedSourceModel(n, ...)`, remove `nx != ny` check, store `self.n`
- [x] 2.2 Update `light()` method to use `self.n` with `build_source_grid`
- [x] 2.3 Update docstring, remove references to `nx`/`ny`

## 3. Regularization (`regularization.py`) ⭐

- [x] 3.1 `DenseRegularizationBuilder(nx, ny, ...)` → `DenseRegularizationBuilder(n, ...)`, store `self.n`, update `n_pixels = n * n`
- [x] 3.2 Merge `_get_scales` to return single scalar `scale_factor`; update all callers
- [x] 3.3 `RegData(scale, scale_x, scale_y)` → `RegData(scale, scale_factor)`; remove `scale_x`/`scale_y` fields
- [x] 3.4 Simplify `_weighted_first_order_matvec` — `scale_x*out_x + scale_y*out_y` → `scale_factor * (out_x + out_y)`
- [x] 3.5 Simplify `_weighted_second_order_matvec` — same merge
- [x] 3.6 Simplify `_weighted_first_order_dense` — same merge
- [x] 3.7 Simplify `_weighted_second_order_dense` — same merge
- [x] 3.8 Simplify `_weighted_first_order_block` — use single `scale_factor`
- [x] 3.9 Simplify `_weighted_second_order_block` — use single `scale_factor`
- [x] 3.10 Simplify `_weighted_first_order_block_vec` — use single `scale_factor`
- [x] 3.11 Simplify `_weighted_second_order_block_vec` — use single `scale_factor`
- [x] 3.12 Simplify `diag_R` — merge `scale_x*diag_x + scale_y*diag_y` → `scale_factor * (diag_x + diag_y)`
- [x] 3.13 `make_reg_data()` — return `RegData(scale=scale, scale_factor=scl)`
- [x] 3.14 `source_template_scale_map(nx, ny, ...)` → `source_template_scale_map(n, ...)`
- [x] 3.15 Update `_build_unit_coordinates`, `_gp_matrix`, `_build_first_difference_operators`, `_build_curvature_difference_operators` — `self.nx`/`self.ny` → `self.n`
- [x] 3.16 `block_diag_R` — update to use `self.n`; simplify `n_bx == n_by` logic
- [x] 3.17 Update `matrix()` docstring from "rectangular" to "square"

## 4. Dense forward simulator (`pixelized.py`)

- [x] 4.1 `self.source_nx`/`self.source_ny` → `self.source_n`; `n_source_pixels = source_n * source_n`
- [x] 4.2 Update `build_mapping_matrix`, `design_matrix`, `simulate`, `forward` — pass `self.source_n` to `build_source_grid`
- [x] 4.3 Update `__repr__` — show single `source_n`
- [x] 4.4 Update `infer_source_bbox` callers — `infer_square_source_bbox` → `infer_source_bbox`

## 5. Operator forward simulator (`pixelized_operator.py`)

- [x] 5.1 `self.source_nx`/`self.source_ny` → `self.source_n`; update `n_source_pixels`
- [x] 5.2 `_A_matvec_jit` static argnames `nx,ny` → `n`; update internal reshape to `(n, n)`
- [x] 5.3 `_weighted_first_order_matvec_jit` — `nx,ny` → `n` in signature and static argnames
- [x] 5.4 `_weighted_second_order_matvec_jit` — same
- [x] 5.5 `precompute_operator_data` — update `build_source_grid` and `lens_mapping_operator_bilinear_from` calls
- [x] 5.6 `build_block_diag_preconditioner` — simplify `is_uniform` check to `n % block_size == 0`; update block partitioning
- [x] 5.7 `_build_block_diag_precond_legacy` — update `self.source_ny` → `self.source_n`
- [x] 5.8 `_build_block_diag_precond_scan` — update to use `self.source_n`
- [x] 5.9 Update imports: `lens_mapping_operator_bilinear_rectangular_from` → `lens_mapping_operator_bilinear_from`, `infer_square_source_bbox` → `infer_source_bbox`
- [x] 5.10 Update `__repr__`

## 6. Dense observation model (`pixelized_image_model.py`)

- [x] 6.1 Update `source_nx`/`source_ny` → single `source_n` usage
- [x] 6.2 Update `DenseRegularizationBuilder` construction to single-`n` API
- [x] 6.3 Update `infer_source_bbox` calls

## 7. Operator observation model (`pixelized_image_model_operator.py`)

- [x] 7.1 `source_nx`/`source_ny` → `source_n`
- [x] 7.2 `DenseRegularizationBuilder(source_nx, source_ny, ...)` → `DenseRegularizationBuilder(source_n, ...)`
- [x] 7.3 Update `_get_reg_scale` — `source_template_scale_map` call with single `n`
- [x] 7.4 Update `_validate_fixed_reg_template` — accept single `n`
- [x] 7.5 Update `_regularization_data` / `RegData` usage — single `scale_factor`

## 8. Other library files

- [x] 8.1 `Inference/prior_passing.py` — update any `PixelizedSource` width references if they use `nx`/`ny`
- [x] 8.2 `visualizer/_plot_pix_src.py` — update if uses `nx`/`ny`

## 9. Tests

- [x] 9.1 `test_pixelized_source_model.py` — `PixelizedSourceModel(nx=X, ny=X)` → `PixelizedSourceModel(n=X)`, update assertions
- [x] 9.2 `test_pixelized_source_utils.py` — `build_source_grid(3, 4, ...)` → `build_source_grid(n=...)`; update all callers and `TestBuildSourceGridOffset` tests
- [x] 9.3 `test_pixelized_operator.py` — replace all `(nx, ny) = (x, y)` with `n = X`; update `_fixed_scale`/`_fixed_template` helpers; update `DenseRegularizationBuilder` construction
- [x] 9.4 `test_regularization.py` — remove `asymmetric_grid` fixture; update all `DenseRegularizationBuilder(nx, ny, ...)` → `DenseRegularizationBuilder(n, ...)`; convert rectangular test scenarios to square
- [x] 9.5 `test_pixelized_inversion.py` — update `build_source_grid` and model construction calls
- [x] 9.6 Run full test suite: `pytest -m "not slow"` and verify all pass

## 10. Examples

- [x] 10.1 `examples/pix_src_demo/*.py` — search and replace `nx=X, ny=X` → `n=X`
- [x] 10.2 `examples/pix_src_demo_operator/*.py` — same
- [x] 10.3 Verify at least one dense example and one operator example run without error
