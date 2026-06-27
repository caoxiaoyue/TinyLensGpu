## 1. PixelizedSourceModel — new parameters

- [x] 1.1 Add `adaptive_reg_mode` (`"brightness_only"` | `"brightness_weighted"`), `adaptive_reg_smooth_sigma` (default 1.0), and `adaptive_reg_freeze` (default False) to `PixelizedSourceModel.__init__`
- [x] 1.2 Validate `adaptive_reg_mode` against allowed set; validate `adaptive_reg_smooth_sigma > 0`
- [x] 1.3 `alpha=0` path unchanged — no new logic triggered when adaptive reg is off

## 2. DenseRegularizationBuilder — smoothing and scale utilities

- [x] 2.1 Add `smooth_scale_map` method with configurable `sigma` and auto-adaptive kernel size (`ksize = max(5, 2*ceil(3*sigma) + 1)`)
- [x] 2.2 Add `_compute_scale_formula` static method implementing `scale_i = floor + (1 - floor) / (1 + alpha * b_norm_i)`
- [x] 2.3 Add `_normalize_brightness` static method: `b_norm = b_smooth / max(mean(b_smooth), eps)`, global-mean normalization (Option B)
- [x] 2.4 Verify `make_reg_data`, `matvec_free`, `logdet_free`, `block_diag_R`, `to_dense_free` already accept `scale` — no API changes needed

## 3. Dense evidence model — `_compute_reg_scale_from_betas` refactor

- [x] 3.1 Extract inv-var from `self.noise_map` at seed pixels: `inv_var = 1 / noise_map.ravel()[seed_flat_indices]²`
- [x] 3.2 Implement `brightness_only` path: `N = segsum(brightness * inv_var * weights)`, `C = segsum(inv_var * weights)`, `b_raw = smooth(N) / (smooth(C) + eps)`
- [x] 3.3 Implement `brightness_weighted` path: `q = segsum(brightness * inv_var * weights)`, `b_raw = smooth(q)`
- [x] 3.4 Route `b_raw` through shared normalization (`_normalize_brightness`) → shared scale formula (`_compute_scale_formula`)
- [x] 3.5 Extract core computation into `_compute_scale_core` (no freeze/alpha-0 checks) shared by the JIT path and the eager freeze path
- [x] 3.6 Implement `freeze_scale()` / `unfreeze_scale()` eager API: compute scale once outside JIT, store as `self._frozen_scale`; `_compute_reg_scale_from_betas` returns the cached array at trace time so JIT captures it as a constant
- [x] 3.7 Warn when `freeze=True` but no scale has been stored (user forgot `freeze_scale()`)
- [x] 3.8 Keep `alpha=0` fast-path: return `None` immediately

## 4. Operator evidence model — same refactor

- [x] 4.1 Mirror all changes from section 3 into `PixelizedImageProbModelOperator._compute_reg_scale_from_betas`
- [x] 4.2 Ensure `_get_bbox()` already returns `beta_x_seed, beta_y_seed` needed by the new brightness estimator
- [x] 4.3 Verify `_regularization_data` passes `scale` through to `RegData` (already supported, confirm no changes needed)
- [x] 4.4 Implement `_compute_scale_core`, `freeze_scale()` / `unfreeze_scale()` identically to dense model
- [x] 4.5 Fix return type annotation `-> Array` → `-> Array | None` (alpha=0 and freeze paths can return `None`)

## 5. Tests

- [x] 5.1 Unit test: `brightness_only` mode produces `scale ≈ 1` when all image brightness is uniform and magnification varies (verify magnification cancellation)
- [x] 5.2 Unit test: `brightness_weighted` mode produces lower scale in higher-magnification regions for identical brightness (verify magnification preserved)
- [x] 5.3 Unit test: inv-var weighting — a pixel with σ=100 contributes 1e-4× the weight of a pixel with σ=1
- [x] 5.4 Unit test: global-mean normalization — sparse bright coverage produces `b_norm ≫ 1` for bright pixels
- [x] 5.5 Unit test: scale formula endpoints (`b_norm=0 → scale=1`, `b_norm→∞ → scale→floor`)
- [x] 5.6 Unit test: `alpha=0` returns `None` without computing brightness
- [x] 5.7 Unit test: `smooth_sigma` changes kernel size correctly
- [x] 5.8 Unit test: `freeze` stores and returns cached scale on subsequent calls
- [x] 5.9 Integration test: `PixelizedImageProbModel` with `brightness_only` mode produces finite log-evidence
- [x] 5.10 Integration test: `PixelizedImageProbModelOperator` with `brightness_only` mode produces finite log-evidence via PCG

## 6. Example and documentation

- [x] 6.1 Update `examples/pix_src_demo_operator/pipe/no_lens_light/model_adpt_reg.py` to use new parameter names (`adaptive_reg_mode`, `adaptive_reg_smooth_sigma`)
- [ ] 6.2 Verify example runs end-to-end with default `brightness_only` mode (requires GPU)
