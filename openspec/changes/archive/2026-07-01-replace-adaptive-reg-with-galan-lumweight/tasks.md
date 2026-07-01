## 1. Core API

- [x] 1.1 Update `PixelizedSourceModel` to accept `adaptive_reg_rho` as a scalar or `ParamU`, validate `rho >= 0`, and expose dynamic `rho` through Caskade traversal.
- [x] 1.2 Remove or clearly reject active use of `adaptive_reg_alpha` and `adaptive_reg_floor` in the source-template adaptive path.
- [x] 1.3 Update docstrings and error messages in `PixelizedSourceModel`, dense `PixelizedImageProbModel`, and operator `PixelizedImageProbModelOperator` to refer to `adaptive_reg_rho`.

## 2. Scale Map Formula

- [x] 2.1 Replace `source_template_scale_map()` arguments with `rho`, `ref_percentile=99.5`, and `eps`.
- [x] 2.2 Implement Galan-style precision scale construction: clip negative S0 values, compute the 99.5 percentile reference, clip normalized brightness to `[0, 1]`, and return `exp(rho * (1 - u))`.
- [x] 2.3 Preserve the static `rho == 0` uniform fast path and traced-rho all-ones behavior without Python branching on traced values.
- [x] 2.4 Keep output validation compatible with existing regularization builders: flat shape `(n * n,)`, finite values, and strictly positive scale.

## 3. ObservationModel Integration

- [x] 3.1 Update `PixelizedImageProbModelOperator._adaptive_reg_enabled()` to detect scalar or dynamic `adaptive_reg_rho`.
- [x] 3.2 Update `PixelizedImageProbModelOperator._get_reg_scale()` to call `source_template_scale_map(..., rho=...)` for `fixed_reg_template`.
- [x] 3.3 Preserve `fixed_reg_scale` compatibility for externally supplied positive scale maps.
- [x] 3.4 Confirm finite-difference matrix and matrix-free regularization assembly need no structural changes for scale values in `[1, exp(rho)]`.

## 4. Examples And Pipeline Metadata

- [x] 4.1 Update `examples/pix_src_demo_operator/pipe/no_lens_light/model_adpt_reg.py` to sample and store `adaptive_reg_rho` instead of `adaptive_reg_alpha` and `adaptive_reg_floor`.
- [x] 4.2 Update stage-m1 posterior medians, cache validation, plot/replot paths, and stage-m2 fixed hyperparameter handoff to use `adaptive_reg_rho`.
- [x] 4.3 Update any other examples found by `rg "adaptive_reg_alpha|adaptive_reg_floor"` when they construct adaptive source-template likelihoods.
- [x] 4.4 Make stale cache handling fail clearly when old alpha/floor metadata is present without `adaptive_reg_rho`.

## 5. Tests

- [x] 5.1 Update `tests/test_regularization.py` source-template scale-map tests for 99.5-percentile normalization, bright-pixel `scale=1`, dark-pixel `scale=exp(rho)`, negative clipping, invalid shapes, and traced-rho behavior.
- [x] 5.2 Update `tests/test_pixelized_source_model.py` for scalar/dynamic `adaptive_reg_rho`, prior extraction names, and negative-rho rejection.
- [x] 5.3 Update `tests/test_pixelized_operator.py` helpers and adaptive operator tests to construct sources with `adaptive_reg_rho` and fixed S0 template or scale inputs.
- [x] 5.4 Add or update a regularization edge-weight test showing adjacent bright/dark pixels produce edge weight `exp(rho / 2)`.

## 6. Verification

- [x] 6.1 Run from the tests directory: `source ~/anaconda3/bin/activate && conda activate tinylens_gpu && pytest test_regularization.py`.
- [x] 6.2 Run from the tests directory: `source ~/anaconda3/bin/activate && conda activate tinylens_gpu && pytest test_pixelized_source_model.py test_pixelized_operator.py`.
- [x] 6.3 Run the fast suite if local runtime permits: `source ~/anaconda3/bin/activate && conda activate tinylens_gpu && pytest -m "not slow"`.
- [x] 6.4 Record any GPU-specific test limitations; the listed unit/operator tests should not require a dedicated GPU, but full demo runs may benefit from one.
