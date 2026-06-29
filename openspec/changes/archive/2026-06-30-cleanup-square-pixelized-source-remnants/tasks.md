## 1. S0 Package Schema

- [x] 1.1 Update `examples/pix_src_demo_operator/pipe/no_lens_light/model_adpt_reg.py` so S0 package creation writes `n=NSRC` instead of `nx=NSRC, ny=NSRC`
- [x] 1.2 Update S0 package validation to require `n`, validate `source_pixels.shape == (n * n,)`, validate source axes shape `(n,)`, and reject legacy packages that only provide `nx`/`ny`
- [x] 1.3 Update S0 adaptive scale-map construction and fingerprinting to use `n` only
- [x] 1.4 Verify S0 fixed kwargs still pass `fixed_source_bbox` and flat `fixed_reg_template` into `PixelizedImageProbModelOperator`

## 2. Source-Plane Naming Cleanup

- [x] 2.1 Replace source-plane example patterns like `nx = source.n; ny = source.n; reshape(ny, nx)` with `n = source.n; reshape(n, n)`
- [x] 2.2 Keep image-plane `ny, nx = image.shape` variables unchanged unless they are incorrectly representing source-plane grids
- [x] 2.3 Update source-plane docstrings and error text from `Nx*Ny`, `nx * ny`, or `(nx * ny,)` to `(n * n,)`
- [x] 2.4 Run targeted `rg` checks for source-plane `nx/ny` remnants and review remaining matches as intentional image-plane usage or tests

## 3. Tests

- [x] 3.1 Update `tests/test_adaptive_demo_s0_package.py` fixtures to build S0 packages with `n`
- [x] 3.2 Replace the rectangular `nx != ny` package test with a stale legacy `nx`/`ny` package rejection test
- [x] 3.3 Add or update tests for invalid `(n * n,)` source vector shape and rectangular `source_bbox` rejection
- [x] 3.4 Run `pytest tests/test_adaptive_demo_s0_package.py` from the `tests/` directory using the `tinylens_gpu` environment

## 4. Verification

- [x] 4.1 Run focused pixelized source tests that cover fixed templates and operator adaptive regularization paths
- [x] 4.2 Run `pytest -m "not slow"` using the `tinylens_gpu` environment
- [x] 4.3 Run `openspec status --change "cleanup-square-pixelized-source-remnants"` and confirm all artifacts remain coherent
