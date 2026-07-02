## 1. Matrix-Free FISTA Solver

- [x] 1.1 Add a JAX-compatible FISTA solver utility under `TinyLensGpu/utils/` with an info tuple containing iteration count, convergence metric, and convergence status.
- [x] 1.2 Implement matrix-free gradient steps using the existing prebound `A(s)` callback and RHS vector without constructing dense design or curvature matrices.
- [x] 1.3 Add a fixed-iteration matrix-free power-iteration step-size estimate with a safety factor and validation for finite positive step sizes.
- [x] 1.4 Add focused unit tests for projection, non-negative output, convergence diagnostics, and JIT compatibility on a small synthetic quadratic problem. No GPU required.

## 2. Operator Backend Integration

- [x] 2.1 Add `solver_type` validation to `PixelizedImageProbModelOperator` with supported values `"pcg"` and `"fista"` and default `"pcg"`.
- [x] 2.2 Route `_solve_source()` through PCG or FISTA based on `solver_type`, reusing `build_A_matvec()`, `build_rhs()`, precomputed operator data, and regularization data.
- [x] 2.3 Ensure `__call__()` and `forward_model(return_source=True)` use the same configured solver path and gate non-converged FISTA solves with the existing large negative evidence penalty behavior.
- [x] 2.4 Document in class and method docstrings that `solver_type="fista"` enforces hard source non-negativity while retaining the existing operator logdet approximation.

## 3. Remove Soft Non-Negativity Prior

- [x] 3.1 Remove `source_nonnegativity_sigma` from the `PixelizedImageProbModelOperator` constructor, validation helpers, stored attributes, and evidence calculation.
- [x] 3.2 Remove or rewrite tests that assert half-Gaussian soft positivity penalties, replacing them with hard FISTA non-negativity tests.
- [x] 3.3 Update examples and pipelines that pass `source_nonnegativity_sigma` to use `solver_type="fista"` when non-negative source reconstruction is required.

## 4. Correctness and Parity Tests

- [x] 4.1 Add an operator test showing `forward_model(return_source=True)` returns a source vector of shape `(n * n,)` with all pixels non-negative for `solver_type="fista"`.
- [x] 4.2 Add an operator evidence test showing valid converged FISTA solves return finite scalar evidence and invalid `solver_type` values raise `ValueError`.
- [x] 4.3 Add a small-grid parity test comparing operator FISTA against dense `PixelizedImageProbModel(..., solver_type="nnls")` using objective values and non-negativity checks. No GPU required.
- [x] 4.4 Add a test or controlled configuration that exercises non-converged FISTA gating and verifies a large negative evidence penalty.

## 5. Verification

- [x] 5.1 Run from the tests directory containing the pixelized operator tests: `source ~/anaconda3/bin/activate && conda activate tinylens_gpu && pytest test_pixelized_operator.py`.
- [x] 5.2 Run focused regularization tests if FISTA integration touches regularization data paths: `source ~/anaconda3/bin/activate && conda activate tinylens_gpu && pytest test_regularization.py`.
- [x] 5.3 Run a repository-wide fast subset if runtime permits: `source ~/anaconda3/bin/activate && conda activate tinylens_gpu && pytest -m "not slow"`.
- [x] 5.4 Record whether verification was CPU-only or GPU-backed; GPU is not required for correctness tests but should be used for any performance sanity check.

Verification device: JAX reported `[CudaDevice(id=0)]`.
