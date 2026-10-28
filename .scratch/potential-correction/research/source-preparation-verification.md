# Ticket 01: source preparation verification

Implementation: `bdfa055` on `codex/pc-01`. This record concerns the existing
pixelized-source inversion, before source–potential reconstruction.

## Changes and numerical contract

`FiniteDifferenceRegularization` prepares dense index-space operators only
when `matrix` or `to_dense_free` is requested. Compact prior preparation,
matvecs, diagonals, and block matrices do not allocate those operators.
Compile-time constant evaluation keeps the first dense request inside JIT
compatible with later traces and differentiation.

Explicitly typed float64 source bounds survive both existing observation-model
constructors and the compact prior's physical-spacing factors, block matrices,
diagonals, and determinant paths. Plain Python bounds retain historical
float32 compact preparation. JAX precision defaults are unchanged; float64
checks use `jax.experimental.enable_x64()` locally. Dense fixed-grid cache
matching uses the stored bounds' dtype rather than unconditionally rounding
all comparisons to float32.

The source prior coefficients, source-boundary fallback, PSF, solvers,
source–lens-light Schur preconditioner, and public call patterns are unchanged.
No optimizer path was changed in this ticket.

## Focused numerical checks

Run in the `tinylens_gpu` environment, from `tests/`, with
`XLA_PYTHON_CLIENT_PREALLOCATE=false`:

```bash
pytest test_regularization.py -q
pytest test_pixelized_source_precision.py test_pixelized_inversion.py \
  -k 'precision or cache or fixed_grid or fixed_bbox' -q
```

Results: **62 passed** and **23 passed / 80 deselected**, respectively.

The independent small Gaussian reference uses 25 image pixels, a 3-by-3 source,
2-by-2 image subpixels, a normalized 3-by-3 PSF, no macro mass, noise 0.05,
second-order source regularization at `log_lambda=-3`, and explicitly typed
float64 bounds `(-0.600000000123, 0.600000000321)` on each axis. It compares both
solvers' recovered source and model image with a NumPy Gaussian solution using
independently written curvature rows, at `rtol=atol=1e-9`. PCG uses `rtol=1e-11`;
observed coefficient differences reached `2.2e-10`. Dense complete Gaussian
evidence agrees at `rtol=atol=1e-12`. Existing operator evidence retains its
established approximation; this test does not claim complete evidence for it.

For all three finite-difference source priors, explicit float64 spacing factors
agree at `rtol=1e-14`, and dense/block/diagonal prior values agree at
`rtol=atol=1e-13`, including reuse after a previous float32 dense request.
First dense use inside JIT and subsequent gradients are checked at float32
`rtol=1e-6`. Allocation, dtype, and fixed-bbox tests failed before the changes
and passed afterward.

## Observable preparation allocations

Reproduce from the repository root:

```bash
source ~/anaconda3/bin/activate && conda activate tinylens_gpu
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. \
  python .scratch/potential-correction/experiments/source_preparation_memory.py
```

Context: Python 3.11.14, JAX 0.7.1, CUDA device 0, NVIDIA GeForce RTX 5080,
16303 MiB reported total, driver 610.57.04, global x64 disabled. The script
counts new live JAX array bytes after constructing a prior and compact
`RegData`; it does not infer memory from a private storage class.

| Source side | Unknowns | Preparation live bytes, each prior | One dense float32 matrix |
| --- | --- | --- | --- |
| 16 | 256 | 4 | 262144 bytes |
| 32 | 1024 | 4 | 4194304 bytes |
| 64 | 4096 | 4 | 67108864 bytes |

All three supported priors gave the reported 4-byte compact scalar allocation.
Before the change, the maintained `n=48` allocation regression observed
21233668 bytes for zero-order and 42467332 bytes for first-/second-order.
Its allowed 294912-byte bound permits linear-size preparation but excludes even
one dense source-sized operator. This measurement is live array storage, not
allocator-pool reservation or peak workflow memory; ticket 13 owns the latter.

## Existing source and differentiation regressions

From `tests/` in the same environment:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false pytest \
  test_pixelized_source_model.py test_pixelized_source_utils.py \
  test_pixelized_inversion.py test_pixelized_operator.py \
  test_operator_autodiff.py test_solver_autodiff.py \
  test_pixelized_source_precision.py -q
```

Result: **372 passed in 257.78 seconds**. This covers existing dense/operator
reconstruction, fixed-source boundaries, PSF/subpixel paths, source–lens-light
inversion and Schur handling, basis validation, solver failures, and supported
operator/solver differentiation. The focused cache/precision command above was
also rerun after the final cache-matching adjustment. Core tests were run in a
separate process from examples; no example configuration was imported into
these checks. No PyAuto runtime was used.
