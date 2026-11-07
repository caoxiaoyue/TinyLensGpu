# 06: Verify bounded memory for operator fits

**What to build:** A researcher can exercise the public operator fit at a representative larger size, verify its matrix-free allocation contract, and use the same measurement approach to report dense and operator costs during real-data validation.

**Blocked by:** 04 — Fixed-strength operator L-BFGS-B fit.

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] A benchmark exercises the public operator fit on regenerable inputs large enough to distinguish bounded blocks from full response, prior, joint curvature, and data-space covariance arrays.
- [x] The matrix-free check covers whitening preparation, fit execution, endpoint certification, and evidence. It demonstrates bounded block work in place of the full matrix allocations and reports violations.
- [x] The measurement entry point records synchronized stage timings and peak device memory with the device, precision, configuration, and measurement method. It can be applied to both backends during ticket 07. Report compilation and warm execution separately where both are measured.
- [x] Resource checks run in the benchmark suite with a documented invocation. Structural matrix-free behavior is the acceptance gate. Timings and peak memory are reported measurements.

## Comments

Claimed by `codex/lbfgs-06-operator-resources` for integration into `codex/lbfgs-potential-correction`.

Implemented in `8665eef` and merged in `3fe904d`. In the activated environment from `benchmarks/`, `JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 pytest -xq test_lbfgs_operator_resources.py` passed, as did the same check with `JAX_PLATFORMS=cpu`. The generated fit used 8,705 coefficients with a 32-column cap and found zero prohibited shapes across 196 compiled CUDA modules. See the [benchmark report and reproducer](../../../docs/validation/potential-correction-lbfgs-operator-resources.md) and [retained measurements](../output/operator-resource-benchmark.json).
