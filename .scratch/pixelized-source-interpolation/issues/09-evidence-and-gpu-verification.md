# 09: Measure evidence approximation and GPU costs

**What to build:** A modeler evaluating opt-in Hermite can inspect measured operator evidence errors on controlled problems and the resource cost of a representative GPU solve. The report distinguishes fixed-parameter reconstruction accuracy from the evidence approximation.

**Blocked by:** 02 — Reconstruct the same Hermite source with the operator backend, and 06 — Run single-step and iterative potential correction with Hermite.

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] For ordinary and potential-correction operator fits, use small problems with available dense references and multiple explicit block configurations. Independently verify the applicable block determinant and report evidence differences for the same interpolation and strengths.
- [x] Report fixed-parameter objectives, reconstruction differences, residuals or KKT checks, and solve failures separately from the evidence differences. Identify the approximation and block configuration in each result.
- [x] Run bilinear and Hermite in the same representative GPU setup with a `100 × 100` source grid and `nsub=4`. Verify that preparation and solve complete within the original-node and matrix-free assembly contracts.
- [x] Record mapping preparation, preconditioner construction, peak device memory, and synchronized solve time, distinguishing compilation from warm execution. Record inputs, precision, tolerances, commands, and the device used.
- [x] Run the relevant core, example, and benchmark checks in separate processes in the documented environment and retain the results with their configurations and limits.

## Comments

Claimed for implementation on `codex/psi-09-verification`.

Implemented in `114a2d0`, merged as `3319ed3`. The
[validation report](../../../docs/validation/pixelized-source-interpolation.md)
records the separate core, example and benchmark commands (22 CPU core,
6 CUDA evidence, 4 CPU example and 3 CUDA benchmark checks passed). Both
fresh 100 × 100, `nsub=4` RTX 5080 runs converged in 12 PCG iterations.
Independent block log determinants agreed within `8.53e-14` across the
reported controlled comparisons. [Raw evidence and GPU records](../experiments/).

The six-case CUDA evidence command ran from `tests/` in `tinylens_gpu`.

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 \
  pytest -xq test_pixelized_interpolation_evidence.py
```
