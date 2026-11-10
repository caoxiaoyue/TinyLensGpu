# 02: Fit Hermite pixelized sources with squared-weight adaptation

**What to build:** A modeler can select cubic Hermite interpolation on either backend and use the same adapt image interface. Each proposal uses the signed Hermite stencil for image prediction and squared per-ray weights for the adaptive projection.

**Blocked by:** [01 — Fit bilinear pixelized sources with an adapt image](01-bilinear-adapt-image.md).

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] An independent reference with more than one subpixel ray per native pixel verifies the square-before-aggregation rule, native averaging factor and Hermite support. The forward prediction uses the signed physical stencil at the same proposal.
- [x] Use an internal coverage offset starting at 10⁻⁴ and the fixed reference normalization from ticket 01. Zero and weak coverage produce finite scales, and diagnostics record the scale used by each backend.
- [x] Matched dense and operator fits agree on the projected scale and reconstruction in a controlled Hermite case. Fixed and changing mass cases retain the specified cache behavior and preserve the configured numerical precision.
- [x] A Float64 conditional-prior derivative check across a controlled ray crossing verifies the continuous Hermite projection. Include a weak-coverage probe and retain the tested parameters and tolerances.

## Comments

- Claimed on `codex/overlap-02-hermite`.
- Implemented in `44077fc`. From `tests/` in the `tinylens_gpu` environment, CUDA `pytest -q test_adapt_image.py` passed the new Hermite cases in isolated runs, including the independent reconstruction comparison. The other 13 projection, validation, and cache cases also passed. The Float64 conditional-prior check used shifts `(-0.15002, -0.15, -0.14998)` across the crossing, `x=-0.33` for weak coverage, `nsub=2`, and a finite-difference step of `1e-6`. The gradient tolerances were `rtol=2e-5` and `atol=1e-7`.
