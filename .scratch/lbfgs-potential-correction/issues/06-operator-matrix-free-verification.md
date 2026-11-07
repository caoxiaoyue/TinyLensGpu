# 06: Verify bounded memory for operator fits

**What to build:** A researcher can exercise the public operator fit at a representative larger size, verify its matrix-free allocation contract, and use the same measurement approach to report dense and operator costs during real-data validation.

**Blocked by:** 04 — Fixed-strength operator L-BFGS-B fit.

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] A benchmark exercises the public operator fit on regenerable inputs large enough to distinguish bounded blocks from full response, prior, joint curvature, and data-space covariance arrays.
- [ ] The matrix-free check covers whitening preparation, fit execution, endpoint certification, and evidence. It demonstrates bounded block work in place of the full matrix allocations and reports violations.
- [ ] The measurement entry point records synchronized stage timings and peak device memory with the device, precision, configuration, and measurement method. It can be applied to both backends during ticket 07. Report compilation and warm execution separately where both are measured.
- [ ] Resource checks run in the benchmark suite with a documented invocation. Structural matrix-free behavior is the acceptance gate. Timings and peak memory are reported measurements.
