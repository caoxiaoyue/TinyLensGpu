# 09: Measure evidence approximation and GPU costs

**What to build:** A modeler evaluating opt-in Hermite can inspect measured operator evidence errors on controlled problems and the resource cost of a representative GPU solve. The report distinguishes fixed-parameter reconstruction accuracy from the evidence approximation.

**Blocked by:** 02 — Reconstruct the same Hermite source with the operator backend, and 06 — Run single-step and iterative potential correction with Hermite.

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] For ordinary and potential-correction operator fits, use small problems with available dense references and multiple explicit block configurations. Independently verify the applicable block determinant and report evidence differences for the same interpolation and strengths.
- [ ] Report fixed-parameter objectives, reconstruction differences, residuals or KKT checks, and solve failures separately from the evidence differences. Identify the approximation and block configuration in each result.
- [ ] Run bilinear and Hermite in the same representative GPU setup with a `100 × 100` source grid and `nsub=4`. Verify that preparation and solve complete within the original-node and matrix-free assembly contracts.
- [ ] Record mapping preparation, preconditioner construction, peak device memory, and synchronized solve time, distinguishing compilation from warm execution. Record inputs, precision, tolerances, commands, and the device used.
- [ ] Run the relevant core, example, and benchmark checks in separate processes in the documented environment and retain the results with their configurations and limits.

