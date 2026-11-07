# 04: Fixed-strength operator L-BFGS-B fit

**What to build:** A modeler can select the operator backend on the same prepared L-BFGS-B fitter and obtain a qualified fixed-strength reconstruction using the shared nonlinear model, matrix-free curvature, and bounded principal blocks.

**Blocked by:** 03 — J0946 dense refit against trial 497.

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] `fit` accepts `backend="operator"` with the same strength triple, start, whitening reference, physical objective, and JAX gradient as the dense path.
- [ ] The operator applies the complete coupled GN curvature through Jacobian, transpose, noise, and prior actions. It uses bounded principal blocks for reference whitening and final preconditioning and determinants.
- [ ] The default partition uses separate contiguous source and potential blocks with a cap of 64. Explicit joint partitions follow existing coverage and size rules. Expose and record the agreed partition, PCG, and solve controls.
- [ ] The returned fit uses the same native-exit, finite-value, and GN stationarity rules as dense. Coupled PCG computes the final undamped inverse action and is accepted using a recomputed true residual. The optimizer diagnostics include that certificate and the result exposes evidence only for a qualified fit.
- [ ] The final prior and posterior principal-block log determinants supply the operator's local GN evidence, with distinct method metadata. The operator reports the same physical result fields as dense.
- [ ] Independent small-matrix tests with nonzero coupling verify the curvature action, block whitening, coupled residual certificate and stationarity, and explicit principal-block log determinants. Include candidate-strength reuse and a failure case that withholds evidence.
