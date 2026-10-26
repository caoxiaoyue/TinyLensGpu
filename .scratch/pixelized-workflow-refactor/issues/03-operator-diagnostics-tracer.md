# 03: Deliver operator diagnostics through a complete example

**What to build:** An operator-backend user obtains a reconstruction, convergence-aware diagnostics, and requested image components through the same public result contract, with a working source-only example and explicit joint-diagnostic limitations.

**Blocked by:** 02: Deliver dense diagnostics through a complete example

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] Add the operator `evaluate_with_diagnostics` entry point using the common contract and one source reconstruction inversion, with optional calculations disabled by default.
- [ ] Expose the specified basic, adaptive-regularization, and enabled position diagnostics; preserve the existing source-only block-preconditioner effective-parameter approximation and identify its method.
- [ ] For joint semi-linear inversion, return the actual total and requested component images while retaining block-Schur preconditioning and unit-amplitude lens-light basis semantics.
- [ ] Return `None` with an explanatory unsupported reason for requested operator joint effective parameter count, residual degrees of freedom, and reduced chi-square; retain valid raw chi-square and image products without marking the reconstruction invalid.
- [ ] Mark explicit non-convergence invalid even when the existing core projection returns finite zero arrays, and reject nonfinite reconstruction values; preserve core evidence penalties and reconstruction failure returns.
- [ ] Migrate the basic operator source-only fitting example without position constraints to this public path and stop formal output for invalid reconstruction.
- [ ] Verify small real source-only and joint problems, supported PCG/PNPG behavior, optional calculation, components, failure status, and the single-reconstruction contract through public outputs.
- [ ] Preserve existing likelihood, batching, and differentiation behavior; keep unmigrated callers usable and record focused core and example verification separately.
