# 02: Deliver dense diagnostics through a complete example

**What to build:** A dense pixelized-source user can request reconstruction and diagnostics through one public call and produce a diagnostic plot in the basic dense source-only fitting example without managing solver internals.

**Blocked by:** 01: Restore native JAX array truthiness

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] Introduce the common diagnostic result contract and the dense `evaluate_with_diagnostics` entry point, returning the existing `InversionResult` plus diagnostics; `include_effective_dof` and `include_components` default to false.
- [ ] Support the specified basic, adaptive-regularization, and enabled position diagnostics, and requested component images, for dense source-only and joint semi-linear inversion.
- [ ] Preserve the current dense effective-parameter convention, lens-light regularization treatment, and supported edge-case handling; identify the method without adding a new constrained-solver interpretation.
- [ ] Produce paired results from one source reconstruction inversion; distinguish unrequested values from invalid reconstruction and retain dense's absence of a solver convergence status.
- [ ] Migrate the basic dense source-only fitting example without position constraints to the public diagnostic path, preserving its scientific quantities and display conventions and rejecting nonfinite reconstruction before formal output.
- [ ] Verify the public result against small real dense problems, including joint components and optional diagnostics, and enforce the single-reconstruction contract with a narrowly scoped invocation probe.
- [ ] Keep existing `evaluate`, forward-model, likelihood, batching, and differentiation contracts intact; ordinary likelihood evaluation performs no optional diagnostic work.
- [ ] Leave other callers working through their existing interfaces during expansion and record focused core and example verification separately.
