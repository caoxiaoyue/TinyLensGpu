# 02: Dense mock fitting and result round trip

**What to build:** A modeler can run the dense L-BFGS-B fitter on a regenerated subhalo injection and matched no-subhalo control and inspect both reconstructions through the common result export and readback workflow.

**Blocked by:** 01 — Fixed-strength dense L-BFGS-B fit.

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] A maintained example fits the existing Pseudo-Jaffe injection and no-subhalo control at recorded strengths using the same noise, mask, and observation setup for the pair.
- [ ] Both dense fits qualify. The example records convergence recovery on the valid domain and nonlinear image residuals for the injection and control, with the configuration needed to regenerate the pair.
- [ ] The common exporter and reader round-trip the fitted fields, valid-domain metadata, strengths, evidence method, and `optimizer_info` with an empty `solver_info`. Numerical failure exports the available reason and diagnostics.
- [ ] Existing single-step and GN/LM products still load through the shared reader, and their export regressions pass.
- [ ] Example-suite checks exercise the matched run and persistence with regenerable inputs in a process separate from the library tests. The example configuration is reusable for ticket 05.
