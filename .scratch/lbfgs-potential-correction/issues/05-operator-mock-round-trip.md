# 05: Operator mock fitting and result round trip

**What to build:** A modeler can repeat the same matched mock and control fits with the operator backend and compare both backends through the maintained example and shared result products.

**Blocked by:** 04 — Fixed-strength operator L-BFGS-B fit.

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] Run the operator backend on the same regenerated injection and control from ticket 02 with the same observations, actual starts, and fixed strengths. Record the operator partition and controls.
- [ ] Both operator fits qualify. Report their objectives, image residuals, and convergence fields on matching valid domains alongside the dense results. Record each evidence value with its determinant method.
- [ ] Export and reload the operator products with the physical fields, partition metadata, principal-block determinant method, and terminal stationarity and residual diagnostics. The dense products remain readable through the same interface.
- [ ] Example regressions run the comparison and readback in a separate process from core tests using regenerable data.
