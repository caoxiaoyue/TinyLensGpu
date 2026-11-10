# 12: Qualify full-score multi-start M3 L-BFGS-B

**What to build:** A researcher can qualify the complete Overlap score for bounded mass optimization in the original mock. The result identifies the Hermite coverage offset that passed the intended full-score workflow.

**Blocked by:** [11 — Reproduce the 80×80 Euclid mock with image-driven adaptation](11-euclid-mock-evidence-replay.md).

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] Run the slope scan and Float64 derivative comparisons before the optimizer qualification. Separate the conditional source prior check from the complete score and identify the position-penalty regime.
- [ ] Maximize the complete Overlap score with its adaptive source prior and configured position penalty over all free M3 mass parameters and their existing physical bounds.
- [ ] Predeclare reproducible starting points, solver settings and projected-gradient qualification criteria. Record line-search status, endpoint validity, finite scores and gradients, projected gradients and endpoint basins for each start.
- [ ] Start qualification with the internal offset of 10⁻⁴. If it changes, repeat the affected M2 calibration, M3 and matched score comparisons. Record the passing setting and supporting results.
