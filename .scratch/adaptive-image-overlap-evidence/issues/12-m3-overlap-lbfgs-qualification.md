# 12: Qualify full-score multi-start M3 L-BFGS-B

**What to build:** A researcher can qualify the complete Overlap score for bounded mass optimization in the original mock. The result identifies the Hermite coverage offset that passed the intended full-score workflow.

**Blocked by:** [11 — Reproduce the 80×80 Euclid mock with image-driven adaptation](11-euclid-mock-evidence-replay.md).

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Run the slope scan and Float64 derivative comparisons before the optimizer qualification. Separate the conditional source prior check from the complete score and identify the position-penalty regime.
- [x] Maximize the complete Overlap score with its adaptive source prior and configured position penalty over all free M3 mass parameters and their existing physical bounds.
- [x] Predeclare reproducible starting points, solver settings and projected-gradient qualification criteria. Record line-search status, endpoint validity, finite scores and gradients, projected gradients and endpoint basins for each start.
- [x] Start qualification with the internal offset of 10⁻⁴. If it changes, repeat the affected M2 calibration, M3 and matched score comparisons. Record the passing setting and supporting results.

## Comments

- Claimed on `codex/overlap-12-optimizer-qualification`.
- Controls were committed in `6037a02` and the runner and results in `166e1c7`. The CUDA `qualify_m3_optimizer.py probe` phase produced 19 valid slope points and passed all nine conditional-prior and full-score finite-difference checks. The subsequent Float64 `optimize` phase converged from all three L-BFGS-B starts to basin 1, with endpoint score 1509.785915959, slope 2.19226494 and scaled projected-gradient norms 7.518e-6, 2.335e-6 and 8.427e-7. The line searches rejected three invalid trials. The Hermite offset of 1e-4 qualified. See the [validation report](../../../docs/validation/adaptive-image-overlap-m3-optimizer.md) and [retained results](../experiments/m3_optimizer_results.json).
