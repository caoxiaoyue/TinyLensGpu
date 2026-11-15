# 03: J0946 dense refit against trial 497

**What to build:** A researcher can supply the local J0946 archive, run a fresh dense fixed-strength fit, and compare its objective, evidence, and complete valid convergence field with trial 497. The retained refit is the dense reference for the final two-backend comparison.

**Blocked by:** 02 — Dense mock fitting and result round trip.

**Status:** ready-for-agent
**State:** resolved

> Retired workflow: this resolved ticket records an archive-specific implementation. Its validation entry point and report are no longer maintained. Use the [current J0946 example](../../../examples/potential_correction/j0946/README.md) for new analyses; the checked items below are historical, not current verification claims.

**Spec:** [Parent specification](../spec.md)

- [x] A maintained validation entry point accepts the local archive inputs and checks their identity. It uses the archived M3 macro, observation, noise, mask, PSF, integration, grid geometry, adaptive source scale, and trial 497's three fixed strengths.
- [x] The fresh fit starts at the saved source-only field and zero potential. It uses the distinct saved hard-box bilinear M3 source as the whitening reference and fits the Hermite image. It preserves the archived image sampling precision before promotion to Float64.
- [x] The dense fit qualifies with the common \(10^{-3}\) stationarity threshold. The comparison verifies matching potential topology and derivative-valid domains.
- [x] The absolute objective difference from trial 497 is at most \(10^{-3}\), the absolute log evidence difference is at most \(0.01\), and the maximum absolute \(\delta\kappa\) difference over all valid nodes is at most \(10^{-4}\).
- [x] Retain the dense comparison and result with the input identity, controls, measured differences, and command needed to reproduce it. Routine automated checks use regenerable data.

## Comments

Claimed by `codex/lbfgs-03-j0946-dense` for integration into `codex/lbfgs-potential-correction`.

Implemented in `dd677ec` and merged in `198f1d6`. CUDA `pytest -xq test_potential_correction_lbfgs_j0946.py` from `examples/tests/` passed all 5 checks. The then-current qualification report passed every gate. The comparison and reconstruction records were retained at that revision.
