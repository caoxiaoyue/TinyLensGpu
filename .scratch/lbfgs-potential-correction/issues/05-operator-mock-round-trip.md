# 05: Operator mock fitting and result round trip

**What to build:** A modeler can repeat the same matched mock and control fits with the operator backend and compare both backends through the maintained example and shared result products.

**Blocked by:** 04 — Fixed-strength operator L-BFGS-B fit.

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Run the operator backend on the same regenerated injection and control from ticket 02 with the same observations, actual starts, and fixed strengths. Record the operator partition and controls.
- [x] Both operator fits qualify. Report their objectives, image residuals, and convergence fields on matching valid domains alongside the dense results. Record each evidence value with its determinant method.
- [x] Export and reload the operator products with the physical fields, partition metadata, principal-block determinant method, and terminal stationarity and residual diagnostics. The dense products remain readable through the same interface.
- [x] Example regressions run the comparison and readback in a separate process from core tests using regenerable data.

## Comments

Claimed by `codex/lbfgs-05-operator-mock` for integration into `codex/lbfgs-potential-correction`.

Implemented in `d565f74` and merged in `baa3963`. CUDA `pytest -xq test_potential_correction_lbfgs_mock.py` from `examples/tests/` passed both tests. The documented `python -m examples.potential_correction.lbfgs_mock --backend both` run qualified all four fits and certified both coupled residuals. See the [reproducer and full-domain measurements](../../../examples/potential_correction/README.md#fixed-strength-l-bfgs-b-mock) and [retained comparison](../output/mock-comparison/run.json).
