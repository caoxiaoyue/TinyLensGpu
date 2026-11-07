# 02: Dense mock fitting and result round trip

**What to build:** A modeler can run the dense L-BFGS-B fitter on a regenerated subhalo injection and matched no-subhalo control and inspect both reconstructions through the common result export and readback workflow.

**Blocked by:** 01 — Fixed-strength dense L-BFGS-B fit.

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] A maintained example fits the existing Pseudo-Jaffe injection and no-subhalo control at recorded strengths using the same noise, mask, and observation setup for the pair.
- [x] Both dense fits qualify. The example records convergence recovery on the valid domain and nonlinear image residuals for the injection and control, with the configuration needed to regenerate the pair.
- [x] The common exporter and reader round-trip the fitted fields, valid-domain metadata, strengths, evidence method, and `optimizer_info` with an empty `solver_info`. Numerical failure exports the available reason and diagnostics.
- [x] Existing single-step and GN/LM products still load through the shared reader, and their export regressions pass.
- [x] Example-suite checks exercise the matched run and persistence with regenerable inputs in a process separate from the library tests. The example configuration is reusable for ticket 05.

## Comments

Claimed by `codex/lbfgs-02-dense-mock` for integration into `codex/lbfgs-potential-correction`.

Implemented in `04e9079` and merged in `792db04`. CUDA `pytest -xq test_potential_correction_lbfgs_mock.py test_potential_correction_scientific.py test_potential_correction_iterative_scientific.py test_potential_correction_operator_example.py test_potential_correction_calibration.py::test_saved_public_results_keep_pre_review_truth_domains_and_recovery` from `examples/tests/` passed all 18 cases in `tinylens_gpu`. The documented `python -m examples.potential_correction.lbfgs_mock` run qualified the injection and control. See the [reproducer and measurements](../../../examples/potential_correction/README.md#fixed-strength-l-bfgs-b-mock) and [retained run](../output/mock-dense/run.json).
