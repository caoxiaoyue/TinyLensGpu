# 07: Qualify the J0946 two-backend refit

**What to build:** A researcher can qualify the operator J0946 reconstruction against the validated dense reconstruction at the same fixed strengths and inspect both numerical and resource measurements.

**Blocked by:** 05 — Operator mock fitting and result round trip, 06 — Verify bounded memory for operator fits.

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] The maintained J0946 entry point runs the operator fit with the same archived observation, geometry, actual start, whitening reference, and strengths as the dense refit from ticket 03. Retain the exact operator partition and controls used for qualification.
- [x] Both fits qualify at the common \(10^{-3}\) GN stationarity threshold, including the operator true-residual certificate. Check matching potential topology and derivative-valid domains before comparing fields.
- [x] The operator objective differs from the dense objective by at most \(0.01\). The maximum absolute \(\delta\kappa\) difference over the full valid domain is at most \(0.005\). The positive peak positions differ by at most one potential grid spacing.
- [x] Retain both local GN evidence values, determinant methods, and the measured difference. Use the small explicit principal-block check from ticket 04 as the operator evidence verification.
- [x] The validation report links the input identity and fit products, records the comparison gates and measured differences, and reports stage timings and peak device memory for both backends. Include the commands and measurement configuration needed to repeat the qualification.

## Comments

Claimed by `codex/lbfgs-07-j0946-qualification` for integration into `codex/lbfgs-potential-correction`.

Implemented in `a4d39e3` and merged in `1484f43`. `pytest -xq test_potential_correction_lbfgs_j0946.py` passed all 14 regenerated workflow checks on CPU and CUDA from `examples/tests/`. The independent CUDA principal-block check passed with `pytest -xq test_potential_correction_lbfgs.py::test_operator_fixed_strengths_qualify_against_principal_block_reference` from `tests/`. Both fresh isolated CUDA commands in the [qualification report](../../../docs/validation/potential-correction-lbfgs-j0946.md) passed. The [dense fit](../output/j0946-qualified/dense/result.json) and [operator comparison](../output/j0946-qualified/operator/result.json) retain their fields, methods, partition, diagnostics and resource measurements.
