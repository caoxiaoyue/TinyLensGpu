# 01: Fixed-strength dense L-BFGS-B fit

**What to build:** A modeler can run a Float64 cubic Hermite source–potential reconstruction at a supplied strength triple through the public L-BFGS-B fitter. The dense fit returns the recovered fields, nonlinear image, convergence diagnostics, and qualified local GN evidence.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] `ObservationModel` exports `LBFGSPotentialCorrection`. The fitter uses the prepared renderer, fixed macro and geometry, observation and noise, signed initial source, source regularization, and optional fixed adaptive `source_scale`.
- [x] Each call takes three finite positive strengths through `PotentialCorrectionStrengths` and starts from the supplied source and zero potential. A public whitening reference defaults to that Hermite source and can supply distinct source values and a reference interpolant on the same grid. The fitted image always uses the Hermite model.
- [x] The default dense path minimizes the shared nonlinear objective with JAX gradients and data-space whitening. The fitted objective includes the accumulated source prior, complete fourth-difference penalty, and all-node amplitude prior at the supplied strengths.
- [x] The result contains the raw fields and valid-domain maps, nonlinear image and residuals, and the final objective. The complete undamped data-space determinant ratio supplies qualified evidence labeled `local_gauss_newton`, with the determinant method and `evidence_is_exact=False`.
- [x] Fit success requires native L-BFGS-B success, finite endpoint quantities, and the GN stationarity check. The public defaults are `max_iter=3000` and `gn_stationarity_tol=1e-3`. `optimizer_info` records the optimizer exit and endpoint diagnostics, with `solver_info=None`. Invalid configuration raises, and numerical failure returns its reason, available fields, and `log_evidence=None`.
- [x] An independent small Hermite case with nonzero source–potential coupling verifies the objective and gradient, dense whitening, endpoint stationarity, and complete determinant ratio. An A → B → A sequence on one object checks per-call amplitude strength, normalization, and repeatability.
- [x] Applicable single-step and GN/LM fit and inference regressions pass after the shared-code changes.

## Comments

Claimed by `codex/lbfgs-01-dense` for integration into `codex/lbfgs-potential-correction`.

Implemented in `305e3a4` and merged in `cdc19c1`. In the activated `tinylens_gpu` environment from `tests/`, `XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 pytest test_potential_correction_lbfgs.py` passed all 5 tests on CUDA. The related legacy suite (`test_potential_correction_{dense,operator,iterative,iterative_operator,inference,iterative_inference,jax_compatibility,precision,rendering}.py`) passed 150 checks on CPU. Its two subnormal prior cases also fail on the unchanged baseline in CPU mode. Both passed on CUDA with `pytest test_potential_correction_iterative.py::test_nonfinite_trial_is_rejected_without_replacing_last_finite_fields`.
