# 03: Deliver operator diagnostics through a complete example

**What to build:** An operator-backend user obtains a reconstruction, convergence-aware diagnostics, and requested image components through the same public result contract, with a working source-only example and explicit joint-diagnostic limitations.

**Blocked by:** 02: Deliver dense diagnostics through a complete example

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Add the operator `evaluate_with_diagnostics` entry point using the common contract and one source reconstruction inversion, with optional calculations disabled by default.
- [x] Expose the specified basic, adaptive-regularization, and enabled position diagnostics; preserve the existing source-only block-preconditioner effective-parameter approximation and identify its method.
- [x] For joint semi-linear inversion, return the actual total and requested component images while retaining block-Schur preconditioning and unit-amplitude lens-light basis semantics.
- [x] Return `None` with an explanatory unsupported reason for requested operator joint effective parameter count, residual degrees of freedom, and reduced chi-square; retain valid raw chi-square and image products without marking the reconstruction invalid.
- [x] Mark explicit non-convergence invalid even when the existing core projection returns finite zero arrays, and reject nonfinite reconstruction values; preserve core evidence penalties and reconstruction failure returns.
- [x] Migrate the basic operator source-only fitting example without position constraints to this public path and stop formal output for invalid reconstruction.
- [x] Verify small real source-only and joint problems, supported PCG/PNPG behavior, optional calculation, components, failure status, and the single-reconstruction contract through public outputs.
- [x] Preserve existing likelihood, batching, and differentiation behavior; keep unmigrated callers usable and record focused core and example verification separately.

## Comments

- Claimed and resolved by Sidekick on branch `pwr/03-operator-diagnostics` (worktree /tmp/tinylens-pwr-03).
- Implementation: `_evaluate` gained `return_details` capturing
  `(block_chols, reg_blocks, lambda_reg, model_source, model_lens)` —
  reused active component vectors; the convergence zero projection and 2D
  scatter run only inside `include_components`;
  new `evaluate_with_diagnostics` reuses the common `reconstruction_diagnostics`
  helper. Source-only effective DOF uses the block-preconditioner trace
  (`operator_source_block_preconditioner`); joint requests return None triple
  with method `operator_joint_unsupported`. Tracer
  `examples/pix_src_demo_operator/simple/pix_src/fit_lens_src.py` migrated.
- Verification (conda tinylens_gpu, `XLA_PYTHON_CLIENT_PREALLOCATE=false`):
  `pytest test_pixelized_operator.py -k 'diagnostics or evaluate or forward_model' -q`
  from `tests/` -> 31 passed.
  `MPLBACKEND=Agg pytest test_operator_diagnostics_plotting.py -q` from
  `examples/tests/` -> 2 passed.
  Review fix + rerun: `pytest test_pixelized_operator.py -k
  'diagnostics_effective_dof or diagnostics_components or diagnostics_joint or
  diagnostics_source_only or diagnostics_match or diagnostics_unconverged' -q`
  -> 16 passed (block_size {2,5} x rho {0,1} x {pcg,pnpg} ragged/adaptive coverage).
