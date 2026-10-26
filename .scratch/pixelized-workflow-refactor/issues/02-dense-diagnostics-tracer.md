# 02: Deliver dense diagnostics through a complete example

**What to build:** A dense pixelized-source user can request reconstruction and diagnostics through one public call and produce a diagnostic plot in the basic dense source-only fitting example without managing solver internals.

**Blocked by:** 01: Restore native JAX array truthiness

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Introduce the common diagnostic result contract and the dense `evaluate_with_diagnostics` entry point, returning the existing `InversionResult` plus diagnostics; `include_effective_dof` and `include_components` default to false.
- [x] Support the specified basic, adaptive-regularization, and enabled position diagnostics, and requested component images, for dense source-only and joint semi-linear inversion.
- [x] Preserve the current dense effective-parameter convention, lens-light regularization treatment, and supported edge-case handling; identify the method without adding a new constrained-solver interpretation.
- [x] Produce paired results from one source reconstruction inversion; distinguish unrequested values from invalid reconstruction and retain dense's absence of a solver convergence status.
- [x] Migrate the basic dense source-only fitting example without position constraints to the public diagnostic path, preserving its scientific quantities and display conventions and rejecting nonfinite reconstruction before formal output.
- [x] Verify the public result against small real dense problems, including joint components and optional diagnostics, and enforce the single-reconstruction contract with a narrowly scoped invocation probe.
- [x] Keep existing `evaluate`, forward-model, likelihood, batching, and differentiation contracts intact; ordinary likelihood evaluation performs no optional diagnostic work.
- [x] Leave other callers working through their existing interfaces during expansion and record focused core and example verification separately.

## Comments

- Claimed by Sidekick on branch `pwr/02-dense-diagnostics` (worktree /tmp/tinylens-pwr-02).
- Resolved: `InversionDiagnostics`/`PositionDiagnostics` added to `inversion_result.py`,
  exported via both ObservationModel packages; shared `reconstruction_diagnostics`
  in `_inversion_diagnostics.py`; dense `evaluate_with_diagnostics` added with
  `include_effective_dof`/`include_components` and the `dense_curvature` method;
  `_evaluate` gained `return_details` so one solve feeds both outputs; tracer
  `examples/pix_src_demo/simple/pix_src/fit_lens_src.py` migrated to the public
  call (40x40 literal -> `pix_src.n`) and rejects nonfinite reconstructions (real NaN observation input) before plotting.
  Verification (conda tinylens_gpu, `XLA_PYTHON_CLIENT_PREALLOCATE=false`):
  `pytest test_pixelized_inversion.py -k 'diagnostics or dense_evaluate or dense_forward_model' -q`
  from `tests/` -> 26 passed.
  `MPLBACKEND=Agg pytest test_joint_fit_plotting.py -q` from `examples/tests/` -> 4 passed.
  Review fixes: `_evaluate` return annotation widened for `return_details`;
  nonfinite tests now drive invalidity through NaN `image_data` on public paths
  (no `_solve_source` monkeypatch); adaptive scale expectation uses public
  `source_template_scale_map`; position test derives separations/penalty
  independently via public `deflection` and the documented penalty expression;
  tracer helper plots `prob_model.image_data` and closes figures via fixture.
  Post-review rerun: `pytest test_pixelized_inversion.py -k
  'diagnostics_nonfinite or diagnostics_adaptive or diagnostics_position' -q`
  -> 3 passed; `MPLBACKEND=Agg pytest test_joint_fit_plotting.py -k tracer -q`
  -> 2 passed.
