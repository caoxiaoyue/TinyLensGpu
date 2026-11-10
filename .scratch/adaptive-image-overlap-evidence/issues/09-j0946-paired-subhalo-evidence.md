# 09: Migrate J0946 paired subhalo evidence

**What to build:** A J0946 researcher can run the M4 and M5 comparison with reference-image adaptation and verify that the baseline and candidate use compatible scoring configurations before interpreting their Bayes factor.

**Blocked by:** [06 — Run Euclid M1 through M3 with a shared reference arc](06-euclid-adapt-image-stages.md).

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Both M4 and M5 construct their adaptive fits using `adapt_image`, with a common reference identity, fixed grid, preparation rule and calibrated strengths.
- [x] The comparison fingerprints the relevant observation, interpolation, adaptation and evidence settings and records the actual scoring method for each fit.
- [x] A saved M4 can serve as a comparison denominator when its provenance matches. Detect an old fixed-template M4 or a different score and require the applicable matched refit.
- [x] Focused J0946 regressions exercise both matched comparison and mismatched recovery through the maintained entry point.

## Comments

- Claimed on `codex/overlap-09-j0946-evidence`.
- Implemented in `a9f0007`. From `examples/tests/` in `tinylens_gpu` on CPU, 14 targeted J0946 cases passed, including eight stage cases in `test_joint_shared_support.py` and six diagnostic cases in `test_operator_diagnostics_plotting.py`. The driver passed `--help` and Python compilation checks. The tests cover matched M4/M5 reference and score settings, M5-only rejection of a stale fixed-template denominator or an actual Block denominator for Overlap before writing output, and a matched refit through the maintained entry point.

Verification ran from `examples/tests/` in `tinylens_gpu` with the checkout on `PYTHONPATH`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `JAX_PLATFORMS=cpu`, and `MPLBACKEND=Agg`.

```bash
pytest -q \
  test_joint_shared_support.py::test_j0946_resumes_before_and_after_source_package \
  test_joint_shared_support.py::test_j0946_rejects_incompatible_comparison_caches_before_reuse \
  'test_joint_shared_support.py::test_joint_sampling_and_cache_contract[pipe/subhalo_0946/model.py-100-True]' \
  'test_joint_shared_support.py::test_subhalo_source_package_normalization_and_identity[pipe/subhalo_0946/model.py]' \
  'test_operator_diagnostics_plotting.py::test_subhalo_plots_reject_failed_reconstruction_before_output[unconverged-pix_src_demo_operator/pipe/subhalo_0946/model.py]' \
  'test_operator_diagnostics_plotting.py::test_subhalo_plots_reject_failed_reconstruction_before_output[nonfinite-pix_src_demo_operator/pipe/subhalo_0946/model.py]' \
  'test_operator_diagnostics_plotting.py::test_subhalo_exports_reject_failed_reconstruction[unconverged-pix_src_demo_operator/pipe/subhalo_0946/model.py]' \
  'test_operator_diagnostics_plotting.py::test_subhalo_exports_reject_failed_reconstruction[nonfinite-pix_src_demo_operator/pipe/subhalo_0946/model.py]' \
  'test_operator_diagnostics_plotting.py::test_subhalo_exports_keep_raw_package_contract_with_unsupported_joint_stats[pix_src_demo_operator/pipe/subhalo_0946/model.py]' \
  'test_operator_diagnostics_plotting.py::test_subhalo_plots_preserve_position_scale_and_subhalo_overlays[pix_src_demo_operator/pipe/subhalo_0946/model.py]'
```
