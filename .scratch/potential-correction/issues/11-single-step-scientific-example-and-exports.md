# 11: Single-step scientific examples, diagnostics, and exports

**What to build:** A researcher can run an approximately 200-by-200-image single-step Pseudo-Jaffe experiment with either backend and obtain reproducible saved reconstructions and figures that distinguish recovery from linearization error.

**Blocked by:** 07 — Operator tuning and single-step backend calibration.

**Status:** ready-for-agent
**State:** resolved

**Parent:** [Approved potential-correction spec](../spec.md).

- [x] A documented example composes the approved mock, fixed true macro, truth-derived gradient initialization, and dense/operator single-step fitting/tuning.
- [x] Image size, source resolution, potential spacing, integration settings, strengths/search bounds, dtype, and output destination are configurable.
- [x] The example includes matched subhalo and no-subhalo runs, with adopted weak-prior calibration and numerical settings recorded.
- [x] Numerical exports preserve source and potential fields, core/support geometry, derivative validity, strengths, method annotations, initialization provenance, configuration, and solver status.
- [x] Figures separately identify the linear fitted image and nonlinear re-render, with corresponding residual diagnostics; exported single-step evidence remains tied to the linear model.
- [x] Diagnostic products include source, potential, deflection, and convergence, plus perturbation localization, convergence error, source recovery, and image residual summaries on documented domains.
- [x] Raw fitted fields are retained; any affine-aligned truth comparison is explicitly identified and does not replace inference outputs.
- [x] Failure records remain available, but a failed fit cannot be exported or labeled as a successful scientific reconstruction.
- [x] Plotting and persistence consume the public result rather than independently assembling or solving the inversion.
- [x] Small example tests verify assembly, saved metadata and numerical products, distinct rendering semantics, valid-domain handling, and failure behavior.
- [x] The scientific run and its reproduction instructions record actual configurations and results without claiming formal detection significance or mass confidence intervals.

## Scope and handoff

This slice may proceed alongside dense iterative development. Establish
reusable result-based example persistence and plotting where it is genuinely
shared, while keeping the explicit single-step workflow visible.


## Comments

- 2026-10-29: Claimed on `codex/pc-11` in `/tmp/tinylens-pc-11`, based on integration `35d0318`; ticket 07 is resolved. Public result/artifact/example seams were approved in the spec.

- 2026-10-29: Implemented on `codex/pc-11`; numerical/export runner commits
  `ece1dcd`, `34954a0`, and completed validation/docs/public loader commit
  `72b5f04a80199872b19005023456d829145d7ca8`. Consumed integration's bounded
  dense-column materializer through `4f8dda4`; no ObservationModel changes
  belong to ticket 11.
- Public seams: `single_step_scientific.run_scientific` explicitly assembles
  one frozen-gradient single-step problem and fits/tunes it; shared
  `result_products.export_result`, `load_result`, `summarize_result`, and
  `plot_result_products` consume public results plus observation/truth context.
  Safe NPZ/JSON loading restores stored geometry and arrays without a fit.
  Failure records cannot be loaded as a successful result; all-invalid tuning
  replaces previous arrays/figures with diagnostic records only.
- Red/green evidence: the first artifact test failed at the missing runner;
  a real one-iteration PCG/all-invalid tuning test exposed stale successful
  artifacts; supplied-truth metadata failed before implementation; public
  loading failed before implementation and failure loading exposed a raw
  missing-file error. Each slice was implemented and passed at the approved
  result/artifact/example seams. No pixel snapshots were introduced.
- Verification in `tinylens_gpu`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, from
  `examples/tests/`, in a separate process from core tests:
  `pytest test_potential_correction_scientific.py test_potential_correction_mock.py test_potential_correction_operator_example.py test_potential_correction_tuning.py test_potential_correction_calibration.py -q`
  — **19 passed in 57.09 s**. After final failure-provenance and presentation
  changes, `pytest test_potential_correction_scientific.py -q` — **3 passed
  in 20.52 s**. Python compilation and `git diff --check` pass. Integration's
  dense batching fix has its own 33-reference-check verification record.
- Actual scientific reference command from repository root:
  `python examples/potential_correction/single_step_scientific.py --config examples/potential_correction/scientific_configuration.json --output /home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/.scratch/potential-correction/output/single-step-scientific/reference`
  in the activated environment with preallocation disabled. Follow with
  `python .scratch/potential-correction/experiments/validate_single_step_scientific.py --reference-already-run --output /home/xycao/workspace/tinylens_dev/TinyLensGpu-potential-correction/.scratch/potential-correction/output/single-step-scientific`
  for independently fixed amplitude probes. `--replot-only` subsequently
  regenerated figures from the retained public NPZ/JSON products with identical
  observation context, common intensity scales and explicit U/core labels;
  no numerical refit was made for presentation changes.
- Actual float64 configuration: 200x200 at 0.012 arcsec/pixel (2.4 arcsec field),
  mock/inversion nsub=2/2, Gaussian PSF sigma=0.05 arcsec/size 27, fixed
  SIS theta_E=0.8/shear (0.05,0.02), PseudoJaffeSubhalo b=0.04 at (0.75,0.1),
  smooth source widths (0.16,0.12)/clump scale 0.7, fixed 49x49 source box
  [-0.4,0.4]^2, h=0.075, lambda0=10000, tile24/cap1024, PCG rtol=1e-12,
  maxiter=20000, actual acceptance=1e-8. Counts: data22016/source2401/core724/
  support956/query88064. This recorded physical scenario preserves the
  generator defaults and differs explicitly from the 64x64 PSF sampling.
- Demonstrated fixed strengths: subhalo lambda_s=6e-4/lambda4=1.6e7;
  control lambda_s=6e-4/lambda4=1e11. Both backends succeed and agree in raw
  fields to <=6.43e-11, with actual relative system residuals <=2.01e-12.
  Block log evidence differs from complete evidence by +108.526/+72.683;
  exact/approximate determinant methods remain separate. No optimized 200x200
  strength or independently tuned 200x200 optimum agreement is claimed.
- Scientific evidence: peak(0.75,0.075), 0.025 arcsec from injection;
  recovered kappa peak0.29014 versus discrete truth0.63144. All-core kappa
  RMSE0.02764 versus zero-field baseline0.03086; local RMSE0.06089 versus
  baseline0.09917. Source relative L2=0.07190. Linear/nonlinear residual
  RMS/noise=1.00349/1.02966 and fitted image discrepancy0.22762 noise.
  Control all-core/local kappa RMS=0.000430/0.000236. All selected rays stay
  inside the fixed source box. Lambda0 probes1000/10000/100000 preserve the
  peak location and local RMSE0.06247/0.06089/0.06153, but raw potential
  modes and local mean kappa remain prior dependent (up to ~12.6%).
- Evidence and remaining limits:
  [validation report](../../../docs/validation/potential-correction-single-step-scientific.md),
  [summary/configs/hashes](../research/single-step-scientific/summary.json),
  [reproduction script](../experiments/validate_single_step_scientific.py).
  Complete numerical products and inspected PNG/PDF figures are retained at
  the authorized ignored integration-worktree output path above, outside this
  ticket worktree. Raw fields are preserved; optional affine products are
  separately labeled and never substituted. Approximately 54% peak suppression,
  modest whole-core improvement, broad-mode/prior dependence and linearization
  error remain explicit. This is reconstruction validation, without detection
  significance, mass intervals, posterior sampling or unmeasured performance
  claims. Float32/performance and iterative scientific work remain downstream.
