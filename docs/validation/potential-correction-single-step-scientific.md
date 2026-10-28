# Single-step scientific reconstruction on 200x200 images

Completed 2026-10-29 in `tinylens_gpu`, using implementation `34954a0` plus
retained per-file SHA256 hashes. Both backends reconstruct the same localized
Pseudo-Jaffe feature at identical fixed strengths. Its convergence peak is
suppressed by approximately 54%, whole-core recovery is modest, and broad
potential/deflection modes remain imperfect. The nonlinear re-render reveals a
substantial single-step linearization discrepancy. The matched control is
small and nonzero. These results validate the stated reconstruction example;
they establish neither detection significance nor a mass interval.

## Reproduce and inspect

From the repository root, after activating `tinylens_gpu`:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python examples/potential_correction/single_step_scientific.py \
  --config examples/potential_correction/scientific_configuration.json \
  --backend both --output output/potential_correction_scientific

XLA_PYTHON_CLIENT_PREALLOCATE=false python .scratch/potential-correction/experiments/validate_single_step_scientific.py \
  --output output/potential_correction_scientific_validation
```

The second command repeats the reference pair and independently fixed
amplitude values 1000 and 100000 with unchanged source/D4 strengths; the
reference uses 10000. It saves a compact comparison summary. Use
`--reference-already-run` when the chosen output already contains `reference`,
`--replot-only` to load public results and regenerate figures without a fit,
or `--summarize-only` to recompute only the retained summary.

The retained [summary](../../.scratch/potential-correction/research/single-step-scientific/summary.json)
contains full configurations, implementation hashes, actual sizes, block sizes,
backend field differences, recovery summaries, supplied-truth diagnostics,
and prior probes. The
[reproduction script](../../.scratch/potential-correction/experiments/validate_single_step_scientific.py)
and [scientific configuration](../../examples/potential_correction/scientific_configuration.json)
are version controlled. Large numerical products and PNG/PDF figures are
retained in the integration worktree's ignored
`.scratch/potential-correction/output/single-step-scientific/`, outside the
temporary ticket worktree. `reference` contains four complete result exports;
`amplitude-1000` and `amplitude-100000` contain the independent operator pairs.
`preliminary-operator` is an earlier fixed-strength preparation check and is
not the reference evidence.

The numerical implementation files in the reference and prior-probe records match
`34954a0`; `git_dirty=true` records pending documentation and reproduction
material at execution. Exact numerical implementation hashes are saved. Figures were subsequently
replotted from those public result arrays with shared intensity scales and
explicit U/core labels; the separate plotting hash and output-file hashes are
retained in the summary. No numerical refit was performed for that change.
Python 3.11.14, JAX 0.7.1 and NumPy 2.4.6 were observed on an NVIDIA GeForce
RTX 5080. Float64 is used throughout. This record does not measure compilation,
warm workflow performance or peak GPU memory; those are separate validations.

## Physical scenario and actual sampling

This is a new recorded configuration, not a claim that the separate
[64x64 calibration](potential-correction-single-step-calibration.md)
automatically transfers. Both use a 2.4-arcsec field of view, true SIS
Einstein radius 0.8 arcsec, shear (0.05,0.02), a singular `PseudoJaffeSubhalo`
with `b_sub=0.04` at (0.75,0.1) arcsec, and
`rt=sqrt(0.04*0.8)=0.178885` arcsec. The source Gaussian widths are
(0.16,0.12) arcsec, clump scale 0.7, with all other clump values in JSON.
The mock generator's original default 200x200 image at 0.02 arcsec/pixel,
Einstein radius 1, `b_sub=0.01` and narrower source remains unchanged and
available. Its scientific recovery is not established by this report.

The scientific images have 200x200 pixels at 0.012 arcsec/pixel. This increases
active data rather than padding the 64-pixel problem. The mock and inversion
both use nsub=2, recorded as separate controls. Noise sigma is 0.01 per native
pixel with seed 17, and both observations share the exact same Gaussian draw.
The Gaussian PSF retains physical sigma 0.05 arcsec, resampled with a 27x27
kernel (sampled radius 0.156 arcsec). The small calibration had a 5x5 kernel
at 0.0375 arcsec/pixel and a different finite truncation. Existing native-grid
circular FFT/masked-input-light conventions are preserved.

The annulus selects 0.45<=R<=1.1 arcsec. Actual counts are **22016 data**,
**2401 source**, **724 derivative-valid core** and **956 potential support**
unknowns (232 support-only), with **88064 inversion subpixel queries**. Source
geometry is fixed at 49x49 in `[-0.4,0.4]^2`, spacing 0.016667 arcsec. Potential
spacing is 0.075 arcsec. Geometry remains fixed during every fit and search;
there is no source-box expansion or mask erosion. The source-incidence
partition uses source tiles 24, cap 1024, and actual block sizes
(788,840,24,820,812,24,24,24,1). Complete block indices are saved in `run.json`
and each operator result configuration.

Every source-gradient evaluation comes from the analytic mock truth at the
fixed true macro rays, frozen for the single-step fit. Both full source and
potential fields are signed unknowns, with no clipping or alignment. Strengths
were fixed before the reference run, starting from the separate small-case
selection and scaling approximately with active data: subhalo
`lambda_source=6e-4`, `lambda4=1.6e7`; control `lambda_source=6e-4`,
`lambda4=1e11`. The independent amplitude precision is **lambda0=10000**.
These are demonstrated fixed settings, not optimized 200x200 strengths. The
reference amplitude penalty is about 32.05 versus chi2=22170.11; field/prior
sensitivity, rather than this small scalar ratio alone, supports its adoption.

The runner also supports bounded DIRECT through `--mode tune`, with explicit
natural-log bounds and independent initialization for every candidate. Small
example tests exercise successful budget-limited and all-invalid searches,
retaining optimizer/boundary/failure status. The scientific fixed-strength
comparison does not establish equality of independently tuned 200x200
operator and complete-evidence optima.

## Recovery, control and approximation limits

The following values are the dense result; operator differences at the same
strengths are negligible at the displayed precision.

| Diagnostic | Subhalo | Matched control |
| --- | ---: | ---: |
| Source relative L2, all fixed source nodes | 0.071902 | 0.051656 |
| Raw potential RMSE versus center-zero truth | 0.015423 | 0.000322 |
| Potential RMSE versus mean-centered truth | 0.003609 | 0.000322 |
| Diagnostic affine potential RMSE | 0.001450 | 0.000065 |
| Raw deflection-vector RMSE, valid core | 0.008495 arcsec | 0.000527 arcsec |
| Whole-core kappa RMSE / zero-field truth RMS | 0.027636 / 0.030863 | 0.000430 / 0 |
| Illuminated-core kappa RMSE / truth RMS | 0.028486 / 0.036190 | 0.000411 / 0 |
| Local-core kappa RMSE / truth RMS | 0.060890 / 0.099169 | 0.000236 / 0 |
| Local mean kappa / truth mean | 0.034081 / 0.043402 | 0.000124 / 0 |
| Peak kappa / discrete truth peak | 0.290141 / 0.631435 | 0.001276 / 0 |
| Peak position (arcsec) | (0.75,0.075) | (-0.9,-0.675) |
| Linear residual RMS / noise | 1.003494 | 0.995835 |
| Nonlinear residual RMS / noise | 1.029664 | 0.995807 |
| Fitted nonlinear-minus-linear RMS / noise | 0.227617 | 0.002121 |

The subhalo peak is 0.025 arcsec from the injected center. The local domain
contains 70 core nodes in the matched radius-0.357771-arcsec disk. The
illuminated domain contains 526 core nodes within the observation annulus
and analytic macro-source brightness >=5% of the core maximum. All-core
metrics include all 724 derivative-valid nodes, including weakly illuminated
regions. Analytic convergence is point-sampled outside R<1e-8 arcsec; it is
retained separately from the centered finite-difference convergence of nodal
analytic psi. No physical core or pixel-average truth is invented. Analytic
and discrete truth peaks are 0.68927 and 0.63144 respectively.

Whole-core improvement versus zero field is limited, and the recovered peak
is only 46% of discrete truth. Raw potential comparison preserves its additive
constant and affine modes: center-zero true potential has nonzero mean,
whereas the proper amplitude prior selects a fitted mean near zero. Optional
least-squares constant+x+y removal is stored only as a separately named
diagnostic array/metric and changes no fit, raw array, image or evidence.
About 19.2% of subhalo source nodes are negative, with negative magnitude
2.30% of the positive-node sum; nothing is clipped.

All 88064 macro, analytic-truth-corrected and fitted rays stay inside the
source box. Minimum signed source-box margins are 0.02461 arcsec (macro),
0.01911 (analytic subhalo truth) and 0.02984 (fitted subhalo). Thus the
recorded differences are not caused by source-box truncation. The sampled
truth source/potential differs from the noiseless analytic mock by
0.247361 noise RMS. Source interpolation at discrete potential contributes
0.112001; analytic-source/discrete-potential versus analytic truth contributes
0.223525. These differences need not add in quadrature. The corresponding
control source-interpolation mismatch is 0.109418. The analytic-source Taylor
remainder is 0.231448, distinct from the fitted frozen-gradient versus fitted
pixelized-source mismatch of 0.227617. Mock/inversion integration matches to
approximately 1e-14 noise RMS in this nsub-matched case.

PCG uses maxiter 20000, rtol 1e-12, atol 0 and independent actual-system
acceptance 1e-8. All four reference fits succeed. Actual system relative
residuals are 1.03e-15/8.65e-13 (dense/operator subhalo), and
2.33e-13/2.01e-12 (control). Across both cases, maximum absolute
source/potential/convergence differences are 6.43e-11/1.08e-11/6.21e-11;
nonlinear-image RMS differences stay below 6.03e-11 noise units.

Complete conditional log evidence is 68989.98446 (subhalo) and 69486.83243
(control). Block evidence is higher by 108.52609 and 72.68271 respectively.
These values describe different observations and determinant methods; no
detection interpretation is drawn from their subtraction. `evidence_method`
remains `conditional_linear_gaussian`; dense `determinant_method` is
`full_cholesky`, operator is `principal_block_cholesky` with
`evidence_is_exact=false`. Both single-step evidence calculations use the
linear image. The nonlinear re-render never replaces that image in evidence.

## Independently fixed amplitude sensitivity

Each probe keeps the same data, geometry, frozen gradient, source/D4 strengths,
partition and solver controls. Lambda0 is fixed for each complete pair and
is never optimized, rescaled or repaired inside a fit.

| Operator subhalo | lambda0=1000 | lambda0=10000 | lambda0=100000 |
| --- | ---: | ---: | ---: |
| Source relative L2 | 0.08513 | 0.07190 | 0.06867 |
| Raw potential RMS | 0.003306 | 0.001831 | 0.001274 |
| Whole-core kappa RMSE | 0.031777 | 0.027636 | 0.029332 |
| Local kappa RMSE | 0.062472 | 0.060890 | 0.061532 |
| Local mean kappa | 0.032393 | 0.034081 | 0.029786 |
| Peak kappa | 0.281538 | 0.290141 | 0.295679 |
| Control whole-core kappa RMS | 0.000432 | 0.000430 | 0.000421 |

All peaks retain the same (0.75,0.075) position. The largest actual system
relative residual across probes is 1.81e-11. The adopted value preserves
localization and similar local error, improves raw broad-mode behavior
relative to 1000, and has the lowest whole-core error among these probes.
Changing it one decade changes local mean kappa by up to 12.6%, and raw
potential RMS changes much more. This sensitivity and peak suppression limit
the interpretation; “weak amplitude prior” does not mean all potential modes
are measured. No precision claim follows from a stable peak location.

## Artifact and regression checks

Each public result export contains source axes/box, raw potential unknowns,
full core/support/query/interpolation/derivative geometry, derivative validity,
window coverage, all three strengths, initialization, method annotations,
actual solver status, both model images/residuals, observation/noise/PSF/mask
and analytic/discrete truth products. Support-only derivative maps remain
NaN and blank in figures. Images and raw field figures were inspected at
their saved size. Source/image comparison panels share intensity scales.
No fitted field is clipped or truth aligned; residual color scales saturate
at the explicitly displayed +/-4 noise units while numerical arrays remain raw.

The maintained example tests cover matched assembly, numerical and metadata
persistence, distinct renderings, validity domains, CLI tuning with retained
budget status, and all-invalid tuning replacing a previous successful export
with diagnostic failure records only. Failed fixed-strength results also go
through the same failure exporter. Plotting and persistence receive only a
public `PotentialCorrectionResult` and observation/truth context; they do not
construct or access a solver.

Verification commands and final outcomes are recorded in ticket 11. Core and
example processes are separate. The bounded dense-column materializer from
integration `4f8dda4` permits the actual dense run without changing the
mathematical model; this report makes no memory or speed comparison claim.
