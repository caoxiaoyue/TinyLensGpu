# Single-step backend calibration on a 64x64 mock

Completed 2026-10-29 in `tinylens_gpu`, with numerical implementation
`694812d`. Ticket 07 passes for the configuration recorded here: the operator
selects a similar localized reconstruction to the complete dense calculation,
with a small, nonzero matched-control correction. Full potential and
convergence recovery remain limited by low-order modes, regularization and
resolution. The recovered convergence peak is substantially suppressed.
This is reconstruction/backend calibration, without a detection-significance
or subhalo-mass-precision claim.

The adopted demonstration uses independently fixed **lambda0=10000**,
source bounds **[1e-5,1e-3]**, D4 bounds **[1e5,1e11]**, and float64.
The incidence partition has source tiles of width 24, cap 1024 and actual block sizes
(787,834,24,814,805,24,24,24,1). PCG uses maxiter 20000, rtol 1e-12, atol 0;
the actual coupled-system residual must be below 1e-8. These choices apply to
this geometry and initialization. A different scientific configuration needs
its own approximation and prior-sensitivity check.

## Reproduce and inspect

From the repository root:

```bash
source ~/anaconda3/bin/activate && conda activate tinylens_gpu
XLA_PYTHON_CLIENT_PREALLOCATE=false python examples/potential_correction/calibrate_single_step.py \
  --config examples/potential_correction/calibration_configuration.json \
  --output output/potential_correction_calibration
```

The complete [configuration](../../examples/potential_correction/calibration_configuration.json)
contains all three independently fixed amplitude values 100,1000,10000.
Every candidate starts with the same fixed macro and frozen analytic truth
gradient; no fitted state is passed to another candidate. The driver saves
raw public fields and distinct linear/nonlinear renderings, scalar candidate
status, initialization provenance, geometry and truth conventions. Optional
affine alignment is diagnostic only and changes no fitted field or evidence.
The reusable example-level `source_incidence_blocks(renderer, mapping, gx, gy,
source_tile=..., block_size=...)` consumes explicit initial arrays and retains
no truth callable. Future iterative callers can use initial pixelized-source
gradients while independently refreshing their numerical Jacobian.

Retained evidence:

- [Summary and scan](../../.scratch/potential-correction/research/single-step-calibration/summary.json).
- [Complete candidate records](../../.scratch/potential-correction/research/single-step-calibration/complete-record.json.gz),
  [inputs](../../.scratch/potential-correction/research/single-step-calibration/inputs.npz),
  [raw selected fields](../../.scratch/potential-correction/research/single-step-calibration/selected-fields.npz).
- [Independent layout exploration](../../.scratch/potential-correction/research/calibration-exploration-summary.json)
  and [dense determinant-reference script](../../.scratch/potential-correction/experiments/calibration_block_reference.py).
- [Boundary probe](../../.scratch/potential-correction/research/calibration-boundary-probe.json)
  and [script](../../.scratch/potential-correction/experiments/calibration_boundary_probe.py).

The three amplitude cases were run in independent processes with separate
configuration/log/output paths `/tmp/pc07-inverse-calibration-{100,1000,10000}`.
Each process ran both observations, with all other controls unchanged.
The [consolidation script](../../.scratch/potential-correction/experiments/summarize_single_step_calibration.py)
requires complete case pairs, unique run identities, identical input arrays,
identical geometry/partition and successful order/failure checks. Its retained
manifest contains component configurations and file hashes. The original
formatter omitted the public initialization-provenance dictionary; consolidation
labels its completion from the explicit runner construction in `694812d`.
A failing artifact test exposed that omission and the current driver now saves
and verifies the public dictionary directly. Numerical fields were unchanged.

Environment: Python 3.11.14, JAX 0.7.1, NumPy 2.4.6, SciPy 1.17.1,
NVIDIA GeForce RTX 5080, driver 610.57.04. The retained summary records the
observed software versions. These are the installed verification versions.

## Physical model, domains and discretization

The image is 64x64 at 0.0375 arcsec/pixel, with nsub=2 for generation and
reconstruction, noise sigma 0.01 and shared seed 17. The fixed true macro is
SIS theta_E=0.8 plus shear (0.05,0.02). The singular Pseudo-Jaffe mock has
b=0.04 at (0.75,0.1) arcsec and rt=sqrt(0.04*0.8)=0.178885 arcsec. This
explicit stronger small-case perturbation does not replace the default
200x200/b=0.01 scientific example. Source Gaussian widths are (0.16,0.12)
arcsec with clump_scale=0.7 and the full two-clump parameters in JSON.

The fixed 49x49 source spans [-0.4,0.4]^2 with 0.016667 arcsec spacing. The
PSF has sigma 0.05 arcsec, size 5, and the existing native circular-FFT/mask
semantics. The selected annulus is 0.45<=R<=1.1 arcsec. Potential spacing
is 0.075 arcsec. Active counts are 2252 data, 2401 source, 704 core and 936
potential unknowns, 3337 total. No source-box expansion, mask erosion or
changed PSF convention is used.

Potential truth is analytic nodal psi with psi(subhalo center)=0. Discrete
truth derivatives apply the reconstruction's centered physical differences to
those nodes, on all 704 valid core positions. Analytic convergence is also
compared at core point samples, excluding R<1e-8, with no invented core or
pixel averaging. Analytic peak kappa is 0.68927; discrete peak is 0.63144.
All-core errors retain every derivative-valid core node. Two additional
predeclared diagnostic domains are 526 illuminated core nodes (inside the
annulus and analytic macro-source brightness>=5% of the core maximum), and 70
local nodes in the matched disk of radius max(2*rt,2*h). They do not replace
`derivative_valid`; support-only derivatives remain undefined/NaN.

The supplied sampled truth differs from the noiseless analytic subhalo image
by 0.27081 noise RMS. Its source-interpolation contribution at the discrete
potential is 0.11428; the analytic-source/potential-discretization difference
is 0.24837. These contributions need not add in quadrature. The control's
source-interpolation mismatch is 0.11168. Analytic-source Taylor remainder is
0.19989 noise RMS; conditional-linear versus nonlinear sampled-truth mismatch
is 0.20093. The fitted conditional/nonlinear discrepancy is separately saved,
because its frozen analytic gradient differs from the fitted source gradient.

All 9008 macro/truth-corrected subpixel rays are inside the fixed source box,
with minimum signed margins 0.02977/0.02431 arcsec. All selected fitted rays
are inside; the minimum across all priors/backends/cases is 0.02207 arcsec.
These checks identify source interpolation and potential discretization
without attributing a source-box truncation error to either backend.

## Selected fields and evidence at lambda0=10000

| Subhalo result | Complete dense | Operator block evidence |
| --- | ---: | ---: |
| lambda_source | 5.66352e-5 | 6.10954e-5 |
| lambda4 | 1.66810e6 | 1.66810e6 |
| Source relative L2 error | 0.08163 | 0.08206 |
| Raw potential RMSE, center-zero truth | 0.015354 | 0.015354 |
| Potential RMSE, truth mean-centered only | 0.0034265 | 0.0034286 |
| Diagnostic affine-aligned potential RMSE | 0.0015222 | 0.0015234 |
| Raw deflection-vector RMSE (arcsec) | 0.007749 | 0.007778 |
| Whole-core kappa RMSE (truth RMS 0.03130) | 0.030718 | 0.030773 |
| Illuminated-core kappa RMSE (truth RMS 0.03619) | 0.028813 | 0.028849 |
| Local kappa RMSE (truth RMS 0.09917) | 0.059631 | 0.059554 |
| Local mean kappa (truth 0.04340) | 0.032227 | 0.032352 |
| Peak kappa (discrete truth 0.63144) | 0.306001 | 0.306976 |
| Peak position (arcsec) | (0.75,0.075) | (0.75,0.075) |
| Linear residual RMS/noise | 0.96968 | 0.97035 |
| Nonlinear residual RMS/noise | 0.99849 | 0.99908 |

Both peaks are 0.025 arcsec from the injection. The whole-core kappa error
remains close to the zero-field baseline; local improvement and localization
do not establish complete field recovery. The raw potential error includes
its unconstrained additive constant: fitted mean is near zero, whereas
center-zero truth mean is 0.0149663. The separately labeled affine diagnostic
removes a least-squares constant+x+y difference on U. It changes neither
stored fields, source, renderings nor inference. Deflection and broad modes
remain imperfect even after diagnostic alignment. The signed Gaussian source
has about 22% negative nodes; their summed magnitude is about 2.7% of the
positive-node sum. Nothing is clipped or substituted in the evidence.

The complete-evidence difference between the dense selection and a complete
dense evaluation at the operator selection is 0.14592. This is distinct from
the absolute block-evidence error: at the dense selected strengths, operator
logZ differs by 88.4353 despite essentially identical fitted fields/objective.
The determinant approximation is therefore not numerically interchangeable
with complete evidence. At the two selected reconstructions, linear/nonlinear
image RMS differences are 0.00378/0.00509 noise units.

| Matched control result | Complete dense | Operator block evidence |
| --- | ---: | ---: |
| lambda_source | 6.71677e-5 | 6.71677e-5 |
| lambda4 | 9.71973e10 (upper edge) | 1.18597e10 |
| Source relative L2 error | 0.06780 | 0.06803 |
| Whole-core kappa RMS | 0.002476 | 0.002790 |
| Local kappa RMS | 0.001288 | 0.001563 |
| Local mean kappa (truth 0) | 0.000260 | -0.000175 |
| Largest kappa | 0.005185 | 0.006331 |
| Linear/nonlinear residual RMS/noise | 0.96976/0.96983 | 0.96892/0.96903 |

The control correction is nonzero, with its peak at (-0.825,-0.825) rather
than the injected subhalo location. Its shape difference and all raw maps
are retained. The complete-evidence cost of the operator choice is 1.58618;
its changed optimum lies in a high-smoothing region and does not introduce a
misleading localized perturbation for this mock. The selected control
nonlinear-image difference is 0.01955 noise RMS. This observation is specific
to the recorded signal/noise and initialization, not a false-positive rate.

## Independent amplitude sensitivity and bounds

Each amplitude value was held fixed throughout its scans and two searches.
The source box, macro, gradient, discretization, partition and noise were
identical. Evidence differences between these physical priors were recorded,
without optimizing or rescaling lambda0 inside a run.

| Operator-selected subhalo | lambda0=100 | lambda0=1000 | lambda0=10000 |
| --- | ---: | ---: | ---: |
| Source relative L2 error | 0.10374 | 0.08673 | 0.08206 |
| Raw fitted potential RMS | 0.003605 | 0.002117 | 0.001342 |
| Diagnostic affine potential RMSE | 0.002318 | 0.001665 | 0.001523 |
| Whole-core kappa RMSE | 0.038654 | 0.031803 | 0.030773 |
| Local kappa RMSE | 0.059329 | 0.058123 | 0.059554 |
| Local mean kappa | 0.032586 | 0.036928 | 0.032352 |
| Peak kappa | 0.274464 | 0.290587 | 0.306976 |
| Complete-evidence selection difference | 0.95113 | 0.40387 | 0.14592 |

The adopted 10000 retains the localized feature and stable source/local error,
improves whole-core/control behavior and reduces determinant-induced selection
changes. Its amplitude penalty is 16.8 versus chi2 about 2117, but the decision
uses the field/evidence response above. Weakness is limited to this localized
reconstruction interpretation: the raw potential changes substantially and
one decade weaker prior changes local mean kappa by about 14%. The data do not
fix all low-order modes. Peak suppression and prior sensitivity prevent a
precise mass or full-potential interpretation.

The source optimum is interior to [1e-5,1e-3]. Subhalo D4 optima are interior
to [1e5,1e11]. Dense controls approach the upper D4 bound and remain flagged.
At selected control source strength, the separate same-geometry probe gives
complete logZ 6916.94702/6917.10923/6917.12024 for D4 strengths 1e11/1e12/1e13.
Thus extending 1e11 to 1e13 gains 0.17322 log units while whole-core control
kappa RMS changes 0.0024746 to 0.0024888 (under 0.6%). The finite upper bound
is a practical plateau truncation, not an enclosed finite global optimum.
The block evidence has its different interior control maximum near 1e10.

All 12 DIRECT searches retained budget termination (status 1), with 201–213
actual evaluations for approximate maxfun 200. The total is 2452 successful
inner evaluations; no inner failure was rewarded. These are bounded best
candidates, not claims of optimizer convergence or proven global maxima.
Independent dense determinant-reference searches also reproduce the lambda0=1000 choices. Rejected smaller-block layouts remain in the exploration record;
finite evidence alone did not qualify them for adoption.

## Numerical implementation and verification

Across 90 matched scan pairs, maximum source/potential/convergence differences
are 4.18e-8/2.13e-8/1.33e-9; maximum actual system relative residual is 3.50e-10.
All six reversed-order checks pass (max field difference 7.20e-10, objective
3.64e-12). Deliberate one-iteration PCG failures return an invalid objective
and retain `pcg_iteration_limit`. The callable and all-invalid optimizer tests
also verify failure propagation. This supports rtol 1e-12/maxiter 20000 with
actual acceptance 1e-8; no lambda0 factorization rescue is used.

The expensive triangular-vector preconditioner application was replaced by
cached inverse lower factors and two highest-precision bounded matvecs.
The same principal SPD preconditioner, original Cholesky/log determinants,
complete prior and full coupled curvature are retained. The legacy three-arg
constructor still works; source–lens-light Schur paths are unchanged. Extra
storage is O(N*b), with no full cross-field or joint inverse. The
[12 actual extreme-strength comparisons](../../.scratch/potential-correction/research/inverse-factor-fixed-strengths.json)
include both cases, all three amplitude values and lambda4=1e11 with weak
source regularization. All converge and match dense. The
[interruption record](../../.scratch/potential-correction/research/interrupted-calibration-searches.json)
labels the old unfinished searches; they produced no accepted run.

The [isolated synthetic application benchmark](../../.scratch/potential-correction/research/block-application-benchmark.json)
measures 9 well-conditioned 834-wide float64 blocks: median 3.038 ms triangular
versus 0.14045 ms cached application, with 11.90 ms factor preparation and 3.4e-16
relative difference. This is an application-kernel result, not a whole-workflow
speed claim. The bounded allocation regression passes at 13,099,264 JAX peak
buffer bytes for 18,465 coefficients/cap8, a storage stress case rather than
an astrophysical configuration; CUDA context/driver memory is excluded.

Commands use `tinylens_gpu` and `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
From `tests/`: curvature/operator/inference 39 passed; dense/rendering/shared
solver/autodiff 62 passed; selected legacy PCG/Schur 23 passed. The exact file
commands are recorded in the ticket. In separate processes from
`examples/tests/`, the four relevant example files passed 16; the final
initialization-metadata regression passed both layouts. From `benchmarks/`,
the application and bounded-allocation checks each passed. New tests first
failed at the missing public behavior before implementation. Whole-feature
scientific plots, 200x200 exports, float32 scientific accuracy and workflow
performance remain in tickets 11–14.
