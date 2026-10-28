# Small single-step backend calibration

**Status: in progress.** This record preserves the physical-resolution and
block-layout checks completed before the full public operator searches and
amplitude-prior sensitivity runs finish. Ticket 07 remains claimed; finite
evidence and the preliminary comparisons below do not resolve that gate.

The runnable workflow is
[`calibrate_single_step.py`](../../examples/potential_correction/calibrate_single_step.py),
with the complete
[physical/numerical configuration](../../examples/potential_correction/calibration_configuration.json).
It writes inputs, raw selected fields and scalar scan/search records. The
[exploration summary](../../.scratch/potential-correction/research/calibration-exploration-summary.json)
retains earlier models/configurations and rejected layouts. The
[selected dense reference](../../.scratch/potential-correction/research/selected-dense-calibration-reference.json)
contains the recovered-field checks associated with the independently
selected complete and approximate candidates below. Those approximate
candidates use a dense reference solve, not an operator solve.

## Physical and discrete problem

The chosen small case has a 64x64 image, 0.0375 arcsec pixels, nsub=2,
noise sigma 0.01 and seed 17. Subhalo and control share the same noise draw.
The fixed true macro is SIS theta_E=0.8 with shear (0.05,0.02). The injected
singular Pseudo-Jaffe has b=0.04 at (0.75,0.1) arcsec and
rt=sqrt(0.04*0.8)=0.178885 arcsec. This stronger, explicitly configured
perturbation is a small calibration case; it does not replace the default
200x200/b=0.01 scientific mock or establish calibrated mass precision.

Source widths are (0.16,0.12) arcsec, with the documented two-Gaussian clump
model and clump_scale=0.7. The fixed 49x49 source spans [-0.4,0.4]^2 with
0.016667 arcsec spacing. The Gaussian PSF has sigma 0.05 arcsec and size5,
and retains the existing native circular FFT/mask semantics. The annulus is
0.45<=R<=1.1 arcsec. The potential spacing is 0.075 arcsec. Counts are
2252 data, 2401 source, 704 core and 936 potential unknowns, 3337 total.
No source bounds are expanded and no observation pixels are eroded.

The noiseless analytic mock is rendered independently from the sampled truth
fields. Sampling its source onto this grid gives 0.11168 noise RMS mismatch
in the control. Sampling both source and potential gives 0.27081 noise RMS
mismatch in the subhalo image. The conditional-linear versus nonlinear
sampled-truth discrepancy is 0.20093 noise RMS; the frozen source gradient is
analytic, so this is not solely the bilinear source's Taylor remainder.
The workflow separately computes the analytic-source Taylor remainder and
source-interpolation/potential-discretization contributions.

Potential truth is analytic nodal psi with psi(subhalo center)=0. Its
constant and slope are not imposed on the fit. Raw fitted fields remain
unchanged. All-core convergence uses the same centered physical differences
on recovered and analytic nodal potential. Analytic point convergence is also
reported, with R<1e-8 excluded and no invented core or pixel averaging.
All derivative-valid core nodes remain in the whole-core errors. Two additional
comparison domains were fixed before the fit: a local disk of radius
max(2*rt,2*h), and core nodes inside the observation annulus with analytic
macro-source brightness >=5% of its core maximum. The control uses the same
local disk. These diagnostic domains do not replace derivative_valid.

## Layout comparison at independently fixed lambda0=1000

The independent dense diagnostic retains the complete coupled solve while
changing only principal-block determinant terms. It uses the complete
amplitude-augmented prior in both terms. The exact conditional evidence
selected log10(lambda_s,lambda4)=(-4.24691,6.39506), logZ=6741.70581.
Small incidence tiles and some field-separated layouts give worse control
structure and remain rejected; the retained exploration records include them.

The current large incidence candidate uses source tiles24 and cap1024.
Sparse source-query/potential-stencil votes assign potential unknowns to source
tiles, unilluminated support nodes attach spatially, and fields interleave
before bounded splitting. It uses no dense cross matrix. Its actual block
sizes are (787,834,24,814,805,24,24,24,1). It selects
(-4.20576,6.32099), with complete logZ=6741.30194, a difference of 0.40387
relative to the complete search selection. DIRECT budget termination remains
visible; no global-maximization proof is claimed.

| Physical subhalo, dense solves at selected strengths | Complete selection | Block selection |
| --- | ---: | ---: |
| Source relative L2 error | 0.08542 | 0.08673 |
| Potential RMSE, truth constant mean-centered only | 0.003639 | 0.003696 |
| Recovered kappa peak position (arcsec) | (0.75,0.075) | (0.75,0.075) |
| Distance to injected position (arcsec) | 0.025 | 0.025 |
| Peak kappa (discrete truth 0.63144) | 0.27976 | 0.29059 |
| Whole-core kappa RMSE (truth RMS 0.03130) | 0.03071 | 0.03180 |
| Local kappa RMSE (truth RMS 0.09917) | 0.05852 | 0.05812 |
| Local mean kappa (truth 0.04340) | 0.03676 | 0.03693 |
| Linear image residual RMS/noise | 0.97287 | 0.97197 |
| Nonlinear image residual RMS/noise | 1.00468 | 1.00332 |

Raw center-zero-truth potential errors and separately named affine diagnostic
alignment are retained in JSON. The mean-centered truth row above only fixes
a truth additive constant; it does not center, align or change fitted p.
The whole-core error, peak attenuation, negative/low-order structure and
mean potential remain important limits. Local recovery does not establish
complete potential, deflection, convergence or subhalo-mass recovery.

For the control, the complete search approaches the upper lambda4 bound
(log10=10.98765), whereas this block candidate selects 10.0. Complete logZ
difference is 1.79215. Source relative errors are 0.06805/0.06838;
whole-core kappa RMS is 0.00271/0.00331, local RMS 0.00138/0.00188 and peak
0.00576/0.00685. Thus the control correction is small compared with the
injected peak but is nonzero. Its changed smoothing optimum must be assessed
with these field differences, not excused solely as approximate evidence.

## Solver and workflow checks completed

The public operator/DIRECT test plus existing operator cases passed32.
It independently constructs the full coupled solution and principal prior/
posterior determinants at the selected point, and checks that an all-failed
PCG search returns no valid selected fit. The example driver first failed at
its missing public seam, then passed; the missing incidence-layout test also
failed before implementation. The final separate example check passed16,
including both layouts, both physical observations, independent fixed
amplitude strengths, raw field/domain exports, reversed candidates, failed
PCG records and DIRECT budget/boundary metadata. These tiny test grids check
assembly and provenance rather than scientific recovery.

A physical fixed-strength cap128 preflight agreed with dense source/potential/
convergence/images at roughly1e-10 or better through lambda4=1e9. At
lambda4=1e11 it needed about6800 PCG iterations and a recomputed actual
relative residual around2e-10. In one representative subhalo fit, all9008
corrected subpixel rays remained inside the fixed source box, with minimum
signed margin0.0275 arcsec. Full-run records retain actual macro, truth and
fit margins. The candidate numerical controls are PCG maxiter20000,
rtol1e-12, atol0 and actual full-system solve tolerance1e-8.

The cap1024 padded triangular preconditioner application made full searches
expensive. The unfinished workers were stopped after their configurations, empty logs
and explicit lack of completed runs were retained in the
[interruption record](../../.scratch/potential-correction/research/interrupted-calibration-searches.json).
A cached inverse lower factor now applies the same principal SPD preconditioner
through two highest-precision bounded matvecs per PCG step. Original Cholesky
factors, determinant terms, prior and curvature are retained; the legacy
three-argument preconditioner still uses triangular solves. No full inverse,
new determinant method or physical strength rescue is introduced.

The new padded-SPD public operator contract first failed for the missing
optional inverse factor and then passed in float64 and float32. The combined
operator/inference/curvature check passed39. The
[physical fixed-strength record](../../.scratch/potential-correction/research/inverse-factor-fixed-strengths.json)
contains12 dense/operator pairs: both cases, all three independent amplitude
values, a representative selected region and the weak-source/high-smoothing
endpoint (lambda_s=1e-5,lambda4=1e11). All converged. Max absolute differences
were2.93e-8 in source,8.07e-9 in potential,2.53e-10 in convergence,8.8e-11 in
linear image and2.91e-8 in nonlinear image. Maximum recomputed relative system
residual was2.57e-10. These actual extreme-strength checks establish numerical
agreement, not complete scientific recovery.

The isolated production-code
[benchmark](../../.scratch/potential-correction/research/block-application-benchmark.json)
uses9 padded834-wide synthetic well-conditioned float64 blocks. Median apply
time was3.038ms for triangular solves and0.14045ms for cached factors, with
11.90ms inverse-factor preparation per candidate and3.4e-16 relative numerical
difference. This is an application-kernel comparison, not a22x workflow or
ill-conditioned accuracy claim. The allocation regression still passed at
13,099,264 JAX peak buffer bytes for18,465 coefficients and cap8; inverse
factors remain bounded block arrays.

## Remaining gate

Complete the actual public operator scans/DIRECT selections for independently
fixed lambda0 in {100,1000,10000}, compare matched-strength solver errors and
selected recovered fields, retain raw products and all failures/boundaries,
and assess amplitude-prior influence on both evidence and recovery. Only then
adopt a demonstration amplitude precision, bounds and solve tolerances and
resolve ticket07 if no materially misleading approximation remains. New
scientific geometries, source grids and initializations need their own check;
this small case does not validate every partition at 200x200.
