# Dense three-strength Float32 Nautilus posterior and median reconstruction

On 2026-11-01, the existing 200×200 seed-17 scientific subhalo mock was fit
with the single-step Dense algorithm, jointly sampling lambda_s, lambda4
and lambda0 with Nautilus. The requested Float32 experiment retains Float64
numerical checks and fallback for invalid Float32 likelihood evaluations.
Library code and default settings are unchanged.

The [research entry point](../../.scratch/potential-correction/research/three-strength-float32-nautilus/README.md)
contains the numerical target, precision policy, scripts and reproduction
commands. [Compact records and weighted samples](../../.scratch/potential-correction/research/three-strength-float32-nautilus/)
are retained alongside it. Working outputs are under
`.scratch/potential-correction/output/three-strength-nautilus-float32-20261101/subhalo-nlive300/`.

## Model and sampling

The macro is fixed at the true SIS plus shear. Source gradients are frozen
analytic truth gradients at macro rays; source brightness and potential
coefficients remain signed unknowns, analytically marginalized in the inner
conditional Gaussian evidence. Source mesh is 49×49 in [-0.4,0.4]²; potential
spacing is 0.075 arcsec, with 956 fitted support and 724 derivative-valid core
nodes. There are 22016 active image pixels and 88064 nsub=2 subpixel queries.
Noise, PSF and observation match the preceding
[single-step precision comparison](potential-correction-single-step-float32.md).

Independent bounded log-uniform hyperpriors are specified as
log10(lambda_s) in [-8,-1], log10(lambda4) in [1,11], and log10(lambda0)
in [0,8]. The potential prior is lambda4 D4ᵀD4 + lambda0 I. Thus **lambda0
is sampled**, rather than retained at the earlier fixed value of 10000.
The cached likelihood still evaluates complete Dense Cholesky determinants
and actual image residuals. No operator/block determinant is used.

Nautilus uses 300 live points, batch size 8, seed 271828, f_live=0.01 and
target effective size 5000, with exploration points discarded. It completes
after **12408 likelihood calls**, retaining **5496 weighted posterior points**
with effective sample size **5002.18**. One progress-serialization interruption
was corrected and resumed from the persisted sampler/RNG state; settings and
the numerical target remained fixed.

Float32 candidates use the explicitly declared residual gate 1e-5; failed
Float32 evaluations are recalculated in Float64 with gate 1e-8. There are
**696 fallback calls (5.61%)**, and no candidate remains invalid after fallback.
The largest fallback log evidence is 69456.28, over 73 below the largest
returned Float32 score, so fallback evaluations are concentrated outside
the posterior's high-density region. This is a Float32-primary mixed-precision
likelihood, not a pure-Float32 wide-prior validity claim.

## Weighted posterior

All summaries use normalized importance weights, without equal-weight
resampling. Intervals are marginal 16th–84th percentiles.

| Strength | Median | 68% marginal interval |
| --- | ---: | ---: |
| lambda_s | 7.77463e-5 | [6.91653e-5, 8.71861e-5] |
| lambda4 | 6.72880e5 | [5.62194e5, 8.04322e5] |
| lambda0 | 3.04549e4 | [2.31234e4, 3.90745e4] |

The corner plot includes all three marginals and three pairwise distributions,
with 68% and 95% joint-probability contours. Orange lines mark the medians;
titles give median [16%,84%] in log10 coordinates. Display limits show central
99.9% marginal probability; all samples remain in the numerical summaries.

Weighted Pearson log-coordinate correlations are lambda0/lambda4 -0.2150,
lambda0/lambda_s -0.0604, and lambda4/lambda_s +0.0393. The two potential
strengths show a mild anticorrelation; these three hyperparameters do not
show a strong narrow degeneracy in this fixed model. This statement does
not exclude degeneracies among source/potential fields or uncertain macro
models. The medians agree closely with the preceding independent
[Float64 posterior](../../.scratch/potential-correction/research/three-strength-nautilus/README.md),
whose lambda0/lambda4/lambda_s medians were approximately
29998 / 673493 / 7.79147e-5. A single new sampler seed does not establish
independent-run convergence.

All 5496 retained points were recomputed with the Float64 Dense likelihood.
The posterior-weighted mean and RMS Float32-minus-Float64 log-evidence
differences are +0.01320 and 0.02847. The **largest absolute difference is
23.529**, so Float32 evidence is not uniformly accurate across retained tails.
From the weighted RMS and this maximum, the original posterior weight of
that worst point is bounded above by 1.47e-6; this bound does not identify
its coordinate or establish uniform tail coverage.

Importance reweighting of the same points to Float64 changes the normalized
discrete distribution by total variation 0.009991 (about 1.00%). The
reweighted effective size is 5007.79; median strengths become
lambda_s=7.77500e-5, lambda4=672879.23 and lambda0=30449.50. Their largest
relative median shift is about 0.0178%. These checks support the reported
central posterior for this mock while retaining the tail accuracy limitation.
They are importance checks on the sampled support, not a new independent
Float64 sampler run or a guarantee about unsampled regions.

## Conditional reconstruction at the three marginal medians

The public Dense fitter solves the source and potential at the median
strengths above. This is a conditional reconstruction, **not** a
maximum-evidence fit, posterior-mean map or map uncertainty interval.

| Diagnostic | Float32 median fit |
| --- | ---: |
| Recovered peak position (arcsec) | (0.75,0.075) |
| Distance to injected position (arcsec) | 0.025 |
| Recovered peak κ / discrete truth peak | 0.480494 / 0.631435 |
| Recovered peak fraction | 76.1% |
| Local-core κ RMSE / zero-field truth RMS | 0.058111 / 0.099169 |
| All-core κ RMSE / zero-field truth RMS | 0.037157 / 0.030863 |
| Source relative L2 error to truth | 7.67% |
| Linear residual RMS / noise | 0.992206 |
| Nonlinear re-rendered residual RMS / noise | 1.016589 |
| True relative linear-system residual | 5.72e-7 |

The localized peak increases from the earlier fixed-strength reference's
about 0.290 (46% of truth) to 0.480 (76%). However, visible ring-like background
structures remain, and **all-core κ RMSE exceeds the zero-field baseline**.
Higher conditional evidence and a stronger local peak therefore do not imply
uniformly accurate κ recovery. No mass measurement or detection significance
is inferred. Signed negative source coefficients remain unclipped.

At the **same new median strengths**, an independent public Float64 fit gives
peak κ=0.481881 at the same location. Float32-minus-Float64 peak change is
about -0.288%; whole-core κ relative L2 difference is 2.92%, raw potential
relative L2 difference is 3.24%, and source relative L2 difference is 0.210%.
The public-fit log-evidence difference is +0.0780. The ring-like background
structure is present in Float64 too; it cannot be attributed entirely to
Float32 rounding. Float32/Float64 illumination-threshold node counts can
differ slightly, so the quoted cross-precision κ comparison uses the identical
724-node derivative-valid core, not separately thresholded illuminated sets.

## Figures and validation

The output directory contains `posterior_corner`, `posterior_median_recovery`
(observed/fit/residual, true/recovered/error source, true/recovered/error κ),
and `subhalo_median_recovery` (the three κ panels) in PNG, editable PDF and SVG.
Truth and reconstruction share intensity/κ scales; error maps use their own
labelled symmetric scales. Black crosses indicate the injected subhalo center.
No physical coefficients are clipped or affine-aligned, and undefined
support-only derivatives are masked.

All three figures pass final rendered plot-area alignment checks at 1.5 pt,
PDF text floors (minimum 8–8.4 pt), collision audits (zero failures/warnings),
and visual inspection. The larger diagnostic canvases and 300-dpi PNG exports
are intentional; these are not journal-specific submission files. Static
preflight has no failures; its TIFF/600-dpi/journal-width warnings are outside
this diagnostic export contract. The numerical audit re-evaluates every
retained posterior point in Float64 and reports reweighted medians separately.

The focused existing single-step precision regressions pass (2 tests;
2 iterative cases deselected). Black and Ruff pass for both new experiment
scripts, and `git diff --check` passes. The complete sampler run, Float64
point/median checks, public result exports and three figure render/audit
paths provide the experiment's numerical and delivery validation.
