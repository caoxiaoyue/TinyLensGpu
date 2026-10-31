# Nautilus posterior for three regularization strengths

User-requested 2026-10-31 follow-up to the
[three-strength point optimization](../three-strength-single-step/README.md).
The objective here is the joint hyperparameter posterior and its degeneracies,
not another maximization of the conditional evidence.

## Target and fixed model

Use the same seed-17 scientific **subhalo** image, 200 x 200, source49,
potential spacing 0.075 arcsec, fixed true macro and frozen analytic truth
source gradient. There are 22,016 active data, 2,401 source coefficients,
and 956 potential coefficients. The source prior remains second order;
potential precision is lambda4 D4^T D4 + lambda0 I on all potential unknowns.
The signed source and potential coefficients are analytically integrated out
using the full dense conditional Gaussian evidence for each hyperparameter
candidate. This is conditional on the frozen single-step response; it does
not integrate over the macro, source-gradient uncertainty or linearization
error of the nonlinear simulated image.

The sampled coordinates, in storage order, are:

| Coordinate | Independent hyperprior |
| --- | --- |
| log10(lambda_source) | Uniform(-8,-1) |
| log10(lambda_potential / lambda4) | Uniform(1,11) |
| log10(lambda_amplitude / lambda0) | Uniform(0,8) |

These ranges match the previous optimization. Thus the target density in
log-strength coordinates is proportional to the conditional evidence. The
priors are proper, bounded log-uniform priors in linear-strength coordinates.
No optimization-derived narrowing or initial posterior cloud is supplied.
Previous evidence maxima are plotted as reference lines, not injected truths
for the hyperparameters.

## Sampler and numerical checks

Use installed nautilus-sampler 1.0.6 directly, with **n_live=300 for the whole
three-dimensional problem**, sampler seed 314159, vectorized batches of 8,
f_live=0.01 and target n_eff=5000. Exploration points are discarded before
the final importance-sampling estimate (`discard_exploration=True`). Every
40-second run interval writes a resumable HDF5 checkpoint and progress JSON.
The local installed API and
[official sampler documentation](https://nautilus-sampler.readthedocs.io/en/stable/api_high.html)
define these settings and the distinction between live points and effective
posterior sample size.

To avoid recomputing the fixed data Gram matrix for every likelihood call,
the experiment caches A^T C_n^-1 A and A^T C_n^-1 d. Every candidate still
builds its complete prior, computes full dense Cholesky determinants of Q
and H, solves for the joint fields, evaluates the actual image residual, and
uses the unchanged relative residual gate 1e-8. No block determinant,
eigenvalue approximation or surrogate evidence replaces the Gaussian formula.

Eleven public-fitter comparisons include the original settings, the previous
mode, six displaced points, and three wide-prior numerical checks. The
largest log-evidence discrepancy is 7.93e-6 at an ill-conditioned remote
point; checks near the posterior mode agree to about 1e-9 or better.
Warm batches of 8 take about 0.288 s (27.8 candidates/s) on the RTX 5080
in float64. This is reuse of invariant data products, not a change to the
physical model. Exact warm timings and validation coordinates are retained.

## Posterior diagnostics

All summaries and contour densities use normalized importance weights, not
an equal-weight resampling. Report weighted Pearson correlations in both
log and linear coordinates, weighted Spearman correlations, partial log
correlations, and eigenvalues of the standardized correlation matrix.
The latter distinguishes a thin correlation direction from mere differences
in parameter units. Alternating sample-index subsets provide a consistency
check, not an independent-run convergence test.

The triangle plot is ordered lambda0, lambda4, lambda_s and displays central
99.9% marginal probability ranges so the broad prior tails do not hide the
posterior. All samples remain in the numerical summaries. The 2D contours
enclose 68% and 95% joint probability; marginal intervals are weighted
16th/50th/84th percentiles. Histogram smoothing is 0.8 bins and is applied
only to the plot, not to reported moments or quantiles.

## Result: weak correlation, no strong hyperparameter degeneracy found

The run met its stopping conditions with **300 live points, 12,520 likelihood
calls, 5,688 retained posterior points and weighted effective sample size
5005.51**. All likelihood calls pass the original numerical acceptance gate;
the maximum relative linear residual across all calls is 1.52e-11.
Exploration is complete and discarded in the saved final posterior. Two
progress-serialization issues were fixed during the run; the sampler resumed
its persisted HDF5 state and RNG without changing the target or settings.
These were reporting interruptions, not failed likelihood evaluations.

| Strength | Posterior median | 16th–84th percentile interval |
| --- | ---: | --- |
| lambda0 | 2.99976e4 | [2.28010e4, 3.89718e4] |
| lambda4 | 6.73493e5 | [5.62081e5, 8.03045e5] |
| lambda_source | 7.79147e-5 | [6.92377e-5, 8.71676e-5] |

The preceding evidence optimum lies inside every marginal 68% interval.
All three marginals are well away from the broad prior boundaries. The
posterior samples give the following weighted correlations:

| Pair | Pearson, log10 coordinates | Weighted Spearman |
| --- | ---: | ---: |
| lambda0 / lambda4 | -0.1848 | -0.1860 |
| lambda0 / lambda_source | -0.0434 | -0.0419 |
| lambda4 / lambda_source | +0.0273 | +0.0216 |

Lambda0 and lambda4 show a mild trade-off, with larger amplitude precision
slightly favoring weaker D4 smoothing. Neither source-strength pair exhibits
substantial correlation. Partial correlations are also small, and the
standardized correlation matrix has condition number **1.470**, supporting
the absence of a nearly singular three-parameter correlation direction.
The weighted corner contours show a single central concentration without
a long narrow or strongly curved degeneracy band. Small contour irregularities
are finite-sample histogram structure; the figure is not a fitted Gaussian.

This conclusion applies to the stated fixed-macro, frozen-gradient, dense
single-step subhalo experiment and its declared hyperpriors. Weak correlations
of these three strengths do not establish absence of degeneracies between
the underlying source/potential fields, or removal of mass-sheet degeneracy.
One sampler seed was used; alternating-index summaries are consistency
diagnostics rather than independent-run uncertainty estimates.

Nautilus reports hyperprior-integrated log evidence 69518.3624, which includes
integration over the declared hyperprior volume. This is distinct from the
maximum conditional log evidence 69529.6343 in the preceding optimization.

Retained numerical products:

- [Weighted posterior samples](posterior.npz).
- [Quantiles, full correlations, partial correlations and PCA](posterior-summary.json).
- [Completion state](completion.json) and [validation/trace audit](verification.json).
- [Manifest and hyperpriors](manifest.json) and [public-fitter comparisons](validation.json).

Working figures:

- [Corner PNG](../../output/three-strength-nautilus/seed17/subhalo-nlive300/posterior_corner.png)
  and [PDF](../../output/three-strength-nautilus/seed17/subhalo-nlive300/posterior_corner.pdf).
- [Correlation PNG](../../output/three-strength-nautilus/seed17/subhalo-nlive300/posterior_correlation.png)
  and [PDF](../../output/three-strength-nautilus/seed17/subhalo-nlive300/posterior_correlation.pdf).

## Reproduction

From the repository root:

```bash
source ~/anaconda3/bin/activate && conda activate tinylens_gpu
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_COMPILATION_CACHE=false \
  python .scratch/potential-correction/experiments/sample_single_step_three_strengths.py \
  --output .scratch/potential-correction/output/three-strength-nautilus/seed17/subhalo-nlive300 \
  --n-live 300 --n-eff 5000 --batch-size 8
python .scratch/potential-correction/experiments/plot_three_strength_posterior.py \
  .scratch/potential-correction/output/three-strength-nautilus/seed17/subhalo-nlive300
```

Use a new output directory for a fresh run, or add `--resume` with matching
settings to continue its checkpoint. Numerical artifacts include the
checkpoint, every likelihood coordinate/score/residual, weighted posterior
samples, completion state, numerical validation, corner PNG/PDF, correlation
PNG/PDF and a machine-readable posterior summary. Compact records and the
final weighted posterior are retained beside this note; working checkpoints
and figures remain in the ignored output directory. The experiment does not
change the public fixed-amplitude interface or any default configuration.
