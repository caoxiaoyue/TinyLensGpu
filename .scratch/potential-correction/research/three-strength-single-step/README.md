# Three-strength single-step evidence experiment

This 2026-10-31 experiment follows the user's request to explore joint
selection of source regularization, fourth-difference potential smoothing,
and the independent potential-amplitude precision. It deliberately relaxes
the fixed-amplitude assumption **for this experiment only**; the public
interfaces, scientific configuration, and first-release defaults are unchanged.

## Model and selection

The unchanged scientific mock is 200 x 200 at 0.012 arcsec/pixel, seed 17,
noise sigma 0.01, mock/inversion nsub=2, source49 on [-0.4,0.4]^2, potential
spacing 0.075 arcsec, and Pseudo-Jaffe b_sub=0.04 at (0.75,0.1). There are
22,016 data, 2,401 source coefficients, 956 potential coefficients, and
724 derivative-valid core nodes. Macro parameters are fixed to truth;
single-step source gradients are the same frozen analytic truth gradients
as in the original demonstration. Source regularization is second order.

Every candidate calls the public **dense** single-step fitter in float64
on the GPU. Its evidence is the exact Gaussian integral conditional on the
fixed linear response and declared prior. This does not make the linearized
model exact for the nonlinear simulated image.

The objective uses evidence alone. Truth recovery, peak location and
background errors are calculated only after selections are frozen. The
search is empirical Bayes, not posterior sampling of the hyperparameters.
The potential precision includes both terms, with their full normalization:

Q_psi = lambda4 D4^T D4 + lambda0 I.

Search intervals in log10 coordinates are:

| Strength | Minimum | Maximum |
| --- | ---: | ---: |
| lambda_source | 1e-8 | 1e-1 |
| lambda_potential / lambda4 | 1e1 | 1e11 |
| lambda_amplitude / lambda0 | 1 | 1e8 |

DIRECT provides a budget-limited global exploration (maxfun=300), followed
by bounded finite-difference L-BFGS-B refinements from different starting
points. A separate two-dimensional DIRECT/local search reoptimizes source
and D4 at lambda0=10000, making the comparison isolate the added freedom.
Conditional one-dimensional evidence slices keep the other strengths fixed;
they are not profile likelihoods or credible intervals. Search statuses and
failed candidates are retained rather than relabelled as convergence.

The experimental adapter creates a separate shallow copy for each positive
amplitude value, sharing only strength-independent dense arrays and geometry.
An independent freshly constructed problem at lambda0=30000 agrees exactly
in evidence with this sharing strategy. No public fixed-amplitude problem
is mutated during an evaluation.

## Subhalo results

| Quantity | Original hand selection | Optimize source/D4, fixed lambda0 | Optimize all three |
| --- | ---: | ---: | ---: |
| lambda_source | 6e-4 | 7.90198e-5 | 7.80471e-5 |
| lambda4 | 1.6e7 | 7.57719e5 | 6.74860e5 |
| lambda0 | 10000 | 10000 | 30718.97 |
| log evidence | 68989.984461 | 69523.102346 | 69529.634274 |
| Recovery at the nodal truth peak | 45.9495% | 74.6349% | 76.2898% |
| Local kappa RMSE | 0.0608898 | 0.0557225 | 0.0579883 |
| Whole-core kappa RMSE | 0.0276357 | 0.0350476 | 0.0367015 |
| Background kappa RMS | 0.0211016 | 0.0318753 | 0.0333989 |
| Source relative L2 | 0.0719018 | 0.0767799 | 0.0755265 |
| Linear residual RMS / noise | 1.003494 | 0.992273 | 0.992215 |
| Nonlinear rerender residual RMS / noise | 1.029664 | 1.020751 | 1.016473 |

The joint optimum gains **539.649813** in log evidence over the original
fixed values, and **6.531928** over reoptimized source/D4 with lambda0 fixed.
All three selected strengths are more than three dex from the nearest search
edge. Independent local refinements reproduce essentially the same point.
DIRECT itself exhausted its exploration budget; this is a stable interior
optimum found by the declared searches, not a proof of the global maximum.

The 709 unique search/slice candidates all pass the original solve_rtol=1e-8
gate, with maximum relative linear residual about 1.59e-12. Original inputs
are bitwise identical to the saved scientific reference; baseline log
evidence differs by less than 1e-8. A final independent repeat agrees at
roundoff. An export-path error was repaired and the completed search was
recovered from its persisted candidate cache; it was not a numerical fit
failure. Final records retain the pre-resume source hashes.

Independent local-curvature checks with steps 0.01 and 0.02 dex give positive
eigenvalues of the negative-log-evidence Hessian: approximately 71.3, 177.1
and 394.6. The two step sizes agree to better than 0.01% in these eigenvalues.
This supports an isolated local maximum rather than a flat local ridge.
All 37 additional audit evaluations succeed. The constant-potential unit
mode has exactly zero D4 response and maximum image response 4.44e-16,
consistent with the expected unobservable gauge. Its normalized prior
integral contributes no information for choosing lambda0 in exact evidence;
the selected amplitude is constrained through other modes.

Evidence optimization recovers a less suppressed local peak, but it does
**not** minimize every truth-error metric. Whole-core and background errors
increase relative to the original hand selection. Freeing lambda0 adds only
about 1.65 percentage points of peak recovery over reoptimized source/D4,
and slightly worsens local kappa RMSE. These distinctions matter when
interpreting what the evidence objective actually selects.

## Matched no-subhalo control

The same search bounds, budgets, objective and source/potential geometry
were used for the matched control image. All 773 unique candidates succeed;
the largest relative linear residual is 2.02e-11, below the unchanged 1e-8
acceptance gate. An initial process received SIGTERM after 207 saved candidates;
the run resumed from its candidate cache without relaxing numerical controls.

| Quantity | Original hand selection | Optimize source/D4, fixed lambda0 | Optimize all three |
| --- | ---: | ---: | ---: |
| lambda_source | 6e-4 | 9.30641e-5 | 9.43371e-5 |
| lambda4 | 1e11 | 1e11 | 1e11 |
| lambda0 | 10000 | 10000 | 1e8 |
| log evidence | 69486.832426 | 69778.804430 | 69808.978859 |
| Whole-core kappa RMS | 4.30168e-4 | 6.42317e-4 | 7.00102e-5 |
| Maximum core kappa | 1.27649e-3 | 1.77891e-3 | 1.59599e-4 |

Both potential strengths select the upper search boundary. These values
are **bounded-search selections, not identified finite interior optima**;
the search does not determine whether or where a maximum occurs beyond
these bounds. Within the tested range, the evidence favors stronger
suppression of a correction field that has no injected perturber. The
joint-selected control kappa RMS is about six times smaller than the original
control and about nine times smaller than the reoptimized fixed-amplitude
control. This one matched realization does not establish a false-positive rate.

In particular, the subhalo and control log evidences are evaluated on
different images and must not be subtracted to report a Bayes factor.

## Reproduction and artifacts

Activate tinylens_gpu and run from the repository root:

```bash
source ~/anaconda3/bin/activate && conda activate tinylens_gpu
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_COMPILATION_CACHE=false \
  python .scratch/potential-correction/experiments/optimize_single_step_three_strengths.py \
  --case subhalo --output .scratch/potential-correction/output/three-strength-single-step/seed17/subhalo
python .scratch/potential-correction/experiments/plot_three_strength_single_step.py \
  .scratch/potential-correction/output/three-strength-single-step/seed17/subhalo
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_COMPILATION_CACHE=false \
  python .scratch/potential-correction/experiments/optimize_single_step_three_strengths.py \
  --case control --output .scratch/potential-correction/output/three-strength-single-step/seed17/control
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_COMPILATION_CACHE=false \
  python .scratch/potential-correction/experiments/audit_three_strength_optimum.py \
  .scratch/potential-correction/output/three-strength-single-step/seed17/subhalo
```

Use a new output directory for a fresh experiment; `--resume` reuses a saved
candidate cache after an interrupted run. The original/fixed-amplitude/joint
exports include full raw fields, linear and nonlinear images, noise and
truth diagnostics. Large arrays, diagnostic PNG/PDF files and complete
working logs are under the ignored output directory. Compact JSON records
and candidate traces are retained beside this note.

The convergence comparison uses one shared color scale and the same
derivative-valid core domain, with centered finite differences of nodal
truth. Raw fitted fields are never aligned, clipped or replaced. Optional
affine comparisons remain separately labelled diagnostics. Neither these
fits nor differences between maximized conditional evidences constitute a
subhalo detection significance, a physical mass interval, or proof that
mass-sheet degeneracy has been removed.
