# Fixed-bbox gradients and local optimization for the 20-MGE example

Validation date: 2026-10-25 (Asia/Shanghai).

## Model and objective

This validation reuses the physical model and FITS data from
`examples/pix_src_demo_operator/simple/pix_src_lens_light_mge/fit_joint.py`:
twenty Gaussian lens-light bases, an 80 x 80 first-order-regularized source,
`nsub=4`, 5024 fitted image pixels, and the original position constraint.
The data are the existing seed-42 mock. No light basis is removed and the
physical lens-light regularization remains `1e-6`.

The source bbox is inferred once at the reference mass and then frozen. The
effective stored bbox is approximately
`(-0.3760319352, 0.5948321223, -0.3653272986, 0.6055367589)` arcsec.
Mass parameters, four shared lens-light geometry parameters, and
`log_lambda_reg` remain dynamic: ten nonlinear parameters and 6420 jointly
solved non-negative linear coefficients.

The optimized quantity is the model's existing approximate log evidence,
including its position penalty. This is a local likelihood/evidence maximum,
not a posterior mode: Gaussian log-priors are not added. Bounds preserve the
original parameter limits and uniform regularization-prior support. These
fixed-bbox results should not be compared as the same statistical model as
earlier runs that re-inferred the source grid for every mass proposal.

## Problems exposed by the large case

1. Default float32, 1000-iteration forward solves pass their convergence gate
   but do not supply valid implicit gradients in this case. Float64 alone
   does not fix the insufficient solve accuracy.
2. The original relative complementarity margin `1e-4` is overly conservative
   for some of these bright correlated bases. At the reference point after
   30000 float64 iterations, the KKT residual is approximately `1e-15` and
   active dual residuals are positive, but several lens-light components have
   dual residuals 0.002--0.5 while the relative margin requires 0.6--4.
   The probability model now exposes the existing low-level control as
   `pnpg_comp_rel_tol`; its default remains unchanged.
3. The position constraint included `sqrt(0)` self-distances in its VJP,
   poisoning mass gradients even when the penalty was inactive. The zero
   distance branch now uses a finite square-root argument, preserving forward
   values and retaining NaN propagation for invalid rays. Zero-threshold
   division is also guarded.
4. Bilinear mapping unconditionally cast ray coordinates to float32. Small
   double-precision parameter steps could disappear in finite differences.
   Mapping now preserves input coordinate precision. A regression checks a
   `1e-9` coordinate step against the analytic interpolation derivative.

## Gradient checks

The checks use float64 observations/parameters and the same original FITS
values, `pnpg_comp_rel_tol=1e-10`, and fully resolved positive duals. They
retain KKT, complementarity, and adjoint convergence checks.

Three outputs are checked against central finite differences over all ten
nonlinear parameters: log evidence, a deterministic weighted source-pixel
projection, and a deterministic weighted lens-amplitude projection. Testing
reconstructions as well as evidence prevents the envelope theorem from hiding
a missing solution derivative.

Parameter scales are 0.1 for Einstein radius/ellipticities, 0.05 arcsec for
centers, and 1 for log regularization. Finite differences perturb coordinate
`i` by `h * scale[i]`. Per-component criteria, specified in the harness, use
2% relative tolerance and absolute tolerances of `1e-2`, `1e-6`, and `1e-5`
for the scaled derivatives of the three outputs respectively.

At `h=1e-4` all thirty components pass. A 30000-iteration step sweep gave:

| h | Evidence relative L2 error | Source projection relative L2 error | Lens projection relative L2 error |
|---:|---:|---:|---:|
| 1e-4 | 0.000311 | 0.00992 | 0.0000460 |
| 3e-5 | 0.000594 | 0.0101 | 0.0000444 |
| 1e-5 | 0.00129 | 0.00270 | 0.0000216 |

The smaller steps do not pass every individual component: one small source
derivative misses the 2% criterion at `3e-5`, and one small evidence
derivative misses it at `1e-5`. This step sensitivity is recorded rather than
interpreted as arbitrarily high derivative accuracy. Possible sources include
bilinear support/cell and NNLS active-set boundaries and numerical error in the
ill-conditioned determinant calculation; this sweep alone does not identify
which term causes each discrepancy.

Tightening PCG from 400 iterations/`rtol=1e-5` to 2000/`rtol=1e-10` changes
the analytic Jacobians by at most approximately `3.7e-6`, `2.7e-8`, and
`2.6e-7` respectively. The step-sweep discrepancy is therefore not explained
by the adjoint PCG tolerance. The final optimization uses the tighter setting;
its implicit adjoint can use up to 4000 iterations.

## Optimization validation

A 30000-iteration solve is not sufficient at every point: at the perturbed
optimizer start its free-variable stationarity residual is about `4.3e-4`,
larger than the smallest positive active dual. The wrapper correctly rejects
the gradient. At the same point, 100000 iterations with forward `rtol=1e-11`
reduce the normalized KKT residual to approximately `3e-15` and all three
Jacobian rows become finite.

The full local optimization therefore uses 100000 PNPG iterations,
`rtol=1e-11`, float64, complementarity margin `1e-10`, and the tighter PCG
configuration. The runner can retry a rejected gradient at a larger iteration
budget, recompiling that budget explicitly rather than mutating an already
compiled likelihood. It records every valid evaluation and rejected retry.
The first ten-parameter start perturbs all reference coordinates; log
regularization starts at -7.9. The first run was stopped after four accepted
steps; a continuation restarts L-BFGS-B from its saved best point with no
optimizer history and uses
parameter scales `(0.03, 0.08, 0.08, 0.05, 0.05, 3, 0.03, 0.03, 0.02, 0.02)`
in the reported parameter order. Its exact starting coordinates are saved in
`optimization_start.json`. SciPy L-BFGS-B minimizes negative log evidence
divided by the number of fitted data, with a constant offset that does not
change the optimum. The tolerances are `gtol=1e-6`, `ftol=1e-10`,
`maxls=30`, `maxiter=100`. Objective improvement or an `ftol` termination alone
does not meet this validation's gradient convergence criterion.

A separate check at the fifth accepted continuation step resolves the inner
problem with 300000 iterations: normalized KKT `6.8e-16`, maximum free
stationarity error `2.2e-11`, strict complementarity passes. Freezing this MAP
solution and differentiating the objective explicitly gives the envelope
gradient. It agrees with the full implicit gradient to `2.7e-7` maximum
absolute error over the ten coordinates. Thus the large evidence derivatives
at this point are not caused by a missing PNPG solution derivative.
The lens-center derivatives are about `-29276` and `-32291`; their frozen
finite differences at scaled `h=1e-5` are `-29273` and `-32279`. This diagnostic
supports the implicit derivative at this point, but does not certify the
entire optimization trajectory or arbitrary finite-difference step sizes.
Stopping only the approximate curvature-logdet derivative in this frozen
diagnostic changes those center derivatives to `+805` and `+3919`:
the logdet contributions are approximately `-30081` and `-36210`.
This identifies the dominant contribution to the steep center gradient at
this point; it does not establish the cause of every line-search difficulty.
The actual optimization retains the original objective and all its terms.

**The complete ten-parameter optimization did not pass.** The continuation
accepted 22 L-BFGS-B steps and recorded 46 valid value/gradient evaluations.
Log evidence improved from `2718.515734` at its saved start to
`2747.440003` at its last accepted point. The next candidate had a finite
objective, but all ten evidence derivatives were NaN at both 100000 and
300000 PNPG iterations. The runner saved its diagnostics and exited with
status 3; it did not report SciPy convergence. The last accepted gradient L2
norm was approximately `0.284`, well above `gtol=1e-6` even after converting
to an infinity norm. There were 33 rejected budget attempts in total.

Consequently the local gradient checks pass under the precision/accuracy
conditions above, while stable convergence of the full optimizer is **not
validated**. These are separate results. The saved `best_point.json` and
`failed_point.json` preserve the final accepted and rejected coordinates.

An independent 300000-iteration reconstruction at the last accepted point
returns log evidence `2747.440003`, normalized KKT `9.2e-16`, and
`chi2/Ndata = 0.936043`. Source and lens-amplitude minima are both zero,
all arrays are finite, and strict complementarity passes. This is a valid
reconstruction at a non-converged outer iterate, not a converged fit.

The rejected point was independently reproduced at 300000 iterations. Its
forward solve succeeds, normalized KKT is `6.7e-16`, free-variable
stationarity error is `2.2e-11`, and the reconstruction is finite and
non-negative (`chi2/Ndata = 0.936101`). Exactly one coefficient fails the
configured strict-complementarity **margin**: MGE component 20 has zero
amplitude and positive dual residual `2.98e-7`, below its required margin
`6.10e-7`. Thus all three Jacobian rows are deliberately rejected by the
wrapper. This is not evidence that its exact mathematical dual is negative
or zero: it fails the selected numerical safety margin. Increasing the solve
budget has already resolved stationarity to machine precision at this point.
Lowering the margin again would require another gradient/trajectory
validation; it was not used to turn this failed run into a passing result.

## Reproduction and regression coverage

Run from the repository root in `tinylens_gpu`:

If the example's ignored FITS data are absent, first generate the original
seed-42 mock with default float32 simulation:

```bash
PYTHONPATH="$PWD" XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg \
  python examples/pix_src_demo_operator/simple/pix_src_lens_light_mge/sim_data.py
```

```bash
python .scratch/pnpg-mge-gradient-validation/validate_gradients.py \
  --x64 --iterations 30000 --pcg-iterations 2000 --pcg-rtol 1e-10 \
  --comp-rtol 1e-10 --check-fd --fd-steps 1e-4 --label gradient_check
python .scratch/pnpg-mge-gradient-validation/validate_gradients.py \
  --x64 --iterations 100000 --rtol 1e-11 \
  --pcg-iterations 2000 --pcg-rtol 1e-10 --comp-rtol 1e-10 \
  --optimize --maxiter 100 --refine-iterations 300000 \
  --start-json .scratch/pnpg-mge-gradient-validation/optimization_start.json \
  --optimizer-scales 0.03 0.08 0.08 0.05 0.05 3 0.03 0.03 0.02 0.02 \
  --label optimization
python .scratch/pnpg-mge-gradient-validation/validate_gradients.py \
  --x64 --iterations 300000 --rtol 1e-11 \
  --pcg-iterations 2000 --pcg-rtol 1e-10 --comp-rtol 1e-10 \
  --point-json .scratch/pnpg-mge-gradient-validation/diagnostic_point.json \
  --check-envelope --fd-steps 1e-4 1e-5 1e-6 1e-7 --label envelope
python .scratch/pnpg-mge-gradient-validation/validate_gradients.py \
  --x64 --iterations 300000 --rtol 1e-11 \
  --pcg-iterations 2000 --pcg-rtol 1e-10 --comp-rtol 1e-10 \
  --point-json .scratch/pnpg-mge-gradient-validation/failed_point.json \
  --label failed_point
python .scratch/pnpg-mge-gradient-validation/validate_gradients.py \
  --x64 --iterations 300000 --rtol 1e-11 \
  --pcg-iterations 2000 --pcg-rtol 1e-10 --comp-rtol 1e-10 \
  --point-json .scratch/pnpg-mge-gradient-validation/best_point.json \
  --forward-only --label final_reconstruction
```

The harness pins imports to this checkout. Its ignored `output/` directory
contains JSON diagnostics and reconstructed arrays. A failed gradient check
or optimization returns a nonzero exit status. The runner separately reports
SciPy's success flag and the projected gradient threshold; an `ftol` stop with
gradient above `gtol` also returns a nonzero status.
The retained `.scratch/pnpg-mge-gradient-validation/results.json` summarizes
the completed checks and failed optimization without depending on ignored
output files; the four saved parameter JSON files allow the key points to be
reproduced.

Regression: 728 non-slow tests pass. Coverage includes analytic position-penalty
gradients with coincident rays, active/inactive penalties and disabled
thresholds; preservation of invalid-ray failure; double-precision mapping
derivatives; a bright coupled NNLS example with a small positive dual; and
reconstruction/evidence finite differences with an enabled position constraint.
