# Single-step physical-mock Float32 comparison

Measured on 2026-11-01, on an NVIDIA GeForce RTX 5080. For the existing
fixed-strength 200×200 subhalo mock, Float32 retains almost the same localized
convergence signal as Float64 **when the true linear residual acceptance
threshold is explicitly relaxed from 1e-8 to 1e-5**. Dense is about 20.9×
faster after warm-up; the operator backend is about 1.46× faster. The original
strict configuration rejects both Float32 subhalo fits. The matched no-subhalo
Float32 control still fails in both backends, including the relaxed probe.
These observations do not establish general Float32 evidence optimization.

The [compact numerical record](../../.scratch/potential-correction/research/single-step-float32-20261101/summary.json)
and [reproduction instructions](../../.scratch/potential-correction/research/single-step-float32-20261101/README.md)
retain 16 fresh isolated workers. The [driver](../../.scratch/potential-correction/experiments/compare_single_step_precision.py)
uses the existing workflow benchmark without changing library code or defaults.
Earlier [strict scientific Float32 failures](potential-correction-precision-performance.md)
remain valid; the new successful subhalo results use a separately declared
acceptance threshold.

## Matched physical and numerical setup

Only the fixed-gradient single-step algorithm is exercised. Dense workers
complete before operator workers. Within each backend, the original threshold
is tested first, followed by the relaxed probe; both precisions and both
subhalo/control observations are included. Source, D4 and amplitude strengths
are fixed throughout; there is no hyperparameter search or prior adjustment.

This is the [scientific reference mock](potential-correction-single-step-scientific.md):
200×200 at 0.012 arcsec/pixel, 22016 active pixels, mock/inversion nsub=2/2,
49×49 source nodes in [-0.4,0.4]², potential spacing 0.075 arcsec, 724 valid
core nodes, 956 support unknowns, Gaussian PSF sigma 0.05 arcsec with 27×27
samples, noise sigma 0.01 and seed 17. The true fixed macro is SIS plus shear;
the Pseudo-Jaffe subhalo has b=0.04 at (0.75,0.1) arcsec. The gradient is the
same frozen analytic truth gradient at macro rays; the source brightness
itself remains an unknown.

Strengths are `(lambda_s,lambda4,lambda0)=(0.0006,1.6e7,10000)` for subhalo
and `(0.0006,1e11,10000)` for control. Thus the control uses the historical
control-specific prior; this is not a same-prior detection-significance test.
Both precisions cast the same saved Float64 observations, noise, PSF and
gradient. Float32 geometry is genuinely built at Float32: integer node/core
sets remain equal, while 3464/88064 query-corner choices change at grid lines.
The recorded operator block partition is unchanged.

In the probe, **both Float64 and Float32** use `solve_rtol=1e-5`. This changes
the final residual acceptance certificate, not the fitted prior or Dense
factorization method. Operator recursive PCG remains `rtol=1e-12`, `atol=0`,
maximum 20000 iterations in every worker. This is not an optimized Float32
PCG-tolerance study. Data, interpolation, gradients, forward/transpose/prior
actions, fitted fields and available evidence terms are audited at the
requested precision; FFT is complex64/complex128 as appropriate.

## Subhalo reconstruction with the explicit residual probe

Each Float32 result is compared with the same-backend Float64 probe. Relative
L2 differences use the complete source/support vectors, or all 724
derivative-valid core nodes for κ; no clipping or affine alignment is applied.
The table uses the final warm fit's saved fields.

| Quantity | Dense Float32 vs Float64 | Operator Float32 vs Float64 |
| --- | ---: | ---: |
| Source relative L2 difference | 0.0968% | 0.00771% |
| Raw potential relative L2 difference | 1.742% | 0.2234% |
| Core κ relative L2 difference | 1.274% | 0.1443% |
| Core κ maximum absolute difference | 0.001187 | 0.0001837 |
| Float64 peak κ | 0.290141 | 0.290141 |
| Float32 peak κ | 0.289449 | 0.290193 |
| Float32 peak change | -0.238% | +0.0179% |
| Float32/Float64 peak location | (0.75,0.075) | (0.75,0.075) |
| Nonlinear image difference RMS / noise | 0.003450 | 0.0003047 |
| Float32 minus Float64 log evidence | -0.07040 | +0.005071 |

Both peaks are 0.025 arcsec from the injection, as in Float64. This comparison
establishes retention of the Float64 reconstruction, not exact recovery of
the physical truth: the historical fixed-prior single-step model already
suppresses the discrete-truth peak (about 0.6314) to about 0.2901.

All cold and five warm subhalo probe fits report success and finite evidence
in each backend/dtype. Small Float32 variation between independent workers is
visible in the retained strict/probe fields; the driver does not assume bitwise
GPU reproducibility. Same-backend evidence differences here do not calibrate
the operator's principal-block determinant approximation against full Dense
evidence, nor test ranking of nearby regularization candidates.

## Timing

Each row is one fresh GPU worker. Setup and the first fit are separate;
the first fit includes compilation and lazy preparation. Warm times are the
median and observed min–max of five complete synchronized public fits,
including evidence and derived image/map products. No other test workload
runs on the GPU during the campaign. Mock generation, audits and NPZ export
are outside these fit timings. Five repeated fits share kernels in one
worker; they are not independent process repetitions.

| Subhalo, solve_rtol=1e-5 | First fit, seconds | Warm median, seconds | Warm min–max, seconds | Warm speedup vs Float64 |
| --- | ---: | ---: | ---: | ---: |
| Dense Float64 | 5.207 | 0.6453 | 0.6424–0.6804 | 1× |
| Dense Float32 | 4.717 | 0.03088 | 0.02972–0.03239 | 20.90× |
| Operator Float64 | 6.413 | 0.2734 | 0.2718–0.2804 | 1× |
| Operator Float32 | 5.486 | 0.1868 | 0.1849–0.1901 | 1.46× |

The large Dense warm speedup does not apply to a single cold execution:
compilation and preparation dominate that use. PCG iteration counts also
increase in Float32 (934 vs 674 in the saved final operator probe fits),
limiting its net speedup in this configuration. Successful Float64 strict
timings and all failed-worker first-fit times are retained in JSON; a failed
Float32 fit has no warm timing or claimed speedup.

Observed environment: Python 3.11.14, JAX/jaxlib and CUDA plugin/PJRT 0.7.1,
NumPy 2.4.6, SciPy 1.17.1, CUDA runtime 12.8.90. Preallocation and persistent
compilation caching are disabled. Software, implementation and input hashes
are retained with each worker.

## Failure boundary and interpretation

At the original `solve_rtol=1e-8`, Dense/Operator Float32 subhalo true relative
residuals are about 5.48e-7 / 8.96e-7. The fields are close to the reference,
but both results correctly retain `success=False` and null evidence. The
relaxed probe admits residuals of about 5.62e-7 / 1.01e-6 without changing
the priors; it does not improve the actual linear algebra accuracy to 1e-8.

The Float32 control fails Dense prior Cholesky factorization at either
threshold. Operator control fails the true residual gate even at 1e-5.
Float64 controls pass both thresholds. The control's stronger D4 prior poses
a concrete representability problem: its maximum smoothing diagonal is
approximately 1.4e13, whose Float32 ULP is 1048576. Adding the physical
amplitude prior of 10000 makes no representable difference on 912/956
diagonal entries. For the subhalo prior, the maximum diagonal is 2.24e9,
ULP is 256, and amplitude additions survive on every node. The driver's
independent stencil-diagonal calculation records these values. This explains
a mechanism for losing the proper weak-mode prior in Float32 dense/block
assembly; it is not a proof that every failure has only this cause.

For this fixed subhalo setting, Float32 Dense is a promising fast path and
Float32 operator also retains the signal. Keep Float64 as the reference for
the stronger control prior and evidence-based hyperparameter selection until
the numerical formulation is made reliable across that range. A future
mixed-precision or better-scaled formulation should be validated separately;
neither is implemented or timed here.

## Verification

All 16 workers exit normally; numerical failures are returned as explicit
fit results. Summary generation validates equal frozen physical inputs,
unchanged strengths, matching integer topology and derivative domains, and
the requested intermediate/output dtypes. Nonfinite fields from failed
fits are recorded as unavailable rather than used for reconstruction metrics.

From `tests/`, with `tinylens_gpu` active and preallocation disabled:
`pytest -q test_potential_correction_precision.py -k single_step`:
**2 passed, 2 deselected** in 7.34 s. Black and Ruff checks pass for the
new reproduction driver; `git diff --check` passes. No library implementation
or default setting changed.
