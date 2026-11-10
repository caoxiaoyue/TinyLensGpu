# Float64 L-BFGS-B qualification on the Euclid mock

The source-only M3 model uses the archived 200×200 no-noise image, RMS weights, PSF, mask, four subpixels per side and fixed 80×80 source grid. The reference arc comes from the archived SIE plus shear M1 fit. The run retains the freshly calibrated M2 strengths `log_lambda_reg = -9.685380417074992` and `adaptive_reg_rho = 5.623951902964207` from the Float32 replay. It uses the same first-order source precision, cubic Hermite interpolation and internal coverage offset of 10⁻⁴. The qualification promotes the data, reference, and physical parameter arrays to Float64. [Euclid mock replay](adaptive-image-overlap-euclid-mock.md)

The objective is the array-returning operator model's complete Overlap score, including its position penalty. The EPL and shear vary over the six original M3 physical bounds. The center stays at the archived Stage A medians. Overlap and the PCG preconditioner both use a tile width of ten. The tighter primary PCG relative tolerance is 10⁻¹⁰ with 1,200 iterations allowed. The implicit adjoint uses the same tolerance and 2,400 iterations. Both use absolute tolerance 10⁻¹². The run used JAX 0.7.1 on the RTX 5080.

## Predeclared controls

The three starts use the injected free-mass coordinates, the peak of the archived prototype Overlap posterior, and a fixed perturbation of the injection. The historical start comes from `m3_ablation/no_noise_default_overlap_output/dynamic_arc_operator_overlap.npz`. Every exact vector is in the tracked configuration. SciPy works in injection-centered coordinates scaled by (0.04, 0.12, 0.04, 0.04, 0.03, 0.03) in the order `theta_E, gamma, e1_mass, e2_mass, gamma1, gamma2`. The transformed bounds preserve the original physical box.

The declared L-BFGS-B settings allow 100 iterations, 220 objective calls, and 40 line-search steps per iteration, with `maxcor=15`, `ftol=1e-15`, and `gtol=2e-5`. Each start must return SciPy success, a finite endpoint score and gradient, a valid primary solve and evidence, an endpoint inside the physical bounds, a nonnegative score gain, and a scaled projected-gradient infinity norm at most 10⁻³. The projected step is `clip(z + ∇S, lower, upper) - z` for the maximized score S and scaled coordinates z. Endpoint basins use a scaled infinity distance at most 10⁻³ and a score difference at most 10⁻⁶. The configuration was committed at `6037a02` before running either phase. [Predeclared configuration](../../.scratch/adaptive-image-overlap-evidence/experiments/m3_optimizer_config.json)

## Slope and derivative probes

The conditional prior probe holds the 6,400 source coefficients at their valid injection reconstruction and recomputes only the mass-dependent source precision in `sᵀH(θ)s / 2`. An independent banded calculation at the injection gives 70.48497119, agreeing to about 2.5 × 10⁻¹². The full score re-solves the source and includes the Overlap determinant and configured position term at each proposal.

The 19-point scan holds the other five mass coordinates at the injection and varies γ from 2.02 to 2.38 in increments of 0.02. All 19 solves and scores are valid. The fixed-source energy decreases from 103.2963 to 55.1253, while the full score reaches its sampled maximum of 1509.2564 at γ = 2.20. The maximum ray-traced position separation over the scan is 0.0600 arcsec, below the 0.1 arcsec penalty threshold. The position contribution is zero at every scanned point.

Central differences cover all six injection coordinates and the slope on both sides of an actual Hermite stencil change. The crossing is an active subpixel ray at y-node 45, γ = 2.2004691096820332. The ray offsets at slope shifts of ±10⁻⁵ are approximately ±9.31 × 10⁻⁵ source grid pixels. The central difference at the knot straddles the change. All nine comparisons pass for both objectives using the declared relative tolerance of 3 × 10⁻⁴ and absolute tolerance of 5 × 10⁻⁴. Maximum absolute errors are 4.80 × 10⁻⁷ for the conditional prior energy and 8.23 × 10⁻⁷ for the complete score. The full-score derivative at the crossing is −61.88639262 by autodiff and −61.88639259 by central difference. All finite-difference stencils stay on the zero-penalty branch.

## Multi-start fits

| Start | SciPy status | Iterations / calls | Score gain | Endpoint score | Scaled projected gradient | Basin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| injection | 0 | 22 / 29 | 0.529506 | 1509.785915959 | 7.518e-06 | 1 |
| archive_peak | 0 | 22 / 29 | 0.072626 | 1509.785915959 | 2.335e-06 | 1 |
| perturbed | 0 | 31 / 43 | 62.986290 | 1509.785915959 | 8.427e-07 | 1 |

All three runs report `CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL`. Each endpoint is in bounds, has a finite gradient and a valid score, and passes the declared stationarity threshold. The endpoint slopes agree to about 5.2 × 10⁻⁹ around γ = 2.19226494. The position contribution is zero at all endpoints, with a maximum separation of about 0.00500913 arcsec. The runner flagged one injection trial and two perturbed trials for a nonfinite score or gradient, or a score at or below the failure sentinel. The line searches still reached valid endpoints. The probe phase took about 46 s and the three fits took about 146 s on this host.

These runs qualify the existing Hermite coverage offset of 10⁻⁴ for the recorded mock model and controls. The retained record includes the individual derivatives, terminal line-search messages, scores, physical parameters and projected gradients. [Qualification results](../../.scratch/adaptive-image-overlap-evidence/experiments/m3_optimizer_results.json)

## Reproduction

Run the mock replay first to produce the matching M2 calibration, M3 fit, and determinant comparison. From the candidate checkout, run the probe phase before the optimizer phase. The optimizer checks the probe record, input identities, and configuration before running. [Qualification runner](../../.scratch/adaptive-image-overlap-evidence/experiments/qualify_m3_optimizer.py)

```bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false

mock_archive=/path/to/workspace/euclid_mock
mock_output=.scratch/adaptive-image-overlap-evidence/output/mock
qualification_output=.scratch/adaptive-image-overlap-evidence/output/optimizer
runner=.scratch/adaptive-image-overlap-evidence/experiments/qualify_m3_optimizer.py
python "$runner" probe --archive "$mock_archive" --mock-output "$mock_output" --output "$qualification_output"
python "$runner" optimize --archive "$mock_archive" --mock-output "$mock_output" --output "$qualification_output"
```
