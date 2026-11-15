# Adaptive-image Overlap replay on the no-noise Euclid mock

The archived observation has a 200×200 native image, a 21×21 PSF, and the original RMS map with values from 0.1 to 0.14860533. The FITS header records `NOISEADD=False`. The 2.5-arcsec circular mask combined with the archived feature mask leaves 1,339 active native pixels. The fit uses a pixel scale of 0.1 arcsec, four subpixels per side, and a fixed 80×80 source grid with a spacing of 0.02 arcsec. The input manifest identifies each FITS, grid, posterior, and native arc file by SHA-256. [Archive manifest](../../.scratch/adaptive-image-overlap-evidence/experiments/euclid_mock_inputs.json)

The reference is the PSF-convolved arc from the archived SIE plus shear M1 fit. Re-rendering the saved pixels at the recorded M1 mass with bilinear interpolation reproduced the native reference within \(9.54\times10^{-7}\) in Float32. M2 and M3 use cubic Hermite interpolation with fixed zero exterior nodes. Their prior uses squared per-ray interpolation weights, an active-native-pixel 99.5th-percentile brightness normalization with floor \(10^{-10}\), and the internal Hermite coverage offset of \(10^{-4}\). The imaging operator uses the signed stencil.

## Fresh M2 calibration and M3 fit

Both stages used explicit Overlap, PCG with 400 maximum iterations and relative tolerance \(10^{-5}\), first-order source precision, and a source tile width of 10 for both the preconditioner and the evidence windows. The position penalty used the two recorded Stage A image positions, a separation threshold of 0.1 arcsec, and a floor of \(-10^{10}\). Sampling used Float32 on the RTX 5080 with JAX 0.7.1. Nautilus used seed 73, an outer batch of 100, and explicit Overlap chunks of five.

M2 fixed SIE plus shear to the recorded M1 mass and varied \((\log\lambda,\rho)\) over \([-\log(10^6),\log(10^6)]\) and \([0,8]\). It reached 1,337 effective posterior samples in 4,300 likelihood calls with 250 live points and a target of 600 effective samples. The highest-scoring candidate among the posterior peak, posterior medians, bounded Powell refinements, and both \(\rho\) boundaries selected \(\log\lambda=-9.6853804\) and \(\rho=5.6239519\). The selected \(\lambda\) was \(6.21860\times10^{-5}\). The score was 1500.2263 at a valid 61-iteration primary solve, with image \(\chi^2=28.3953\) and zero position penalty.

M3 kept those strengths and the M1 arc and grid fixed. It varied all six original EPL plus shear parameters using the archived mass-prior construction. The lens center was fixed at the Stage A medians \((0.0023381,-0.0076449)\) arcsec. The injected slope was \(\gamma=2.2\).

M3 used 180 live points and reached 1,485 effective posterior samples in 7,900 likelihood calls. The slope median was \(2.19607\), with a 68% interval of \([2.14861,2.23863]\) and a 95% interval of \([2.10800,2.27194]\). The injection lay inside both intervals. The injected free-mass vector, the posterior marginal medians, and the sampled posterior peak all gave valid primary and evidence evaluations with zero position penalty. Six sampler records were at or below the failure sentinel and carried zero posterior weight. The retained record includes the actual mass priors, solver settings, selected strengths, and diagnostics. [Run results](../../.scratch/adaptive-image-overlap-evidence/experiments/euclid_mock_results.json)

## Matched determinant comparison

The 19 proposals were the injected free-mass vector, posterior marginal medians, sampled peak, and 16 mass vectors stratified by posterior weight in slope order. Each proposal used a single primary PCG solve. The three scores shared its residual, prior energy, Gaussian noise normalization, and position penalty. Block used the Cholesky factors from the same preconditioner used by PCG. Overlap used the production source-window determinants.

The complete reference formed a Float64 whitened design from the proposal's signed subpixel stencil and an independently assembled sparse native PSF. It built the weighted first-order prior in lower banded form, including the upper row and column boundary terms. With whitened design \(F\) and prior \(H\), the ratio \(\log\det(F^\mathsf{T}F+H)-\log\det H\) was evaluated as \(\log\det(I+F H^{-1}F^\mathsf{T})\). A small independent dense check matched this banded calculation to \(10^{-12}\). On the archive, the independent design applied to the shared coefficients reproduced the native active-pixel predictions within \(1.12\times10^{-6}\). All 19 solves converged in 54–60 iterations and all position penalties were zero.

Offsets below are the approximate score minus the complete score. Centered absolute errors subtract the median signed offset before taking absolute values.

| Method | Median signed offset | Offset range | Centered absolute error P95 | Spearman rank | Kendall rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| Block | −60.13438 | [−63.80840, −53.94206] | 4.60097 | 0.49123 | 0.35673 |
| Overlap | +6.46065 | [+6.40339, +6.48226] | 0.03969 | 0.99825 | 0.98830 |

The complete reference and Overlap both ranked the posterior marginal-median vector first. Block ranked the second slope stratum first. These offsets and rankings apply to the recorded Float32 M3 configuration and its 19 matched proposals. The complete determinant was evaluated in Float64 on the Float32 proposal geometry and shared linear solution.

## Reproduction

Run from the candidate checkout. The archive path is the local directory containing `data_no_noise`, `pix_src_fit_no_noise_upstream`, and `m3_ablation`. The retained replay used separate `--phase fit` and `--phase compare` commands. The default runs both phases. [Replay runner](../../.scratch/adaptive-image-overlap-evidence/experiments/euclid_mock_replay.py)

```bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
export PYTHONPATH="$PWD"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export NVIDIA_TF32_OVERRIDE=0 OPENBLAS_NUM_THREADS=8

mock_archive=/path/to/euclid_mock_archive
mock_output=.scratch/adaptive-image-overlap-evidence/output/mock
python .scratch/adaptive-image-overlap-evidence/experiments/euclid_mock_replay.py \
    --archive "$mock_archive" --output "$mock_output"
```
