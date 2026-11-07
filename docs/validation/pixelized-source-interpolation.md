# Pixelized source interpolation verification

The verification uses the library at `ee1c67d`. Runs used the `tinylens_gpu` environment with Python 3.11.14, JAX and jaxlib 0.7.1, and NumPy 2.4.6. [Fixed-strength records](../../.scratch/pixelized-source-interpolation/experiments/evidence-comparison.json) · [GPU measurements](../../.scratch/pixelized-source-interpolation/experiments/gpu-costs.json)

## Fixed-strength evidence

The comparisons use Float64 on CPU and hold the observations and prior strengths fixed within each problem across interpolation modes, backends, and partitions. [Comparison tests](../../tests/test_pixelized_interpolation_evidence.py)

| Problem | Configuration |
| --- | --- |
| Ordinary source-only | 4 × 4 nodes at 0.25 arcsec spacing, 9 × 9 image at 0.125 arcsec per pixel, 48 selected pixels, `nsub=2`, asymmetric 3 × 3 PSF, and first-order source precision with strength 0.7. |
| Single-step correction | 2 × 2 source grid, 3 × 3 image with nine selected pixels, frozen gradient (1.2, −0.8), and 33 potential coefficients. The source, potential roughness, and potential amplitude strengths are 1.3, 0.3, and 0.4. |
| Iterative correction | 2 × 2 source grid, 4 × 4 image with 16 selected pixels, and 33 potential coefficients. Both modes start from the same nodal intensities and zero potential. The observed image comes from a bilinear control with a small potential and a deterministic additive perturbation. The strengths are 0.3, 0.02, and 20. |

The ordinary operator uses prior blocks of the first-order precision and posterior blocks from the native-resolution preconditioner. The reference constructs the regularization from first differences, uses the native source design and independently convolves inverse noise variance with the reversed squared PSF. It takes the resulting principal blocks and applies the respective prior and posterior Cholesky jitters before evaluating their log determinants in NumPy. The 4 × 4 layout covers the whole source grid, with the native data approximation still in effect. The dense reference uses the full subpixel-integrated, PSF-convolved data curvature.

For correction, the reference forms the full posterior curvature from the single-step linear response or from automatic differentiation of the final iterative image. It sums log determinants of the principal blocks of that posterior and the complete prior. The two partitions split the joint coefficient indices into groups of eight with a shorter final block. One uses contiguous order and the other lists even indices before odd ones. The dense iterative reference uses a full final Gauss–Newton curvature for its local evidence.

The largest absolute difference between a block log determinant and its independent reference was 8.53e−14, against an absolute test tolerance of 1e−10. The table shows the operator log evidence minus the dense value.

| Problem | Interpolation | Dense log evidence | First partition | Second partition |
| --- | --- | ---: | ---: | ---: |
| Ordinary | Bilinear | 28.430 | −3.585 (2 × 2 tiles) | −4.346 (4 × 4 tile) |
| Ordinary | Hermite | 55.353 | +2.422 (2 × 2 tiles) | +1.675 (4 × 4 tile) |
| Single-step | Bilinear | −12.614 | −4.238 (contiguous) | −0.206 (even then odd) |
| Single-step | Hermite | −12.651 | −4.997 (contiguous) | −0.931 (even then odd) |
| Iterative | Bilinear | 8.979 | −7.920 (contiguous) | −9.740 (even then odd) |
| Iterative | Hermite | 7.231 | −8.831 (contiguous) | −10.827 (even then odd) |

## Reconstruction at the same strengths

The dense and operator reconstructions succeeded in all 12 comparisons. The objective is half the sum of weighted squared residuals and the prior penalty. Single-step uses its linear image in the objective. The image differences below use active pixels of the nonlinear image for correction and the lensed-source prediction for ordinary inversion. Differences are maxima across the two partitions.

| Problem | Interpolation | Dense objective | Absolute objective difference | Max source difference | Max potential difference | Max image difference | Max relative normal residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary | Bilinear | 41.847039 | 0 | 7.22e−16 | — | 2.78e−16 | 4.45e−16 |
| Ordinary | Hermite | 8.050413 | 1.78e−15 | 1.04e−12 | — | 6.68e−13 | 4.81e−13 |
| Single-step | Bilinear | 2.006936 | 0 | 9.88e−14 | 3.32e−13 | 8.10e−15 | 4.11e−14 |
| Single-step | Hermite | 1.989033 | 4.45e−16 | 3.62e−13 | 6.87e−13 | 4.03e−13 | 1.97e−13 |
| Iterative | Bilinear | 0.446380 | 2.22e−16 | 6.67e−16 | 2.23e−16 | 4.45e−16 | 7.68e−11 |
| Iterative | Hermite | 0.289446 | 1.11e−16 | 8.88e−16 | 2.21e−16 | 4.44e−16 | 8.92e−10 |

The residual is evaluated independently from the image response and complete prior, normalized by the norm of the data right-hand side. The iterative rows use the final stationarity residual. In both backends, every accepted iterative solve had a true relative residual below 2e−12. These measurements characterize the reported fixed-strength configurations.

## Representative GPU solve

The benchmark uses a 100 × 100 fitted source grid with bounds [−1.2, 1.2] arcsec on each axis, a 100 × 100 image at 0.05 arcsec per pixel, and `nsub=4`. The fixed SIE has Einstein radius 0.8 arcsec, ellipticity components (0.08, -0.04), and center (0.02, -0.03) arcsec. The selected annulus retains 4,008 pixels and 64,128 rays. Both runs use the same deterministic ring data, spatially varying noise, asymmetric PSF, first-order source prior of strength 0.08, and 10 × 10 source tiles. The script specifies the input formulas and PSF. [Benchmark](../../benchmarks/pixelized_source_interpolation.py)

Each mode ran in a fresh process on an NVIDIA GeForce RTX 5080 with 16,303 MiB and driver 615.71.09. Float32 ran with highest matmul precision, preallocation disabled, and the persistent compilation cache disabled. The native preconditioner includes constructing and factoring the data and regularization blocks. The right-hand-side preparation is outside the solve timing and took 128.9 ms for bilinear and 134.9 ms for Hermite on first execution. Executions were synchronized and warm times are medians of ten runs.

| Interpolation | Phase | Lower (ms) | Compile (ms) | First execution (ms) | Warm (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| Bilinear | Mapping | 30.81 | 107.16 | 1.16 | 0.081 |
| Hermite | Mapping | 43.47 | 138.98 | 4.25 | 0.057 |
| Bilinear | Native preconditioner | 190.78 | 424.69 | 14.55 | 10.65 |
| Hermite | Native preconditioner | 211.07 | 450.19 | 138.82 | 127.79 |
| Bilinear | PCG solve | 33.79 | 280.83 | 56.18 | 1.95 |
| Hermite | PCG solve | 34.78 | 303.68 | 63.18 | 2.16 |

| Interpolation | Stencil shape | Stencil storage (MiB) | Peak allocated on device (MiB) | PCG iterations | True relative residual |
| --- | --- | ---: | ---: | ---: | ---: |
| Bilinear | 64,128 × 4 | 1.96 | 26.40 | 12 | 4.63e−6 |
| Hermite | 64,128 × 16 | 7.83 | 30.65 | 12 | 4.97e−6 |

The peak is cumulative bytes in use from the process's JAX CUDA allocator and excludes driver context memory. Both modes keep 10,000 source coefficients and 100 preconditioner factors of size 100 × 100. A dense subpixel mapping would require 2,446 MiB in Float32. PCG completed with its relative tolerance of 1e−5, absolute tolerance of 1e−12, and limit of 400 iterations. Hermite increases warm preconditioner time by about 12× and warm solve time by about 1.1× in this setup.

## Reproduction

Activate the `tinylens_gpu` environment from the repository root.

```bash
source ~/anaconda3/bin/activate && conda activate tinylens_gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1
```

From `tests/`, the following CPU run passed 25 tests. The JUnit `comparisons` properties hold the fixed-strength results.

```bash
JAX_PLATFORMS=cpu pytest -xq -o junit_family=legacy \
  --junitxml=/tmp/interpolation-evidence.xml test_pixelized_interpolation_evidence.py \
  test_pixelized_source_interpolation.py test_operator_source_interpolation.py
```

A separate CUDA process passed all six evidence cases from `tests/`.

```bash
JAX_PLATFORMS=cuda pytest -xq test_pixelized_interpolation_evidence.py
```

From the repository root, run both modes with exclusive GPU access.

```bash
export PYTHONPATH="$PWD" JAX_PLATFORMS=cuda JAX_ENABLE_COMPILATION_CACHE=false
python benchmarks/pixelized_source_interpolation.py bilinear
python benchmarks/pixelized_source_interpolation.py cubic_hermite
```

The example and benchmark regression commands ran as separate processes.

| Working directory | Command | Outcome |
| --- | --- | --- |
| `examples/tests/` | `JAX_PLATFORMS=cpu MPLBACKEND=Agg pytest -xq test_potential_correction_operator_example.py test_potential_correction_iterative_example.py` | 4 passed |
| `benchmarks/` | `JAX_PLATFORMS=cuda pytest -xq test_potential_correction_memory.py test_potential_correction_preconditioner.py` | 3 passed |
