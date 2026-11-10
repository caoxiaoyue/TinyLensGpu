# Overlap evidence GPU qualification

Both configurations ran in Float64 on one NVIDIA GeForce RTX 5080 under JAX 0.7.1. They used the archived no-noise Euclid mock and its RMS weights. The 80×80 source grid, four subpixels per side and original PSF came from the mock replay. The models used cubic Hermite interpolation, the archived native M1 arc and coverage offset `10^-4`. The fixed M2 calibration supplied log λ = -9.685380417 and ρ = 5.623951903. Both evidence methods used PCG with a 1,200-iteration limit, relative tolerance `1e-10`, and solver block width 10. Overlap used evidence tile width 10. [Mock input and fit provenance](adaptive-image-overlap-euclid-mock.md) · [Float64 M3 qualification](adaptive-image-overlap-m3-optimizer.md)

The source-only fit used the archived 1,339 active pixels. The joint fit used 1,976 active pixels from a 2.5-arcsec circular mask that included the lens core. Its mock observation added eight unit-amplitude Gaussian lens-light bases with widths geometrically spaced from 0.14 to 1.25 arcsec, ellipticity components (0.12, 0.02), and center (0.02, −0.01) arcsec. The injected amplitudes followed a geometric sequence from 0.1 to 1 before scaling the foreground peak to five times the source image peak. Both joint fits solved the amplitudes with lens-light precision `1e-6` and shared the same generated observation and reference arc.

## Score and gradient cost

Each row ran in a fresh CUDA process with preallocation and the persistent compilation cache disabled. Preparation includes model construction and synchronization after loading the fixture. First compilation includes lowering and compilation. The warmed times are medians of three synchronized executions after one first execution. The peak reports JAX allocator bytes in use from initialization through execution. CUDA context and graphics residency are outside that counter. The retained record also includes first execution times, the allocator pool peak and compiled temporary bytes. [Machine-readable results](../../.scratch/adaptive-image-overlap-evidence/experiments/overlap_gpu_results.json)

| Case | Evidence | Call | Prep (s) | Compile (s) | Warm (s) | Peak (MiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Source-only | Block | score | 2.387 | 1.885 | 0.0640 | 53.6 |
| Source-only | Block | gradient | 2.419 | 5.130 | 0.1197 | 292.0 |
| Source-only | Overlap | score | 2.524 | 2.075 | 0.3278 | 73.4 |
| Source-only | Overlap | gradient | 2.758 | 6.100 | 0.9856 | 169.4 |
| Joint | Block | score | 2.436 | 2.758 | 0.1540 | 59.0 |
| Joint | Block | gradient | 2.618 | 7.714 | 0.3090 | 406.2 |
| Joint | Overlap | score | 3.056 | 4.164 | 0.5191 | 108.5 |
| Joint | Overlap | gradient | 2.518 | 7.983 | 1.3798 | 300.5 |

## Matched sampler batches

Each sampler call submitted 23 proposals along the line from the injected mass to the Hermite M3 posterior peak. Width five evaluates four complete chunks and a remainder of three proposals. All runs returned 23 valid scores. The maximum absolute difference between widths was 1.1e-10 across both cases and methods.

| Case | Evidence | Inner width | Prep (s) | Compile (s) | Warm batch (s) | Peak (MiB) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Source-only | Block | 1 (explicit) | 2.384 | 1.941 | 1.4631 | 53.6 |
| Source-only | Block | 5 (explicit) | 2.384 | 4.358 | 0.3824 | 152.8 |
| Source-only | Overlap | 1 (automatic) | 3.077 | 2.445 | 7.5292 | 73.4 |
| Source-only | Overlap | 5 (explicit) | 2.488 | 5.091 | 2.2838 | 379.2 |
| Joint | Block | 1 (explicit) | 2.967 | 3.359 | 3.5474 | 59.0 |
| Joint | Block | 5 (explicit) | 2.752 | 7.766 | 1.1231 | 283.0 |
| Joint | Overlap | 1 (automatic) | 2.808 | 4.101 | 11.9752 | 108.5 |
| Joint | Overlap | 5 (explicit) | 2.515 | 8.632 | 4.7355 | 392.2 |

Automatic Overlap width **1** keeps the allocator peaks at 73.4 MiB for the source-only fit and 108.5 MiB for the joint fit. Explicit width five increased those peaks to 379.2 and 392.2 MiB. The sampler retains its outer batch size when `vectorized_chunk_size` changes the number of proposals evaluated together.

## Workspace audit

With tile width 10 on the tested 80×80 grid, the lowered and optimized GPU HLO and live device arrays contained no complete native or active PSF matrix, full image-by-source design or global source curvature for scalar scores, their reverse-mode gradients, or either sampler batch width. The check covered permuted and batched axes and flattened element counts. Optimized byte workspaces from CUDA custom calls enter the device allocator measurements as bytes. The source-only score used a 400×3,240 FFT panel. Explicit width five produced 5×400×3,240 and 3×400×3,240 panels for full chunks and the remainder. The same width control appears in the joint HLO.

The local matrices for an exact window have one row and column per node in that window. A tile width of 40 on this grid would give one exact full-grid window and would allocate full-size local curvature matrices and factors. That path assembles its Gram with bounded normal-operator column batches. The measured configurations use multiple smaller windows.

## Reproduce

Run from the repository root with the original archive. The runner validates the FITS, mask, source-grid and native reference identities against the retained mock manifest. It creates the joint observation on CPU and checks that no CUDA compute process is active before each measured worker.

```bash
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
mock_archive=/path/to/euclid_mock
qualification_output=.scratch/adaptive-image-overlap-evidence/output/gpu
OPENBLAS_NUM_THREADS=1 NVIDIA_TF32_OVERRIDE=0 \
  python benchmarks/overlap_evidence_resources.py \
  --archive "$mock_archive" --output "$qualification_output" --outer-batch 23 --repeats 3
```

The run records the exact revision and hardware query with the timings. [Benchmark runner](../../benchmarks/overlap_evidence_resources.py)
