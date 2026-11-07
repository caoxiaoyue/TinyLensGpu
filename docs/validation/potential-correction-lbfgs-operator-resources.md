# L-BFGS-B operator resource check

The generated fixture has 1,296 active image pixels, 4,096 source coefficients, and
4,609 potential coefficients. The potential grid includes 4,096 isolated support nodes
that exercise the amplitude prior without enlarging the derivative domain. The Float64
fit uses a perturbed Hermite source image, a fixed shear, a PSF, seeded noise, and the
default field-separated partition with a block width of 32. The benchmark records the
fit controls and partition digest. [Benchmark
driver](../../benchmarks/lbfgs_operator_resources.py)

From the repository root, activate the modeling environment and run the probe and
regression from `benchmarks/`.

```bash
source ~/anaconda3/bin/activate && conda activate tinylens_gpu
cd benchmarks
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1 NVIDIA_TF32_OVERRIDE=0
python lbfgs_operator_resources.py
pytest -q test_lbfgs_operator_resources.py
```

The probe runs the public operator fit in a fresh subprocess and captures XLA HLO before
optimization with the persistent compilation cache disabled. It scans for full source,
potential, and joint responses and prior or curvature matrices, the source–potential
cross block, and the data-space covariance. It includes transposes, arrays containing
the complete matrix axes, and reshapes with the same element count. The probe also
inspects live device arrays after synchronization and records each violation with its
shape and location. The successful fit must execute L-BFGS-B and endpoint PCG, return
principal-block evidence, and expose bounded principal blocks in the compiled shapes.

The RTX 5080 run inspected 196 compiled modules. It observed the `(273, 32, 32)` block
stack and found zero prohibited shapes. The fit took 310 L-BFGS-B iterations and 47 PCG
iterations. The recomputed residual was `1.48e-13` against a threshold of `1.61e-11`,
with GN stationarity `2.37e-12` and local GN log evidence `84.770524`.
[Retained resource record](../../.scratch/lbfgs-potential-correction/output/operator-resource-benchmark-final.json)

| Stage | Seconds |
| --- | ---: |
| Preparation | 1.914 |
| Fixed reference | 0.538 |
| Initial objective | 0.191 |
| Whitening | 0.239 |
| Optimizer | 0.853 |
| Endpoint | 1.465 |
| Result products | 0.966 |
| Cold fit including compilation | 4.261 |

The peak JAX allocator use was 26,216,704 bytes. Measurements include HLO capture
overhead and used JAX 0.7.1, SciPy 1.17.1, and NumPy 2.4.6 with GPU preallocation
disabled. The measurement helper synchronizes the fit and samples the allocator
high-water from device initialization. It can be reused for J0946 by running each
backend in a fresh process. Timing and peak memory are recorded measurements.
[Measurement helper](../../examples/potential_correction/_lbfgs_resources.py)
