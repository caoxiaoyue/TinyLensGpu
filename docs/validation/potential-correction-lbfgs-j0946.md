# J0946 L-BFGS-B two-backend qualification

Both fresh CUDA refits qualified at the fixed trial 497 strengths
`(0.0002532641285041916, 538677.9228218549, 10572.677018080045)` in source,
fourth-difference, and amplitude order. The fits used the same M3 sample
21342, archived lens-light-subtracted image, noise, mask, PSF, integration,
first-order adaptive source scale, source-only start, and zero initial potential.
The distinct saved M3 source provided the hard-box bilinear whitening reference.
Both nonlinear fits used Float64 cubic Hermite with the archived native Float32
image sampling promoted to Float64. [Dense archive procedure](potential-correction-lbfgs-j0946-dense.md)

The operator command checks the dense reference against the same six archive
checksums, strengths, fixed fit controls, source initialization provenance, and
reconstruction checksum. It repeats the archive comparison before running the
operator fit. Both comparisons match the potential node indices, physical
coordinates, and derivative-valid mask before measuring convergence differences.
[Validation entry point](../../examples/potential_correction/validate_j0946_lbfgs.py)

| Comparison | Measurement | Limit |
| --- | ---: | ---: |
| Dense vs. trial 497 absolute objective difference | 2.73e−12 | 1e−3 |
| Dense vs. trial 497 absolute log evidence difference | 6.13e−7 | 0.01 |
| Dense vs. trial 497 max absolute convergence difference | 3.48e−8 | 1e−4 |
| Operator vs. dense absolute objective difference | 1.18e−10 | 0.01 |
| Operator vs. dense max absolute convergence difference | 6.02e−7 | 0.005 |
| Operator vs. dense positive peak separation | 0 arcsec | 0.1 arcsec |

The convergence comparisons include all 802 derivative-valid nodes. Both peaks
lie at ((-0.7'', 1.1'')). Dense and operator objectives are 605.0380047097674
and 605.0380047098852. Native L-BFGS-B status was 0 for both fits after 135 and
901 iterations, respectively. The GN stationarity statistics were 1.21e−12
and 1.12e−10 at the common 1e−3 threshold. The operator's recomputed coupled
PCG residual was 1.79e−12 against a 1.89e−10 limit after 951 iterations.

The operator used the `field_separated_contiguous_v1` partition of 175 blocks
with cap 64. The result records every index and the partition SHA-256
`d67b0403a1e0da66cc71bb3d311df3a79121341fd7f209e178052e65c013dd24`.
The endpoint used `pcg_max_iter=2000`, `pcg_rtol=1e-10`, `pcg_atol=0`,
`solve_rtol=1e-8`, and `solve_atol=0`. The reports also record the shared
L-BFGS-B controls.

| Backend | Local Gauss–Newton log evidence | Determinant method |
| --- | ---: | --- |
| Dense | 6307.427178834962 | `full_data_space` |
| Operator | 6409.152194588976 | `principal_block_cholesky` |

The operator estimate exceeds the dense estimate by 101.725015754.
The operator determinant is verified separately against explicit principal
blocks in the independent small coupled problem.
[Principal-block reference test](../../tests/test_potential_correction_lbfgs.py)

The retained records include the archive SHA-256 values, configuration,
optimization diagnostics, evidence terms, comparison gates, and reconstruction
checksums. The operator report also records the exact dense products used.
[Dense report](../../.scratch/lbfgs-potential-correction/output/j0946-qualified/dense/result.json) ·
[Dense reconstruction](../../.scratch/lbfgs-potential-correction/output/j0946-qualified/dense/reconstruction.npz) ·
[Operator report](../../.scratch/lbfgs-potential-correction/output/j0946-qualified/operator/result.json) ·
[Operator reconstruction](../../.scratch/lbfgs-potential-correction/output/j0946-qualified/operator/reconstruction.npz)

## Resources

Each first fit ran in its own CUDA process on an NVIDIA GeForce RTX 5080 with
JAX 0.7.1, SciPy 1.17.1, NumPy 2.4.6, Float64 arithmetic, highest matmul
precision, and device preallocation disabled. The timings are synchronized
wall-clock measurements that include first-use compilation. The first objective
and gradient evaluation is part of the optimizer stage. The memory metric is
JAX device allocator `peak_bytes_in_use` from initialization through the
synchronized fit. [Matrix-free benchmark and measurement method](potential-correction-lbfgs-operator-resources.md)

| Stage or resource | Dense | Operator |
| --- | ---: | ---: |
| Archive read, s | 0.011 | 0.011 |
| Synchronized preparation, s | 2.808 | 2.934 |
| Fixed reference, s | 2.607 | 0.905 |
| Initial objective, s | 0.306 | 0.311 |
| Whitening, s | 0.766 | 0.265 |
| Optimizer, s | 2.264 | 1.817 |
| First objective and gradient within optimizer, s | 0.978 | 0.644 |
| Endpoint, s | 2.541 | 2.724 |
| Result assembly, s | 1.382 | 1.502 |
| Total first fit including compilation, s | 9.872 | 7.532 |
| Peak device bytes in use | 3,216,142,592 | 194,374,912 |

## Reproduction

Run from the repository root with the original local M3 handoff and
`cubic_evidence_joint_v1` archive. Use new output directories for the two
processes.

```bash
source ~/anaconda3/bin/activate && conda activate tinylens_gpu
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1 NVIDIA_TF32_OVERRIDE=0
saved="$PWD/workspace/iteration"
qualified="$PWD/.scratch/lbfgs-potential-correction/output/j0946-qualified-repeat"

python -m examples.potential_correction.validate_j0946_lbfgs \
    --macro-dir "$saved/macro" \
    --archive-dir "$saved/cubic_evidence_joint_v1" \
    --backend dense --output "$qualified/dense"

python -m examples.potential_correction.validate_j0946_lbfgs \
    --macro-dir "$saved/macro" \
    --archive-dir "$saved/cubic_evidence_joint_v1" \
    --backend operator --dense-reference "$qualified/dense" --pcg-max-iter 2000 \
    --output "$qualified/operator"
```

Run `pytest -xq test_potential_correction_lbfgs_j0946.py` from
`examples/tests/` for the regenerated provenance and comparison checks. Run
`pytest -xq test_potential_correction_lbfgs.py::test_operator_fixed_strengths_qualify_against_principal_block_reference`
from `tests/` in a separate process for the determinant check.
