# J0946 dense L-BFGS-B refit

A fresh Float64 cubic Hermite fit passed all trial 497 acceptance gates on the
NVIDIA GeForce RTX 5080. The local Gauss–Newton evidence uses the full
data-space determinant ratio. The fit used fixed strengths
`(0.0002532641285041916, 538677.9228218549, 10572.677018080045)` in source,
fourth-difference, and amplitude order.

| Quantity | Trial 497 | Fresh dense fit | Absolute difference | Limit |
| --- | ---: | ---: | ---: | ---: |
| Objective | 605.0380047097701 | 605.0380047097673 | 2.84e−12 | 1e−3 |
| Local GN log evidence | 6307.427179447846 | 6307.427179001290 | 4.47e−7 | 0.01 |
| Convergence over all 802 valid nodes | | | 3.29e−8 max | 1e−4 |

L-BFGS-B returned native success with status 0 after 135 iterations and 143
function and gradient evaluations. The endpoint stationarity statistic was
9.63e−13 at a threshold of 1e−3. The peak convergence was 0.174331760 at
((-0.7'', 1.1'')), the same location as in trial 497. The validator checked
potential node indices, coordinates, and the derivative-valid mask before
comparing fields.

## Inputs and products

The refit uses M3 retained sample 21342 with its fixed EPL and shear parameters,
lens-light-subtracted image, noise, mask, 15 × 15 PSF, and the first-order
adaptive source scale with `rho=5.530494995245579`. Native 201 × 201 image
sampling at 0.05 arcsec and `nsub=4` uses Float32 before the renderer promotes
the coordinates. The fitted source grid has 100 × 100 nodes and the potential
grid has spacing 0.1 arcsec and 1,134 coefficients. The observation contributes
2,293 active pixels.

The source-only start is the source segment of `initial_values.npz:joint_values`.
The potential starts at zero. The distinct `saved_M3_source` field supplies
the hard-box bilinear whitening reference. The fit uses Hermite interpolation
through the entire optimization.

The validator pins all six input SHA-256 hashes.
[Validation entry point](../../examples/potential_correction/validate_j0946_lbfgs.py)

The retained comparison records those digests, source bounds, exact macro
parameters, optimizer controls, and determinant terms.
[Comparison and fit metadata](../../.scratch/lbfgs-potential-correction/output/j0946-dense/result.json)

The reconstruction contains raw fields, valid-domain metadata, the nonlinear
image, and residuals. Its SHA-256 is
`5552ac1f3d96706bc8ac90e34cfb5a27b4b10f4882096ca2132395446d4deebd`.
[Retained reconstruction](../../.scratch/lbfgs-potential-correction/output/j0946-dense/reconstruction.npz)

## Reproduction

Run from the repository root with the original local M3 handoff and
`cubic_evidence_joint_v1` archive. Use a fresh output directory.

```bash
source ~/anaconda3/bin/activate && conda activate tinylens_gpu
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1 NVIDIA_TF32_OVERRIDE=0
saved="$PWD/workspace/iteration"
python -m examples.potential_correction.validate_j0946_lbfgs \
    --macro-dir "$saved/macro" \
    --archive-dir "$saved/cubic_evidence_joint_v1" \
    --output .scratch/lbfgs-potential-correction/output/j0946-dense-repeat
```

The recorded first fit took 9.88 s including compilation, with 2.86 s of
synchronized setup before fitting. Fit stages were 2.60 s for fixed-reference
preparation, 0.30 s for the initial objective, 0.76 s for whitening, 2.27 s for
the optimizer, 2.57 s for the endpoint, and 1.36 s for result assembly. The first
objective-and-gradient evaluation took 0.97 s within the optimizer stage. The
JAX allocator reported a peak of 3,216,142,592 bytes in use through the fit.
The run used JAX 0.7.1, SciPy 1.17.1, NumPy 2.4.6, Float64 fit arithmetic,
highest matmul precision, and disabled device preallocation.

The example regression uses regenerated inputs for archive rejection and
convergence comparisons. Run `pytest -xq test_potential_correction_lbfgs_j0946.py`
from `examples/tests/`.
