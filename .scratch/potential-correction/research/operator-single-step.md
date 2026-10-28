# Fixed-strength operator source–potential reconstruction

Ticket [06](../issues/06-operator-single-step-reconstruction.md), implemented
on `codex/pc-06` starting from integration `cdda3e9`. Implementation commit:
`20b5ebd`; integrations `4301d8b` and `f7d96e5` were merged before completion.

## Public API and numerical model

The existing `SingleStepPotentialCorrection` accepts
`fit(lambda_source, lambda_potential, backend="operator", ...)`, keeping the
same geometry, frozen source gradient, signed variables, complete declared
prior, imaging semantics and public result as `backend="dense"`.

Operator controls are `block_size=64`, optional `block_indices`,
`pcg_max_iter=400`, `pcg_rtol=None`, and `pcg_atol=0.0`. The constructor's
`solve_rtol` remains the final coupled-system relative-residual acceptance
threshold (1e-8 for float64, 1e-3 for float32 by default). Default PCG rtol is
the smaller of one tenth of that threshold and 1e-10/1e-5 respectively.
No candidate warm-start is used.

The curvature is the full `M.T(noise**-2 * M(x)) + Q(x)`, where the joint
forward first adds source and potential responses. This retains source–potential
cross terms. Principal blocks approximate only preconditioning and determinants.
The exact residual is recomputed from this complete curvature after PCG;
recursive PCG convergence alone cannot produce a successful result.

The default partition separates fields in their established node order, with
at most `block_size` coefficients per block and at least two blocks per field.
An explicit complete disjoint partition may mix fields. Full joint/potential
blocks are rejected. Configuration records the partition, block sizes/count,
partition digest, solver controls and recursive versus recomputed residuals.

Preparation scans bounded blocks. Each block forms only
`one_hot(block_indices, n_joint)` and its image panel, then stores the bounded
likelihood, unit-source-prior and unit-fourth-difference principal blocks.
Full stencil actions are restricted to a block, so windows crossing block
boundaries contribute correctly. Cached components combine with dynamic
strengths without retracing. There are no full difference, joint curvature,
potential curvature, source-by-potential cross or joint design matrices on
this path. Dense materialization remains an explicit reference operation.

For every block, `Q_B` includes the complete sum
`lambda_source * Rs_B + lambda_potential * D4_B + lambda_amplitude * Ipot_B`.
Every source and potential support coefficient belongs to exactly one block.
Padding is outside the physical model, has identity diagonal, zero response,
and zero log determinant. A dedicated padded principal-block preconditioner
uses masked scatter-add to avoid overwriting real coefficients with padding.
The existing source–lens-light Schur representation is unchanged.

Evidence uses the exact coupled fit, chi-squared and prior penalty with
`sum(logdet(H_B)) - sum(logdet(Q_B))`. Cholesky factors use diagonal
equilibration, without changing the physical prior or adding jitter. Results
declare `evidence_method="conditional_linear_gaussian"`,
`determinant_method="principal_block_cholesky"`, `evidence_is_exact=False`.
The determinant calculation has an internal replacement boundary; SLQ,
Lanczos and source–potential Schur construction are not implemented.

Block Gram products and solve/evidence kernels locally request highest matmul
precision. Both explicit dtypes preserve the process's x64 and matmul defaults.
Repeated strength evaluations after warm-up produce no JAX compilation log
entries in the public-fit test.

Invalid configuration raises `ValueError`. Numerical failures return failed
public results with no evidence and explicit reasons for nonfinite systems,
failed factors, PCG breakdown, iteration limits, or unacceptable actual
residuals. The zero-data test exposed a generic PCG edge case at zero absolute
tolerance: exact residual zero now satisfies `<= tolerance` and converges
without iterations. Existing solver and source–lens-light regressions cover
this shared change.

## Fixed-strength numerical comparisons

Environment: `tinylens_gpu`, Python 3.11.14, JAX 0.7.1, NVIDIA GeForce RTX 5080.
The independent 3x3 test uses source2, dpix=0.1, potential spacing=0.2,
four source and 33 potential coefficients, gradient (1.2,-0.8), noise
standard deviations 0.2–0.4, lambda_source=1.3 and lambda_amplitude=0.4.
Field-separated cap8 blocks have sizes (2,2,8,8,8,8,1). The fixture has
nonzero cross-curvature and signed coefficients; an uncoupled solve moves
the source by more than 0.2 and fails the comparison.

A measured matched-strength check used PCG rtol=1e-12 in float64 and 1e-5 in
float32 (zero atol), with the following errors versus dense. The commands in
the tests recreate the same model; GPU reductions can vary at solver accuracy.

| dtype | lambda_potential | PCG iterations | actual relative residual | max source error | max potential error | max linear-image error | max nonlinear-image error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| float64 | 0.03 | 39 | 4.54e-14 | 1.01e-13 | 2.04e-13 | 2.19e-13 | 8.52e-14 |
| float64 | 2.0 | 42 | 2.99e-13 | 8.15e-14 | 8.10e-13 | 2.75e-13 | 2.60e-14 |
| float32 | 0.03 | 41 | 6.22e-6 | 4.21e-5 | 2.79e-4 | 2.75e-5 | 3.53e-5 |
| float32 | 2.0 | 44 | 7.30e-6 | 6.61e-6 | 5.34e-5 | 6.68e-6 | 5.28e-6 |

Objective absolute differences were at most 1.33e-15 in float64 and 4.77e-7
in float32. These small fixed-strength checks do not establish float32
scientific recovery accuracy.

For the same float64 cases, complete dense log evidence was -15.17135297 and
-11.41326520, while cap8 block evidence was -25.23134864 and -10.81072863.
The approximation error is strength dependent despite matching fitted fields
and objectives. This is evidence for ticket 07's calibration work, not a
claim that block evidence preserves a useful optimum.

Maintained tests also compare an explicit mixed irregular partition with
independently constructed full-prior and full-posterior principal determinants
at lambda_potential=0.03 and 2.0, holding lambda_amplitude fixed. A separate
5x5 nsub=3 case includes a mask hole/disconnected pixels, an asymmetric PSF and
nonzero macro shear; dense/operator fields, both images, objective, selected
residuals and adjoint inner products agree at float64 tolerances. Other cases
exercise dtype preservation, candidate repetition, JIT reuse, nonfinite
strength overflow, failed PCG, and rejection after false solver convergence.

## Observable allocation check

`benchmarks/potential_correction_memory.py` is an isolated reproducible GPU
buffer probe. It uses a 3x3 zero-data image, source96 (9216 coefficients),
33 local potential nodes plus 9216 extra isolated support-only nodes,
18,465 total coefficients, and cap=8 (2309 blocks). Additional nodes are
fitted amplitude-prior unknowns, without changing the valid derivative
domain or adding observations. This is a storage stress test, not an
astrophysical geometry or a representative scientific runtime benchmark.

The CUDA run reported JAX peak buffer bytes in use **13,099,264** and retained
bytes **4,932,352**. The largest live array had **147,776** elements. The
observable live-array scan found no full source, potential or joint square,
cross-field or joint-design shape. A single dense source square would require
679,477,248 bytes in float64. The benchmark checks a generous peak bound
below half of either dense-field square, plus bounded retained arrays.
JAX allocator statistics exclude CUDA context/driver memory and are not a
total `nvidia-smi` process-memory measurement. Allocation-path inspection
confirms only bounded basis/image panels occur inside the scan.

## Runnable example and verification

`examples/potential_correction/single_step_operator.py` uses a 12x12 Pseudo-Jaffe
mock, source5 and fixed strengths (0.01,100,100), cap=16, maxiter=600. It saves
only successful public reconstruction NPZ and JSON products; failures save
diagnostic JSON and raise. The JSON records every block and approximation.
The subhalo and matched control commands completed in 109 PCG iterations,
with actual relative residuals 8.80e-11 and 8.37e-11. Output paths used for the
checks were `/tmp/tinylens-pc-06-example` and `/tmp/tinylens-pc-06-control`;
the script regenerates them. The amplitude value 100 has no demonstrated
scientific weakness or calibrated recovery interpretation.

Activate `tinylens_gpu` and set `XLA_PYTHON_CLIENT_PREALLOCATE=false` for all
commands. Core checks run from `tests/`:

```bash
pytest test_potential_correction_operator.py test_potential_correction_dense.py \
  test_potential_correction_rendering.py test_solver_autodiff.py \
  test_operator_autodiff.py test_curvature_operator.py -q
pytest test_pixelized_operator.py -q \
  -k 'pcg or schur or joint_curvature or ragged_source_blocks or preconditioner_is_spd'
```

The selected pixelized regression run passed **23**, with 173 deselected,
in 36.48s. The 86-case combined run passed 85 and encountered one existing
float32 finite-difference Jacobian failure. That case passed an isolated
rerun both here and on unchanged `cdda3e9`. The retained baseline repeat
diagnostic below confirms the old fixture's variability. With only its PCG
verification tolerance tightened to 1e-8, the full 12-case operator-autodiff
suite passed in 73.13s; model defaults and comparison tolerances are unchanged.
After merging ticket 05, `pytest test_potential_correction_operator.py
test_potential_correction_inference.py -q` passed **31 in 25.42s**.

In a separate process from `examples/tests/`:

```bash
pytest test_potential_correction_operator_example.py -q
```

Result: **1 passed in 6.85s**. In another process from `benchmarks/`:

```bash
pytest test_potential_correction_memory.py -q -s
```

Result: **1 passed in 5.70s**. Both standalone example commands also succeeded,
and `git diff --check` passed. Scientific strength selection, amplitude-prior
sensitivity, 200x200 reconstruction, recovery assessment and total GPU
runtime/memory comparison remain in downstream tickets 07–13.

After merging the tuning example, the separate example command
`pytest test_potential_correction_operator_example.py
test_potential_correction_tuning.py test_potential_correction_mock.py -q`
passed **14 in 14.61s**.

A single-candidate cross-layer smoke check also evaluated the merged
`PotentialCorrectionEvidence(problem, backend="operator",
fit_options={"block_size": 8})` interface successfully and confirmed that
`pcg_max_iter=1` returns positive infinity with retained reason
`pcg_iteration_limit`. This did not perform an operator strength search.

## Existing float32 Jacobian-fixture variability

The combined-suite failure is in the existing 10x10, source5, adaptive joint
source–lens-light PCG model, separate from source–potential reconstruction.
The exact baseline `cdda3e9` fixture was repeated with the existing float32
JIT Jacobian, eager finite differences, and unchanged assertion tolerances
(rtol=0.05, atol=0.03). Initial 20-repeat results were four failures at the
default PCG rtol=1e-5 (maximum assertion-tolerance ratio 1.457), and zero at
rtol=1e-7 (maximum 0.867). The
[initial compact record](legacy-gradient-initial-repeat.json) retains these
exact numerical results. The
[repeat script](../experiments/legacy_gradient_repeat.py) reproduces the
comparison against an explicitly selected checkout, for example:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python \
  .scratch/potential-correction/experiments/legacy_gradient_repeat.py \
  --checkout /path/to/unchanged-cdda3e9 --tight-rtol 1e-7
```

A second 20-repeat baseline probe compared default rtol=1e-5 with the fixture's
new rtol=1e-8; it produced four failures versus one, with maximum tolerance
ratios 1.452 versus 1.077. The full
[second record](legacy-gradient-repeat.json) is retained. Tighter solves
improve this check but do not establish elimination of float32 variability.
Counts are empirical GPU results and can vary between runs. No public model
defaults, source–lens-light preconditioning, finite-difference steps or
assertion thresholds were changed. The retained compatibility limitation
must not be presented as a new operator-backend recovery failure or as a
fully cured baseline numerical issue.
