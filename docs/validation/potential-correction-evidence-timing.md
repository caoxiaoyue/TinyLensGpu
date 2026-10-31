# One source–potential inversion plus evidence: measured cost

Measured 2026-10-30 on an **NVIDIA GeForce RTX 5080, float64, JAX 0.7.1**.
The current 200 × 200 subhalo mock has 22,016 active pixels, 2,401 source
coefficients and 956 potential coefficients: **3,357 joint unknowns**. The
macro, 49 × 49 source geometry, potential spacing 0.075 arcsec, PSF, seed 17
noise and nsub = 2 match the saved scientific results.

One evaluation means a fixed (λs, λ4), a complete joint source/potential fit
and its evidence. The dense backend uses full Cholesky determinants. For
iteration, one evaluation includes **all inner iterations to convergence**.
Single-step evidence is conditional linear Gaussian; iterative evidence is
local Gauss–Newton, with full determinants of the local curvature.

| Configuration | First fit, including JIT / lazy preparation (s) | Warm median (s) | Warm range (s) | Warm repeats | Accepted nonlinear updates |
| --- | ---: | ---: | --- | ---: | ---: |
| Original single-step | 5.292 | **0.645** | 0.645–0.677 | 5 | — |
| Selected single-step | 5.269 | **0.649** | 0.649–0.680 | 5 | — |
| Original iterative | 38.719 | **33.819** | 33.815–33.858 | 3 | 46 |
| Selected iterative | 19.889 | **14.905** | 14.897–14.943 | 3 | 19 |

One-time worker setup, including imports, device/geometry and PSF preparation,
takes **3.46–3.53 s** separately. Setup plus first fit is 8.82 / 8.80 s for
the two single-step settings and 42.19 / 23.35 s for the two iterative
settings. Mock/partition fixture generation, plotting, export and post-fit
diagnostic audits are outside these fit timings. First minus warm is not a
measurement of compiler time alone.

Settings reproduce the already recorded reconstructions:

| Configuration | λs | λ4 | λ0 | Iterative gradient atol / maximum iterations |
| --- | ---: | ---: | ---: | --- |
| Original single-step | 0.0006 | 16,000,000 | 10,000 | — |
| Selected single-step | 0.00006 | 400,000 | 10,000 | — |
| Original iterative | 0.0006 | 16,000,000 | 10,000 | 0.0005 / 80 |
| Selected iterative | 0.0006 | 800,000 | 10,000 | 0.00003 / 120 |

The shorter selected nonlinear runtime reflects fewer updates under its
recorded priors and stopping controls. It is not a faster implementation.
These measurements are costs at two parameter points, not bounds on the cost
of every hyperparameter candidate or a measurement of complete optimization.

Each configuration runs in a separate GPU process, serially, with persistent
JAX compilation caching disabled and preallocation off. Every public result
and grid array is synchronized before stopping the timer. Warm evaluations
reuse the fixed geometry, frozen single-step design and compiled kernels;
**each call resets the original fields**, solves again and recomputes
evidence. Numerical matrices are refreshed where required by the algorithm.

All **20 dense timed fits succeeded**. Cold and warm evidence agree with their
saved scientific references within **2.92 × 10⁻¹¹** in log evidence. Maximum
field discrepancy in the final warm exports is **1.41 × 10⁻¹²**. Original
observation/noise/PSF/mask/initial-source arrays are checked for exact equality.
The compact [measurement record](../../.scratch/potential-correction/research/evidence-timing/summary.json)
retains every warm sample, effective controls, counts, source/reference hashes
and software versions. Three or five warm repeats characterize these observed
runs; they do not establish a broad performance distribution.

Reproduce from the repository root, using a fresh output directory:

```bash
source ~/anaconda3/bin/activate && conda activate tinylens_gpu
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_COMPILATION_CACHE=false \
  python .scratch/potential-correction/experiments/benchmark_evidence_evaluation.py \
  --output .scratch/potential-correction/output/evidence-evaluation-benchmark-new
```

The driver reuses `benchmarks/potential_correction_workflow.py`; it imports no
JAX in the coordinator and validates measured reconstructions against the
original and selected public exports. Full input/worker JSON/NPZ/log files are
retained in ignored `output/evidence-evaluation-benchmark-2026-10-30/`.

After timing completed, the worker's isolated workflow regression passed
**1 test in 8.18 s**, run separately from `benchmarks/`:

```bash
pytest -q test_potential_correction_workflow.py::test_isolated_workflow_reports_synchronized_phases_and_preparation_peak
```

The compact record includes this verification command and log hash. No
library algorithm or benchmark-worker implementation changed for this run.

## Matched operator backend

A second serial campaign uses **the exact same immutable fixtures**, hardware,
float64 precision, physical priors, source/potential initialization and
nonlinear controls as the dense measurements above. It preserves the packaged
nine-block partition (cap 1,024 coefficients), PCG rtol 10⁻¹² and budget
20,000. Actual-system acceptance remains solve rtol 10⁻⁸, with solve atol
5 × 10⁻⁵ for nonlinear iteration. No tolerance is relaxed for speed.

| Configuration | Dense warm median (s) | Operator warm median (s) | Operator warm range (s) | Operator first fit including JIT (s) | Warm repeats |
| --- | ---: | ---: | --- | ---: | ---: |
| Original single-step | 0.645 | **0.279** | 0.276–0.280 | 6.462 | 5 |
| Selected single-step | 0.649 | **0.214** | 0.210–0.217 | 6.416 | 5 |
| Original iterative | 33.819 | **39.497** | 39.491–39.518 | 46.565 | 3 |
| Selected iterative | 14.905 | **16.434** | 16.431–16.464 | 23.484 | 3 |

In this 3,357-unknown problem, the operator single-step evaluation is
**2.31× / 3.03× faster** for original/selected settings. Operator nonlinear
evaluation is **16.8% / 10.3% slower**. Both nonlinear backends accept the same
46 / 19 updates. Operator setup takes 3.43–3.65 s separately. First fit plus
setup is approximately 10.12 / 9.96 s for single-step and 50.00 / 26.95 s
for iteration. These are implementation-specific measurements at this size,
not asymptotic scaling or a universal operator speed claim.

Every call restarts the fields. The warm single-step operator reuses fixed
data-curvature components and RHS, assembles the new prior/posterior blocks,
factors them, solves the full coupled system and recomputes evidence. Current
data blocks and derivatives refresh during nonlinear iteration.

All **20 operator timed fits succeeded**. Their final fields match the dense
results with maximum discrepancy **below 7.4 × 10⁻¹⁰**; cold/warm objectives
meet the declared 0.001 comparison guard. Original-setting evidence also
reproduces the existing operator scientific exports. Selected-setting
operator evidence is checked for cold/warm repeatability.

The evidence computations have different determinant contracts: operator
uses **principal-block Cholesky approximations**, while dense uses full
coupled Cholesky determinants. The fitted fields still solve the full
coupled problem. Operator-minus-dense log evidence is approximately
108.526 / 66.309 for original/selected single-step and 108.508 / 95.218 for
original/selected iteration. This offset varies with settings; identical
fields do not establish equivalent evidence rankings over hyperparameters.
Timing therefore compares the complete existing implementations, rather than
identical determinant accuracy.

The compact [operator record](../../.scratch/potential-correction/research/evidence-timing/operator-summary.json)
retains all timing samples, settings, numerical checks and unchanged dense
fixture/worker hashes. No numerical-library or shared benchmark-worker code
changed. After producing a dense run, reproduce with a fresh operator output:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_COMPILATION_CACHE=false \
  python .scratch/potential-correction/experiments/benchmark_operator_evidence_evaluation.py \
  --dense-run .scratch/potential-correction/output/evidence-evaluation-benchmark-2026-10-30 \
  --output .scratch/potential-correction/output/evidence-evaluation-operator-benchmark-new
```

Full worker JSON/NPZ/logs are retained in ignored
`output/evidence-evaluation-operator-benchmark-2026-10-30/`. The immutable
input fixtures remain in the matched dense run's `fixtures/` directory.
