# Source–potential precision and GPU workflow measurements

Float64 reproduces both completed reconstruction algorithms in both backends.
Float32 is supported here for the named **coupled37** numerical configuration
at the declared accuracy below. All eight matched 200×200 scientific float32
fits fail their existing numerical gates and retain null evidence. Their
elapsed times are failure diagnostics, not speedups.

A later [single-step-only Float32 comparison](potential-correction-single-step-float32.md)
keeps these strict failures and adds a separately declared `solve_rtol=1e-5`
probe with unchanged priors. Subhalo fits pass that probe in both backends;
the strong-prior controls still fail. Its successful warm timings apply only
to that explicitly relaxed residual certificate.

The [compact measurement record](../../.scratch/potential-correction/research/precision-performance/summary.json)
contains 45 isolated workers, numerical comparisons, dtype audits, input and
implementation hashes, failed statuses, and the rejected calibration.
[Declared budgets](../../.scratch/potential-correction/research/precision-performance/numerical-budgets.json)
and the [reproduction driver](../../.scratch/potential-correction/experiments/validate_precision_performance.py)
are versioned. Large inputs, per-worker JSON/NPZ/logs, process-memory samples,
and measurement-driver source snapshots identified by SHA256 are retained in
the root checkout's ignored
`.scratch/potential-correction/output/precision-performance/`.

## Reference order and matched inputs

The campaign first audited all eight saved float64 scientific results from
tickets 11/12, then established four fresh strict float64 coupled37 references.
Both algorithms and both backends passed before float32 comparisons began.
Eight fresh scientific float64 workers also passed before the eight matched
scientific float32 workers. The reference gate records the saved artifact
hashes; it does not substitute old numerical results for new performance runs.
The [fresh-reference comparison](../../.scratch/potential-correction/research/precision-performance/reference-reproduction.json)
reproduces the canonical fields to maximum source difference 9.1e-11 and
log-evidence difference 1.72e-8 across all eight fits.

Every comparison loads the same saved float64 observations, noise, PSF,
initial source and frozen single-step gradient and casts these to the
requested dtype. It does not regenerate an independent float32 observation.
The macro, mask, source box, potential spacing, initialization, all three
priors, and the partition are fixed within a comparison. Iterative gradients
still refresh from the current pixelized source. Geometry is constructed at
the requested precision: float64 topology/weights are not imposed on float32.

For the 200 case, integer core/support sets remain identical and the exact
recorded partition is reused after validating its float64 geometry and the
actual float32 integer topology. Float32 changes **3464 of 88064** query-corner
choices at grid lines. These expected rounding-dependent bilinear branches,
query/source-coordinate hashes and unchanged node sets are recorded. No
geometry union, spacing change, layout substitution or source-box expansion
is used to make hashes match.

## Named small precision configuration

Coupled37 is the existing nonzero-coupling numerical model: 4×4 data at
0.1 arcsec/pixel, a delta PSF, nsub=1, four source nodes in `[-0.5,0.5]^2`,
33 fitted potential nodes at spacing 0.2, no macro deflection, and the same
representable nodal observation plus fixed sinusoidal residual. Source
regularization is zero-order; `(lambda_s,lambda4,lambda0)=(0.3,0.02,20)`.
It is distinct from the physical scientific Pseudo-Jaffe configuration.

The named `small32` profile uses the following explicit controls for **both
float64 and float32**. Public solver defaults are unchanged.

| Control | Value |
| --- | ---: |
| true linear relative tolerance | 1e-5 |
| iterative true linear absolute tolerance | 1e-6 |
| recursive PCG relative tolerance / budget | 1e-8 / 400 |
| nonlinear gradient absolute / relative tolerance | 1e-4 / 1e-5 |
| objective / scaled-update tolerance | 1e-6 / 1e-4 |

The original strict float32 public test was reproduced red at `rejection_limit`
in 8.39 s before adopting this named configuration. Earlier diagnostic
captures show positive independent decreases around 1.4e-9 while a float32
objective near 0.44638 has ULP **2.98e-8**. A saved nearby state has physical
scaled gradient 8.67e-5 and remaining objective decrease 2.25e-8. Rounded
forward images also affect the reduction sign; changing scalar subtraction
alone was not established as a sufficient fix. No controller rewrite or
post-hoc convergence reclassification was made.

The independent original guard remains **atol=2e-4 and rtol=2e-4** for source,
potential, nonlinear image and objective against strict same-method float64.
This is an absolute-plus-relative guard, not a claim of an absolute 2e-4
bound on every source coefficient. A predeclared evidence budget is
`abs(delta logZ)<0.002` against that same-method reference. One cold and five
warm fits per algorithm/backend/dtype pass both guards. These are observed
repeats with reused kernels, not a failure-rate estimate.

An initial PCG `rtol=1e-5` shared profile was **rejected**: single-step operator
source error was 0.00832 even in float64 (0.00836 in float32), beyond the
unchanged guard. Its records remain in `rejected-calibration-pcg1e-5/`.
Tightening PCG to 1e-8 removes that error. Measured float32 dense residuals
near small RHS norms reach 6–7e-7, motivating the named iterative absolute
certificate of 1e-6; the independent field/evidence checks still apply.

| Same controls: float32 minus float64 | max source | max potential | max nonlinear image | delta logZ |
| --- | ---: | ---: | ---: | ---: |
| single-step dense | 1.275e-4 | 1.113e-5 | 1.761e-5 | -2.810e-5 |
| single-step operator | 2.497e-4 | 1.643e-5 | 4.783e-5 | -6.923e-6 |
| iterative dense | 2.351e-7 | 4.887e-8 | 1.031e-7 | +1.302e-5 |
| iterative operator | 2.698e-7 | 3.713e-8 | 1.377e-7 | -5.125e-6 |

Stopping accuracy is separate from dtype accuracy: named-control **float64**
iteration differs from strict float64 by maximum source 1.2733e-4, potential
2.1900e-5, image 1.6814e-6, and objective 2.2551e-8. Local/full and local/block
log evidence changes are -0.0004094 and -0.0013542. Thus the corresponding
float32-versus-strict differences are principally the declared stopping
accuracy, not hidden dtype loss. These controls are not transferred to the
scientific model.

## Dtypes and bounded strength selection

Actual query/source coordinates, interpolation weights, gradients, forward,
transpose and prior actions, source/potential/image outputs, objectives,
normalization and determinant terms have the requested real dtype. The PSF
FFT is complex128 for float64 and complex64 for float32. Core public-operator
tests also request each dtype inside the *opposite* ambient x64 setting.
The surrounding x64/matmul defaults remain unchanged; the measurement workers
start and end at `(False, None)`. This audits the stated public/intermediate
numerical contracts, not every compiler temporary.

DIRECT coordinates/bounds remain host float64 even for float32 fits. In the
small physical model, natural-log bounds correspond to source strengths
`[0.25,0.35]` and fourth-difference strengths `[0.015,0.025]`, with lambda0=20.
Each of the eight searches evaluates 27 valid candidates for the nominal
25-evaluation soft budget. Every candidate resets the same source and zero
potential. All select the same sampled point `(0.34782594,0.02476462)`;
recorded boundary sides are null. All stop at the budget with
`optimizer_success=False`: these are best sampled candidates, not claims of
a converged global optimum.

Float32-minus-float64 selected log-evidence differences have magnitude at most
1.20e-5, and selected fields meet the same guard. Within float64, operator
minus dense log evidence is **-10.4183** for single-step and **-9.33312** for
iteration at the selected point, despite nearly identical fitted fields.
Matching selection in this small bounded box does not make the evidence
methods interchangeable or establish unrestricted strength-selection accuracy.
Single-step dense evidence is conditional linear Gaussian; iterative evidence
is a local Gauss–Newton approximation, with a separate principal-block
determinant approximation in the operator backend.

## Scientific float32 limits

The scientific configuration is exactly the delivered 200×200 case: 22016
data, 2401 source, 724 core C, 956 support U and 88064 subpixel queries;
mock/inversion nsub=2/2, source box `[-0.4,0.4]^2`, potential spacing 0.075,
Gaussian PSF sigma 0.05 with 27×27 samples, noise 0.01 and seed 17. The true
macro and Pseudo-Jaffe mock/control are unchanged. Smoothing is
`lambda_s=0.0006`, lambda4 `1.6e7/1e11` for subhalo/control and lambda0=10000.
Both dtypes use the same recorded partition and PCG `rtol=1e-12`, budget 20000.
Iterative example-only controls remain gradient `5e-4/3e-5`, gradient relative
1e-8, objective relative 1e-10, update relative 1e-7, and true residual
`5e-5 + 1e-8*norm(rhs)`.

| Float32 case | single-step dense / operator | iterative dense / operator |
| --- | --- | --- |
| subhalo | true residual rejected / true residual rejected | linear solve failed / linear solve failed |
| control | prior factorization failed / true residual rejected | prior factorization failed / linear solve failed |

All eight have null evidence. Subhalo single-step true relative residuals are
5.62e-7/9.75e-7 against 1e-8; control operator is 0.001856. Iterative failures
retain the actual absolute-plus-relative certificate and phase diagnostics.
For example, failed subhalo single-step source differences are 8.09e-4 dense
and 2.00e-4 operator; nonlinear residual differences are 0.00369/0.000458 RMS
in noise units. These are diagnostic endpoints, not successful reconstructions.
The complete record includes all fields, residuals and prior/posterior terms.
No prior or tolerance was changed to rescue these fits.

This is a limitation of the stated stiff prior/discretization/solver
configuration. It does not prove all float32 correction models fail. Preserve
the separate [iterative scientific validation](potential-correction-iterative-scientific.md)
and its amplitude-prior limits: 10000→30000 has stable localized kappa but
prior-sensitive raw potential/deflection and changing evidence; lambda0=3000
failed with null evidence. This campaign does not broaden that sensitivity
claim or establish a universally weak amplitude prior.

## Timings and memory on the RTX 5080

Measured context: NVIDIA GeForce RTX 5080, 16303 MiB, driver 610.57.04;
Python 3.11.14, JAX/jaxlib and CUDA plugin/PJRT 0.7.1, NumPy 2.4.6,
SciPy 1.17.1, CUDA runtime 12.8.90, cuBLAS 12.8.4.1 and cuDNN 9.10.2.21.
Preallocation and persistent compilation caching are disabled. The allocator
choice/fraction are left at their recorded defaults; reported device allocator
limit is about 12.49 GB, distinct from total VRAM.

The parent imports no JAX and launches one fresh worker per actual
case/algorithm/backend/dtype, serially. The first public fit includes lazy
backend preparation, compilation, solve/evidence, Python nonlinear control
and scalar round trips, history and derived images/maps. Setup includes input
loading and static geometry. Cold workflow spans setup through the synchronized
first complete numerical result; audit/export costs and launch-to-exit process
wall time are separately retained. Observations, initial fields and the fixed
partition are inputs to this cold reconstruction workflow. Mock simulation
and source-incidence partition construction belong to the separate fixture
preparation command and are outside these timings. Warm fits reuse prepared
state and kernels while resetting the
original fields. Every result/grid dataclass array is explicitly synchronized;
waiting for `source_values` or an unregistered result dataclass alone is
insufficient. First-minus-warm is not pure compiler time.

Each large row has one warm repetition, with no variance claim. All times are
seconds. Allocator/process peaks below are MiB.

| 200 case / algorithm / backend | setup | first fit | warm fit | cold workflow | allocator peak | sampled process peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| subhalo single-step dense | 3.655 | 5.284 | 0.675 | 8.938 | 1505 | 2468 |
| subhalo single-step operator | 3.499 | 6.448 | 0.283 | 9.947 | 1989 | 2874 |
| subhalo iterative dense | 3.439 | 38.747 | 33.873 | 42.186 | 1633 | 3456 |
| subhalo iterative operator | 3.452 | 46.526 | 39.520 | 49.978 | 2238 | 3054 |
| control single-step dense | 3.494 | 5.239 | 0.688 | 8.733 | 1505 | 2468 |
| control single-step operator | 3.485 | 7.453 | 1.293 | 10.937 | 1989 | 2874 |
| control iterative dense | 3.399 | 18.923 | 13.984 | 22.322 | 1634 | 3456 |
| control iterative operator | 3.447 | 39.799 | 33.151 | 43.246 | 2238 | 3054 |

There is no universal operator speed or memory advantage at this layout.
Operator allocator peak exceeds dense here; sampled process residency can
have a different ordering because it includes context and retained pools.
Dense and operator determinant work is also different. Small supported
float32/F64 warm medians are retained without extrapolation: for example,
iterative dense 0.0113/0.0128 s and iterative operator 0.0310/0.0248 s.
Float32 is not uniformly faster even in the supported small case.

`peak_bytes_in_use` is the cumulative JAX allocator high-water mark, including
preparation and execution in that worker. Snapshots after setup/first/warm do
not constitute independent phase peaks; peaks are not subtracted. Process GPU
residency is sampled at 100 ms for that PID/UUID and is a lower bound on the
true instantaneous process peak, including non-JAX allocations/reserved pools.
Host RSS, allocator pool statistics and live-array shapes are separate values.
Live-array snapshots do not reveal dead/transient allocations. Optional memory
profiles/traces are not needed for these primary measurements.

## Actual scaling and allocation evidence

These are actual observations with a 2.4-arcsec field, nsub=2/2 and fixed
annular selection, not padding a fixed active image or adding isolated nodes.
At 32/64/200, source sides are 17/33/49, potential spacings .15/.1/.075 and
PSF sides 5/9/27. These are explicitly different discretizations, not claims
that tuned strengths or scientific recovery transfer automatically.

| image | active data | source | C / U | queries | single-step allocator dense / operator MiB | iterative allocator dense / operator MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32×32 | 568 | 289 | 200 / 312 | 2272 | 26.5 / 15.1 | 29.3 / 20.7 |
| 64×64 | 2252 | 1089 | 416 / 592 | 9008 | 223.6 / 208.5 | 241.5 / 191.3 |
| 200×200 | 22016 | 2401 | 724 / 956 | 88064 | 1505 / 1989 | 1633 / 2238 |

All twelve subhalo float64 size/algorithm/backend fits converge. Source tile
and block caps are 8/128, 16/512 and 24/1024 respectively. Every partition is
saved. At fixed **64×64 observations**, a separately named source axis copies
image/noise/PSF/mask bit for bit and changes source side 33→65
(1089→4225 source coefficients), holding potential spacing .1 and C/U=416/592.
The operator single-step fit converges: first/warm 3.696/0.242 s, allocator
peak **218.5 MiB** versus 208.5 MiB for side 33, sampled process peak 712 MiB.
Its partition is rebuilt for the explicitly changed source grid, with the
same bounded cap 512; this is a storage-path check, not a new recovery study.

No full source/potential/joint square, cross-field rectangle or complete
joint-by-data design is observed among live operator arrays in these real
cases. Dense reference storage is observed as intended. Allocation-path review
also confirms lazy source differences, compact potential stencils and bounded
scan panels/basis blocks; neither operator backend calls complete dense
prior/design assembly. Principal-block storage and temporary image panels
can dominate these sizes. Peak numbers alone would not prove absence of a
temporary dense matrix. The separate synthetic source96²/isolated-support
stress remains complementary evidence and does not replace this image/source
scaling. No OOM was induced or claimed.

## Reproduction and verification

Activate `tinylens_gpu`, set `XLA_PYTHON_CLIENT_PREALLOCATE=false`, and use a
fresh output directory to retain the existing evidence. `PC_CANONICAL` points
to the root checkout's existing ticket 11/12 output directory. From the repo:

```bash
PC_CANONICAL=/path/to/root/.scratch/potential-correction/output
PC_PRECISION=/path/to/fresh/precision-performance
for phase in fixtures small-reference small-precision scientific scaling source-axis search; do
  JAX_ENABLE_COMPILATION_CACHE=false python \
    .scratch/potential-correction/experiments/validate_precision_performance.py \
    --phase "$phase" --output "$PC_PRECISION" --canonical-output "$PC_CANONICAL"
done
python .scratch/potential-correction/experiments/validate_precision_performance.py \
  --phase summary --output "$PC_PRECISION" --canonical-output "$PC_CANONICAL" \
  --summary /path/to/fresh/summary.json
```

Existing records are retained rather than overwritten. The summary enforces
the named small field/evidence guards, records all scientific float32 failures,
and keeps numerical, determinant-approximation and stopping differences
separate. Worker crashes/timeouts are distinct from numerical fit failures.
The supported profile can be invoked directly with saved coupled37 inputs:

```bash
python benchmarks/potential_correction_workflow.py --inputs /path/to/coupled37.npz \
  --algorithm iterative --backend operator --dtype float32 --profile small32 \
  --repeats 5 --output /path/to/fresh/measurement.json
```

Core, example and benchmark verification runs in separate processes, with
specific files run from their own directories. Final command/outcome records
are also retained in issue 13.

- `tests/`: `pytest -q test_potential_correction_*.py`: **132 passed** in 188.69 s.
- `examples/tests/`: `pytest -q test_potential_correction_*.py`: **30 passed** in 90.35 s.
- `benchmarks/`: `pytest -q -s test_potential_correction_memory.py
  test_potential_correction_preconditioner.py test_potential_correction_workflow.py`:
  **5 passed** in 37.46 s. This includes the complementary bounded-support
  allocation probe, independent synthetic preconditioner application benchmark,
  isolated phase/metadata checks and actual fixed-image source-size axis.
- Black default-format checks, Ruff default checks on the new Python files,
  and `git diff --check` pass.
- The five focused dtype/supported-stopping checks pass again after the final
  test-contract edit: **5 passed** in 14.16 s.
