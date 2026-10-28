# Potential-correction requirement coverage and final integration

Requirements are the approved [parent spec](../../.scratch/potential-correction/spec.md)
and its designated [14 tickets](../../.scratch/potential-correction/issues/).
Research notes and validation reports below are configuration-specific evidence,
not new requirements. This record maps all 33 implementation decisions, 13
testing decisions, 50 user stories and 14 delivery tickets. It preserves
failed probes and numerical/scientific limits. The parent spec remains open
until the coordinator completes the independent final Standards/Spec review
against original baseline `5bbfdfa` and any required fixes.

Ticket 14 starts at integration `5170da8e5ec6f162ff6b66ab7b75b8a4b6fd2bed`.
Implementation is `6cdfbe9`; latest integration `ef80f51` (progress README only)
was merged without conflicts as `14b2fcf`.
The [user guide](../guides/potential-correction.md), root README, main guide,
example README and codebase map provide the public discovery path. Current
final verification is recorded below; earlier per-ticket commands are historical
checks at their named revisions, not a claim that they were all rerun here.

## Implementation, test and evidence index

Paths below are relative to the repository root. Test groups are complete
files, allowing each table entry to be reproduced without private helper tests.

| Key | Concrete implementation/public boundary | Maintained verification |
| --- | --- | --- |
| G | `ForwardSimulation.PotentialCorrectionRenderer`, `TinyLensGpu/utils/lensing/potential.py::PotentialGrid`, `TinyLensGpu/utils/lensing/mapping.py` | `tests/test_potential_correction_rendering.py`: geometry, irregular masks, physical derivatives/sign, transpose, source boundaries, macro seam |
| R | `TinyLensGpu/utils/inversion/{regularization,potential_regularization}.py`, `TinyLensGpu/ObservationModel/LensImage/_potential_correction_common.py` | `tests/test_regularization.py`, `test_pixelized_source_precision.py`, `test_potential_correction_dense.py::test_complete_fourth_windows_include_support_only_runs_and_do_not_bridge_holes` |
| S | `TinyLensGpu.ObservationModel.SingleStepPotentialCorrection` and `TinyLensGpu/ObservationModel/LensImage/potential_correction_operator.py` | `tests/test_potential_correction_{dense,operator}.py`: independent Gaussian solve/evidence, signed coupling, normalization, both images, failure, dtype/JIT |
| I | `TinyLensGpu.ObservationModel.IterativePotentialCorrection`, `PotentialLinearization`, `TinyLensGpu/ObservationModel/LensImage/potential_correction_iterative{,_operator}.py` | `tests/test_potential_correction_{iterative,iterative_operator,residual_refinement}.py`: independent nonlinear MAP, current cells/gradients, acceptance, accumulated priors, final curvature, strict total-budget refinement |
| N | `TinyLensGpu.Inference.PotentialCorrectionEvidence`, `optimize_potential_correction` and optimization/evaluation result types | `tests/test_potential_correction_{inference,iterative_inference}.py`: independent scans, bounds, ordering/reset, invalid candidates, approximation/recovery guards |
| O | `TinyLensGpu.ObservationModel.PotentialCorrectionResult`, `PotentialSolverInfo`, `PotentialCorrectionStrengths`; `examples/potential_correction/result_products.py` | `examples/tests/test_potential_correction_{scientific,iterative_scientific}.py`: safe public export/reload, None linear products, comparison/history, failures/stale-success cleanup |
| M | `examples/potential_correction/{mock,generate_mock}.py`, existing `PseudoJaffeSubhalo` mass component | `examples/tests/test_potential_correction_mock.py`, `tests/test_pseudo_jaffe_subhalo.py`: independent truth relations, singular/zero limits, shared-noise control, persistence |
| P | `tests/test_potential_correction_precision.py`; benchmark workflow/measurement driver | `benchmarks/test_potential_correction_{memory,preconditioner,workflow}.py`: isolated allocation, synchronized workflow metadata, real source-size axis; scientific precision campaign below |
| C | Existing pixelized source/dense/operator/PSF, `cg_solver.py`, `curvature_operator.py`, public exports | `tests/test_{pixelized_source_model,pixelized_source_utils,pixelized_source_precision,pixelized_inversion,pixelized_operator,operator_autodiff,solver_autodiff,curvature_operator}.py`; `test_potential_correction_jax_compatibility.py` |

Evidence keys link exact model/configuration, commands, results and provenance:

- **E01** [Source preparation](../../.scratch/potential-correction/research/source-preparation-verification.md): lazy finite differences/typed bounds, 372 existing regressions and live allocation probe.
- **E03** [Supplied-field geometry/rendering](../../.scratch/potential-correction/research/supplied-field-rendering.md): 17 focused cases and separate supplied-truth 200×200 discretization evidence.
- **E04** [Dense single-step](../../.scratch/potential-correction/research/dense-single-step.md): 31 dense/rendering cases and independent small Gaussian model.
- **E05** [Dense strength search](../../.scratch/potential-correction/research/dense-strength-optimization.md): independent bounded scan, callable/optimizer semantics and reproducible example.
- **E06** [Operator single-step](../../.scratch/potential-correction/research/operator-single-step.md): coupled fixed-strength comparisons, block determinants, lazy/bounded storage and legacy float32-Jacobian variability provenance.
- **E07** [Single-step calibration](potential-correction-single-step-calibration.md): actual 64×64, source49, h=.075, lambda0=10000 adopted among 100/1000/10000; saved layout and bounded searches.
- **E08/E09** [Dense iteration](potential-correction-dense-iteration.md) and [operator iteration](potential-correction-operator-iteration.md): independent small MAP/local evidence, changed-cell/JIT/failure comparisons and bounded allocation. Later refinement/precision findings are recorded in E12/E13.
- **E10** [Nested iterative search](potential-correction-iterative-inference.md): 37-unknown independent 5×5 scan, finite-difference locality check, rejected 19/18 layout and accepted cap20 partition; both searches have 115 evaluations for requested soft maxfun100, upper-D4 boundary and incomplete optimizer status.
- **E11** [Single-step science](potential-correction-single-step-scientific.md) and [configuration](../../examples/potential_correction/scientific_configuration.json): actual 200×200 fixed-strength subhalo/control results, raw/aligned diagnostics and prior probes.
- **E12** [Iterative science](potential-correction-iterative-scientific.md) and [configuration](../../examples/potential_correction/iterative_scientific_configuration.json): matched 200×200 comparisons, original failed controls, bounded residual correction, measured stopping/prior probes and saved final arrays/history.
- **E13** [Precision/performance](potential-correction-precision-performance.md), [compact 45-worker summary](../../.scratch/potential-correction/research/precision-performance/summary.json) and [declared budgets](../../.scratch/potential-correction/research/precision-performance/numerical-budgets.json): matched same-input references, small32, failed scientific float32, actual size/source axes, timing/memory semantics and source/input hashes.

## All implementation decisions

| Spec decision | Delivered behavior and concrete trace | Tests/evidence; boundary |
| --- | --- | --- |
| D01 scope/terminology | G/S/I: one lens-light-subtracted band, fixed macro, source–potential terminology | C/M; no simultaneous lens light or fitted parametric-subhalo substitute |
| D02 architecture/API | Public renderer, distinct S/I classes, shared O; numerical adapters separate from N and example diagnostics | Core public-fit tests and O examples; guide imports |
| D03 compatibility/ADRs | C preserves source–lens-light Schur, unit-amplitude bases and actual-prior semantics | Broad C checks; ADRs 0001–0003; JAX API compatibility regression |
| D04 fixed source | G/R reuse square grid, ordering, bilinear/endpoint/outside conventions and nonadaptive source priors | G/R/S tests; E01/E03; no box expansion/adaptation |
| D05 initialization | S freezes supplied analytic gradient at macro rays; I resets truth source nodes/zero potential then current gradients | S/I/N/O tests; E08/E10/E12 |
| D06 signed fields | S/I solve full real source and potential; O preserves raw negatives | Signed-fit/ridge/precision tests; no clipping/NNLS/PNPG |
| D07 core nodes | G derives C from actual subpixel interpolation corners | G irregular-mask/subpixel tests; E03 |
| D08 constructive support | G builds unique axial ±1/±2 support U and preserves observations | G isolated/hole/edge tests; E03 |
| D09 support/coverage | All U values fitted; derivative validity only C; malformed explicit geometry rejected | G/S/O valid-domain and NaN tests; E03/E11/E12 |
| D10 resolution/coordinates | G defaults h=2*dpix; physical angular derivatives and beta_macro-alpha convention | G independent quadratic/sign tests; scientific configs independently set h |
| D11 physical operators | Identical centered deflection/interpolation and kappa=(psi_xx+psi_yy)/2 across S/I/render | G transpose and derivative tests; I finite-difference Jacobian |
| D12 source gradient | Analytical derivative of current bilinear branch, including endpoint/outside conventions | `test_nonaffine_source_gradient_differentiates_current_bilinear_branch`; changed-cell tests |
| D13 imaging reuse | G reuses native FFT PSF/masking/subpixel kernels for both response arms | G/S asymmetric-PSF, masked and nsub tests; C regressions |
| D14 D4 windows | R enumerates all complete five-node windows in U once, no holes/fallback | Complete-window/low-order tests in S; E03/E04 |
| D15 complete prior | R uses independent source, D4 and all-node amplitude terms; lambda0 fixed in N | S/operator mixed-block determinant tests; E07/E12 sensitivity and failures |
| D16 low modes/gauge | O preserves raw fields, coverage and separately named affine diagnostics; no hard affine gauge | O scientific export tests; E11/E12 raw-mode limits |
| D17 single-step equation | S solves complete signed (source,potential) with frozen coupled design | Independent Gaussian/coupling/sign tests; E04/E06 |
| D18 iterative equation | I uses nonlinear residual, current J and accumulated-state -Q*r RHS | Independent nonlinear MAP/ridge/current-J tests; E08/E09 |
| D19 acceptance/failure | One I controller owns actual decrease, LM policy, stopping, honest terminal failures | I kink/nonfinite/rejection/stall/limit tests; E12 measured controls |
| D20 dense reference | S/I explicitly assemble complete H/Q and Cholesky determinants | Independent Gaussian/MAP/evidence comparisons; E04/E08 |
| D21 operator coupling | Exact current J.T W J+Q actions and PCG; only bounded preconditioner/determinants approximate | S/I coupled comparisons and true residuals; C Schur regression |
| D22 lazy memory | R compact source/potential actions; operator blocks/panels, no full cross/curvature allocation | R allocation tests and P isolated/live/peak checks; E01/E06/E13 |
| D23 normalized evidence | S uses chi2+prior+logdetH-logdetQ+noise terms on every fitted coefficient | Independent Gaussian and lambda4-varying complete-prior determinant tests |
| D24 approximations | O/N distinguish conditional vs local-GN and full vs principal-block determinants; I final H undamped | S/I/N annotation/formula/damping tests; E08–E13 |
| D25 approximation validation | E07 single-step and E10 nested model compare selection/recovery; rejected layouts retained | N independent scan/recovery guards; no SLQ or source–potential Schur |
| D26 outer search | N bounded natural-log DIRECT; complete fresh candidates, fixed lambda0, failed +infinity | N reset/order/boundary/all-invalid tests; E05/E07/E10; scientific 200 results fixed-strength |
| D27 future seam/cache | G `prepare_macro`, S explicit `macro_mapping`; I current linearization is dynamic | Macro-rebuild and interleaved/fresh/JIT tests; no actual macro inference |
| D28 execution/precision | JAX kernels, scoped dtype/defaults; Python orchestration; declared older-map API fallback | P/C and JAX compatibility tests; E13; small32 supported, scientific float32 failure explicit |
| D29 result contract | O holds geometry/maps/images/strengths/status/methods/provenance and scalar history | Public-fit and export/reload/failure tests; safe JSON/NPZ |
| D30 images | S keeps linear inference and nonlinear diagnostic; I sets linear products None | S/O image/objective tests; matched algorithm-comparison tests |
| D31 scientific truth | M uses singular PseudoJaffeSubhalo, correct tidal relation, true macro and matched noise/control | Independent truth/zero/center tests; E11/E12 sampling/domain definitions |
| D32 scientific/performance | Configurable small/200 runners, numerical/figure products and measured RTX 5080 resources | O/P; E11–E13 exact sizes, recovery, sampling, prior and timing/memory limitations |
| D33 delivery sequence | Tickets 01–13 precede 14; publication and implementation states kept separate | Ticket table and final verification below; parent remains open for review/fix |

## All testing decisions

| Testing decision | Maintained seam and verification/evidence |
| --- | --- |
| T01 approved seams | G/S/I/N public numerical results, operators, O artifacts, P measurements; no testing-only public API |
| T02 independent/prior-art tests | S Gaussian construction, I independent nonlinear least-squares, N independent scan/locality; C legacy regressions |
| T03 geometry/support | G tests irregular/hole/disconnected/isolated/subpixel/edge observations and invalid explicit geometry |
| T04 numerical operators | G physical polynomials/adjoints/sign, S small-perturbation response, I current finite-difference/branch gradients |
| T05 prior semantics | R/S complete windows, missing-node/no-fallback, independent amplitude and dense/operator penalties; E01/E04/E06 |
| T06 coupled solves | S/I fixed matched dense/operator fields/images/objectives/residuals at declared dtype/solver tolerances |
| T07 evidence/search | S independent normalized Gaussian; N fixed lambda0 scans, reordered candidates, invalid fits, selected recovery and bounds; E07/E10 |
| T08 iteration/failure | I accepted objective monotonicity, accumulated prior, damping-independent final evidence, failures; two slow public64 refinement/budget regressions |
| T09 scientific validation | M truth helpers, O matched subhalo/control outcomes, raw/aligned valid-domain errors; E11/E12 exact physical models and limitations |
| T10 precision/memory | R/P dtype/intermediate-facing/default audits, matched float64/32 and allocation-path/live/peak checks; E13 |
| T11 artifacts/compatibility | O safe saved/reloaded results, None fields, histories and failures; C source/PSF/AD/Schur/public imports; older external map signature |
| T12 organization/provenance | Separate core/example/benchmark processes in tinylens_gpu, specific files from their directories; actual commands/configs/hashes below |
| T13 calibration/acceptance | E07/E10 rejected partitions, E12 retained strict/lambda0=3000 failures, E13 rejected loose-PCG/small32 and all scientific float32 failures remain explicit |

## Every user story

Each story points to the decision rows above, which name the concrete public
implementation, test group and configuration-specific evidence. IDs retain the
approved parent numbering; none is removed or rewritten to fit a limitation.

| Story | User-visible delivery | Trace |
| --- | --- | --- |
| U01 | Pixelized lens-potential reconstruction | D01/D17/D18; S/I; E11/E12 |
| U02 | Existing regular square source | D04; G/R/C |
| U03 | One lens-light-subtracted band | D01; S/I/M |
| U04 | Separate algorithms | D02/D05; S/I |
| U05 | Full source responds jointly | D06/D17; S coupling tests |
| U06 | Truth-derived initialization | D05; S/I/O |
| U07 | Updated rays/gradients | D12/D18/D27; I current-cell tests |
| U08 | Fixed macro | D01/D27; G |
| U09 | Future macro preparation seam | D27; G prepare_macro/S macro_mapping |
| U10 | Signed source/potential | D06; S/I signed tests |
| U11 | Source boundaries preserved | D04/D12; G/C |
| U12 | Irregular-mask pixels retained | D07–D09; G |
| U13 | Narrow/isolated-pixel support | D08/D09; G |
| U14 | Local potential support | D08/D22; G/P |
| U15 | Support-only coefficients labeled | D09/D16/D29; O |
| U16 | Independent potential spacing | D10; renderer/scientific configs |
| U17 | Same deflection map everywhere | D11/D13; G/S/I |
| U18 | Brightness-consistent current gradient | D12; I branch/finite-difference tests |
| U19 | Full fourth differences | D14; R/S window tests |
| U20 | Pixel-unit smoothing | D14/D15; R |
| U21 | Explicit amplitude prior | D15; R/S |
| U22 | Amplitude normalization | D15/D23; independent S/operator determinants |
| U23 | Separate source/D4 strengths | D26; N |
| U24 | Amplitude fixed during search | D15/D26; N reset/override tests |
| U25 | Raw fit preserved | D16; O raw/affine export tests |
| U26 | Dense reference | D20; S/I |
| U27 | Matrix-free joint action | D21/D22; S/I/P |
| U28 | Coupling retained | D21; S/I matched references |
| U29 | Evidence method and selection effect | D24/D25; N/E07/E10 |
| U30 | Lazy regularization storage | D22; R/P/E01 |
| U31 | Actual nonlinear decrease | D18/D19; I monotonicity/history |
| U32 | Accumulated-field prior | D18; I ridge/nonlinear MAP |
| U33 | Explicit numerical failure | D19/D29; I/O |
| U34 | Invalid candidates rejected | D26; N +infinity/reason tests |
| U35 | Reproducible initialization per candidate | D05/D26; N interleaved/fresh/order tests |
| U36 | Bounded derivative-free log search | D26; N DIRECT |
| U37 | Callable inference boundary | D27; N exports |
| U38 | JAX/GPU numerical execution | D28; I/P JIT and E13 workers |
| U39 | Float64 reference/explicit float32 | D28; P/E13 supported and failed profiles |
| U40 | Imports/defaults/old behavior | D03/D28; C and broad integration |
| U41 | Common structured result | D29; O |
| U42 | Two single-step renderings | D30; S/O |
| U43 | Iterative objective/termination history | D19/D29/D30; I/O |
| U44 | Pseudo-Jaffe truth fields | D31; M |
| U45 | Matched no-subhalo control | D31/D32; M/E11/E12 |
| U46 | Localization/field/source/residual errors | D32; O/E11/E12 valid domains |
| U47 | Saved reproducibility inputs/results | D29/D32; O/N exports/E13 hashes |
| U48 | Compile/warm/size/memory evidence | D28/D32; P/E13 distinct phase semantics |
| U49 | Public result/artifact regressions | T01/T02; S/I/N/O/P |
| U50 | Independently verifiable stages | D33; tickets 01–14, historical reports and final gate |

## Ticket delivery and historical verification

The tracker remains the lifecycle authority. This table indexes concrete
completion evidence without replacing each ticket's full checklist/comments.
Named focused checks below were run in their reported revisions/environments;
final broad integration appears in the next section.

| Ticket | Delivery/evidence | Recorded focused checks and configuration |
| --- | --- | --- |
| [01](../../.scratch/potential-correction/issues/01-prepare-source-regularization-and-precision.md) | R/C, E01 | `test_regularization.py`: 62; typed-source/cache selection: 23; existing source/operator/AD family: 372. Explicit float64 bounds and lazy allocation |
| [02](../../.scratch/potential-correction/issues/02-generate-pseudo-jaffe-mocks-and-truth.md) | M, example README | `test_potential_correction_mock.py`: truth relations, singularity, shared noise, persistence; generator default b=.01 separate from scientific b=.04 |
| [03](../../.scratch/potential-correction/issues/03-render-supplied-potential-corrections.md) | G/R, E03 | `test_potential_correction_rendering.py`: 17; separate supplied-truth 200×200 discretization record |
| [04](../../.scratch/potential-correction/issues/04-dense-single-step-reconstruction.md) | S, E04 | Dense + rendering: 31; independent 3×3/source2 coupled Gaussian and nsub3 masked PSF cases |
| [05](../../.scratch/potential-correction/issues/05-optimize-dense-single-step-strengths.md) | N, E05 | Callable/DIRECT independent bounded Gaussian scan and example; explicit log bounds, fixed amplitude, candidate reasons |
| [06](../../.scratch/potential-correction/issues/06-operator-single-step-reconstruction.md) | S/P, E06 | Operator/inference: 31; selected existing Schur/PCG: 23; isolated allocation and runnable example; legacy float32 variability retained |
| [07](../../.scratch/potential-correction/issues/07-calibrate-single-step-backends.md) | N/P, E07 | Curvature/operator/inference: 39; dense/render/AD: 62; Schur: 23; examples: 16. Six physical 64×64 sensitivity runs and bounded selections |
| [08](../../.scratch/potential-correction/issues/08-dense-iterative-reconstruction.md) | I, E08 | Iterative/current AD check: 33; inference: 13; examples: 4. Nonlinear independent MAP, ridge-limit correction and honest kink/failure policy |
| [09](../../.scratch/potential-correction/issues/09-operator-iterative-reconstruction.md) | I/P, E09 | Correction family: 108; Schur/PCG: 23; examples: 6; memory: 2. Changed 11 cells at source5/nsub2; dense fields agree at recorded tolerances |
| [10](../../.scratch/potential-correction/issues/10-optimize-iterative-strengths.md) | N, E10 | Inference family: 31; tuning examples: 4. Independent 25-point scan/refined boundary reference; cap20 calibrated partition; 115 valid candidates per final search |
| [11](../../.scratch/potential-correction/issues/11-single-step-scientific-example-and-exports.md) | M/O, E11 | Actual 200×200 both backends/control, saved numeric/figure results and independent amplitude probes; scientific examples regression |
| [12](../../.scratch/potential-correction/issues/12-iterative-scientific-example-and-exports.md) | I/O, E12 | Refinement+nested core: 20; science examples: 9; memory: 2. Four converged 200×200 float64 fits, retained strict/3000 failures, explicit example controls |
| [13](../../.scratch/potential-correction/issues/13-validate-precision-and-gpu-performance.md) | P, E13 | Correction core: 132; examples: 30; benchmarks: 5; final dtype/stopping check: 5. 45 isolated workers; actual 32/64/200 and source33→65 axes |
| [14](../../.scratch/potential-correction/issues/14-integrate-document-and-verify.md) | User guide, this full coverage record, discovery links and older-JAX dense compatibility | Final broad/targeted results below. Parent spec stays open for independent review/fix after integration |

## Exact scientific and precision interpretation

E11/E12 use the explicit 200×200, dpix=.012, nsub=2/2, source49 bbox±.4,
h=.075, PSF27/sigma .05, noise .01/seed17 model: true SIS .8 + shear(.05,.02),
Pseudo-Jaffe b=.04 at(.75,.1), matched shared-noise control. Counts are
22016 data/2401 source/724 C/956 U/88064 queries; fixed strengths are
lambda_s=.0006, lambda4=1.6e7 or1e11, lambda0=10000. Iterative example controls
are gradient atol5e-4/3e-5 and true residual `5e-5+1e-8*norm(rhs)`; defaults
were not changed. Scientific fixed-strength success is distinct from outer
optimizer completion. Local kappa is localized with suppressed peak; raw
potential/deflection remain broad-mode/prior sensitive. Lambda0 10000→30000
has measured local-kappa stability with raw-field/evidence changes; 3000 failed
and has null evidence. No globally weak prior, detection significance or mass
interval is established.

E13 validates float32 only for the named coupled37/small32 profile and retains
all eight scientific float32 failures/null evidence. Its field guard remains
atol=rtol=2e-4 and evidence budget .002 against strict same-method float64.
Host DIRECT coordinates are float64; requested numerical dtype and ambient
precision defaults are separate. Timings synchronize all result dataclass
arrays; first-fit includes compilation/lazy preparation and is not pure compiler
time. Allocator peaks, sampled process residency and live arrays have distinct
semantics. At the scientific layout operator allocator memory exceeds dense;
there is no blanket speed/memory or float32-performance claim. The synthetic
extra-support probe complements actual image/source scaling, not substitutes
for it. See E13's per-worker hashes and phase definitions before comparing costs.

## Final integration verification

The **full core run passed 957 tests in 567.50s**, including both slow
scientific-control refinement/exhausted-budget regressions, existing PNPG
stress and large-scale integration, legacy pixelized/PSF/Schur/AD and public
interfaces. No marker selection or skip waived those cases. This process began
before the bounded older-map compatibility edit and before its four new cases
were collected. Its complete pre-edit record is preserved separately.

After the compatibility edit, a separate GPU command from `tests/` passed
**67 tests in 60.95s**:

```bash
pytest -q test_potential_correction_jax_compatibility.py \
  test_potential_correction_dense.py test_potential_correction_rendering.py \
  test_potential_correction_iterative.py test_potential_correction_precision.py
```

These post-fix public dense/render/iteration/dtype checks supplement the full
pre-edit run; they are not reported as a single 961-case full run. The measured
modern native map path is retained. The 45-worker scientific/performance
campaign was not rerun for this API guard, and its results remain attributed
to E13's recorded numerical revision/hashes.

A separate full example process, `pytest -q examples/tests` from the
repository root, passed **286 tests in 248.88s**. This includes the completed
single-step/iterative scientific/nested export/reload/history/failure cases,
source compatibility, legacy pipelines, plotting and shared example support.
The separate full benchmark command `pytest -q -s benchmarks` passed
**14 tests in 44.20s**, including legacy timing/scaling checks, bounded
single-step/iterative allocation, cached principal factors and isolated workflow/
actual source-size checks. No GPU timing workers ran concurrently. The iterative
synthetic probe retained its honest `iteration_limit` status and measured peak
14,567,424 allocator bytes; it is not a converged scientific memory result. Core, example and benchmark suites run in separate serial processes
on the available RTX 5080 in `tinylens_gpu`, with
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, `MPLBACKEND=Agg` and
`PYTHONPATH=/tmp/tinylens-pc-14`. Full logs are retained in the root checkout's
ignored `.scratch/potential-correction/output/final-integration/`; compact
versioned commands/outcomes/hashes are linked below.

The JAX minimum declared by setup/requirements is preserved. JAX 0.4.20's
[official map source](https://raw.githubusercontent.com/jax-ml/jax/jax-v0.4.20/jax/_src/lax/control_flow/loops.py)
has `map(f, xs)` without the newer batch-size keyword. The new correction dense
path therefore needs a feature-detected bounded fallback. Public single-step
and iterative dense fits under an external legacy-signature adapter test the
real unsupported-keyword failure, 37-column remainder/order and both dtypes.
The first CPU public single-step case reproduced the exact
`TypeError: ... unexpected keyword argument 'batch_size'` in **3.31s** before
the fix. After feature detection and mapped 16-column panels plus a separate
remainder, **four CPU cases passed in 6.44s**; the same four also passed in the
67-case GPU command. Both algorithms match native-map fields/images/objectives/
determinants/evidence while preserving requested float64/float32 and the
five-column remainder. Modern JAX keeps its measured native 16-column path. This targeted compatibility
check is not an actual CUDA run on the older JAX release and makes no blanket
minimum-version environment claim. The unrelated historical optional chunked
likelihood path is outside this feature change.

The [versioned compact verification record](../../.scratch/potential-correction/research/final-integration-verification.json)
retains exact commands, working directories, source/log hashes, environment,
pre/post-edit scope and outcomes. Full logs are retained at:

- `.scratch/potential-correction/output/final-integration/pc14-core.log`
- `.scratch/potential-correction/output/final-integration/pc14-post-compatibility-core.log`
- `.scratch/potential-correction/output/final-integration/pc14-examples.log`
- `.scratch/potential-correction/output/final-integration/pc14-benchmarks.log`

Observed environment: Python 3.11.14, JAX/jaxlib 0.7.1, NumPy 2.4.6, SciPy
1.17.1, NVIDIA GeForce RTX 5080, driver 610.57.04, 16303 MiB. Recorded installed
versions are not inferred from dependency declarations. No PyAuto environment
was used. No assertion was relaxed, numerical failure reclassified, test
skipped or scientific model/default changed for final integration.

Black/Ruff default checks on the changed Python files and `git diff --check`
pass. All local links in the new guide/coverage record resolve; explicit D01–33,
T01–13 and U01–50 coverage IDs are complete. All three guide Python blocks
compile, and the complete minimal public single-step/iterative example was
executed successfully on CPU under the documented scoped x64 context. The
load/export and remaining scientific commands use the maintained public
examples and separately verified artifacts; the 200×200 campaigns were not
rerun as part of this documentation ticket.

All ticket-14 acceptance checks are met. The original approved spec and prior
failed scientific records are unchanged. The parent spec is deliberately still
open for the coordinator's mandatory independent review/fix gate.
