# Iterative scientific source–potential reconstruction

This record describes the original fixed-strength runs. The later
[peak-bias investigation](potential-correction-peak-recovery.md) evaluates a
common evidence-selection policy, sharper peaks and their noise trade-off;
the original numerical results below remain the baseline.

The recorded float64 runs validate nonlinear reconstruction of the ticket 11
scientific mock in both backends. They support localized recovery with
substantial peak suppression. Raw potential/deflection low modes remain
prior-sensitive. The chosen amplitude precision is a model-specific value;
it is not established as globally weak or broadly robust.

## Model, initialization and matched comparison

The analytic `PseudoJaffeSubhalo` observation uses `b_sub=0.04`,
`theta_E=0.8`, position `(0.75,0.1)` arcsec, an SIS plus shear `(0.05,0.02)`,
and a matched zero-subhalo control with the identical noise realization.
Images are 200×200 at 0.012 arcsec/pixel, spanning 2.4 arcsec, with noise
sigma 0.01, seed 17, and a 27×27 Gaussian PSF of physical sigma 0.05 arcsec.
Mock and inversion integration are separately recorded as `nsub=2/2`.
This is the explicit ticket 11 scenario, not the generator default
`b_sub=0.01` or the 64-pixel calibration with its different sampling/PSF.

The fixed source is 49×49 on `[-0.4,0.4]^2`, with analytic source widths
0.16/0.12 arcsec and clump scale 0.7. Potential spacing is 0.075 arcsec.
Actual counts are **22,016 data, 2,401 source, 724 core C, 956 support U,
and 88,064 query subpixels**. Source second-order regularization,
`lambda_amplitude=10000`, and fixed `(lambda_source,lambda_potential)` are
`(0.0006,1.6e7)` for subhalo and `(0.0006,1e11)` for control.

Every fit/candidate starts from truth intensity at source **nodes** and zero
potential. Subsequent derivatives come from the current bilinear source at
corrected rays. This differs from the single-step frozen analytic truth
gradient. Neither geometry expands during fitting or search.

The exact ticket 11 operator partition is reused after a geometry-hash check:
block sizes `788,840,24,820,812,24,24,24,1`, cap 1024. The partition is a fixed
numerical choice; the iterative Jacobian still refreshes. Dense single-step
versus dense iterative is the primary algorithm comparison. Operator-minus-
dense within each algorithm isolates determinant approximation. Both operator
algorithm comparisons confirm identical partition indices.

## Residual correction and measured stopping controls

The public 64-pixel control regression first reproduced a true residual
`4.75e-5` despite recursive PCG convergence, at the original physical
certificate. The adapter now permits **one separately accumulated residual
correction within the remaining total PCG budget**, using the same operator,
factors and recursive tolerances. It attempts correction only after a finite,
recursively converged solve misses the true certificate, and selects only a
finite improvement. True acceptance limits, priors, damping policy, generic
CG and block-Schur code remain unchanged.

The correction calls the standard wrapped Python body of `pcg_solve` inside
the adapter's existing JIT: its public JIT wrapper makes `max_iter` static,
whereas the remaining budget is dynamic. A public changing-count regression
observes one adapter-solve compilation. A 4,000-iteration budget regression
still fails honestly at exactly 4,000 aggregate iterations, with null evidence.
Phase counts sum to the aggregate, and final certification uses the original
full-system RHS. The legacy `pcg_recursive_residual_norm` and aggregate
`PCGInfo.residual_norm` describe the **initial phase on the original RHS**;
per-phase diagnostics give the correction's own recursive norm. No aggregate
recursive residual is invented. Breakdown, budget exhaustion, or an
unsatisfied true certificate remains failure.

The completed 64-pixel check at `gradient_atol=3e-5`, `solve_atol=2e-5`
converges in all four fits. Control operator selects six corrections, with
maximum 8,987 aggregate iterations within 20,000; dense/operator fields and
images agree around `1e-12`. Subhalo needs no correction. Local/block log
evidence offsets are 86.52597255 and 54.84327570. This does not itself
validate the 200-pixel setting.

Strict 200-pixel subhalo fits stalled in both backends at
`Phi=11734.82651376806`. Their minimum pre-step gradient `4.137e-4` falsified
the proposed `1e-4` stopping hypothesis. A named failed dense replay retained
raw diagnostic arrays and independently measured final gradient `6.443e-4`;
its evidence stays null. Dense control failed its true gate
`2.3127e-5 > 2.2025e-5`, while strict operator control genuinely converged.
The initial strict records and the named replay remain distinct.

Separate successful subhalo controls at `gradient_atol=5e-4/1e-3` differ by
`Phi=1.993e-6`, maximum source `3.468e-7`, potential `3.187e-8`, core kappa
`7.387e-6`, and nonlinear image RMS `3.497e-6` noise. Dense local log evidence
changes `-9.331e-6`; operator block log evidence changes `-4.092e-6`.
The `5e-4` state differs from the strict stalled replay by `Phi=1.768e-8`
and image RMS `4.172e-8` noise, without assigning evidence to the failed state.
Control at `solve_atol=5e-5` matches the stricter converged operator state to
`6.1e-13` source and `2.25e-13` image RMS/noise; block log evidence changes
only `1.5e-11`.

The adopted **example-only** controls are therefore subhalo/control
`gradient_atol=5e-4/3e-5`, `gradient_rtol=1e-8`, `solve_atol=5e-5`, and
`solve_rtol=1e-8`. The actual true residual threshold is
`solve_atol + solve_rtol*norm(rhs)`. PCG `rtol=1e-12`, budget 20,000, and
all public defaults remain unchanged. The gradient threshold includes its
recorded initial-baseline relative term. These controls are calibrated for
this configuration, not automatically transferable.

## Actual 200-pixel reconstruction

All four final fits genuinely terminate at `converged_gradient`, at clean
numerical revision `764e4b1`. The independent terminal-gradient audit uses
public current-J/Q actions in **16-column panels**, without a full basis vmap.
It agrees with the public refreshed final gradient.

| Case/backend | accepted/rejected trials | final gradient | final true residual | objective Phi | log evidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Subhalo dense | 46/17 | 4.2900e-4 | 4.49e-14 | 11734.82651379 | 69037.25724906 |
| Subhalo operator | 46/17 | 4.2900e-4 | 3.87e-11 | 11734.82651379 | 69145.76524575 |
| Control dense | 18/0 | 2.5648e-5 | 5.95e-8 | 11404.44365566 | 69487.83779926 |
| Control operator | 18/0 | 2.5648e-5 | 4.83e-7 | 11404.44365566 | 69560.59240706 |

Operator control selects four residual corrections; maximum aggregate
iterations are 7,996. Subhalo selects none, with maximum 729. Every phase
sum, total-budget check and true residual certificate passes. Across all
trials, the maximum operator true residual is `3.27e-5` for control and
`5.93e-6` for subhalo; final values above are separate quantities.
Dense/operator maximum source differences are `4.2e-16/5.6e-13`, and image
RMS differences are `1.23e-14/1.59e-13` noise for subhalo/control.

| Dense recovery diagnostic | Subhalo | Matched control |
| --- | ---: | ---: |
| Source relative L2 error | 0.07029 | 0.05171 |
| Raw potential RMSE, center-zero truth | 0.015397 | 0.0005191 |
| Deflection-vector RMSE, discrete truth | 0.007281 | 0.0008046 |
| Kappa RMSE, all 724 core nodes | 0.026779 | 0.0005329 |
| Kappa RMSE, local 70-node disk | 0.058278 | 0.0005714 |
| Nonlinear residual RMS/noise | 1.001837 | 0.995828 |

The recovered subhalo peak is **0.303586 at `(0.75,0.075)`**, 0.025 arcsec
from the injection, versus discrete truth maximum **0.631435**: about 52%
peak suppression remains. Local/all-core kappa errors improve over zero-field
baselines `0.099169/0.030863`, but this is not calibrated mass recovery.
The illuminated 526-node subset has RMSE 0.025901. The local disk radius is
0.35777 arcsec; illumination is the observation annulus with macro-source
intensity at least 5% of its core maximum. The matched control uses the same
**nominal** position/disk without an injection.

Analytic point-sampled convergence and centered differences of analytic nodal
potential remain separate truth conventions, retaining the profile's
singular-center convention. Derivatives are undefined outside C; potential
lives on U. Raw fields are neither clipped nor aligned. Subhalo potential
RMSE falls to 0.001096 only in the separately named affine diagnostic; that
field never replaces the fit, image or evidence. The unconstrained source
retains signs: 20.7% negative nodes, negative/positive absolute intensity sum
0.0240. All fitted query rays remain within the fixed box.

At identical dense settings, the single-step recovered fields evaluated
**nonlinearly** have Phi `12366.52710/11404.88170`, versus iterative
`11734.82651/11404.44366` for subhalo/control. The single-step stored inference
objectives `11780.80342/11405.49454` belong to its linear image and remain
unchanged. Algorithm image differences are `0.226783/0.00253456` noise RMS.

Iterative operator-minus-dense log evidence offsets are
**108.50799669/72.75460780**, entirely from principal-block versus full
determinants. Dense single-step-to-iterative log evidence changes
`47.27278849/1.00537277` compare different evidence models: conditional linear
Gaussian versus local Gauss–Newton. Final iterative curvature is refreshed
and **undamped**; even full dense determinants do not make it an exact
nonlinear integral. None of these evidence differences is a detection
statistic.

## Nonlinear amplitude-prior sensitivity and limits

At fixed smoothing, physical configuration and adopted numerical controls,
`lambda_amplitude=30000` converges for both dense cases:

| Diagnostic, lambda0 10000 → 30000 | Subhalo | Control |
| --- | ---: | ---: |
| Source relative L2 | 0.07029 → 0.06655 | 0.05171 → 0.05170 |
| All-core kappa RMSE | 0.02678 → 0.02747 | 0.0005329 → 0.0005305 |
| Local kappa RMSE | 0.05828 → 0.05846 | 0.0005714 → 0.0005693 |
| Raw potential RMSE | 0.015397 → 0.015369 | 0.0005191 → 0.0001909 |
| Deflection RMSE | 0.007281 → 0.006925 | 0.0008046 → 0.0003787 |
| Nonlinear image RMS difference/noise | 0.007755 | 0.0006170 |
| Local log evidence change | -12.13460650 | +6.18869029 |

The subhalo peak stays at the same node and changes from 0.303586 to
0.302805. Stable localized kappa does **not** establish weak global modes:
control potential/deflection change strongly, and normalized local evidence
changes measurably. The baseline 10000 is an explicit validated model choice,
not an inferred universally weak prior.

The declared lower `lambda0=3000` probes fail at the same controls: subhalo
stalls at Phi 11720.0379871 with minimum pre-step gradient `5.554e-4`, and
dense control misses its true residual gate (`5.539e-5` versus approximately
`5.353e-5`). Their raw terminal arrays and histories are explicitly diagnostic,
with null evidence and no successful scientific export. **Lower-side
convergence remains unverified.** No closer successful interval replaces
this failed range, and no tolerance is raised to rescue it. The supported
sensitivity statement is the measured **10000-to-30000** comparison.

## Saved products, reproduction and verification

Actual arrays, histories, figures and complete inputs are retained in the
root checkout's ignored
`.scratch/potential-correction/output/iterative-scientific/`:

- `validated-reference/`: all four final public results, comparison records,
  truth arrays, plots and separate presentation provenance.
- `reference/`, `strict-subhalo-replay/`: strict outcomes and named failed
  terminal replay; failed status/evidence are preserved.
- `subhalo-gradient5e-4/`, `subhalo-gradient1e-3/`,
  `operator-gradient1e-3/`, `control-residual5e-5/`: measured stopping probes.
- `amplitude3000/`, `amplitude30000/`: both declared prior neighbors.
- `calibration64/` and `historical-diagnosis/`: complete current 64 fits and
  frozen earlier failures/independent residual diagnosis. Historical scripts
  are archived evidence, not a current successful workflow.

[Compact records and hashes](../../.scratch/potential-correction/research/iterative-scientific/)
include the final configuration/summary, scalar recovery table, stopping
comparisons, amplitude sensitivity, artifact manifest and historical manifest.
[Final source](../../.scratch/potential-correction/output/iterative-scientific/validated-reference/subhalo/dense/source.png),
[fields](../../.scratch/potential-correction/output/iterative-scientific/validated-reference/subhalo/dense/fields.png),
[images](../../.scratch/potential-correction/output/iterative-scientific/validated-reference/subhalo/dense/images.png)
and [control history](../../.scratch/potential-correction/output/iterative-scientific/validated-reference/control/operator/history.png)
were visually inspected. Shared intensity limits, raw signed source, C/U
coverage, and initial-recursive versus true residual labels are explicit.
Control crosses are labeled “nominal subhalo position.” Presentation-only
regeneration loads the public result and preserves numerical NPZ/JSON hashes.

Activate `tinylens_gpu`, set `XLA_PYTHON_CLIENT_PREALLOCATE=false`, and run
from the repository root. Use a **fresh output directory** to preserve the
retained evidence. The ticket 11 saved reference must exist for comparisons:

```bash
PC_OUTPUT=/path/to/fresh/iterative-scientific
PC_SINGLE=.scratch/potential-correction/output/single-step-scientific/reference
python .scratch/potential-correction/experiments/validate_iterative_scientific.py \
  --output "$PC_OUTPUT/validated-reference" \
  --summary "$PC_OUTPUT/validated-reference/summary.json" \
  --single-step-reference "$PC_SINGLE" --audit-gradients
python .scratch/potential-correction/experiments/validate_iterative_scientific.py \
  --case subhalo --backend operator --gradient-atol 0.001 --no-plots \
  --audit-gradients --output "$PC_OUTPUT/operator-gradient1e-3" \
  --summary "$PC_OUTPUT/operator-gradient1e-3/summary.json"
```

For each declared prior neighbor, run the same driver with `--backend dense
--lambda-amplitude 3000` or `30000`, `--no-plots`, and separate output/summary
paths. The 3000 command intentionally exits nonzero after retaining diagnostics.
The strict configuration and every probe's complete actual configuration are
retained. `summarize_iterative_scientific.py --output-root ...
--summary-directory ...` reads the complete retained campaign using host
float64 only; it never fits or aligns fields. `replot_iterative_scientific.py
--run .../validated-reference` regenerates presentation from saved public
results. The maintained runner also supports `--mode tune`, independently
resetting every full reconstruction; bounded nested assembly/failure tests
exercise both backends. These 200-scale results are **fixed-strength**, not
claims of converged outer searches.

Focused tests run in separate core/example/benchmark processes, with specific
files run from their own directories:

- `tests/`: `pytest -q test_potential_correction_residual_refinement.py
  test_potential_correction_iterative_inference.py`: **20 passed**.
- `examples/tests/`: `pytest -q test_potential_correction_iterative_scientific.py
  test_potential_correction_scientific.py`: **9 passed**. This covers both
  backends, complete nested assembly, saved-reference comparison, public
  reload/None linear products, per-case controls and failure cleanup/history.
- `benchmarks/`: `pytest -q -s test_potential_correction_memory.py`:
  **2 passed**, retaining bounded toy-workflow allocation and honest
  iteration-limit status; this is not production scientific memory/timing.

The earlier broader operator check passed 18 cases and reproduced the known
float32 coupled stopping failure at
`test_operator_float32_matches_float64_coupled_reference_without_changing_defaults`. One unchanged-integration isolated baseline
also fails; the refinement path is inactive there. This separate ticket 13
handoff is preserved without changing tests/defaults. No float32 scientific
precision, timing, peak-memory or speedup claim is made here. Those measurements
remain ticket 13, and whole-spec integration remains ticket 14.
