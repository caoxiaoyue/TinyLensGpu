# Scientific subhalo peak bias and matched-control recovery

## Result and scope

The original nonlinear scientific demo recovers **48.08%** of the injected
discrete convergence peak. A finite, data-driven comparison of converged
nonlinear fits raises this to **85.30%** on the original noise realization,
while retaining the original no-subhalo control to numerical precision.
Both observations use the same candidate set, initialization and numerical
controls. The selection rule reads convergence status and full dense local
Gauss–Newton evidence; it never reads injected location, recovered peak,
truth error or control-map amplitude.

This is a **peak versus variance trade-off**, not a uniformly better mass map.
The subhalo local κ RMSE improves, but whole-core κ RMSE increases. The
default scientific configuration remains the recorded historical baseline.
The study supplies a reproducible alternative and its limitations rather than
silently replacing that configuration.

All measurements are for the existing 200 × 200 analytic mock: 0.012 arcsec
pixels; matched mock/inversion nsub = 2; Gaussian PSF σ = 0.05 arcsec, 27 × 27;
noise σ = 0.01; true fixed SIS θE = 0.8 plus shear (0.05, 0.02); singular
Pseudo-Jaffe subhalo b = 0.04 at (0.75, 0.1) arcsec. Source geometry stays
49 × 49 in [-0.4, 0.4]²; potential spacing stays 0.075 arcsec. Counts are
22,016 data, 2,401 source nodes, 724 derivative-valid core nodes and 956
potential unknowns. The macro, mask, PSF, source box and integration do not
change. Fits start from truth-derived source **node values**, then solve the
full signed source and potential. No parametric subhalo is fitted.

## What causes the low peak?

### D4 smoothing broadens the cusp

The potential precision is λ4 D4ᵀD4 + λ0 I, with complete pixel-unit fourth
differences and independent amplitude precision on all unknowns. At the
original λ4 = 1.6 × 10⁷, the injected nodal potential has unhalved D4 penalty
**2575.76**, versus **392.82** for the recovered potential. The Gaussian prior
strongly prefers the broader solution. Holding everything else fixed and
reducing λ4 raises the peak in both algorithms.

The original nonlinear reconstruction already recovers **98.15%** of the
integrated κ within radius 0.15 arcsec, and **102.17%** within the documented
0.35777-arcsec local disk. Its peak is low largely because the profile is
broader. These aperture sums use the same finite-difference grid and cell
area; they are not physical mass estimates or calibrated mass uncertainties.

The target peak **0.631435** is the centered finite-difference convergence of
analytic nodal potential on the actual inversion grid. The distinct analytic
point-sampled maximum is about 0.68927. Thus the reported 52% suppression
already compares fields on the same discrete grid.

### The source and potential priors compete

Reducing potential smoothing universally creates structure in a null image.
With λs = 0.0006 and λ4 = 400,000, the exploratory nonlinear control has
maximum |κ| **0.2961**, versus **0.001506** for the original strong-prior
control. The exploratory fit is accepted at the original gradient tolerance
but stalls at the stricter study tolerance; it is a rejected diagnostic.

This is not entirely noise fitting. An exact **single-step frozen-linear**
decomposition at λs = 0.0006, λ4 = 400,000 shows control κ core RMS **0.05681**
from regularized reconstruction of perfectly matched, noise-free linear
source data, compared with **0.02141** from the noise contribution. The source
prior itself pushes the potential to compensate. Weakening λs to 0.00006
reduces the single-step control RMS at that λ4 from **0.06129 to 0.02470**.
Strengthening source smoothing is therefore not a safe general remedy.

### Representation error can make a sharper peak look better

For the single-step model, let H = MᵀC⁻¹M + Q. Decompose the actual data into
matched linear truth, nonlinear-minus-linear discrete truth, analytic-minus-
discrete representation error, and noise. Applying H⁻¹MᵀC⁻¹ to each component
reproduces the complete fitted coefficient vector within **2.3 × 10⁻¹²**.

At the true peak node, the subhalo contributions are:

| Frozen-linear contribution to κ | λ4 = 1.6 × 10⁷ | λ4 = 400,000 |
| --- | ---: | ---: |
| Regularized matched linear truth | 0.219119 | 0.352525 |
| Nonlinear minus linear discrete truth | 0.003432 | -0.005059 |
| Analytic minus discrete representation | 0.071826 | 0.225669 |
| Noise | -0.004236 | -0.016668 |
| Sum: actual single-step recovered peak | 0.290141 | 0.556466 |

The analytic-to-discrete image mismatch is only **0.24736 noise RMS**, but
its projection into poorly constrained potential modes matters. Some of the
apparent peak improvement compensates for forward-model mismatch. This exact
linear decomposition is not an exact decomposition of the nonlinear fit.
The higher peak alone does not establish more accurate potential or mass.

### Amplitude precision and stopping do not explain the original 52%

The retained nonlinear amplitude probe changes λ0 from 10,000 to 30,000 and
changes the peak only from **0.303586 to 0.302805**. The retained 5 × 10⁻⁴
versus 10⁻³ gradient-tolerance comparison changes κ by at most **7.39 × 10⁻⁶**.
These saved-product checks reproduce the earlier validation findings; they
are not new amplitude or stopping sweeps.

Stricter checks nevertheless matter when selecting new settings. At high
potential precision, a coarse common gradient tolerance can stop the control
before its weak modes settle and make its map look artificially quiet. The
final study uses gradient atol **3 × 10⁻⁵** for **every** candidate in both
observations, gradient rtol 10⁻⁸, solve atol 5 × 10⁻⁵ and solve rtol 10⁻⁸,
with at most 120 iterations. Stalled or failed candidates retain null evidence
and cannot win selection. We do not relax tolerances to rescue sharper maps.

## Shared selection policy and measured trade-offs

The nonlinear study keeps λs = 0.0006 and λ0 = 10,000 fixed and tries the same
λ4 values **[400,000; 800,000; 1,600,000; 10¹¹]** on each observation. It
selects the largest finite log evidence among converged fits independently
for each dataset. On development seed 17 it selects λ4 = **800,000** for the
subhalo and **10¹¹** for the control. Unlike the original case-specific fixed
settings, the rule does not use knowledge of which dataset has a subhalo.

| Seed-17 nonlinear diagnostic | Original demo | Selected study fit |
| --- | ---: | ---: |
| Recovered κ peak / truth 0.631435 | 0.303586 | 0.538583 |
| Peak recovery | 48.08% | 85.30% |
| Peak suppression | 51.92% | 14.70% |
| Subhalo local κ RMSE | 0.058278 | 0.053637 |
| Subhalo whole-core κ RMSE | 0.026779 | 0.050799 |
| Subhalo background κ RMS | 0.020732 | 0.050317 |
| Aperture-integrated κ ratio, radius 0.15 arcsec | 0.9815 | 1.2165 |
| Aperture-integrated κ ratio, radius 0.35777 arcsec | 1.0217 | 0.9686 |
| Nonlinear image residual RMS / noise | 1.001837 | 0.993995 |
| Control κ RMS | 0.000532935 | 0.000532935 |
| Control maximum absolute κ | 0.00150646 | 0.00150646 |

The original baseline was calibrated with subhalo gradient atol 5 × 10⁻⁴;
the selected subhalo passes the stricter 3 × 10⁻⁵ check. The control is fit
with the same tighter tolerance in both records and agrees numerically.
The final selected subhalo peak remains at (0.75, 0.075) arcsec.

The tempting **95.38%** peak at λ4 = 400,000 is excluded: its stricter
nonlinear fit stalls, and applying that setting to the null introduces strong
structure. Three weaker-source subhalo candidates also stall at the strict
tolerance. Their diagnostic peaks are retained but are not successful fits.

A separate complete-conditional single-step comparison over the same 23
candidate pairs for both observations selects (λs, λ4) = (0.00006, 400,000)
for the subhalo and (0.00006, 10¹¹) for the control. It recovers **82.09%**
of the subhalo peak, with control RMS **0.0007023** and maximum |κ|
**0.0019585**. These are single-step results with frozen analytic gradients;
their evidence values are not interchangeable with nonlinear local evidence.

## Independent noise checks and boundaries

The nonlinear candidate grid, shared controls, selection rule and evaluation
limits were recorded in the
[validation policy](../../.scratch/potential-correction/research/peak-recovery/validation-policy.json)
before fitting noise seeds **18 and 19**. The policy SHA256 is
`a714f0772eba9d63e3f6300fbe0bb5b7d90330c8482ea13af826e57290278ecf`.
Both observations share the same noise draw within each seed. All candidate
statuses and selected results are retained in the
[compact study record](../../.scratch/potential-correction/research/peak-recovery/summary.json).

| Noise seed | Role | Selected subhalo λ4 | Peak recovery | Control κ RMS | Control maximum absolute κ |
| --- | --- | ---: | ---: | ---: | ---: |
| 17 | Development | 800,000 | 85.30% | 0.0005329 | 0.0015065 |
| 18 | Held out | 800,000 | 86.65% | 0.0007742 | 0.0030125 |
| 19 | Held out | 400,000 | 97.05% | 0.0009842 | 0.0025769 |

All three select control λ4 = 10¹¹ and pass the declared gate. Seed 17
subhalo has one converged candidate of four; seed 18 has two; seed 19 has
three. Respective control counts are two, one and two. Every attempted point
is retained, including stalled points. This constrains the interpretation of
the selection rather than establishing a continuous optimum.

The diagnostic gate is peak recovery at the **true peak node** ≥ 60%, control
core RMS ≤ 0.0011, and control maximum |κ| ≤ 0.0031, approximately twice the
original control amplitudes. This is an explicit comparison envelope, not a
statistical false-positive probability. The development draw and two new
draws cannot establish broad noise robustness, detection significance,
population performance or mass precision.

Other boundaries remain: the grid is finite and development-chosen; source
strength stays fixed in the nonlinear policy; many candidates stall at
bilinear interpolation kinks; failed fits are excluded rather than assigned
evidence; the selected control lies on the largest tested λ4; and full dense
determinants still give a local Gaussian approximation to nonlinear evidence.
No converged continuous hyperparameter optimum is claimed. The new study
does not independently calibrate operator/block evidence for this selection.

## Reproduce and inspect

Activate `tinylens_gpu` from the repository root. Use fresh output and
selection directories to preserve prior evidence:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PC_RECOVERY_OUTPUT=.scratch/potential-correction/output/peak-recovery/reproduce
python .scratch/potential-correction/experiments/probe_peak_recovery.py \
  --algorithm iterative --sources 0.0006 \
  --potentials 400000,800000,1600000,100000000000 \
  --gradient-atol 0.00003 --max-iter 120 --seed 17 \
  --output "$PC_RECOVERY_OUTPUT"
python .scratch/potential-correction/experiments/select_recovery_candidates.py \
  "$PC_RECOVERY_OUTPUT" --output "$PC_RECOVERY_OUTPUT-selected"
python .scratch/potential-correction/experiments/diagnose_peak_recovery.py \
  --subhalo "$PC_RECOVERY_OUTPUT-selected/subhalo/dense" \
  --control "$PC_RECOVERY_OUTPUT-selected/control/dense" --assert-target
```

Repeat with seeds 18 and 19 and distinct output paths for the fixed-policy
noise checks. `decompose_peak_bias.py` reproduces the exact linear
superposition diagnostic. `summarize_peak_recovery.py` rebuilds the retained
compact record from the study's saved public exports. The
[study entry point](../../.scratch/potential-correction/research/peak-recovery/README.md)
links the exact-symptom replay and declared hypotheses.

Generated full products are in ignored `output/peak-recovery/`, with selected
public JSON/NPZ exports under `selected-iterative-final/`,
`selected-single-step-final/` and `heldout-selected-{18,19}/`.
The comparison figure is `figures/peak-recovery.{png,pdf,svg}`. It preserves
signed values, uses one shared κ scale and labels the rejected exploratory
column. Alignment, rendered-text and collision checks pass; the PNG was
visually inspected. Numerical source products were not modified for plotting.

The mathematical checks follow the coupled source–potential formulation and
regularization discussion in [Powell et al., Section 1.4](https://arxiv.org/abs/2510.07382)
and [Cao et al., free-form substructure modelling](https://doi.org/10.1093/mnras/staf2179).
The local reference implementation was inspected for matrix flow. These
sources motivate the diagnostics; all numbers above come from this stated
TinyLensGpu configuration, with no new covariance or non-Gaussian prior added.

## Verification

The retained study comprises **96 public-fitter candidates** plus four
single-step superposition checks. Both development algorithms and both
held-out nonlinear pairs pass the declared recovery/control gate. All eight
selected public results load successfully, their raw κ agrees with the
potential-grid derivatives, copied products match the selected source hashes,
and both observations attempted identical candidate pools. The predeclared
policy and original baseline numerical products remain unchanged.

In a separate process from the study fits, the following example regressions
passed **9 tests** in 46.01 seconds, from `examples/tests/`:

```bash
pytest -q test_potential_correction_iterative_scientific.py \
  test_potential_correction_scientific.py
```

The compact [verification record](../../.scratch/potential-correction/research/peak-recovery/verification.json)
retains commands, source and log hashes and numerical-product checks. No
library algorithm or physical-prior implementation was changed by this study.
