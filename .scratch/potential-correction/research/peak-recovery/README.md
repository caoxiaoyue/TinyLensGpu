# Scientific subhalo peak recovery investigation

Question: why does the 200 × 200 nonlinear scientific demo recover only
48.08% of the injected discrete κ peak, and can recovery improve without
introducing substantial structure in the no-subhalo control?

The model is the existing scientific configuration: fixed true SIS plus
shear, singular Pseudo-Jaffe injection b = 0.04 arcsec at (0.75, 0.1),
PSF σ = 0.05 arcsec, image pixels 0.012 arcsec, source grid 49 × 49 in
[-0.4, 0.4]², potential spacing 0.075 arcsec, float64. The initial development
pair uses the same noise draw, seed 17 and σ = 0.01. This is a reconstruction
study; it does not establish detection significance or calibrated mass errors.

## Reproduction and feedback

Activate `tinylens_gpu` from the repository root and set
`XLA_PYTHON_CLIENT_PREALLOCATE=false` for fitting.

The fast saved-product replay catches the exact original symptom:

```bash
python .scratch/potential-correction/experiments/diagnose_peak_recovery.py --assert-target
```

It reports 51.9213% suppression and exits nonzero on the original products.
Its operational improvement gate requires at least 60% recovery **at the true
discrete peak node**, control core RMS ≤ 0.0011 and maximum |κ| ≤ 0.0031.
These control limits are approximately twice the original matched-control
values, an explicit engineering comparison envelope, not a statistical
false-positive calibration. Location, background error and local aperture
integrals must also be inspected; a global maximum alone can reward noise.

The public-fitter ablation harness is:

```bash
python .scratch/potential-correction/experiments/probe_peak_recovery.py \
  --algorithm iterative --potentials 400000,1600000 \
  --output .scratch/potential-correction/output/peak-recovery/iterative-d4-probes
```

Every candidate cold-starts from the same truth-derived source nodes and zero
potential. The source remains a signed unknown. Both observations receive
identical candidate settings. Each output saves complete arguments, source
hashes, configuration, scalar summaries and public numerical products.
Failures retain diagnostic status and null evidence, and cannot be selected.
`--data linear`, `--data discrete`, and `--noise-factor 0` are explicitly named
synthetic diagnostic ablations; they do not replace the analytic noisy mock.

## Ranked hypotheses, declared before fitting

1. D4 smoothing penalizes the compact cusp and broadens the peak: lowering
   only D4 strength should raise the peak and increase null structure.
2. Source flexibility trades against potential structure: changing only source
   strength should alter peak/background recovery and null artifacts.
3. Analytic-to-grid mismatch and finite resolution: matched discrete synthetic
   data should improve recovery if representation error dominates.
4. Independent amplitude precision: changing it at fixed smoothing should
   restore a substantial peak if it causes the suppression.
5. Numerical stopping: tightening controls should substantially change the
   peak if the baseline is merely an under-converged fit.

## Results and useful files

The [validation report](../../../../docs/validation/potential-correction-peak-recovery.md)
records the causal checks, alternative fits, source-prior effects and recovery
trade-offs. The [compact numerical record](summary.json) retains candidate
statuses, source hashes, selection outcomes and aperture diagnostics. The
[validation policy](validation-policy.json) fixes the numerical controls,
candidate grid and engineering comparison limits before held-out seeds 18/19.

For seed 17, strict converged evidence selection raises the nonlinear peak
recovery from 48.08% to 85.30%, with the same small control as the original.
The subhalo map has higher background variance and whole-core error. The
original aperture already recovered 98.15% of the integrated κ inside 0.15
arcsec, so peak suppression was largely broadening. The original scientific
exports and default configurations remain the historical baseline.

The same predeclared selection policy recovers 86.65% and 97.05% of the peak
on held-out noise seeds 18 and 19. Both controls pass the original engineering
envelope (maximum absolute κ ≤ 0.0031, core RMS ≤ 0.0011); these two noise
checks do not calibrate detection significance or population false positives.

Supporting scripts in `../../experiments/`:

- `diagnose_peak_recovery.py`: deterministic saved-product red/green gate.
- `probe_peak_recovery.py`: actual public-fitter, one-factor ablations and
  common candidate grids; preserves failed statuses and null evidence.
- `select_recovery_candidates.py`: same converged-evidence rule and candidate
  set for each observation; no truth metrics enter selection.
- `decompose_peak_bias.py`: exact single-step superposition check separating
  regularized truth, nonlinear remainder, representation mismatch and noise.
- `summarize_peak_recovery.py`: rebuild the compact study record.
- `plot_peak_recovery.py`: draw signed saved maps, with the panel-alignment
  helper available on `PYTHONPATH`; no fitting or image manipulation.

Full saved products are under ignored `output/peak-recovery/`. The report
distinguishes direct new fits, saved-product checks, development choices and
independent noise-seed validation. Use fresh output directories to reproduce
the study; the selector refuses to merge into an existing selection export.
