# Scientific single-step evidence

`summary.json` is the retained compact record for ticket 11's actual
200x200/dpix=0.012/b_sub=0.04 fixed-strength experiment, distinct from the
64x64 calibration and from the unchanged mock generator defaults. It contains
both backend comparisons, matched control, independently fixed amplitude
probes, field/residual/source metrics on declared domains, actual sizes,
configuration and software/implementation hashes. Artifact hashes refer to
the retained integration-worktree output, with figure replot provenance.

Reproduce from the repository root with
[the validation script](../../experiments/validate_single_step_scientific.py),
as documented in the
[scientific validation report](../../../../docs/validation/potential-correction-single-step-scientific.md).
Large arrays and figures remain at the explicitly retained root path
`.scratch/potential-correction/output/single-step-scientific/{reference,amplitude-1000,amplitude-100000}`.

`preliminary-configuration.json` records the earlier operator-only preparation
run, whose products remain under `preliminary-operator` at the same root
output. That preparation did not include the later public supplied-truth
or implementation-hash metadata and is not the accepted reference record.

The useful result is localization with modest full-domain error improvement
and a small control, alongside approximately 54% peak suppression, broad-mode
and prior dependence, and a consequential linear/nonlinear image difference.
Neither finite evidence nor a visually localized peak establishes a calibrated
detection or mass measurement. The 200x200 strengths are fixed, not optimized.
