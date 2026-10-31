# Cost of one joint inversion and evidence evaluation

The [measurement report](../../../../docs/validation/potential-correction-evidence-timing.md)
defines the timed work and records the 200 × 200 float64 dense GPU results.
The [compact record](summary.json) retains hardware, configuration, hashes,
all warm samples and reference checks for the four measured settings.

The coordinator is `../../experiments/benchmark_evidence_evaluation.py` and
the isolated worker is `benchmarks/potential_correction_workflow.py` from the
repository root. Original and selected scientific exports are required to
prepare the identical mock and validate numerical results. Use `tinylens_gpu`
and a fresh output directory; see the report for the exact command.

All 20 complete timed fits succeeded. Warm medians are 0.645/0.649 seconds
for original/selected single-step settings and 33.819/14.905 seconds for
original/selected nonlinear settings. Each fit starts over; warm refers to
fixed-geometry/kernel reuse, not reuse of a converged reconstruction.

Full fixtures, numerical exports and logs remain under ignored
`output/evidence-evaluation-benchmark-2026-10-30/`. These timings describe
specific settings, not a full source/lens regularization optimization.

The [matched operator record](operator-summary.json) uses the identical frozen
fixtures and repeats. Operator warm medians are 0.279/0.214 seconds for
original/selected single-step settings and 39.497/16.434 seconds for
original/selected nonlinear settings. All 20 additional fits pass the dense
field/objective guards. Operator evidence uses principal-block determinant
approximations and is checked separately from dense evidence.

`../../experiments/benchmark_operator_evidence_evaluation.py` accepts the
existing dense output via `--dense-run` and a fresh operator `--output`.
The report gives the exact command, first-fit costs, PCG settings and numerical
checks. The core library and shared benchmark worker remain unchanged.
