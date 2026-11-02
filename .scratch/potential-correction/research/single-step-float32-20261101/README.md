# Single-step physical-mock Float32 comparison, 2026-11-01

Question: at fixed regularization, does Float32 retain the Float64 subhalo
signal, and how fast is it on Dense followed by the operator backend?

The [validation report](../../../../docs/validation/potential-correction-single-step-float32.md)
interprets the [compact results](summary.json). This experiment is separate
from the earlier precision campaign: it uses only the single-step method and
adds an explicitly declared `solve_rtol=1e-5` probe. Library defaults and all
three regularization strengths remain unchanged.

From the repository root, activate `tinylens_gpu`, then run:

```bash
python .scratch/potential-correction/experiments/compare_single_step_precision.py \
  --fixtures .scratch/potential-correction/output/precision-performance/fixtures \
  --output .scratch/potential-correction/output/single-step-float32-20261101 \
  --repeats 5 --probe-solve-rtol 1e-5 \
  --summary .scratch/potential-correction/research/single-step-float32-20261101/summary.json
```

Use a fresh output directory for a new campaign; existing worker records
are protected from overwriting. `--summarize-only` rebuilds the compact
summary from retained worker records and arrays. Fixtures and worker
artifacts remain read-only; only the requested summary output is written.
The driver will not create missing probe fixtures, worker records or input
directories. It requires every worker record to
match its declared case/observation, backend, dtype, algorithm, fixture
configuration and input SHA256, with `result.configuration.solve_rtol`
equal to the actual fixture tolerance. Cached probe inputs must equal the
original fixture's arrays with only `solve_rtol` replaced; the summary's
`residual_probe_rtol` is derived from the validated probe worker records,
not from the CLI argument, so a CLI/cache disagreement is rejected before
any summary is written. Individual worker JSON, NPZ and logs are retained
under the ignored output directory above.

Note: the archived summary and results in this directory describe the
historical run. Later driver changes only add these read-only/state
reproduction safeguards; they do not recompute or alter the archived
numerical evidence.

Inputs are the prior campaign's frozen `scientific200-{subhalo,control}.npz`.
To rebuild equivalent fixtures into a fresh directory, use
`benchmarks/potential_correction_workflow.py --prepare --case scientific200
--observation subhalo --output /path/to/fixtures/scientific200-subhalo.npz`
and repeat for `control`. This regenerates the configured seed-17 mock;
the existing saved fixtures are used for the recorded comparison. See the
[earlier campaign](../../../../docs/validation/potential-correction-precision-performance.md)
for canonical-artifact-based fixture preparation. Exact input and implementation
hashes, current software, topology changes and requested-dtype audits are
retained with each run.

The driver launches isolated workers serially, completing Dense before the
operator backend. Each of the 16 workers evaluates one cold fit; successful
workers repeat five complete fits with explicit device synchronization.
Failures retain diagnostic fields and null evidence, without warm timing or
a claimed speedup. The residual probe uses identical physical arrays and
changes only the metadata's solve acceptance threshold. Recursive PCG remains
`rtol=1e-12`, `atol=0`, with 20000 iterations maximum in both precisions.

Comparison checks require identical frozen physical inputs, fixed prior
settings, potential node topology and derivative validity. A separate NumPy
diagonal audit records whether adding the amplitude prior survives Float32
rounding at each potential node. It explains a numerical mechanism and does
not replace the solver's factorization and residual checks.
