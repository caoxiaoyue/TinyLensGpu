# Saved public consumer fixtures

These small dense results and diagnostic values were captured at integration
commit `e14ea508f53eec3ebe4c6f57f887aa6716a725fc`, before sharing the consumer
math. `baseline.json` records the complete input configuration, original
consumer source hashes, calibration schemas/metrics and scientific summaries.
The NPZ/JSON pairs restore through public `load_result()` without fitting.

The 12-by-12 paired mock deliberately uses nominal `b_sub=0.4` so that the
local diagnostic disk exceeds two potential spacings. Its zero-strength
control must retain that same disk. This is a domain/consumer regression,
not a scientific recovery-acceptance configuration. Both fits have signed
source and potential values; support-only derivatives remain NaN.

Regenerate into an ignored directory from the explicitly selected source tree:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python \
  .scratch/potential-correction/experiments/capture_review_consumers.py \
  --source-root PATH_TO_SOURCE --output IGNORED_OUTPUT
```

The script runs a small public calibration, exports selected public dense
results without another solve, and writes `baseline.json`. Copy only that
compact record and the `subhalo`/`control` result pairs when intentionally
updating this fixture; normal tests never regenerate their expected values.
