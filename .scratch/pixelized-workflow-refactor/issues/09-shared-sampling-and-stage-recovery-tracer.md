# 09: Run sampling, persistence, and recovery through shared support

**What to build:** One representative operator SLACS stage runs with explicit configuration through reusable sampling and stage persistence, returning StagePosterior and recovering the saved stage through the same supported workflow.

**Blocked by:** 08: Complete and verify the public diagnostics migration

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Introduce reusable example support for explicit run configuration, sampling execution, and stage persistence, and use it in a complete representative operator SLACS stage.
- [x] Return StagePosterior from shared sampling while preserving parameter ordering, posterior weights and schema, prior transformation, and the chosen stage's actual sampler controls.
- [x] Make batch size, live-point settings, vectorization, chunking where supported, and output location explicit rather than reading case-module globals or silently selecting new scientific defaults.
- [x] Read both existing legacy and schema-bearing stage payloads; write the original format selected by the case, preserving filenames, fields, numerical contents, and relevant representations.
- [x] Demonstrate the selected stage's sample-to-save-to-recovery path with controlled sampler output and temporary artifacts, including legacy/current artifact compatibility checks.
- [x] Shared support imports add no working-directory, environment-policy, or JAX precision mutations beyond existing library initialization, and the representative entry point remains usable.
- [x] Keep unmigrated cases functional through their existing helpers or thin adapters; do not introduce a generic workflow engine or migrate cache formats.
- [x] Record focused verification. Waiting for the diagnostics migration is the approved P1 delivery-order gate, not an invented numerical dependency.

## Comments

- Claimed/resolved by Sidekick on `pwr/09-shared-stage-support`. New
  `TinyLensGpu/example_support` package: `SamplingConfig`/`run_sampling`
  (lazy external-Sampler factory, before/after hooks, optional stabilized
  weights, n_batch forwarding only when set) + `posterior_quantiles`, and
  `StageStorageConfig`/`save_stage`/`load_stage`/`stage_from_payload`
  (schema-bearing vs legacy formats). SLACS pipeline `_run_sampler`/
  `_dump_stage`/`_load_stage`/`_stage_from_payload` are thin adapters
  (`n_batch=100` when vectorized); stage-return schema unchanged.
- Verification (conda tinylens_gpu, XLA_PYTHON_CLIENT_PREALLOCATE=false,
  MPLBACKEND=Agg, examples/tests):
  `pytest test_pixelized_support.py -q` -> 8 passed (controlled-sampler
  controls/prior order/weights/hooks, legacy+schema persistence, real SLACS
  run_stage_a save/recovery, isolated no-mutation import, --help entrypoint).
  `pytest test_source_compatibility.py -k stage_propagates -q` -> 2 passed.
- Review rework: `posterior_quantiles` now returns the ordered dict
  `{name: quantile_array}` (nautilus_posterior_summary convention, callers use
  `.values()`). Tests strengthened: `run_sampling` control test exercises
  `vectorized_chunk_size`, real loglike calls on a 3-row batch vs public
  `likelihood(theta)` (expected computed before traced calls to avoid tracer
  leaks), exact `sampler_kwargs`/`run_kwargs` pass-through; stabilized-weights
  case now uses log_w ~ -1000 so the flag is decisive; persistence test adds
  top-level `prior_specs` recovery and raw JAX float32 grid round-trip;
  `--help` targets the known `model.py` with PYTHONPATH set. Rerun:
  `pytest test_pixelized_support.py -q` -> 8 passed.
