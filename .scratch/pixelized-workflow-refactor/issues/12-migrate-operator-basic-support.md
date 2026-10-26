# 12: Migrate basic operator workflows to shared run support

**What to build:** Basic operator, no-lens-light, adaptive, and Galan workflows reuse the shared support without changing their source templates, grid policies, sampling choices, or scientific outputs.

**Blocked by:** 10: Transfer a source template and render it through shared support

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Migrate the affected reusable support in basic operator workflows, standard and adaptive no-lens-light pipelines, and inferred- and predefined-bounding-box Galan pipelines.
- [x] Use explicit configuration and StagePosterior-based shared sampling while preserving each case's actual parameters, batch behavior, and input/output locations.
- [x] Preserve adaptive template propagation, fixed-grid interpretation, source-package identity and normalization, and optional expected-bounding-box enforcement.
- [x] Preserve each case's stage payload format, resume behavior, example launch method, and explicit model and stage assembly.
- [x] Use shared plotting and export support while keeping the public diagnostic results, existing render variants, and approved invalid-reconstruction handling.
- [x] Verify representative complete stage transitions and saved artifacts for the materially different source-template and grid variants; record focused verification without requiring full scientific sampling.

## Comments

- Claimed by `/root/basic_support` on `codex/pwr-12-basic-support` in `/tmp/tinylens-pwr-12`; based on integration commit `83f6002`.

- Implemented in `3e53c44`: no-lens-light standard/adaptive and Galan
  inferred/predefined pipelines delegate sampling, stage storage, source-package
  validation/export and panel drawing to shared support. All basic source,
  compact-source and EPL simple fitting/demo scripts use the same support.
  Entry-point model/stage assembly and runtime policies remain explicit.
- Preserved fixed vectorized batch size 100 (omitted for scalar sampling),
  simple-demo stabilized weights, the adaptive pipeline's schema-less pickle
  payload, schema-bearing payloads for the other three pipelines, dictionary
  identity and float32 scale-map normalization, and Galan's optional expected
  bbox/axis checks. Adaptive private sampling consumers now use StagePosterior.
- Verification in conda `tinylens_gpu`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
  `MPLBACKEND=Agg`, with `PYTHONPATH=/tmp/tinylens-pwr-12`, from `examples/tests`:
  `pytest test_basic_shared_support.py -q` -> **14 passed** (28.66 s).
  Covers direct module sampling/cache restoration for all four pipeline
  variants; real source export to adaptive next-stage construction and five
  rendered arrays for each variant; expected-bbox rejection; real adaptive
  stage-A artifact; a complete simple-script launch with temporary FITS and
  a controlled external sampler, CSV/gzip-pickle/PNG outputs; and mask FITS,
  plot outputs and distinct viridis/magma variants. The initial red run exposed
  the adaptive sampler's tuple-returning behavior before migration.
- `python -m compileall -q examples/pix_src_demo_operator/pipe/no_lens_light
  examples/pix_src_demo_operator/pipe/galan24_test
  examples/pix_src_demo_operator/simple` and `git diff --check` passed.
  Full-suite regression and compatibility-test harness consolidation are
  coordinated by ticket 15; no long scientific sampling was needed.
