# 14: Migrate operator joint and subhalo workflows to shared run support

**What to build:** Operator joint semi-linear inversion and subhalo workflows reuse the common support while retaining their specialized sampling, posterior transfer, stage validation, and protection of saved scientific results.

**Blocked by:** 10: Transfer a source template and render it through shared support

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Migrate affected operator simple joint examples, the operator simultaneous MGE workflow, and the 0946 and B1938 subhalo pipelines; the dense joint workflow remains assigned to the dense-support ticket.
- [x] Preserve explicit case configuration, vectorized chunking and other sampling controls, parameter ordering, prior semantics, and stage posterior transfer.
- [x] Preserve existing stage payload formats and restore behavior, including the B1938 M4/M5 schema checks and protection against overwriting the protected M5 artifact.
- [x] Retain case-specific model and stage assembly while reusing applicable sampling, persistence, source-package, and plotting support.
- [x] Preserve the corrected actual joint component images, unsupported-joint-statistic behavior, subhalo overlays, and approved failure handling from the diagnostics migration.
- [x] Verify representative stage transitions, cache restoration and schema rejection, refusal to overwrite protected output, and relevant chunking behavior with controlled sampling and small artifacts.
- [x] Record focused results without modifying scientific definitions or requiring full-data nested sampling.

## Comments

- Claimed by joint_support on `codex/pwr-14-joint-support` in `/tmp/tinylens-pwr-14`; base integration commit `83f6002`.

- Implemented shared support in both operator simple joint fits, simultaneous
  MGE, and 0946/B1938 pipelines. Samplers return `StagePosterior`; legacy
  simultaneous-MGE payloads remain schema-free and subhalo payloads retain
  `stage_schema`. Batch sizes, vectorized chunking, MGE `f_live`, shifted simple
  weights, B1938 seed/checkpoint/resume, Stage-A warm-up, and sampling JSON
  remain case-specific. Case model/stage assembly and M4/M5 cache/overwrite
  protections stay explicit.
- Source export/validation and figure panels now delegate to shared support.
  Subhalo export retains its failure-message context. Joint figures keep actual
  fitted components and the unavailable-effective-DOF explanation; source and
  subhalo overlays and mask/zoom behavior are preserved.
- Test seams are the spec-approved sampling controls, returned posterior,
  serialized artifacts, source-package identity/normalization, protected output,
  and rendered image data. Initial controlled-sampler test failed on the legacy
  simultaneous-MGE tuple result before migration.
- Verification (conda `tinylens_gpu`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
  `MPLBACKEND=Agg`, working directory `examples/tests`):
  `pytest test_joint_shared_support.py test_b1938_pipeline.py -q` -> **16 passed**.
  The seven new tests cover preserved sampler controls and legacy/schema cache
  round trips, source-package normalization, B1938 rejection of protected M5 and
  differing M4/M5 schemas, and actual joint-component artists. Existing B1938
  tests exercise real PCG stage transitions and invalid reconstruction retention.
  Repeated float32 PCG artist comparisons use rtol=1e-5/atol=1e-7 (observed
  absolute source-intensity variation below 1e-7).
- Integration follow-up: old AST-extraction namespaces in
  `test_operator_diagnostics_plotting.py` need shared helper imports; parent
  coordinator owns that harness update and full-suite verification in ticket 15.
  Normal-import joint plotting and all dedicated ticket checks pass.
