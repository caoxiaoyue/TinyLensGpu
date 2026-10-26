# 09: Run sampling, persistence, and recovery through shared support

**What to build:** One representative operator SLACS stage runs with explicit configuration through reusable sampling and stage persistence, returning StagePosterior and recovering the saved stage through the same supported workflow.

**Blocked by:** 08: Complete and verify the public diagnostics migration

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] Introduce reusable example support for explicit run configuration, sampling execution, and stage persistence, and use it in a complete representative operator SLACS stage.
- [ ] Return StagePosterior from shared sampling while preserving parameter ordering, posterior weights and schema, prior transformation, and the chosen stage's actual sampler controls.
- [ ] Make batch size, live-point settings, vectorization, chunking where supported, and output location explicit rather than reading case-module globals or silently selecting new scientific defaults.
- [ ] Read both existing legacy and schema-bearing stage payloads; write the original format selected by the case, preserving filenames, fields, numerical contents, and relevant representations.
- [ ] Demonstrate the selected stage's sample-to-save-to-recovery path with controlled sampler output and temporary artifacts, including legacy/current artifact compatibility checks.
- [ ] Shared support imports add no working-directory, environment-policy, or JAX precision mutations beyond existing library initialization, and the representative entry point remains usable.
- [ ] Keep unmigrated cases functional through their existing helpers or thin adapters; do not introduce a generic workflow engine or migrate cache formats.
- [ ] Record focused verification. Waiting for the diagnostics migration is the approved P1 delivery-order gate, not an invented numerical dependency.
