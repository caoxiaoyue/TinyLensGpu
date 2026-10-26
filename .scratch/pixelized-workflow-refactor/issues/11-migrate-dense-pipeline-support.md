# 11: Migrate dense pipelines to shared run support

**What to build:** Dense pipelines use reusable configuration, sampling, persistence, source-transfer, and plotting support while keeping their existing scientific stages and legacy artifacts usable.

**Blocked by:** 10: Transfer a source template and render it through shared support

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] Migrate affected dense pipelines, including SLACS, the pre-subtraction family, dense simultaneous joint MGE, and the dense matrix variant located among operator examples.
- [ ] Replace private tuple-returning sampling consumption with StagePosterior while retaining each case's prior semantics, parameter ordering, and numerical sampling settings.
- [ ] Preserve intentional sampler differences, including cases whose batch size follows the live-point setting rather than another case's fixed value.
- [ ] Preserve legacy cache filenames, formats, fields without a stage schema where applicable, numerical representations, and recovery behavior despite the unified in-memory posterior type.
- [ ] Use explicit configuration and shared source-transfer and plotting support wherever applicable while retaining visible model construction, stage order, and existing launch methods.
- [ ] Verify representative dense stage transitions, saved-artifact compatibility, restoration, and plot data; keep core numerical behavior unchanged and record focused results.
