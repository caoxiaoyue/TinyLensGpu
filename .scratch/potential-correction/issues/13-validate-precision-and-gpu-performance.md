# 13: Validate float32 and measure GPU performance

**What to build:** A user can judge the numerical and resource trade-offs of the completed correction workflows from reproducible float64/float32 comparisons and GPU timing/memory measurements at explicitly recorded problem sizes.

**Blocked by:** 12 — Iterative scientific examples, diagnostics, and exports.

**Status:** ready-for-agent
**State:** open

**Parent:** [Approved potential-correction spec](../spec.md).

- [ ] Float64 references cover both algorithms and both backends before float32 is presented as a performance option.
- [ ] Precision comparisons report model residuals, recovered-field differences, solver behavior, and relevant evidence/strength-selection differences at matched configurations.
- [ ] Numerical paths are checked for hidden intermediate downcasts; requested precision and global default behavior remain distinct.
- [ ] Any float32 limitation is documented with the affected conditioning/prior/configuration and is not hidden by silently changing the physical model or declaring failed fits converged.
- [ ] Benchmarks distinguish first compilation, warm numerical solve/evaluation, and complete workflow costs, using appropriate synchronization.
- [ ] Reports record GPU identity, software/runtime context, dtype, observation and source sizes, core/support node counts, integration settings, prior strengths, and solver controls.
- [ ] Peak memory and scaling checks include preparation as well as solve execution, exposing eager dense source differences or full joint/cross-field allocations if present.
- [ ] Measurements include the available approximately 16 GiB RTX 5080 and explicitly configured small-to-larger cases that remain within available resources.
- [ ] Timing/scaling checks live in the benchmark suite; numerical correctness remains in core or example tests as appropriate.
- [ ] A reproducible report and benchmark invocation describe measured results without promising an unmeasured speedup or treating approximate and complete evidence as interchangeable.

## Scope and handoff

This is numerical/resource validation of delivered workflows, not a new solver,
SLQ project, or blanket repository precision change. Correct regressions in
the approved paths when found and record limits rather than changing the spec.

