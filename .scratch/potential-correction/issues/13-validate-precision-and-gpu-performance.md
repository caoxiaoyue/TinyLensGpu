# 13: Validate float32 and measure GPU performance

**What to build:** A user can judge the numerical and resource trade-offs of the completed correction workflows from reproducible float64/float32 comparisons and GPU timing/memory measurements at explicitly recorded problem sizes.

**Blocked by:** 12 — Iterative scientific examples, diagnostics, and exports.

**Status:** ready-for-agent
**State:** resolved

**Parent:** [Approved potential-correction spec](../spec.md).

- [x] Float64 references cover both algorithms and both backends before float32 is presented as a performance option.
- [x] Precision comparisons report model residuals, recovered-field differences, solver behavior, and relevant evidence/strength-selection differences at matched configurations.
- [x] Numerical paths are checked for hidden intermediate downcasts; requested precision and global default behavior remain distinct.
- [x] Any float32 limitation is documented with the affected conditioning/prior/configuration and is not hidden by silently changing the physical model or declaring failed fits converged.
- [x] Benchmarks distinguish first compilation, warm numerical solve/evaluation, and complete workflow costs, using appropriate synchronization.
- [x] Reports record GPU identity, software/runtime context, dtype, observation and source sizes, core/support node counts, integration settings, prior strengths, and solver controls.
- [x] Peak memory and scaling checks include preparation as well as solve execution, exposing eager dense source differences or full joint/cross-field allocations if present.
- [x] Measurements include the available approximately 16 GiB RTX 5080 and explicitly configured small-to-larger cases that remain within available resources.
- [x] Timing/scaling checks live in the benchmark suite; numerical correctness remains in core or example tests as appropriate.
- [x] A reproducible report and benchmark invocation describe measured results without promising an unmeasured speedup or treating approximate and complete evidence as interchangeable.

## Scope and handoff

This is numerical/resource validation of delivered workflows, not a new solver,
SLQ project, or blanket repository precision change. Correct regressions in
the approved paths when found and record limits rather than changing the spec.

## Comments

- 2026-10-29: Claimed by `/root/performance_exploration` on `codex/pc-13`,
  isolated worktree `/tmp/tinylens-pc-13`, based on integration
  `8b153edd8ec5b19e10902bf2e843a1373e61a2fd`. Issue 12 is resolved.
  Public fit/evidence/operator/result-artifact and benchmark seams were
  preapproved in the parent spec and the implementation assignment.
- 2026-10-29: Completed the bounded campaign: four fresh strict float64
  coupled37 references, eight audited and freshly reproduced scientific
  float64 fits, then matched float32 comparisons. The named small profile
  passes the unchanged `atol=rtol=2e-4` field guard and declared 0.002
  same-method evidence budget in six cold/warm observations per
  algorithm/backend/dtype. Strict-versus-named-control float64 comparisons
  isolate stopping effects. Rejected loose-PCG calibration is retained.
  No public solver default, physical prior/model or controller was changed.
- All eight scientific float32 fits fail honestly with null evidence at the
  unchanged stiff configuration. Corner-rounding differences, actual dtype
  contracts and all limitations are reported. Eight bounded DIRECT searches
  each retain 27 valid candidates for a soft budget of 25; identical sampled
  strengths across dtype/method are budget-limited selections, with method
  evidence offsets separately identified.
- Serial RTX 5080 measurements cover real 32/64/200 images in both algorithms
  and backends and a fixed-64-image source side 33-to-65 operator axis.
  Parent processes do not initialize JAX; workers synchronize dataclass
  arrays, report setup/compile-inclusive first/warm/cold workflow costs and
  allocator/process GPU/host memory with their distinct semantics. Operator
  allocator peak exceeding dense at the scientific layout is retained.
- Report: `docs/validation/potential-correction-precision-performance.md`.
  Versioned budgets, 45-worker compact summary, 26 comparisons, rejected
  calibration and canonical-reference reproduction live in
  `.scratch/potential-correction/research/precision-performance/`; reproduction
  script is `experiments/validate_precision_performance.py`. Large inputs,
  raw worker JSON/NPZ/logs, memory samples and hashed measurement-driver
  snapshots are preserved in the root checkout's ignored
  `.scratch/potential-correction/output/precision-performance/`.
- Verification in `tinylens_gpu`, separate processes and each file's directory:
  `tests/: pytest -q test_potential_correction_*.py` — **132 passed**, 188.69 s;
  `examples/tests/: pytest -q test_potential_correction_*.py` — **30 passed**,
  90.35 s; `benchmarks/: pytest -q -s test_potential_correction_memory.py
  test_potential_correction_preconditioner.py test_potential_correction_workflow.py`
  — **5 passed**, 37.46 s. Final focused dtype/supported-stopping recheck:
  **5 passed**, 14.16 s. Black/Ruff default checks and `git diff --check` pass.
  TDD red observations: missing workflow driver; unsupported source-axis
  invocation; original strict float32 stopping node (`rejection_limit`,
  8.39 s). The calibrated supported node and public CLI regressions pass
  with the original numerical field guard unchanged.

