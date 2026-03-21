# INVERSION KB

## OVERVIEW
This directory is the inversion core for pixelized source reconstruction. It contains the dense analytic path and the matrix-free operator path.

## STRUCTURE
```text
TinyLensGpu/utils/inversion/
|- linear_solver.py       # LinearInversion, NNLSInversion
`- operator_solver.py     # OperatorInversion, OperatorNNLSInversion
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Dense inversion / log evidence | `TinyLensGpu/utils/inversion/linear_solver.py` | Explicit matrix path |
| Matrix-free solve / SLQ logdet | `TinyLensGpu/utils/inversion/operator_solver.py` | Highest-risk numerical engine |

## CONVENTIONS
- Use the dense path for diagnostics and small systems; use the operator path when matrix materialization is the memory bottleneck.
- `OperatorInversion` is the unconstrained solve path; `OperatorNNLSInversion` is the nonnegative path and relies on FISTA-style updates.
- Small systems can fall back to dense log-determinant evaluation; large systems use SLQ with multiple probes and steps.
- Numerical floors on noise variance, eigvals, and Lipschitz estimates are deliberate stability guards.

## ANTI-PATTERNS
- Do not describe NNLS log evidence as exact; the implementation uses a Laplace-style approximation around the MAP solution.
- Do not lower SLQ / Lipschitz / noise floors casually to chase speed.
- Do not collapse operator and dense code paths into one branch; both are relied on for different debugging and memory tradeoffs.

## NOTES
- `operator_solver.py` is one of the largest files in the repository and the first place to inspect when pixelized inference regresses.
- CG, FISTA, and SLQ defaults are distributed across solver config and operator code; keep them consistent with `SolverConfig` semantics upstream.
