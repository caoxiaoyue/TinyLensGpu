## Context

The operator pixelized-source backend lives across the ForwardSimulation and ObservationModel layers. `PixelizedLensOperator` already provides matrix-free forward, adjoint, RHS, and `A(s)` matvec operations for source-only inversions, while `PixelizedImageProbModelOperator` currently uses PCG to solve the unconstrained MAP source.

The current soft positivity option, `source_nonnegativity_sigma`, penalizes negative source pixels after the unconstrained PCG solve. It does not guarantee physical non-negative source brightness. The dense pixelized backend already supports hard non-negativity through dense NNLS, but the operator backend cannot use that path without materializing dense design or curvature matrices.

## Goals / Non-Goals

**Goals:**

- Add a JAX-friendly matrix-free FISTA solver for the quadratic constrained problem `min_s>=0 0.5 s^T A s - b^T s`.
- Integrate FISTA into `PixelizedImageProbModelOperator` as an explicit solver choice while retaining the existing PCG path.
- Guarantee `source_pixels >= 0` for `solver_type="fista"` in evidence and `forward_model(return_source=True)`.
- Keep memory usage O(Ns) beyond existing operator data by storing only FISTA state vectors and reusing existing matvec/RHS machinery.
- Remove `source_nonnegativity_sigma` and the associated soft evidence penalty.

**Non-Goals:**

- No active-set Laplace correction in this change.
- No exact truncated-Gaussian evidence calculation.
- No stochastic logdet estimation.
- No lens-light joint inversion support for the operator backend.
- No GP regularization support for the operator backend.

## Decisions

1. **Use FISTA rather than adapting PCG to constraints.**

   FISTA handles the non-negative orthant with a simple projection `max(x, 0)` after each gradient step. PCG is appropriate for unconstrained SPD systems but does not naturally enforce bound constraints. Alternatives considered were projected gradient descent and active-set NNLS. PGD is simpler but typically slower; active-set NNLS is more complex and less naturally matrix-free for the current operator structure.

2. **Place the reusable solver in `TinyLensGpu/utils/`.**

   The FISTA implementation should be a JAX utility similar to `TinyLensGpu/utils/cg_solver.py`. It should accept `A_data`, `b`, a static prebound matvec callable, and solver settings, then return `(source_pixels, info)`. This keeps the ObservationModel layer focused on likelihood/evidence assembly and allows focused unit tests for the solver.

3. **Expose `solver_type` on `PixelizedImageProbModelOperator`.**

   The constructor should accept `solver_type="pcg"` by default and `solver_type="fista"` for hard non-negative source reconstruction. Keeping `pcg` as default preserves existing unconstrained behavior except for the removed soft prior argument. The Caskade-facing callable remains `__call__` and `forward_model`; only internal source solving changes based on the configured solver.

4. **Estimate or configure the FISTA step size without dense matrices.**

   The solver should support a matrix-free step-size strategy that estimates the largest eigenvalue of `A` using a fixed-count power iteration, with a safety factor before computing `step_size = 1 / L`. Fixed iteration counts keep JIT tracing predictable. A user-provided positive `step_size` can be considered as an implementation detail if it simplifies testing, but dense spectral calculations are out of scope.

5. **Use fixed JAX loops and explicit diagnostics.**

   The FISTA body should use `jax.lax.scan` or another JIT-compatible fixed-shape loop. Diagnostics should include objective or gradient-based convergence information and a boolean convergence flag. `PixelizedImageProbModelOperator` should gate evidence similarly to the current PCG path when the solver fails to converge.

6. **Keep the existing evidence logdet approximation for FISTA.**

   For `solver_type="fista"`, the MAP source is constrained, but the evidence keeps the existing operator approximation: `logdet(A) ~= logdet(P)` and `logdet(R)` from `reg_builder.logdet_free(...)`. This is an intentional first-step approximation. The docstring should state that FISTA evidence is a constrained-MAP score with the existing unconstrained-style logdet approximation.

7. **Delete the soft positivity prior API.**

   `source_nonnegativity_sigma` should be removed from the operator constructor, validation, evidence calculation, tests, and examples. Users requiring hard non-negativity should use `solver_type="fista"`.

## Risks / Trade-offs

- **Approximate evidence semantics** -> The FISTA path enforces non-negative MAP source pixels but does not compute the exact constrained evidence. Mitigation: document the approximation and keep active-set Laplace as future work.
- **FISTA convergence can be slower than PCG** -> Ill-conditioned systems or weak regularization may require more iterations. Mitigation: expose iteration/tolerance settings and return diagnostics; penalize non-converged solves in evidence.
- **Power iteration adds matvec cost** -> Each likelihood evaluation may do extra `A(s)` calls before FISTA. Mitigation: use a small fixed iteration count and reuse existing matrix-free matvecs; no dense memory growth.
- **Breaking removal of `source_nonnegativity_sigma`** -> Existing examples using the soft prior must migrate. Mitigation: update examples and tests in the same change.
- **Non-smooth solver output near active constraints** -> Projected solves can create small non-smoothness as pixels hit zero. Mitigation: this is acceptable for nested sampling use; future active-set evidence work can refine the approximation.
