# 06: Operator single-step reconstruction and block evidence

**What to build:** A caller can reconstruct the same fixed-strength single-step source–potential model using PCG and matrix-free products, with an explicitly labeled block approximation to evidence and results directly comparable to the dense backend.

**Blocked by:** 04 — Dense single-step reconstruction at fixed strengths.

**Status:** ready-for-agent
**State:** open

**Parent:** [Approved potential-correction spec](../spec.md).

- [ ] Backend selection reaches an operator implementation using the same geometry, initialization, prior, imaging semantics, and result contract as the dense fit.
- [ ] Joint forward/transpose and curvature actions retain the full source–potential coupling; block approximation is confined to preconditioning and determinant evaluation.
- [ ] PCG uses a suitable block preconditioner and exposes convergence status and residual diagnostics.
- [ ] Operator preparation and execution do not require full dense source/potential difference matrices, joint curvature, or a dense source-by-potential cross matrix.
- [ ] The block determinant calculation reflects the complete amplitude-augmented potential prior and accounts for support-only coefficients.
- [ ] Evidence and result metadata explicitly identify the block approximation; the extension boundary permits later determinant methods without implementing SLQ here.
- [ ] Fixed-strength tests compare recovered fields, linear/nonlinear images, objectives, and solve residuals against the dense reference at matched settings.
- [ ] A test with nonzero cross-curvature would fail for an accidentally decoupled solve, and forward/transpose inner-product checks pass at declared tolerances.
- [ ] Failed PCG or nonfinite results propagate as failed fits rather than valid scientific outputs.
- [ ] JIT/GPU execution and explicit precision are retained without changes to existing source–lens-light Schur preconditioning.
- [ ] A small runnable fixed-strength operator example and focused verification record demonstrate the slice independently of automatic tuning.

## Scope and handoff

Finite, correctly labeled approximate evidence is necessary but not the final
scientific acceptance of the approximation. Ticket 07 evaluates its effect on
selected strengths and recovery. Do not add an unapproved dense Schur or
stochastic determinant method to bypass that validation.

