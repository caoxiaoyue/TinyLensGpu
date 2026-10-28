# 09: Operator iterative reconstruction at fixed strengths

**What to build:** The iterative correction workflow can use the operator backend for each coupled update, retaining the dense workflow's objective, control policy, and failure semantics without assembling the full iteration curvature.

**Blocked by:** 08 — Dense iterative reconstruction at fixed strengths.

**Status:** ready-for-agent
**State:** claimed

**Parent:** [Approved potential-correction spec](../spec.md).

- [ ] Operator backend selection uses the shared iterative control policy and fixed-strength scientific model.
- [ ] Every linearization uses current mappings and source gradients in the forward, adjoint, and joint curvature actions.
- [ ] Static geometry is reused, while dynamic iteration values cannot be mistaken for fixed cached inputs.
- [ ] PCG retains full source–potential coupling and uses the approved block preconditioning without full dense curvature or cross-field matrices.
- [ ] Step damping is applied consistently in the numerical solve but is absent from the final physical prior and evidence.
- [ ] Successful evidence is labeled separately for local Gauss–Newton linearization and block determinant approximation, including the complete amplitude-augmented prior.
- [ ] Small matched problems compare dense/operator nonlinear images, recovered fields, objectives, and linear-solve residuals at declared tolerances.
- [ ] Tests detect stale-gradient/mapping reuse and verify that unsuccessful PCG, nonfinite states, and rejected updates propagate through the common termination contract.
- [ ] JIT/GPU execution, explicit dtype, observation-mask preservation, and valid-domain diagnostics remain consistent with the earlier slices.
- [ ] A small fixed-strength operator iteration is runnable and returns the same result structure and history semantics as the dense workflow.

## Scope and handoff

The operator single-step machinery is an already-resolved transitive
dependency through ticket 08's stage gate. Do not add redundant direct blocking
edges or introduce a second iteration controller. Nested optimization remains
ticket 10.


## Comments

- 2026-10-29: Claimed by delegated ticket-09 agent on `codex/pc-09` in `/tmp/tinylens-pc-09`, based on integration `f00868e` (ticket 08 resolved). This branch owns only fixed-strength operator iteration; approved public-result/operator/allocation seams from the parent spec apply.
