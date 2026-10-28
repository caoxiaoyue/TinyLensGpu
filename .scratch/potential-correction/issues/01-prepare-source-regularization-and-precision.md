# 01: Prepare source regularization and precision

**What to build:** Existing pixelized-source reconstruction can prepare and run operator inversions without eagerly allocating dense source-difference matrices, while explicit precision is preserved through the shared source preparation needed by potential correction. Existing callers retain their numerical and public-interface behavior.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent
**State:** resolved

**Parent:** [Approved potential-correction spec](../spec.md).

- [x] Operator preparation for the supported finite-difference source priors does not eagerly construct full dense difference matrices; observable allocation or memory evidence accompanies numerical checks.
- [x] Dense callers can still request the corresponding matrices, with the same coefficients, boundary rules, regularization penalties, and determinant behavior.
- [x] Explicit float64 values remain float64 through the relevant shared source-grid and regularization preparation; existing default dtype behavior is preserved.
- [x] Small real dense and operator source reconstructions retain their model images, recovered source, objective/evidence conventions, and solver behavior within justified tolerances.
- [x] Existing nonadaptive regularization choices remain usable; this is not a change to their mathematical prior or an adaptive-regularization project.
- [x] Public imports, existing constructor/call patterns, and supported differentiation behavior remain compatible.
- [x] Existing source–lens-light Schur preconditioning, lens-light basis validation, PSF behavior, and fixed-source boundary semantics are unchanged.
- [x] Relevant core regressions run in the documented project environment, with commands, precision, numerical tolerances, and results recorded.

## Scope and handoff

This is the approved prefactoring slice, not a potential-correction implementation.
Keep changes limited to lazy source regularization and the necessary opt-in
precision propagation. Optimizer-specific precision integration belongs to
ticket 05. Do not change process-wide precision defaults or introduce a general
backend redesign.


## Comments

- Claimed by `/root/ticket01` on `codex/pc-01`, worktree `/tmp/tinylens-pc-01`; based on integration commit `5bbfdfa`. Tests use the pre-approved public numerical, operator, and memory seams from the parent spec.

- Implemented in `bdfa055`. Verification and reproducible allocation evidence: [source-preparation-verification.md](../research/source-preparation-verification.md); executable evidence: [source_preparation_memory.py](../experiments/source_preparation_memory.py).
- In `tinylens_gpu`, from `tests/`, with `XLA_PYTHON_CLIENT_PREALLOCATE=false`: `pytest test_regularization.py -q` — 62 passed; `pytest test_pixelized_source_precision.py test_pixelized_inversion.py -k 'precision or cache or fixed_grid or fixed_bbox' -q` — 23 passed; full relevant source/operator/autodiff command in the verification record — 372 passed. Float64 compact-prior comparisons use 1e-13 tolerance; real dense/operator Gaussian reconstruction comparisons use 1e-9 (PCG rtol 1e-11), and dense evidence uses 1e-12.
- Matrix-free preparation measured 4 live JAX array bytes for each supported prior at n=16,32,64. Dense materialization remains explicitly available, including a first call inside JIT. Existing process precision defaults and prior/solver conventions remain intact. Peak full-workflow memory and optimizer integration remain with tickets 13 and 05 respectively.
