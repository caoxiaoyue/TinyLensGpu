# 09: Operator iterative reconstruction at fixed strengths

**What to build:** The iterative correction workflow can use the operator backend for each coupled update, retaining the dense workflow's objective, control policy, and failure semantics without assembling the full iteration curvature.

**Blocked by:** 08 — Dense iterative reconstruction at fixed strengths.

**Status:** ready-for-agent
**State:** resolved

**Parent:** [Approved potential-correction spec](../spec.md).

- [x] Operator backend selection uses the shared iterative control policy and fixed-strength scientific model.
- [x] Every linearization uses current mappings and source gradients in the forward, adjoint, and joint curvature actions.
- [x] Static geometry is reused, while dynamic iteration values cannot be mistaken for fixed cached inputs.
- [x] PCG retains full source–potential coupling and uses the approved block preconditioning without full dense curvature or cross-field matrices.
- [x] Step damping is applied consistently in the numerical solve but is absent from the final physical prior and evidence.
- [x] Successful evidence is labeled separately for local Gauss–Newton linearization and block determinant approximation, including the complete amplitude-augmented prior.
- [x] Small matched problems compare dense/operator nonlinear images, recovered fields, objectives, and linear-solve residuals at declared tolerances.
- [x] Tests detect stale-gradient/mapping reuse and verify that unsuccessful PCG, nonfinite states, and rejected updates propagate through the common termination contract.
- [x] JIT/GPU execution, explicit dtype, observation-mask preservation, and valid-domain diagnostics remain consistent with the earlier slices.
- [x] A small fixed-strength operator iteration is runnable and returns the same result structure and history semantics as the dense workflow.

## Scope and handoff

The operator single-step machinery is an already-resolved transitive
dependency through ticket 08's stage gate. Do not add redundant direct blocking
edges or introduce a second iteration controller. Nested optimization remains
ticket 10.

## Comments

- 2026-10-29: Claimed by delegated ticket-09 agent on `codex/pc-09` in `/tmp/tinylens-pc-09`, based on integration `f00868e` (ticket 08 resolved). This branch owns only fixed-strength operator iteration; approved public-result/operator/allocation seams from the parent spec apply.

- 2026-10-29: Implemented in `7c642bd`; latest integration `f26cc49` merged without conflicts as `1597ca9`. The same `IterativePotentialCorrection` controller selects a numerical operator adapter with explicit dynamic current mappings/gradients, exact coupled PCG and exact undamped step measures. Cached bounded unit-prior blocks and a fixed partition are separate from refreshed current data blocks; damping affects step factors, never final prior/evidence. No CG or source–lens-light Schur code changed.
- Verified in `tinylens_gpu`, with `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `PYTHONPATH=/tmp/tinylens-pc-09`. From `tests/`, `pytest -q test_potential_correction_iterative_operator.py test_potential_correction_iterative.py test_potential_correction_dense.py test_potential_correction_operator.py test_potential_correction_rendering.py test_potential_correction_inference.py`: **108 passed in 95.23s**. The source–lens-light command `pytest -q test_pixelized_operator.py -k 'pcg or schur or joint_curvature or ragged_source_blocks or preconditioner_is_spd'`: **23 passed, 173 deselected in 34.95s**.
- Separate process from `examples/tests/`, `pytest -q test_potential_correction_iterative_example.py test_potential_correction_operator_example.py test_potential_correction_scientific.py`: **6 passed in 30.65s**. Standalone `python examples/potential_correction/iterate_fixed_strength.py --backend operator --block-size 16 --pcg-max-iter 800 --pcg-rtol 1e-12` converged in six accepted updates, objective 0.0460958021481, true relative residual 3.06e-12.
- Separate process from `benchmarks/`, `pytest -q -s test_potential_correction_memory.py`: **2 passed in 11.10s**. The operator iteration probe used 18,465 coefficients, two current points and nine PCG solves/retries; peak JAX buffers 14,563,840 bytes with no prohibited dense shapes. This is a deliberately iteration-limited storage probe, not a scientific convergence result. `black --check` on nine changed Python files and `git diff --check` pass.
- [Validation and limitations](../../../docs/validation/potential-correction-operator-iteration.md) record declared tolerances, independently checked complete prior/current undamped determinants, failures, float32, JIT and allocation evidence. [Reproduction script](../experiments/iterative_operator_comparison.py) and [small numerical record](../research/operator-iterative-comparison.json) retain configurations. The nsub=2 coupled fixture changes 11 source cells and matches dense fields at <1e-14 in the recorded run. Coarse-block evidence offsets remain explicit; nonlinear strength selection/calibration and scientific recovery remain later tickets, not claims of this slice.
