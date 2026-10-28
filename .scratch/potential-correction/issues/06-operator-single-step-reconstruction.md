# 06: Operator single-step reconstruction and block evidence

**What to build:** A caller can reconstruct the same fixed-strength single-step source–potential model using PCG and matrix-free products, with an explicitly labeled block approximation to evidence and results directly comparable to the dense backend.

**Blocked by:** 04 — Dense single-step reconstruction at fixed strengths.

**Status:** ready-for-agent
**State:** resolved

**Parent:** [Approved potential-correction spec](../spec.md).

- [x] Backend selection reaches an operator implementation using the same geometry, initialization, prior, imaging semantics, and result contract as the dense fit.
- [x] Joint forward/transpose and curvature actions retain the full source–potential coupling; block approximation is confined to preconditioning and determinant evaluation.
- [x] PCG uses a suitable block preconditioner and exposes convergence status and residual diagnostics.
- [x] Operator preparation and execution do not require full dense source/potential difference matrices, joint curvature, or a dense source-by-potential cross matrix.
- [x] The block determinant calculation reflects the complete amplitude-augmented potential prior and accounts for support-only coefficients.
- [x] Evidence and result metadata explicitly identify the block approximation; the extension boundary permits later determinant methods without implementing SLQ here.
- [x] Fixed-strength tests compare recovered fields, linear/nonlinear images, objectives, and solve residuals against the dense reference at matched settings.
- [x] A test with nonzero cross-curvature would fail for an accidentally decoupled solve, and forward/transpose inner-product checks pass at declared tolerances.
- [x] Failed PCG or nonfinite results propagate as failed fits rather than valid scientific outputs.
- [x] JIT/GPU execution and explicit precision are retained without changes to existing source–lens-light Schur preconditioning.
- [x] A small runnable fixed-strength operator example and focused verification record demonstrate the slice independently of automatic tuning.

## Scope and handoff

Finite, correctly labeled approximate evidence is necessary but not the final
scientific acceptance of the approximation. Ticket 07 evaluates its effect on
selected strengths and recovery. Do not add an unapproved dense Schur or
stochastic determinant method to bypass that validation.

## Comments

- 2026-10-28: Claimed by ticket06 agent on branch `codex/pc-06` in
  `/tmp/tinylens-pc-06`, based on integration `cdda3e9`. Implements the
  fixed-strength operator backend only; tuning and calibration remain in
  tickets 05 and 07.

- Implemented in `20b5ebd` (`Add coupled matrix-free potential correction with
  principal block evidence`). Merged integration `4301d8b` in `2765e25` and
  ticket-05 integration `f7d96e5` in `2ec636e` before final handoff. The only
  content conflict was the example README; both the operator and dense-tuning
  sections are retained.
- API, exact model/configuration, numerical tolerances, approximation limits,
  memory evidence and reproduction commands are in the
  [operator verification record](../research/operator-single-step.md).
  `backend="operator"` uses full-coupled PCG, cached bounded principal blocks,
  dynamic strengths and highest-precision block products. The complete
  amplitude prior and all support coefficients enter both block determinants.
  Actual coupled residuals, solver controls, exact partition and approximation
  metadata are public result configuration. Dense remains the explicit
  complete reference; source–lens-light Schur preconditioning is unchanged.
- Final post-merge core command from `/tmp/tinylens-pc-06/tests`, after
  activating `tinylens_gpu`: `XLA_PYTHON_CLIENT_PREALLOCATE=false pytest
  test_potential_correction_operator.py test_potential_correction_inference.py
  -q` → **31 passed in 25.42 s** (19 operator cases, 12 merged inference cases).
  Tests cover coupled signed fields, both images, objective and true residual,
  independent complete-prior block determinants at two D4 strengths, explicit
  mixed/padded partitions, subpixels/mask holes/asymmetric PSF, adjoints, both
  dtypes, process precision, repeated candidates/JIT reuse, zero data, failure
  status and invalid controls.
- Shared source/PCG/Schur command from the same directory/environment:
  `XLA_PYTHON_CLIENT_PREALLOCATE=false pytest test_pixelized_operator.py -q
  -k 'pcg or schur or joint_curvature or ragged_source_blocks or
  preconditioner_is_spd'` → **23 passed, 173 deselected in 36.48 s**.
  The combined dense/rendering/solver/autodiff/curvature command in the record
  passed **85 of 86**; one existing float32 adaptive joint source–lens-light
  finite-difference Jacobian check failed. It passed isolated reruns here and
  on unchanged `cdda3e9`. Baseline repetition reproduced four failures in 20
  at default rtol=1e-5; its exact script and both numerical repeat records are
  retained. A fixture-only solve tightening leaves assertions/public defaults
  unchanged. `pytest test_operator_autodiff.py -q` then passed **12 in 73.13 s**.
  Tighter solves reduce but do not establish elimination of this pre-existing
  float32 variability (a second baseline probe had one failure in 20 at 1e-8).
- Separate final example process from `/tmp/tinylens-pc-06/examples/tests`:
  `XLA_PYTHON_CLIENT_PREALLOCATE=false pytest
  test_potential_correction_operator_example.py test_potential_correction_tuning.py
  test_potential_correction_mock.py -q` → **14 passed in 14.61 s**. The runnable
  `single_step_operator.py` also succeeded for subhalo and control observations
  with 109 PCG iterations each and actual residuals below 9e-11. The public NPZ
  fields/images and JSON configuration/approximation metadata are checked.
- Separate allocation process from `/tmp/tinylens-pc-06/benchmarks`:
  `XLA_PYTHON_CLIENT_PREALLOCATE=false pytest
  test_potential_correction_memory.py -q -s` → **1 passed in 5.70 s**. On CUDA,
  9216 source plus 9249 potential coefficients, cap8 and 2309 blocks used
  13,099,264 peak JAX buffer bytes; no full dense field/joint/cross/design
  live-array shape was observed. This synthetic support-only stress check is
  not a 200x200 scientific runtime/memory benchmark. `git diff --check` passed.
- Fixed-strength matches do not validate block-evidence strength selection:
  in the 3x3/source2 example the block-minus-exact log evidence differs by
  approximately -10.06 at D4 strength 0.03 and +0.60 at strength 2.0, despite
  agreeing fields/objectives. Ticket 07 owns optimization/calibration and
  amplitude-prior sensitivity; downstream tickets own scientific recovery,
  iterative workflows and measured performance.

