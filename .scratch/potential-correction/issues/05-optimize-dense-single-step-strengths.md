# 05: Optimize dense single-step regularization strengths

**What to build:** A researcher can automatically select source and potential smoothing strengths for a dense single-step fit using bounded DIRECT optimization, while retaining fixed initialization and an explicit fixed amplitude prior.

**Blocked by:** 04 — Dense single-step reconstruction at fixed strengths.

**Status:** ready-for-agent
**State:** resolved

**Parent:** [Approved potential-correction spec](../spec.md).

- [x] DIRECT searches the logarithms of the positive source and fourth-difference strengths through the public fit/evidence boundary.
- [x] Search bounds and controls are explicit and returned or saved with the result; an optimum at a search boundary is reported.
- [x] The independent weak zero-order strength remains fixed and does not scale with the searched fourth-difference strength.
- [x] Fixed source/potential geometry and the same initial source-gradient provider are used for every candidate.
- [x] Each candidate performs the required single-step joint fit; there is no search-order-dependent reconstruction warm start.
- [x] Configuration errors and numerical candidate failures remain distinguishable; numerical failures use the optimizer's invalid-objective convention and retain their reasons.
- [x] Optimizer integration preserves the requested numerical precision instead of forcing a float32 parameter path.
- [x] A callable remains available for other optimizers or samplers without introducing a required posterior-sampling workflow.
- [x] A small runnable example returns selected strengths and their reconstruction; tests compare with an independent bounded scan and check repeated/reordered candidate evaluations.
- [x] Tests cover explicit bounds, boundary reporting, fixed amplitude-prior semantics, and failure propagation.

## Scope and handoff

Use the dense evidence reference. The operator adapter and cross-backend
calibration are ticket 07; this ticket does not select the final scientific
demo's weak-prior value or search limits without the agreed sensitivity record.


## Comments

- Claimed by ticket05 agent on branch `codex/pc-05`, worktree `/tmp/tinylens-pc-05`, based on integration commit `cdda3e9767ae9a6c6ba93d0bac9c4e572238d4a5`. Public fit/evidence and inference-callable seams are approved by the parent spec; implementation follows TDD vertical slices.

- Implemented in `ca82c4c` (`Add bounded DIRECT potential-correction strength inference`). Merged integration `4301d8b` in `97325d7` before final verification. The new `Inference` callable and DIRECT wrapper use complete public fits, retain scalar evaluation records and the best successful public reconstruction, reject numerical failures with positive-infinity objective and retained reasons, and preserve configuration exceptions. Natural-log bounds, search controls and boundary proximity are explicit. No fitted candidate is used as an initializer for another. Amplitude strength remains fixed on the prepared problem.
- API, numeric configuration, retained independent comparison and reproduction commands: [dense strength optimization record](../research/dense-strength-optimization.md), [small numerical summary](../research/dense-strength-optimization-summary.json), and [verification script](../experiments/verify_dense_strength_search.py). The independent 31x31 Gaussian-covariance scan differs from the DIRECT best objective by 2.4675e-4; independent evidence at that candidate agrees within 6.03e-13. The potential upper-bound proximity is reported. These bounds and amplitude strength are numerical test inputs, without a scientific weak-prior calibration claim.
- Final core verification, from `/tmp/tinylens-pc-05/tests`, after `source ~/anaconda3/bin/activate && conda activate tinylens_gpu`: `XLA_PYTHON_CLIENT_PREALLOCATE=false pytest test_potential_correction_inference.py test_potential_correction_dense.py test_potential_correction_rendering.py test_prior.py test_stage_posterior.py test_caskade_models.py -q` → **181 passed in 41.91 s**. This includes 12 new inference cases, the 31 dense/rendering cases, and existing inference/physical-model compatibility checks. Tests cover real repeated/reordered fits, fixed geometry/gradient/amplitude, independent scan/evidence, both dtypes in opposite precision contexts, boundary metadata, numerical failure/underflow/overflow, invalid configuration and all-invalid search outcome.
- Separate final example process, from `/tmp/tinylens-pc-05/examples/tests` in the same environment: `XLA_PYTHON_CLIENT_PREALLOCATE=false pytest test_potential_correction_tuning.py test_potential_correction_mock.py -q` → **13 passed in 11.71 s**. The tuning regression verifies saved bounds, candidate records, initialization provenance, source/potential fields, distinct linear/nonlinear renderings and budget-limit status without relabeling it convergence.
- Standalone runnable check, from the worktree root: `XLA_PYTHON_CLIENT_PREALLOCATE=false python examples/potential_correction/tune_single_step.py --lambda-amplitude 0.4 --log-source-bounds -2 2 --log-potential-bounds -4 2 --maxfun 80 --output output/ticket05-dense-tuning` → valid dense reconstruction; selected source/D4 strengths 0.13570708323176198 / 0.01839116718971317, fixed amplitude strength 0.4; lower-bound proximity and DIRECT budget termination reported explicitly. Saved `tuning.json` and `reconstruction.npz`; generated outputs are reproducible with this command.
- Retained experiment, from the worktree root in the same environment: `XLA_PYTHON_CLIENT_PREALLOCATE=false python .scratch/potential-correction/experiments/verify_dense_strength_search.py` → passed numerical assertions and wrote the tracked summary. `git diff --check` passed. Environment: Python 3.11.14, JAX 0.7.1, SciPy 1.17.1, NVIDIA GeForce RTX 5080.
- Scope limits: ticket 07 owns operator optimization/calibration and the final scientific prior/search settings; ticket 10 owns complete iterative-run tuning. The wrapper forwards `backend` and fixed `fit_options` to the same public fit boundary for those later stages. No generic optimizer refactor or ObservationModel changes were needed.
