# 10: Export new J0946 macro handoffs with the actual source scale

**What to build:** Both documented J0946 macro producers can create a fresh handoff after an adaptive fit. A downstream fixed-scale potential-correction fit can read the selected M3 draw, its actual precision scale and the scoring provenance together.

**Blocked by:** [06 — Run Euclid M1 through M3 with a shared reference arc](06-euclid-adapt-image-stages.md).

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Both producers use the new reference-image adaptive input and record the reference and scoring configuration.
- [x] At the selected M3 draw, obtain the fitted fields, actual adaptive scale and score method from the same valid model evaluation. Save a finite positive scale on the handoff's fitted source grid.
- [x] A focused handoff round trip verifies the existing fixed `source_scale` consumer can read the new product. An archived product remains readable using its recorded scale and configuration.
- [x] Limit the migration to the two documented macro entry points and their handoff behavior.

## Comments

- Claimed on `codex/overlap-10-j0946-handoffs`.
- Implemented in `c9cd8e5`. From `examples/tests/` in `tinylens_gpu` on CPU, `pytest -q test_j0946_single_step_workflow.py test_potential_correction_lbfgs_j0946.py` passed all 59 cases. Both macro drivers passed CLI `--help` and Python compilation checks. The regressions cover a fresh handoff with the actual scale, score and fields from one valid M3 solve, the fixed-scale consumer round trip and archived handoff preservation, and matching stage and sampler recovery.
