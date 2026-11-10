# 11: Reproduce the 80×80 Euclid mock with image-driven adaptation

**What to build:** A researcher can rerun the original Euclid mock with the new library and inspect the determinant approximation separately from reconstruction error. The run supplies a reproducible M3 configuration for optimizer qualification.

**Blocked by:** [05 — Differentiate and batch the full Overlap score](05-overlap-gradients-and-batching.md).

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Use the archived no-noise mock with its RMS weighting, 80×80 source grid, mask, PSF, subpixel integration and recorded SIE M1 reference producer.
- [x] Run a new M2 calibration and M3 fit with the image-driven prior. Record the actual model, interpolation, adaptive settings, fitted strengths and evidence method.
- [x] At matched proposals, compare Block, Overlap and complete determinant scores using the same primary solution. Report absolute offsets, centered errors and ranking as separate quantities.
- [x] Retain a reproducible runner, input identities, settings and small result summaries. Record slope recovery and validity for the actual M3 configuration.

## Comments

- Claimed on `codex/overlap-11-mock-replay`.
- Implemented in `18d207d`. The retained `euclid_mock_replay.py` completed archive verification, fresh M2 and M3 fitting, and the determinant comparison on CUDA in `tinylens_gpu`. The SIE M1 arc re-render agreed with the archived arc to a maximum absolute difference of `9.54e-7`. All 19 shared primary solves converged. Commands, configuration, slope recovery, score offsets, centered errors, and ranks are in the [validation report](../../../docs/validation/adaptive-image-overlap-euclid-mock.md) and [machine-readable results](../experiments/euclid_mock_results.json).
