# 13: Qualify Overlap GPU workspace and chunk width

**What to build:** A GPU user can choose a sampler batch size from matched measurements of Overlap and Block on the qualified implementation. The report provides the evidence for a conservative automatic Overlap chunk width.

**Blocked by:** [12 — Qualify full-score multi-start M3 L-BFGS-B](12-m3-overlap-lbfgs-qualification.md).

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Use the same device for the 80×80 source-only mock and a representative joint semi-linear inversion. Match physical inputs, precision and solver settings when comparing methods.
- [x] Record preparation time, first compilation, warmed score and gradient time, peak device memory and inner chunk effects at matched outer sampler batches.
- [x] Verify bounded forward and reverse scoring workspace on the tested configurations and equivalent scores for the compared chunk widths.
- [x] Select and record a conservative automatic width from the results and demonstrate that explicit `vectorized_chunk_size` continues to control batching. Retain commands and measured configurations.

## Comments

- Claimed on `codex/overlap-13-gpu-qualification`.
- Runner verified at `2a3b93d` and results recorded in `40b0ae2`. From the repository root in `tinylens_gpu`, `OPENBLAS_NUM_THREADS=1 NVIDIA_TF32_OVERRIDE=0 python benchmarks/overlap_evidence_resources.py --archive <archive> --output <output> --outer-batch 23 --repeats 3` completed all 16 isolated CUDA workers. Both matched cases recorded preparation, first compilation, warmed score and gradient times, peak device allocation, and forward, reverse and batched workspace audits. Widths 1 and 5 agreed within `1.15e-10`. Compiled panels showed full chunks of five and a remainder of three. The measurements support automatic Overlap width 1. See the [validation report](../../../docs/validation/adaptive-image-overlap-gpu.md) and [retained results](../experiments/overlap_gpu_results.json).
