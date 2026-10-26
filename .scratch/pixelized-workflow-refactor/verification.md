# Pixelized workflow refactor verification

Date: 2026-10-27. Requirements: [spec](spec.md), [tickets](issues/).
The resumed implementation starts at `83f6002` (tickets 01–10 resolved).
Tickets 11–15 complete the shared example-support deliverable on
`integration/pixelized-workflow-refactor`; the earlier truthiness and diagnostic
changes retain their separate implementation history.

## Environment and inputs

- Conda environment: `tinylens_gpu`, Python 3.11.14, JAX 0.7.1.
- GPU: NVIDIA GeForce RTX 5080 (16 GB), JAX CUDA backend.
- Commands run from the repository root with
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `MPLBACKEND=Agg`.
- Core fixtures retain their documented dtype, solver budgets and tolerances.
  Example checks use small real dense/operator models, temporary FITS/cache
  artifacts, and controlled external samplers. Model construction, prior
  transforms, source inversions, stage transfer, persistence and rendering are
  exercised by the actual code. No full-data nested-sampling campaign is claimed.

```bash
source ~/anaconda3/bin/activate && conda activate tinylens_gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
pytest -q
pytest examples/tests -q
pytest benchmarks -q
```

Each suite runs in its own process. Benchmarks run after correctness tests have
finished, with the existing thresholds unchanged.

## Final suite results

| Suite | Result | Duration |
| --- | --- | --- |
| Core library | 813 passed | 367.59 s |
| Examples and pipelines | 256 passed | 154.77 s |
| Existing benchmarks | 9 passed | 8.37 s |

The core run includes unchanged operator/solver autodifferentiation, batched
likelihood, source-grid, regularization, prior, and posterior-transfer checks.
The example inventory and deliberately retained non-stage serialization are
recorded in [the overview](README.md#shared-support-inventory-tickets-1115).

## Focused compatibility checks

The new directly imported example tests cover:

- Dense: six stage-A sampling/artifact variants, a real SLACS pixelized stage
  transition and plot, a full small FITS/CSV/gzip simple-script launch, and dense
  matrix M2 batch size 1 (9 checks).
- Basic operator: four sampling/cache variants, four export→adaptive-stage→plot
  variants, predefined bbox rejection, adaptive stage-A legacy payload, a small
  simple-script launch, and four mask/FITS rendering variants (14 checks).
- Surveys: six stage-A artifact/recovery paths, five in-place template
  validators and three real SLACS/Euclid/MGE source-transfer/render paths
  (14 checks). Existing survey tests also compare masks, crops, colors,
  position overlays and failure propagation.
- Joint/subhalo: sampler/checkpoint/chunking/metrics and cache contracts,
  B1938 M4/M5 schema rejection and M5 overwrite protection, source dictionary
  normalization and identity, actual nonzero joint component artists with
  explicitly unavailable effective degrees of freedom (7 checks).

Commands and results are retained with tickets 11–14. The focused rendering
and source-export compatibility run passed all 133 checks. During integration,
these checks caught and corrected dense demo title initialization order and
adapted isolated script namespaces to the shared imports and informative
reconstruction errors. The first complete examples run additionally exposed
three old B1938 fixture references to a removed private axis helper. The fixture
now uses public `source_axes_from_bbox`; its 9 tests pass. The final complete
examples rerun is reported above.

## Standards review

One P2 finding: the new joint-test module loader did not restore the entry
point's working-directory change. Corrected in `c917a4e` by importing inside a
scoped monkeypatch context; a real import/teardown probe and all 7 joint checks
pass. Follow-up review found no residual actionable issue, including in the
B1938 public-fixture correction.

## Spec review

No actionable findings in the remaining implementation against the full spec
and tickets 11–15. The review compared physical-model and stage-builder
constructor arguments against the resume baseline and found them unchanged.
It also checked sampling controls, stage schemas/representations, source-package
identity/bbox variants, failure propagation, plot variants and launch behavior.
No new operator joint effective-parameter mathematics was introduced.

Review totals: Standards 1 finding (P2, fixed); Spec 0 findings. Final validation
and lifecycle closure are recorded in [ticket 15](issues/15-close-shared-support-and-final-verification.md).
