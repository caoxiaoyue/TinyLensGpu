# Local work records

This directory holds specs, tickets, research, and reproduction material. Follow [the tracker conventions](../docs/agents/issue-tracker.md) when publishing, claiming, resolving, or retaining work. This index locates records; each source file owns its requirements, state, and supersession notes.

## Specs and implementation records

| Record | Entry point | Reading note |
| --- | --- | --- |
| Adaptive source regularization | [Spec](adaptive-regularization/spec.md) | Requirements record; read before choosing a backend or template workflow. |
| Dense fixed-grid cache | [Spec](dense-pixelized-fixed-grid-cache/spec.md); [tickets](dense-pixelized-fixed-grid-cache/issues/) | The spec has partial supersession notes. Existing tickets have checked acceptance lists and `ready-for-agent` labels but no lifecycle state; classify them from evidence before claiming or marking complete. |
| Operator constrained solvers | [Historical spec](operator-fista-nnls/spec.md) | FISTA requirements are explicitly superseded by the solver-consolidation note at the top. |
| Square pixelized source grids | [Spec](square-pixelized-source-grid/spec.md) | Requirements record; completion is not asserted by this index. |
| Stage posterior transfer | [Spec](stage-posterior-transfer/spec.md) | Requirements record; completion is not asserted by this index. |
| Pixelized workflow structural refactor | [Overview](pixelized-workflow-refactor/README.md); [spec](pixelized-workflow-refactor/spec.md) | Three agreed P1 changes; the spec defines compatibility, diagnostic availability, test seams, and delivery order. |

## Research and reproduction records

| Record | Entry points |
| --- | --- |
| Solver comparison | [Script](fista-mge-validation/compare_solvers.py); [validation report](../docs/validation/pix-src-lens-light-mge-fista-comparison.md) |
| PNPG fixed-grid gradients | [Script](pnpg-mge-gradient-validation/validate_gradients.py); [retained results](pnpg-mge-gradient-validation/results.json); [validation report](../docs/validation/pnpg-mge-fixed-bbox-gradients.md) |
| Pseudo-Jaffe model comparison | [Primary-source notes](pseudo-jaffe-comparison/vegetti-comparison.md) |
| Subhalo prior sensitivity | [Overview](subhalo-prior-audit/README.md); [study](subhalo-prior-audit/sensitivity-study.md); [regularization literature](subhalo-prior-audit/vegetti-source-regularization.md) |

Existing script and data paths are retained for reproduction. Generated `output/` directories follow `.gitignore` and are not necessarily present in a fresh checkout. Read the linked report for the model, configuration, and command associated with a result.

## Adding material

Keep a feature's spec and one-file-per-ticket issues at the tracker paths. Temporary scripts, experiments, reproduction material, and results may also live alongside the work. Use a short feature README or a linked existing note as its entry point. Follow [the tracker conventions](../docs/agents/issue-tracker.md#research-experiments-and-retained-history) for optional subdirectories, retained evidence, cleanup, and promotion into maintained tests or tools. Add a link here when introducing a feature or changing its main entry point.
