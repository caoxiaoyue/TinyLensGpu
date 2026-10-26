# Pixelized workflow structural refactor

[The specification](spec.md) is the authoritative requirements and readiness
record for the three agreed P1 changes: native array truthiness, public inversion
diagnostics, and shared example-pipeline support.

The specification defines the test seams, compatibility requirements, approved
behavior corrections, and delivery order. Use the environment and suite rules
in [AGENTS.md](../../AGENTS.md) and the
[project guide](../../docs/guides/guide.md) when implementation begins.

Validation will use existing small core fixtures and temporary example artifacts;
full scientific datasets and long sampling runs are not required acceptance
inputs. Record verification commands and output evidence with the corresponding
implementation work under the [local tracker conventions](../../docs/agents/issue-tracker.md).
No implementation or validation outputs have been produced for this record yet.

## Implementation tickets

The approved breakdown is published as [individual tickets](issues/), each with
its own acceptance criteria, blockers, triage status, and lifecycle state.
Start with [01: Restore native JAX array truthiness](issues/01-restore-native-array-truthiness.md),
then select open tickets whose blockers are resolved. The parent specification
remains the authority for scope and scientific behavior.
