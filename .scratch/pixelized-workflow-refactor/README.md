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

## Maintained diagnostics inventory (ticket 08)

All maintained formal plot/source-export callers now use the public
`evaluate_with_diagnostics` endpoint; the settled search
`rg '_solve_source|_regularization_matrix|_regularization_data|_get_bbox|_position_likelihood_penalty_jax|_get_reg_scale|reconstruct_component_images' examples/pix_src_demo examples/pix_src_demo_operator -g '*.py' -g '!bench*' -g '!profile*' -g '!dump_jaxpr.py' -g '!sim_data.py'`
returns no matches. Inventory distinctions:

- `pipe/no_lens_light/model_matrix.py` lives in the operator tree but is the
  dense backend; it was migrated by ticket 04 (dense), not the operator tickets.
- Profiling/timing and data-generation scripts (`bench_inversion.py`,
  `profile_inversion.py`, `dump_jaxpr.py`, `sim_data.py`) legitimately inspect
  or feed internals and are not migration targets. The `demo_inversion.py`
  scripts produce formal plots and were migrated.
- Dead `_source_param_value` helpers removed from all pipe models except
  `galan24_test/model_predefine_src_box.py`, which still uses it for the
  stage-A source Gaussian sigma.
- Transitional `evaluate`-vs-diagnostics compatibility was removed in ticket
  08; the export probe now asserts callers use `evaluate_with_diagnostics`.


## Implementation tickets

The approved breakdown is published as [individual tickets](issues/), each with
its own acceptance criteria, blockers, triage status, and lifecycle state.
Start with [01: Restore native JAX array truthiness](issues/01-restore-native-array-truthiness.md),
then select open tickets whose blockers are resolved. The parent specification
remains the authority for scope and scientific behavior.
