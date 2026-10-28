# Potential correction

This feature adds single-band source–potential reconstruction to TinyLensGpu,
with separate single-step and iterative algorithms and dense/operator backends.
The approved requirements and scope are in the [spec](spec.md).

- Spec triage: ready-for-agent.
- Implementation is in progress on `codex/potential-correction-integration`.
  Consult individual tickets for their current lifecycle and verification.
- The design uses the existing square source grid, a local potential support
  constructed around actual interpolation corners, complete fourth-difference
  windows, and a fixed independent weak amplitude prior.
- Planned validation uses Pseudo-Jaffe and no-subhalo mocks, fixed true macro
  parameters, and truth-derived source initialization. Numerical comparisons
  precede scientific examples and performance measurements.

## Implementation tickets

### Implementation progress

Tickets 01 and 02 are resolved and merged. Source preparation passed 62
regularization, 23 precision/cache, and 372 source/operator/autodiff checks;
see [the verification record](research/source-preparation-verification.md).
The mock generator passed 12 example and 6 existing profile checks and ran its
default 200-by-200 float64 configuration; see ticket 02 and the
[mock documentation](../../examples/potential_correction/README.md).
The integration checkout also passed all 268 current example tests in a
separate process. The original core baseline passed 810 tests (3 slow tests
deselected) before shared-code changes. These are stage checks, not final
whole-spec verification.

Supplied-field rendering and dense fixed-strength single-step reconstruction
are implemented. The combined focused check passed 31 cases (14 dense plus
17 rendering); see the [renderer record](research/supplied-field-rendering.md)
and [dense fit API and verification record](research/dense-single-step.md).
The dense result provides signed coupled fields, complete conditional Gaussian
evidence, distinct linear/nonlinear images and explicit failure metadata.
Strength optimization, operator calibration, iterative fits and scientific
reconstruction/performance validation remain outstanding downstream stages.

The user approved the following 14 tickets and their direct blocking edges.
All tickets were published as ready-for-agent / open; none was claimed or
implemented during publication. Each ticket owns its current lifecycle state
and acceptance criteria. The parent spec is unchanged.

| Ticket | What it delivers | Blocked by |
| --- | --- | --- |
| [01: Source preparation](issues/01-prepare-source-regularization-and-precision.md) | Lazy source regularization and explicit precision with existing behavior preserved | None |
| [02: Mock and truth](issues/02-generate-pseudo-jaffe-mocks-and-truth.md) | Reproducible Pseudo-Jaffe and control observations with truth products | None |
| [03: Supplied-field rendering](issues/03-render-supplied-potential-corrections.md) | Local support, physical derivatives, and corrected images | 01 |
| [04: Dense single-step fit](issues/04-dense-single-step-reconstruction.md) | Fixed-strength coupled reconstruction and complete conditional evidence | 03 |
| [05: Dense single-step tuning](issues/05-optimize-dense-single-step-strengths.md) | Bounded DIRECT strength selection and candidate failure handling | 04 |
| [06: Operator single-step fit](issues/06-operator-single-step-reconstruction.md) | Coupled PCG reconstruction and block evidence | 04 |
| [07: Single-step calibration](issues/07-calibrate-single-step-backends.md) | Operator tuning, backend comparison, and weak-prior sensitivity | 02, 05, 06 |
| [08: Dense iterative fit](issues/08-dense-iterative-reconstruction.md) | Fixed-strength LM reconstruction, history, and local evidence | 07 |
| [09: Operator iterative fit](issues/09-operator-iterative-reconstruction.md) | Matrix-free updates with shared iteration and failure semantics | 08 |
| [10: Iterative tuning](issues/10-optimize-iterative-strengths.md) | Outer DIRECT around complete inner reconstructions in both backends | 09 |
| [11: Single-step scientific outputs](issues/11-single-step-scientific-example-and-exports.md) | Scientific examples, two renderings, diagnostics, and exports | 07 |
| [12: Iterative scientific outputs](issues/12-iterative-scientific-example-and-exports.md) | Scientific examples, histories, exports, and algorithm comparisons | 10, 11 |
| [13: Precision and performance](issues/13-validate-precision-and-gpu-performance.md) | Float32 validation and GPU timing/memory/scaling evidence | 12 |
| [14: Final integration](issues/14-integrate-document-and-verify.md) | Complete usage documentation and whole-spec verification | 13 |

At publication, the immediately available frontier is **01 and 02**.
Once 04 is resolved, 05 and 06 can proceed independently. Once 07 is resolved,
08 and 11 can proceed independently. The 08 dependency on 07 is the approved
single-step-before-iteration validation gate. Indirect dependencies are not
repeated in ticket blocking fields.

Each slice includes its own verification. Ticket 14 performs final integration;
it does not postpone earlier numerical or example testing. Recompute the
frontier from ticket states before claiming work.

No feature runner or generated output was added during ticket publication.
Implementation records should add runnable commands, required inputs, exact
configurations, output locations, and verification results when those exist.
Follow the [local tracker conventions](../../docs/agents/issue-tracker.md)
when claiming, implementing, or resolving a ticket.
