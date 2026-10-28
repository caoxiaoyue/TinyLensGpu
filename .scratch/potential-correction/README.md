# Potential correction

This feature adds single-band source–potential reconstruction to TinyLensGpu,
with separate single-step and iterative algorithms and dense/operator backends.
The approved requirements and scope are in the [spec](spec.md).

- Spec triage: ready-for-agent.
- Implementation is in progress on `codex/potential-correction-integration`.
  Consult individual tickets for their current lifecycle and verification.
- The design uses the existing square source grid, a local potential support
  constructed around actual interpolation corners, complete fourth-difference
  windows, and a fixed independent amplitude prior.
- Planned validation uses Pseudo-Jaffe and no-subhalo mocks, fixed true macro
  parameters, and truth-derived source initialization. Numerical comparisons
  precede scientific examples and performance measurements.

## Implementation tickets

### Implementation progress

Tickets 01–13 are resolved. Tickets 01–12 are merged; precision/performance
validation (13) is ready for integration. Final integration (14) remains
outstanding.

The [small-case calibration](../../docs/validation/potential-correction-single-step-calibration.md)
records six matched subhalo/control runs and adopts amplitude precision 10000
for its explicitly configured physical model. It supports localized recovery,
with peak suppression, low-mode sensitivity, boundary optima and approximation
limits retained. Those results do not automatically validate different image,
source, potential or physical configurations.

Focused commands, configurations and outcomes are retained in:

- [Source preparation](research/source-preparation-verification.md).
- [Mock generation](../../examples/potential_correction/README.md) and ticket 02.
- [Supplied-field rendering](research/supplied-field-rendering.md), including
  the independent 200-by-200 supplied-truth discretization check.
- [Dense single-step reconstruction](research/dense-single-step.md).
- [Dense strength optimization](research/dense-strength-optimization.md).
- [Operator single-step reconstruction](research/operator-single-step.md),
  including allocation evidence and the reproduced baseline float32
  finite-difference variability.
- [Single-step calibration summary](research/single-step-calibration/summary.json)
  and the linked complete inputs, selected fields and candidate records.
- [Scientific single-step validation](../../docs/validation/potential-correction-single-step-scientific.md)
  for the explicitly configured 200-by-200 runs, their saved public results,
  figures, matched controls and amplitude-prior probes.
- [Dense nonlinear iteration](../../docs/validation/potential-correction-dense-iteration.md)
  for current bilinear derivatives, accumulated-field priors, objective-based
  LM acceptance, failure histories and undamped local-Gauss–Newton evidence.
- [Operator nonlinear iteration](../../docs/validation/potential-correction-operator-iteration.md)
  for current coupled actions, the shared control policy, fixed partition,
  damped-step/undamped-evidence separation, dense/float32 comparisons,
  failure propagation, JIT reuse and bounded observable allocation.
- [Nested nonlinear strength selection](../../docs/validation/potential-correction-iterative-inference.md)
  and its [retained scan/search record](research/iterative-strength-search.json)
  for independent nonlinear references, actual DIRECT selections, cold-start
  reproducibility, bounded partition calibration, rejected layouts, recovery
  effects, budget/boundary status and explicit scalar candidate diagnostics.
- [Precision and measured GPU workflows](../../docs/validation/potential-correction-precision-performance.md)
  and [compact records](research/precision-performance/) distinguish the named
  supported small float32 configuration from failed stiff scientific float32
  fits, preserve the original numerical guards, and record isolated
  compilation-inclusive/warm timing, memory and actual resolution scaling.

The original core baseline passed 810 tests (3 slow tests deselected) before
shared-code changes. An early integration checkout passed all 268 example
tests separately. Subsequent focused checks are recorded per ticket; these
earlier broad runs are not final whole-spec verification.

Supplied-field rendering and dense fixed-strength single-step reconstruction
are implemented. The combined focused check passed 31 cases (14 dense plus
17 rendering); see the [renderer record](research/supplied-field-rendering.md)
and [dense fit API and verification record](research/dense-single-step.md).
The dense result provides signed coupled fields, complete conditional Gaussian
evidence, distinct linear/nonlinear images and explicit failure metadata.
Dense strength optimization now uses bounded DIRECT with a reusable
log-strength callable, complete candidate fits, explicit failure/boundary
metadata, and a small runnable tuning example. Its focused/shared core check
passed 181 cases and its separate example check passed 13; see the
[API and independent-scan record](research/dense-strength-optimization.md).
The fixed-strength operator backend now solves the full coupled system with
PCG, bounded principal-block preconditioning and explicitly approximate H/Q
determinants. Its post-merge core check passed 31 operator/inference cases,
the separate example check passed 14, and the allocation benchmark passed;
see the [operator API and verification record](research/operator-single-step.md).
That record also preserves an existing float32 source–lens-light Jacobian
fixture limitation and its unchanged-baseline repeat evidence.
Single-step backend calibration is now complete for the recorded 64x64
Pseudo-Jaffe/control configuration, with a 49x49 source and 0.075 arcsec
potential spacing. Six independent amplitude-prior sensitivity runs, bounded
scans and both-backend DIRECT selections retain raw fields, failures,
optimizer budgets/boundaries, and physical/discretization diagnostics. The
adopted amplitude precision is 10000 for this configuration. Local recovery
is useful while raw low modes, global errors and suppressed peaks remain
explicit limitations; block evidence is approximate. See the
[validation report](../../docs/validation/potential-correction-single-step-calibration.md)
and [compact retained products](research/single-step-calibration/summary.json).
Cached bounded inverse lower factors accelerate the same principal
preconditioner, with focused numerical, legacy Schur/autodiff and allocation
regressions passing. Both iterative backends and bounded nested strength
selection are now implemented with focused numerical checks. Scientific
iterative outputs are validated below; workflow precision and performance
remain downstream stages.

Ticket 11 adds the documented 200x200 fixed-strength single-step experiment
with both backends and matched control. The explicit b=0.04 scenario retains
the calibration field of view at 0.012 arcsec/pixel, with a resampled physical
PSF. Actual counts, numerical products, source/potential/deflection/convergence
plots, two renderings, raw/optional affine diagnostics, and independent
amplitude sensitivity are retained. See the [scientific validation report](../../docs/validation/potential-correction-single-step-scientific.md)
and [compact summary](research/single-step-scientific/summary.json).
Peak localization is useful, while peak suppression, global low modes, prior
dependence and linearization mismatch remain explicit limitations. The
shared result exporter/loader does not reconstruct a numerical solver.

Ticket 12 adds both-backend nonlinear scientific reconstruction with fixed
strengths and complete nested-search assembly, truth-node/zero-potential resets,
current nodal derivatives, saved public results and actual objective/damping
histories. All four matched 200x200 adopted-model fits genuinely converge;
dense/operator fields agree numerically while their local/block evidence
offsets remain explicit. A narrow residual correction preserves the total PCG
budget and true physical certificate, with failed budgets/endpoints retained.
The [iterative scientific report](../../docs/validation/potential-correction-iterative-scientific.md)
and [compact records](research/iterative-scientific/) preserve stopping-control
calibration, exact single-step partition reuse, algorithm comparisons, raw
versus optional affine diagnostics, and the measured amplitude-prior
sensitivity. Localized kappa is useful with substantial peak suppression;
psi/alpha low modes remain strongly prior-sensitive, and lower-prior numerical
convergence is unverified. No globally weak prior or broad robustness is
established. Failed probes keep null evidence and diagnostic status. Production
scientific results use fixed smoothing; small tests verify the complete nested
workflow without claiming a converged 200-pixel outer search.

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
