# Pixelized workflow structural refactor

**Status:** ready-for-agent
**State:** resolved

## Problem Statement

Researchers maintaining TinyLensGpu must currently understand solver internals to update some reconstruction plots and pipeline exports. Although dense and operator probability models already expose `evaluate()` and `InversionResult`, several callers independently assemble and solve inversions. A change to the supported inversion path can consequently leave a plotting path behind.

Related example pipelines also maintain repeated sampling, stage persistence, source-template validation, and plotting support. Global configuration and import-time process changes make these functions difficult to reuse and test independently. Differences between pipelines include intentional scientific settings and artifact formats, so consolidating them without preserving those differences would change existing workflows.

Importing the library additionally changes JAX array truthiness for the entire process. This hides invalid optional-array selection patterns and makes behavior depend on whether TinyLensGpu has been imported.

Users need these three structural problems resolved while retaining established numerical behavior, public model calls, saved results, and recognizable example workflows. They also need two explicitly approved corrections: failed reconstructions must not silently become formal scientific outputs, and the legacy operator joint-inversion example must display actual reconstructed image components instead of zero placeholders.

## Solution

Deliver three separately reviewable and reversible changes, in this order:

1. Restore native JAX array truthiness after correcting dependencies and verifying supported execution paths.
2. Provide one optional public entry point for a reconstruction and its diagnostics, then migrate affected plotting and export callers to public interfaces.
3. Consolidate repeated example-pipeline support behind explicit configuration while preserving case-specific model construction, stage flow, sampling settings, and artifact contracts.

The diagnostic interface will report which quantities were computed and how. Dense diagnostics and the existing source-only operator approximation retain their current definitions. For operator joint semi-linear inversion, effective parameter count, residual degrees of freedom, and reduced chi-square remain unavailable in this effort; the interface explains that limitation and still provides the valid reconstruction, component images, and raw chi-square.

## User Stories

1. As a lens-modeling researcher, I want existing public model imports and calls to remain usable, so that structural improvements do not require rewriting my analysis.
2. As a researcher sharing a Python process with other scientific libraries, I want JAX arrays to retain native truthiness, so that importing TinyLensGpu does not change unrelated code.
3. As a library maintainer, I want optional array arguments to use explicit missing-value handling, so that valid PSF arrays do not depend on a global compatibility patch.
4. As a user of supported dependencies, I want real modeling and inference paths verified after the patch is removed, so that compatibility is demonstrated rather than assumed.
5. As a researcher evaluating a posterior parameter vector, I want one inversion to produce both its reconstruction and requested diagnostics, so that displayed quantities describe the same model state.
6. As a sampling user, I want ordinary likelihood evaluation to avoid optional diagnostic work, so that analysis features do not burden the sampling path.
7. As an analysis-script author, I want a single-parameter diagnostic entry point, so that I can inspect a fitted model without managing solver internals.
8. As a caller of `evaluate()`, I want its result fields and failure behavior preserved, so that existing consumers continue to work.
9. As a user of gradients and batched likelihoods, I want existing differentiation and JIT behavior preserved, so that structural refactoring does not remove inference capabilities.
10. As a plotting author, I want the same diagnostic result structure across dense and operator backends, so that plotting logic can consume named scientific quantities.
11. As a researcher checking a reconstruction, I want normalized residuals and chi-square computed with the existing mask and noise conventions, so that comparisons remain meaningful.
12. As a user of adaptive regularization, I want the current regularization strength, adaptive parameter, and scale-map information available for display, so that I can interpret the reconstruction.
13. As a user of position constraints, I want the position penalty, source-plane separation, and associated coordinates available through a public interface, so that plotting does not inspect private attributes.
14. As a user of joint semi-linear inversion, I want the total model and actual lensed-source and lens-light images, so that each plotted component corresponds to the fitted model.
15. As a user who needs only a total reconstruction, I want component-image calculation to be optional, so that unused analysis products are not generated.
16. As a dense-backend user, I want existing effective-parameter diagnostics preserved, so that historical comparisons retain their numerical convention.
17. As a source-only operator user, I want the existing block-preconditioner diagnostic approximation preserved and identified, so that I know which approximation produced the reported value.
18. As an operator joint-inversion user, I want unavailable effective-parameter diagnostics represented explicitly, so that a source-only approximation is not presented as a complete joint-model statistic.
19. As an operator joint-inversion user, I want raw chi-square and valid image products even when effective degrees of freedom are unavailable, so that the reconstruction remains usable.
20. As a researcher reading diagnostics, I want unrequested quantities distinguished from unsupported quantities and failed reconstructions, so that missing values are interpretable.
21. As a pipeline user, I want an explicitly unconverged operator reconstruction to stop formal plotting and source export, so that failed iterates do not enter subsequent scientific work.
22. As a pipeline user, I want nonfinite reconstructions rejected before formal plotting or source export, so that invalid values do not propagate into later stages.
23. As a dense-backend user, I want validity reporting to acknowledge the absence of a convergence-status object, so that finite results are not misrepresented as proof of solver convergence.
24. As a pipeline author, I want configuration passed explicitly to shared support, so that helpers do not depend on another module's globals.
25. As a researcher maintaining multiple targets, I want each case to retain its own numerical settings, so that code consolidation does not silently standardize scientific choices.
26. As a reader of an example, I want its model construction and stage ordering to remain visible, so that I can understand what is fitted and what is transferred between stages.
27. As a pipeline author, I want shared sampling to return `StagePosterior`, so that downstream code uses an established posterior representation.
28. As a sampling user, I want live-point, batch, vectorization, and chunking settings preserved, so that consolidation does not change resource use or the sampling procedure.
29. As a multi-stage inference user, I want posterior-to-parameter transfer and prior metadata preserved, so that stages retain their established inference semantics.
30. As a user with saved stage results, I want legacy and current caches to remain readable, so that previous computation remains useful.
31. As a user of downstream analysis scripts, I want newly written artifacts to retain each pipeline's filenames, formats, and field organization, so that existing consumers continue to work.
32. As a user resuming an interrupted pipeline, I want stage selection, schema checks, and overwrite protections preserved, so that resumption does not replace protected results.
33. As a source-template consumer, I want centralized validation to retain the existing dictionary interface, so that cached packages and callers remain compatible.
34. As a caller relying on source-template normalization, I want validation to update and return the same dictionary, so that existing in-place behavior remains reliable.
35. As a pixelized-source user, I want source-grid size, square bounding-box, template, and optional expected-bounding-box checks preserved, so that shared helpers enforce the established grid contract.
36. As an export consumer, I want raw reconstruction packages distinguished from prepared template packages, so that each stage retains its own required fields.
37. As a developer importing shared example support, I want imports to avoid additional working-directory, environment, or JAX precision changes, so that helpers can be used safely in tests and other workflows.
38. As an existing example user, I want current launch methods to continue working, so that the refactor does not require a new runner.
39. As a researcher comparing diagnostic figures, I want layouts, color choices, masking, and overlays retained, so that the structural change does not obscure visual comparisons.
40. As a maintainer, I want all affected maintained callers migrated, so that old private reconstruction paths do not remain alongside the supported interface.
41. As a test author, I want tests to exercise public results and artifacts, so that internal reorganization does not require rewriting tests for equivalent behavior.
42. As a reviewer, I want small real numerical comparisons and focused compatibility checks, so that correctness can be assessed without full scientific sampling runs.
43. As a maintainer, I want the three changes independently reviewable and reversible, so that failures can be isolated and corrected.
44. As a reader of project history, I want historical experiments retained with their original context, so that this refactor does not rewrite the evidence behind earlier conclusions.

## Implementation Decisions

1. **Scope and delivery.** Cover affected library modules, maintained examples, tests, benchmarks, and relevant usage documentation. Deliver native truthiness first, public diagnostics second, and shared pipeline support third. Each change must migrate its affected callers and pass its relevant checks before its delivery is considered complete.

2. **Compatibility baseline.** Preserve public import paths, existing model calls, parameter ordering, prior definitions and transformations, solver controls, evidence approximations, fixed-input preparation rules, and supported differentiation behavior. Preserve existing supported diagnostic definitions and their edge-case handling. The approved behavior changes are specified explicitly below.

3. **Native array truthiness.** Remove the library's global modifications to JAX array boolean behavior after repairing confirmed dependencies. Optional arrays use explicit missing-value selection. Verify actual supported dependency paths; static search alone does not establish compatibility. Keep required Caskade backend initialization intact. Provide migration guidance without adding a switch that reinstalls the global patch.

4. **Existing numerical interface.** Keep `evaluate()` and the existing `InversionResult` field contract. Its evidence, model image, source pixels, lens-light intensities, source bounding box, and solver diagnostics remain available with their current backend-specific failure conventions. Existing forward-model calls also remain compatible.

5. **Additional diagnostic interface.** Add `evaluate_with_diagnostics` to both pixelized probability models. It accepts the same selected parameter state as the existing public evaluation interface and returns an `InversionResult` together with a named diagnostic result. The keyword options `include_effective_dof` and `include_components` both default to false.

6. **Single-inversion contract.** The paired outputs come from one source reconstruction inversion at one parameter state. Reuse its relevant numerical intermediates rather than performing a second reconstruction for plotting. Diagnostic linear algebra is permitted when requested; solver warm starts already belonging to an inversion are not additional reconstructions. Keep results explicit rather than adding an implicit last-result cache.

7. **Execution scope.** The new diagnostic entry point serves a single parameter vector for analysis and plotting. It may use JAX and GPU computation internally, but supporting JIT transformation, batching, or automatic differentiation of this new entry point is not a requirement. Existing numerical entry points retain their current capabilities and do not perform optional diagnostic work.

8. **Common diagnostic content.** The result reports reconstruction validity and an explanatory reason when invalid; normalized residuals and chi-square using existing conventions; regularization strength, adaptive regularization parameter, and information sufficient to render the existing uniform or adaptive scale map. When position constraints are enabled, it also exposes the penalty, maximum source-plane separation, and associated image-plane and source-plane coordinates.

9. **Optional image components.** When requested, provide the reconstructed lensed-source and lens-light model images corresponding to the returned joint reconstruction. Use fitted lens-light intensities and the existing unit-amplitude lens-light bases. On active image pixels, the components must reproduce the returned total model within the applicable numerical tolerance. A physically absent lens-light contribution may be zero; a fitted nonzero contribution must not be replaced with a placeholder.

10. **Effective-parameter support.** The diagnostic fields are effective parameter count, residual degrees of freedom, reduced chi-square, and a method identifier. For dense source-only and joint semi-linear inversion, preserve the current curvature-based diagnostic convention, including its existing treatment of lens-light regularization. For source-only operator inversion, preserve the current source-block preconditioner approximation. These identifiers describe the implemented convention without claiming a new exact interpretation for constrained solvers.

11. **Operator joint limitation.** For operator joint semi-linear inversion, effective parameter count, residual degrees of freedom, and reduced chi-square are unavailable in this effort. Return `None` for these quantities and explain the unsupported diagnostic when requested. Do not replace them with zero, a source-only value, or a newly introduced joint approximation. Raw chi-square and valid reconstruction products remain available. Unsupported diagnostics do not make an otherwise valid reconstruction invalid.

12. **Unrequested results.** Optional results that were not requested remain `None` and do not incur their associated calculation. The result distinguishes an unsupported diagnostic from a quantity that was simply not requested. Position diagnostics are optional when no position constraint is configured.

13. **Failure semantics.** Existing likelihood and `evaluate()` failure returns remain unchanged. In the new diagnostic result, explicit operator non-convergence makes the reconstruction invalid even when the existing failure projection has produced finite zero arrays. Nonfinite reconstructed model, source, or lens-light values also make it invalid. Dense results have no fabricated convergence status; finiteness checking does not claim solver convergence.

14. **Approved failure correction.** At the case/pipeline layer, formal plotting and source-package export raise an informative error and stop the current stage for an explicitly unconverged operator reconstruction or nonfinite reconstruction from either backend. Available solver status and failure context should be reported. Already saved stage artifacts are retained. Missing optional diagnostics, including unsupported operator joint effective degrees of freedom, do not trigger this failure policy.

15. **Approved legacy joint-plot correction.** Migrate the legacy operator joint-inversion plot to the public joint result, including the actual source and lens-light component images. Replace its zero component placeholders and outdated joint-support assumptions. Retain layout and display style, show raw chi-square where appropriate, and represent the unsupported joint statistics according to the limitation above.

16. **Shared example support.** Locate reusable pipeline orchestration support, sampling execution, stage persistence, source-package helpers, and plotting helpers in the example-support namespace. Keep numerical inversion and diagnostics in the core library and reuse existing `StagePosterior` capabilities. Each example continues to express its model assembly, stage sequence, and stage-to-stage data transfer explicitly.

17. **Configuration and sampling.** Pass explicit configuration objects from example entry points to shared support. Preserve per-case pixel scales, grid sizes, masks, solver settings, sampling controls, and input/output locations. Shared sampling returns `StagePosterior`; migrate private tuple-returning callers accordingly. In particular, retain differences in batch size and vectorized chunking rather than replacing them with a shared default.

18. **Stage artifacts.** Preserve each pipeline's current saved filenames, serialization format, field organization, parameter ordering, numerical content, and relevant array representations. Readers accept the existing legacy payloads without a stage schema and the existing schema-bearing payloads. Writers retain the format expected for their particular pipeline even though the in-memory sampler result is unified. Preserve resume behavior, schema validation, and case-specific overwrite guards.

19. **Source-package contracts.** Retain dictionaries and centralize their existing validation and normalization. Validation returns the identical dictionary object and preserves in-place scale-map generation or coercion. Pass expected grid size, optional expected bounding box, and other existing case-specific validation inputs explicitly. Retain square-grid and bounding-box checks and current rejection of obsolete rectangular-grid keys. Distinguish a raw reconstruction export from a prepared source-template package that additionally requires regularization metadata.

20. **Import and launch behavior.** New shared support introduces no working-directory changes, environment-policy changes, or JAX precision changes during import. Runtime policy belongs to entry points, and current example launch methods remain supported. Import checks account for the library's existing Caskade backend initialization and verify that shared support adds no further process mutations.

21. **Plot migration.** Shared plotting consumes public reconstruction and diagnostic outputs. Preserve case-specific layouts, colors, mask presentation, zoom, and overlays. Extract genuinely shared drawing behavior while retaining meaningful plot variants. Complete the migration across affected dense and operator examples, including cases whose directory names do not identify their actual backend.

22. **Architectural decisions.** Respect ADR 0001, “Use block-Schur preconditioning for joint operator inversion”: preserve source–lens-light cross curvature, the supported joint preconditioner, and the distinction between numerical preconditioner damping and the physical prior. Respect ADR 0002, “Require unit-amplitude lens-light bases in joint inversion”: retain that validation and the positive lens-light regularization contract. Preserve the dedicated robust mixture prior semantics recorded in ADR 0003.

## Testing Decisions

1. **Test seams already confirmed.** The design interview approved the public numerical interfaces, public example-support behavior and resulting artifacts, and normal import behavior as the test surfaces. Prefer these highest useful seams over separate tests for every extracted helper. The only additional numerical seam is the requested diagnostic entry point; no testing-only public interface or generic workflow framework is required.

2. **Observable behavior.** Good tests assert returned numerical products, validity and availability semantics, persisted artifact contracts, resume behavior, and process effects. They should survive equivalent internal refactoring. Use a limited invocation probe where necessary to verify the explicitly required single-reconstruction contract, without prescribing private call order or intermediate representation.

3. **Numerical equivalence.** Use small real dense and operator problems covering source-only and joint semi-linear inversion, relevant supported solvers, fixed and inferred source grids, uniform and adaptive regularization, masks, PSF convolution, and position constraints. Compare the new result with the existing public result at the same parameter vector. For retained diagnostic methods, use an independent expression of the established calculation or a documented pre-migration reference. Justify tolerances for dtype, backend, and solver accuracy.

4. **Diagnostic availability and components.** Exercise requested and unrequested components and effective-parameter diagnostics. Verify component reconstruction against the returned total model on active pixels. Verify retained dense and source-only operator diagnostic values and method descriptions. Operator joint requests must return unavailable statistics with a reason while still permitting valid raw chi-square, images, plotting, and export. Verify that ordinary likelihood calls do not execute optional diagnostic work.

5. **Failure cases.** Exercise an actually unconverged operator solve and a nonfinite reconstruction through the public diagnostic interface and case-level consumers. Verify existing core penalties and reconstruction conventions separately from the approved case-level exception policy. Verify that formal plot/export operations stop before producing a success artifact or passing an invalid source to a later stage. Keep dense “no convergence status” distinct from operator convergence reporting.

6. **Pipeline and artifact compatibility.** Test shared sampling with a controlled sampler adapter where deterministic control of its output is needed, asserting the preserved configuration and resulting `StagePosterior`. Cover legacy and schema-bearing stage payloads, write/read compatibility, resume schema checks, and protected-stage overwrite behavior. Preserve raw source-export keys, prepared-template validation, expected-bounding-box variants, dictionary identity, and in-place scale-map behavior. Use small inputs and temporary output locations.

7. **Import and runtime compatibility.** Use isolated processes to verify native non-scalar and scalar JAX truthiness around library import and to verify that shared support adds no working-directory, environment, or JAX precision changes. Cover the repaired optional-array call sites and actual supported modeling/inference dependency paths. Search findings are useful for migration inventory but do not replace runtime checks.

8. **Prior art and placement.** Extend suitable existing core tests for dense/operator inversion, public evaluation, solver and operator autodifferentiation, and stage posterior transfer. Extend suitable existing example tests for source-package exports, adaptive source-package validation, joint-fit plotting, the B1938 pipeline, and example compatibility. Newly extracted shared support should be directly importable in its tests instead of requiring AST extraction of script functions. Keep core behavior tests, example assembly tests, and timing checks in their established separate suites.

9. **Verification sequence and evidence.** Run focused checks for each independently delivered change. After the full migration, run the core and example regression suites once each in separate processes using the documented project environment. Retain the existing gradient and batching regressions for the unchanged numerical interfaces. Run affected existing benchmark checks where relevant; do not invent a new timing threshold for this refactor. Record commands, results, and any limitations with the implementation work. Full-data, long-running scientific sampling is not an acceptance gate.

## Out of Scope

- New effective-parameter or residual-degree-of-freedom mathematics for operator joint semi-linear inversion, including stochastic trace estimation, coordinate-wise trace evaluation, or a new joint-preconditioner approximation.
- Redefining retained dense or source-only operator diagnostics, adding an active-set interpretation for constrained solutions, or standardizing their mathematical conventions as part of structural cleanup.
- Changes to physical models, priors, sampler settings, solver tolerances, regularization semantics, evidence approximations, or existing gradient support.
- JIT, vectorization, or automatic-differentiation guarantees for the new diagnostic entry point.
- A generic workflow engine, a new public pipeline framework, or replacement of explicit case-specific stage assembly.
- A new stage-cache format, a cache migration campaign, or replacement of source-package dictionaries with a new serialized object model.
- The previously discussed P2/P3 projects: broad dense/operator consolidation, redesign of observation-data containers or prepared-state lifecycles, relocation of shared parameter definitions, and general directory cleanup.
- Bulk rewriting of historical research records, figure redesign, and mandatory full scientific sampling campaigns.

## Further Notes

- This specification is the synthesis of the completed design interview. The testing seams and acceptance approach were explicitly agreed; publishing it does not require another interview.
- The user's final Q22 decision selects unavailable operator joint effective-parameter diagnostics for this effort. A subsequently discussed alternative that would add a new numerical estimator was not selected and is outside this specification.
- Unavailability is a scope and implementation statement, not a mathematical claim that matrix-free operators cannot compute effective parameter counts.
- `ready-for-agent` records specification readiness. Publication itself did not assert implementation or verification. Implementation is now resolved; see the completion record below.
- Historical experiments and validation reports remain evidence for their stated models and configurations. They do not override this specification or establish validation of the new interfaces.

## Completion record

Completed on 2026-10-27 on `integration/pixelized-workflow-refactor`. All 15
implementation tickets are resolved. The resumed work completed tickets 11–15
after the existing tickets 01–10, preserving the separately reviewable native
truthiness, public diagnostics and shared-support deliveries.

Final separate-process verification: **813 core**, **256 example/pipeline**, and
**9 benchmark** tests passed. The Standards review's one P2 test-isolation issue
was fixed and rechecked; the Spec review found no actionable gaps. See the
[verification record](verification.md), [caller inventory](README.md#shared-support-inventory-tickets-1115),
and [ticket 15](issues/15-close-shared-support-and-final-verification.md) for
commands, configuration, evidence and the small-data/controlled-sampling scope.
