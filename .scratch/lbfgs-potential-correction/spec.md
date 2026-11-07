# L-BFGS-B source–potential reconstruction

**Status:** ready-for-agent
**State:** open

## Problem Statement

TinyLensGpu users can reconstruct a pixelized source and a lens-potential perturbation with the single-step and Gauss–Newton/Levenberg–Marquardt (GN/LM) interfaces. The J0946 experiments demonstrate another route that jointly optimizes the source and accumulated potential with L-BFGS-B and a continuously differentiable cubic Hermite source. Researchers currently access that route through experiment scripts that combine the fit, archived-input loading, strength search, and reporting.

A lens modeler needs a reusable fit for a supplied set of prior strengths, a complete numerical reference, and a matrix-free route for larger grids. Both routes need the same physical target, a consistent convergence decision, and results that existing potential-correction tools can read. The migration needs to reproduce the archived J0946 result at fixed strengths and test the new operator route on the same observation.

## Solution

Add `LBFGSPotentialCorrection` as a peer of the existing source–potential fit interfaces. Use the library's Float64 cubic Hermite renderer, fixed macro lens mass model, and existing physical priors to reconstruct a signed pixelized source and a signed lens-potential perturbation from a lens-light-subtracted image.

The caller supplies all three regularization strengths for each fit. L-BFGS-B optimizes the source and potential at those strengths and returns a shared reconstruction result with qualified local GN evidence. The default dense backend follows the archived data-space calculation. The operator backend uses the same nonlinear objective with matrix-free coupled curvature and bounded principal blocks for whitening and determinant approximation.

Validate both backends on independent small references, a matched injection and control pair, and a fresh fixed-strength J0946 refit. Expose the fitted fields, diagnostics, and evidence approximation through the public result and the common mock export and readback workflow.

## User Stories

1. As a lens modeler, I want a maintained L-BFGS-B fit interface, so that I can reconstruct a pixelized source and lens-potential perturbation from my observation.
2. As an existing potential-correction caller, I want the established single-step and GN/LM interfaces to remain compatible, so that I can add the new fitter to my workflow incrementally.
3. As a researcher with an existing macro fit, I want to supply my fixed macro lens model and lens-light-subtracted image, so that I can study a local correction conditional on that model.
4. As a researcher comparing methods, I want the fit to use my selected mask, PSF, integration, source geometry, and potential support, so that the comparison uses the same image model.
5. As a modeler using moving rays, I want the built-in Float64 cubic Hermite source, so that the objective and its spatial derivatives follow the same continuously differentiable interpolation.
6. As a researcher who fits residual images, I want signed source and potential coefficients, so that I can inspect the reconstructed fields directly.
7. As a modeler continuing from a macro source, I want to supply my initial source nodes and start at zero potential correction, so that the fit has an explicit starting point.
8. As a researcher who uses adaptive regularization, I want to pass a fixed source-node weight map, so that I can preserve the prior shape used in my macro fit.
9. As a modeler testing the prior, I want to set the source, fourth-difference, and amplitude strengths on every call, so that I can compare distinct strength triples.
10. As a modeler interpreting the fit, I want all three strengths fixed during each L-BFGS-B run, so that the result is conditional on the strengths I supplied.
11. As a researcher evaluating several triples, I want to reuse the prepared fit object, so that fixed observation and reference preparation can be shared.
12. As a researcher checking candidate repeatability, I want each fit to restart from the same source and zero potential, so that earlier candidates do not determine later starting points.
13. As a modeler selecting the source prior, I want the existing source regularization choices, so that I can compare the same physical penalties across fit methods.
14. As a modeler setting the potential prior, I want independent smoothness and all-node amplitude penalties, so that their effects on the correction can be distinguished.
15. As a modeler starting a new run, I want whitening to use my Hermite starting source by default, so that I can prepare a fit with a single source input.
16. As a researcher reproducing J0946, I want to supply a separate bilinear M3 whitening reference, so that I can reproduce the archived optimization setup with the Hermite fit objective.
17. As a researcher comparing algorithms, I want both backends to evaluate the same nonlinear objective and gradient, so that I can compare their solutions at fixed strengths.
18. As a modeler running a reference calculation, I want dense data-space whitening and complete local GN determinants, so that I can compare against the archived result.
19. As a researcher fitting a larger grid, I want a matrix-free operator backend, so that I can use bounded blocks in place of full response and curvature matrices.
20. As a researcher studying conditioning, I want to choose or explicitly supply the operator partition, so that I can reproduce the whitening and determinant approximation.
21. As a modeler qualifying an operator fit, I want PCG controls and a recomputed coupled residual, so that I can check the endpoint stationarity calculation.
22. As a researcher comparing fit status, I want one GN stationarity criterion across both backends, so that the same rule qualifies their endpoints.
23. As a modeler choosing a candidate, I want evidence normalized with the supplied prior strengths, so that the score corresponds to the fitted model.
24. As a researcher comparing scores, I want separate labels for local GN evidence and the determinant method, so that I can interpret scores from complete and block calculations.
25. As a researcher diagnosing a failed run, I want the native optimizer exit, evaluation counts, and endpoint checks, so that I can identify the reason a result failed qualification.
26. As a caller comparing several fits, I want numerical failures to return a failed result with available physical fields, so that I can diagnose them and exclude their evidence.
27. As a modeler inspecting a correction, I want source values, potential values, deflection, and convergence on their established domains, so that I can compare them to other fits.
28. As a researcher checking image residuals, I want the final nonlinear image and residuals, so that I can assess the reconstruction on the observed pixels.
29. As a user of the shared result tools, I want to export and reload new L-BFGS-B products, so that I can inspect them through the existing mock workflow.
30. As a user with saved single-step or GN/LM products, I want to read those products alongside new results, so that I can compare methods across releases.
31. As a developer checking the numerical implementation, I want small independent matrix references, so that I can verify the gradient, whitening, stationarity, and determinants.
32. As a developer checking strength reuse, I want to fit A, B, and A on one object, so that I can catch a cached amplitude strength or a changed initialization.
33. As a researcher checking perturbation recovery, I want both backends to fit the same injected subhalo mock, so that I can compare the recovered perturbation to the known injection.
34. As a researcher checking artifacts, I want both backends to fit the matched no-subhalo control, so that I can inspect structures produced in the absence of the injection.
35. As a researcher migrating the J0946 analysis, I want a fresh dense fixed-strength refit against trial 497, so that I can verify parity with the archive.
36. As a researcher evaluating the operator backend, I want a J0946 refit compared on the full valid convergence field, so that I can assess agreement beyond the peak alone.
37. As a researcher repeating the real-data comparison, I want to supply the local archive and preserve its inputs and configuration in the report, so that I can reproduce the same run.
38. As a developer running routine regression tests, I want regenerable small inputs, so that the suite can run without the local J0946 archive.
39. As a researcher planning GPU runs, I want stage timings and peak device-memory measurements for both backends, so that I can assess their costs at J0946 scale.
40. As a researcher performing my own strength comparison, I want a qualified fit and its evidence for a supplied triple, so that I can consume the result from an outer workflow.

## Implementation Decisions

1. **Public fit.** Export the new peer from `ObservationModel`. Prepare it with a `PotentialCorrectionRenderer`, image and noise arrays, a finite signed `initial_source`, and the existing source regularization type with default `second-order`. Accept an optional fixed `source_scale` of finite positive values, one per fitted source node, plus initialization provenance and the optional whitening reference. Use the existing observation and geometry validation. The first release supports Float64 and explicitly selected `cubic_hermite`.
2. **Per-call strengths.** `fit` takes `PotentialCorrectionStrengths` in the existing source, potential, amplitude order. All values are finite, positive physical strengths. Pass them explicitly to the objective, candidate-dependent metric, priors, evidence, and result assembly, including across compiled calls. Cache preparation that depends on fixed inputs and rebuild factors that depend on a changed strength. Each call starts from the supplied source and zero potential.
3. **Physical objective.** Use the renderer's nonlinear image at the current accumulated source and potential with the fixed macro, grids, noise, mask, PSF, and subpixel integration. The minimized objective is

   \[
   F(s,\delta\psi)=\tfrac12\left[\chi^2+\lambda_s s^\mathsf T R_s s+
   \lambda_4\lVert D_4\delta\psi\rVert^2+\lambda_0\lVert\delta\psi\rVert^2\right].
   \]

   Here \(R_s\) is the existing source precision shape, including the optional adaptive scale, and \(D_4\) uses the existing complete horizontal and vertical fourth-difference windows. The amplitude term applies to every fitted potential coefficient. Share the objective and its JAX gradient across backends. Use SciPy L-BFGS-B on the whitened coordinates to fit the signed fields.
4. **Whitening reference.** A public `whitening_reference` supplies reference source values on the same fitted grid and their interpolation mode. Use zero potential and the same fixed geometry and macro rays for this reference. Default to the Hermite initial source. Keep the reference metric fixed within each fit. The actual fit and endpoint always use the Hermite forward model, including when the supplied reference uses hard-box bilinear interpolation.
5. **Backend selection.** `fit` selects `dense` or `operator` and defaults to `dense`. Expose `max_iter` with default 3000 and `gn_stationarity_tol` with default \(10^{-3}\). Follow the established operator controls `block_size`, `block_indices`, `pcg_max_iter`, `pcg_rtol`, and `pcg_atol` at fit time, with `solve_rtol` and `solve_atol` at construction for the endpoint linear-residual check.
6. **Dense backend.** Port the prototype's scaled prior factors and data-space whitening. Evaluate the full coupled, undamped GN determinant ratio in data space at the final fields. Preserve the prototype's mathematical path as the reference for the fixed-strength J0946 refit.
7. **Operator backend.** Apply the complete coupled \(H_{\mathrm{GN}}=J^\mathsf T W J+R\) through current Jacobian and transpose actions, noise precision \(W\), and prior actions. Assemble bounded principal blocks for the reference whitening and for terminal preconditioning and determinants. The default partition uses separate contiguous source and potential blocks with a cap of 64. Explicit joint partitions obey the existing coverage and size rules. Record the selected partition. The operator fit uses only operator actions and bounded blocks for whitening, endpoint certification, and evidence.
8. **Endpoint qualification.** Both backends require native optimizer success, finite fitted coefficients and endpoint quantities, and the same undamped GN stationarity statistic \(\tfrac12 g^\mathsf T H_{\mathrm{GN}}^{-1}g\) at or below the configured tolerance. Here \(g\) is the physical objective gradient at the returned fields. The operator backend obtains the inverse action by coupled PCG and checks the recomputed residual against the absolute and relative solve tolerances. Use the existing convention of raising for invalid configuration and returning a failed result for numerical failure, with available fields, a reason, and `log_evidence=None`.
9. **Evidence semantics.** Compute qualified local GN evidence from the final nonlinear objective, noise normalization, and posterior-minus-prior log determinant. Use the full data-space determinant ratio for dense and the difference of principal-block log determinants for operator. Label both as `local_gauss_newton` with `evidence_is_exact=False`. Distinguish the full data-space and principal-block determinant methods and record the prior, posterior, and noise log determinant terms.
10. **Result and persistence.** Reuse `PotentialCorrectionResult` for the raw signed fields, valid derivative maps, final nonlinear image, residuals, objective, and evidence. Add `optimizer_info` for native L-BFGS-B status, iterations and evaluations, and the endpoint stationarity diagnostics, including the operator residual certificate when applicable. Use `solver_info=None` for L-BFGS-B results. The common mock exporter and reader support the new diagnostics and existing single-step and GN/LM products.
11. **Delivery order.** Implement the shared fit and dense numerical route first. Establish dense J0946 parity before qualifying the operator route. Reuse the existing rendering, priors, partition and factorization logic, result assembly, and example products where they apply. Keep archive loading and validation in the maintained example workflow. Preserve the established single-step and GN/LM fit, renderer, and inference contracts.

## Testing Decisions

1. **Confirmed seams.** Use public `fit` inputs and `PotentialCorrectionResult` as the primary library test seam. Exercise exported products through the common mock export and readback seam. Run J0946 and resource measurement through the maintained validation entry point. Use focused numerical checks for independently verifiable algebra and avoid testing-only public interfaces.
2. **Small matrix reference.** Use a deterministic Float64 Hermite case with nonzero source–potential coupling. Compare the nonlinear objective and gradient with an independent calculation. Check the full and block whitening against explicitly assembled reference curvature and compare the two stationarity calculations with a complete solve. Check the dense log determinant ratio against a full small calculation and the operator log determinant terms against explicit principal blocks at the same endpoint. Existing Hermite derivative, coupled potential-correction solve, and evidence tests provide the patterns.
3. **Candidate reuse and failure.** Run A → B → A on one prepared object with different amplitude strengths and compare the repeated A results. Verify the objective, prior normalization, evidence, and reported strengths correspond to each call. Exercise qualified and failed fit statuses, the coupled residual certificate for operator, and consistent output of available fields and diagnostics on failure.
4. **Mock and control.** Use the existing Pseudo-Jaffe injection generator and matched no-subhalo control with shared noise and mask. Fit both cases with both backends at fixed strengths. Compare reported recovery on the valid convergence domain and image residuals under the recorded configuration. Verify the new fields and diagnostics survive export and readback, and cover legacy products through the existing example regressions.
5. **J0946 inputs.** Run a fresh fit from trial 497's fixed-strength configuration on each backend. Use the archived M3 macro, observation and noise, mask, PSF, integration and grid geometry, and the fixed adaptive source scale. Use the saved source-only starting field and zero potential as the actual start. Use the distinct saved M3 source and hard-box bilinear interpolation for the whitening reference. Preserve the archived image sampling precision before promotion to Float64. Record the input identity and check matching potential topology and derivative-valid domains before comparing fields.
6. **J0946 gates.** Both runs must qualify with the same \(10^{-3}\) stationarity threshold. The dense fit is compared with the archive. The operator fit is compared with the new dense fit and may use an explicitly fixed, recorded partition.

   | Comparison | Acceptance criterion |
   | --- | --- |
   | Dense vs. archived trial 497 objective | Absolute difference at most \(10^{-3}\) |
   | Dense vs. archived trial 497 log evidence | Absolute difference at most \(0.01\) |
   | Dense vs. archived trial 497 convergence | Maximum absolute \(\delta\kappa\) difference at most \(10^{-4}\) over the full derivative-valid field |
   | Operator vs. dense objective | Absolute difference at most \(0.01\) |
   | Operator vs. dense convergence | Maximum absolute \(\delta\kappa\) difference at most \(0.005\) over the full derivative-valid field |
   | Operator vs. dense positive peak position | Separation at most one potential grid spacing |

   Verify the operator evidence against its explicit block reference on the small problem. Report the two J0946 evidence values, determinant methods, and their difference as measurements.
7. **Matrix-free and performance checks.** Exercise the operator path with a problem large enough to distinguish bounded blocks from full response, prior, joint-curvature, and data-space covariance allocations. Use the existing potential-correction memory benchmark as prior art. Record per-stage J0946 timings and peak device memory for both backends, with device, precision, configuration, and measurement method. Matrix-free structure and successful real-data fits are the release gates. Timing and memory values are reported measurements.
8. **Suite placement.** Put core numerical and compatibility tests in the library suite, mock assembly and persistence in the example suite, and resource measurements in benchmarks. Run the suites in separate processes in the `tinylens_gpu` environment. The maintained J0946 validation accepts local archive paths. Routine tests use regenerable inputs.

## Out of Scope

- Automatic two- or three-strength selection for the new fitter, including porting the archived DIRECT/Powell search.
- Float32 fits, other fitted source interpolants, or nonzero starting potentials in the new interface.
- Refitting the macro model or lens light, fitting a parametric subhalo mass component, or changing the source and potential geometry or prior definitions.
- Changes to the existing joint semi-linear inversion of source and lens light or its block-Schur preconditioner.
- Calibrated subhalo detection significance, mass intervals, and posterior sampling.
- A prescribed runtime, speed ratio, or device-memory cap.
- B1938+666 workflows, modeling, replays, and tests.

## Further Notes

- The [potential-correction spec](../potential-correction/spec.md) and [source interpolation spec](../pixelized-source-interpolation/spec.md) define the established geometry, priors, and Hermite representation this feature uses. The [block-Schur ADR](../../docs/adr/0001-use-block-schur-for-joint-operator-inversion.md) concerns the separate joint semi-linear source and lens-light workflow.
- The [trial 497 report](../../workspace/iteration/cubic_evidence_joint_results.md) is the historical reference for the fixed-strength J0946 refit. It used \((\lambda_s,\lambda_4,\lambda_0)=(0.0002532641285041916, 538677.9228218549, 10572.677018080045)\), a 100 × 100 source grid, 2,293 selected pixels, 1,134 potential coefficients, and 802 derivative-valid nodes. Its reported objective is 605.0380047097701 and local GN log evidence is 6307.427179447846. Its \(\delta\kappa\) positive peak is approximately 0.17433 at \((-0.7'',1.1'')\) on a \(0.1''\) potential grid.
- The [archived experiment runner](../../workspace/iteration/run_cubic_evidence_joint.py) records how the frozen fit start and distinct bilinear whitening reference were prepared. The [previous fixed-state Hermite replay](../../docs/validation/pixelized-source-interpolation-j0946-replay.md) establishes renderer parity on a different archived trial. The gates above apply to the new refits at trial 497's strengths.
