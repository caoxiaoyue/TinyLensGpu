# Single-band pixelized potential correction

**Status:** ready-for-agent
**State:** open
**Design approved:** 2026-10-28

## Problem Statement

Researchers using TinyLensGpu need to reconstruct non-parametric lens-potential perturbations from extended lensed images while retaining the library's regular square source grids, JAX execution, GPU acceleration, and operator-based memory savings. The existing pixelized-source models do not provide a complete source–potential reconstruction workflow.

The reference potential-correction implementation demonstrates single-step and iterative reconstruction, but its PyAuto-specific containers, source representation, dense covariance priors, mask cleaning, and execution paths cannot be copied directly without losing these properties. In particular, identifying the potential support with the observed arc mask can leave insufficient neighbors for physical derivatives and interpolation; repeatedly removing pixels complicates geometry and can change the modeled data.

Users need both algorithms to recover a simulated subhalo perturbation, an explicit distinction between exact and approximate evidence calculations, reproducible numerical products, and a clear failure contract. The first release is a reconstruction and algorithm-validation capability, not a calibrated subhalo-detection or mass-measurement pipeline.

## Solution

Add separate single-step and iterative potential-correction entry points with a shared discretization, regularization model, result contract, and dense/operator backend selection. Operate on single-band lens-light-subtracted images with a fixed macro lens mass model and an unconstrained, regular square pixelized source.

Construct a local potential support from the actual image-subpixel interpolation requirements and add the neighboring potential unknowns required by centered physical derivatives and complete fourth-difference windows. Preserve the observation mask rather than iteratively eroding it. Use complete fourth-difference windows plus an independent, explicit weak zero-order potential prior.

Use mock truth to initialize the source information in the first examples. Deliver a dense reference first, then an operator implementation with block-approximate evidence, then the iterative workflow. Optimize source and potential smoothing strengths through an outer derivative-free search. Provide Pseudo-Jaffe subhalo and no-subhalo examples, saved numerical products, diagnostic figures, and measured performance.

## User Stories

1. As a lens-modeling researcher, I want to reconstruct a pixelized lens-potential perturbation, so that I can study image structure not explained by a fixed macro lens mass model.
2. As a TinyLensGpu user, I want to retain the existing regular square source grid, so that potential correction integrates with my source-modeling workflow.
3. As a researcher using lens-light-subtracted observations, I want a single-band correction workflow, so that I can analyze the residual lensed-source image directly.
4. As a researcher comparing algorithms, I want independent single-step and iterative entry points, so that their assumptions and results remain distinguishable.
5. As a single-step user, I want the full source brightness and potential correction solved together, so that the source can respond to the perturbation.
6. As a mock-validation user, I want initial source values and gradients derived from the mock truth, so that I can isolate correction-algorithm behavior.
7. As an iterative user, I want source gradients and corrected ray positions updated after accepted changes, so that each step describes the current reconstruction.
8. As a researcher, I want the macro lens mass model fixed during reconstruction, so that the first release has a controlled modeling scope.
9. As a future extension author, I want macro-dependent preparation separated from fixed geometry, so that variable-macro single-step inference can be added later.
10. As a researcher using Gaussian linear inference, I want signed source and potential coefficients, so that the solver and evidence describe the same unconstrained model.
11. As a user of fixed source grids, I want their existing boundary and out-of-bounds behavior preserved, so that this feature does not change established source modeling.
12. As a user with an irregular arc mask, I want all selected observation pixels retained, so that stencil preparation does not discard my data.
13. As a user with narrow arcs or isolated selected pixels, I want sufficient local potential support, so that the required physical derivatives and interpolation are defined.
14. As a GPU user, I want the potential support localized around the observed arcs, so that I do not have to solve a full rectangular potential field.
15. As a researcher interpreting boundary pixels, I want support-only unknowns identified, so that they are not mistaken for directly constrained measurements.
16. As a researcher choosing resolution, I want potential spacing configurable independently of source resolution, so that I can balance spatial detail and cost.
17. As a researcher, I want the same physical deflection operator in linearization and nonlinear rendering, so that their difference reflects linearization error.
18. As an iterative user, I want source gradients consistent with the brightness interpolation, so that the local Jacobian matches the forward model within grid cells.
19. As a researcher selecting a prior, I want complete fourth-difference penalties, so that smoothing does not silently change derivative order at the boundary.
20. As a researcher using a fixed potential grid, I want pixel-unit smoothing, so that regularization follows the approved discrete convention.
21. As a researcher, I want an independent weak amplitude prior, so that modes not controlled by fourth differences have a proper, explicit prior.
22. As a researcher interpreting evidence, I want the amplitude prior included in normalization, so that it is not treated as invisible numerical jitter.
23. As a user tuning smoothing, I want source and fourth-difference strengths optimized separately, so that each field can have appropriate regularization.
24. As a researcher, I want the weak amplitude strength fixed during that optimization, so that its meaning does not change with smoothing strength.
25. As a researcher inspecting low-order potential modes, I want the original fitted solution preserved, so that diagnostic alignment does not alter the inference result.
26. As a dense-backend user, I want complete joint matrices and determinants, so that I have a numerical reference for the new model.
27. As an operator-backend user, I want matrix-free joint products, so that large reconstructions avoid unnecessary dense storage.
28. As an operator-backend user, I want source–potential coupling retained in the true solve, so that preconditioning approximations do not change the fitted model.
29. As a researcher using approximate evidence, I want its method recorded and compared with the dense reference, so that I can assess its effect on regularization selection.
30. As a GPU user, I want regularization preparation to avoid eager dense matrices, so that memory savings begin before the first solve.
31. As an iterative user, I want accepted updates to reduce the actual regularized nonlinear objective, so that a misleading local prediction does not determine acceptance.
32. As an iterative user, I want regularization applied to the accumulated fields, so that repeated updates do not evade the specified prior.
33. As a researcher, I want nonconvergence and numerical failures reported explicitly, so that failed fits are not presented as successful scientific results.
34. As an optimization user, I want failed reconstruction candidates rejected with reasons, so that the search does not reward invalid evidence values.
35. As a reproducibility-focused user, I want every hyperparameter candidate to start from the same initialization, so that evaluation does not depend on search order.
36. As a researcher, I want bounded derivative-free optimization in log-strength space, so that positive regularization strengths can be searched without automatic differentiation.
37. As an integration author, I want callable inference interfaces, so that other optimizers and samplers can be connected later.
38. As a GPU user, I want expensive numerical kernels compiled with JAX, so that Python orchestration does not move large systems to NumPy.
39. As a numerical-validation user, I want a float64 reference and explicit float32 selection, so that precision trade-offs are measured rather than hidden.
40. As a library user, I want existing imports, pixelized-model behavior, and process precision defaults preserved, so that adding correction does not disrupt other analyses.
41. As an analysis author, I want a common result structure across algorithms and backends, so that plotting and persistence do not reconstruct solver internals.
42. As a single-step user, I want both the linear prediction and a nonlinear re-render, so that I can inspect the approximation used by the fit.
43. As an iterative user, I want objective history and termination information, so that I can assess the final reconstruction.
44. As a researcher validating recovery, I want Pseudo-Jaffe perturbation truth maps, so that I can compare reconstructed potential, deflection, and convergence.
45. As a researcher checking artifacts, I want a matched no-subhalo control, so that spurious correction structure is visible.
46. As a researcher, I want recovered position, convergence error, source recovery, and image residuals reported, so that success is assessed beyond a finite evidence value.
47. As a user reproducing a run, I want configurations, geometry, strengths, initialization provenance, and numerical results saved, so that I can reconstruct the conditions of the fit.
48. As a performance-focused user, I want compilation, warm-run timing, active sizes, and peak memory reported, so that comparisons reflect the actual computational problem.
49. As a maintainer, I want tests centered on public numerical results and artifacts, so that implementation refactoring does not invalidate equivalent behavior.
50. As a maintainer, I want the work delivered in independently verifiable stages, so that dense-reference, operator, and iterative issues can be isolated.

## Implementation Decisions

1. **Model scope and terminology.** Support one band of lens-light-subtracted imaging with the existing image/noise/PSF/mask conventions. The macro lens mass model is fixed in both algorithms. Use “source–potential reconstruction” for the new coupled problem; do not redefine the glossary's “joint semi-linear inversion,” which currently denotes source–lens-light fitting. A parametric subhalo mass component is mock truth, not the fitted replacement for the pixelized correction.

2. **Architecture and public boundary.** Provide separate single-step and iterative algorithm entry points, backend selection, and a common structured result. Share potential-grid preparation, physical derivatives, source mapping and gradients, regularization, and joint numerical operators. Keep geometry, numerical fitting, inference orchestration, and example visualization in their corresponding architectural responsibilities. Exact class names are implementation choices, not requirements of this spec. Do not add a generic workflow engine or require callers to assemble private solver internals.

3. **Compatibility with existing decisions.** Preserve established pixelized public interfaces and behavior outside the approved changes. ADR 0001 mandates block-Schur handling for existing source–lens-light inversion; that contract remains unchanged. The explicitly approved block approximation here concerns two large fields, source and potential, and does not authorize an uncoupled replacement for the existing source–lens-light preconditioner. Preserve ADR 0002's unit-amplitude lens-light bases and ADR 0003's actual-prior semantics. No lens-light amplitudes are fitted in this feature.

4. **Source representation.** Reuse the regular square pixelized source, its node ordering, bilinear interpolation, fixed bounding box, and existing out-of-bounds and boundary-branch conventions. Fix source geometry throughout each correction run and all its hyperparameter evaluations. Do not expand the source grid, rerun with a larger box, or change its boundary behavior automatically. Default to existing second-order source regularization with one scalar strength; retain configuration of existing supported nonadaptive regularization types without introducing new source-adaptive regularization.

5. **Initialization contract.** The first examples derive initial source information from the mock's true source. Single-step reconstruction receives a source-gradient provider evaluated at the macro ray-traced coordinates and freezes it for the correction run. An analytic source derivative or a documented small-step finite difference of the true source function is acceptable; sampling the truth onto the pixelized grid first is not required for this gradient. Iterative reconstruction starts from truth-derived source-node intensities and zero potential correction, then recomputes gradients from the current reconstructed pixelized source. It must not continue using the truth gradient after initialization. Do not add a source-only inference stage as the default initializer.

6. **Signed variables.** Both source intensities and potential coefficients are unconstrained real variables. Do not apply the existing all-variable NNLS or PNPG constraint to the joint vector. Negative source intensities may be reported but must not be clipped after fitting or silently substituted into the evidence calculation.

7. **Core potential nodes.** Define the core node set \(C\) from the actual image-plane subpixel positions used by the existing imaging forward model: collect all four regular coarse-grid corners needed for bilinear deflection interpolation at each such position. Do not identify this set with a downsampled image mask or require a coarse image block to be wholly unmasked. Fix this geometry for the run.

8. **Constructive potential support.** Solve for potential values on
   \[
   U=C\cup(C\pm e_x)\cup(C\pm2e_x)\cup(C\pm e_y)\cup(C\pm2e_y),
   \]
   where \(e_x,e_y\) are one potential-grid index step. Use the union, without duplicate unknowns. This axial two-step support is the approved local alternative to a full rectangular unknown field. Ensure the coarse-grid bounding representation includes the support rather than clipping it at the image edge. Preserve the observation mask, including isolated selected pixels; do not use iterative erosion to satisfy derivative stencils. This supersedes the earlier proposal to confine every potential unknown strictly inside the original arc mask.

9. **Support semantics and coverage.** All values on \(U\) are fitted unknowns, including those outside the observed arc; they are not fixed-zero padding, copied boundary values, or additional observed data. Compute required physical derivatives on \(C\), using neighbors in \(U\). Do not recursively require complete derivatives at every support-only node. Identify core/support membership and valid derivative coverage in results; do not claim full deflection or convergence coverage throughout \(U\). Handle inadequate explicit user geometry with an informative validation error, not silent loss of observation pixels. Missing or malformed numerical inputs retain the library's appropriate validation behavior.

10. **Grid resolution and coordinates.** Make potential spacing configurable and default it to twice the image-pixel spacing; source resolution remains independently configurable. Keep the geometry fixed during correction and hyperparameter optimization. Use the library's coordinate ordering, orientation, and flattening consistently. Physical potential derivatives use angular spacings even though the potential smoothing operator uses pixel units. The corrected source-plane position is the macro ray-traced position minus the correction deflection; respect the existing macro interface's distinction between a ray-traced coordinate and a component deflection.

11. **Physical potential operators.** On core nodes, use centered finite differences for deflection and the corresponding centered second derivatives for convergence. Interpolate the core-node deflection to image subpixels with the prepared four-corner bilinear weights. Use this identical potential-to-deflection map in the single-step response, iterative Jacobian, and nonlinear rendering. Core support guarantees complete templates, so the forward model needs neither nearest-valid-cell searches nor extrapolation across mask holes. Use \(\delta\kappa=(\partial_{xx}\delta\psi+\partial_{yy}\delta\psi)/2\) for numerical diagnostics on their valid domain.

12. **Pixelized-source gradients.** Differentiate the existing bilinear brightness formula analytically within each cell, using its same source nodes, spacings, validity rules, and boundary-branch conventions. Do not substitute finite differences at source nodes followed by gradient interpolation. This is an explicitly coded spatial derivative, not an automatic-differentiation requirement. Gradients may jump across cell boundaries; step acceptance is based on the actual nonlinear objective.

13. **Imaging reuse.** Reuse the current pixelized forward/adjoint PSF, masking, and subpixel integration semantics. Do not substitute a different convolution helper with different padding or edge behavior under the guise of reuse. Both source and potential-response columns pass through the same relevant imaging operations. This feature adds potential support, not a PSF halo or expanded source domain.

14. **Fourth-difference prior.** In each horizontal and vertical direction, enumerate every contiguous five-node window lying wholly in \(U\), once, with coefficients \([1,-4,6,-4,1]\) in pixel units. Do not jump over absent nodes, duplicate shifted edge windows, or restrict enumeration to windows centered on \(C\). Forward enumeration and center-based labeling describe the same complete window. The difference operators may be rectangular. There is no automatic third-, second-, first-, or zero-order boundary fallback in these operators.

15. **Complete prior and explicit amplitude scale.** Let \(p\) be the potential values on \(U\). Use
   \[
   R_\psi=\lambda_4(D_{4x}^{\mathsf T}D_{4x}+D_{4y}^{\mathsf T}D_{4y})+\lambda_0 I,\qquad
   Q=\operatorname{diag}(\lambda_sR_s,R_\psi).
   \]
   The fourth-difference part is generally singular; the positive, explicit \(\lambda_0\) supplies a proper amplitude prior on every potential coefficient, not just its nullspace. Fix \(\lambda_0\) during smoothing optimization and do not tie it to \(\lambda_4\). Choose and record the demo value using sensitivity checks of both recovery and evidence; “weak” is not established merely by comparing the two strengths numerically. Do not conceal this term as numerical jitter or silently alter it to rescue a failed factorization.

16. **Gauge and low-order modes.** Do not impose additional hard constant-potential or linear-slope constraints in the first release. A positive amplitude prior makes the prior proper; it does not prove that the data remove physical degeneracies. Preserve the raw fitted solution. Optional affine-aligned truth-comparison diagnostics must identify the alignment and must not replace the fitted fields, forward image, or evidence. Record per-node fourth-window coverage so prior-dominated support regions can be recognized.

17. **Single-step model.** With fixed macro source mapping \(F_0\), fixed truth-derived initial source gradient, and linear potential response \(G_0\), solve the full vector \(r=(s,p)\) in \(d\simeq Mr\), \(M=[F_0,G_0]\). The potential response has the lens-equation minus sign and applies the same image operators as the source response. Solve
   \[
   H r=M^{\mathsf T}C_n^{-1}d,\qquad H=M^{\mathsf T}C_n^{-1}M+Q.
   \]
   The unknown source is the complete source brightness, not merely an increment and not a fixed truth source.

18. **Iterative model.** Minimize
   \[
   \Phi(r)=\tfrac12[d-f(r)]^{\mathsf T}C_n^{-1}[d-f(r)]+\tfrac12r^{\mathsf T}Qr
   \]
   with fixed macro parameters and nonlinear re-rendering through the corrected ray positions. A damped Gauss–Newton/LM step satisfies
   \[
   (J^{\mathsf T}C_n^{-1}J+Q+\text{damping})\Delta r
   =J^{\mathsf T}C_n^{-1}[d-f(r)]-Qr.
   \]
   Update source and potential together. The regularizer acts on the accumulated state, and the right-hand-side prior term is required; repeatedly solving the single-step problem and adding its solution is not this algorithm.

19. **Acceptance, stopping, and failure.** Accept updates according to decrease in the actual regularized nonlinear objective and adjust damping on acceptance/rejection. Use configurable, recorded checks involving objective change, scaled update size, and gradient, with iteration limits. Do not label an iteration limit, failed linear solve, nonfinite state, or repeated rejection as successful convergence. Return an explicit termination reason and available diagnostics. Invalid configuration is distinguishable from failure of a valid numerical candidate. Damping used to obtain a step is not part of the physical prior or final evidence curvature.

20. **Dense reference backend.** Assemble the complete coupled linear system and solve the unconstrained positive-definite problem with Cholesky-based linear algebra. Compute complete prior and posterior log determinants. Use the same prepared geometry and mathematical operators as the operator backend. Dense results are a numerical reference at fixed configuration; they are not a different scientific discretization.

21. **Operator solve and preconditioning.** Implement joint forward and transpose actions and use PCG with block preconditioning. The actual curvature action must retain source–potential cross terms: applying a block-diagonal approximation in place of the true joint system is prohibited. Approximation is permitted in the preconditioner and determinant evaluation. Neither solve preparation nor the preconditioner may require a dense source-by-potential cross matrix or full dense potential curvature merely to reproduce the existing source–lens-light Schur implementation.

22. **Memory preparation.** Make dense source-difference matrix construction lazy along the reused finite-difference regularization path, preserving its mathematical results and established callers. New potential stencils are prepared without materializing dense square matrices for the operator backend. The dense backend explicitly requests materialization. Do not broaden this into an unrelated regularization or backend rewrite.

23. **Evidence definition.** For the conditional single-step linear Gaussian model, use
   \[
   \log Z=-\tfrac12\left[\chi^2+r^{\mathsf T}Qr+\log|H|-\log|Q|+
   \log|C_n|+N_{\rm data}\log(2\pi)\right].
   \]
   All fitted support coefficients belong to the prior and posterior normalization, including support-only nodes. In particular, the potential prior determinant is that of the complete sum \(R_\psi\); it cannot be replaced by a pure-\(\lambda_4\) scaling formula for the singular fourth-difference matrix.

24. **Evidence approximations.** The dense backend evaluates complete determinants numerically. The operator backend initially uses block-based approximations to the relevant determinant terms, consistent with the approved complete prior. Mark the chosen approximation explicitly. For the iterative workflow, evaluate evidence at a converged state using the local Gauss–Newton curvature without LM damping; identify this as a local-linearization approximation even for dense determinants. Record this separately from operator determinant approximation. Do not describe the iterative evidence as an exact nonlinear integral.

25. **Approximation validation boundary.** Compare operator evidence with the dense reference on small identical problems, including the effect on selected strengths and recovered fields, not just finiteness. A materially misleading optimum or recovery remains an unmet validation requirement; it is not excused by labeling the method approximate. Retain an internal determinant-method extension boundary, but do not automatically add SLQ/Lanczos or a dense cross-field Schur method to this release.

26. **Outer optimization.** Reuse DIRECT or its suitable existing integration for bounded derivative-free search in \((\log\lambda_s,\log\lambda_4)\), preserving the required dtype. Make bounds explicit; the demo may establish them through a recorded preliminary scan and must report an optimum at the configured boundary. Each single-step candidate performs one joint solve. Each iterative candidate holds strengths fixed during a complete inner run and evaluates evidence only after successful convergence. Every candidate starts from the same truth-derived source initialization and zero potential correction; do not use search-order-dependent warm starts. Reject failed candidates through the optimizer's invalid-objective convention while retaining the reason.

27. **Future inference and macro seam.** Expose an appropriate callable for fixed-strength fits/evidence so other optimizers and samplers can be attached. Separate macro-dependent mapping and linearization preparation from invariant geometry and data. Future variable-macro single-step fitting must be able to rebuild those dependencies correctly; actual macro variation and a complete posterior-sampling example are not part of this release. Iterative updates must not reuse caches containing obsolete source gradients or corrected ray positions.

28. **Execution and precision.** Prepare static topology, indices, and fixed geometry once; run expensive forward/adjoint operations, system products, linear solves, and principal evidence computations with JAX JIT/GPU. Python may manage outer iterations, acceptance, damping, hyperparameter search, and histories, transferring small control scalars rather than large systems to NumPy. Establish float64 numerical references and support explicit float32 configuration. Avoid hidden intermediate downcasts in reused bounds, regularization, or optimizer paths. Do not globally change library precision defaults. Float32 performance examples require numerical comparison first. New automatic-differentiation guarantees are not required; existing supported differentiation must not regress through shared-code changes.

29. **Result contract.** Return source coefficients and geometry, potential coefficients and core/support geometry, valid-domain deflection and convergence, fitted images and residuals, strengths, evidence and its method annotations, dtype, solver status, and initialization/configuration provenance. Iterative results additionally contain termination reason and iteration history sufficient to inspect objective progression and accepted/rejected updates. A failed result is not advertised or persisted as a successful reconstruction; diagnostic failure records may be retained. Plotting and exports consume this public result instead of independently rebuilding a fit.

30. **Single-step and iterative images.** Single-step output distinguishes the fitted linear prediction from a nonlinear re-render using the recovered source and potential. Provide corresponding diagnostics so linearization error is visible; do not substitute the nonlinear image into the single-step evidence formula. Iterative output contains the final nonlinear image, residuals, and convergence history.

31. **Scientific mock and truth products.** Use TinyLensGpu's PseudoJaffeSubhalo subhalo mass component with its existing strength, host-Einstein-radius, position, and tidal-radius prescription. It is not interchangeable with the separate cored PseudoJaffe parameterization. The mock need only reproduce the physical scenario, not the reference script's exact data, NFW profile, or sampling. Use the correct mock macro parameters during reconstruction and include a matched no-subhalo control. Add independent analytic diagnostic helpers for true potential and convergence, since the existing component exposes deflection but not those products. Respect the zero-strength limit and the profile's central singularity/numerical convention; do not change the profile or invent a core to simplify truth-error metrics. State the comparison sampling or averaging convention.

32. **Examples, persistence, and performance.** Deliver small numerical cases for backend comparison and an approximately 200-by-200-image scientific example with configurable source resolution, potential spacing, and integration settings. Save numerical results and reproducibility configuration and generate source, potential, deflection, convergence, residual, and relevant convergence diagnostics. Report recovered perturbation location, convergence error, source recovery, and image residuals with the relevant resolution and mask. On the available RTX 5080 with approximately 16 GiB memory, measure initial compilation, warm solve/workflow times, peak memory, and actual data/source/core/support counts. Do not promise an unmeasured speedup or confuse simulated-image oversampling with inversion oversampling.

33. **Delivery sequence.** Deliver shared geometry/operators and lazy regularization preparation first; single-step dense reconstruction and exact conditional evidence second; single-step operator reconstruction and approximate evidence third; both-backend iterative reconstruction with outer optimization fourth; and complete scientific examples, documentation, diagnostics, and performance evidence fifth. Earlier stages require focused numerical verification; completion of a dense prototype alone does not satisfy this spec.

## Testing Decisions

1. **Confirmed seams.** The completed interview explicitly approved public reconstruction results, backend comparisons, operator contracts, saved example artifacts, and performance measurements as the test surfaces. Use the public fit/evaluation-to-result boundary as the primary seam for both algorithms and backends. Use focused numerical operator contracts only where an adjoint, derivative, or support invariant needs independent verification. Do not add testing-only public interfaces or a separate test for every helper.

2. **Good tests and prior art.** Assert externally meaningful numerical behavior, status, approximation metadata, and artifacts rather than internal call order or a copied implementation. Follow existing dense/operator pixelized-inversion, regularization, solver, Pseudo-Jaffe subhalo, and public-result diagnostic tests. Follow existing example assembly/export/plotting tests and benchmark conventions for the corresponding boundaries. Independent small analytical constructions and explicit dense references are preferable to mocks of the numerical solve.

3. **Geometry and physical support.** Exercise ordinary masks, narrow arcs, holes, disconnected regions, isolated selected pixels, subpixel integration, and selected pixels near the image boundary. Verify observation pixels are preserved; interpolation corners and all required core derivative neighbors exist; support is unique and includes the required outer coordinates; and derivative validity is not claimed outside the supported output domain. Invalid explicit configurations must fail informatively. Do not require the physical data to constrain every added support coefficient.

4. **Numerical operator contracts.** Verify physical derivatives against smooth known potentials away from singularities, including the lens-equation sign and coordinate orientation. Check forward/transpose inner products and dense/operator action equivalence on small real systems. Check source spatial gradients against brightness perturbations within cells without crossing a grid boundary; separately check the established branch/out-of-bounds conventions. Verify that the single-step response approaches the nonlinear image change in the small-perturbation regime.

5. **Regularization semantics.** Check that equivalent forward/center window enumeration gives the same quadratic penalty, that absent nodes are not bridged, and that windows are not duplicated or silently replaced by lower-order penalties. Known low-order fields must have the expected fourth-difference behavior. Verify the complete amplitude-augmented prior, its dependence on both strengths, and dense/operator penalty agreement. Confirm that regularization changes do not change existing source results or eager-materialization callers.

6. **Coupled solve correctness.** At identical fixed geometry, strengths, precision, and initialization, compare dense and operator model images, recovered fields on supported domains, quadratic objective, and linear-system residuals. Include a case where source–potential coupling is nonzero so a decoupled solve cannot pass accidentally. Use declared tolerances appropriate to precision, conditioning, and linear-solver accuracy, rather than requiring identical PCG iteration counts.

7. **Evidence and optimization.** Verify dense conditional evidence against an independent small Gaussian calculation. Exercise multiple fourth-difference strengths while holding the positive amplitude prior fixed, detecting the invalid pure-strength determinant shortcut. Compare block approximations with the complete calculation across a small hyperparameter scan, including changes to optima and recovered fields. Check candidate reproducibility under changed search order, explicit bounds and boundary reporting, and rejection of unsuccessful inner fits. Identify the local-linearization and determinant approximations separately in returned results.

8. **Iteration and failure.** Verify that accepted updates reduce the actual nonlinear regularized objective within justified numerical tolerances and that the prior acts on the accumulated fields. Exercise convergence, iteration limits, unsuccessful inner linear solves, nonfinite candidates, and rejection/stalling. Check status propagation into optimization and diagnostic/export consumers. The final evidence must not depend on the artificial damping used to obtain the last step.

9. **Scientific validation.** Use a fixed true macro model, a Pseudo-Jaffe subhalo mock, and a matched no-subhalo control. Validate truth helpers against independent profile relations, including zero strength and documented handling of the singular center. Report perturbation localization, convergence recovery, source recovery, and image residuals with their masks, grid resolution, noise, integration, and prior configuration. Preserve raw and any explicitly aligned truth comparisons. Do not turn a visually plausible image or a finite evidence into a claim of calibrated detection or mass precision.

10. **Precision and memory.** Establish float64 baselines before validating float32. Check actual result and intermediate-facing numerical contracts for accidental precision loss in the newly reused paths. Verify that operator preparation and execution do not require full dense difference, joint-curvature, or source-by-potential cross matrices. Use observable memory/scaling evidence in the benchmark suite, supplemented by review of allocation paths; do not tie correctness to a private storage class.

11. **Artifacts and compatibility.** Through example entry points, verify configuration/result persistence, geometry and method metadata, valid-domain maps, the two distinct single-step renderings, iterative history, and failure records. Preserve existing source boundary behavior, current pixelized PSF semantics, established source–lens-light inversion, and relevant public interfaces. Shared-code changes require the applicable existing numerical and differentiation regressions even though new correction autodifferentiation is out of scope.

12. **Verification organization.** Put core library behavior and cross-module correctness in the core suite, example assembly/export/plotting behavior in the example suite, and timing/scaling assertions in benchmarks. Use the documented tinylens_gpu environment. Run these suites in separate processes because example imports can change JAX and plotting configuration. Run focused checks during each stage and the relevant broader regressions after integration. Record commands, outcomes, exact configurations, numerical tolerances, and resource measurements. The reference PT_jax environment is for reference-code checks only. No full posterior-sampling campaign is required.

13. **Calibration versus acceptance.** Solver tolerances, demonstration search bounds, and the weak amplitude-prior value are explicit numerical configuration, not hidden new scientific assumptions. Select and record them through the agreed small-scale comparisons and sensitivity checks. Document any remaining approximation limitation with the affected model/configuration; an unresolved recovery or approximation failure remains unfinished work rather than a silently relaxed acceptance condition.

## Out of Scope

- Multi-band correction, simultaneous lens-light fitting, or a lens-light subtraction algorithm.
- Actual variation or inference of the macro lens mass model in either algorithm.
- Adaptive/Voronoi source meshes, automatic source-box expansion, or changed existing source interpolation/boundary semantics.
- Automatic data-mask erosion, the reference coarse-block-all-valid mask rule, or nearest-valid-cell potential extrapolation as the production geometry strategy.
- Gaussian, exponential, Matérn, or other dense covariance potential priors.
- Boundary derivative-order fallback, hidden amplitude jitter, or new hard affine gauge constraints.
- Nonnegative or mixed-bound joint source–potential solvers.
- SLQ/Lanczos determinants, a dense cross-field Schur construction for source–potential inversion, or altered source–lens-light preconditioning.
- Changes to current pixelized PSF padding, convolution edges, or mask-adjacent light treatment.
- A default data-inferred source initialization stage; first examples use truth-derived initialization.
- Required new automatic-differentiation support or compiling all Python iteration control into one JIT function.
- Complete hyperparameter posterior-sampling examples, full pixel covariance, subhalo mass confidence intervals, or calibrated detection significance.
- Macro-model-mismatch stress tests or exact replication of the reference simulation's data and profile.
- Fitting a parametric subhalo in place of the potential correction.
- General mass-profile API expansion, unrelated numerical refactoring, or repository-wide precision-policy changes.
- Unmeasured speedup targets or claims that matrix and approximate operator evidence are numerically interchangeable.

## Further Notes

- This spec synthesizes the design interview and the user's explicit final approval. The testing seams were included in that approval; publication does not reopen the interview.
- The final decisions supersede intermediate suggestions: initialize from mock truth rather than a source-only fit; preserve existing fixed-source boundaries; use constructive local potential support rather than a full rectangular field or strict-mask erosion; use pixel-unit complete fourth differences plus independent weak zero-order regularization; add neither macro stress tests nor PSF boundary changes.
- The principal algorithm sources are [Cao et al., arXiv:2504.19177](https://arxiv.org/abs/2504.19177), [the iterative-method paper, arXiv:2510.07382](https://arxiv.org/abs/2510.07382), and [Koopmans, Gravitational Imaging of CDM Substructure](https://arxiv.org/abs/astro-ph/0501324). The single-step paper was inspected in its v2 form during planning. The complete-window/support construction and explicit amplitude prior are approved design choices, not claims that these papers prescribe that exact implementation.
- The user's JAX lensing_potential_correction checkout and its PT_jax environment were the reference implementation. It is a source of algorithmic evidence, not a runtime dependency or a mandate to preserve PyAuto containers, mask cleaning, host-side solves, or reference evidence shortcuts.
- A Gaussian prior and a well-defined discrete solve do not establish that all low-order modes or support-only pixels are measured by the data. Interpret truth comparisons and the first release's scientific claims accordingly.
- No physical unit convention, uncertainty claim, or exact backend agreement should be inferred solely from a saved scalar evidence. Persist its configuration and approximation provenance.
- Status ready-for-agent means the approved specification is ready for implementation. State open means implementation and verification remain outstanding. Publishing this document does not claim that code has been added, examples have run, or acceptance tests have passed.
