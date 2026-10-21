Status: ready-for-agent

## Problem Statement

When a lens modeler samples a dense pixelized-source model, every likelihood evaluation rebuilds the source regularization matrix from the current source grid. If the source grid is held fixed, that geometry does not change between proposals, but the dense evidence model has no place to keep the matrix or its log determinant. The same gap applies to the joint design matrix: once the source bounding box, every lens-mass component, and every unit-amplitude lens-light basis are fixed, the design matrix, its noise-weighted Gram matrix, and its right-hand side are invariant, yet they are rebuilt on every evaluation. Sampling therefore repeats work that the operator evidence model already skips. The dense model also cannot accept a fixed source bounding box, so a mass proposal still changes the pixelized source basis even when the modeler intended to freeze it.

## Solution

The dense pixelized evidence model accepts the same fixed square source bounding box as the operator evidence model. At construction it stores the regularization quantities that are invariant on that grid, and, when the lens mapping and lens-light bases are also invariant, the joint design matrix and the data-dependent normal-equation terms. Later evidence and reconstruction calls reuse those stored values. Regularization strength stays outside the cache, and anything that still depends on the proposal is built again. Without a fixed bounding box, evidence behaves as it does today.

## User Stories

1. As a lens modeler sampling lens mass on a frozen pixelized source grid, I want the dense evidence model to accept a fixed source bounding box, so that every mass proposal uses the same source basis and the same regularization geometry.
2. As a lens modeler, I want a missing fixed source bounding box to keep today's behavior, so that existing dense-model runs continue to infer a source bounding box from seed rays on every evaluation.
3. As a lens modeler, I want an inferred bounding box to remain detached from the gradient, so that freezing the grid is a separate choice from the existing stop-gradient behavior.
4. As a pipeline author, I want the fixed bounding box validated as a finite square with positive width and height, so that a rectangular or invalid grid is rejected before sampling starts.
5. As a pipeline author, I want non-finite bounds rejected, so that a bad reference bbox cannot enter the cache.
6. As a pipeline author, I want inverted or zero-width bounds rejected, so that the source grid has a usable physical spacing.
7. As a pipeline author, I want a square check consistent with the operator evidence model, so that a bbox accepted by one backend is accepted by the other.
8. As a pipeline author, I want `infer_source_bbox` on the dense evidence model, so that I can read the current reference model's square bbox and pass it to a second, cached model.
9. As a pipeline author, I want `infer_source_bbox` on an already fixed model to return that fixed bbox, so that the helper reports the grid the model actually uses.
10. As a pipeline author, I want `infer_source_bbox` on an unfixed model to return the square bbox inferred from the current seed rays, so that the helper matches the grid the unfixed evidence path would use.
11. As a lens modeler using finite-difference source regularization, I want the dense matrix and its log determinant built once for a fixed grid, so that sampling does not reassemble or refactorize that matrix per proposal.
12. As a lens modeler, I want zero-order, first-order, and second-order regularization to share that cache, so that the speedup does not depend on which finite-difference order I chose.
13. As a lens modeler, I want the regularization strength to stay outside the cached matrix, so that changing `log_lambda_reg` still changes the evidence and the reconstructed source.
14. As a lens modeler, I want the evidence path to use the stored log determinant on a cache hit, so that the model does not pay for another `slogdet` of the same matrix.
15. As a caller of the regularization helper, I want finite-difference calls to keep returning no covariance log determinant, so that existing callers that branch on that absence keep working.
16. As a lens modeler using a Gaussian-process source prior with a static kernel scale, I want the precision matrix and `log|K|` stored with the fixed grid, so that the kernel is not rebuilt while I sample mass or regularization strength.
17. As a lens modeler sampling the kernel scale on a fixed grid, I want the precision matrix and `log|K|` rebuilt for each kernel proposal, so that the cache does not freeze a kernel that is still being inferred.
18. As a lens modeler sampling the kernel scale on a fixed grid, I want those rebuilds to use the fixed bounding box, so that the source basis stays put while the kernel changes.
19. As a caller of the regularization helper, I want a Gaussian-process cache hit to return the stored precision and stored covariance log determinant, so that the helper and the evidence path agree.
20. As a caller of the regularization helper, I want a different bounding box to be evaluated freshly, so that I can inspect another grid without disturbing the cache used by sampling.
21. As a lens modeler, I want that fresh evaluation to leave the stored fixed-grid matrix in place, so that the next likelihood call still hits the cache.
22. As a lens modeler, I want `forward_model` to use the same fixed grid and the same cached regularization as the evidence, so that a reconstruction matches the image that produced the evidence.
23. As a lens modeler, I want both the Cholesky solver and the non-negative solver to share the cache, so that switching solver does not disable it.
24. As a lens modeler, I want the Cholesky factor itself left uncached, so that each proposal still factorizes the curvature that includes the current regularization strength.
25. As a lens modeler running a joint semi-linear inversion with fixed lens mass and fixed unit-amplitude lens-light bases, I want the joint design matrix, the noise-weighted data Gram matrix, and the data right-hand side stored at construction, so that repeated solves at that mass only update the regularization terms.
26. As a lens modeler, I want lens-light intensities to remain the solved linear amplitudes, so that fixing a unit-amplitude lens-light basis does not freeze the fitted intensities.
27. As a lens modeler, I want the weak lens-light regularization added on every evaluation, so that the stored Gram matrix stays the data term only.
28. As a lens modeler sampling lens mass on a fixed source grid, I want the design-matrix cache skipped while the regularization cache remains, so that ray tracing still follows the mass proposal.
29. As a lens modeler sampling a lens-light basis shape while the mass and source grid are fixed, I want the whole design-matrix cache skipped, so that a stale cross-term between source and lens light cannot enter the solve.
30. As a lens modeler with no lens-light components, I want the design-matrix cache to depend only on the fixed grid and fixed lens mass, so that an empty lens-light list does not block it.
31. As a lens modeler with no lens-mass components, I want the design-matrix cache to be allowed when the source grid and any lens-light bases are fixed, so that an empty mass list is treated as fixed.
32. As a lens modeler, I want a pointer parameter counted as fixed only when the parameters it points to are fixed, so that a static mass parameter driven by a dynamic parameter does not freeze the lens mapping.
33. As a lens modeler, I want a pointer-driven kernel scale treated as dynamic, so that the precision matrix is rebuilt as that driver changes.
34. As a lens modeler, I want a static parameter with no pointer, and a pointer whose children are fixed, to count as fixed, so that the cache closes over values that cannot change during sampling.
35. As a lens modeler, I want adaptive source regularization to keep being rejected by the dense model, so that this cache does not silently reintroduce a scale map the dense path no longer supports.
36. As a lens modeler, I want the cache filled when the evidence model is constructed, so that the sampling loop does not pay the setup cost on the first proposal.
37. As a lens modeler, I want parameter modes set before construction, so that the model can see which inputs are actually fixed.
38. As a lens modeler, I want a documented requirement to rebuild the evidence model and its compiled likelihood after changing a fixed value, a parameter mode, the observations, or the bounding box, so that a compiled trace cannot keep a stale matrix.
39. As a lens modeler, I want the position-likelihood penalty left unchanged and outside the cache, so that image-plane position constraints still apply on top of the cached solve.
40. As a lens modeler using sub-pixel ray tracing, I want the cached design matrix to be the same native-resolution, PSF-convolved matrix the solver uses today, so that `nsub` still matches an uncached evaluation.
41. As a lens modeler, I want masked pixels, the noise map, and the PSF taken from construction, so that the cached Gram matrix matches the observation the model was built with.
42. As a lens modeler, I want rays that fall outside the fixed source grid to contribute zero rows, so that fixing the grid does not change the existing out-of-bounds mapping.
43. As a lens modeler, I want a non-finite evidence value to become the same large negative penalty as today, so that cache hits do not change the failure contract.
44. As a caller of the simulator, I want `design_matrix` on the simulator to keep building from the current mass and the bbox it is given, so that diagnostic scripts that bypass the evidence model do not silently receive a cached matrix.
45. As a reader of the sampling guide, I want the dense fixed-grid cache described next to the operator cache, so that I know when each tier applies and when I must rebuild the compiled likelihood.
46. As a reader of the fixed-grid technical note, I want it to describe the dense model's fixed bounding box and two cache tiers, so that the note matches the implementation.
47. As a maintainer of the existing pixelized demos, I want those demos left unchanged, so that this work does not retune their source grids.
48. As a sampler evaluating a batch of regularization strengths, I want two different `log_lambda_reg` values to produce different sources under JIT and `vmap`, so that the cached geometry is shared while the regularization strength is not.
49. As a reviewer, I want a cached evaluation to match the same model with the cache cleared, so that storing the matrices does not change the evidence or the reconstruction.
50. As a reviewer, I want a fixed-grid mass sweep to avoid rebuilding regularization while still rebuilding the design matrix, so that the two tiers can be told apart.
51. As a reviewer, I want a fully fixed mass and lens-light evaluation to avoid calling design-matrix construction, so that the second tier is actually used.
52. As a reviewer, I want the unfixed evidence path to match the current dense model, so that the cache cannot change results for callers that never freeze the grid.

## Implementation Decisions

- The only module whose interface grows is the dense pixelized evidence model. The pixelized source simulator, the regularization builder, and the operator evidence model keep their current public behavior.
- Add an optional fixed source bounding box, defaulting to absent. Validation matches the operator evidence model: four finite values, positive width, positive height, and square within the operator's existing numeric tolerance. Store the accepted bounds as `float32`.
- Add `infer_source_bbox`. It returns four Python floats. On a fixed model it returns the stored bbox. On an unfixed model it returns the square bbox inferred from the current seed rays, using the padding and outlier fraction the dense model already uses.
- Fixedness uses the operator rule, including transitive dependencies. A parameter is fixed when it is not dynamic and, if it is a pointer, it has children. Apply that rule to every lens-mass component, every lens-light component, and, for Gaussian-process regularization, to the kernel scale. An empty component list is fixed. Lens-light intensities are not parameters in this check; they are the solved amplitudes of unit-amplitude lens-light bases, consistent with the unit-amplitude basis decision.
- The cache lives on the evidence model and is stored so the module registry does not treat it as a parameter. It is filled in the constructor. There is no later invalidation. Changing a fixed physical value, a parameter mode, the observations, or the bbox requires a new evidence model and a newly compiled likelihood.
- Tier one is filled only when a fixed bbox is present and the regularization geometry is invariant. Finite-difference regularization is invariant on a fixed grid. Gaussian-process regularization is invariant only when the kernel scale is fixed. Adaptive regularization remains rejected whenever its strength is dynamic or meaningfully nonzero, which is the dense model's current rule.
- Tier one stores the dense regularization matrix. It also stores the log determinant the evidence uses: `logdet(R)` for finite differences, including a negative-infinity result when the matrix is not positive definite, and `log|K|` for a fixed Gaussian-process kernel. The evidence path reads that stored determinant and does not call `slogdet` again. The regularization helper still returns `(matrix, covariance log determinant)`. The covariance log determinant stays absent for finite differences and is the stored `log|K|` for a cached Gaussian process.
- The helper returns the stored matrix when its bbox is the stored fixed bbox. Any other bbox is built immediately and does not replace the stored matrix or determinant.
- Regularization strength is applied at evaluation time as `lambda * R`, with the usual `0.5 * N_source * log(lambda)` evidence term. It is not multiplied into the stored matrix.
- Tier two is all or nothing. It is filled only when tier one's bbox is fixed and every lens-mass component and every lens-light component is fixed. It stores the joint design matrix `M`, the noise-weighted data Gram matrix `M^T C^{-1} M`, and the data right-hand side `M^T C^{-1} d`. These are exactly the objects the current solve builds before adding priors. Sub-pixel aggregation and PSF convolution are already included in `M`.
- If either the mass or a lens-light basis is dynamic, store none of the three tier-two objects. Do not cache the source block alone or the lens-light block alone.
- On a tier-two hit, evaluation adds `lambda * R` on the source block and the existing lens-light regularization on the lens-light block, then solves with the configured Cholesky or non-negative solver. The residual uses the stored `M`. The Cholesky factor is not stored.
- On a tier-two miss with a fixed bbox, evaluation still uses the fixed bbox for the design matrix and does not infer a new bbox. It rebuilds the design matrix from the current mass and lens-light bases. On a tier-one miss, it also rebuilds the regularization matrix on that same fixed bbox.
- With no fixed bbox, evaluation keeps the current sequence: infer the bbox, detach it when the simulator is configured to, build the design matrix, and build the regularization matrix. No tier is stored.
- Evidence, the scalar likelihood entry point, and `forward_model` all read the cache. `forward_model` keeps its current return values. The position-likelihood penalty stays where it is now, outside the cached solve.
- The simulator's own design-matrix method stays a pure function of the current physical parameters and the bbox it receives. The evidence model passes the fixed bbox into that method. Direct simulator calls do not consult the evidence-model cache.
- Existing pixelized demos are not switched onto a fixed bbox.
- Document the two tiers, the fixedness rule, and the rebuild requirement in the sampling guide beside the operator cache, and update the fixed-grid technical note so it describes this dense-model behavior.

## Testing Decisions

- Test only the dense pixelized evidence model, in the existing dense-inversion test module. Do not add a simulator seam or a regularization-builder seam for this feature.
- A good test checks evidence, reconstruction, `infer_source_bbox`, the regularization helper, and constructor rejection. Cache hits are proved the way the operator cache tests prove them: after construction, replace design-matrix construction or regularization construction and require the cached path to keep working, and clear the prepared cache on a reference model to show numerical agreement.
- Compare cached and cache-cleared evidence, model image, and source pixels at `rtol=1e-6` and `atol=1e-6`.
- Cover the agreed cases: no fixed bbox; fixed bbox with dynamic mass; static and dynamic kernel scale; fully fixed mass and lens-light bases; a pointer-driven mass parameter; a non-square bbox rejected at construction; a helper call with a different bbox; `forward_model` sharing the cache; and two JIT/`vmap` proposals with different regularization strengths.
- Prior art is the operator tests that clear prepared regularization and mapping and then forbid those builders to run, together with the existing dense tests for evidence, joint semi-linear inversion, both solvers, and vectorized likelihood.
- Do not assert the private names of cached arrays beyond the attribute the reference test clears.

## Out of Scope

- Adaptive regularization, fixed scale maps, and fixed source templates on the dense model.
- A partial design-matrix cache when only the mass or only the lens-light bases are fixed.
- Caching the Cholesky factor or the non-negative solver's intermediate state.
- Caching inside the simulator for callers that do not go through the evidence model.
- Changing pixelized demo pipelines or their source grids.
- Operator-backend cache behavior, operator solvers, and implicit differentiation.
- New constructor arguments for source-bbox padding or outlier fraction.
- Gradients of the dense evidence model.
- Changes to the position-likelihood penalty.

## Further Notes

- The project glossary does not yet define pixelized source, fixed source bbox, or source regularization. This spec uses those terms as they already appear on the evidence-model interface and in the fixed-grid technical note.
- The unit-amplitude lens-light basis decision still holds. The cache watches the basis module, not the solved lens-light intensity.
- The operator preconditioner decision is untouched. This cache is for the dense normal equations.
- The fixed-grid technical note currently says the dense constructor has no fixed bbox. Updating that note is part of this work, not a conflicting architecture decision.
