# 03: 缓存固定质量下的联合设计矩阵

**What to build:** When the source grid, every lens-mass component, and every unit-amplitude lens-light basis are fixed, the dense evidence model stores the joint design matrix, the noise-weighted data Gram matrix, and the data right-hand side. Later solves only add the current source regularization and the weak lens-light regularization, then run the configured Cholesky or non-negative solver. Lens-light intensities stay solved amplitudes. If mass or a lens-light basis is still dynamic, that stored design triple is absent and the regularization cache from the fixed grid remains. The sampling guide and the fixed-grid technical note describe both tiers and the requirement to rebuild the model and its compiled likelihood.

**Blocked by:** 02: 缓存固定网格上的源正则化

**Status:** ready-for-agent

- [x] With a fixed bbox and every lens-mass and lens-light component fixed, evidence and `forward_model` still succeed after design-matrix construction is replaced. Empty mass and lens-light lists count as fixed. Sub-pixel ray tracing matches a cache-cleared evaluation.
- [x] Each evaluation adds `lambda * R` on the source block and the existing weak lens-light regularization on the lens-light intensities. The stored Gram matrix remains the data term. The Cholesky factor is not reused for a different regularization strength.
- [x] Cached and cache-cleared evidence, model image, and source pixels agree at `rtol=1e-6` and `atol=1e-6` for both solvers, including a joint semi-linear inversion. Two JIT/`vmap` proposals with different `log_lambda_reg` produce different sources.
- [x] If any lens-mass component or any lens-light basis is dynamic, none of the design matrix, data Gram matrix, or right-hand side is stored. A pointer-driven mass parameter counts as dynamic. The fixed-grid regularization cache still applies, and design-matrix construction still runs.
- [x] A direct design-matrix call on the simulator still follows the current mass and the bbox it is given. The position-likelihood penalty and the non-finite evidence penalty are unchanged. Existing pixelized demo pipelines are left unchanged.
- [x] The sampling guide, beside the operator cache, and the fixed-grid technical note describe the fixed dense bbox, both cache tiers, the fixedness rule, and the need to rebuild the evidence model and its compiled likelihood after a fixed value, parameter mode, observation, or bbox changes.
