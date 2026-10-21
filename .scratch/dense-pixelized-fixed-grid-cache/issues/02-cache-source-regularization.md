# 02: 缓存固定网格上的源正则化

**What to build:** Once the dense evidence model has a fixed source grid, a lens modeler can sample mass or regularization strength without rebuilding an invariant source regularization matrix. Finite-difference regularization and a Gaussian-process prior with a fixed kernel scale are stored at construction, including the log determinant the evidence uses. Regularization strength stays outside that stored matrix. A kernel scale that is still being sampled, or a helper call for a different bbox, is computed for that call and does not replace the stored grid.

**Blocked by:** 01: 固定稠密模型的 source bbox

**Status:** ready-for-agent

- [ ] For zero-order, first-order, and second-order regularization on a fixed grid, construction stores the dense matrix and the evidence log determinant. A later evidence call still succeeds after regularization construction is replaced, and it does not compute `slogdet` again.
- [ ] The regularization helper keeps returning no covariance log determinant for finite differences. On a cache hit it returns the stored matrix. A different bbox is built immediately and leaves the stored matrix and determinant in place.
- [ ] A static Gaussian-process kernel scale stores the precision matrix and `log|K|`, and the helper returns both. A dynamic kernel scale, including one reached through a pointer, rebuilds both on the fixed bbox at each evaluation.
- [ ] Fixedness uses the operator rule: transitive dependencies, with a pointer counting as fixed only when it has children.
- [ ] Changing `log_lambda_reg` still changes the evidence and the reconstructed source. Cached and cache-cleared evidence, model image, and source pixels agree at `rtol=1e-6` and `atol=1e-6` for both the Cholesky solver and the non-negative solver.
- [ ] While lens mass is dynamic, regularization construction is not repeated and design-matrix construction still is. `forward_model` uses the same stored regularization as the evidence. Adaptive source regularization remains rejected, and existing pixelized demo pipelines are left unchanged.
