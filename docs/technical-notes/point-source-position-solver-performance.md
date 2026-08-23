# Point-source image-position solver: performance study

## Scope and conclusion

This note investigates the time to solve lensed point-image positions using
the exact optimization configuration in
[`examples/point_source/sim_data.py`](../../examples/point_source/sim_data.py):
an SIE plus external shear, a `200 x 200` coarse grid, `k_keep=30`, and 20
Newton iterations.  Measurements were made on an NVIDIA GeForce RTX 5080 with
JAX on the CUDA backend, float32, and
`XLA_PYTHON_CLIENT_PREALLOCATE=false`.

There was substantial acceleration headroom for the *position-returning API*.
Before this implementation, `solve_image_positions()` called the dynamic-length
public solver and returned **428.5 ms mean** over ten warm steady-state calls
(median 418.6 ms). The same candidate-generation calculation plus fixed-size
deduplication, placed under one `jax.jit`, took **0.745 ms mean** (100 calls,
median 0.743 ms). It returned the same four images to within `9.54e-7` arcsec.

The refactored `solve_image_positions()` now delegates to that fixed-shape JIT
path and only trims its result at the API boundary. It measured **0.923 ms
mean** over 30 warm calls (median 0.899 ms), a roughly **464x** reduction from
the pre-refactor dynamic path. This is not a claim that Newton/root evaluation
itself became 464x faster: the old dominant cost was eager Python/JAX dispatch
and synchronization rather than GPU numerical work. The inference path is
already structured correctly: [`PointSourceProbModel.__call__`](../../TinyLensGpu/ObservationModel/LensImage/point_source_model.py#L318-L396)
is outer-JITed and uses the fixed-size
[`select_unique_images_fixed`](../../TinyLensGpu/utils/lensing/point_source_solver.py#L455-L535)
selector.  With four observed positions, that full jitted likelihood measured
0.742 ms mean under the same numerical configuration (the likelihood value is
irrelevant to this timing).

## Current execution paths

`solve_image_positions()` calls
`solve_lens_equation_optimization()`, which then calls `post_process_images()`.
The latter uses a Boolean index (`sorted_images[final_mask]`) and so produces
a data-dependent leading dimension.  Such an output cannot be the result of a
single normally-JITed JAX computation.

By contrast, the likelihood uses
`solve_lens_equation_optimization_core()` and retains static shapes through
`select_unique_images_fixed()`: it returns a padded `(n_select, 2)` array, a
Boolean validity mask, and a count.  This lets XLA compile and fuse the entire
search/refinement/selection computation.

The distinction is consistent with JAX's guidance to JIT the outermost
function, keep array shapes stable, and synchronize before timing an
asynchronous GPU result.  See the official [benchmarking guide](https://docs.jax.dev/en/latest/benchmarking.html),
[asynchronous-dispatch documentation](https://docs.jax.dev/en/latest/async_dispatch.html),
and [JIT-compilation guide](https://docs.jax.dev/en/latest/jit-compilation.html).

## Measurements

Every timed call was followed by `jax.block_until_ready(output)`.  One warm-up
was excluded from steady-state timing.  The first ever dynamic call took about
2.46 s because it includes compilation; it should not be confused with the
steady solver rate.

| Path / configuration | Steady-state result |
| --- | ---: |
| Pre-refactor `solve_image_positions()`, 200x200 / 30 / 20 | 428.5 ms mean (10 calls) |
| Current `solve_image_positions()`, 200x200 / 30 / 20 | 0.923 ms mean (30 calls) |
| Current `solve_image_positions_fixed()`, 200x200 / 30 / 20 | 0.757 ms (warm call) |
| JIT `PointSourceProbModel.__call__`, 200x200 / 30 / 20 | 0.742 ms mean (20 calls) |
| JIT fixed result, 200x200 / 10 / 20 | 0.615 ms mean (100 calls) |
| JIT fixed result, 200x200 / 30 / 10 | 0.507 ms mean (100 calls) |

Changing only `n_x=n_y` from 100 to 200 did not materially change the
measured fixed-JIT runtime in this tiny model (0.740 vs 0.747 ms).  Therefore
reducing the coarse grid is not the first optimization to make.  Reducing
candidate count or Newton iterations did help, but these are numerical
accuracy/completeness trade-offs.  In this symmetric demo both reduced
settings still found four images and had maximum source-plane residual below
`1.4e-7`; this must be checked over representative lens populations before
adopting a new default.

## Implemented design

The implementation adds a separate public fixed-shape API while retaining the
compatibility semantics of `solve_image_positions()`:

```python
images, dists, valid_mask, count = solver.solve_image_positions_fixed()
```

It is an outer-JITed function that:

1. runs `solve_lens_equation_optimization_core()` (or the AMR core);
2. calls the fixed selector with `n_select=k_keep`;
3. returns padded images and residuals, a mask, and a count.

For the simulation script, copy only `images[valid_mask]` to NumPy after the
one device synchronization.  The static `max_images` must be part of the
compiled specialization, but continuous mass and source parameters must
remain dynamic JAX arguments.  The fixed-shape approach follows JAX's
[cache-miss guidance](https://docs.jax.dev/en/latest/debugging/slow_tracing_compilation.html).

The existing dynamic method remains for callers which need a host-side
variable-length array and trims the fixed result only at its API boundary.

## Secondary opportunities

- **Batch independent solves only when the workload has a batch axis.** A
  sampler evaluating many independent lens/source states can use `vmap` over a
  fixed-shape jitted solver.  This may increase GPU occupancy and reduce cost
  per solve; it is not expected to materially improve a single system.  See
  JAX's [automatic-vectorization guide](https://docs.jax.dev/en/latest/automatic-vectorization.html).
- **Profile before changing numerical algorithms.** Capture a warmed trace
  after introducing the fixed API, then inspect GPU kernels and host gaps.
  JAX documents the Perfetto/XProf workflow in its
  [profiling guide](https://docs.jax.dev/en/latest/profiling.html).
- **Use a persistent compilation cache for short-lived scripts.** This can
  reduce the roughly 0.5--2.5 s cold compilation cost but cannot reduce the
  0.745 ms warmed runtime.  See the official
  [persistent compilation cache guide](https://docs.jax.dev/en/latest/persistent_compilation_cache.html).

## Regression coverage

The point-source tests compare masks, sorted valid positions, and residuals
between the fixed and dynamic APIs for both `optimization` and `amr`. They
also cover the likelihood's extra-root semantics: observed positions are
matched to the best one-to-one subset of all valid roots, while extra roots
are not penalized. Future changes should preserve these cases, distinguish
warmed from cold timing, and synchronize every timed GPU result.
