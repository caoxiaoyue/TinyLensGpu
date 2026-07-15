# Joint pixel-source + MGE operator backend: GPU-memory study

## Scope and conclusion

This note examines the joint pixelized-source and lens-light path used by
[`fit_joint.py`](../../examples/pix_src_demo_operator/simple/pix_src_lens_light_mge/fit_joint.py)
at commit `2d6c6ee` (2026-07-14). The concrete configuration is a `100 x 100`
image, `N_d = 5024` active image pixels, an `80 x 80` pixelized source
(`N_s = 6400`), 20 MGE lens-light components (`N_l = 20`), `nsub = 4`, and a
vectorized likelihood chunk of 50 samples.

The user's intuition is substantially correct: the final lens-light design
matrix is dense, and trying to make that matrix sparse is not the useful
optimization. Each Gaussian is nonzero everywhere in exact arithmetic, as is
clear from the profile implementation
([`GaussianEllipse.light`](../../TinyLensGpu/PhysicalModel/LensImage/Parametric/Light/gaussian.py#L67-L115)),
and the final dense matrix is only `5024 x 20 = 100480` float32 values, or
**0.383 MiB per likelihood sample**.

There is nevertheless meaningful memory headroom. It comes mainly from:

1. the `50`-sample vectorized chunk multiplying all sample-dependent
   intermediates;
2. constructing all 20 MGE bases together on the `400 x 400` subgrid before
   binning, a nominal **12.207 MiB per sample**;
3. retaining the source block preconditioner, **2.441 MiB per sample**; and
4. the sparse source mapping and joint source--lens workspaces, rather than the
   final lens design matrix itself.

The measured likelihood evaluation is not close to exhausting the 16 GiB
device: its fresh-process peak is **1.305 GiB** for a 50-sample chunk. (This is
not a claim about every allocation in the complete Nautilus process.)
Optimization is most valuable for larger images, larger `nsub`, more
simultaneous models, or for turning the current memory allowance into a
larger/faster sampling chunk.

## What the joint system stores

Write the image model as

\[
    m = F s + L a,
\]

where `F` maps source pixels to the PSF-convolved image and `L` contains the
PSF-convolved lens-light bases. The joint curvature has the block form

\[
  A = \begin{bmatrix}
      F^T W F + \lambda R & C \\
      C^T                 & G
  \end{bmatrix},
  \qquad
  C = F^T W L,\quad G = L^T W L + \epsilon I.
\]

These objects have different memory behavior and should not be conflated.

| Object | Current representation and shape | float32/int32 payload per sample | Interpretation |
|---|---:|---:|---|
| Lens design matrix `L` | dense `(N_d, N_l) = (5024, 20)` | **0.383 MiB** | Dense but thin; sparsifying it is not worthwhile. |
| MGE subgrid bases during construction | dense `(400, 400, 20)` | **12.207 MiB nominal** | A transient before binning and PSF convolution; much larger than final `L`. |
| Source mapping weights | four bilinear weights per subpixel, `(80384, 4)` | **1.227 MiB** | Sparse/local source representation. |
| Source mapping indices | four int32 indices per subpixel, `(80384, 4)` | **1.227 MiB** | Sparse/local source representation. |
| Dense source design, if formed | `(N_d, N_s) = (5024, 6400)` | **122.656 MiB** | Avoided by the operator backend; 50 samples alone would require about 5.99 GiB. |
| Joint cross block `C` | dense `(N_s, N_l) = (6400, 20)` | **0.488 MiB** | Generally dense after lensing, PSF convolution, and weighting, but still thin. |
| Lens Gram/Schur factor | dense `(20, 20)` | **0.0015 MiB** | Negligible. Dense linear algebra is appropriate here. |
| Source block Cholesky factors | `(64, 100, 100)` for `block_size=10` | **2.441 MiB** | Retained for PCG, PNPG scaling, and the approximate determinant. |
| PNPG/PCG vectors | length `N_s + N_l = 6420` | about **0.0245 MiB each** | Loop-carried state is bounded; iteration count does not create a 1000-iterate history. |

The source operator data are built once per likelihood evaluation and reused by
the iterative matvecs
([`LensOperatorData` and the joint matvec](../../TinyLensGpu/ForwardSimulation/LensImage/pixelized_operator.py#L44-L126)).
The bilinear forward operation gathers only four source neighbors per subpixel
([`_apply_L_jit`](../../TinyLensGpu/ForwardSimulation/LensImage/pixelized_operator.py#L142-L165)).
That is the major sparsity win: weights plus indices use **2.453 MiB**, roughly
50 times less payload than the final dense source design matrix.

By contrast, lens light is explicitly built as a stack of component images,
binned, convolved component-by-component with `vmap`, and finally sliced down
to `(N_d, N_l)`
([`build_lens_light_matrix`](../../TinyLensGpu/ForwardSimulation/LensImage/pixelized_operator.py#L596-L621)).
The important distinction is therefore **dense final matrix versus dense
construction transient**.

The block-Schur preconditioner stores source Cholesky blocks, the dense cross
block, and a tiny dense Schur factor
([`BlockSchurPreconditioner`](../../TinyLensGpu/utils/cg_solver.py#L23-L29)).
Its construction computes `C`, `G`, `P_s^{-1} C`, and the Schur complement
([`_build_block_schur_preconditioner`](../../TinyLensGpu/ObservationModel/LensImage/pixelized_image_model_operator.py#L434-L502)).
These are the right places to use dense algebra because `N_l = 20`; replacing
the `20 x 20` operations with sparse kernels would add complexity without a
material payload reduction.

## Reproducible measurements

### Environment and method

- GPU: NVIDIA GeForce RTX 5080, 16303 MiB
- driver: 595.71.05
- JAX/jaxlib: 0.7.1, CUDA device backend, float64 disabled
- allocation setting: `XLA_PYTHON_CLIENT_PREALLOCATE=false`
- model input: the setup prefix of `fit_joint.py`, evaluated at its truth/default
  parameter vector

For compiler-planned temporary memory, the probe obtained the jitted batched
likelihood from the wrapper closure and ran:

```python
compiled = batch_loglike.lower(theta_batch).compile()
temporary_bytes = compiled.memory_analysis().temp_size_in_bytes
```

For allocator-observed memory, every batch size ran in a separate fresh Python
process. After `jax.block_until_ready(output)`, the probe read
`jax.devices()[0].memory_stats()`, including `peak_bytes_in_use`,
`bytes_in_use`, and `largest_alloc_size`. Thus `memory_analysis()` below is the
compiler's buffer-plan estimate, while `peak_bytes_in_use` is the allocator
high-water mark for that fresh process. They measure different things and
should not be added together. The official JAX memory-analysis example likewise
defines total compiled memory from temporary, argument, output, and alias sizes
([JAX host-offloading notebook](https://docs.jax.dev/en/latest/notebooks/host-offloading.html#device-memory-analysis)).

### Chunk scaling

| Joint likelihood batch/chunk | Compiler temporary | Fresh-process device peak |
|---:|---:|---:|
| 1 | not recorded in this probe | 704,601,344 B = **671.96 MiB** |
| 10 | 169,988,568 B = **162.11 MiB** | 727,717,120 B = **694.01 MiB** |
| 50 (current) | 848,025,568 B = **808.74 MiB** | 1,401,004,800 B = **1336.10 MiB** |

For batch 50, `largest_alloc_size` was 848,025,600 B, essentially the same as
the compiler temporary plan. The non-scaling part of the device peak includes
resident program, constants, library/runtime state, and allocator effects.
The compiler temporary grows almost exactly fivefold from chunk 10 to chunk 50,
showing that vectorized sample-dependent state is the dominant scaling axis.

This is consistent with the code: `make_likelihood(...,
vectorized_chunk_size=50)` implements a `lax.map` whose body is vectorized in
bounded chunks
([`build_likelihood.py`](../../TinyLensGpu/Inference/build_likelihood.py#L92-L104)).
JAX documents `lax.map(batch_size=...)` specifically as a memory-efficient
alternative to a full `vmap`, with the selected batch executed using `vmap`
([official `jax.lax.map` documentation](https://docs.jax.dev/en/latest/_autosummary/jax.lax.map.html)).

### Controlled batch-50 comparisons

These comparisons change one model/configuration axis at a time, but XLA's
fusion and buffer reuse mean the differences are whole-program effects, not an
additive attribution ledger.

| Batch-50 configuration | Compiler temporary | Change from current |
|---|---:|---:|
| Current: joint, `N_l=20`, `nsub=4`, block 10 | **808.74 MiB** | baseline |
| Source-only, `N_l=0` | **544.74 MiB** | -264.00 MiB (-32.6%) |
| Joint, `N_l=10` | **777.24 MiB** | -31.50 MiB (-3.9%) |
| Joint, `nsub=1` | **431.43 MiB** | -377.31 MiB (-46.7%) |
| Joint, `block_size=5` | **658.39 MiB** | -150.35 MiB (-18.6%) |

Three conclusions follow.

First, the joint lens path costs much more than the 19.17 MiB needed to retain
50 final lens matrices: basis construction, cross/Schur construction, and joint
matvec workspaces account for the rest. Second, halving the number of Gaussian
columns from 20 to 10 saves only 3.9% in this compiled program; removing lens
light entirely also removes fixed joint-path work, so its 32.6% difference must
not be divided by 20 and assigned to each Gaussian. Third, the quadratic
`nsub` dependence and the batched source preconditioner are quantitatively
important.

## Recommendations, ordered by expected value

| Priority | Change | Expected memory benefit | Numerical/scientific risk |
|---:|---|---|---|
| 1 | Make `vectorized_chunk_size` a memory/throughput tuning knob; use 10--25 when memory is constrained. | Direct and nearly linear; measured 808.74 MiB at 50 versus 162.11 MiB at 10 for compiler temporaries. | None to the model; possible throughput loss, so benchmark wall time. |
| 2 | Build `L` once per sample and explicitly thread it through preconditioner construction, the joint system, and reconstruction. | Removes source-level duplicate construction and gives explicit lifetime control. | Low; identical values if implemented carefully. Actual gain depends on current XLA CSE. |
| 3 | Stream/fuse MGE basis construction over components instead of materializing `(400,400,20)` before binning. | Potentially replaces a 12.207 MiB/sample stack by one/few component workspaces plus the 0.383 MiB final matrix. | Low-to-moderate engineering risk; validate exact image parity and profile XLA output. |
| 4 | Precompute exact dense `C` and `G` once, then use them in every joint PNPG matvec. | Reduces repeated image-space and FFT workspaces; likely a larger speed win than a retained-memory win. | Moderate; preserve an undamped exact `C` for the physical operator and use damping only in the preconditioner. Revalidate float32 solver/evidence parity. |
| 5 | Compress the bilinear source mapping to base index + fractional coordinates + validity, reconstructing four neighbors/weights in kernels. | Current weights+indices are 2.453 MiB/sample; a compact representation can save roughly 1--1.5 MiB/sample before compiler reuse. | Moderate implementation risk; boundary and adjoint tests are essential. |
| 6 | Evaluate a smaller source preconditioner block, for example 5 instead of 10. | Measured compiler temporary reduction of 18.6% at batch 50. | Medium-to-high: weaker PCG/PNPG preconditioning and a different approximate logdet can alter runtime and evidence. Requires end-to-end validation. |
| 7 | Decouple lens-light integration from source `nsub`, preferably component-aware. | Large ceiling: global `nsub=1` reduced compiler temporary by 46.7%. | High scientific risk for the current basis: its smallest Gaussian has `sigma=0.01` arcsec while image pixels are 0.05 arcsec. Do not simply set global `nsub=1`. |

### 1. Chunk size is the safest control

The current 50-sample chunk is reasonable on this 16 GiB GPU, but it is not a
memory minimum. Reducing Nautilus `n_batch` below the chunk can also reduce the
effective `vmap`, but it couples memory tuning to the sampler's submission
strategy. `vectorized_chunk_size` is the direct device-work bound while keeping
the current `n_batch=100`. Conversely, because the measured peak is only 1.305
GiB, a larger chunk may improve throughput on this specific device if
compilation and runtime benchmarks support it.

`XLA_PYTHON_CLIENT_PREALLOCATE=false` prevents JAX from reserving most GPU
memory at startup, but it does not reduce the buffers required by the compiled
calculation. JAX's official GPU allocation guide distinguishes preallocation
policy from actual program memory and warns that disabling preallocation can
increase fragmentation
([JAX GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)).

### 2. Construct and pass the lens matrix once

At the Python source level, the log-evidence path requests the same lens matrix
at least three times:

1. for the block-Schur preconditioner
   ([`_log_evidence`](../../TinyLensGpu/ObservationModel/LensImage/pixelized_image_model_operator.py#L688-L696));
2. inside `build_joint_system`
   ([`build_joint_system`](../../TinyLensGpu/ForwardSimulation/LensImage/pixelized_operator.py#L623-L657)); and
3. during component reconstruction, because no existing `lens_matrix` is
   passed
   ([`reconstruct_components`](../../TinyLensGpu/ForwardSimulation/LensImage/pixelized_operator.py#L659-L672)).

The compiler may common-subexpression-eliminate some or all identical pure
calls, so source counting does **not** prove three live allocations. Still,
constructing once and passing `L` explicitly makes reuse and lifetime
intentional and testable rather than dependent on a particular XLA version.
The optimized HLO and `memory_analysis()` should be compared before and after.

### 3. Stream the MGE construction, not the final dense matrix

The present implementation stacks all component images on the subgrid before
binning. A component `lax.scan`/`lax.map` can instead:

1. evaluate one Gaussian (or a small component tile) at `400 x 400`;
2. bin it to `100 x 100`;
3. convolve it with the PSF; and
4. write the `5024` active values into one output column.

This retains the appropriate dense `(5024,20)` output but bounds the large
subgrid workspace by component tile size. Care is needed with nesting order:
the outer 50-sample vectorization will still multiply a one-component
`400 x 400` workspace by 50. JAX's `lax.map` documentation supports using an
element-wise or bounded batch for reduced memory
([official documentation](https://docs.jax.dev/en/latest/_autosummary/jax.lax.map.html)).

All 20 Gaussians share center and ellipticity in this example. A specialized
MGE kernel can compute the transformed elliptical radius once per sample and
evaluate the fixed sigma sequence from it. That is primarily a compute
optimization; streaming the resulting exponentials is what controls memory.

### 4. Use dense sufficient statistics in the repeated joint matvec

The current joint matvec computes `C a` and `C^T s` indirectly on every
iteration: it forms image-space lens/source vectors and applies the source
forward/adjoint operators
([`_joint_curvature_kernel`](../../TinyLensGpu/ForwardSimulation/LensImage/pixelized_operator.py#L127-L158)).
But preconditioner setup already computes the exact cross block and lens Gram.
Keeping exact `C` and `G` allows the repeated operation to be written as

```text
source_out = source_curvature(s) + C @ a
lens_out   = C.T @ s + G @ a
```

This is not an attempt to make lens light sparse. It exploits the fact that the
lens subspace is **small and dense**. It should remove one extra source forward
and one extra source adjoint from each joint matvec, as well as their
image-space temporaries. The existing cross block is spectrally scaled for the
preconditioner; the physical operator must instead use the undamped exact
cross. A clean representation would retain exact `C` plus a scalar
preconditioner scale, avoiding a second full cross allocation.

This change is especially attractive for wall time because PNPG performs a
fixed 1000-iteration scan
([`pnpg_nnls_solve`](../../TinyLensGpu/utils/pnpg_solver.py#L156-L249)). It must
be validated against the current matvec over random vectors and against the
end-to-end likelihood because forming `C` once changes float32 operation
association even when the algebra is exact.

### 5. The remaining source sparsity can be compressed further

The four neighbor indices are deterministic from one clipped lower-left source
index, and the four weights are deterministic from two fractional coordinates
plus validity
([`lens_mapping_operator_bilinear_from`](../../TinyLensGpu/utils/lensing/mapping.py#L41-L83)).
A compact record such as `(base_index, fx, fy, valid)` can reconstruct the four
indices and weights inside forward/adjoint kernels. This targets the part where
sparsity genuinely exists and could save tens of MiB at chunk 50, at the cost
of extra integer/floating-point work each PNPG iteration.

### 6. Treat block size as a numerical choice, not merely memory tuning

For a divisible `n x n` source grid and spatial block width `k`, the stored
Cholesky payload scales as approximately `N_s k^2`; hence the current `k=10`
uses 640,000 floats while `k=5` uses 160,000. The measured whole-program saving
is smaller than this 75% local reduction because other workspaces remain.

More importantly, the same factors determine both the PCG/PNPG preconditioner
and the operator evidence's approximate log determinant
([`_log_evidence`](../../TinyLensGpu/ObservationModel/LensImage/pixelized_image_model_operator.py#L737-L747)).
Changing block size can therefore change convergence and the sampled evidence,
not just memory. It needs a representative-point KKT comparison, likelihood
comparison, and full posterior/runtime validation.

## Changes not recommended as first-line optimizations

- **Sparse `L`, `C`, or the `20 x 20` lens block.** MGE columns are dense in
  exact arithmetic, `L` and `C` are already thin, and the tiny lens block is not
  a memory concern. Approximate tail truncation would introduce a new modeling
  tolerance and irregular sparse GPU kernels.
- **Recompute `L` in every PNPG matvec.** Retaining 0.383 MiB per sample is far
  cheaper than repeating 20 subgrid evaluations and PSF convolutions 1000
  times.
- **float16/bfloat16 for solver curvature.** The joint 20-MGE system already
  required float32 equilibration and stabilization. Lower precision is poorly
  aligned with the numerical-stability goal.
- **Global `nsub=1`.** Although the memory result is large, it is a different
  forward model. The MGE sigma sequence starts at 0.01 arcsec
  ([knot generation in the example](../../examples/pix_src_demo_operator/simple/pix_src_lens_light_mge/fit_joint.py#L121-L124)),
  so the narrowest basis is substantially under-resolved by 0.05-arcsec native
  pixels.
- **`jax.remat` as a generic fix.** The likelihood is not being differentiated;
  JAX documents checkpoint/rematerialization as controlling intermediates saved
  for automatic differentiation
  ([`jax.checkpoint`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html)).
  It is not the primary tool for this forward iterative solve.
- **Buffer donation of the sampler input.** Donation only reduces memory when
  an input buffer can back a same-shape/type output
  ([JAX buffer donation](https://docs.jax.dev/en/latest/buffer_donation.html)).
  Here the batch input is only `100 x 10` float32 values and the output is 100
  scalars, so it cannot address the large internal workspaces.
- **Host-offloading repeatedly used operator/preconditioner arrays.** These
  arrays are consumed on every iterative matvec; transfers would likely trade
  modest resident memory for substantial latency. Consider it only after
  profiling a much larger model that does not fit even with a smaller chunk.

## Suggested implementation/validation sequence

No code was changed as part of this research. If memory optimization is taken
forward, the least risky sequence is:

1. benchmark chunk sizes 10, 25, and 50 for both peak memory and likelihoods per
   second;
2. thread one explicitly constructed `L` through the complete likelihood path
   and compare optimized HLO memory plus exact output parity;
3. stream the 20 MGE columns and require image/lens-matrix parity before timing;
4. introduce exact `C`/`G` joint matvecs and require random-vector matvec parity,
   PNPG KKT parity, likelihood parity, and a shortened sampler validation;
5. only then prototype compact bilinear mapping or smaller preconditioner
   blocks.

For every variant, collect both `Compiled.memory_analysis()` and fresh-process
`Device.memory_stats()`: the former explains the compiled buffer plan, while
the latter captures the complete runtime high-water mark. Continue to record
`XLA_PYTHON_CLIENT_PREALLOCATE`, because JAX allocation policy materially
changes what process-level GPU monitors show even when the program's logical
arrays are unchanged.
