# J0946 fixed-state source interpolation replay

The replay uses the M3 handoff at retained sample 21342 and the archived Hermite reconstruction `cubic_source_v1/trials/0023`, profile `single_step_lens`, repeat 2. The M3 handoff omits a source mode and resolves to hard-box bilinear. Trial 0023 records `cubic` with fixed zero exterior nodes and uses hard-box bilinear only for its preconditioner reference. The replay takes its source mode from the trial record and its fixed macro model from the handoff. [Input checksums and raw results](../../.scratch/pixelized-source-interpolation/experiments/j0946-hermite-replay.json)

The trial supplies 10,000 Float64 source intensities on the original 100 × 100 fitted grid and 1,134 Float64 potential coefficients, including 802 core nodes. The fitted source bounds are `[-0.5641691088676453, 0.43583089113235474, -0.6536976099014282, 0.3463023602962494]` arcsec. The original Float32 native image geometry is promoted to Float64 before ray tracing, as in the archive. The 201 × 201 image has 0.05 arcsec pixels, a 15 × 15 PSF, 2,293 selected pixels, and `nsub=4`. The potential spacing is 0.1 arcsec.

The replay verified the handoff checksums against the archived protocol and the experimental Hermite source checksum against the archived implementation. Both renderers used the same corrected rays and geometric support for all 36,688 subpixel queries. The image comparison uses the full source-only prediction after subpixel integration and PSF convolution. The replay ran on CPU with the library at `ee1c67d`, JAX 0.7.1, Float64 and highest matmul precision. The acceptance tolerance is 1e−11 absolute for each compared value. [Replay script](../../workspace/iteration/replay_source_interpolation.py) · [Experimental Hermite reference](../../workspace/iteration/source_cubic.py)

| Comparison | Maximum absolute difference | RMS difference |
| --- | ---: | ---: |
| Production versus experimental ray brightness | 8.44e−15 | 4.58e−16 |
| Production versus experimental x derivative | 1.10e−12 | 5.05e−14 |
| Production versus experimental y derivative | 1.60e−12 | 6.65e−14 |
| Production versus experimental full source image | 1.50e−15 | 4.61e−17 |
| Production versus archived full source image | 2.33e−15 | 5.25e−17 |
| Experimental versus archived full source image | 3.61e−15 | 7.22e−17 |

Two controls used the same fixed handoff and their own recorded fields and boundaries. The hard-box control `fixed_boundary_v2/hard/single_step/repeat_02` reproduced its saved source image to 3.89e−15 maximum absolute error. The zero-ring control `cubic_source_v1/trials/0022` reproduced its saved source image to 3.66e−15 using the experimental one-cell zero extension. [Control records](../../.scratch/pixelized-source-interpolation/experiments/j0946-boundary-controls.json)

## Replay commands

From the repository root, point `saved` to the `workspace/iteration` directory containing the original M3 and experiment archives.

```bash
source ~/anaconda3/bin/activate && conda activate tinylens_gpu
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1
saved="$PWD/workspace/iteration"

for trial in 0023 0022; do
  python workspace/iteration/replay_source_interpolation.py \
    --macro-dir "$saved/macro" \
    --trial-dir "$saved/cubic_source_v1/trials/$trial" \
    --protocol "$saved/cubic_source_v1/protocol.json"
done
python workspace/iteration/replay_source_interpolation.py \
  --macro-dir "$saved/macro" \
  --trial-dir "$saved/fixed_boundary_v2/hard/single_step/repeat_02" \
  --protocol "$saved/fixed_boundary_v2/protocol.json"
```

## New handoffs

Both macro producer entry points accept `--source-interpolation cubic_hermite` for a fresh output directory. To use the completed handoff, point the corresponding correction driver at that directory with `--macro-dir`. Resuming the macro fit uses the recorded mode when the option is omitted. A conflicting explicit mode requires a fresh run. [Iterative experiment macro producer](../../workspace/iteration/macro_model.py) · [SLACS J0946 workspace macro producer](../../workspace/slacs0946_potential_correction/macro_model.py)

The CUDA example suite passed 43 J0946 cases, including both producers, legacy and Hermite consumers, and the archive boundary selection. The CPU workspace suite passed 13 numerical cases for the iterative handoff, experimental Hermite, and zero-ring workflows.

```bash
# From examples/tests/
JAX_PLATFORMS=cuda MPLBACKEND=Agg pytest -xq test_j0946_single_step_workflow.py

# From workspace/iteration/tests/
JAX_PLATFORMS=cpu pytest -xq test_j0946_iteration.py \
  test_j0946_cubic_source.py test_j0946_zero_boundary.py
```
