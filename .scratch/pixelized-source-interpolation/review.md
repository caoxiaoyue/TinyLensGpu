# Pixelized source interpolation review

The independent reviews examined `git diff dbb914a...b443388` on
`codex/pixelized-source-interpolation`. The findings were addressed in
`ee1c67d`, with refreshed numerical records in `38e37fd`. [Specification](spec.md)

## Standards

- **P2, partition reuse.** Two macro models could share image-plane geometry
  and trace different source-plane rays. A change in the Einstein radius
  from 0.10 to 0.35 arcsec changed the newly voted blocks while the second
  renderer automatically reused the first partition. The reuse check now
  includes traced rays and the incidence gradients, with a checksum-pinned
  exception for the documented legacy paired calibration. [Partition selection](../../examples/potential_correction/_scientific_partitions.py)
- **P2, shared exporter scope.** The shared export helper initially added
  interpolation fields for every caller, including the excluded B1938
  workflow. The migrated callers now opt in to the new fields. The shared
  diagnostics regression expects those fields for J0946 and uses the legacy
  key set for B1938. [Source export](../../TinyLensGpu/example_support/source.py)
- **P3, verification commands.** The joint and J0946 verification record now
  gives the scoped pytest node IDs. The evidence verification record gives
  the CUDA command. [Joint and J0946 ticket](issues/05-joint-subhalo-source-interpolation.md) · [Evidence ticket](issues/09-evidence-and-gpu-verification.md)
- **P3, duplicated checkpoint policy.** The J0946 macro producers now share
  their manifest, sampler, and stage interpolation checks. [Shared checkpoint policy](../../examples/potential_correction/j0946/_macro_provenance.py)

## Spec

- **P2, excluded workflow.** The spec excludes “B1938+666 workflow changes,
  modeling, replays, and validation.” The shared exporter and its mixed
  regression required the schema correction recorded in Standards. [Out of scope](spec.md#out-of-scope)

Standards reported four findings, with two at P2. Spec reported one P2 finding.

## Support-edge precision

An additional Float64 check at `n=11` reproduced a one-ULP support-edge
shift on CPU and CUDA when JIT replaced division by multiplication with a
rounded reciprocal. The compiled mapping and the frozen callback now use
the same grid spacing. The regression checks both outer edges and their
nearest representable neighbors. [Mapping checks](../../tests/test_pixelized_source_interpolation.py) · [Callback checks](../../tests/test_potential_correction_dense.py)

The historical J0946 archive replay is retired. Maintained support-edge checks
are the mapping and callback regressions linked above; GPU measurements retain
their original configuration. [Evidence and GPU costs](../../docs/validation/pixelized-source-interpolation.md)

## Integrated verification

The runs used `tinylens_gpu`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, and
`OPENBLAS_NUM_THREADS=1`. Core, example, and benchmark checks ran in separate
processes. The CUDA core command from `tests/` at `158c87a` passed 575 tests,
with 3 slow cases deselected.

```bash
JAX_PLATFORMS=cuda pytest -q -m 'not slow' \
  test_pixelized_source_{model,utils,interpolation,precision}.py \
  test_pixelized_inversion.py test_pixelized_operator.py \
  test_operator_source_interpolation.py test_operator_autodiff.py \
  test_solver_autodiff.py test_image_template_light.py \
  test_pixelized_interpolation_evidence.py test_potential_correction_*.py
```

At `ee1c67d`, the CUDA commands below passed 33 mapping, solver and derivative
checks, 4 callback checks, and 6 evidence checks.

```bash
JAX_PLATFORMS=cuda pytest -xq test_pixelized_source_interpolation.py \
  test_operator_source_interpolation.py test_operator_autodiff.py
JAX_PLATFORMS=cuda pytest -xq \
  test_potential_correction_dense.py::test_single_step_uses_frozen_callback_inside_selected_support \
  test_potential_correction_dense.py::test_frozen_hermite_callback_excludes_exact_float64_support_edges
JAX_PLATFORMS=cuda pytest -xq test_pixelized_interpolation_evidence.py
```

The scientific example command from `examples/tests/` passed 29 tests at
`ee1c67d`.

```bash
JAX_PLATFORMS=cpu MPLBACKEND=Agg pytest -q test_potential_correction_calibration.py \
  test_potential_correction_scientific.py test_potential_correction_iterative_scientific.py \
  test_potential_correction_operator_example.py test_potential_correction_iterative_example.py
```

The staged, export, and plotting example command from `examples/tests/` passed
301 tests at `ee1c67d`. The selection expands only the applicable node IDs
from the two mixed modules.

```bash
export JAX_PLATFORMS=cpu MPLBACKEND=Agg
mixed=$(pytest --collect-only -q -o addopts= test_joint_shared_support.py test_operator_diagnostics_plotting.py)
mapfile -t nodes < <(printf '%s\n' "$mixed" | awk '/test_.*::/ && tolower($0) !~ /b1938/ {sub(/^examples\/tests\//, ""); print}')
pytest -q test_basic_shared_support.py test_dense_shared_support.py test_survey_shared_support.py \
  test_pipeline_source_export.py test_source_compatibility.py test_pixelized_support.py \
  test_adaptive_demo_s0_package.py test_joint_fit_plotting.py test_j0946_single_step_workflow.py \
  "${nodes[@]}"
```
