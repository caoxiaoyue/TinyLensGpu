# 03: Save and resume a staged fit with its source interpolation

**What to build:** A representative maintained staged workflow, using the operator SLACS source-template transition, can run with an explicit Hermite choice, save it, and resume with the same forward model. Reusable source-package and checkpoint rules make the workflow a reference for the remaining migrations.

**Blocked by:** 02 — Reconstruct the same Hermite source with the operator backend.

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] New source packages record the actual `source_interpolation` and `source_boundary`, retaining the fitted-node `source_bbox`, axes, and single grid size. The support is derivable from this metadata and the compiled inversion result keeps its existing structure.
- [x] Checkpoint configuration records the interpolation before a source package exists and carries it through the source-template transition to the next stage's model.
- [x] Verify a fresh default run, an explicit Hermite run, a Hermite save-and-resume round trip, and a known legacy checkpoint without the fields that resumes as bilinear.
- [x] A conflicting explicit mode fails before any completed stage or its evidence is reused. A fresh run can explicitly use old node intensities as initialization.
- [x] The guide and a maintained runnable example show how to select Hermite explicitly, how resume chooses the mode, and how to interpret nodal non-negativity with signed weights. Exercise the workflow through its existing configuration, export, and restore entry points.

## Comments

Claimed for implementation on `codex/psi-03-stage-resume`.

Implemented in `a44ea5a`, merged as `f79b030`. The CPU command
`pytest -q test_operator_source_interpolation.py test_operator_autodiff.py test_pixelized_source_interpolation.py`
from `tests/` passed 31 checks, including the actual PCG initial iterate.
From `examples/tests/` in a separate CPU Agg process the following command
passed 122 checks.

```bash
pytest -q test_survey_shared_support.py test_basic_shared_support.py test_pipeline_source_export.py \
  'test_operator_diagnostics_plotting.py::test_subhalo_exports_keep_raw_package_contract_with_unsupported_joint_stats[pix_src_demo_operator/pipe/subhalo_0946/model.py]' \
  test_source_compatibility.py::test_survey_source_export_preserves_public_products \
  test_source_compatibility.py::test_survey_stage_propagates_invalid_plot_and_retains_posterior
```

The final `pytest -q test_survey_shared_support.py -k slacs -x` run passed
8 checks. The reference pipeline tests cover the early M0-only checkpoint,
the S1 transition, legacy restore, and mode conflicts before data loading.
The operator SLACS CLI and guide show fresh initialization from an old S1.
