# 15: Release the adapt-image interface and default Overlap together

**What to build:** The library release exposes reference-image adaptation on both ordinary pixelized backends and uses Overlap by default in operator fits. Maintained callers and the guide use the released configuration, with a clear path to regenerate artifacts for the changed objective.

**Blocked by:** [07 — Migrate SLACS, no-lens-light and Galán workflows](07-migrate-survey-and-source-workflows.md), [08 — Migrate lens-light-subtracted pixelized examples](08-migrate-lens-light-subtracted-examples.md), [09 — Migrate J0946 paired subhalo evidence](09-j0946-paired-subhalo-evidence.md), [10 — Export new J0946 macro handoffs with the actual source scale](10-j0946-macro-handoffs.md), [13 — Qualify Overlap GPU workspace and chunk width](13-overlap-gpu-qualification.md), [14 — Qualify the Euclid M1 through M3 workflow on real VIS data](14-euclid-vis-qualification.md).

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Remove `fixed_reg_template` from both constructors after migrating the in-scope callers. Public input tests confirm acceptance of `adapt_image` and rejection of the removed fixed-template and fixed-scale keywords.
- [x] The omitted operator evidence choice agrees with explicit `overlap` in source-only and joint fits. Explicit `block` reproduces the matched calculation. Apply the qualified Hermite offset and automatic chunk policy.
- [x] Verify the maintained Euclid stage regression and in-scope migration regressions against the released defaults. Check the real VIS artifacts and retained qualification reports against the released objective and rerun affected validation if it changed.
- [x] Update the guide with the reference-image inputs, scoring defaults, coverage and batching behavior, and migration and recovery rules. Link the matched mock, optimizer, GPU and VIS qualification records.
- [x] Run the required core, example and benchmark checks in separate processes under the repository rules. Use targeted in-scope regressions and apply the B1938+666 scope exclusion.

## Comments

- Claimed on `codex/overlap-15-release`.
- Released in `da29187`, with CUDA comparison assertions corrected in `9e972de`. The [guide](../../../docs/guides/guide.md) documents both public constructors, Overlap and Block, coverage and batching, and artifact recovery.
- The broad CUDA core run reported 1,025 passed, 1 failed and 5 deselected. Its single joint default-versus-explicit comparison passed after correcting the repeated-solve tolerance. The broad CUDA example run reported 250 passed and 14 failed. All 22 cases in the focused rerun passed, including the 14 corrected assertions. The example selection excludes B1938+666 and the global source-discovery checks.
- The separate CUDA benchmark run reported 14 passed and 1 failed. The exact parametric component-count timing node passed on the idle device.
- The release VIS resume omitted the evidence method and tile flags. It retained explicit chunk width 5 and seed 73. Fresh diagnostics reproduced every saved reporting-point score, and the stage pickle hashes matched. Four released mock GPU replay checks agreed within `1.55e-11` and confirmed the measured workspace. See the [mock replay](../../../docs/validation/adaptive-image-overlap-euclid-mock.md), [optimizer qualification](../../../docs/validation/adaptive-image-overlap-m3-optimizer.md), [GPU qualification](../../../docs/validation/adaptive-image-overlap-gpu.md), and [real VIS qualification](../../../docs/validation/adaptive-image-overlap-euclid-vis.md).

The release checks used a checkout of the release branch on the RTX 5080.
Each pytest command ran in a separate process from the directory shown.
The run logs are under `/tmp/tinylens-overlap.EIfsZz/15-*.log`.

### Environment

```bash
release_checkout="$(git rev-parse --show-toplevel)"
source ~/anaconda3/bin/activate
conda activate tinylens_gpu
export PYTHONPATH="$release_checkout${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

### Core

```bash
cd "$release_checkout/tests"
pytest -q -m 'not slow and not performance'
JAX_PLATFORMS=cuda pytest -q \
  test_overlap_evidence.py::test_joint_operator_default_matches_explicit_overlap
```

### Examples

```bash
cd "$release_checkout/examples/tests"
export JAX_PLATFORMS=cuda MPLBACKEND=Agg
pytest -q \
  test_euclid_adapt_image_stages.py test_pre_subtracted_adapt_image.py \
  test_pipeline_source_export.py test_pixelized_support.py test_survey_shared_support.py \
  test_basic_shared_support.py test_adaptive_demo_s0_package.py \
  test_j0946_single_step_workflow.py test_potential_correction_lbfgs_j0946.py \
  test_joint_shared_support.py::test_j0946_resumes_before_and_after_source_package \
  test_joint_shared_support.py::test_j0946_rejects_incompatible_comparison_caches_before_reuse \
  test_joint_shared_support.py::test_joint_pipeline_renders_actual_components_with_unavailable_dof \
  test_operator_diagnostics_plotting.py::test_joint_pipeline_plots_fitted_components_and_unavailable_statistics \
  test_operator_diagnostics_plotting.py::test_joint_plots_reject_failed_reconstruction_before_output \
  test_operator_diagnostics_plotting.py::test_simple_joint_plots_preserve_per_pixel_and_arc_summaries \
  test_source_compatibility.py::test_survey_plot_renders_public_reconstruction \
  test_source_compatibility.py::test_survey_source_export_preserves_public_products \
  test_source_compatibility.py::test_survey_invalid_reconstruction_stops_output \
  test_source_compatibility.py::test_survey_stage_propagates_invalid_plot_and_retains_posterior

pytest -q \
  test_basic_shared_support.py::test_adaptive_pipeline_resumes_its_saved_source_model \
  test_basic_shared_support.py::test_source_export_to_adaptive_stage_and_plot \
  test_j0946_single_step_workflow.py::test_macro_exports_the_native_reference_and_selected_m3_fields \
  test_joint_shared_support.py::test_j0946_resumes_before_and_after_source_package \
  test_source_compatibility.py::test_survey_plot_renders_public_reconstruction
```

### Pytest benchmarks

```bash
cd "$release_checkout/benchmarks"
JAX_PLATFORMS=cuda pytest -q test_performance.py test_lbfgs_operator_resources.py \
  test_potential_correction_memory.py test_potential_correction_preconditioner.py \
  test_potential_correction_workflow.py
JAX_PLATFORMS=cuda pytest -q -s \
  test_performance.py::TestScalability::test_component_count_scaling
```

### Matched Overlap replay

```bash
cd "$release_checkout/benchmarks"
archive=/home/xycao/workspace/tinylens_dev/TinyLensGpu/workspace/euclid_mock
bench_output="$(mktemp -d /tmp/tinylens-overlap-release-gpu.XXXXXX)"
mkdir -p "$bench_output"
(
    set -e
    export OPENBLAS_NUM_THREADS=1 NVIDIA_TF32_OVERRIDE=0
    export JAX_ENABLE_X64=true JAX_ENABLE_COMPILATION_CACHE=false
    JAX_PLATFORMS=cpu python overlap_evidence_resources.py \
        --archive "$archive" --output "$bench_output" --worker fixture --case joint
    for case in source joint; do
        for phase in gradient batch; do
            output="$bench_output/$case-$phase"
            mkdir -p "$output"
            if [ "$case" = joint ]; then
                ln -s "$bench_output/joint_image.npy" "$output/joint_image.npy"
            fi
            [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || exit 10
            JAX_PLATFORMS=cuda python overlap_evidence_resources.py \
                --archive "$archive" --output "$output" --worker "$phase" --case "$case" \
                --evidence overlap --chunk auto --outer-batch 23 --repeats 1
        done
    done
)
```

### Post-review verification

The first focused CUDA core run reported 80 passed and one failure from
the mixed-precision dense test fixture. Non-power-of-two Float32 noise had
introduced whitening roundoff in that fixture. After using exact power-of-two
noise, the focused three-case rerun passed on CUDA and CPU. The separate
dense and regularization run passed all 54 selected cases (109 deselected).

The focused example run passed all 18 cases. It covers compatible M1 reuse with stale
M2/M3, refits for malformed M1 metadata or a damaged S1 reference, the orphan cache
conflict, the archived Block M0 replot, and both J0946 macro restore paths. The M0
builder already forwarded its archived evidence settings. The review added the replot
regression to exercise that path.

Use the environment above with the review checkout. Run the core commands from
`tests/` and the example command from `examples/tests/` in separate processes.
The run logs are under `/tmp/tinylens-overlap.EIfsZz/review-*.log`.

```bash
cd "$release_checkout/tests"
export JAX_PLATFORMS=cuda
pytest -q test_adapt_image.py test_overlap_evidence.py test_operator_autodiff.py \
  test_pixelized_source_precision.py test_pixelized_source_model.py \
  test_pixelized_interpolation_evidence.py
pytest -q test_pixelized_inversion.py test_regularization.py -k 'evidence or diagnostics or adaptive'
pytest -q \
  test_adapt_image.py::test_dense_invalid_exact_evidence_preserves_reconstruction_and_rejects_gradients \
  test_adapt_image.py::test_adapt_image_uses_mixed_fit_precision
JAX_PLATFORMS=cpu pytest -q \
  test_adapt_image.py::test_dense_invalid_exact_evidence_preserves_reconstruction_and_rejects_gradients \
  test_adapt_image.py::test_adapt_image_uses_mixed_fit_precision
```

```bash
cd "$release_checkout/examples/tests"
JAX_PLATFORMS=cpu MPLBACKEND=Agg pytest -q test_euclid_adapt_image_stages.py \
  'test_survey_shared_support.py::test_survey_rejects_late_stage_conflicts_before_any_restore[euclid_examples/pipeline.py-m3]' \
  test_basic_shared_support.py::test_adaptive_pipeline_resumes_its_saved_source_model \
  test_j0946_single_step_workflow.py::test_macro_resume_preserves_the_recorded_source_interpolation \
  test_j0946_single_step_workflow.py::test_macro_exports_the_native_reference_and_selected_m3_fields \
  test_j0946_single_step_workflow.py::test_macro_resume_rejects_an_incompatible_scored_checkpoint
```

The Float64 mock source and joint gradient workers used the matched replay
environment and the `gradient` phase above. Their scores and gradients differed
from the archived results by at most `1.82e-12` and `2.69e-11`, respectively,
and the shape audit passed. The Float32 archived M3 injection and posterior peak
were replayed three times each with the archived configuration. The largest score
difference was one Float32 ULP (`1.22e-4`). The real VIS resume used
`python -u model.py --skip-done --vectorized-chunk-size 5 --seed 73 --out-dir "$vis_output"`
with copied artifacts. All six stage pickle hashes and all four M0–M3 reporting-point
scores matched the saved run.
