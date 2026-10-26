# 07: Correct joint reconstruction displays and migrate subhalo diagnostics

**What to build:** Operator joint semi-linear inversion and subhalo cases display actual fitted image components and obtain all reconstruction diagnostics from public interfaces with explicit availability and failure semantics.

**Blocked by:** 03: Deliver operator diagnostics through a complete example

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Migrate the operator simultaneous MGE stage, operator simple joint-fitting examples, and the 0946 and B1938 subhalo cases as needed for the public diagnostic contract.
- [x] Replace zero component placeholders in the legacy simultaneous example with actual lensed-source and lens-light model images from the public joint reconstruction, preserving its layout and display style.
- [x] Retain block-Schur joint inversion and fitted lens-light intensities on unit-amplitude bases; verify that returned component images reproduce the total model on active pixels.
- [x] Treat requested operator joint effective parameter count, residual degrees of freedom, and reduced chi-square as unavailable with a reason; display raw chi-square without failing an otherwise valid reconstruction.
- [x] Preserve already established, explicitly labeled per-data-pixel or arc-region summaries where applicable, without relabeling them as reduced chi-square based on unavailable joint degrees of freedom.
- [x] Replace private regularization and position-diagnostic reads in subhalo plotting with the public diagnostic result and retain its existing overlays and maps.
- [x] Stop formal plotting and source export for explicit non-convergence or nonfinite reconstruction; keep stage sampling, cache, and overwrite policies unchanged in this ticket.
- [x] Verify the real joint reconstruction, corrected display data, diagnostic availability, and subhalo output behavior through appropriate existing core and example test seams.

## Comments

- Claimed by the ticket 07 subagent on `pwr/07-operator-joint-diagnostics`,
  worktree `/tmp/tinylens-pwr-07`, based on integration `9c9e09b` (ancestry
  verified). Ticket 03 is resolved. Applying the `tdd` skill at the approved
  public plot/export seams, using real small operator inversions, without
  private solver mocks. Changes remain uncommitted for lead review.

- Reclaimed on the same branch for the authorized review correction: use
  `InvalidReconstructionError` for invalid guards and propagation handlers,
  preserving unrelated rendering `RuntimeError`s as nonfatal. Integration
  `6eb15d3` merged before this test-first correction.

- Migration inventory (all under `examples/pix_src_demo_operator/`):
  - `simple/pix_src_lens_light/fit_joint.py` and
    `simple/pix_src_lens_light_mge/fit_joint.py`: one diagnostics call requesting
    components, validity guard before PNG/CSV output, returned total/component
    images and amplitudes, unchanged chi-square-per-data-pixel and arc summaries.
  - `pipe/simult_fit_lens_source/mge_lens_light/model.py::_plot_stage_b`: one
    public joint reconstruction requesting components/effective DOF; actual
    fitted components replace placeholders. Raw chi-square plus explicitly
    unavailable effective-parameter/residual-DOF/reduced-chi-square labels and
    the public unsupported reason. Fixed the legacy 2D-axes overlay indexing
    and evaluate overlays in the fitted mass context.
  - `pipe/subhalo_0946/model.py` and `pipe/subhalo_B1938/model.py`: migrated
    `_plot_pix_stage` and `_solve_pixel_source_for_package` using default
    optional flags. Position, regularization, residual, and chi-square fields
    come from diagnostics; raw package keys and float64 conversions retained.
    Subhalo markers, caustics, source-axis bounds, masks, and adaptive maps
    remain in place. B1938 stage-A parametric plotting is unchanged.
  - All seven plot/export invalid guards and the 21 pixelized run/main/replot
    propagation handlers use `InvalidReconstructionError`, imported from
    `TinyLensGpu.ObservationModel.LensImage.inversion_result`. This authorized
    review correction preserves the existing error messages and makes only
    invalid reconstructions fatal; unrelated rendering `RuntimeError`s retain
    the existing nonfatal warning policy. Saved sampling and stage caches
    remain intact; schema/overwrite checks, priors, scientific settings, and
    sampling configuration are unchanged.

- TDD evidence:
  - `examples/tests/test_operator_diagnostics_plotting.py`: real masked 10x10
    image, 5x5 pixelized source, unit-amplitude Gaussian lens-light basis, PCG;
    source/lens panels reproduce the total within `atol=1e-6`. Optional fixed
    bbox/adaptive template and position constraints exercise public overlays
    and scale maps. Joint unsupported statistics still permit plots and exports.
    The legacy plot first failed at its invalid 2D axes indexing; after migration
    its artifact, components, amplitudes, raw chi-square and unavailable labels
    pass. Real `pcg_max_iter=0` and NaN inputs initially produced success artifacts
    in the simple plots and 0946 plot; migrated callers raise before PNG/CSV
    creation. Export tests verify public failure context and exact raw keys/dtypes.
  - `examples/tests/test_b1938_pipeline.py`: extends actual stage-M1 construction
    to PCG/NaN plot and export rejection, preserving `plotting stage-M1` error
    text. A controlled external sampler with the real M2 PCG model demonstrated
    that the old broad handler swallowed reconstruction failure; the stage now
    propagates it while retaining its saved schema-bearing posterior cache.
    The review regression uses the controlled external sampler with a real
    valid M2 reconstruction and makes external `matplotlib.pyplot.savefig`
    raise `RuntimeError("render backend failed")`: the stage must still return
    its posterior/medians and retain its cache, reporting a nonfatal warning.
    Both review tests failed before narrowing (wrong invalid-error type and
    escaping rendering error); genuine PCG failure now asserts the specific
    `InvalidReconstructionError` type.
    No private solver mocks, full-data sampling, or servers were used.
  - Repeat-solve GPU roundoff affected a small normalized arc-residual mean
    comparison by about `4.4e-6`; the retained summary check independently
    computes NumPy residual statistics from the displayed model, rather than
    amplifying roundoff between separate solves. Component checks retain their
    `1e-6` absolute tolerance.

- Verification: activate conda `tinylens_gpu`, then from `examples/tests` run
  `XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg pytest
  test_operator_diagnostics_plotting.py test_b1938_pipeline.py -q`
  → **32 passed** (44.36 s after merging `6eb15d3` and the authorized exception
  narrowing). The two stage-M2 review tests also passed in the focused
  red/green rerun (`pytest test_b1938_pipeline.py -k stage_m2 -q`).
  `git diff --check` passes. AST inventory finds exactly seven diagnostics
  call sites (five plots, two exports) and no remaining private reconstruction,
  position, or regularization-diagnostic reads in the assigned scripts.
  The review AST audit additionally confirms seven typed invalid guards and
  21 narrowed propagation handlers, with no broad `except RuntimeError` left
  in the five owned scripts.
  Fast-forwarded latest integration to `6eb15d3`; original `9c9e09b` ancestry
  remains confirmed. No implementation commit was created, as requested.
