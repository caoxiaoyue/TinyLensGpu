# 02: Generate Pseudo-Jaffe mocks and truth

**What to build:** A researcher can generate a reproducible Pseudo-Jaffe subhalo observation and a matched no-subhalo control, together with true source initialization information and potential, deflection, and convergence products suitable for later correction workflows.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent
**State:** resolved

**Parent:** [Approved potential-correction spec](../spec.md).

- [x] A runnable generator produces single-band, lens-light-subtracted mock data using the existing PseudoJaffeSubhalo strength, host Einstein radius, position, and tidal-radius prescription.
- [x] A matched control removes the subhalo while preserving and recording the shared macro, source, PSF, observation settings, and noise-generation convention.
- [x] Saved or reproducibly reconstructible products include the image, noise information, PSF, observation mask, physical configuration, seed, units, and integration settings.
- [x] Source truth can provide brightness at requested source nodes and spatial gradients at requested source-plane coordinates, without requiring a potential-correction solver or a PyAuto runtime.
- [x] Independent diagnostic helpers provide potential and convergence truth at requested coordinates; deflection truth agrees with the existing mass component.
- [x] The zero-strength limit is defined, and tests cover the profile's singular center and existing numerical radius convention without inventing a new core.
- [x] Potential reference level and the sampling, averaging, or invalid-point convention used for convergence truth are recorded.
- [x] Image size, grid sampling, physical parameters, and output destination are configurable; a small deterministic invocation is suitable for maintained example tests.
- [x] Tests verify reproducibility and truth helpers against independent profile relations, including the relation between potential derivatives and deflection away from the singular center.
- [x] The generator and its input/output contract are documented without claiming recovery, detection significance, or correspondence to the reference repository's exact simulation.

## Scope and handoff

This ticket is independently verifiable through generated observations and
analytic truth. It does not fit a correction or extend the entire mass-profile
API. Do not confuse PseudoJaffeSubhalo with the separately parameterized cored
PseudoJaffe component.


## Comments

- Claimed by `/root/ticket02` on branch `codex/pc-02`, based on integration `5bbfdfa`, in `/tmp/tinylens-pc-02`. Approved test seams: public mock generation, analytic truth helpers, and saved observation artifacts (parent spec Testing Decisions 1, 9, 11).

- Implemented in `67cbf524dfe737daa0cc8701b174c64624cda64b`.
  Public inputs, truth conventions, artifact schema, and reproducible commands
  are in [the example README](../../../examples/potential_correction/README.md).
  `generate_mock_pair` exposes the fixed true macro `PhysicalModel`, matching
  `SimulatorConfig`, observations, and analytic source/subhalo truth.
- Verification in `tinylens_gpu` with `XLA_PYTHON_CLIENT_PREALLOCATE=false`:
  from `examples/tests/`, `pytest test_potential_correction_mock.py -q`:
  **12 passed** (8.48 s). Float64 truth/component deflections agree at
  `rtol=2e-13`; potential finite differences agree at `rtol=2e-9` away from
  the singular numerical domain. Source analytic derivatives agree at
  `rtol=2e-9, atol=1e-9`. Boundary tests cover zero strength, exact center,
  radii below/at 1e-8 arcsec, float32 delta-PSF imaging, and invalid inputs.
  Generator tests verify shared noise, precision/default preservation, fixed
  macro, serialized configuration, saved truth validity, and exact regeneration.
- Separate core process from `tests/`:
  `pytest test_pseudo_jaffe_subhalo.py -q`: **6 passed** (2.34 s).
  No core mass-profile code or prior semantics changed.
- Default scientific-size generator invocation from repository root:
  `python examples/potential_correction/generate_mock.py --output /tmp/tinylens-pc-02-mock-scientific`
  completed successfully. Configuration: 200×200 image, 0.02 arcsec pixels,
  mock integration nsub=2, float64, SIS theta_E=1 arcsec plus shear (0.05,0.02),
  default two-Gaussian source, b_sub=0.01 arcsec at (0.9,0.1), rt=0.1 arcsec,
  noise sigma=0.01 and PCG64 seed=17, 9×9 Gaussian PSF sigma=0.04 arcsec,
  annulus 0.6–1.4 arcsec. It selects 12,552 native pixels. Both observation
  images are finite with identical noise draws. The noiseless subhalo/control
  difference has selected-pixel RMS 0.0037291652982337645 intensity units;
  subhalo peak brightness is 0.9581243440965788. These describe generated
  inputs only, not recovery or significance. Artifacts are regenerable from
  the maintained generator and these defaults; `/tmp` outputs are not required
  retained scientific evidence.
- Scope: physical analytic potential has psi(center)=0; convergence is a
  point sample with invalid values for R<1e-8 arcsec at nonzero strength.
  Deflection preserves the component's numerical radius floor. No core,
  potential correction fit, detection claim, or mass uncertainty is added.
