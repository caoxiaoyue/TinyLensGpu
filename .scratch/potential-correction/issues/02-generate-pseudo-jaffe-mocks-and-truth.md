# 02: Generate Pseudo-Jaffe mocks and truth

**What to build:** A researcher can generate a reproducible Pseudo-Jaffe subhalo observation and a matched no-subhalo control, together with true source initialization information and potential, deflection, and convergence products suitable for later correction workflows.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent
**State:** open

**Parent:** [Approved potential-correction spec](../spec.md).

- [ ] A runnable generator produces single-band, lens-light-subtracted mock data using the existing PseudoJaffeSubhalo strength, host Einstein radius, position, and tidal-radius prescription.
- [ ] A matched control removes the subhalo while preserving and recording the shared macro, source, PSF, observation settings, and noise-generation convention.
- [ ] Saved or reproducibly reconstructible products include the image, noise information, PSF, observation mask, physical configuration, seed, units, and integration settings.
- [ ] Source truth can provide brightness at requested source nodes and spatial gradients at requested source-plane coordinates, without requiring a potential-correction solver or a PyAuto runtime.
- [ ] Independent diagnostic helpers provide potential and convergence truth at requested coordinates; deflection truth agrees with the existing mass component.
- [ ] The zero-strength limit is defined, and tests cover the profile's singular center and existing numerical radius convention without inventing a new core.
- [ ] Potential reference level and the sampling, averaging, or invalid-point convention used for convergence truth are recorded.
- [ ] Image size, grid sampling, physical parameters, and output destination are configurable; a small deterministic invocation is suitable for maintained example tests.
- [ ] Tests verify reproducibility and truth helpers against independent profile relations, including the relation between potential derivatives and deflection away from the singular center.
- [ ] The generator and its input/output contract are documented without claiming recovery, detection significance, or correspondence to the reference repository's exact simulation.

## Scope and handoff

This ticket is independently verifiable through generated observations and
analytic truth. It does not fit a correction or extend the entire mass-profile
API. Do not confuse PseudoJaffeSubhalo with the separately parameterized cored
PseudoJaffe component.

