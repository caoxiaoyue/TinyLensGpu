# 07: Correct joint reconstruction displays and migrate subhalo diagnostics

**What to build:** Operator joint semi-linear inversion and subhalo cases display actual fitted image components and obtain all reconstruction diagnostics from public interfaces with explicit availability and failure semantics.

**Blocked by:** 03: Deliver operator diagnostics through a complete example

**Status:** ready-for-agent
**State:** open

**Spec:** [Parent specification](../spec.md)

- [ ] Migrate the operator simultaneous MGE stage, operator simple joint-fitting examples, and the 0946 and B1938 subhalo cases as needed for the public diagnostic contract.
- [ ] Replace zero component placeholders in the legacy simultaneous example with actual lensed-source and lens-light model images from the public joint reconstruction, preserving its layout and display style.
- [ ] Retain block-Schur joint inversion and fitted lens-light intensities on unit-amplitude bases; verify that returned component images reproduce the total model on active pixels.
- [ ] Treat requested operator joint effective parameter count, residual degrees of freedom, and reduced chi-square as unavailable with a reason; display raw chi-square without failing an otherwise valid reconstruction.
- [ ] Preserve already established, explicitly labeled per-data-pixel or arc-region summaries where applicable, without relabeling them as reduced chi-square based on unavailable joint degrees of freedom.
- [ ] Replace private regularization and position-diagnostic reads in subhalo plotting with the public diagnostic result and retain its existing overlays and maps.
- [ ] Stop formal plotting and source export for explicit non-convergence or nonfinite reconstruction; keep stage sampling, cache, and overwrite policies unchanged in this ticket.
- [ ] Verify the real joint reconstruction, corrected display data, diagnostic availability, and subhalo output behavior through appropriate existing core and example test seams.
