# 03: Score source-only operator fits with Overlap

**What to build:** A modeler can select Overlap evidence for a source-only operator fit and compare it with explicit Block scoring on the same reconstruction. The score uses accurate local curvature with bounded window workspace.

**Blocked by:** [02 — Fit Hermite pixelized sources with squared-weight adaptation](02-hermite-adapt-image.md).

**Status:** ready-for-agent
**State:** resolved

**Spec:** [Parent specification](../spec.md)

- [x] Expose explicit `overlap` and `block` evidence choices and an Overlap tile width independent of the solver block width. Use the prototype tile width of 10 as the baseline.
- [x] A single window covering the grid agrees with a complete dense determinant reference. Multiple windows agree with an independently assembled signed sum, including partial tiles and global finite-difference contributions at window edges.
- [x] Small reference cases exercise zero-, first- and second-order precision, the smallest supported grid, both interpolations, an asymmetric PSF, a mask and subpixel integration.
- [x] Matched Block and Overlap evaluations use the same linear solution and data and prior energy terms. Explicit Block agrees with the existing calculation on the matched baseline.
- [x] The operator assembles exact window Gram matrices with bounded source columns and the masked PSF action. Workspace checks cover the prohibited image-wide PSF, full design and global curvature allocations.

## Comments

- Claimed on `codex/overlap-03-source-overlap`.
- Implemented in `e8023bb`. From `tests/` in `tinylens_gpu`, CPU `pytest -q test_overlap_evidence.py test_pixelized_interpolation_evidence.py::test_ordinary_evidence_uses_native_data_and_prior_tile_determinants --maxfail=1` passed all 7 cases. Coverage includes the full determinant, periodic and cropped asymmetric PSF, adaptive partial windows for all prior orders, matched reconstructions, the explicit Block reference, and a compiled HLO shape audit for moving mass.
