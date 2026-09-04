# Technical notes

This directory collects reusable engineering and modeling experience that does
not belong to a user guide, an architecture decision record, or a single-run
validation report. Notes may cover code implementation, numerical algorithms,
solver behavior, and practical lens-modeling choices.

## Notes

- [Stabilizing FISTA for joint pixel-source and MGE lens-light inversion](fista-joint-pixel-source-mge.md)
- [Replacing scalar-step FISTA with PNPG](fista-joint-pixel-source-mge.md#pnpg-replacement)
- [GPU-memory study for the joint pixel-source + MGE operator backend](joint-pixel-source-mge-operator-memory.md)
- [Point-source image-position solver performance study](point-source-position-solver-performance.md)
- [Fixed pixelized-source grids and source-bbox boundary behavior](fixed-pixelized-source-grid-and-bbox-boundaries.md)
