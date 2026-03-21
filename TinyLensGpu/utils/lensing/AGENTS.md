# LENSING NUMERICS KB

## OVERVIEW
This directory owns shared lensing numerics: interpolation/mapping, PSF operators, regularization builders, and point-source solving.

## STRUCTURE
```text
TinyLensGpu/utils/lensing/
|- mapping.py             # bilinear / dense mapping helpers
|- psf.py                 # dense, sparse, FFT PSF paths
|- regularization.py      # GP and sparse rectangular penalties
`- point_source_solver.py # lens equation solving and image matching
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Interpolation / mapping weights | `TinyLensGpu/utils/lensing/mapping.py` | Consumed by simulators and tests |
| PSF matrix construction | `TinyLensGpu/utils/lensing/psf.py` | Dense, sparse, and FFT utilities |
| GP or rectangular regularization | `TinyLensGpu/utils/lensing/regularization.py` | Irregular GP and rectangular sparse modes |
| Point-source image solving | `TinyLensGpu/utils/lensing/point_source_solver.py` | Newton / AMR + assignment matching |

## CONVENTIONS
- Backend choice is split between dense `matrix` paths and memory-saving `operator` paths in higher layers; this directory provides the shared operators for both.
- Regularization families are mode-specific: irregular grids use GP kernels (`exp`, `gauss`, `matern32`, `matern52`), rectangular grids use sparse finite-difference operators.
- Point-source matching uses brute-force permutations only for small image counts; Hungarian matching takes over when factorial growth becomes wasteful.
- String config values are normalized with lowercase / strip semantics before branching.

## ANTI-PATTERNS
- Do not reduce numerical floors such as GP jitter, ridge floors, or Jacobian stabilization without profiling the downstream solvers.
- Do not brute-force image permutations for large point-source multiplicities; the code deliberately switches strategies.
- Do not use irregular GP regularization assumptions for rectangular-grid code paths.

## NOTES
- `point_source_solver.py` and `regularization.py` are major complexity hotspots; changes here ripple into both observation models and forward simulators.
- `psf.py` is one of the main memory pressure sources in large-image workflows.
