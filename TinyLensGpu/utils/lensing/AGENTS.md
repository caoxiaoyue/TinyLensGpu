# LENSING NUMERICS KB

## OVERVIEW
This directory owns shared lensing numerics: interpolation/mapping, PSF operators, and point-source solving.

## STRUCTURE
```text
TinyLensGpu/utils/lensing/
|- mapping.py             # bilinear / dense mapping helpers
|- psf.py                 # dense, sparse, FFT PSF paths
`- point_source_solver.py # lens equation solving and image matching
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Interpolation / mapping weights | `TinyLensGpu/utils/lensing/mapping.py` | Consumed by simulators and tests |
| PSF matrix construction | `TinyLensGpu/utils/lensing/psf.py` | Dense, sparse, and FFT utilities |
| Point-source image solving | `TinyLensGpu/utils/lensing/point_source_solver.py` | Newton / AMR + assignment matching |

## CONVENTIONS
- Point-source matching uses brute-force permutations only for small image counts; Hungarian matching takes over when factorial growth becomes wasteful.
- String config values are normalized with lowercase / strip semantics before branching.

## ANTI-PATTERNS
- Do not brute-force image permutations for large point-source multiplicities; the code deliberately switches strategies.

## NOTES
- `psf.py` is one of the main memory pressure sources in large-image workflows.
