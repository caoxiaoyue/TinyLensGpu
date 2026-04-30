# UTILS KB

## OVERVIEW
`TinyLensGpu/utils/` is the shared numerical substrate used by physical models, simulators, and probability models. Prefer extending these helpers over re-implementing math in feature packages.

## STRUCTURE
```text
TinyLensGpu/utils/
|- geometry/              # coordinate transforms, ellipticity helpers
|- interpolation/         # Wendland kernels and weights
|- lensing/               # mapping, PSF, point-source solving
|- linear_solver.py       # parametric NNLS / normal solver
`- misc.py                # FITS loading, small filesystem helpers
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Coordinate / ellipticity math | `TinyLensGpu/utils/geometry/` | Reused by many mass and light profiles |
| Kernel interpolation | `TinyLensGpu/utils/interpolation/` | Feeds mapping code |
| Lens mapping / PSF | `TinyLensGpu/utils/lensing/` | Detailed child AGENTS exists |
| FITS data loading | `TinyLensGpu/utils/misc.py` | Demo entry surface |

## CONVENTIONS
- Keep this layer low-level and reusable; package-specific orchestration belongs higher up.
- JAX and Numba performance assumptions are baked in; preserve vectorized, device-friendly interfaces.
- Numerical floors, jitter terms, and stability clamps are part of the contract, not cleanup targets.
- `load_lens_data()` in `misc.py` is the canonical demo/data-loading helper.

## ANTI-PATTERNS
- Do not duplicate geometry transforms inside new profiles.
- Do not move likelihood- or sampler-specific policy into utils.
- Do not silently remove numerical safeguards to make formulas look cleaner.

## NOTES
- `geometry/transforms.py` is imported across much of `PhysicalModel`.
- `linear_solver.py` is for parametric linear-intensity solving.
