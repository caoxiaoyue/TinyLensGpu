# PACKAGE KNOWLEDGE BASE

## OVERVIEW
`TinyLensGpu/` is the installable package surface. It exports the modern Caskade/JAX API and keeps most user-facing work inside `PhysicalModel`, `ForwardSimulation`, `ObservationModel`, `Inference`, and `utils`.

## STRUCTURE
```text
TinyLensGpu/
|- __init__.py              # version + forced Caskade backend
|- PhysicalModel/           # physics definitions
|- ForwardSimulation/       # simulators
|- ObservationModel/        # probability models
|- Inference/               # ParamU, priors, samplers, optimizers
|- utils/                   # shared numerics
`- visualizer.py            # plotting helper
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| New physical component | `TinyLensGpu/PhysicalModel/` | Mass and light abstractions |
| New simulator behavior | `TinyLensGpu/ForwardSimulation/` | Config + orchestration |
| New likelihood model | `TinyLensGpu/ObservationModel/` | Thin wrappers over simulators |
| New sampler / optimizer glue | `TinyLensGpu/Inference/` | Keep demo-facing APIs stable |
| Shared math helper | `TinyLensGpu/utils/` | Prefer reuse over local copies |

## CONVENTIONS
- `TinyLensGpu/__init__.py` sets `CASKADE_BACKEND=jax` on import; assume JAX backend throughout the package.
- Modern imports are `TinyLensGpu.PhysicalModel`, `TinyLensGpu.ObservationModel`, `TinyLensGpu.Inference`, `TinyLensGpu.ForwardSimulation`.
- Some demos still use legacy `TinyLensGpu.Models` paths; treat those as compatibility residue, not the preferred surface.
- Export surfaces are controlled from package `__init__.py` files; many leaf modules intentionally skip their own `__all__`.

## ANTI-PATTERNS
- Do not add alternate backend setup in package init.
- Do not create new top-level convenience modules when the existing subpackage already owns the concept.
- Do not duplicate utility code inside feature packages; route it into `TinyLensGpu/utils/`.

## NOTES
- `visualizer.py` is post-processing only; it is not part of the core modeling pipeline.
- Child AGENTS files under this package cover the high-complexity hotspots in more detail.
