# SIMULATION KB

## OVERVIEW
This directory is the simulator orchestration layer. Start here for workflow changes; drop into `pixelized_core/` only when you need algorithm details.

## STRUCTURE
```text
TinyLensGpu/ForwardSimulation/LensImage/
|- config.py               # SimulatorConfig, grids, mask handling
|- parametric.py           # LensSimulator
|- pixelized.py            # PixelizedLensSimulator
`- pixelized_core/
   |- artifacts.py         # immutable intermediate payloads
   |- grid_strategies.py
   |- mapping_strategies.py
   |- regularization_strategies.py
   `- inversion_assembler.py
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Parametric image synthesis | `TinyLensGpu/ForwardSimulation/LensImage/parametric.py` | High-level forward model |
| Pixelized workflow wiring | `TinyLensGpu/ForwardSimulation/LensImage/pixelized.py` | Main entry for grids, PSF, inversion |
| Grid / mapping / regularization algorithm | `TinyLensGpu/ForwardSimulation/LensImage/pixelized_core/` | Strategy-pattern implementation |
| Backend selection | `TinyLensGpu/ForwardSimulation/LensImage/pixelized_core/inversion_assembler.py` | Matrix vs operator, linear vs NNLS |

## CONVENTIONS
- `SimulatorConfig` is the shared configuration object; thread new simulation knobs through it first.
- `pixelized.py` owns orchestration, cache policy, and public reconstruction methods.
- `pixelized_core/` uses strategy classes plus frozen artifact containers; preserve that separation.
- Operator caching is keyed by geometry state; keep cache invalidation semantics aligned with `OperatorCacheKey`.

## ANTI-PATTERNS
- Do not patch deep strategy files for a top-level workflow tweak before checking whether `parametric.py`, `pixelized.py`, or `config.py` already owns that policy.
- Do not add a new inversion backend without wiring it through `inversion_assembler.py`.
- Do not bypass cache-policy handling in `pixelized.py`; `safe`, `off`, and `unsafe_static` are deliberate modes.

## NOTES
- `pixelized.py` is one of the biggest contributor-facing files in the repo; read it before changing pixelized behavior.
- `artifacts.py` is the fastest way to understand data flow through the pixelized path.
