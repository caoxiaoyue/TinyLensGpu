# SIMULATION KB

## OVERVIEW
This directory is the simulator orchestration layer for parametric lens-image forward modeling.

## STRUCTURE
```text
TinyLensGpu/ForwardSimulation/LensImage/
|- config.py               # SimulatorConfig, grids, mask handling
|- parametric.py           # LensSimulator
`- results.py              # SimulationResult
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Parametric image synthesis | `TinyLensGpu/ForwardSimulation/LensImage/parametric.py` | High-level forward model |

## CONVENTIONS
- `SimulatorConfig` is the shared configuration object; thread new simulation knobs through it first.

## ANTI-PATTERNS
- Do not patch deep math files for a top-level workflow tweak before checking whether `parametric.py` or `config.py` already owns that policy.

## NOTES
- `parametric.py` is the main contributor-facing simulator file.
