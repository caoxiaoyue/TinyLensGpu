# PIXELIZED SOURCE KB

## OVERVIEW
This directory defines the pixelized-source model surface: frozen configuration objects, public exports, and the `PixelizedSourceModel` wrapper consumed by simulators and probability models.

## STRUCTURE
```text
TinyLensGpu/PhysicalModel/LensImage/Pixelized/
|- __init__.py            # public exports + convenience re-exports
|- config.py              # grid / mapping / regularization / solver dataclasses
|- pixelized_source.py    # PixelizedSourceModel
`- README.md              # legacy-heavy deep dive; partially stale
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add or validate a config field | `TinyLensGpu/PhysicalModel/LensImage/Pixelized/config.py` | All user-facing knobs live here |
| Change regularization wrapper behavior | `TinyLensGpu/PhysicalModel/LensImage/Pixelized/pixelized_source.py` | Dense GP vs sparse rectangular helpers |
| Adjust public exports | `TinyLensGpu/PhysicalModel/LensImage/Pixelized/__init__.py` | Re-exports helpers from `utils/` |
| Learn the full workflow | `doc/pixelized_source_guide.md` | More authoritative than local README |

## CONVENTIONS
- `PixelizedSourceConfig` is the aggregate entry point; it nests `grid`, `mapping`, `regularization`, and `solver` config objects.
- Grid family and regularization family must stay aligned: irregular grids pair with `irregular_gp_*`, rectangular grids pair with `rectangular_*`.
- Config dataclasses are `frozen=True`; change behavior by constructing new config objects, not by mutating fields.
- `PixelizedSourceModel` keeps `config` on the module with `object.__setattr__`; preserve that pattern when extending the wrapper.
- `reg_scale` and `reg_coefficient` are `ParamU`-compatible so the hyperparameters can participate in inference.

## ANTI-PATTERNS
- Do not add a new solver/backend keyword here without also wiring the corresponding implementation in `ForwardSimulation/LensImage/pixelized_core/` and `utils/inversion/`.
- Do not treat `regularization_matrix()` as a universal entry point; it is for dense irregular-GP matrices, while rectangular mode uses `regularization_sparse_rectangular()`.
- Do not document `README.md` as the canonical API source without cross-checking `doc/pixelized_source_guide.md`; the README still contains stale import examples.
- Do not mix matrix/operator guidance with config validation rules; config stays declarative and backend assembly happens downstream.

## NOTES
- `__init__.py` intentionally re-exports inversion, PSF, and mesh helpers for convenience, so public imports here reach beyond this directory.
- `config.py` centralizes accepted kernels, interpolation choices, and solver backend strings; keep new literals normalized with the existing lowercase validation style.
- `pixelized_source.py` is a wrapper and regularization helper, not the full reconstruction engine; the heavy assembly logic lives downstream in `TinyLensGpu/ForwardSimulation/LensImage/pixelized.py`.
