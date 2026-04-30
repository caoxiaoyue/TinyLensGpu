# LENSIMAGE MODELING KB

## OVERVIEW
This directory is the physics definition layer for image-plane modeling. It contains the composition root and parametric mass/light profiles.

## STRUCTURE
```text
TinyLensGpu/PhysicalModel/LensImage/
|- composite.py            # PhysicalModel container
|- Parametric/
|  |- Mass/                # SIE, EPL, TNFW, multipoles, flexion, etc.
|  `- Light/               # Sersic, Gaussian, shapelet, backgrounds
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add mass profile | `TinyLensGpu/PhysicalModel/LensImage/Parametric/Mass/` | Keep `center_x` / `center_y` naming |
| Add light profile | `TinyLensGpu/PhysicalModel/LensImage/Parametric/Light/` | `light(x, y)` is the core surface |
| Change component composition | `TinyLensGpu/PhysicalModel/LensImage/composite.py` | Handles list registration and summation |

## CONVENTIONS
- Every profile is a `ck.Module`; forward math lives behind `@ck.forward` methods.
- Parameter values should be `ParamU`-compatible so inference modes (`dynamic`, `static`, `linear`) remain available.
- `PhysicalModel` composes `lens_mass`, `source_light`, and `lens_light` as plain Python lists, then registers children with unique names.
- Parametric mass and light profiles standardize on `center_x` and `center_y` for centers.

## ANTI-PATTERNS
- Do not replace `object.__setattr__` list registration with normal assignment; Caskade `NodeList` conversion breaks duplicate profile collections.
- Do not call composite child `deriv()` / `light()` through the normal bound wrapper when manually threading parameter values; existing composite profiles use `__wrapped__` intentionally.

## NOTES
- `Parametric/Mass/__init__.py` exports 18 public symbols; `Parametric/Light/__init__.py` exports 10.
- `Multipole` keeps `m` as a static Python integer; treat that branching pattern as intentional JAX-trace-time behavior.
