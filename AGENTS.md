# PROJECT KNOWLEDGE BASE

**Generated:** 2026-05-06 Asia/Shanghai
**Commit:** 13aa053
**Branch:** main

## OVERVIEW
TinyLensGpu is a JAX/Caskade strong-lensing library. Core work is programmatic Python, with runnable examples under `paper/demo/` and dense regression coverage under `tests/`.

## STRUCTURE
```text
./
|- TinyLensGpu/               # installable package
|  |- PhysicalModel/         # mass/light/source definitions
|  |- ForwardSimulation/     # image synthesis and inversion assembly
|  |- ObservationModel/      # likelihood / evidence wrappers
|  |- Inference/             # ParamU, priors, samplers, optimizers
|  `- utils/                 # shared numerics, PSF, inversion, geometry
|- tests/                    # flat pytest suite + perf/benchmark files
|- doc/                      # authoritative guides; prefer over README prose
`- paper/demo/               # runnable examples and paper assets; mixed primary + archival
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add mass or light profile | `TinyLensGpu/PhysicalModel/LensImage/` | Caskade `ck.Module` patterns live here |
| Change image simulation | `TinyLensGpu/ForwardSimulation/LensImage/` | `parametric.py` is the entry surface |
| Change likelihood / evidence | `TinyLensGpu/ObservationModel/LensImage/` | Thin bridge over simulators |
| Change priors / sampler wiring | `TinyLensGpu/Inference/` | `ParamU`, `build_prior.py`, `build_likelihood.py` |
| Change mapping / PSF / regularization | `TinyLensGpu/utils/lensing/` | Shared by simulators and observation models |
| Change dense or operator inversion | `TinyLensGpu/utils/inversion/` | Highest-risk numerical core |
| Verify behavior | `tests/` | Flat suite; markers and perf tests matter |
| Learn canonical workflow | `doc/GUIDE.md` | Real setup/use authority |

## CODE MAP
| Symbol | Type | Location | Refs | Role |
|--------|------|----------|------|------|
| `PhysicalModel` | class | `TinyLensGpu/PhysicalModel/LensImage/composite.py` | package-wide | Composition root for mass / source / lens light |
| `LensSimulator` | class | `TinyLensGpu/ForwardSimulation/LensImage/parametric.py` | demos + tests | Parametric forward model |
| `ImageProbModel` | class | `TinyLensGpu/ObservationModel/LensImage/parametric_image_model.py` | demos + tests | Image chi-square likelihood |
| `ParamU` | class | `TinyLensGpu/Inference/param_u.py` | all modern demos | Prior-aware parameter wrapper |
| `make_prior_transformation` | function | `TinyLensGpu/Inference/build_prior.py` | all demos | Unit-cube to physical transform |
| `make_likelihood` | function | `TinyLensGpu/Inference/build_likelihood.py` | all demos | Sampler-facing callable builder |


## CONVENTIONS
- Activate the environment with `source ~/anaconda3/bin/activate && conda activate tinylens_gpu` before running test programs.
- When running a specific test, run it from the directory where that file lives unless a task says otherwise.
- Treat `doc/GUIDE.md` as the authority for install, quickstart, tests, and troubleshooting; README is overview-first.
- Modern workflow is programmatic only. Legacy ModelParser / YAML runner code is removed from the current layout.
- Importing `TinyLensGpu` forces `CASKADE_BACKEND=jax`; do not document or introduce alternate backends here.
- Demo scripts under `paper/demo/*/run_model.py` are the real entry points; there is no package CLI.
- Exclude `.worktrees/` from structural reasoning; it is a duplicate git worktree already ignored by `.gitignore`.
- Use Python 3.10+ style with 4-space indentation and standard naming: `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_CASE` for constants.
- Keep modules focused and numerics explicit; shape assumptions and units should be clear in code or docstrings.
- Prefer type hints for public APIs, especially in `TinyLensGpu/Inference/` and model-builder surfaces.
- Write documentation and inline comments in English. Favor detailed docstrings for public classes/functions and comments for non-obvious numerical logic.
- Keep structure concise and readable; avoid unnecessary helper functions and over-engineering.
- During refactors, remove duplicate logic, preserve single-responsibility boundaries, and reduce excessive nesting.

## ANTI-PATTERNS (THIS PROJECT)
- Do not scan or document `.worktrees/` as real package structure.
- Do not follow the stale `CASKADE_GUIDE.md` README link; use `doc/GUIDE.md` instead.
- Do not reintroduce removed YAML-first workflows when documenting current behavior.
- Do not treat `paper/demo/` data and `output/` folders as source modules; only the scripts matter.

## UNIQUE STYLES
- Architecture is pipeline-shaped: `PhysicalModel -> ForwardSimulation -> ObservationModel -> Inference`.
- Tests are flat, not mirrored by subpackage, but they still map tightly onto source subsystems.

## COMMANDS
```bash
source ~/anaconda3/bin/activate && conda activate tinylens_gpu

pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .

pytest
pytest -m "integration"
pytest -m "not slow"
mypy TinyLensGpu/

python paper/demo/lens_only/run_model.py
python paper/demo/point_source/sim_data.py && python paper/demo/point_source/run_model.py
```

## NOTES
- `README.md` still points to a missing `CASKADE_GUIDE.md`; contributors should follow `doc/GUIDE.md`.
- `.gitignore` has a broad `*yml` rule; this repo intentionally does not treat YAML files as tracked configuration.
- GPU-constrained work often needs `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- When sharing refactors, provide the full updated code plus a brief list of key changes and why they were made.
