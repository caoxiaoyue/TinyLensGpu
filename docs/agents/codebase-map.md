# Codebase Map

Paths below are relative to the repository root. Start with the row for the behavior being changed, then follow imports to the other layers. Read the relevant glossary and ADRs as described in [domain.md](domain.md).

## Code and regression entry points

| Area | Code entry points | Regression entry points |
| --- | --- | --- |
| Physical composition and profiles | `TinyLensGpu/PhysicalModel/LensImage/composite.py`; `TinyLensGpu/PhysicalModel/LensImage/Parametric/` | `tests/test_caskade_models.py`; `tests/test_mass_profile.py`; `tests/test_light_profile.py`; `tests/test_pseudo_jaffe_subhalo.py` |
| Parametric image simulation and likelihood | `TinyLensGpu/ForwardSimulation/LensImage/parametric.py`; `TinyLensGpu/ObservationModel/LensImage/parametric_image_model.py` | `tests/test_integration.py`; `tests/test_linear_solver.py` |
| Pixelized source and dense inversion | `TinyLensGpu/PhysicalModel/LensImage/Pixelized/Light/pixelized_source.py`; `TinyLensGpu/ForwardSimulation/LensImage/pixelized.py`; `TinyLensGpu/ObservationModel/LensImage/pixelized_image_model.py` | `tests/test_pixelized_source_model.py`; `tests/test_pixelized_inversion.py` |
| Pixelized operator inversion and derivatives | `TinyLensGpu/ForwardSimulation/LensImage/pixelized_operator.py`; `TinyLensGpu/ObservationModel/LensImage/pixelized_image_model_operator.py`; `TinyLensGpu/utils/curvature_operator.py`; `TinyLensGpu/utils/cg_solver.py`; `TinyLensGpu/utils/pnpg_solver.py` | `tests/test_pixelized_operator.py`; `tests/test_operator_autodiff.py`; `tests/test_solver_autodiff.py` |
| Fixed-macro source–potential reconstruction | `TinyLensGpu/ForwardSimulation/LensImage/potential_correction.py`; `TinyLensGpu/ObservationModel/LensImage/potential_correction.py`; `TinyLensGpu/ObservationModel/LensImage/potential_correction_operator.py`; `TinyLensGpu/utils/lensing/potential.py`; `TinyLensGpu/utils/inversion/potential_regularization.py` | `tests/test_potential_correction_rendering.py`; `tests/test_potential_correction_dense.py`; `tests/test_potential_correction_operator.py`; `examples/tests/test_potential_correction_operator_example.py`; `benchmarks/test_potential_correction_memory.py` |
| Fixed-macro nonlinear source–potential iteration | `TinyLensGpu/ObservationModel/LensImage/potential_correction_iterative.py`; `TinyLensGpu/ObservationModel/LensImage/potential_correction_iterative_operator.py` | `tests/test_potential_correction_iterative.py`; `tests/test_potential_correction_iterative_operator.py`; `tests/test_potential_correction_residual_refinement.py` (slow current scientific control/budget); `examples/tests/test_potential_correction_iterative_example.py`; `examples/tests/test_potential_correction_iterative_scientific.py` |
| Fixed-strength L-BFGS-B source–potential reconstruction | `TinyLensGpu/ObservationModel/LensImage/potential_correction_lbfgs.py`; `TinyLensGpu/ObservationModel/LensImage/potential_correction_lbfgs_operator.py` | `tests/test_potential_correction_lbfgs.py`; `examples/tests/test_potential_correction_lbfgs_*.py`; `benchmarks/test_lbfgs_operator_resources.py` |
| Source–potential inference, exports and precision | `TinyLensGpu/Inference/potential_correction.py`; `examples/potential_correction/shared/result_products.py`; `examples/potential_correction/mock/{single_step,iterative}/run.py`; `examples/potential_correction/j0946/{macro_model,pt,pt_iter_lbfgs,run_pipeline}.py` (self-contained J0946 workflow) | `tests/test_potential_correction_{inference,iterative_inference,precision,jax_compatibility}.py`; `examples/tests/test_potential_correction_*.py`; `benchmarks/test_potential_correction_{memory,preconditioner,workflow}.py` |
| Source mapping, PSF, and regularization | `TinyLensGpu/utils/lensing/`; `TinyLensGpu/utils/inversion/`; `TinyLensGpu/ObservationModel/LensImage/_pixelized_regularization.py` | `tests/test_pixelized_source_utils.py`; `tests/test_regularization.py` |
| Multi-band and point-source likelihoods | `TinyLensGpu/ObservationModel/LensImage/multi_band_image_model.py`; `TinyLensGpu/ObservationModel/LensImage/point_source_model.py`; `TinyLensGpu/ObservationModel/LensImage/_position_likelihood.py` | `tests/test_multiband_parametric.py`; `tests/test_point_source_model.py`; `tests/test_position_likelihood.py` |
| Priors, likelihood wrappers, and stage transfer | `TinyLensGpu/Inference/build_prior.py`; `TinyLensGpu/Inference/build_likelihood.py`; `TinyLensGpu/Inference/stage_posterior.py`; `TinyLensGpu/Inference/NestedSampler/`; `TinyLensGpu/Inference/Optimizer/` | `tests/test_prior.py`; `tests/test_stage_posterior.py` |
| Plotting and example assembly | `TinyLensGpu/visualizer/`; `examples/pix_src_demo/`; `examples/pix_src_demo_operator/` | `tests/test_visualizer.py`; `examples/tests/` |

## Imports and backend selection

Common profiles and `PhysicalModel` are re-exported by `TinyLensGpu.PhysicalModel`. Parametric and dense observation models are re-exported by `TinyLensGpu.ObservationModel`. Check the relevant `__init__.py` for the available symbols; the package root initializes runtime behavior rather than exporting the modeling API.

Use the defining modules for these entry points:

```python
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import (
    PixelizedImageProbModelOperator,
)
```

Solver names depend on the observation model:

| Model | Supported solver choices |
| --- | --- |
| Parametric `ImageProbModel` | `nnls` (non-negative), `normal` (unconstrained) |
| Dense `PixelizedImageProbModel` | `nnls` (non-negative), `cholesky` (unconstrained) |
| Operator `PixelizedImageProbModelOperator` | `pnpg` (non-negative), `pcg` (unconstrained) |

The historical FISTA spec and experiments remain as records of earlier work; consult their supersession notes and the current operator constructor before selecting a solver. Fixed-input preparation, supported differentiation, and evidence approximations are described in [the guide](../guides/guide.md) and the relevant model's docstring.

`TinyLensGpu.ObservationModel` exports the signed source–potential fit APIs
`SingleStepPotentialCorrection`, `IterativePotentialCorrection`, and
`LBFGSPotentialCorrection`. `TinyLensGpu.ForwardSimulation` exports their
renderer. `TinyLensGpu.Inference` exports the callable evidence/DIRECT
interfaces. See the
[usage guide](../guides/potential-correction.md) and
[requirement/verification record](../validation/potential-correction-coverage.md).

## Test and example boundaries

Core behavior tests live in `tests/`, example assembly and pipeline regressions in `examples/tests/`, and timing assertions in `benchmarks/`. Run the suites in separate processes because example imports can change JAX or plotting configuration. Follow the environment and test commands in `AGENTS.md` and the fixture/path rules in [the guide](../guides/guide.md).

Use `README.md` for the demo index and `examples/` for complete runnable assembly. Find research and reproduction records through [the scratch index](../../.scratch/README.md), and reusable findings through [technical notes](../technical-notes/README.md).
