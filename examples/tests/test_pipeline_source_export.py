"""Regression checks for pixelized pipeline source-package exports."""

import ast

from TinyLensGpu import example_support

# Script blocks execute in isolated namespaces; supply their shared imports.
_SUPPORT_IMPORTS = {
    name: getattr(example_support, name) for name in example_support.__all__
}

import pytest

import TinyLensGpu  # noqa: F401 -- select the JAX backend before importing caskade


_PIPE_PACKAGE_HELPERS = (
    "pre_subtract_lens/pix_src_pipe/model.py",
    "pre_subtract_lens/pix_src_pipe_sersic/model.py",
    "pre_subtract_lens/pix_src_pipe_mge/model.py",
    "slacs_examples/pipeline.py",
    "euclid_examples/pipeline.py",
    "galan24_test/model.py",
    "galan24_test/model_predefine_src_box.py",
)

_ADAPT_IMAGE_PACKAGE_HELPERS = {
    "pre_subtract_lens/pix_src_pipe/model.py",
    "pre_subtract_lens/pix_src_pipe_sersic/model.py",
    "pre_subtract_lens/pix_src_pipe_mge/model.py",
    "slacs_examples/pipeline.py",
    "euclid_examples/pipeline.py",
    "galan24_test/model.py",
    "galan24_test/model_predefine_src_box.py",
}


def _load_source_package_helper(path):
    """Execute only the two source-package helpers, not pipeline side effects."""
    import caskade as ck
    import jax.numpy as jnp
    import numpy as np

    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_source_axes_from_bbox", "_solve_pixel_source_for_package"}
    ]
    from TinyLensGpu.ObservationModel.LensImage.inversion_result import (
        InvalidReconstructionError,
    )
    from TinyLensGpu.example_support import (
        export_source_package, validate_source_package,
        require_valid_reconstruction, source_axes_from_bbox,
        draw_image_panel,
    )

    namespace = {
        **_SUPPORT_IMPORTS,
        "np": np, "jnp": jnp, "ck": ck, "NSRC": 3,
        "InvalidReconstructionError": InvalidReconstructionError,
        "export_source_package": export_source_package,
        "validate_source_package": validate_source_package,
        "require_valid_reconstruction": require_valid_reconstruction,
        "source_axes_from_bbox": source_axes_from_bbox,
        "draw_image_panel": draw_image_panel,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
         namespace)
    return namespace["_solve_pixel_source_for_package"]


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", _PIPE_PACKAGE_HELPERS)
@pytest.mark.parametrize("converged", [True, False])
def test_source_package_helper_uses_public_diagnostics(relative_path, converged, examples_dir):
    """Pipe source-package helpers go through one public diagnostics call."""
    import caskade as ck
    import jax.numpy as jnp
    import numpy as np
    from types import SimpleNamespace

    from TinyLensGpu.ObservationModel.LensImage.inversion_result import (
        InversionResult,
    )

    helper = _load_source_package_helper(
        examples_dir / "pix_src_demo_operator" / "pipe" / relative_path
    )

    calls = []
    records_arc = relative_path in _ADAPT_IMAGE_PACKAGE_HELPERS

    class ProbeLikelihood(ck.Module):
        solver_type = "pcg"

        def __init__(self):
            super().__init__("probe_likelihood")
            self.amplitude = ck.Param("amplitude", 0.0)
            if records_arc:
                self.image_data = jnp.zeros((5, 5))
                self.noise_map = jnp.ones((5, 5))
                mask = jnp.zeros((5, 5), dtype=bool)
                self.sim_obj = SimpleNamespace(sim_config=SimpleNamespace(
                    psf_kernel=jnp.ones((1, 1)), mask=mask, source_seed_mask=mask,
                    dpix=0.1, nsub=1,
                ))
                self.phys_model = SimpleNamespace(lens_mass=[], lens_light=[])

        def get_dynamic_params(self):
            return self.dynamic_params

        @ck.forward
        def evaluate(self):
            raise AssertionError("source export must use diagnostics")

        @ck.forward
        def evaluate_with_diagnostics(self, **_kwargs):
            calls.append(_kwargs)
            pixels = jnp.arange(9, dtype=jnp.float32) + self.amplitude.value
            info = SimpleNamespace(
                converged=self._converged,
                residual_norm=0.1,
                convergence_metric=0.1,
                n_iter=3,
            )
            result = InversionResult(
                jnp.asarray(0.0), jnp.zeros((3, 3)), pixels, None,
                (-0.3, 0.3, -0.3, 0.3), info,
            )
            diagnostics = SimpleNamespace(
                valid=bool(self._converged),
                invalid_reason=None if self._converged else "probe not converged",
                source_image=jnp.ones((5, 5)),
            )
            return result, diagnostics

    probe = ProbeLikelihood()
    probe.amplitude.to_dynamic()
    object.__setattr__(probe, "_converged", converged)

    if not converged:
        with pytest.raises(RuntimeError, match="failed"):
            helper(probe, {"amplitude": 2.5}, ["amplitude"])
        assert len(calls) == 1
        return

    package = helper(probe, {"amplitude": 2.5}, ["amplitude"])
    assert calls == ([{"include_components": True}] if records_arc else [{}])
    expected = {
        "source_pixels", "source_image", "source_bbox",
        "source_x_axis", "source_y_axis", "n", "source_interpolation", "source_boundary",
    }
    if records_arc:
        expected.update(("adapt_image", "reference_provenance"))
        np.testing.assert_array_equal(package["adapt_image"], np.ones((5, 5)))
    assert set(package) == expected
    assert package["n"] == 3
    assert package["source_pixels"].dtype == np.float64
    assert package["source_pixels"].shape == (9,)
    np.testing.assert_allclose(
        package["source_pixels"], np.arange(9) + 2.5,
    )
    np.testing.assert_allclose(
        package["source_image"], (np.arange(9) + 2.5).reshape(3, 3),
    )
    np.testing.assert_allclose(
        package["source_bbox"], (-0.3, 0.3, -0.3, 0.3),
    )
    np.testing.assert_allclose(package["source_x_axis"], np.linspace(-0.3, 0.3, 3))
    np.testing.assert_allclose(package["source_y_axis"], np.linspace(-0.3, 0.3, 3))


def _small_source_export_model(image_data=None):
    import TinyLensGpu  # ensure CASKADE_BACKEND selected before caskade
    import jax.numpy as jnp
    from TinyLensGpu.PhysicalModel import PhysicalModel, SIE
    from TinyLensGpu.PhysicalModel.LensImage.Pixelized.Light import PixelizedSourceModel
    from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
    from TinyLensGpu.ForwardSimulation.LensImage.pixelized import PixelizedLensSimulator
    from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import PixelizedImageProbModelOperator
    from TinyLensGpu.Inference import ParamU

    mass = SIE(theta_E=0.12, e1=0.05, e2=0.0, center_x=0.0, center_y=0.0)
    for p in (mass.theta_E, mass.e1, mass.e2, mass.center_x, mass.center_y):
        p.to_static()
    mass.theta_E.to_dynamic()
    lam = ParamU("log_lambda_reg", 0.0, prior_type="uniform", prior_settings=[-3.0,3.0], limits=[-3.0,3.0])
    rho = ParamU("adaptive_reg_rho", 0.6, prior_type="uniform", prior_settings=[0.0,3.0], limits=[0.0,3.0])
    lam.to_dynamic()
    rho.to_dynamic()
    source = PixelizedSourceModel(n=3, log_lambda_reg=lam, regularization_type="first-order", adaptive_reg_rho=rho)
    phys = PhysicalModel(lens_mass=[mass], source_light=[source], lens_light=[])
    psf = jnp.asarray([[1.0]])
    bbox = (-0.4,0.4,-0.4,0.4)
    sim = PixelizedLensSimulator(phys, SimulatorConfig(dpix=0.08,npix=10,psf_kernel=psf))
    reference = sim.simulate(jnp.linspace(0.1,1.0,9), source_bbox=bbox)
    image = reference if image_data is None else jnp.asarray(image_data)
    return PixelizedImageProbModelOperator(
        image, jnp.ones_like(image)*0.05, psf, 0.08, phys,
        solver_type="pcg", block_size=2, fixed_source_bbox=bbox,
        adapt_image=reference,
    )


@pytest.fixture(scope="module")
def small_source_export_model():
    return _small_source_export_model()


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", _PIPE_PACKAGE_HELPERS)
@pytest.mark.parametrize("rho", [0.0,1.2])
def test_source_package_matches_legacy_real_operator(
    relative_path, rho, small_source_export_model, monkeypatch, examples_dir,
):
    import caskade as ck
    import jax.numpy as jnp
    import numpy as np
    model = small_source_export_model
    names = [p.name for p in model.get_dynamic_params()]
    medians = {"theta_E":0.15, "log_lambda_reg":0.2, "adaptive_reg_rho":rho}
    initial = np.asarray(model.get_values("flat")).copy()
    theta = jnp.asarray([medians[name] for name in names])
    # Literal pre-migration solve path, using the same current parameters.
    with ck.ActiveContext(model):
        model.fill_params(theta)
        strength = jnp.exp(model.phys_model.source_light[0].log_lambda_reg.value)
        xmin,xmax,ymin,ymax,bx,by,_,_ = model._get_bbox()
        scale = model._get_reg_scale()
        reg_data = model._regularization_data(xmin,xmax,ymin,ymax,scale=scale)
        op_data = model.sim_obj.precompute_operator_data(
            xmin,xmax,ymin,ymax,_betas_sub=(bx,by),
        )
        blocks = model.sim_obj.build_block_diag_preconditioner(
            model.noise_1d,xmin,xmax,ymin,ymax,strength,model.reg_builder,
            block_size=model.block_size,scale=scale,
        )
        expected_source, info = model._solve_source(
            xmin,xmax,ymin,ymax,strength,reg_data,blocks,op_data=op_data,
        )
        expected_bbox = np.asarray([xmin,xmax,ymin,ymax])
    assert bool(np.asarray(info.converged))
    expected_source = np.asarray(expected_source)
    assert np.all(np.isfinite(expected_source))
    assert np.any(expected_source != 0.0)

    solve = model._solve_source
    calls = []
    def counted_solve(*args, **kwargs):
        calls.append(None)
        return solve(*args, **kwargs)
    monkeypatch.setattr(model,"_solve_source",counted_solve)
    helper = _load_source_package_helper(examples_dir / "pix_src_demo_operator" / "pipe" / relative_path)
    package = helper(model,medians,names)
    assert len(calls) == 1
    expected_keys = {
        "source_pixels", "source_image", "source_bbox",
        "source_x_axis", "source_y_axis", "n", "source_interpolation", "source_boundary",
    }
    if relative_path in _ADAPT_IMAGE_PACKAGE_HELPERS:
        expected_keys.update(("adapt_image", "reference_provenance"))
        with ck.ActiveContext(model):
            model.fill_params(theta)
            expected_arc, _, _ = model.reconstruct_component_images(expected_source, None)
        np.testing.assert_allclose(package["adapt_image"], expected_arc, rtol=2e-5, atol=2e-5)
    assert set(package) == expected_keys
    assert package["n"] == 3
    assert package["source_pixels"].dtype == np.float64
    np.testing.assert_allclose(package["source_pixels"],expected_source,rtol=2e-5,atol=2e-5)
    np.testing.assert_allclose(package["source_image"],expected_source.reshape(3,3),rtol=2e-5,atol=2e-5)
    np.testing.assert_allclose(package["source_bbox"],expected_bbox,rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(package["source_x_axis"],np.linspace(expected_bbox[0],expected_bbox[1],3),rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(package["source_y_axis"],np.linspace(expected_bbox[2],expected_bbox[3],3),rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(np.asarray(model.get_values("flat")),initial,rtol=0,atol=0)


_MIGRATED_PACKAGE_HELPERS = (
    "galan24_test/model.py",
    "galan24_test/model_predefine_src_box.py",
)


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", _MIGRATED_PACKAGE_HELPERS)
@pytest.mark.parametrize("failure", ["nan_image", "unconverged"])
def test_source_package_helper_rejects_invalid_reconstruction(
    relative_path, failure, examples_dir,
):
    """Migrated helpers raise 'failed' for real invalid reconstructions."""
    import jax.numpy as jnp
    import numpy as np

    helper = _load_source_package_helper(
        examples_dir / "pix_src_demo_operator" / "pipe" / relative_path
    )
    if failure == "unconverged":
        model = _small_source_export_model()
        model.pcg_max_iter = 0
    else:
        model = _small_source_export_model(
            image_data=np.full((10, 10), np.nan)
        )
    names = [p.name for p in model.get_dynamic_params()]
    medians = {n: float(np.asarray(p.value)) for n, p in
               zip(names, model.get_dynamic_params())}
    with pytest.raises(RuntimeError, match="failed"):
        helper(model, medians, names)


# ------------------------------------------------------------------
# Plot-block and stage-flow coverage for migrated basic diagnostics
# ------------------------------------------------------------------

_SIMPLE_PLOT_FITS = (
    "simple/pix_src/fit_lens_src.py",
    "simple/pix_src/fit_src_pos_like.py",
    "simple/pix_src/fit_lens_src_pos_like.py",
    "simple/pix_compact_src/fit_src_pos_like.py",
    "simple/pix_src_epl_sh/fit_lens_src.py",
    "simple/pix_src_epl_sh/fit_lens_src_pos_like.py",
)

_DEMO_INVERSION_FILES = (
    "simple/pix_src/demo_inversion.py",
    "simple/pix_src_epl_sh/demo_inversion.py",
)


@pytest.fixture
def close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _simple_plot_block(fit_path):
    """Statements from `best_params` onward (reconstruction + plot section)."""
    tree = ast.parse(fit_path.read_text(), filename=str(fit_path))
    for i, node in enumerate(tree.body):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "best_params" for t in node.targets)
        ):
            return tree.body[i:]
    raise AssertionError(f"plot block not found in {fit_path}")


def _small_operator_namespace(tmp_path, monkeypatch, image_data=None):
    """Small real operator model plus the names migrated plot blocks expect."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import jax.numpy as jnp
    import caskade as ck

    from TinyLensGpu.Inference import ParamU
    from TinyLensGpu.ObservationModel.LensImage.inversion_result import (
        InvalidReconstructionError,
    )
    from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import (
        PixelizedImageProbModelOperator,
    )
    from TinyLensGpu.PhysicalModel import PhysicalModel, SIE
    from TinyLensGpu.PhysicalModel.LensImage.Pixelized.Light import PixelizedSourceModel
    from TinyLensGpu.visualizer import overlay_critical_and_caustics

    sie = SIE(theta_E=0.12, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    for p in (sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y):
        p.to_static()
    sie.theta_E.to_dynamic()
    lam = ParamU(
        "log_lambda_reg", 0.0, prior_type="uniform",
        prior_settings=[jnp.log(1e-3), jnp.log(1e3)],
        limits=[jnp.log(1e-6), jnp.log(1e6)],
    )
    lam.to_dynamic()
    pix_src = PixelizedSourceModel(n=5, log_lambda_reg=lam)
    phys = PhysicalModel(lens_mass=[sie], source_light=[pix_src], lens_light=[])
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
    data = np.full((10, 10), 0.05) if image_data is None else image_data
    prob_model = PixelizedImageProbModelOperator(
        image_data=data, noise_map=np.full((10, 10), 0.1),
        psf_kernel=_delta_kernel(), dpix=0.08, phys_model=phys,
        mask=mask, nsub=1,
    )
    for outdir in ("output", "output_pos_like"):
        (tmp_path / outdir).mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)
    namespace = {
        **_SUPPORT_IMPORTS,
        "np": np, "jnp": jnp, "plt": plt, "ck": ck,
        "InvalidReconstructionError": InvalidReconstructionError,
        "prob_model": prob_model, "pix_src": pix_src,
        "q50_list": [float(np.asarray(p.value)) for p in prob_model.get_dynamic_params()],
        "param_names": [p.name for p in prob_model.get_dynamic_params()],
        "image_data": np.asarray(prob_model.image_data),
        "noise_map": np.asarray(prob_model.noise_map),
        "mask": mask, "DPIX": 0.08,
        "overlay_critical_and_caustics": overlay_critical_and_caustics,
    }
    return namespace, tmp_path


def _delta_kernel():
    import numpy as np

    return np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])


def _plot_artifacts(tmp_path):
    return list(tmp_path.glob("output/*.png")) + list(tmp_path.glob("output_pos_like/*.png"))


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", _SIMPLE_PLOT_FITS)
def test_simple_plot_block_uses_public_diagnostics(
    relative_path, tmp_path, monkeypatch, examples_dir, close_figures,
):
    """Migrated simple plot blocks run on a real small operator model."""
    path = examples_dir / "pix_src_demo_operator" / relative_path
    namespace, tmp = _small_operator_namespace(tmp_path, monkeypatch)
    exec(compile(ast.Module(body=_simple_plot_block(path), type_ignores=[]),
                 str(path), "exec"), namespace)
    assert _plot_artifacts(tmp)


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", _SIMPLE_PLOT_FITS)
def test_simple_plot_block_rejects_invalid_reconstruction(
    relative_path, tmp_path, monkeypatch, examples_dir, close_figures,
):
    """A nonfinite observation raises InvalidReconstructionError before artifacts."""
    import numpy as np

    from TinyLensGpu.ObservationModel.LensImage.inversion_result import (
        InvalidReconstructionError,
    )

    path = examples_dir / "pix_src_demo_operator" / relative_path
    namespace, tmp = _small_operator_namespace(
        tmp_path, monkeypatch, image_data=np.full((10, 10), np.nan)
    )
    with pytest.raises(InvalidReconstructionError):
        exec(compile(ast.Module(body=_simple_plot_block(path), type_ignores=[]),
                     str(path), "exec"), namespace)
    assert not _plot_artifacts(tmp)


def _demo_loop_body(path):
    """Body of the per-lambda inversion loop in the demo scripts."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.For) and getattr(node.target, "id", "") == "LAMBDA":
            return node.body
    raise AssertionError(f"LAMBDA loop not found in {path}")


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", _DEMO_INVERSION_FILES)
def test_demo_inversion_block_uses_public_diagnostics(
    relative_path, tmp_path, monkeypatch, examples_dir, close_figures,
):
    """Demo inversion loop iteration plots from one diagnostics call."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import jax.numpy as jnp
    import os

    from TinyLensGpu.ObservationModel.LensImage.inversion_result import (
        InvalidReconstructionError,
    )
    from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import (
        PixelizedImageProbModelOperator,
    )
    from TinyLensGpu.PhysicalModel import (
        EPL, PhysicalModel, Shear, SIE, PixelizedSourceModel,
    )
    from TinyLensGpu.visualizer import overlay_critical_and_caustics

    path = examples_dir / "pix_src_demo_operator" / relative_path
    epl = EPL(theta_E=1.0, gamma=2.2, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
    shear = Shear(gamma1=0.05, gamma2=0.05)
    sie = SIE(theta_E=1.0, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
    mask = np.zeros((10, 10), dtype=bool)
    monkeypatch.chdir(tmp_path)
    namespace = {
        **_SUPPORT_IMPORTS,
        "np": np, "jnp": jnp, "plt": plt, "os": os,
        "InvalidReconstructionError": InvalidReconstructionError,
        "PixelizedImageProbModelOperator": PixelizedImageProbModelOperator,
        "PhysicalModel": PhysicalModel, "PixelizedSourceModel": PixelizedSourceModel,
        "SIE": SIE, "EPL": EPL, "Shear": Shear,
        "overlay_critical_and_caustics": overlay_critical_and_caustics,
        "image_data": np.full((10, 10), 0.05),
        "noise_map": np.full((10, 10), 0.1),
        "psf_kernel": _delta_kernel(),
        "mask": mask, "DPIX": 0.05,
        "sie": sie, "epl": epl, "shear": shear,
        "results_summary": [], "LAMBDA": 1.0,
    }
    exec(compile(ast.Module(body=_demo_loop_body(path), type_ignores=[]),
                 str(path), "exec"), namespace)
    assert namespace["results_summary"] and list(namespace["results_summary"][0]) == [
        "lambda", "log_ev", "chi2_nu",
    ]
    assert list(tmp_path.glob("output/demo_inversion_lambda*.png"))


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", _DEMO_INVERSION_FILES)
def test_demo_inversion_block_rejects_invalid_reconstruction(
    relative_path, tmp_path, monkeypatch, examples_dir, close_figures,
):
    """Demo loop raises InvalidReconstructionError before writing a figure."""
    import numpy as np

    import caskade as ck
    import jax.numpy as jnp

    from TinyLensGpu.ObservationModel.LensImage.inversion_result import (
        InvalidReconstructionError,
    )
    from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import (
        PixelizedImageProbModelOperator,
    )
    from TinyLensGpu.PhysicalModel import (
        EPL, PhysicalModel, Shear, SIE, PixelizedSourceModel,
    )

    path = examples_dir / "pix_src_demo_operator" / relative_path
    monkeypatch.chdir(tmp_path)
    epl = EPL(theta_E=1.0, gamma=2.2, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
    shear = Shear(gamma1=0.05, gamma2=0.05)
    sie = SIE(theta_E=1.0, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    from TinyLensGpu.visualizer import overlay_critical_and_caustics

    namespace = {
        **_SUPPORT_IMPORTS,
        "np": np, "jnp": jnp, "plt": plt, "os": os, "ck": ck,
        "InvalidReconstructionError": InvalidReconstructionError,
        "PixelizedImageProbModelOperator": PixelizedImageProbModelOperator,
        "PhysicalModel": PhysicalModel, "PixelizedSourceModel": PixelizedSourceModel,
        "SIE": SIE, "EPL": EPL, "Shear": Shear,
        "overlay_critical_and_caustics": overlay_critical_and_caustics,
        "image_data": np.full((10, 10), np.nan),
        "noise_map": np.full((10, 10), 0.1),
        "psf_kernel": _delta_kernel(),
        "mask": np.zeros((10, 10), dtype=bool), "DPIX": 0.05,
        "sie": sie, "epl": epl, "shear": shear,
        "results_summary": [], "LAMBDA": 1.0,
    }
    with pytest.raises(InvalidReconstructionError):
        exec(compile(ast.Module(body=_demo_loop_body(path), type_ignores=[]),
                     str(path), "exec"), namespace)
    assert not list(tmp_path.glob("output/*.png"))


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", _SIMPLE_PLOT_FITS)
def test_simple_fit_imports_keep_jax_backend(relative_path, examples_dir):
    """Original import order selects the JAX Caskade backend in a fresh process."""
    import os
    import subprocess
    import sys

    path = examples_dir / "pix_src_demo_operator" / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    imports = [
        ast.unparse(node)
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    script = "\n".join(imports) + """
import caskade
assert caskade.backend.backend == "jax", caskade.backend.backend
from TinyLensGpu.Inference import ParamU
import jax.numpy as jnp
p = ParamU("theta_E", 1.0)
assert isinstance(jnp.asarray(p.value), jnp.ndarray)
"""
    repo_root = examples_dir.parent
    env = {k: v for k, v in os.environ.items() if k != "CASKADE_BACKEND"}
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=repo_root, env=env,
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


_OPERATOR_PIPE_PLOTS = (
    "galan24_test/model.py",
    "galan24_test/model_predefine_src_box.py",
)


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", _OPERATOR_PIPE_PLOTS)
def test_pipe_plot_overlay_uses_fitted_parameters(
    relative_path, tmp_path, monkeypatch, examples_dir, close_figures,
):
    """_plot_pix_stage overlays ray-trace with the median parameters, not defaults."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import jax.numpy as jnp
    import caskade as ck

    from matplotlib.colors import LogNorm
    from TinyLensGpu.ObservationModel.LensImage.inversion_result import (
        InvalidReconstructionError,
    )
    from TinyLensGpu.visualizer import overlay_critical_and_caustics

    path = examples_dir / "pix_src_demo_operator" / "pipe" / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_plot_pix_stage"
    ]
    assert len(selected) == 1

    model = _small_source_export_model()
    captured = {}

    def overlay_probe(*, image_axes, source_ax, lens_mass):
        captured["theta_E"] = float(
            np.asarray(lens_mass.lens_mass[0].theta_E.value)
        )

    namespace = {
        **_SUPPORT_IMPORTS,
        "np": np, "jnp": jnp, "plt": plt, "ck": ck,
        "InvalidReconstructionError": InvalidReconstructionError,
        "overlay_critical_and_caustics": overlay_probe,
        "LogNorm": LogNorm, "DPIX": 0.08,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
         namespace)

    names = [p.name for p in model.get_dynamic_params()]
    medians = {
        n: float(np.asarray(p.value))
        for n, p in zip(names, model.get_dynamic_params())
    }
    medians["theta_E"] = 0.45  # distinct from the 0.12 construction value
    assert medians["theta_E"] != float(np.asarray(model.phys_model.lens_mass[0].theta_E.value))

    save_path = tmp_path / "stage_m_model.png"
    monkeypatch.chdir(tmp_path)
    namespace["_plot_pix_stage"](
        "stage-M", model, medians, names, str(save_path)
    )
    assert save_path.exists() and save_path.stat().st_size > 0
    assert captured["theta_E"] == pytest.approx(0.45)
    # Original library overlay still usable with the fitted context semantics.
    namespace["overlay_critical_and_caustics"] = overlay_critical_and_caustics


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", _SIMPLE_PLOT_FITS)
def test_simple_plot_overlay_uses_fitted_parameters(
    relative_path, tmp_path, monkeypatch, examples_dir, close_figures,
):
    """Simple fit overlays see the fitted theta_E, not construction defaults."""
    import numpy as np

    path = examples_dir / "pix_src_demo_operator" / relative_path
    namespace, tmp = _small_operator_namespace(tmp_path, monkeypatch)
    names = namespace["param_names"]
    assert "theta_E" in names
    q50 = list(namespace["q50_list"])
    q50[names.index("theta_E")] = 0.45  # distinct from construction value
    namespace["q50_list"] = q50

    captured = {}
    real_overlay = namespace["overlay_critical_and_caustics"]

    def overlay_probe(*, image_axes, source_ax, lens_mass):
        captured["theta_E"] = float(
            np.asarray(lens_mass.lens_mass[0].theta_E.value)
        )
        return real_overlay(
            image_axes=image_axes, source_ax=source_ax, lens_mass=lens_mass
        )

    namespace["overlay_critical_and_caustics"] = overlay_probe
    exec(compile(ast.Module(body=_simple_plot_block(path), type_ignores=[]),
                 str(path), "exec"), namespace)
    assert _plot_artifacts(tmp)
    assert captured["theta_E"] == pytest.approx(0.45)
