"""Regression checks for pixelized pipeline source-package exports."""

import ast

import pytest

import TinyLensGpu  # noqa: F401 -- select the JAX backend before importing caskade


_PIPE_PACKAGE_HELPERS = (
    "no_lens_light/model.py",
    "no_lens_light/model_adpt_reg.py",
    "pre_subtract_lens/pix_src_pipe/model.py",
    "pre_subtract_lens/pix_src_pipe_sersic/model.py",
    "pre_subtract_lens/pix_src_pipe_mge/model.py",
    "slacs_examples/pipeline.py",
    "euclid_examples/pipeline.py",
    "galan24_test/model.py",
    "galan24_test/model_predefine_src_box.py",
)


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
    namespace = {"np": np, "jnp": jnp, "ck": ck, "NSRC": 3}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
         namespace)
    return namespace["_solve_pixel_source_for_package"]


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", _PIPE_PACKAGE_HELPERS)
@pytest.mark.parametrize("converged", [True, False])
def test_source_package_helper_uses_public_evaluate(relative_path, converged, examples_dir):
    """Pipe source-package helpers go through one public evaluate() call."""
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

    class ProbeLikelihood(ck.Module):
        solver_type = "pcg"

        def __init__(self):
            super().__init__("probe_likelihood")
            self.amplitude = ck.Param("amplitude", 0.0)

        @ck.forward
        def evaluate(self):
            calls.append(None)
            pixels = jnp.arange(9, dtype=jnp.float32) + self.amplitude.value
            info = SimpleNamespace(
                converged=self._converged,
                residual_norm=0.1,
                convergence_metric=0.1,
                n_iter=3,
            )
            return InversionResult(
                jnp.asarray(0.0), jnp.zeros((3, 3)), pixels, None,
                (-0.3, 0.3, -0.3, 0.3), info,
            )

    probe = ProbeLikelihood()
    probe.amplitude.to_dynamic()
    object.__setattr__(probe, "_converged", converged)

    if not converged:
        with pytest.raises(RuntimeError, match="failed"):
            helper(probe, {"amplitude": 2.5}, ["amplitude"])
        assert len(calls) == 1
        return

    package = helper(probe, {"amplitude": 2.5}, ["amplitude"])
    assert len(calls) == 1
    assert set(package) == {
        "source_pixels", "source_image", "source_bbox",
        "source_x_axis", "source_y_axis", "n",
    }
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


@pytest.fixture(scope="module")
def small_source_export_model():
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
    image = sim.simulate(jnp.linspace(0.1,1.0,9), source_bbox=bbox)
    return PixelizedImageProbModelOperator(
        image, jnp.ones_like(image)*0.05, psf, 0.08, phys,
        solver_type="pcg", block_size=2, fixed_source_bbox=bbox,
        fixed_reg_template=jnp.abs(jnp.linspace(-1.0,1.0,9)),
    )


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
    assert set(package) == {"source_pixels","source_image","source_bbox","source_x_axis","source_y_axis","n"}
    assert package["n"] == 3
    assert package["source_pixels"].dtype == np.float64
    np.testing.assert_allclose(package["source_pixels"],expected_source,rtol=2e-5,atol=2e-5)
    np.testing.assert_allclose(package["source_image"],expected_source.reshape(3,3),rtol=2e-5,atol=2e-5)
    np.testing.assert_allclose(package["source_bbox"],expected_bbox,rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(package["source_x_axis"],np.linspace(expected_bbox[0],expected_bbox[1],3),rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(package["source_y_axis"],np.linspace(expected_bbox[2],expected_bbox[3],3),rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(np.asarray(model.get_values("flat")),initial,rtol=0,atol=0)
