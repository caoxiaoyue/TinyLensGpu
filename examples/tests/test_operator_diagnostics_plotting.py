"""Regression checks for the operator tracer's diagnostic plotting block."""

import ast

import pytest


_TRACER_FIT = "pix_src_demo_operator/simple/pix_src/fit_lens_src.py"


def _tracer_plot_block(fit_path):
    """Return the top-level statements of the tracer reconstruction/plot block."""
    tree = ast.parse(fit_path.read_text(), filename=str(fit_path))
    for i, node in enumerate(tree.body):
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "best_params" for t in node.targets
            )
        ):
            return tree.body[i:]
    raise AssertionError(f"reconstruction block not found in {fit_path}")


@pytest.fixture
def close_figures():
    """Close all matplotlib figures after a tracer plot test."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _tracer_namespace(tmp_path, monkeypatch, image_data=None):
    """Build a small real operator model and the names the tracer block expects."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import jax.numpy as jnp

    from TinyLensGpu.Inference import ParamU
    from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import (
        PixelizedImageProbModelOperator,
    )
    from TinyLensGpu.PhysicalModel import PhysicalModel, SIE
    from TinyLensGpu.PhysicalModel.LensImage.Pixelized.Light import PixelizedSourceModel
    from TinyLensGpu.visualizer import overlay_critical_and_caustics

    sie = SIE(theta_E=0.12, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    for param in (sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y):
        param.to_static()
    sie.theta_E.to_dynamic()
    log_lambda = ParamU(
        "log_lambda_reg", 0.0,
        prior_type="uniform",
        prior_settings=[jnp.log(1e-3), jnp.log(1e3)],
        limits=[jnp.log(1e-6), jnp.log(1e6)],
    )
    log_lambda.to_dynamic()
    pix_src = PixelizedSourceModel(n=5, log_lambda_reg=log_lambda)
    phys_model = PhysicalModel(lens_mass=[sie], source_light=[pix_src], lens_light=[])

    mask = np.zeros((10, 10), dtype=bool)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
    data = np.full((10, 10), 0.05) if image_data is None else image_data
    prob_model = PixelizedImageProbModelOperator(
        image_data=data,
        noise_map=np.full((10, 10), 0.1),
        psf_kernel=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        dpix=0.08,
        phys_model=phys_model,
        mask=mask,
        nsub=1,
    )
    # The plotted image is the same observation the diagnostics describe.
    image_data = np.asarray(prob_model.image_data)

    out_dir = tmp_path / "output"
    out_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    param_names = [p.name for p in prob_model.get_dynamic_params()]
    q50_list = [float(np.asarray(p.value)) for p in prob_model.get_dynamic_params()]
    namespace = {
        "np": np,
        "jnp": jnp,
        "plt": plt,
        "prob_model": prob_model,
        "pix_src": pix_src,
        "q50_list": q50_list,
        "param_names": param_names,
        "image_data": image_data,
        "noise_map": np.asarray(prob_model.noise_map),
        "mask": mask,
        "DPIX": 0.08,
        "overlay_critical_and_caustics": overlay_critical_and_caustics,
    }
    return namespace, out_dir / "fit_map_visualization.png"


@pytest.mark.unit
def test_tracer_plot_block_uses_public_diagnostics(
    tmp_path, monkeypatch, examples_dir, close_figures
):
    """The migrated operator tracer plots a real small reconstruction."""
    fit_path = examples_dir / _TRACER_FIT
    block = _tracer_plot_block(fit_path)
    namespace, artifact = _tracer_namespace(tmp_path, monkeypatch)
    exec(
        compile(ast.Module(body=block, type_ignores=[]), str(fit_path), "exec"),
        namespace,
    )
    assert artifact.exists() and artifact.stat().st_size > 0


@pytest.mark.unit
def test_tracer_plot_block_rejects_invalid_reconstruction(
    tmp_path, monkeypatch, examples_dir, close_figures
):
    """A nonfinite reconstruction raises before any figure artifact is written."""
    import numpy as np

    fit_path = examples_dir / _TRACER_FIT
    block = _tracer_plot_block(fit_path)
    namespace, artifact = _tracer_namespace(
        tmp_path, monkeypatch, image_data=np.full((10, 10), np.nan)
    )
    with pytest.raises(RuntimeError, match="Cannot plot reconstruction"):
        exec(
            compile(ast.Module(body=block, type_ignores=[]), str(fit_path), "exec"),
            namespace,
        )
    assert not artifact.exists()
