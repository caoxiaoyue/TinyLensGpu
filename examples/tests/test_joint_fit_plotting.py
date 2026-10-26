"""Regression checks for joint-fit example plotting."""

import ast

import pytest


_JOINT_FIT_FILES = (
    "pix_src_demo/simple/pix_src_lens_light/fit_joint.py",
    "pix_src_demo/simple/pix_src_lens_light_mge/fit_joint.py",
)


@pytest.mark.unit
@pytest.mark.parametrize("fit_path", _JOINT_FIT_FILES, ids=["fit_path0", "fit_path1"])
def test_joint_fit_overlay_uses_correct_axes_and_parameter_context(fit_path, examples_dir):
    """The critical-curve overlay uses 2D axes indexing and the fitted context."""
    import TinyLensGpu  # noqa: F401 -- import first so caskade picks 'jax' backend
    import caskade as ck
    import numpy as np

    fit_path = examples_dir / fit_path
    tree = ast.parse(fit_path.read_text(), filename=str(fit_path))
    overlay_stmt = None
    for node in tree.body:
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == "overlay_critical_and_caustics"
            ):
                overlay_stmt = node
                break
        if overlay_stmt is not None:
            break
    assert overlay_stmt is not None, f"overlay call not found in {fit_path}"

    calls = []
    axes = np.asarray(
        [[object() for _ in range(4)] for _ in range(2)], dtype=object,
    )
    prob_model = ck.Module("overlay_probe")
    prob_model.phys_model = ck.Module("mass_probe")
    prob_model.phys_model.theta_E = ck.Param("theta_E", 1.0)
    prob_model.phys_model.theta_E.to_dynamic()

    def overlay_critical_and_caustics(*, image_axes, source_ax, lens_mass):
        calls.append(dict(
            image_axes=image_axes,
            source_ax=source_ax,
            lens_mass=lens_mass,
            theta_E=float(np.asarray(lens_mass.theta_E.value)),
        ))

    namespace = {
        "np": np, "ck": ck,
        "axes": axes,
        "prob_model": prob_model,
        "best_params": [1.1],
        "overlay_critical_and_caustics": overlay_critical_and_caustics,
    }
    exec(
        compile(
            ast.Module(body=[overlay_stmt], type_ignores=[]),
            str(fit_path), "exec",
        ),
        namespace,
    )
    assert len(calls) == 1
    call = calls[0]
    assert call["image_axes"] == [axes[0, 0], axes[0, 1], axes[0, 2]]
    assert call["source_ax"] is axes[0, 3]
    assert call["lens_mass"] is prob_model.phys_model
    assert call["theta_E"] == pytest.approx(1.1)


_TRACER_FIT = "pix_src_demo/simple/pix_src/fit_lens_src.py"


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
    """Build a small real dense model and the names the tracer block expects."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import jax.numpy as jnp

    from TinyLensGpu.Inference import ParamU
    from TinyLensGpu.ObservationModel.LensImage import PixelizedImageProbModel
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
    prob_model = PixelizedImageProbModel(
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
    """The migrated tracer block plots a real small dense reconstruction."""
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
