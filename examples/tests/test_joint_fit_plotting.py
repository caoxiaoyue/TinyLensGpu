"""Regression checks for joint-fit example plotting."""

import ast

from TinyLensGpu import example_support

# Script blocks execute in isolated namespaces; supply their shared imports.
_SUPPORT_IMPORTS = {
    name: getattr(example_support, name) for name in example_support.__all__
}

import pytest

from TinyLensGpu.ObservationModel.LensImage.inversion_result import (
    InvalidReconstructionError,
)


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
        **_SUPPORT_IMPORTS,
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
_REMAINING_SOURCE_FITS = (
    "pix_src_demo/simple/pix_src/fit_lens_src_pos_like.py",
    "pix_src_demo/simple/pix_src_epl_sh/fit_lens_src.py",
    "pix_src_demo/simple/pix_src_epl_sh/fit_lens_src_pos_like.py",
)


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


def _tracer_namespace(
    tmp_path, monkeypatch, image_data=None, *, joint=False, positions=False,
    lens_profile="gaussian", source_n=5,
):
    """Build a small real dense model and the names the tracer block expects."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import jax.numpy as jnp

    from TinyLensGpu.Inference import ParamU
    from TinyLensGpu.ObservationModel.LensImage import PixelizedImageProbModel
    from TinyLensGpu.PhysicalModel import (
        GaussianEllipse, PhysicalModel, SersicEllipse, SIE,
    )
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
    pix_src = PixelizedSourceModel(n=source_n, log_lambda_reg=log_lambda)
    lens_light = []
    if joint and lens_profile == "sersic":
        sersic = SersicEllipse(
            R_sersic=ParamU("R_sersic_lens", 0.13), n_sersic=2.0, Ie=1.0,
            e1=0.0, e2=0.0, center_x=0.0, center_y=0.0,
        )
        for param in (
            sersic.n_sersic, sersic.Ie, sersic.e1, sersic.e2,
            sersic.center_x, sersic.center_y,
        ):
            param.to_static()
        sersic.R_sersic.to_dynamic()
        lens_light.append(sersic)
    elif joint:
        gaussian = GaussianEllipse(
            flux=1.0, sigma=0.13, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0,
        )
        for param in (
            gaussian.flux, gaussian.sigma, gaussian.e1, gaussian.e2,
            gaussian.center_x, gaussian.center_y,
        ):
            param.to_static()
        lens_light.append(gaussian)
    phys_model = PhysicalModel(
        lens_mass=[sie], source_light=[pix_src], lens_light=lens_light,
    )

    mask = np.zeros((10, 10), dtype=bool)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
    data = np.full((10, 10), 0.05) if image_data is None else image_data
    if joint and image_data is None:
        y, x = np.mgrid[:10, :10]
        data = data + 0.4 * np.exp(-((x - 4.5) ** 2 + (y - 4.5) ** 2) / 5.0)
    prob_model = PixelizedImageProbModel(
        image_data=data,
        noise_map=np.full((10, 10), 0.1),
        psf_kernel=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        dpix=0.08,
        phys_model=phys_model,
        mask=mask,
        nsub=1,
        position_likelihood=(
            dict(
                positions=[[0.16, 0.04], [-0.16, 0.04]],
                threshold_arcsec=0.01, min_log_like=-1e6,
            )
            if positions else None
        ),
    )
    # The plotted image is the same observation the diagnostics describe.
    image_data = np.asarray(prob_model.image_data)

    out_dir = tmp_path / "output"
    out_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    param_names = [p.name for p in prob_model.get_dynamic_params()]
    q50_list = [float(np.asarray(p.value)) for p in prob_model.get_dynamic_params()]
    namespace = {
        **_SUPPORT_IMPORTS,
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
        "InvalidReconstructionError": InvalidReconstructionError,
    }
    return namespace, out_dir / "fit_map_visualization.png"


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", (_TRACER_FIT, *_REMAINING_SOURCE_FITS))
def test_tracer_plot_block_uses_public_diagnostics(
    relative_path, tmp_path, monkeypatch, examples_dir, close_figures
):
    """The migrated source-only fitting blocks plot real small reconstructions."""
    fit_path = examples_dir / relative_path
    block = _tracer_plot_block(fit_path)
    namespace, artifact = _tracer_namespace(tmp_path, monkeypatch)
    import caskade as ck

    namespace["ck"] = ck
    if relative_path.endswith("fit_lens_src_pos_like.py"):
        artifact = tmp_path / "output_pos_like" / artifact.name
        artifact.parent.mkdir()
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
    with pytest.raises(InvalidReconstructionError, match="plotting.*nonfinite"):
        exec(
            compile(ast.Module(body=block, type_ignores=[]), str(fit_path), "exec"),
            namespace,
        )
    assert not artifact.exists()


_JOINT_STAGE = "pix_src_demo/pipe/simult_fit_lens_source/mge_lens_light/model.py"
_SOURCE_STAGES = (
    "pix_src_demo/pipe/pre_subtract_lens/pix_src_pipe/model.py",
    "pix_src_demo/pipe/pre_subtract_lens/pix_src_pipe_mge/model.py",
    "pix_src_demo/pipe/pre_subtract_lens/pix_src_pipe_sersic/model.py",
    "pix_src_demo/pipe/slacs_examples/pipeline.py",
    "pix_src_demo_operator/pipe/no_lens_light/model_matrix.py",
)


def _load_stage_plot(path, namespace, name="_plot_stage_b"):
    """Load only the legacy drawing function, without executing the pipeline."""
    import caskade as ck

    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(selected) == 1
    namespace.update(
        ck=ck, N_GAUSSIANS_LENS=1,
        InvalidReconstructionError=InvalidReconstructionError,
    )
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
         namespace)
    return namespace[name]


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", (_JOINT_STAGE, *_SOURCE_STAGES))
def test_dense_stage_rejects_nonfinite_before_figure(
    relative_path, tmp_path, monkeypatch, examples_dir, close_figures
):
    """NaN observation data must not produce a formal reconstruction plot."""
    import numpy as np

    joint = relative_path == _JOINT_STAGE
    namespace, artifact = _tracer_namespace(
        tmp_path, monkeypatch, image_data=np.full((10, 10), np.nan), joint=joint,
    )
    plot = _load_stage_plot(
        examples_dir / relative_path, namespace,
        "_plot_stage_b" if joint else "_plot_pix_stage",
    )
    medians = dict(zip(namespace["param_names"], namespace["q50_list"]))

    def unexpected_figure(*args, **kwargs):
        pytest.fail("An invalid reconstruction reached figure creation")

    monkeypatch.setattr(namespace["plt"], "subplots", unexpected_figure)
    with pytest.raises(InvalidReconstructionError, match="plotting.*nonfinite"):
        plot("test", namespace["prob_model"], medians,
             namespace["param_names"], artifact)
    assert not artifact.exists()


@pytest.mark.unit
def test_dense_joint_stage_plots_fitted_components(
    tmp_path, monkeypatch, examples_dir, close_figures
):
    """A real joint plot displays fitted components summing to its total image."""
    import jax.numpy as jnp
    import numpy as np
    from matplotlib.axes import Axes

    namespace, artifact = _tracer_namespace(tmp_path, monkeypatch, joint=True)
    likelihood = namespace["prob_model"]
    names = namespace["param_names"]
    medians = dict(zip(names, namespace["q50_list"]))
    medians["theta_E"] = 0.17  # Distinct from the model's inactive defaults.
    theta = jnp.asarray([medians[name] for name in names])
    expected = likelihood.evaluate(theta)
    displayed = []
    original_imshow = Axes.imshow

    def capture_image(ax, image, *args, **kwargs):
        displayed.append((ax, np.asarray(image).copy()))
        return original_imshow(ax, image, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", capture_image)
    original_overlay = namespace["overlay_critical_and_caustics"]
    overlays = []

    def overlay_at_fitted_parameters(**kwargs):
        overlays.append(kwargs)
        mass = kwargs["lens_mass"].lens_mass[0]
        assert float(mass.theta_E.value) == pytest.approx(medians["theta_E"])
        original_overlay(**kwargs)

    namespace["overlay_critical_and_caustics"] = overlay_at_fitted_parameters
    plot = _load_stage_plot(examples_dir / _JOINT_STAGE, namespace)
    plot("test", likelihood, medians, names, artifact)

    assert artifact.exists() and artifact.stat().st_size > 0
    images = {ax.get_title().split("\n")[0]: image for ax, image in displayed}
    np.testing.assert_allclose(
        images["Joint model"], expected.model_image, rtol=2e-5, atol=2e-5,
    )
    lens_image = images["Lens light only (MGE, N=1)"]
    assert np.max(np.abs(lens_image)) > 0.01
    np.testing.assert_allclose(
        images["Lensed source only"] + lens_image,
        images["Joint model"], rtol=2e-5, atol=2e-5,
    )
    np.testing.assert_allclose(
        images["Source reconstruction"],
        np.asarray(expected.source_pixels).reshape(5, 5),
        rtol=2e-5, atol=2e-5,
    )
    assert len(overlays) == 1
    assert all(isinstance(ax, Axes) for ax in overlays[0]["image_axes"])
    assert isinstance(overlays[0]["source_ax"], Axes)


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", (*_REMAINING_SOURCE_FITS, *_JOINT_FIT_FILES))
def test_dense_simple_fit_rejects_nonfinite_before_figure(
    relative_path, tmp_path, monkeypatch, examples_dir, close_figures
):
    """Legacy top-level fitting scripts reject real NaN reconstructions."""
    import TinyLensGpu  # noqa: F401 -- select JAX before importing caskade
    import caskade as ck
    import numpy as np

    namespace, artifact = _tracer_namespace(
        tmp_path, monkeypatch, image_data=np.full((10, 10), np.nan),
        joint=relative_path in _JOINT_FIT_FILES,
    )
    namespace.update(ck=ck)

    def unexpected_figure(*args, **kwargs):
        pytest.fail("An invalid reconstruction reached figure creation")

    monkeypatch.setattr(namespace["plt"], "subplots", unexpected_figure)
    path = examples_dir / relative_path
    with pytest.raises(InvalidReconstructionError, match="plotting.*nonfinite"):
        exec(compile(ast.Module(body=_tracer_plot_block(path), type_ignores=[]),
                     str(path), "exec"), namespace)
    assert not artifact.exists()
    assert not (tmp_path / "output_pos_like" / artifact.name).exists()


_INVERSION_DEMOS = (
    "pix_src_demo/simple/pix_src/demo_inversion.py",
    "pix_src_demo/simple/pix_src_epl_sh/demo_inversion.py",
)


def _demo_plot_block(path):
    """Extract one lambda's evaluation and output, not its model/data setup."""
    tree = ast.parse(path.read_text(), filename=str(path))
    loop = next(
        node for node in tree.body
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
        and node.target.id == "LAMBDA"
    )
    for i, node in enumerate(loop.body):
        if isinstance(node, ast.Assign) and any(
            isinstance(name, ast.Name) and name.id == "result"
            for target in node.targets for name in ast.walk(target)
        ):
            return loop.body[i:]
    raise AssertionError(f"Reconstruction block not found in {path}")


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", _INVERSION_DEMOS)
@pytest.mark.parametrize("invalid", [False, True])
def test_dense_inversion_demo_outputs_only_valid_reconstruction(
    relative_path, invalid, tmp_path, monkeypatch, examples_dir, close_figures
):
    """Demo plots and summaries retain chi2/pixel, but never export NaN output."""
    import os
    import numpy as np

    namespace, _ = _tracer_namespace(
        tmp_path, monkeypatch,
        image_data=np.full((10, 10), np.nan) if invalid else None,
    )
    namespace.update(os=os, LAMBDA=1.0, results_summary=[])
    path = examples_dir / relative_path
    block = compile(ast.Module(body=_demo_plot_block(path), type_ignores=[]),
                    str(path), "exec")
    artifact = tmp_path / "output" / "demo_inversion_lambda1.0.png"
    if invalid:
        def unexpected_figure(*args, **kwargs):
            pytest.fail("An invalid reconstruction reached figure creation")

        monkeypatch.setattr(namespace["plt"], "subplots", unexpected_figure)
        with pytest.raises(InvalidReconstructionError, match="plotting.*nonfinite"):
            exec(block, namespace)
        assert namespace["results_summary"] == []
        assert not artifact.exists()
    else:
        model = namespace["prob_model"]
        result = model.evaluate()
        residual = np.asarray((model.image_data - result.model_image) / model.noise_map)
        expected_chi2_per_pixel = np.mean(residual[np.asarray(model.unmask)] ** 2)
        exec(block, namespace)
        assert artifact.exists() and artifact.stat().st_size > 0
        summary, = namespace["results_summary"]
        assert set(summary) == {"lambda", "log_ev", "chi2_nu"}
        assert summary["lambda"] == 1.0
        assert summary["log_ev"] == pytest.approx(float(result.log_evidence))
        assert summary["chi2_nu"] == pytest.approx(expected_chi2_per_pixel, rel=2e-5)


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", _SOURCE_STAGES)
def test_dense_source_stage_preserves_images_and_position_overlays(
    relative_path, tmp_path, monkeypatch, examples_dir, close_figures
):
    """Source-stage plots retain residual masks and the fitted overlay context."""
    import jax.numpy as jnp
    import numpy as np
    from matplotlib.axes import Axes

    namespace, artifact = _tracer_namespace(tmp_path, monkeypatch, positions=True)
    import caskade as ck

    likelihood = namespace["prob_model"]
    names = namespace["param_names"]
    medians = dict(zip(names, namespace["q50_list"]))
    medians["theta_E"] = 0.17
    theta = jnp.asarray([medians[name] for name in names])
    result = likelihood.evaluate(theta)
    displayed = []
    original_imshow = Axes.imshow

    def capture_image(ax, image, *args, **kwargs):
        displayed.append((ax, np.asarray(image).copy()))
        return original_imshow(ax, image, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", capture_image)
    original_overlay = namespace["overlay_critical_and_caustics"]
    overlay_parameters = []

    def overlay_at_fitted_parameters(**kwargs):
        overlay_parameters.append(float(kwargs["lens_mass"].lens_mass[0].theta_E.value))
        original_overlay(**kwargs)

    namespace["overlay_critical_and_caustics"] = overlay_at_fitted_parameters
    plot = _load_stage_plot(examples_dir / relative_path, namespace, "_plot_pix_stage")
    plot("test", likelihood, medians, names, artifact)
    assert artifact.exists() and artifact.stat().st_size > 0
    assert overlay_parameters == pytest.approx([medians["theta_E"]])
    assert len(displayed) == 4
    np.testing.assert_allclose(displayed[1][1], result.model_image, rtol=2e-5, atol=2e-5)
    active = np.asarray(likelihood.unmask)
    expected_residual = np.asarray(
        (likelihood.image_data - result.model_image) / likelihood.noise_map,
    )
    assert np.all(np.isnan(displayed[2][1][~active]))
    np.testing.assert_allclose(
        displayed[2][1][active], expected_residual[active], rtol=2e-5, atol=2e-5,
    )
    np.testing.assert_allclose(
        displayed[3][1], np.asarray(result.source_pixels).reshape(5, 5),
        rtol=2e-5, atol=2e-5,
    )
    if relative_path.endswith("model_matrix.py"):
        image_positions = np.asarray([[0.16, 0.04], [-0.16, 0.04]])
        with ck.ActiveContext(likelihood):
            likelihood.fill_params(theta)
            bx, by = likelihood.phys_model.deflection(
                jnp.asarray(image_positions[:, 0]), jnp.asarray(image_positions[:, 1]),
            )
        for ax, expected, label in (
            (displayed[0][0], image_positions, "Marked pos"),
            (displayed[3][0], np.stack((bx, by), axis=-1), "Traced pos"),
        ):
            line, = [line for line in ax.lines if line.get_label() == label]
            np.testing.assert_allclose(
                np.stack(line.get_data(), axis=-1), expected, rtol=2e-5, atol=2e-5,
            )


@pytest.mark.unit
def test_dense_stage_propagates_invalid_reconstruction_and_keeps_posterior(
    tmp_path, examples_dir, close_figures,
):
    """A real NaN reconstruction aborts the stage after preserving its posterior."""
    import pickle
    import time

    import caskade as ck
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    from TinyLensGpu.Inference import ParamU, StagePosterior
    from TinyLensGpu.Inference.build_likelihood import make_likelihood
    from TinyLensGpu.Inference.build_prior import make_prior_transformation
    from TinyLensGpu.ObservationModel.LensImage import PixelizedImageProbModel
    from TinyLensGpu.PhysicalModel import PhysicalModel, PixelizedSourceModel, Shear
    from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import EPL
    from TinyLensGpu.utils.misc import weighted_quantile
    from TinyLensGpu.visualizer import overlay_critical_and_caustics

    sampler_instances = []

    class ControlledSampler:
        """Control only the external sampler; keep assembly, inversion and IO real."""
        log_z = -12.0

        def __init__(self, prior, loglike, *, n_dim, **kwargs):
            self.samples = np.tile(np.asarray(prior(jnp.full(n_dim, 0.5))), (3, 1))
            sampler_instances.append(self)

        def run(self, *, verbose, n_eff):
            pass

        def posterior(self):
            return self.samples, np.zeros(3), np.zeros(3)

    path = examples_dir / "pix_src_demo/pipe/slacs_examples/pipeline.py"
    selected_names = {
        "_run_sampler", "_dump_stage", "_load_stage", "_posterior_median",
        "_print_summary", "_epl_mass_from_stage_a",
        "build_stage_m_likelihood", "_plot_pix_stage", "run_stage_m",
    }
    selected = [
        node for node in ast.parse(path.read_text(), filename=str(path)).body
        if isinstance(node, ast.FunctionDef) and node.name in selected_names
    ]
    assert len(selected) == len(selected_names)
    namespace = dict(
        _SUPPORT_IMPORTS,
        np=np, jnp=jnp, plt=plt, ck=ck, pickle=pickle, time=time,
        ParamU=ParamU, StagePosterior=StagePosterior,
        make_likelihood=make_likelihood,
        make_prior_transformation=make_prior_transformation,
        PixelizedImageProbModel=PixelizedImageProbModel,
        PhysicalModel=PhysicalModel,
        PixelizedSourceModel=PixelizedSourceModel, Shear=Shear, EPL=EPL,
        weighted_quantile=weighted_quantile,
        overlay_critical_and_caustics=overlay_critical_and_caustics,
        InvalidReconstructionError=InvalidReconstructionError,
        Sampler=ControlledSampler, OUT_DIR=tmp_path / "output",
        DPIX=0.08, NSUB_PIX=1, NSRC=5,
    )
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
         namespace)

    names_a = [
        "theta_E", "e1_mass", "e2_mass", "center_x_mass", "center_y_mass",
        "gamma1", "gamma2",
    ]
    center = np.asarray([0.17, 0.1, 0.03, 0.0, 0.0, 0.0, 0.0])
    samples_a = np.stack((center - 0.001, center, center + 0.001))
    position_likelihood = dict(
        positions=[[0.16, 0.04], [-0.16, 0.04]],
        threshold_arcsec=0.01, min_log_like=-1e6,
    )
    with pytest.raises(InvalidReconstructionError, match="nonfinite"):
        namespace["run_stage_m"](
            np.full((10, 10), np.nan), np.full((10, 10), 0.1), np.ones((1, 1)),
            np.zeros((10, 10), dtype=bool),
            np.zeros((10, 10)), samples_a, np.ones(3) / 3, names_a, position_likelihood,
        )

    stage_path = tmp_path / "output" / "stage_m.pkl"
    assert stage_path.exists() and stage_path.stat().st_size > 0
    saved = namespace["_load_stage"]("m")
    assert len(sampler_instances) == 1
    np.testing.assert_array_equal(saved["samples"], sampler_instances[0].samples)
    np.testing.assert_allclose(saved["weights"], np.ones(3) / 3)
    assert saved["log_z"] == ControlledSampler.log_z
    assert saved["extra"]["medians"]
    assert not (tmp_path / "output" / "stage_m_model.png").exists()


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", _JOINT_FIT_FILES)
@pytest.mark.parametrize("source_n", [5, 8])
def test_simple_joint_preserves_raw_dof_and_fitted_component_images(
    relative_path, source_n, tmp_path, monkeypatch, examples_dir, close_figures,
):
    """Simple joint plots keep N_active-N_source-N_lens, not curvature DOF."""
    import caskade as ck
    import jax.numpy as jnp
    import numpy as np
    from matplotlib.axes import Axes

    namespace, artifact = _tracer_namespace(
        tmp_path, monkeypatch, joint=True, source_n=source_n,
        lens_profile="sersic" if relative_path == _JOINT_FIT_FILES[0] else "gaussian",
    )
    namespace.update(ck=ck, N_gaussians_lens=1)
    model = namespace["prob_model"]
    theta = jnp.asarray(namespace["q50_list"])
    result = model.evaluate(theta)
    active = np.asarray(model.unmask)
    residual = np.asarray((model.image_data - result.model_image) / model.noise_map)
    expected_chi2 = float(np.sum(residual[active] ** 2))
    # The fixture has an 8x8 active image and one fitted lens-light amplitude.
    expected_dof = 64 - source_n**2 - 1
    expected_reduced_chi2 = expected_chi2 / expected_dof if expected_dof > 0 else 0.0

    displayed = []
    original_imshow = Axes.imshow

    def capture_image(ax, image, *args, **kwargs):
        displayed.append((ax, np.asarray(image).copy()))
        return original_imshow(ax, image, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", capture_image)
    path = examples_dir / relative_path
    exec(compile(ast.Module(body=_tracer_plot_block(path), type_ignores=[]),
                 str(path), "exec"), namespace)

    assert artifact.exists() and artifact.stat().st_size > 0
    assert namespace["dof"] == expected_dof
    assert namespace["chi2_nu"] == pytest.approx(
        expected_reduced_chi2, rel=2e-5, abs=2e-7,
    )
    diagnostics = namespace["diagnostics"]
    assert diagnostics.effective_parameters is None
    assert diagnostics.residual_dof is None
    assert diagnostics.reduced_chi2 is None
    images = {ax.get_title().split("\n")[0]: image for ax, image in displayed}
    total = images["Joint model (source + lens light)"]
    lens_key = "Lens light only" if relative_path == _JOINT_FIT_FILES[0] else "Lens light only (MGE, N=1)"
    lens_image = images[lens_key]
    assert np.max(np.abs(lens_image)) > 0.01
    np.testing.assert_allclose(total, result.model_image, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        images["Lensed source only"] + lens_image, total, rtol=2e-5, atol=2e-5,
    )
    np.testing.assert_allclose(
        images["Source reconstruction"],
        np.asarray(result.source_pixels).reshape(source_n, source_n),
        rtol=2e-5, atol=2e-5,
    )
    residual_ax, = [
        ax for ax, _ in displayed if ax.get_title().startswith("Norm. residual")
    ]
    assert f"chi^2/ν = {expected_reduced_chi2:.3f}" in residual_ax.get_title()
