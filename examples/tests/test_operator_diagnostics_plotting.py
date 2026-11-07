"""Runtime checks at the operator examples' plotting and export seams."""

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
    import caskade as ck

    namespace = {
        **_SUPPORT_IMPORTS,
        "ck": ck,
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
    with pytest.raises(RuntimeError, match="plot.*reconstruction"):
        exec(
            compile(ast.Module(body=block, type_ignores=[]), str(fit_path), "exec"),
            namespace,
        )
    assert not artifact.exists()


_JOINT_PIPELINE = (
    "pix_src_demo_operator/pipe/simult_fit_lens_source/mge_lens_light/model.py"
)
_SIMPLE_JOINT = [
    f"pix_src_demo_operator/simple/{case}/fit_joint.py"
    for case in ("pix_src_lens_light", "pix_src_lens_light_mge")
]


def _load_pipeline(examples_dir, relative_path, tmp_path, monkeypatch):
    """Import a historical pipeline without leaking its working-directory change."""
    import importlib.util

    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location(
        "operator_plot_test_" + relative_path.replace("/", "_"),
        examples_dir / relative_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "DPIX", 0.08)
    if relative_path == _JOINT_PIPELINE:
        monkeypatch.setattr(module, "N_GAUSSIANS_LENS", 1)
    return module


def _small_joint_operator(
    *, failure=None, positions=False, full_params=False, adaptive=False,
):
    """Real masked joint PCG problem with a static unit-amplitude Gaussian basis."""
    import jax.numpy as jnp
    import numpy as np

    from TinyLensGpu.Inference import ParamU
    from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import (
        PixelizedImageProbModelOperator,
    )
    from TinyLensGpu.PhysicalModel import (
        GaussianEllipse, PhysicalModel, PixelizedSourceModel, SIE,
    )

    mass = SIE(theta_E=0.12, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    for param in (mass.theta_E, mass.e1, mass.e2, mass.center_x, mass.center_y):
        param.to_static()
    mass.theta_E.to_dynamic()
    log_lambda = ParamU("log_lambda_reg", 0.0)
    log_lambda.to_dynamic()
    source = PixelizedSourceModel(
        n=5, log_lambda_reg=log_lambda, adaptive_reg_rho=1.0 if adaptive else 0.0,
    )
    light = GaussianEllipse(
        flux=1.0, sigma=0.15, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0,
    )
    for param in (
        light.flux, light.sigma, light.e1, light.e2, light.center_x, light.center_y,
    ):
        param.to_static()
    if full_params:
        for param in (mass.e1, mass.e2, mass.center_x, mass.center_y, light.sigma):
            param.to_dynamic()
    phys = PhysicalModel(lens_mass=[mass], source_light=[source], lens_light=[light])
    yy, xx = np.mgrid[:10, :10]
    data = 0.05 + 0.2 * np.exp(-((xx - 4.5)**2 + (yy - 4.5)**2) / 5.0)
    if failure == "nonfinite":
        data[:] = np.nan
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
    likelihood = PixelizedImageProbModelOperator(
        image_data=data, noise_map=np.full((10, 10), 0.01),
        psf_kernel=np.ones((1, 1)), dpix=0.08, phys_model=phys,
        mask=mask, nsub=1, solver_type="pcg",
        fixed_source_bbox=(-0.3, 0.3, -0.3, 0.3) if adaptive else None,
        fixed_reg_template=np.abs(np.linspace(-1, 1, 25)) if adaptive else None,
        position_likelihood=(
            dict(positions=[[-0.2, 0.0], [0.2, 0.0]],
                 threshold_arcsec=0.1, min_log_like=-10.0)
            if positions else None
        ),
    )
    if failure == "unconverged":
        likelihood.pcg_max_iter = 0
    medians = {p.name: float(np.asarray(p.value)) for p in likelihood.get_dynamic_params()}
    # Distinct from the initial mass: plotted overlays must use the fitted context.
    medians[mass.theta_E.name] = 0.14
    theta = jnp.asarray(list(medians.values()))
    return likelihood, medians, theta


def _capture_saved_figures(monkeypatch):
    """Inspect rendered artists while still writing a real artifact."""
    import matplotlib.pyplot as plt

    figures = []
    savefig = plt.savefig

    def save(*args, **kwargs):
        figures.append(plt.gcf())
        return savefig(*args, **kwargs)

    monkeypatch.setattr(plt, "savefig", save)
    return figures


@pytest.mark.unit
def test_joint_pipeline_plots_fitted_components_and_unavailable_statistics(
    tmp_path, monkeypatch, examples_dir, close_figures, capsys,
):
    """Unsupported effective DOF does not discard a valid joint reconstruction."""
    import numpy as np

    module = _load_pipeline(examples_dir, _JOINT_PIPELINE, tmp_path, monkeypatch)
    likelihood, medians, theta = _small_joint_operator(positions=True)
    result, diagnostics = likelihood.evaluate_with_diagnostics(
        theta, include_components=True, include_effective_dof=True,
    )
    assert diagnostics.valid
    assert diagnostics.effective_parameters is None
    assert diagnostics.residual_dof is None
    assert diagnostics.reduced_chi2 is None
    figures = _capture_saved_figures(monkeypatch)
    artifact = tmp_path / "joint.png"
    module._plot_stage_b("stage-B", likelihood, medians, list(medians), str(artifact))
    assert artifact.exists() and artifact.stat().st_size > 0
    axes = figures[0].axes[:8]
    total = np.asarray(axes[1].images[0].get_array())
    source = np.asarray(axes[4].images[0].get_array())
    lens = np.asarray(axes[5].images[0].get_array())
    np.testing.assert_allclose(total, result.model_image, atol=1e-6)
    np.testing.assert_allclose(source, diagnostics.source_image, atol=1e-6)
    np.testing.assert_allclose(lens, diagnostics.lens_light_image, atol=1e-6)
    active = np.asarray(likelihood.unmask)
    np.testing.assert_allclose((source + lens)[active], total[active], atol=1e-6)
    assert np.any(source[active] != 0) and np.any(lens[active] != 0)
    assert f"total flux={float(np.sum(result.lens_amplitudes)):.3f}" in axes[5].get_title()
    title = axes[2].get_title()
    assert f"chi^2 = {diagnostics.chi2:.3f}" in title
    assert "unavailable" in title.lower()
    assert diagnostics.effective_dof_reason in capsys.readouterr().out
    np.testing.assert_allclose(axes[0].collections[0].get_offsets(),
                               diagnostics.position.image_positions)
    _assert_fitted_critical_curve(axes[1], 0.14)


def _run_joint_plot(relative_path, likelihood, medians, examples_dir, tmp_path, monkeypatch):
    import time
    import caskade as ck
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    from TinyLensGpu.visualizer import overlay_critical_and_caustics

    if relative_path == _JOINT_PIPELINE:
        module = _load_pipeline(examples_dir, relative_path, tmp_path, monkeypatch)
        module._plot_stage_b(
            "stage-B", likelihood, medians, list(medians), str(tmp_path / "joint.png"),
        )
        return tmp_path / "joint.png"
    fit_path = examples_dir / relative_path
    namespace = dict(
        _SUPPORT_IMPORTS,
        np=np, jnp=jnp, plt=plt, ck=ck, time=time, script_t0=time.time(),
        prob_model=likelihood, q50_list=list(medians.values()),
        param_names=list(medians), image_data=np.asarray(likelihood.image_data),
        noise_map=np.asarray(likelihood.noise_map),
        mask=~np.asarray(likelihood.unmask), DPIX=0.08,
        N_gaussians_lens=1, overlay_critical_and_caustics=overlay_critical_and_caustics,
        InvalidReconstructionError=InvalidReconstructionError,
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output").mkdir(exist_ok=True)
    exec(
        compile(ast.Module(body=_tracer_plot_block(fit_path), type_ignores=[]),
                str(fit_path), "exec"),
        namespace,
    )
    return tmp_path / "output" / "fit_map_visualization.png"


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", [_JOINT_PIPELINE, *_SIMPLE_JOINT])
@pytest.mark.parametrize("failure", ["unconverged", "nonfinite"])
def test_joint_plots_reject_failed_reconstruction_before_output(
    tmp_path, monkeypatch, examples_dir, close_figures, relative_path, failure,
):
    """Real failed PCG/NaN inputs cannot create either figures or diagnostics CSVs."""
    import matplotlib.pyplot as plt

    likelihood, medians, _ = _small_joint_operator(failure=failure, full_params=True)
    with pytest.raises(InvalidReconstructionError, match="Reconstruction failed while plotting"):
        _run_joint_plot(relative_path, likelihood, medians, examples_dir, tmp_path, monkeypatch)
    assert not list(tmp_path.rglob("*.png"))
    assert not list(tmp_path.rglob("*.csv"))
    assert not plt.get_fignums()


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", _SIMPLE_JOINT)
def test_simple_joint_plots_preserve_per_pixel_and_arc_summaries(
    tmp_path, monkeypatch, examples_dir, close_figures, relative_path,
):
    import numpy as np

    likelihood, medians, theta = _small_joint_operator(full_params=True)
    result, diagnostics = likelihood.evaluate_with_diagnostics(theta, include_components=True)
    assert diagnostics.valid
    figures = _capture_saved_figures(monkeypatch)
    artifact = _run_joint_plot(
        relative_path, likelihood, medians, examples_dir, tmp_path, monkeypatch,
    )
    assert artifact.exists() and artifact.stat().st_size > 0
    axes = figures[0].axes[:8]
    source = np.asarray(axes[4].images[0].get_array())
    lens = np.asarray(axes[5].images[0].get_array())
    np.testing.assert_allclose(source, diagnostics.source_image, atol=1e-6)
    np.testing.assert_allclose(lens, diagnostics.lens_light_image, atol=1e-6)
    np.testing.assert_allclose(source + lens, result.model_image, atol=1e-6)
    active = np.asarray(likelihood.unmask)
    # Compare the saved summary with the displayed reconstruction, avoiding
    # amplification of repeat-solve GPU roundoff by the small noise values.
    residuals = (
        np.asarray(likelihood.image_data) - np.asarray(axes[1].images[0].get_array())
    ) / np.asarray(likelihood.noise_map)
    chi2_per_pixel = np.mean(residuals[active]**2)
    assert f"chi^2/Ndata = {chi2_per_pixel:.3f}" in axes[2].get_title()
    assert "DOF" not in axes[2].get_title()
    arc = active & (source / np.asarray(likelihood.noise_map) > 3)
    assert arc.sum() > 1
    expected = [
        chi2_per_pixel, np.mean(residuals[arc]), np.std(residuals[arc]),
        np.corrcoef(residuals[arc], (source / np.asarray(likelihood.noise_map))[arc])[0, 1],
    ]
    csv_path = tmp_path / "output" / "fit_diagnostics.csv"
    assert csv_path.read_text().splitlines()[0] == (
        "chi2_per_data,arc_resid_mean,arc_resid_std,arc_template_corr"
    )
    np.testing.assert_allclose(np.loadtxt(csv_path, skiprows=1, delimiter=","), expected,
                               rtol=1e-5, atol=1e-6)
    _assert_fitted_critical_curve(axes[1], 0.14)


_SUBHALO_PIPELINES = [
    f"pix_src_demo_operator/pipe/{case}/model.py"
    for case in ("subhalo_0946", "subhalo_B1938")
]


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", _SUBHALO_PIPELINES)
@pytest.mark.parametrize("failure", ["unconverged", "nonfinite"])
def test_subhalo_plots_reject_failed_reconstruction_before_output(
    tmp_path, monkeypatch, examples_dir, close_figures, relative_path, failure,
):
    import matplotlib.pyplot as plt

    module = _load_pipeline(examples_dir, relative_path, tmp_path, monkeypatch)
    likelihood, medians, _ = _small_joint_operator(failure=failure)
    destination = tmp_path / "failed.png"
    with pytest.raises(InvalidReconstructionError, match="Reconstruction failed while plotting stage-M1"):
        module._plot_pix_stage("stage-M1", likelihood, medians, list(medians), str(destination))
    assert not destination.exists()
    assert not plt.get_fignums()


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", _SUBHALO_PIPELINES)
@pytest.mark.parametrize("failure", ["unconverged", "nonfinite"])
def test_subhalo_exports_reject_failed_reconstruction(
    tmp_path, monkeypatch, examples_dir, relative_path, failure,
):
    """No invalid source dictionary can be passed to the next pipeline stage."""
    module = _load_pipeline(examples_dir, relative_path, tmp_path, monkeypatch)
    monkeypatch.setattr(module, "NSRC", 5)
    likelihood, medians, theta = _small_joint_operator(failure=failure)
    _, diagnostics = likelihood.evaluate_with_diagnostics(theta)
    assert not diagnostics.valid
    with pytest.raises(InvalidReconstructionError, match="Reconstruction failed while exporting") as error:
        module._solve_pixel_source_for_package(likelihood, medians, list(medians))
    assert diagnostics.invalid_reason in str(error.value)
    assert not list(tmp_path.iterdir())


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", _SUBHALO_PIPELINES)
def test_subhalo_exports_keep_raw_package_contract_with_unsupported_joint_stats(
    tmp_path, monkeypatch, examples_dir, relative_path,
):
    import numpy as np

    module = _load_pipeline(examples_dir, relative_path, tmp_path, monkeypatch)
    monkeypatch.setattr(module, "NSRC", 5)
    likelihood, medians, theta = _small_joint_operator()
    result, diagnostics = likelihood.evaluate_with_diagnostics(theta, include_effective_dof=True)
    assert diagnostics.valid and diagnostics.reduced_chi2 is None
    assert diagnostics.effective_dof_method == "operator_joint_unsupported"
    package = module._solve_pixel_source_for_package(likelihood, medians, list(medians))
    keys = {
        "source_pixels", "source_image", "source_bbox",
        "source_x_axis", "source_y_axis", "n",
    }
    if "subhalo_0946" in relative_path:
        keys.update(("source_interpolation", "source_boundary"))
    assert set(package) == keys
    assert package["n"] == 5
    assert isinstance(package["source_bbox"], tuple)
    assert all(type(value) is float for value in package["source_bbox"])
    for key in ("source_pixels", "source_image", "source_x_axis", "source_y_axis"):
        assert isinstance(package[key], np.ndarray)
        assert package[key].dtype == np.float64
    np.testing.assert_allclose(package["source_pixels"], result.source_pixels, atol=1e-6)
    np.testing.assert_allclose(package["source_image"], np.asarray(result.source_pixels).reshape(5, 5),
                               atol=1e-6)
    np.testing.assert_allclose(package["source_bbox"], result.source_bbox, atol=1e-6)
    xmin, xmax, ymin, ymax = package["source_bbox"]
    np.testing.assert_allclose(package["source_x_axis"], np.linspace(xmin, xmax, 5))
    np.testing.assert_allclose(package["source_y_axis"], np.linspace(ymin, ymax, 5))


def _assert_fitted_critical_curve(axis, theta_e):
    """The circular SIE's outer critical line has its fitted Einstein radius."""
    import numpy as np

    curves = [line for line in axis.lines if len(line.get_xdata()) > 10]
    assert curves
    radius = max(np.median(np.hypot(line.get_xdata(), line.get_ydata())) for line in curves)
    assert radius == pytest.approx(theta_e, abs=0.003)


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", _SUBHALO_PIPELINES)
def test_subhalo_plots_preserve_position_scale_and_subhalo_overlays(
    tmp_path, monkeypatch, examples_dir, close_figures, capsys, relative_path,
):
    import numpy as np

    module = _load_pipeline(examples_dir, relative_path, tmp_path, monkeypatch)
    likelihood, medians, theta = _small_joint_operator(positions=True, adaptive=True)
    result, diagnostics = likelihood.evaluate_with_diagnostics(theta)
    assert diagnostics.valid
    figures = _capture_saved_figures(monkeypatch)
    names = list(medians)
    medians = {**medians, "center_x_subhalo": 0.13, "center_y_subhalo": -0.09}
    artifact = tmp_path / "subhalo.png"
    module._plot_pix_stage("stage-M5", likelihood, medians, names, str(artifact))
    assert artifact.exists() and artifact.stat().st_size > 0
    axes = figures[0].axes[:5]
    np.testing.assert_allclose(axes[1].images[0].get_array(), result.model_image, atol=1e-6)
    np.testing.assert_allclose(axes[4].images[0].get_array(), diagnostics.regularization_scale)
    assert np.ptp(axes[4].images[0].get_array()) > 0
    assert f"rho={diagnostics.adaptive_reg_rho:.2f}" in axes[4].get_title()
    assert f"λ={diagnostics.lambda_reg:.2e}" in axes[3].get_title()
    assert f"χ²={diagnostics.chi2:.1f}" in axes[2].get_title()
    for axis, label, expected in (
        (axes[0], "Marked pos", diagnostics.position.image_positions),
        (axes[3], "Traced pos", diagnostics.position.source_positions),
        (axes[1], "Pseudo-Jaffe subhalo", [[0.13, -0.09]]),
    ):
        line = next(line for line in axis.lines if line.get_label() == label)
        np.testing.assert_allclose(np.column_stack(line.get_data()), expected, atol=1e-6)
    _assert_fitted_critical_curve(axes[1], 0.14)
    xmin, xmax, ymin, ymax = result.source_bbox
    np.testing.assert_allclose(axes[3].get_xlim(), [xmin, xmax])
    np.testing.assert_allclose(axes[3].get_ylim(), [ymin, ymax])
    output = capsys.readouterr().out
    assert f"Position likelihood penalty: {diagnostics.position.penalty:.4e}" in output
    assert f"{diagnostics.position.max_separation:.4e} arcsec" in output
