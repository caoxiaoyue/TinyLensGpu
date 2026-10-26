"""Basic operator pipelines retain sampling, cache and source-transfer contracts."""

import importlib.util

import numpy as np
import pytest

import TinyLensGpu  # Initialize the documented JAX backend.
from TinyLensGpu.Inference import StagePosterior
from test_pixelized_support import _ControlledSampler, _small_likelihood


@pytest.fixture
def load_case(monkeypatch, examples_dir):
    import jax

    precision = jax.config.jax_default_matmul_precision

    def load(relative):
        path = examples_dir / "pix_src_demo_operator" / relative
        spec = importlib.util.spec_from_file_location("basic_" + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        with monkeypatch.context() as context:
            context.chdir(path.parent)
            spec.loader.exec_module(module)
        return module

    yield load
    jax.config.update("jax_default_matmul_precision", precision)


@pytest.mark.parametrize(
    "relative,schema",
    [
        ("pipe/no_lens_light/model.py", True),
        ("pipe/no_lens_light/model_adpt_reg.py", False),
        ("pipe/galan24_test/model.py", True),
        ("pipe/galan24_test/model_predefine_src_box.py", True),
    ],
)
def test_basic_sampling_and_cache_contract(
    relative, schema, load_case, tmp_path, monkeypatch
):
    case = load_case(relative)
    monkeypatch.setattr(case, "OUT_DIR", tmp_path)
    monkeypatch.setattr(case, "Sampler", _ControlledSampler)
    stage = case._run_sampler(_small_likelihood(), 9, 17, "small", vectorized=True)
    assert isinstance(stage, StagePosterior)
    assert _ControlledSampler.instances[-1].kwargs["n_batch"] == 100
    assert _ControlledSampler.instances[-1].run_kwargs["n_eff"] == 17
    assert list(stage.param_names) == ["theta_E"]
    extra = {
        "source_pixels": np.arange(9, dtype=np.float32),
        "medians": stage.medians(),
    }
    kwargs = {"stage": stage} if schema else {}
    case._dump_stage(
        "a",
        stage.samples,
        stage.weights,
        stage.param_names,
        stage.log_z,
        extra=extra,
        **kwargs
    )
    payload = case._load_stage("a")
    assert set(payload) == {"samples", "weights", "param_names", "log_z", "extra"} | (
        {"stage_schema"} if schema else set()
    )
    assert payload["extra"]["source_pixels"].dtype == np.float32
    from TinyLensGpu.example_support import stage_from_payload

    restored = stage_from_payload(payload)
    np.testing.assert_array_equal(restored.samples, stage.samples)
    assert list(restored.param_names) == ["theta_E"]


@pytest.mark.parametrize(
    "relative",
    [
        "pipe/no_lens_light/model.py",
        "pipe/no_lens_light/model_adpt_reg.py",
        "pipe/galan24_test/model.py",
        "pipe/galan24_test/model_predefine_src_box.py",
    ],
)
def test_source_export_to_adaptive_stage_and_plot(
    relative, load_case, tmp_path, monkeypatch
):
    import jax.numpy as jnp
    from matplotlib.axes import Axes
    from test_pixelized_support import _small_source_export_model
    from TinyLensGpu.utils.inversion.regularization import source_template_scale_map

    case = load_case(relative)
    monkeypatch.setattr(case, "NSRC", 3)
    monkeypatch.setattr(case, "NSUB_PIX", 1)
    monkeypatch.setattr(case, "DPIX", 0.08)
    model = _small_source_export_model()
    names = [p.name for p in model.get_dynamic_params()]
    medians = {p.name: float(np.asarray(p.value)) for p in model.get_dynamic_params()}
    package = case._solve_pixel_source_for_package(model, medians, names)
    assert set(package) == {
        "source_pixels",
        "source_image",
        "source_bbox",
        "source_x_axis",
        "source_y_axis",
        "n",
    }
    package.update(lambda_best=1.0, log_lambda_best=0.0)
    kwargs = (
        {"expected_source_bbox": package["source_bbox"]}
        if "predefine" in relative
        else {}
    )
    assert case._validate_source_package(package, **kwargs) is package
    assert package["scale_map"].dtype == np.float32
    if kwargs:
        with pytest.raises(ValueError, match="fixed stage-A"):
            case._validate_source_package(
                package, expected_source_bbox=(-3.0, 3.0, -3.0, 3.0)
            )
    m1 = dict(
        theta_E=0.17,
        gamma=2.1,
        e1_mass=0.1,
        e2_mass=0.03,
        center_x_mass=0.0,
        center_y_mass=0.0,
        gamma1=0.02,
        gamma2=-0.01,
        log_lambda_reg=0.0,
    )
    image = np.full((10, 10), 0.05)
    noise = np.full_like(image, 0.1)
    psf = np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    mask = np.zeros_like(image, dtype=bool)
    builder = (
        case.build_stage_m1_likelihood
        if "adpt_reg" in relative
        else case.build_stage_m2_likelihood
    )
    m2 = builder(image, noise, psf, mask, m1, None, package)
    m2_names = [p.name for p in m2.get_dynamic_params()]
    m2_medians = {p.name: float(np.asarray(p.value)) for p in m2.get_dynamic_params()}
    result, diagnostics = m2.evaluate_with_diagnostics(
        jnp.asarray([m2_medians[n] for n in m2_names]), include_effective_dof=True
    )
    assert diagnostics.valid, diagnostics.invalid_reason
    np.testing.assert_allclose(result.source_bbox, package["source_bbox"], atol=1e-7)
    expected_scale = source_template_scale_map(
        package["source_pixels"], 3, rho=diagnostics.adaptive_reg_rho
    ).reshape(3, 3)
    np.testing.assert_allclose(
        diagnostics.regularization_scale, expected_scale, rtol=2e-5
    )
    rendered = []
    original_imshow = Axes.imshow

    def capture(ax, data, *args, **kwargs):
        rendered.append(np.asarray(data).copy())
        return original_imshow(ax, data, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", capture)
    case._plot_pix_stage("m2", m2, m2_medians, m2_names, tmp_path / "m2.png")
    assert (tmp_path / "m2.png").stat().st_size > 0
    assert len(rendered) == 5
    np.testing.assert_allclose(rendered[1], result.model_image, rtol=2e-5)
    np.testing.assert_allclose(
        rendered[3], result.source_pixels.reshape(3, 3), rtol=2e-5
    )
    np.testing.assert_allclose(rendered[4], expected_scale, rtol=2e-5)


def test_adaptive_stage_a_writes_legacy_payload(load_case, tmp_path, monkeypatch):
    case = load_case("pipe/no_lens_light/model_adpt_reg.py")
    monkeypatch.setattr(case, "OUT_DIR", tmp_path)
    monkeypatch.setattr(case, "N_GAUSSIANS_SRC", 2)
    monkeypatch.setattr(case, "NSUB", 1)
    monkeypatch.setattr(case, "Sampler", _ControlledSampler)
    psf = np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    samples, weights, names, medians = case.run_stage_a(
        np.full((8, 8), 0.05), np.full((8, 8), 0.1), psf
    )
    saved = case._load_stage("a")
    assert set(saved) == {"samples", "weights", "param_names", "log_z", "extra"}
    assert saved["param_names"] == names
    np.testing.assert_array_equal(saved["samples"], samples)
    np.testing.assert_array_equal(saved["weights"], weights)
    assert saved["extra"]["medians"] == medians
    assert _ControlledSampler.instances[-1].kwargs["n_live"] == 300
    assert _ControlledSampler.instances[-1].kwargs["n_batch"] == 100
    assert _ControlledSampler.instances[-1].run_kwargs["n_eff"] == 2000


def test_simple_fit_launch_and_legacy_artifacts(examples_dir, tmp_path, monkeypatch):
    import gzip
    import pickle
    import runpy
    import nautilus
    import matplotlib.pyplot as plt
    from astropy.io import fits

    source = examples_dir / "pix_src_demo_operator/simple/pix_src/fit_lens_src.py"
    script = tmp_path / source.name
    script.write_text(source.read_text())
    data = tmp_path / "data"
    data.mkdir()
    for name, value in [
        ("image", np.full((8, 8), 0.05)),
        ("noise", np.full((8, 8), 0.1)),
        ("mask", np.zeros((8, 8), dtype=np.uint8)),
        ("psf", np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])),
    ]:
        fits.writeto(data / (name + ".fits"), value)
    monkeypatch.setattr(nautilus, "Sampler", _ControlledSampler)
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.chdir(tmp_path)
    namespace = runpy.run_path(str(script), run_name="__main__")
    assert isinstance(namespace["stage"], StagePosterior)
    sampler = _ControlledSampler.instances[-1]
    assert sampler.kwargs == {"n_live": 200, "vectorized": False}
    assert sampler.run_kwargs == {"n_eff": 800}
    with gzip.open(tmp_path / "output/fit_results.pkl.gz", "rb") as stream:
        payload = pickle.load(stream)
    assert set(payload) == {"samples", "weights", "log_z", "param_names"}
    assert len(payload["param_names"]) == 6
    assert payload["samples"].dtype == np.float64
    assert (tmp_path / "output/fit_map_visualization.png").stat().st_size > 0
    assert (tmp_path / "output/fit_samples.csv").stat().st_size > 0
    assert (
        (tmp_path / "output/fit_summary.csv")
        .read_text()
        .startswith("parameter,median,lower,upper\n")
    )
    plt.close("all")


@pytest.mark.parametrize(
    "relative,cmap",
    [
        ("pipe/no_lens_light/model.py", "viridis"),
        ("pipe/no_lens_light/model_adpt_reg.py", "viridis"),
        ("pipe/galan24_test/model.py", "magma"),
        ("pipe/galan24_test/model_predefine_src_box.py", "magma"),
    ],
)
def test_mask_stage_writes_fits_and_retains_render_variant(
    relative, cmap, load_case, tmp_path, monkeypatch
):
    from astropy.io import fits
    from matplotlib.axes import Axes

    case = load_case(relative)
    monkeypatch.setattr(case, "OUT_DIR", tmp_path)
    monkeypatch.setattr(case, "DATA_DIR", tmp_path)
    image = np.zeros((12, 12))
    image[4:8, 4:8] = 5.0
    rendered = []
    original_imshow = Axes.imshow

    def capture(ax, data, *args, **kwargs):
        rendered.append((np.asarray(data).copy(), kwargs.get("cmap")))
        return original_imshow(ax, data, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", capture)
    mask = case.run_stage_b(image, np.ones_like(image))
    np.testing.assert_array_equal(
        fits.getdata(tmp_path / "feature_mask.fits"), mask.astype(np.uint8)
    )
    np.testing.assert_array_equal(rendered[0][0], image)
    assert rendered[0][1] == cmap
    assert rendered[1][1] == cmap
    assert (tmp_path / "stage_b_mask.png").stat().st_size > 0
