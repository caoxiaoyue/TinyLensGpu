"""Basic operator pipelines retain sampling, cache and source-transfer contracts."""

import importlib.util
import pickle
from pathlib import Path

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


@pytest.mark.parametrize("requested", [None, "cubic_hermite"], ids=["legacy", "hermite"])
def test_adaptive_pipeline_resumes_its_saved_source_model(
    load_case, requested, tmp_path, monkeypatch,
):
    import jax.numpy as jnp
    from TinyLensGpu.example_support import export_source_package

    case = load_case("pipe/no_lens_light/model_adpt_reg.py")
    for name, value in dict(
        OUT_DIR=tmp_path, Sampler=_ControlledSampler, NSRC=3, NSUB=1,
        NSUB_PIX=1, N_GAUSSIANS_SRC=2, DPIX=0.08,
    ).items():
        monkeypatch.setattr(case, name, value)
    y, x = np.indices((8, 8))
    image = 0.04 + 0.06 * np.exp(-((x - 5)**2 + (y - 3)**2) / 3)
    noise, psf = np.full_like(image, 0.1), np.array([[1.0]])
    mask = np.zeros_like(image, dtype=bool)
    monkeypatch.setattr(case, "load_lens_data", lambda **kw: (image, noise, psf, None))
    monkeypatch.setattr(case, "run_stage_b", lambda *a, **kw: mask)
    monkeypatch.setattr(case, "_position_likelihood_from_stage_a", lambda *a: None)
    monkeypatch.setattr(case, "plot_model_results", lambda *a, **kw: None)
    reconstructions = {}
    plot_evidence = {}

    def reconstruct(tag, likelihood, medians, names, path):
        result = likelihood.evaluate(jnp.asarray([medians[name] for name in names]))
        assert result.solver_info.converged
        reconstructions[tag] = (likelihood.source_model.interpolation, result)
        plot_evidence[tag] = likelihood.evidence_logdet, likelihood.evidence_block_size
        Path(path).touch()

    monkeypatch.setattr(case, "_plot_pix_stage", reconstruct)
    case.main(source_interpolation=requested)
    fresh = reconstructions.copy()
    expected_mode = requested or "bilinear"
    assert set(fresh) == {"stage-M0", "stage-M1", "stage-M2"}
    assert all(mode == expected_mode for mode, _ in fresh.values())
    for tag in ("a", "m0", "m1", "m2"):
        path = tmp_path / f"stage_{tag}.pkl"
        payload = pickle.loads(path.read_bytes())
        assert "stage_schema" not in payload
        records = [payload["extra"]]
        if tag.startswith("m"):
            assert payload["extra"]["objective"]["evidence_logdet"] == "overlap"
        if tag == "m0":
            s0 = payload["extra"]["s0"]
            records.append(s0)
            assert s0["adapt_image"].shape == image.shape
            np.testing.assert_allclose(s0["source_bbox"], fresh["stage-M2"][1].source_bbox)
        for record in records:
            assert record["source_interpolation"] == expected_mode

    checkpoint_bytes = {path: path.read_bytes() for path in tmp_path.glob("stage_*.pkl")}
    for path in tmp_path.glob("stage_m*_model.png"):
        path.unlink()
    reconstructions.clear()
    case.main(skip_done=True)
    assert reconstructions.keys() == fresh.keys()
    for tag, (mode, result) in reconstructions.items():
        assert mode == expected_mode
        np.testing.assert_allclose(
            result.model_image, fresh[tag][1].model_image, rtol=2e-5, atol=2e-6,
        )
    assert all(path.read_bytes() == before for path, before in checkpoint_bytes.items())

    conflicting = "bilinear" if requested else "cubic_hermite"
    with pytest.raises(ValueError, match="Start a fresh run"):
        case.main(skip_done=True, source_interpolation=conflicting)
    assert all(path.read_bytes() == before for path, before in checkpoint_bytes.items())

    if requested is None:
        m0_path = tmp_path / "stage_m0.pkl"
        archived = pickle.loads(m0_path.read_bytes())
        extra = archived["extra"]
        medians_a = case._load_stage("a")["extra"]["medians"]
        historical = case.build_stage_m0_likelihood(
            image, noise, psf, mask, medians_a, None,
            evidence_logdet="block", evidence_block_size=4,
        )
        log_lambda = extra["lambda_best"]
        producer = export_source_package(
            historical, {"log_lambda_reg": log_lambda}, ["log_lambda_reg"],
            n=3, record_adapt_image=True,
        )
        producer.update(log_lambda_best=log_lambda, lambda_best=np.exp(log_lambda))
        proof = producer["reference_provenance"].copy()
        proof.pop("adapt_image")
        extra["s0"] = {key: value for key, value in producer.items()
                       if key not in {"adapt_image", "reference_provenance"}}
        extra["source_reconstruction_provenance"] = proof
        extra.pop("objective")
        m0_path.write_bytes(pickle.dumps(archived))
        for tag in ("m1", "m2"):
            path = tmp_path / f"stage_{tag}.pkl"
            old_stage = pickle.loads(path.read_bytes())
            old_stage["extra"].pop("objective")
            path.write_bytes(pickle.dumps(old_stage))

        plot = tmp_path / "stage_m0_model.png"
        plot.unlink()
        before = len(_ControlledSampler.instances)
        case.main(skip_done=True)
        assert len(_ControlledSampler.instances) == before + 2
        assert plot.exists() and plot_evidence["stage-M0"] == ("block", 4)
        replayed = case._load_stage("m0")["extra"]
        np.testing.assert_allclose(replayed["s0"]["adapt_image"], producer["adapt_image"], atol=1e-7)
        assert "objective" not in replayed
        assert all("objective" in case._load_stage(tag)["extra"] for tag in ("m1", "m2"))

        extra.pop("source_reconstruction_provenance")
        m0_path.write_bytes(pickle.dumps(archived))
        case.main(skip_done=True)
        assert case._load_stage("m0")["extra"]["objective"]["evidence_logdet"] == "overlap"


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
    import caskade as ck
    import jax.numpy as jnp
    from matplotlib.axes import Axes
    from test_pixelized_support import _small_source_export_model

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
        "source_interpolation",
        "source_boundary",
        "adapt_image",
        "reference_provenance",
    }
    package.update(lambda_best=1.0, log_lambda_best=0.0)
    kwargs = (
        {"expected_source_bbox": package["source_bbox"]}
        if "predefine" in relative
        else {}
    )
    assert case._validate_source_package(package, **kwargs) is package
    assert "scale_map" not in package
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
    image, noise = np.asarray(model.image_data), np.asarray(model.noise_map)
    psf = np.asarray(model.sim_obj.psf_kernel)
    mask = ~np.asarray(model.unmask)
    with ck.ActiveContext(model):
        model.fill_params(jnp.asarray([medians[name] for name in names]))
        reference, _, _ = model.reconstruct_component_images(
            jnp.asarray(package["source_pixels"], dtype=model.image_data.dtype), None,
        )
    np.testing.assert_allclose(package["adapt_image"], reference, atol=1e-7)
    builder = (
        case.build_stage_m1_likelihood
        if "adpt_reg" in relative
        else case.build_stage_m2_likelihood
    )
    m2 = builder(image, noise, psf, mask, m1, None, package)
    tampered = {**package, "adapt_image": package["adapt_image"] + 1.0}
    with pytest.raises(ValueError, match="adapt_image differs"):
        builder(image, noise, psf, mask, m1, None, tampered)
    m2_names = [p.name for p in m2.get_dynamic_params()]
    m2_medians = {p.name: float(np.asarray(p.value)) for p in m2.get_dynamic_params()}
    result, diagnostics = m2.evaluate_with_diagnostics(
        jnp.asarray([m2_medians[n] for n in m2_names]), include_effective_dof=True
    )
    assert diagnostics.valid, diagnostics.invalid_reason
    np.testing.assert_allclose(result.source_bbox, package["source_bbox"], atol=1e-7)
    expected_scale = diagnostics.regularization_scale
    assert diagnostics.evidence_logdet == "overlap"
    assert np.ptp(expected_scale) > 0.0
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


@pytest.mark.parametrize("interpolation", [None, "cubic_hermite"])
def test_simple_fit_launch_and_legacy_artifacts(
    examples_dir, tmp_path, monkeypatch, interpolation,
):
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
    if interpolation is None:
        monkeypatch.delenv("TINYLENS_SOURCE_INTERPOLATION", raising=False)
    else:
        monkeypatch.setenv("TINYLENS_SOURCE_INTERPOLATION", interpolation)
    monkeypatch.chdir(tmp_path)
    namespace = runpy.run_path(str(script), run_name="__main__")
    assert isinstance(namespace["stage"], StagePosterior)
    sampler = _ControlledSampler.instances[-1]
    assert sampler.kwargs == {"n_live": 200, "vectorized": False}
    assert sampler.run_kwargs == {"n_eff": 800}
    with gzip.open(tmp_path / "output/fit_results.pkl.gz", "rb") as stream:
        payload = pickle.load(stream)
    assert set(payload) == {
        "samples", "weights", "log_z", "param_names",
        "source_interpolation", "source_boundary",
    }
    assert payload["source_interpolation"] == (interpolation or "bilinear")
    assert namespace["prob_model"].source_model.interpolation == payload["source_interpolation"]
    assert payload["source_boundary"] == (
        "C1_cardinal_bicubic_fixed_zero_exterior_nodes"
        if interpolation else "hard_zero_outside_bbox"
    )
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


@pytest.mark.parametrize(
    "relative",
    [
        "pipe/no_lens_light/model.py",
        "pipe/galan24_test/model.py",
        "pipe/galan24_test/model_predefine_src_box.py",
    ],
)
def test_m1_reference_pipelines_reuse_matching_pixelized_stages(
    relative, load_case, tmp_path, monkeypatch,
):
    case = load_case(relative)
    for name, value in dict(
        OUT_DIR=tmp_path, Sampler=_ControlledSampler, NSRC=3, NSUB_PIX=1, DPIX=0.08,
    ).items():
        monkeypatch.setattr(case, name, value)
    y, x = np.indices((8, 8))
    image = 0.04 + 0.08 * np.exp(-((x - 5)**2 + (y - 3)**2) / 3)
    noise = np.full_like(image, 0.1)
    psf = np.array([[0., 0.1, 0.], [0.2, 0.6, 0.05], [0., 0.05, 0.]])
    mask = x < 1
    monkeypatch.setattr(case, "load_lens_data", lambda **kw: (image, noise, psf, None))
    monkeypatch.setattr(case, "run_stage_b", lambda *a, **kw: mask)
    monkeypatch.setattr(case, "_position_likelihood_from_stage_a", lambda *a: None)
    monkeypatch.setattr(case, "_plot_pix_stage", lambda *a: Path(a[-1]).touch())

    names = ["theta_E", "e1_mass", "e2_mass", "center_x_mass", "center_y_mass", "gamma1", "gamma2"]
    values = np.array([0.17, 0.1, 0.03, 0.0, 0.0, 0.02, -0.01])
    stage_a = StagePosterior(np.tile(values, (3, 1)), np.ones(3), names)
    extra = {"medians": stage_a.medians()}
    if "predefine" in relative:
        factor = case.SOURCE_BBOX_RADIUS_FACTOR
        extra["source_bbox_info"] = dict(
            center_x=0.0, center_y=0.0, effective_radius=0.4 / factor, axis_ratio=1.0,
            major_axis_effective_radius=0.4 / factor, radius_factor=factor,
            source_bbox=(-0.4, 0.4, -0.4, 0.4),
        )
    case._dump_stage(
        "a", stage_a.samples, stage_a.weights, names, -12.0,
        extra=extra, stage=stage_a, source_interpolation="bilinear",
    )
    before = len(_ControlledSampler.instances)
    case.main(skip_done=True)
    assert len(_ControlledSampler.instances) == before + 3
    m1 = case._load_stage("m1")["extra"]
    m2 = case._load_stage("m2")["extra"]["objective"]
    m3 = case._load_stage("m3")["extra"]["objective"]
    assert m1["s1"]["adapt_image"].shape == image.shape
    assert m2["reference_id"] == m3["reference_id"]
    assert m2["grid"] == m3["grid"]
    if "predefine" in relative:
        np.testing.assert_allclose(m2["grid"]["source_bbox"], (-0.4, 0.4, -0.4, 0.4))
    artifacts = {path: path.read_bytes() for path in tmp_path.glob("stage_m*.pkl")}
    case.main(skip_done=True)
    assert len(_ControlledSampler.instances) == before + 3
    assert all(path.read_bytes() == data for path, data in artifacts.items())
