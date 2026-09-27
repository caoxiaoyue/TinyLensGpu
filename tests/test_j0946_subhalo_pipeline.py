import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import ndtr, ndtri

from TinyLensGpu.Inference import StagePosterior
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.Inference.build_prior import PriorSpec, make_prior_transformation

# TinyLensGpu selects the JAX backend before caskade is imported.
import caskade as ck


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "pix_src_demo_operator"
    / "pipe"
    / "subhalo_0946"
    / "model.py"
)


def _load_pipeline_module():
    spec = importlib.util.spec_from_file_location("subhalo_0946_model", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def comparison(monkeypatch):
    pipeline = _load_pipeline_module()
    monkeypatch.setattr(pipeline, "NSRC", 8)
    monkeypatch.setattr(pipeline, "NSUB_PIX", 1)
    monkeypatch.setattr(pipeline, "DPIX", 0.4)
    centers = {
        "theta_E": 1.44,
        "gamma": 2.0,
        "e1_mass": 0.05,
        "e2_mass": -0.03,
        "center_x_mass": 0.0,
        "center_y_mass": 0.0,
        "gamma1": 0.01,
        "gamma2": -0.02,
    }
    offsets = np.linspace(-1.0, 1.0, 21)[:, None]
    weights = np.ones(21)
    stage_m3 = StagePosterior(
        np.array(list(centers.values())) + offsets * 0.01,
        weights,
        param_names=list(centers),
    )
    stage_m2 = StagePosterior(
        np.array([0.0, 1.0]) + offsets * np.array([0.1, 0.05]),
        weights,
        param_names=["log_lambda_reg", "adaptive_reg_rho"],
    )
    axis = np.linspace(-2.0, 2.0, 8)
    x, y = np.meshgrid(axis, axis)
    source = np.exp(-(x**2 + y**2)) + 0.05
    package = dict(
        source_pixels=source.ravel(),
        source_bbox=(-2.0, 2.0, -2.0, 2.0),
        source_x_axis=axis,
        source_y_axis=axis,
        n=8,
        lambda_best=1.0,
        log_lambda_best=0.0,
    )
    image = np.exp(-((x - 0.2)**2 + (y + 0.1)**2))
    noise = np.full_like(image, 0.2)
    psf = np.zeros((3, 3))
    psf[1, 1] = 1.0
    mask = np.zeros(image.shape, dtype=bool)
    mask[0, 0] = True
    positions = dict(
        positions=[(-1.2, 0.4), (0.6, 1.1)],
        threshold_arcsec=0.1,
        min_log_like=-20.0,
    )

    def build(with_subhalo=False):
        return pipeline.build_comparison_likelihood(
            image, noise, psf, mask, stage_m2, stage_m3, positions, package,
            with_subhalo=with_subhalo,
        )

    return pipeline, stage_m2, stage_m3, image.shape, build, (
        image, noise, psf, mask, positions, package,
    )


def _prior_samples(model, quantiles=(0.1, 0.5, 0.9)):
    transform, specs = make_prior_transformation(model)
    unit = np.broadcast_to(np.asarray(quantiles)[:, None], (len(quantiles), len(specs)))
    return [spec.name for spec in specs], np.asarray(transform(unit))


@pytest.mark.unit
def test_runtime_position_mode_changes_built_position_distribution(comparison, monkeypatch):
    pipeline, _, _, shape, build, _ = comparison
    quantiles = np.array([0.1, 0.5, 0.9])
    for mode in ("gaussian_local", "full_image", "model_square"):
        monkeypatch.setattr(pipeline, "SUBHALO_POSITION_MODE", mode)
        names, samples = _prior_samples(build(with_subhalo=True), quantiles)
        actual = samples[:, [names.index("center_x_subhalo"), names.index("center_y_subhalo")]]
        if mode == "gaussian_local":
            trunc = pipeline.SUBHALO_POSITION_TRUNCATE_SIGMA
            z = ndtri(ndtr(-trunc) + quantiles * (ndtr(trunc) - ndtr(-trunc)))
            expected = np.array(pipeline.SUBHALO_POSITION_CENTER) + z[:, None] * pipeline.SUBHALO_POSITION_SIGMA
        else:
            half_widths = (
                np.array(shape[::-1]) * pipeline.DPIX / 2.0
                if mode == "full_image" else np.full(2, pipeline.MASK_RADIUS)
            )
            expected = (2.0 * quantiles[:, None] - 1.0) * half_widths
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.unit
def test_shared_prior_transforms_follow_live_lambda_width(comparison, monkeypatch):
    pipeline, stage_m2, stage_m3, _, build, _ = comparison
    for width in (0.25, 0.8):
        monkeypatch.setattr(pipeline, "LAMBDA_PRIOR_HALF_WIDTH_DEX", width)
        null_names, null_samples = _prior_samples(build())
        sub_names, sub_samples = _prior_samples(build(with_subhalo=True))
        for name in (*stage_m3.param_names, "log_lambda_reg"):
            np.testing.assert_allclose(
                null_samples[:, null_names.index(name)],
                sub_samples[:, sub_names.index(name)],
                rtol=0.0, atol=0.0,
            )
        expected = stage_m2.median("log_lambda_reg") + np.array([-0.8, 0.0, 0.8]) * width * np.log(10.0)
        np.testing.assert_allclose(
            null_samples[:, null_names.index("log_lambda_reg")], expected,
            rtol=2e-6, atol=2e-6,
        )


@pytest.mark.unit
def test_zero_strength_subhalo_matches_null_and_position_switch_changes_likelihood(comparison, monkeypatch):
    pipeline, _, _, _, build, _ = comparison
    results = {}
    for enabled in (False, True):
        monkeypatch.setattr(pipeline, "COMPARISON_USE_POSITION_LIKELIHOOD", enabled)
        for with_subhalo in (False, True):
            model = build(with_subhalo=with_subhalo)
            names, samples = _prior_samples(model, (0.5,))
            theta = samples[0].copy()
            if with_subhalo:
                theta[names.index("b_subhalo")] = 0.0
            value = make_likelihood(model)(theta)
            assert np.isfinite(value) and value > -1.0e9
            results[enabled, with_subhalo] = value
            if enabled and not with_subhalo:
                with ck.ActiveContext(model):
                    model.fill_params(jnp.asarray(theta))
                    expected_penalty = float(model._position_likelihood_penalty_jax())
        np.testing.assert_allclose(results[enabled, False], results[enabled, True], rtol=0.0, atol=2e-4)
    assert expected_penalty < -1.0
    for with_subhalo in (False, True):
        np.testing.assert_allclose(
            results[True, with_subhalo] - results[False, with_subhalo],
            expected_penalty, rtol=0.0, atol=2e-4,
        )


@pytest.mark.unit
def test_tidal_radius_updates_numerically_with_live_host_radius(comparison):
    _, _, _, _, build, _ = comparison
    model = build(with_subhalo=True)
    names, samples = _prior_samples(model, (0.5,))
    theta = samples[0].copy()
    theta[names.index("b_subhalo")] = 0.04
    for host_radius in (1.44, 1.69):
        theta[names.index("theta_E")] = host_radius
        with ck.ActiveContext(model):
            model.fill_params(jnp.asarray(theta))
            np.testing.assert_allclose(
                model.phys_model.lens_mass[-1].rt.value,
                np.sqrt(0.04 * host_radius), rtol=2e-6,
            )


@pytest.mark.unit
@pytest.mark.parametrize("name,value", [
    ("SUBHALO_POSITION_MODE", "full_image"),
    ("SUBHALO_POSITION_CENTER", (-0.5, 0.8)),
    ("SUBHALO_POSITION_SIGMA", 0.3),
    ("SUBHALO_POSITION_TRUNCATE_SIGMA", 2.0),
    ("SUBHALO_B_RANGE", (0.0, 0.2)),
    ("COMPARISON_USE_POSITION_LIKELIHOOD", False),
    ("LAMBDA_PRIOR_HALF_WIDTH_DEX", 0.8),
    ("SMOOTH_PRIOR_MIN_HALF_WIDTH_SIGMA", 8.0),
])
def test_cache_invalidates_changed_live_prior_choices(comparison, monkeypatch, name, value):
    pipeline, stage_m2, stage_m3, _, _, cache_inputs = comparison
    schema = pipeline._comparison_cache_schema(stage_m2, stage_m3, *cache_inputs)
    with monkeypatch.context() as changed:
        changed.setattr(pipeline, name, value)
        assert pipeline._comparison_cache_schema(stage_m2, stage_m3, *cache_inputs) != schema
    assert pipeline._comparison_cache_schema(stage_m2, stage_m3, *cache_inputs) == schema


@pytest.mark.unit
def test_changed_subpixel_sampling_rejects_saved_comparison_evidence(comparison, monkeypatch):
    pipeline, stage_m2, stage_m3, _, _, cache_inputs = comparison
    saved_schema = pipeline._comparison_cache_schema(stage_m2, stage_m3, *cache_inputs)
    saved = {
        role: {"extra": {"pipeline_schema": saved_schema, "comparison_role": role}}
        for role in ("null", "subhalo")
    }

    monkeypatch.setattr(pipeline, "NSUB_PIX", 2)
    current_schema = pipeline._comparison_cache_schema(stage_m2, stage_m3, *cache_inputs)
    for role, payload in saved.items():
        assert not pipeline._comparison_cache_matches(payload, current_schema, role)


@pytest.mark.unit
@pytest.mark.parametrize("setting,value", [
    ("DPIX", 0.5),
    ("NSRC", 12),
    ("PIXEL_REGULARIZATION_TYPE", "zero-order"),
    ("SOLVER_TYPE", "fista"),
    ("M5_N_LIVE", 1200),
])
def test_changed_comparison_settings_reject_saved_evidence(comparison, monkeypatch, setting, value):
    pipeline, stage_m2, stage_m3, _, _, cache_inputs = comparison
    saved_schema = pipeline._comparison_cache_schema(stage_m2, stage_m3, *cache_inputs)
    saved = {"extra": {"pipeline_schema": saved_schema, "comparison_role": "subhalo"}}

    monkeypatch.setattr(pipeline, setting, value)
    current_schema = pipeline._comparison_cache_schema(stage_m2, stage_m3, *cache_inputs)
    assert not pipeline._comparison_cache_matches(saved, current_schema, "subhalo")


@pytest.mark.unit
@pytest.mark.parametrize("changed_input", [
    "image", "noise", "psf", "mask", "positions", "source_pixels", "source_bbox",
])
def test_changed_comparison_data_rejects_saved_evidence(comparison, changed_input):
    pipeline, stage_m2, stage_m3, _, _, cache_inputs = comparison
    saved_schema = pipeline._comparison_cache_schema(stage_m2, stage_m3, *cache_inputs)
    saved = {"extra": {"pipeline_schema": saved_schema, "comparison_role": "null"}}
    image, noise, psf, mask, positions, package = cache_inputs
    if changed_input == "positions":
        positions = {**positions, "threshold_arcsec": 0.15}
    elif changed_input == "source_pixels":
        pixels = package["source_pixels"].copy()
        pixels[0] += 0.1
        package = {**package, "source_pixels": pixels}
    elif changed_input == "source_bbox":
        package = {**package, "source_bbox": (-1.9, 2.1, -2.0, 2.0)}
    else:
        arrays = dict(image=image, noise=noise, psf=psf, mask=mask)
        changed = arrays[changed_input].copy()
        changed.flat[0] = not changed.flat[0] if changed_input == "mask" else changed.flat[0] + 0.1
        arrays[changed_input] = changed
        image, noise, psf, mask = (arrays[name] for name in ("image", "noise", "psf", "mask"))

    current_schema = pipeline._comparison_cache_schema(
        stage_m2, stage_m3, image, noise, psf, mask, positions, package,
    )
    assert not pipeline._comparison_cache_matches(saved, current_schema, "null")


@pytest.mark.unit
def test_cache_tracks_posterior_bounds_regularization_and_arm(comparison, monkeypatch):
    pipeline, stage_m2, stage_m3, _, _, cache_inputs = comparison
    schema = pipeline._comparison_cache_schema(stage_m2, stage_m3, *cache_inputs)
    for stage, name in ((stage_m3, "theta_E"), (stage_m2, "log_lambda_reg"), (stage_m2, "adaptive_reg_rho")):
        samples = stage.samples.copy()
        samples[:, stage.param_names.index(name)] += 0.01
        changed = StagePosterior(samples, stage.weights, param_names=stage.param_names)
        args = (stage_m2, changed) if stage is stage_m3 else (changed, stage_m3)
        assert pipeline._comparison_cache_schema(*args, *cache_inputs) != schema
    monkeypatch.setattr(pipeline, "SUBHALO_POSITION_MODE", "full_image")
    full_image_schema = pipeline._comparison_cache_schema(stage_m2, stage_m3, *cache_inputs)
    image, noise, psf, mask, positions, package = cache_inputs
    expanded_inputs = (
        np.pad(image, 1), np.pad(noise, 1), psf, np.pad(mask, 1), positions, package,
    )
    assert pipeline._comparison_cache_schema(stage_m2, stage_m3, *expanded_inputs) != full_image_schema
    monkeypatch.setattr(pipeline, "DPIX", 0.5)
    changed_schema = pipeline._comparison_cache_schema(stage_m2, stage_m3, *cache_inputs)
    assert changed_schema != full_image_schema
    payload = {"extra": {"pipeline_schema": schema, "comparison_role": "null"}}
    assert pipeline._comparison_cache_matches(payload, schema, expected_role="null")
    assert not pipeline._comparison_cache_matches(payload, schema, expected_role="subhalo")
    assert not pipeline._comparison_cache_matches(payload, changed_schema, expected_role="null")


@pytest.mark.unit
def test_zero_subhalo_strength_is_reported_separately_from_smooth_prior_edges(
    comparison, capsys, monkeypatch, tmp_path,
):
    pipeline = comparison[0]
    stage = StagePosterior(
        samples=np.array([
            [1.4, 0.0000],
            [1.5, 0.0005],
            [1.6, 0.0010],
        ]),
        weights=np.ones(3),
        prior_specs=[
            PriorSpec("theta_E", "uniform", (1.0, 2.0)),
            PriorSpec("b_subhalo", "uniform", (0.0, 0.1)),
        ],
    )

    edge_report = pipeline._prior_edge_report("stage-M5", stage)
    strength_report = pipeline._subhalo_strength_report("stage-M5", stage)
    output = capsys.readouterr().out

    assert set(edge_report) == {"theta_E"}
    assert not edge_report["theta_E"]["flagged"]
    assert strength_report["near_zero_weight"] == pytest.approx(1.0)
    assert "b_subhalo=0 means no subhalo" in output
    assert "PRIOR EDGE" not in output

    monkeypatch.setattr(pipeline, "OUT_DIR", tmp_path)
    cached = {"extra": {"prior_edge_report": {"b_subhalo": {"flagged": True}}}}
    pipeline._refresh_cached_boundary_reports("m5", cached, stage)
    refreshed = pipeline._load_stage("m5")
    assert set(refreshed["extra"]["prior_edge_report"]) == {"theta_E"}
    assert refreshed["extra"]["subhalo_strength_report"]["near_zero_weight"] == pytest.approx(1.0)
    np.testing.assert_array_equal(refreshed["samples"], stage.samples)
