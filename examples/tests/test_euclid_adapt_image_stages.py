"""Small maintained Euclid fits with independently checked stage artifacts."""

import importlib.util
import pickle

import jax.numpy as jnp
import numpy as np
import pytest

from TinyLensGpu.Inference import StagePosterior


class ControlledSampler:
    """Supply reproducible posterior points at the external sampler boundary."""

    log_z = -12.0
    runs = []

    def __init__(self, prior, loglike, *, n_dim, **kwargs):
        self.points = np.tile(np.asarray(prior(jnp.full(n_dim, 0.5))), (3, 1))

    def run(self, **kwargs):
        self.runs.append(self.points.shape[1])

    def posterior(self):
        return self.points, np.log([0.2, 0.3, 0.5]), np.zeros(3)


@pytest.fixture
def euclid(examples_dir, tmp_path, monkeypatch):
    path = examples_dir / "pix_src_demo_operator/pipe/euclid_examples/pipeline.py"
    spec = importlib.util.spec_from_file_location("euclid_adapt_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ControlledSampler.runs.clear()
    for key, value in dict(
        Sampler=ControlledSampler, NSRC=3, NSUB=1, NSUB_PIX=2, DPIX=0.08,
        N_GAUSSIANS_SRC=2, N_GAUSSIANS_LENS=2,
        OUT_DIR=tmp_path, DATA_DIR=tmp_path,
    ).items():
        monkeypatch.setattr(module, key, value)
    monkeypatch.setattr(module, "overlay_critical_and_caustics", lambda **kw: None)
    return module


@pytest.fixture
def small_fit():
    y, x = np.indices((10, 10))
    image = (0.04 + 0.25 * np.exp(-((x - 7)**2 + (y - 4)**2) / 3)).astype(np.float32)
    noise = np.full_like(image, 0.1)
    psf = np.array([[0., 0.04, 0.], [0.18, 0.63, 0.05], [0., 0.1, 0.]], dtype=np.float32)
    mask = x < 2
    names = ["theta_E", "e1_mass", "e2_mass", "center_x_mass", "center_y_mass", "gamma1", "gamma2"]
    center = np.asarray([0.17, 0.1, 0.03, 0.02, -0.01, 0.02, -0.01])
    stage_a = StagePosterior(np.stack([center - 0.001, center, center + 0.001]), np.ones(3), names)
    return image, noise, psf, mask, stage_a


def test_m1_exports_native_psf_arc_and_distinct_source_raster(euclid, small_fit):
    image, noise, psf, mask, stage_a = small_fit
    stage, point, package = euclid.run_stage_m1(
        image, noise, psf, mask, np.zeros_like(image), stage_a, None, 0.0,
    )

    assert package["adapt_image"].shape == image.shape
    assert package["source_image"].shape == (3, 3)
    assert not np.any(package["adapt_image"][mask])
    likelihood = euclid.build_stage_m1_likelihood(image, noise, psf, mask, stage_a, None, 0.0)
    theta = jnp.asarray([point[name] for name in stage.param_names])
    result, diagnostics = likelihood.evaluate_with_diagnostics(theta, include_components=True)
    assert diagnostics.valid
    np.testing.assert_allclose(package["adapt_image"], diagnostics.source_image, atol=1e-6)
    np.testing.assert_allclose(package["source_image"].ravel(), result.source_pixels, atol=1e-6)
    saved = euclid._load_stage("m1")["extra"]["s1"]
    np.testing.assert_array_equal(saved["adapt_image"], package["adapt_image"])


def test_m2_and_m3_record_one_reference_and_calibrated_strengths(euclid, small_fit):
    image, noise, psf, mask, stage_a = small_fit
    data = (image, noise, psf, mask, np.zeros_like(image))
    stage_m1, point, package = euclid.run_stage_m1(*data, stage_a, None, 0.0)
    mass_point = {**stage_a.medians(), **point}
    stage_m2, _, strengths = euclid.run_stage_m2(*data, mass_point, None, package)
    assert set(stage_m2.param_names) == {"log_lambda_reg", "adaptive_reg_rho"}
    stage_m3, _ = euclid.run_stage_m3(*data, stage_m1, stage_a, None, strengths, package)
    assert "gamma" in stage_m3.param_names
    assert not set(stage_m2.param_names).intersection(stage_m3.param_names)

    m1, m2, m3 = (euclid._load_stage(tag)["extra"] for tag in ("m1", "m2", "m3"))
    assert m2["objective"]["reference_id"] == m3["objective"]["reference_id"]
    assert m2["objective"]["grid"] == m3["objective"]["grid"]
    assert tuple(m2["objective"]["grid"]["source_bbox"]) == package["source_bbox"]
    assert m3["objective"]["fixed_strengths"] == strengths
    assert m1["objective"]["evidence_logdet"] == "overlap"
    assert m2["objective"]["evidence_logdet"] == "overlap"
    assert m3["objective"]["sampling"]["vectorized_chunk_size"] == 1


def _seed_upstream(euclid, small_fit, monkeypatch):
    image, noise, psf, mask, stage_a = small_fit
    monkeypatch.setattr(euclid, "load_lens_data", lambda **kw: (image, noise, psf, None))
    monkeypatch.setattr(euclid, "run_stage_b", lambda *a, **kw: mask)
    monkeypatch.setattr(euclid, "_position_likelihood_from_stage_a", lambda point: None)
    for tag in ("a", "l"):
        euclid._dump_stage(
            tag, stage_a.samples, stage_a.weights, stage_a.param_names, -12.0,
            extra=dict(medians=stage_a.medians(), lens_light_model=np.zeros_like(image)),
            stage=stage_a, source_interpolation="bilinear",
        )


def test_matching_resume_reuses_stages_and_changed_score_refits(euclid, small_fit, monkeypatch):
    _seed_upstream(euclid, small_fit, monkeypatch)
    options = dict(skip_done=True, evidence_block_size=2, vectorized_chunk_size=2, seed=73)
    euclid.main(evidence_logdet="overlap", **options)
    assert ControlledSampler.runs == [7, 2, 6]
    m0 = euclid._load_stage("m0")["extra"]["objective"]
    assert m0["evidence_logdet"] == "overlap"
    assert m0["sampling"]["vectorized_chunk_size"] == 2
    euclid.main(evidence_logdet="overlap", **options)
    assert ControlledSampler.runs == [7, 2, 6]

    euclid.main(evidence_logdet="block", **options)
    assert ControlledSampler.runs == [7, 2, 6, 7, 2, 6]
    for tag in ("m0", "m1", "m2", "m3"):
        assert euclid._load_stage(tag)["extra"]["objective"]["evidence_logdet"] == "block"


def _replace_stage_extra(euclid, tag, extra):
    payload = euclid._load_stage(tag)
    stage = euclid._stage_from_payload(payload)
    euclid._dump_stage(
        tag, stage.samples, stage.weights, stage.param_names, stage.log_z,
        stage=stage, extra=extra, source_interpolation="bilinear",
    )


def test_verified_legacy_m1_renders_reference_but_old_adaptive_stages_refit(
    euclid, small_fit, monkeypatch,
):
    _seed_upstream(euclid, small_fit, monkeypatch)
    euclid.main(skip_done=True)
    from TinyLensGpu.example_support import export_source_package, source_reconstruction_provenance

    image, noise, psf, mask, stage_a = small_fit
    payload = euclid._load_stage("m1")
    original = payload["extra"]
    medians, names = original["medians"], payload["param_names"]
    log_lambda_m0 = euclid._load_stage("m0")["extra"]["log_lambda_best"]
    historical = euclid.build_stage_m1_likelihood(
        image, noise, psf, mask, stage_a, None, log_lambda_m0, evidence_logdet="block",
    )
    legacy = export_source_package(historical, medians, names, n=3)
    legacy.update(log_lambda_best=medians["log_lambda_reg"], lambda_best=np.exp(medians["log_lambda_reg"]))
    theta = jnp.asarray([medians[name] for name in names])
    result, diagnostics = historical.evaluate_with_diagnostics(theta, include_components=True)
    assert diagnostics.evidence_valid
    proof = source_reconstruction_provenance(
        historical, legacy, medians, names, source_dtype=result.source_pixels.dtype,
    )
    legacy_m1 = {
        "medians": original["medians"], "s1": legacy,
        "source_reconstruction_provenance": proof,
    }
    _replace_stage_extra(euclid, "m1", legacy_m1)
    for tag in ("m2", "m3"):
        extra = euclid._load_stage(tag)["extra"]
        _replace_stage_extra(euclid, tag, {key: value for key, value in extra.items()
                                         if key != "objective"})

    euclid.main(skip_done=True)
    assert ControlledSampler.runs == [7, 2, 6, 2, 6]
    replayed = euclid._load_stage("m1")["extra"]
    np.testing.assert_allclose(replayed["s1"]["adapt_image"], diagnostics.source_image, atol=1e-6)
    assert replayed["s1"]["reference_provenance"]["producer_evidence_logdet"] == "block"
    assert "objective" not in replayed

    # Missing independent producer provenance must rebuild M1 and its dependents.
    _replace_stage_extra(euclid, "m1", {"medians": original["medians"], "s1": legacy})
    euclid.main(skip_done=True)
    assert ControlledSampler.runs[-3:] == [7, 2, 6]
    assert euclid._load_stage("m1")["extra"]["objective"]["evidence_logdet"] == "overlap"


@pytest.mark.parametrize("damaged_product", ["adaptive", "reference", "producer_metadata"])
def test_resume_rebuilds_incompatible_products_after_m1_mode_selection(
    euclid, small_fit, monkeypatch, damaged_product,
):
    from TinyLensGpu.example_support import source_interpolation_metadata

    _seed_upstream(euclid, small_fit, monkeypatch)
    euclid.main(skip_done=True)
    assert ControlledSampler.runs == [7, 2, 6]

    m1_path = euclid.OUT_DIR / "stage_m1.pkl"
    saved_m1 = m1_path.read_bytes()
    if damaged_product != "adaptive":
        payload = pickle.loads(saved_m1)
        metadata = payload["extra"]
        if damaged_product == "reference":
            metadata = metadata["s1"]
        metadata["source_boundary"] = "malformed"
        m1_path.write_bytes(pickle.dumps(payload))

    for tag in ("m2", "m3"):
        path = euclid.OUT_DIR / f"stage_{tag}.pkl"
        payload = pickle.loads(path.read_bytes())
        payload["extra"].update(source_interpolation_metadata("cubic_hermite"))
        path.write_bytes(pickle.dumps(payload))

    euclid.main(skip_done=True)
    expected_reruns = [2, 6] if damaged_product == "adaptive" else [7, 2, 6]
    assert ControlledSampler.runs == [7, 2, 6] + expected_reruns
    if damaged_product == "adaptive":
        assert m1_path.read_bytes() == saved_m1

    reference = euclid._load_stage("m1")["extra"]["s1"]
    assert reference["source_boundary"] == "hard_zero_outside_bbox"
    for tag in ("m2", "m3"):
        extra = euclid._load_stage(tag)["extra"]
        assert extra["source_interpolation"] == "bilinear"
        assert extra["objective"]["reference_id"] == euclid.artifact_fingerprint(
            reference["reference_provenance"],
        )
