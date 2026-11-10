"""Native reference artifacts in the maintained lens-light-subtracted fits."""

import importlib.util
import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from TinyLensGpu.Inference import StagePosterior
from TinyLensGpu.example_support.adaptation import (
    artifact_fingerprint,
    pixelized_observation_provenance,
)


class ControlledSampler:
    log_z = -12.0
    unit_point = 0.5
    runs = []

    def __init__(self, prior, loglike, *, n_dim, **kwargs):
        self.points = np.tile(np.asarray(prior(jnp.full(n_dim, self.unit_point))), (3, 1))

    def run(self, **kwargs):
        self.runs.append(self.points.shape[1])

    def posterior(self):
        return self.points, np.log([0.2, 0.3, 0.5]), np.zeros(3)


@pytest.fixture(params=["pix_src_pipe", "pix_src_pipe_mge", "pix_src_pipe_sersic"])
def workflow(request, examples_dir, tmp_path, monkeypatch):
    path = examples_dir / "pix_src_demo_operator/pipe/pre_subtract_lens" / request.param / "model.py"
    spec = importlib.util.spec_from_file_location(request.param, path)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as context:
        context.chdir(path.parent)
        spec.loader.exec_module(module)
    ControlledSampler.runs.clear()
    for name, value in dict(OUT_DIR=tmp_path, NSRC=3, NSUB_PIX=2, DPIX=0.08, Sampler=ControlledSampler).items():
        monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(module, "overlay_critical_and_caustics", lambda **kw: None)
    return module


@pytest.fixture
def small_fit():
    y, x = np.indices((10, 10))
    lens_light = (0.05 + 0.13 * np.exp(-((x - 4)**2 + (y - 4)**2) / 7)).astype(np.float32)
    source_arc = 0.04 + 0.25 * np.exp(-((x - 7)**2 + (y - 4)**2) / 3)
    image = (lens_light + source_arc).astype(np.float32)
    noise = np.full_like(image, 0.1)
    psf = np.array([[0., 0.04, 0.], [0.18, 0.63, 0.05], [0., 0.1, 0.]], dtype=np.float32)
    mask = x < 2
    names = ["theta_E", "e1_mass", "e2_mass", "center_x_mass", "center_y_mass", "gamma1", "gamma2"]
    center = np.asarray([0.17, 0.1, 0.03, 0.02, -0.01, 0.02, -0.01])
    stage_a = StagePosterior(np.stack([center - 0.001, center, center + 0.001]), np.ones(3), names)
    return image, noise, psf, mask, lens_light, stage_a


@pytest.mark.integration
def test_m1_reference_is_the_native_source_component(workflow, small_fit):
    image, noise, psf, mask, lens_light, stage_a = small_fit
    stage, point, package = workflow.run_stage_m1(
        image, noise, psf, mask, lens_light, stage_a, None, 0.0,
    )
    likelihood = workflow.build_stage_m1_likelihood(
        image - lens_light, noise, psf, mask, stage_a, None, 0.0,
    )
    theta = jnp.asarray([point[name] for name in stage.param_names])
    result, diagnostics = likelihood.evaluate_with_diagnostics(theta, include_components=True)
    assert diagnostics.valid
    assert package["adapt_image"].shape == image.shape
    assert package["source_image"].shape == (3, 3)
    np.testing.assert_allclose(package["adapt_image"], diagnostics.source_image, atol=1e-6)
    np.testing.assert_allclose(package["source_image"].ravel(), result.source_pixels, atol=1e-6)
    np.testing.assert_array_equal(package["adapt_image"][mask], 0.0)


@pytest.mark.integration
def test_subtracted_adaptive_stages_resume_matching_objective(workflow, small_fit, tmp_path, monkeypatch):
    image, noise, psf, mask, lens_light, stage_a = small_fit
    stage_a_light = np.zeros_like(image)
    monkeypatch.setattr(workflow, "MASK_RADIUS", 0.42)
    circle = workflow._make_circular_mask(image.shape, workflow.DPIX, radius_arcsec=0.42)
    combined_mask = mask | circle

    def save_foreground(tag, foreground):
        workflow._dump_stage(
            tag, stage_a.samples, stage_a.weights, stage_a.param_names, -1.0,
            extra=dict(medians=stage_a.medians(), lens_light_model=foreground), stage=stage_a,
        )

    def make_feature_mask(data, noise_map, foreground, *args, **kwargs):
        np.testing.assert_array_equal(data, image)
        np.testing.assert_array_equal(noise_map, noise)
        np.testing.assert_array_equal(foreground, stage_a_light)
        return mask

    def record_plot(tag, likelihood, medians, names, path):
        theta = jnp.asarray([medians[name] for name in names])
        _, diagnostics = likelihood.evaluate_with_diagnostics(theta)
        assert diagnostics.valid
        assert diagnostics.evidence_valid
        if tag in {"stage-M2", "stage-M3"}:
            scale = np.asarray(diagnostics.regularization_scale)
            assert np.all(np.isfinite(scale))
            assert np.any(scale > 1.0)
        Path(path).write_text(diagnostics.evidence_logdet)

    save_foreground("a", stage_a_light)
    save_foreground("l", lens_light)
    monkeypatch.setattr(workflow, "load_lens_data", lambda **kw: (image, noise, psf, None))
    monkeypatch.setattr(workflow, "run_stage_b", make_feature_mask)
    monkeypatch.setattr(workflow, "_position_likelihood_from_stage_a", lambda medians: None)
    monkeypatch.setattr(workflow, "_plot_pix_stage", record_plot)

    tags = ["m0", "m1", "m2", "m3"]
    paths = {tag: tmp_path / f"stage_{tag}.pkl" for tag in tags}

    def resume(score="overlap"):
        workflow.main(skip_done=True, out_dir=tmp_path, evidence_logdet=score, evidence_block_size=2)
        return {tag: pickle.loads(path.read_bytes())["extra"] for tag, path in paths.items()}

    first = resume()
    assert len(ControlledSampler.runs) == 3
    package = first["m1"]["s1"]
    uniform = workflow.build_stage_m1_likelihood(
        image - lens_light, noise, psf, mask, stage_a, None, first["m0"]["log_lambda_best"],
        circle, evidence_block_size=2,
    )
    observation = pixelized_observation_provenance(uniform)
    assert package["adapt_image"].shape == image.shape
    assert package["source_image"].shape == (3, 3)
    np.testing.assert_array_equal(package["adapt_image"][combined_mask], 0)
    if workflow.__name__ == "pix_src_pipe":
        s0 = first["m0"]["s0"]
        assert s0["source_image"].shape == (3, 3)
        assert s0["scale_map"].shape == (9,)
        assert "adapt_image" not in s0
    reference_id = artifact_fingerprint(package["reference_provenance"])
    for tag in tags:
        objective = first[tag]["objective"]
        assert objective["observation"] == observation
        assert objective["evidence_logdet"] == "overlap"
        assert (tmp_path / f"stage_{tag}_model.png").read_text() == "overlap"
        assert objective["evidence_block_size"] == 2
        assert objective["reference_id"] == (reference_id if tag in {"m2", "m3"} else None)
        if tag != "m0":
            assert objective["sampling"]["n_batch"] == 100

    snapshots = {tag: path.read_bytes() for tag, path in paths.items()}
    resume()
    assert len(ControlledSampler.runs) == 3
    assert {tag: path.read_bytes() for tag, path in paths.items()} == snapshots

    with monkeypatch.context() as context:
        context.setattr(ControlledSampler, "unit_point", 0.53)
        workflow.run_stage_m1(
            image, noise, psf, mask, lens_light, stage_a, None, first["m0"]["log_lambda_best"],
            circle, upstream=first["m1"]["objective"]["upstream"], evidence_block_size=2,
        )
    assert len(ControlledSampler.runs) == 4
    replacement = paths["m1"].read_bytes()
    changed = resume()
    assert len(ControlledSampler.runs) == 6
    assert paths["m0"].read_bytes() == snapshots["m0"]
    assert paths["m1"].read_bytes() == replacement
    new_reference = artifact_fingerprint(changed["m1"]["s1"]["reference_provenance"])
    assert new_reference != reference_id
    for tag in ("m2", "m3"):
        assert changed[tag]["objective"]["reference_id"] == new_reference
    assert not np.allclose(changed["m1"]["s1"]["adapt_image"], package["adapt_image"])

    block = resume("block")
    assert len(ControlledSampler.runs) == 9
    for tag in tags:
        assert block[tag]["objective"]["evidence_logdet"] == "block"
        assert (tmp_path / f"stage_{tag}_model.png").read_text() == "block"
