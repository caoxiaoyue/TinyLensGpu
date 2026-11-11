"""Compatibility checks at dense example stages and persisted artifacts."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

import TinyLensGpu  # noqa: F401 -- select the JAX backend before components
from TinyLensGpu.example_support import StageStorageConfig, load_stage, stage_from_payload


class ControlledSampler:
    """Replace external nested sampling with a deterministic prior-centre posterior."""

    log_z = -7.25
    instances = []

    def __init__(self, prior, loglike, *, n_dim, **kwargs):
        self.prior, self.n_dim, self.kwargs = prior, n_dim, kwargs
        type(self).instances.append(self)

    def run(self, *, verbose, n_eff):
        self.n_eff = n_eff

    def posterior(self):
        samples = np.tile(np.asarray(self.prior(np.full(self.n_dim, 0.5))), (3, 1))
        return samples, np.array([0.0, -1.0, -2.0]), np.zeros(3)


def import_pipeline(path):
    previous_cwd = Path.cwd()
    try:
        spec = importlib.util.spec_from_file_location("dense_" + path.parent.name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.chdir(previous_cwd)
    return module


PIPELINES = [
    ("pix_src_demo/pipe/slacs_examples/pipeline.py", 300, 100, 2000),
    ("pix_src_demo/pipe/pre_subtract_lens/pix_src_pipe/model.py", 200, 200, 2000),
    ("pix_src_demo/pipe/pre_subtract_lens/pix_src_pipe_mge/model.py", 200, 200, 2000),
    ("pix_src_demo/pipe/pre_subtract_lens/pix_src_pipe_sersic/model.py", 200, 200, 2000),
    ("pix_src_demo/pipe/simult_fit_lens_source/mge_lens_light/model.py", 200, 200, 800),
]


def configure(module, tmp_path, monkeypatch):
    monkeypatch.setattr(module, "OUT_DIR", tmp_path)
    monkeypatch.setattr(module, "NSUB", 1)
    monkeypatch.setattr(module, "Sampler", ControlledSampler)
    for name in ("N_GAUSSIANS_SRC", "N_GAUSSIANS_LENS"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, 2)


def small_data():
    image = np.full((8, 8), 0.05)
    return image, np.full_like(image, 0.1), np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])


@pytest.mark.parametrize("relative_path,n_live,n_batch,n_eff", PIPELINES)
def test_dense_stage_sampling_and_legacy_artifact(
    relative_path, n_live, n_batch, n_eff, examples_dir, tmp_path, monkeypatch,
):
    """Real stage A preserves controls, parameter order, and its legacy pickle."""
    module = import_pipeline(examples_dir / relative_path)
    configure(module, tmp_path, monkeypatch)
    ControlledSampler.instances.clear()
    result = module.run_stage_a(*small_data())
    samples, weights, names, medians = result[:4]
    sampler = ControlledSampler.instances[-1]
    assert sampler.kwargs == dict(n_live=n_live, vectorized=True, n_batch=n_batch)
    assert sampler.n_eff == n_eff
    payload = load_stage(StageStorageConfig(tmp_path, False), "a")
    assert set(payload) == {"samples", "weights", "param_names", "log_z", "extra"}
    assert payload["log_z"] == -7.25
    assert payload["param_names"] == names
    assert payload["samples"].dtype == payload["weights"].dtype == np.float64
    np.testing.assert_array_equal(payload["samples"], samples)
    np.testing.assert_array_equal(payload["weights"], weights)
    np.testing.assert_allclose(weights, [0.6652409558, 0.2447284711, 0.0900305732])
    recovered = stage_from_payload(payload)
    assert recovered.medians() == medians
    assert recovered.prior_specs is None
    if len(result) == 5:
        np.testing.assert_array_equal(payload["extra"]["lens_light_model"], result[4])
    assert (tmp_path / "stage_a_model.png").exists()


@pytest.mark.parametrize("interpolation", [None, "cubic_hermite"])
def test_dense_slacs_stage_transition_and_plot_data(
    examples_dir, tmp_path, monkeypatch, interpolation,
):
    """A saved stage-A posterior feeds the actual pixelized M stage and figure."""
    module = import_pipeline(examples_dir / PIPELINES[0][0])
    configure(module, tmp_path, monkeypatch)
    monkeypatch.setattr(module, "NSUB_PIX", 1)
    monkeypatch.setattr(module, "NSRC", 3)
    image, noise, psf = small_data()
    module.run_stage_a(image, noise, psf, source_interpolation=interpolation)
    payload_a = load_stage(StageStorageConfig(tmp_path, False), "a")
    recovered = stage_from_payload(payload_a)
    feature_mask = np.zeros_like(image, dtype=bool)
    feature_mask[3:5, 3:5] = True
    *_, lens_light_model = module.run_stage_l(
        image, noise, psf, feature_mask, source_interpolation=interpolation,
    )
    recorded = []
    original_save = module.plt.savefig

    def save_figure(*args, **kwargs):
        recorded.append([np.asarray(ax.images[0].get_array())
                         for ax in module.plt.gcf().axes if ax.images])
        return original_save(*args, **kwargs)

    monkeypatch.setattr(module.plt, "savefig", save_figure)
    samples, weights, names, medians = module.run_stage_m(
        image, noise, psf, feature_mask, lens_light_model,
        recovered.samples, recovered.weights, recovered.param_names, None,
        source_interpolation=interpolation,
    )
    assert ControlledSampler.instances[-1].kwargs == dict(n_live=300, vectorized=False)
    likelihood = module.build_stage_m_likelihood(
        image - lens_light_model, noise, psf, feature_mask, recovered, None,
        source_interpolation=interpolation,
    )
    import jax.numpy as jnp
    inversion, diagnostics = likelihood.evaluate_with_diagnostics(
        jnp.array([medians[n] for n in names]), include_effective_dof=True,
    )
    assert diagnostics.valid
    assert len(recorded[-1]) == 4
    np.testing.assert_allclose(recorded[-1][1], inversion.model_image, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        recorded[-1][2][~feature_mask],
        np.asarray(diagnostics.normalized_residuals)[~feature_mask],
        rtol=1e-5, atol=1e-6,
    )
    np.testing.assert_allclose(recorded[-1][3], np.asarray(inversion.source_pixels).reshape(3, 3),
                               rtol=1e-5, atol=1e-6)
    payload_m = load_stage(StageStorageConfig(tmp_path, False), "m")
    assert set(payload_m) == {"samples", "weights", "param_names", "log_z", "extra"}
    mode = interpolation or "bilinear"
    assert likelihood.source_model.interpolation == mode
    assert payload_a["extra"]["source_interpolation"] == mode
    assert payload_m["extra"]["source_interpolation"] == mode
    np.testing.assert_array_equal(payload_m["samples"], samples)
    plot = tmp_path / "stage_m_model.png"
    assert plot.exists()

    monkeypatch.setattr(module, "load_lens_data", lambda **kwargs: (image, noise, psf, None))
    monkeypatch.setattr(module, "run_stage_b", lambda *args, **kwargs: feature_mask)
    monkeypatch.setattr(module, "_position_likelihood_from_stage_a", lambda *args: None)
    stage_path = tmp_path / "stage_m.pkl"
    original = stage_path.read_bytes()
    sampler_count = len(ControlledSampler.instances)
    plot.unlink()
    module.main(skip_done=True)
    assert plot.exists()
    assert len(ControlledSampler.instances) == sampler_count
    assert stage_path.read_bytes() == original
    np.testing.assert_allclose(recorded[-1][1], inversion.model_image, rtol=1e-5, atol=1e-6)


def test_dense_simple_script_artifacts(examples_dir, tmp_path, monkeypatch):
    """The unchanged script launch writes CSV and gzip payloads using shared sampling."""
    import gzip
    import pickle
    import runpy

    import matplotlib.pyplot as plt
    import nautilus
    from astropy.io import fits

    script = tmp_path / "fit_lens_src.py"
    script.write_text((examples_dir / "pix_src_demo/simple/pix_src/fit_lens_src.py").read_text())
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    image, noise, psf = small_data()
    for name, array in (("image", image), ("noise", noise), ("psf", psf),
                        ("mask", np.zeros_like(image, dtype=np.uint8))):
        fits.writeto(data_dir / f"{name}.fits", array)
    monkeypatch.setattr(nautilus, "Sampler", ControlledSampler)
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.delenv("TINYLENS_SOURCE_INTERPOLATION", raising=False)
    previous_cwd = Path.cwd()
    try:
        namespace = runpy.run_path(str(script), run_name="__main__")
    finally:
        os.chdir(previous_cwd)
        plt.close("all")
    stage = namespace["stage"]
    from TinyLensGpu.Inference import StagePosterior
    assert isinstance(stage, StagePosterior)
    sampler = ControlledSampler.instances[-1]
    assert sampler.kwargs == dict(n_live=200, vectorized=False)
    assert sampler.n_eff == 800
    with gzip.open(tmp_path / "output/fit_results.pkl.gz", "rb") as stream:
        payload = pickle.load(stream)
    assert set(payload) == {
        "samples", "weights", "log_z", "param_names",
        "source_interpolation", "source_boundary",
    }
    assert payload["source_interpolation"] == "bilinear"
    assert payload["source_boundary"] == "hard_zero_outside_bbox"
    assert payload["param_names"] == stage.param_names
    np.testing.assert_array_equal(payload["samples"], stage.samples)
    np.testing.assert_array_equal(payload["weights"], stage.weights)
    csv = np.loadtxt(tmp_path / "output/fit_samples.csv", delimiter=",")
    np.testing.assert_array_equal(csv, stage.samples)
    assert (tmp_path / "output/fit_map_visualization.png").exists()
