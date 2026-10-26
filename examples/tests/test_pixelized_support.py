"""Tests for shared example support: sampling and stage persistence."""

import ast

import numpy as np
import pytest

import TinyLensGpu  # noqa: F401 -- select the JAX backend first


def _small_likelihood():
    import jax.numpy as jnp

    from TinyLensGpu.Inference import ParamU
    from TinyLensGpu.ObservationModel.LensImage.parametric_image_model import ImageProbModel
    from TinyLensGpu.PhysicalModel import PhysicalModel, SIE
    from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import GaussianEllipse

    sie = SIE(theta_E=ParamU("theta_E", 0.12, prior_type="uniform",
                             prior_settings=[0.05, 0.2], limits=[0.0, 1.0]),
              e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    sie.theta_E.to_dynamic()
    for p in (sie.e1, sie.e2, sie.center_x, sie.center_y):
        p.to_static()
    src = GaussianEllipse(flux=1.0, sigma=0.2, e1=0.0, e2=0.0,
                          center_x=0.0, center_y=0.0)
    for p in (src.flux, src.sigma, src.e1, src.e2, src.center_x, src.center_y):
        p.to_static()
    phys = PhysicalModel(lens_mass=[sie], source_light=[src], lens_light=[])
    psf = jnp.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    return ImageProbModel(
        image_data=jnp.full((8, 8), 0.05),
        noise_map=jnp.full((8, 8), 0.1),
        psf_kernel=psf, dpix=0.08, nsub=1, phys_model=phys,
        use_linear=True, mask=jnp.zeros((8, 8), dtype=bool),
    )


class _ControlledSampler:
    """External-sampler stand-in recording construction and run kwargs."""

    log_z = -7.5
    instances = []

    def __init__(self, prior, loglike, *, n_dim, **kwargs):
        self.prior, self.loglike, self.n_dim, self.kwargs = prior, loglike, n_dim, kwargs
        self.run_kwargs = None
        _ControlledSampler.instances.append(self)

    def run(self, *, verbose, n_eff, **kwargs):
        self.run_kwargs = dict(n_eff=n_eff, **kwargs)

    def posterior(self):
        samples = np.tile(np.asarray(self.prior(np.full(self.n_dim, 0.5))), (3, 1))
        return samples, np.asarray([0.0, -1.0, -2.0]), np.zeros(3)


@pytest.mark.unit
@pytest.mark.parametrize("vectorized", [True, False])
def test_run_sampling_controls_and_stage(vectorized):
    import jax.numpy as jnp

    from TinyLensGpu.example_support import SamplingConfig, run_sampling

    _ControlledSampler.instances.clear()
    likelihood = _small_likelihood()
    stage = run_sampling(
        likelihood,
        SamplingConfig(n_live=7, n_eff=11, vectorized=vectorized,
                       n_batch=100 if vectorized else None,
                       vectorized_chunk_size=2 if vectorized else None,
                       sampler_kwargs={"seed": 42, "filepath": "ckpt.h5",
                                       "resume": False},
                       run_kwargs={"f_live": 0.1}),
        tag="t", sampler_factory=_ControlledSampler,
    )
    sampler = _ControlledSampler.instances[-1]
    assert sampler.kwargs["n_live"] == 7
    assert sampler.kwargs["vectorized"] is vectorized
    assert sampler.kwargs["seed"] == 42
    assert sampler.kwargs["filepath"] == "ckpt.h5"
    assert sampler.kwargs["resume"] is False
    if vectorized:
        assert sampler.kwargs["n_batch"] == 100
    else:
        assert "n_batch" not in sampler.kwargs
    assert sampler.run_kwargs["n_eff"] == 11
    assert sampler.run_kwargs["f_live"] == 0.1
    # Expected values first: the vectorized wrapper is JIT-traced and its
    # tracers would escape into later direct calls on the same module.
    thetas = np.asarray([[0.08], [0.12], [0.18]])
    expected = np.asarray(
        [float(likelihood(jnp.asarray(t, dtype=jnp.float32))) for t in thetas]
    )
    if vectorized:
        out = np.asarray(sampler.loglike(thetas))
        assert out.shape == (3,)
    else:
        out = np.asarray([sampler.loglike(t) for t in thetas])
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-4)
    # Prior maps unit cube to the dynamic theta_E uniform interval.
    np.testing.assert_allclose(
        sampler.prior(jnp.asarray([0.5])), [0.125], rtol=1e-6,
    )
    assert stage.log_z == _ControlledSampler.log_z
    np.testing.assert_allclose(np.asarray(stage.samples), [[0.125]] * 3, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(stage.weights), np.exp([0.0, -1.0, -2.0]) / np.exp([0.0, -1.0, -2.0]).sum(),
        rtol=1e-6,
    )
    assert list(stage.param_names) == ["theta_E"]


class _DeepLogWSampler(_ControlledSampler):
    def posterior(self):
        samples, log_w, extra = super().posterior()
        return samples, log_w - 1000.0, extra


@pytest.mark.unit
def test_run_sampling_hooks_and_stabilized_weights():
    import jax.numpy as jnp

    from TinyLensGpu.example_support import SamplingConfig, run_sampling

    _ControlledSampler.instances.clear()
    calls = []
    likelihood = _small_likelihood()
    stage = run_sampling(
        likelihood,
        SamplingConfig(n_live=3, n_eff=5, vectorized=False, n_batch=None,
                       stabilize_weights=True, run_kwargs={"f_live": 0.1}),
        tag="t", sampler_factory=_DeepLogWSampler,
        before_sample=lambda prior, loglike, names: calls.append(("before", list(names))),
        after_sample=lambda sampler, st: calls.append(("after", st.log_z)),
    )
    assert calls[0] == ("before", ["theta_E"])
    assert calls[1] == ("after", _ControlledSampler.log_z)
    expected = np.exp([0.0, -1.0, -2.0])
    np.testing.assert_allclose(
        np.asarray(stage.weights), expected / expected.sum(), rtol=1e-6,
    )
    assert np.all(np.isfinite(np.asarray(stage.weights)))


@pytest.mark.unit
def test_stage_persistence_both_formats(tmp_path):
    from TinyLensGpu.Inference import StagePosterior
    from TinyLensGpu.example_support import (
        StageStorageConfig, load_stage, save_stage, stage_from_payload,
    )

    likelihood = _small_likelihood()
    stage = StagePosterior.from_likelihood(
        likelihood, np.asarray([[0.1], [0.2]]), np.asarray([0.5, 0.5]), log_z=-1.0,
    )
    # Schema-bearing write through a real stage payload.
    cfg = StageStorageConfig(output_dir=tmp_path, include_schema=True)
    save_stage(cfg, "x", None, None, None, None, extra={"k": 1}, stage=stage)
    payload = load_stage(cfg, "x")
    assert payload["stage_schema"]["param_names"] == ["theta_E"]
    assert "prior_specs" in payload["stage_schema"]
    assert payload["extra"] == {"k": 1}
    assert payload["samples"].dtype == np.float64
    recovered = stage_from_payload(payload)
    np.testing.assert_allclose(np.asarray(recovered.samples), stage.samples)
    assert list(recovered.param_names) == ["theta_E"]
    # Raw-arrays schema write keeps caller representations.
    raw_cfg = StageStorageConfig(output_dir=tmp_path, include_schema=True)
    save_stage(raw_cfg, "y", [[1.0], [2.0]], np.asarray([0.5, 0.5]),
               ["theta_E"], -2.0, extra={})
    raw = load_stage(raw_cfg, "y")
    assert raw["stage_schema"] == {"param_names": ["theta_E"]}
    rec = stage_from_payload(raw)
    np.testing.assert_allclose(np.asarray(rec.samples), [[1.0], [2.0]])
    # Legacy format: no schema keys at all.
    legacy_cfg = StageStorageConfig(output_dir=tmp_path, include_schema=False)
    save_stage(legacy_cfg, "z", np.asarray([[1.0]]), np.asarray([1.0]),
               ["theta_E"], -3.0, extra={"a": "b"})
    legacy = load_stage(legacy_cfg, "z")
    assert set(legacy) == {"samples", "weights", "param_names", "log_z", "extra"}
    assert legacy["param_names"] == ["theta_E"]
    rec = stage_from_payload(legacy)
    np.testing.assert_allclose(np.asarray(rec.samples), [[1.0]])
    assert list(rec.param_names) == ["theta_E"]
    # Top-level prior_specs (non-nested schema) recovers as well.
    top = dict(legacy)
    top["prior_specs"] = payload["stage_schema"]["prior_specs"]
    rec_top = stage_from_payload(top)
    assert list(rec_top.param_names) == ["theta_E"]
    # Raw JAX float32 grids round-trip without coercion.
    import jax.numpy as jnp

    jax_cfg = StageStorageConfig(output_dir=tmp_path, include_schema=False)
    jax_samples = jnp.asarray([[1.0]], dtype=jnp.float32)
    jax_weights = jnp.asarray([1.0], dtype=jnp.float32)
    save_stage(jax_cfg, "w", jax_samples, jax_weights, ["theta_E"], -4.0)
    loaded = load_stage(jax_cfg, "w")
    assert isinstance(loaded["samples"], jnp.ndarray)
    assert loaded["samples"].dtype == jnp.float32
    assert loaded["weights"].dtype == jnp.float32


@pytest.mark.unit
def test_slacs_stage_a_sampling_save_recovery(tmp_path, monkeypatch, examples_dir):
    """Representative SLACS stage-A runs through shared support end to end."""
    import jax.numpy as jnp

    from TinyLensGpu.Inference import StagePosterior
    from TinyLensGpu.example_support import (
        SamplingConfig, StageStorageConfig, run_sampling, save_stage,
        load_stage, stage_from_payload,
    )

    path = examples_dir / "pix_src_demo_operator" / "pipe" / "slacs_examples" / "pipeline.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [
        n for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_run_sampler", "_stage_storage", "_dump_stage",
                       "_load_stage", "_stage_from_payload",
                       "build_stage_a_likelihood", "run_stage_a", "_print_summary"}
    ]
    assert {
        n.name for n in selected
    } == {"_run_sampler", "_stage_storage", "_dump_stage", "_load_stage",
          "_stage_from_payload", "build_stage_a_likelihood", "run_stage_a",
          "_print_summary"}

    import caskade as ck
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from TinyLensGpu.Inference import ParamU
    from TinyLensGpu.ObservationModel import PointSourceProbModel
    from TinyLensGpu.ObservationModel.LensImage.parametric_image_model import ImageProbModel
    from TinyLensGpu.PhysicalModel import PhysicalModel, SIE, Shear
    from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import GaussianEllipse
    from TinyLensGpu.utils import generate_radial_basis_knots
    from TinyLensGpu.utils.misc import weighted_quantile
    from TinyLensGpu.visualizer import plot_model_results

    class FakeSampler:
        log_z = -9.0
        instances = []

        def __init__(self, prior, loglike, *, n_dim, **kwargs):
            self.prior, self.n_dim, self.kwargs = prior, n_dim, kwargs
            FakeSampler.instances.append(self)

        def run(self, *, verbose, n_eff):
            self.n_eff = n_eff

        def posterior(self):
            samples = np.tile(np.asarray(self.prior(jnp.full(self.n_dim, 0.5))), (3, 1))
            return samples, np.asarray([0.0, -0.5, -1.0]), np.zeros(3)

    namespace = dict(
        np=np, jnp=jnp, plt=plt, ck=ck, time=__import__("time"),
        ParamU=ParamU, StagePosterior=StagePosterior,
        Sampler=FakeSampler,
        SamplingConfig=SamplingConfig, StageStorageConfig=StageStorageConfig,
        run_sampling=run_sampling, save_stage=save_stage,
        load_stage=load_stage, stage_from_payload=stage_from_payload,
        weighted_quantile=weighted_quantile,
        ImageProbModel=ImageProbModel, PointSourceProbModel=PointSourceProbModel,
        PhysicalModel=PhysicalModel, SIE=SIE, Shear=Shear, GaussianEllipse=GaussianEllipse,
        OUT_DIR=tmp_path, DPIX=0.08, NSUB=1, MASK_RADIUS=3.0,
        N_GAUSSIANS_SRC=2, N_GAUSSIANS_LENS=2,
        generate_radial_basis_knots=generate_radial_basis_knots,
        plot_model_results=plot_model_results,
    )
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
         namespace)

    psf = np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    stage, medians, _ = namespace["run_stage_a"](
        np.full((8, 8), 0.05), np.full((8, 8), 0.1), psf,
    )
    assert FakeSampler.instances[0].kwargs["n_live"] == 300
    assert FakeSampler.instances[0].kwargs["n_batch"] == 100
    assert FakeSampler.instances[0].n_eff == 2000
    saved = namespace["_load_stage"]("a")
    assert saved["stage_schema"]["param_names"] == list(stage.param_names)
    recovered = namespace["_stage_from_payload"](saved)
    np.testing.assert_allclose(np.asarray(recovered.samples), stage.samples)
    assert (tmp_path / "stage_a.pkl").exists()


@pytest.mark.unit
def test_example_support_import_has_no_process_mutation():
    """Isolated import adds no cwd/env/JAX-precision mutations."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    script = r'''
import os, json
import jax
import TinyLensGpu
before = (os.getcwd(), dict(os.environ), jax.config.jax_enable_x64,
          jax.config.jax_default_matmul_precision)
import TinyLensGpu.example_support
after = (os.getcwd(), dict(os.environ), jax.config.jax_enable_x64,
         jax.config.jax_default_matmul_precision)
assert before == after, "example_support import mutated process state"
print("ok")
'''
    env = {k: v for k, v in os.environ.items() if k != "CASKADE_BACKEND"}
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-c", script], cwd=repo, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.unit
def test_slacs_entrypoint_help(examples_dir):
    """The SLACS entry point stays launchable from a fresh process."""
    import os
    import subprocess
    import sys

    script = examples_dir / "pix_src_demo_operator" / "pipe" / "slacs_examples" / "model.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(examples_dir.parent) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.unit
def test_posterior_quantiles():
    from TinyLensGpu.Inference import StagePosterior
    from TinyLensGpu.example_support import posterior_quantiles

    stage = StagePosterior(
        np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]),
        np.asarray([0.2, 0.6, 0.2]),
        param_names=["a", "b"],
    )
    qs = posterior_quantiles(stage)
    assert list(qs) == ["a", "b"]
    assert qs["a"].shape == (3,)
    assert qs["a"][1] == pytest.approx(0.5)   # interpolated CDF median
    assert qs["b"][1] == pytest.approx(1.5)
