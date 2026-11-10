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
        load_stage, stage_from_payload, resolve_source_interpolation,
    )

    path = examples_dir / "pix_src_demo_operator" / "pipe" / "slacs_examples" / "pipeline.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [
        n for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"_run_sampler", "_stage_storage", "_dump_stage",
                       "_load_stage", "_stage_from_payload",
                       "_initial_source_kwargs",
                       "build_stage_a_likelihood", "run_stage_a", "_print_summary"}
    ]
    assert {
        n.name for n in selected
    } == {"_run_sampler", "_stage_storage", "_dump_stage", "_load_stage",
          "_stage_from_payload", "_initial_source_kwargs",
          "build_stage_a_likelihood", "run_stage_a",
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
        resolve_source_interpolation=resolve_source_interpolation,
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
import sys
pyplot_before = 'matplotlib.pyplot' in sys.modules
import TinyLensGpu.example_support
import TinyLensGpu.example_support.source
import TinyLensGpu.example_support.plotting
after = (os.getcwd(), dict(os.environ), jax.config.jax_enable_x64,
         jax.config.jax_default_matmul_precision)
assert ('matplotlib.pyplot' in sys.modules) == pyplot_before, \
    "support import loaded matplotlib.pyplot"
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


# ------------------------------------------------------------------
# Shared source-package and drawing helpers (ticket 10)
# ------------------------------------------------------------------


@pytest.fixture
def close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def _small_source_export_model(image_data=None):
    import TinyLensGpu  # ensure CASKADE_BACKEND selected before caskade
    import jax.numpy as jnp
    from TinyLensGpu.PhysicalModel import PhysicalModel, SIE
    from TinyLensGpu.PhysicalModel.LensImage.Pixelized.Light import PixelizedSourceModel
    from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
    from TinyLensGpu.ForwardSimulation.LensImage.pixelized import PixelizedLensSimulator
    from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import PixelizedImageProbModelOperator
    from TinyLensGpu.Inference import ParamU

    mass = SIE(theta_E=0.12, e1=0.05, e2=0.0, center_x=0.0, center_y=0.0)
    for p in (mass.theta_E, mass.e1, mass.e2, mass.center_x, mass.center_y):
        p.to_static()
    mass.theta_E.to_dynamic()
    lam = ParamU("log_lambda_reg", 0.0, prior_type="uniform",
                 prior_settings=[-3.0, 3.0], limits=[-3.0, 3.0])
    rho = ParamU("adaptive_reg_rho", 0.6, prior_type="uniform",
                 prior_settings=[0.0, 3.0], limits=[0.0, 3.0])
    lam.to_dynamic()
    rho.to_dynamic()
    source = PixelizedSourceModel(
        n=3, log_lambda_reg=lam, regularization_type="first-order",
        adaptive_reg_rho=rho,
    )
    phys = PhysicalModel(lens_mass=[mass], source_light=[source], lens_light=[])
    psf = jnp.asarray([[1.0]])
    bbox = (-0.4, 0.4, -0.4, 0.4)
    sim = PixelizedLensSimulator(phys, SimulatorConfig(dpix=0.08, npix=10, psf_kernel=psf))
    reference = sim.simulate(jnp.linspace(0.1, 1.0, 9), source_bbox=bbox)
    image = reference if image_data is None else jnp.asarray(image_data)
    return PixelizedImageProbModelOperator(
        image, jnp.ones_like(image) * 0.05, psf, 0.08, phys,
        solver_type="pcg", block_size=2, fixed_source_bbox=bbox,
        adapt_image=reference,
    )


def _raw_source_package(n=3):
    import numpy as np

    x = np.linspace(-0.3, 0.3, n)
    return dict(
        source_pixels=np.arange(n * n, dtype=np.float64) + 1.0,
        source_image=np.arange(n * n, dtype=np.float64).reshape(n, n) + 1.0,
        source_bbox=(-0.3, 0.3, -0.3, 0.3),
        source_x_axis=x,
        source_y_axis=x.copy(),
        n=n,
    )


@pytest.mark.unit
def test_shared_source_helpers_importable():
    from TinyLensGpu.example_support import (
        draw_image_panel, export_source_package, require_valid_reconstruction,
        source_axes_from_bbox, validate_source_package,
    )

    x, y = source_axes_from_bbox((-0.3, 0.3, -0.2, 0.2), 3)
    np.testing.assert_allclose(x, [-0.3, 0.0, 0.3])
    np.testing.assert_allclose(y, [-0.2, 0.0, 0.2])


@pytest.mark.unit
def test_validate_source_package_raw_vs_prepared():
    from TinyLensGpu.example_support import validate_source_package

    # Raw export: no lambda keys required, no scale_map generated, same dict.
    raw = _raw_source_package()
    assert validate_source_package(
        raw, expected_n=3, rho=1.0, require_regularization=False
    ) is raw
    assert "scale_map" not in raw
    # Prepared path requires the lambda keys.
    with pytest.raises(KeyError, match="lambda_best"):
        validate_source_package(_raw_source_package(), expected_n=3, rho=1.0)
    prepared = _raw_source_package()
    prepared.update(lambda_best=1.0, log_lambda_best=0.0)
    assert validate_source_package(prepared, expected_n=3, rho=1.0) is prepared
    assert prepared["scale_map"].dtype == np.float32
    assert prepared["scale_map"].shape == (9,)
    assert np.all(prepared["scale_map"] > 0)
    # Error boundaries: obsolete keys, shape, axis, bbox, n mismatches.
    bad = _raw_source_package(); bad["nx"] = 3
    bad.update(lambda_best=1.0, log_lambda_best=0.0)
    with pytest.raises(KeyError, match="legacy"):
        validate_source_package(bad, expected_n=3, rho=1.0)
    bad = _raw_source_package() | {"lambda_best": 1.0, "log_lambda_best": 0.0}
    bad["n"] = 4
    with pytest.raises(ValueError, match="n=4"):
        validate_source_package(bad, expected_n=3, rho=1.0)
    bad = _raw_source_package() | {"lambda_best": 1.0, "log_lambda_best": 0.0}
    bad["source_pixels"] = np.ones(7)
    with pytest.raises(ValueError, match="shape"):
        validate_source_package(bad, expected_n=3, rho=1.0)
    bad = _raw_source_package() | {"lambda_best": 1.0, "log_lambda_best": 0.0}
    bad["source_bbox"] = (-0.3, 0.5, -0.3, 0.3)
    with pytest.raises(ValueError, match="square|rectangular"):
        validate_source_package(bad, expected_n=3, rho=1.0)
    bad = _raw_source_package(); bad["scale_map"] = -np.ones(9)
    bad.update(lambda_best=1.0, log_lambda_best=0.0)
    with pytest.raises(ValueError, match="positive"):
        validate_source_package(bad, expected_n=3, rho=1.0)


@pytest.mark.unit
def test_validate_source_package_optional_bbox_axis_checks():
    from TinyLensGpu.example_support import validate_source_package

    pkg = _raw_source_package()
    pkg.update(lambda_best=1.0, log_lambda_best=0.0)
    # Galan predefine options: in-place bbox normalization + expected box + axes.
    out = validate_source_package(
        pkg, expected_n=3, rho=1.0, expected_source_bbox=(-0.3, 0.3, -0.3, 0.3),
        normalize_bbox=True, check_axes=True,
    )
    assert out is pkg and pkg["source_bbox"] == (-0.3, 0.3, -0.3, 0.3)
    with pytest.raises(ValueError, match="fixed stage-A source bbox"):
        validate_source_package(
            _raw_source_package() | {"lambda_best": 1.0, "log_lambda_best": 0.0},
            expected_n=3, rho=1.0, expected_source_bbox=(-1, 1, -1, 1),
            normalize_bbox=True,
        )
    bad = _raw_source_package(); bad["source_x_axis"] = np.zeros(3)
    bad.update(lambda_best=1.0, log_lambda_best=0.0)
    with pytest.raises(ValueError, match="source_x_axis"):
        validate_source_package(
            bad, expected_n=3, rho=1.0, normalize_bbox=True, check_axes=True
        )
    # Axis tolerance boundary: +5e-8 on the zero coordinate passes (atol=1e-7);
    # +6e-7 on a nonzero coordinate fails.
    pkg = _raw_source_package() | {"lambda_best": 1.0, "log_lambda_best": 0.0}
    pkg["source_x_axis"] = pkg["source_x_axis"].copy()
    pkg["source_x_axis"][1] += 5e-8
    validate_source_package(
        pkg, expected_n=3, rho=1.0, normalize_bbox=True, check_axes=True
    )
    bad = _raw_source_package() | {"lambda_best": 1.0, "log_lambda_best": 0.0}
    bad["source_x_axis"] = bad["source_x_axis"].copy()
    bad["source_x_axis"][0] += 6e-7
    with pytest.raises(ValueError, match="source_x_axis"):
        validate_source_package(
            bad, expected_n=3, rho=1.0, normalize_bbox=True, check_axes=True
        )
    # A positive supplied float64 scale_map is coerced to equal float32 in place.
    pkg = _raw_source_package() | {"lambda_best": 1.0, "log_lambda_best": 0.0}
    supplied = np.linspace(0.5, 2.0, 9, dtype=np.float64)
    pkg["scale_map"] = supplied
    out = validate_source_package(pkg, expected_n=3, rho=1.0)
    assert out is pkg
    assert pkg["scale_map"].dtype == np.float32
    np.testing.assert_allclose(pkg["scale_map"], supplied, rtol=1e-6)


@pytest.mark.unit
def test_require_valid_reconstruction():
    from types import SimpleNamespace

    from TinyLensGpu.ObservationModel.LensImage.inversion_result import (
        InvalidReconstructionError,
    )
    from TinyLensGpu.example_support import require_valid_reconstruction

    # Joint-unsupported effective diagnostics stay valid and must pass.
    ok = SimpleNamespace(valid=True, invalid_reason=None)
    require_valid_reconstruction(None, ok, context="test")
    bad = SimpleNamespace(valid=False, invalid_reason="solver did not converge")
    with pytest.raises(InvalidReconstructionError, match="test"):
        require_valid_reconstruction(None, bad, context="test")


@pytest.mark.unit
def test_slacs_export_validate_next_stage_and_plot(
    tmp_path, monkeypatch, examples_dir, close_figures
):
    """SLACS export -> prepared validation -> M2 builder -> plot, all shared."""
    import ast

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import jax.numpy as jnp
    import caskade as ck

    from TinyLensGpu.Inference import ParamU
    from TinyLensGpu.ObservationModel.LensImage.inversion_result import (
        InvalidReconstructionError,
    )
    from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import (
        PixelizedImageProbModelOperator,
    )
    from TinyLensGpu.PhysicalModel import (
        EPL, PhysicalModel, Shear, PixelizedSourceModel,
    )
    from TinyLensGpu.example_support import (
        adapt_image_kwargs, export_source_package, require_reference_observation, validate_source_package,
        require_valid_reconstruction, resolve_source_interpolation, source_axes_from_bbox,
        draw_image_panel,
    )
    from TinyLensGpu.visualizer import overlay_critical_and_caustics

    path = examples_dir / "pix_src_demo_operator" / "pipe" / "slacs_examples" / "pipeline.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    wanted = {
        "_epl_mass_from_medians", "_pnpg_kwargs", "_fixed_source_kwargs",
        "build_stage_m2_likelihood", "_plot_pix_stage",
        "_solve_pixel_source_for_package", "_validate_source_package",
    }
    selected = [
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in wanted
    ]
    assert {n.name for n in selected} == wanted
    namespace = dict(
        np=np, jnp=jnp, plt=plt, ck=ck, time=__import__("time"),
        ParamU=ParamU, EPL=EPL, Shear=Shear,
        PixelizedSourceModel=PixelizedSourceModel,
        PixelizedImageProbModelOperator=PixelizedImageProbModelOperator,
        PhysicalModel=PhysicalModel,
        export_source_package=export_source_package,
        validate_source_package=validate_source_package,
        require_valid_reconstruction=require_valid_reconstruction,
        resolve_source_interpolation=resolve_source_interpolation,
        source_axes_from_bbox=source_axes_from_bbox,
        draw_image_panel=draw_image_panel,
        adapt_image_kwargs=adapt_image_kwargs,
        require_reference_observation=require_reference_observation,
        InvalidReconstructionError=InvalidReconstructionError,
        overlay_critical_and_caustics=overlay_critical_and_caustics,
        DPIX=0.08, NSUB_PIX=1, NSRC=3, SOLVER_TYPE="pcg",
        PIXEL_REGULARIZATION_TYPE="first-order",
        ADAPTIVE_REG_RHO=1.0, ADAPTIVE_REG_RHO_PRIOR_MAX=8.0,
        SOURCE_BBOX_PADDING=0.05,
        PNPG_MAX_ITER=200, PNPG_RTOL=2.0e-2, PNPG_POWER_ITER=10,
    )
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
         namespace)

    model = _small_source_export_model()
    names = [p.name for p in model.get_dynamic_params()]
    medians = {n: float(np.asarray(p.value)) for n, p in
               zip(names, model.get_dynamic_params())}

    # Keep the native source arc separate from the fitted source raster.
    pkg = namespace["_solve_pixel_source_for_package"](model, medians, names)
    assert pkg["source_pixels"].dtype == np.float64
    assert pkg["source_image"].shape == (3, 3)
    x_ax, y_ax = pkg["source_x_axis"], pkg["source_y_axis"]
    np.testing.assert_allclose(x_ax, np.linspace(*pkg["source_bbox"][:2], 3))
    assert len(x_ax) == len(y_ax) == 3

    assert pkg["adapt_image"].shape == model.image_data.shape

    # Validate the fitted source grid before the adaptive builder.
    prepared = validate_source_package(
        pkg | {"lambda_best": 1.0, "log_lambda_best": 0.0},
        expected_n=3, rho=1.0, require_regularization=False,
    )
    m1_medians = dict(
        theta_E=0.17, gamma=2.1, e1_mass=0.1, e2_mass=0.03,
        center_x_mass=0.0, center_y_mass=0.0, gamma1=0.02, gamma2=-0.01,
        log_lambda_reg=0.0,
    )
    m2 = namespace["build_stage_m2_likelihood"](
        model.image_data, model.noise_map, model.sim_obj.psf_kernel,
        ~np.asarray(model.unmask), m1_medians, None, prepared,
    )
    # Public reconstruction uses the fixed grid and native reference arc.
    result, diagnostics = m2.evaluate_with_diagnostics()
    require_valid_reconstruction(result, diagnostics, context="m2 build check")
    assert tuple(float(v) for v in result.source_bbox) == tuple(prepared["source_bbox"])
    assert result.source_pixels.shape == (9,)

    # Plot the M2 stage (not the original model) with its own medians and
    # compare rendered images to its public result/diagnostics.
    m2_names = [p.name for p in m2.get_dynamic_params()]
    m2_medians = {
        n: float(np.asarray(p.value))
        for n, p in zip(m2_names, m2.get_dynamic_params())
    }
    m2_result, m2_diag = m2.evaluate_with_diagnostics(
        jnp.asarray([m2_medians[n] for n in m2_names]),
        include_effective_dof=True,
    )
    require_valid_reconstruction(m2_result, m2_diag, context="m2 plot check")
    expected_scale = np.asarray(m2_diag.regularization_scale)
    assert np.unique(expected_scale).size > 1
    assert m2_diag.evidence_logdet == "overlap"

    from matplotlib.axes import Axes

    rendered = []
    original_imshow = Axes.imshow

    def capture(ax, image, *args, **kwargs):
        rendered.append(np.asarray(image).copy())
        return original_imshow(ax, image, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", capture)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "stage_m.png"
    namespace["_plot_pix_stage"](
        "stage-M", m2, m2_medians, m2_names, str(out)
    )
    assert out.exists() and out.stat().st_size > 0
    assert len(rendered) == 5
    np.testing.assert_allclose(rendered[0], np.asarray(m2.image_data), rtol=2e-5)
    np.testing.assert_allclose(
        rendered[1], np.asarray(m2_result.model_image), rtol=2e-5
    )
    m2_mask = ~np.asarray(m2.unmask, dtype=bool)
    np.testing.assert_allclose(
        rendered[2][~np.isnan(rendered[2])],
        np.asarray(m2_diag.normalized_residuals)[~m2_mask], rtol=2e-5, atol=1e-6,
    )
    np.testing.assert_allclose(
        rendered[3], np.asarray(m2_result.source_pixels).reshape(3, 3), rtol=2e-5, atol=1e-6,
    )
    np.testing.assert_allclose(
        rendered[4], np.asarray(m2_diag.regularization_scale), rtol=2e-5, atol=1e-6,
    )

    bad = _small_source_export_model(image_data=np.full((10, 10), np.nan))
    with pytest.raises(InvalidReconstructionError):
        namespace["_plot_pix_stage"](
            "stage-M", bad, medians, names, str(tmp_path / "bad.png")
        )
    assert not (tmp_path / "bad.png").exists()
