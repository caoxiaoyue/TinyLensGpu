"""Regression checks for the B1938 direct-M1 pixelized pipeline."""
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from TinyLensGpu.Inference import StagePosterior
from TinyLensGpu.Inference.build_prior import make_prior_transformation


SCRIPT = (Path(__file__).resolve().parents[1] / "examples" /
          "pix_src_demo_operator/pipe/subhalo_B1938/model.py")


@pytest.fixture
def model(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("b1938_pipeline_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "NSRC", 20)
    monkeypatch.setattr(module, "NSUB_PIX", 1)
    monkeypatch.setattr(module, "OUT_DIR", tmp_path / "output")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(module, "DATA_DIR", data_dir)
    fits.writeto(data_dir / "B1938_666_F115W_mask.fits", np.zeros((32, 32)))
    return module


def posterior(values):
    names = list(values)
    center = np.array(list(values.values()))
    samples = center + np.array([-0.01, 0, 0.01])[:, None]
    return StagePosterior.from_schema(samples, np.ones(3) / 3, param_names=names)


@pytest.fixture
def stages():
    mass = dict(theta_E=0.48, e1_mass=0.02, e2_mass=0.01,
                center_x_mass=0.0, center_y_mass=0.0, gamma1=0.01, gamma2=0.0)
    return (
        posterior({**mass, "center_x_src": 0.02, "center_y_src": -0.01}),
        posterior({**mass, "log_lambda_reg": 0.0}),
        posterior(dict(log_lambda_reg=0.0, adaptive_reg_rho=2.0)),
        posterior({**mass, "gamma": 2.0}),
    )


def inputs():
    return dict(lens_subtracted_image=np.full((32, 32), 0.1),
                noise_map=np.full((32, 32), 0.05),
                psf_kernel=np.ones((1, 1)), feature_mask=np.zeros((32, 32), bool),
                position_likelihood=None)


def source_package(model, stage_a):
    bbox = model._stage_a_source_bbox(stage_a)
    x, y = model._source_axes_from_bbox(bbox, model.NSRC)
    return dict(n=model.NSRC, source_bbox=bbox,
                source_pixels=np.ones(model.NSRC**2), source_x_axis=x,
                source_y_axis=y, lambda_best=1.0, log_lambda_best=0.0)


def test_m1_prior_is_uniform_in_log_lambda(model, stages):
    lkl = model.build_stage_m1_likelihood(**inputs(), stage_a=stages[0])
    prior, specs = make_prior_transformation(lkl)
    index = next(i for i, spec in enumerate(specs) if spec.name == "log_lambda_reg")
    assert len(specs) == 8  # seven mass/shear parameters plus ln(lambda)
    for quantile, expected in ((0, 1e-6), (0.5, 1.0), (1, 1e6)):
        unit = np.full(len(specs), 0.5)
        unit[index] = quantile
        assert np.exp(np.asarray(prior(unit))[index]) == pytest.approx(expected, rel=2e-6)


def test_all_pixelized_builders_share_grid_and_pcg_budget(model, stages):
    a, m1, m2, m3 = stages
    package = source_package(model, a)
    likelihoods = [
        model.build_stage_m1_likelihood(**inputs(), stage_a=a),
        model.build_stage_m2_likelihood(**inputs(), medians_m1=m1.medians(), s1_package=package),
        model.build_stage_m3_likelihood(**inputs(), stage_m1=m1,
                                       reg_hyperparams_fixed=m2.medians(), s1_package=package),
        *[model.build_comparison_likelihood(**inputs(), stage_m2=m2, stage_m3=m3,
                                           s1_package=package, with_subhalo=subhalo)
          for subhalo in (False, True)],
    ]
    for lkl in likelihoods:
        assert lkl.pcg_max_iter == 800
        assert lkl.sim_obj.n_source_pixels == model.NSRC**2
        np.testing.assert_allclose(lkl._fixed_source_bbox, package["source_bbox"], atol=1e-7)
        xmin, xmax, ymin, ymax = lkl._fixed_source_bbox
        assert xmax - xmin == pytest.approx(0.4)
        assert ymax - ymin == pytest.approx(0.4)
        assert (xmin + xmax) / 2 == pytest.approx(a.median("center_x_src"), abs=1e-7)
        assert (ymin + ymax) / 2 == pytest.approx(a.median("center_y_src"), abs=1e-7)
    assert not hasattr(model, "run_stage_m0")
    assert not hasattr(model, "build_stage_m0_likelihood")


def test_failed_source_solve_does_not_export_a_zero_plot(model, stages):
    lkl = model.build_stage_m1_likelihood(**inputs(), stage_a=stages[0])
    prior, specs = make_prior_transformation(lkl)
    values = np.asarray(prior(np.full(len(specs), 0.5)))
    medians = dict(zip((spec.name for spec in specs), values))
    lkl.pcg_max_iter = 0
    destination = model.OUT_DIR.parent / "failed_plot.png"
    with pytest.raises(RuntimeError, match="plotting stage-M1"):
        model._plot_pix_stage("stage-M1", lkl, medians, list(medians), str(destination))
    assert not destination.exists()


def test_main_runs_m1_directly_after_stage_b(model, stages, monkeypatch):
    calls = []
    a = stages[0]
    data = inputs()
    monkeypatch.setattr(model, "_load_b1938_data", lambda: (
        data["lens_subtracted_image"], data["noise_map"], data["psf_kernel"], data["feature_mask"]))
    def stage_a(*args, **kwargs):
        calls.append("a")
        return a, a.medians(), np.zeros((32, 32))
    def stage_b(*args, **kwargs):
        calls.append("b")
        return data["feature_mask"]
    original_signature = inspect.signature(model.run_stage_m1)
    class ReachedM1(Exception):
        pass
    def stage_m1(*args, **kwargs):
        original_signature.bind(*args, **kwargs)
        calls.append("m1")
        raise ReachedM1
    monkeypatch.setattr(model, "run_stage_a", stage_a)
    monkeypatch.setattr(model, "run_stage_b", stage_b)
    monkeypatch.setattr(model, "run_stage_m1", stage_m1)
    monkeypatch.setattr(model, "_load_stage", lambda tag: {"extra": {"time_taken": 0.0}})
    monkeypatch.setattr(model, "_position_likelihood_from_stage_a", lambda medians: None)
    with pytest.raises(ReachedM1):
        model.main()
    assert calls == ["a", "b", "m1"]
