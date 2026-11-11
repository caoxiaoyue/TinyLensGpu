"""Stage transfer in the standalone Euclid mock pixelized fit."""

import importlib.util
import runpy
import sys

import numpy as np
import pytest
from astropy.io import fits

from TinyLensGpu.Inference import StagePosterior


@pytest.mark.integration
@pytest.mark.parametrize("interpolation", ["bilinear", "cubic_hermite"])
def test_pixelized_fit_reaches_m3_from_mock_fits(examples_dir, tmp_path, monkeypatch, interpolation):
    script = examples_dir / "pix_src_demo_operator/pipe/no_lens_light/pix_src_fit.py"
    spec = importlib.util.spec_from_file_location("mock_pix_src_fit", script)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    monkeypatch.setattr(example, "SOURCE_N", 8)
    monkeypatch.setattr(example, "SOURCE_DPIX", 0.08)
    monkeypatch.setattr(example, "NSUB", 1)

    data = tmp_path / "data"
    data.mkdir()
    axis = (np.arange(20) - 9.5) * 0.08
    radius = np.hypot(axis[:, None], axis[None, :])
    image = np.exp(-0.5 * ((radius - 0.31) / 0.05) ** 2)
    fits.writeto(data / "image.fits", image, fits.Header({"PIXSCALE": 0.08}))
    fits.writeto(data / "noise.fits", np.full_like(image, 0.1))
    fits.writeto(data / "psf.fits", np.ones((1, 1)))

    fixture_point = dict(
        theta_E=0.31, gamma=2.0, e1_mass=-0.05, e2_mass=0.02,
        center_x_mass=0.015, center_y_mass=-0.012, gamma1=0.04, gamma2=0.03,
        center_x_src=0.02, center_y_src=0.05, e1_src=0.0, e2_src=0.0,
        log_lambda_reg=-2.0, adaptive_reg_rho=2.0,
    )
    sampled_stages = []
    operator_inputs = []
    operator_type = example.PixelizedImageProbModelOperator

    def record_operator(*args, **kwargs):
        likelihood = operator_type(*args, **kwargs)
        operator_inputs.append((likelihood.source_model.interpolation, kwargs["adapt_image"]))
        return likelihood

    def fixed_posterior(likelihood, _config, *, tag):
        sampled_stages.append(tag)
        params = likelihood.get_dynamic_params()
        point = {param.name: fixture_point[param.name] for param in params}
        if tag == "Stage M1":
            strength = next(param for param in params if param.name == "log_lambda_reg")
            with np.load(output / "stage_m0_evidence.npz") as scan:
                assert strength.prior_settings[0] == pytest.approx(float(scan["log_lambda"]))
                point["log_lambda_reg"] = float(scan["log_lambda"])
        return StagePosterior.from_likelihood(
            likelihood, np.tile(list(point.values()), (3, 1)), np.ones(3), log_z=-12.0,
        )

    output = tmp_path / "pixelized"
    monkeypatch.setattr(example, "PixelizedImageProbModelOperator", record_operator)
    monkeypatch.setattr(example, "run_sampling", fixed_posterior)
    if interpolation == "cubic_hermite":
        entry = script.with_name("pix_src_fit_3rd_itp.py")
        monkeypatch.setitem(sys.modules, "pix_src_fit", example)
        monkeypatch.setattr(sys, "argv", [str(entry), "--data-dir", str(data), "--out-dir", str(output)])
        runpy.run_path(str(entry), run_name="__main__")
    else:
        example.main(data, output)

    assert sampled_stages == ["Stage A", "Stage M1", "Stage M2", "Stage M3"]
    assert [mode for mode, _ in operator_inputs] == [interpolation] * 4
    with np.load(output / "stage_m2_posterior.npz") as posterior:
        assert set(posterior["names"]) == {"log_lambda_reg", "adaptive_reg_rho"}
    with np.load(output / "stage_m3_posterior.npz") as posterior:
        assert set(posterior["names"]) == {"theta_E", "gamma", "e1_mass", "e2_mass", "gamma1", "gamma2"}
    with np.load(output / "stage_m1_model.npz") as m1, np.load(output / "stage_m3_model.npz") as m3:
        np.testing.assert_array_equal(m1["source_bbox"], m3["source_bbox"])
        np.testing.assert_array_equal(m1["model_image"], operator_inputs[2][1])
        np.testing.assert_array_equal(m1["model_image"], operator_inputs[3][1])
        assert m3["source_interpolation"].item() == interpolation
        assert m3["source_image"].shape == (8, 8)
        assert np.isfinite(m3["chi2"])
        assert m3["lambda_reg"] == pytest.approx(np.exp(-2.0))
        assert m3["adaptive_reg_rho"] == pytest.approx(2.0)
    assert (output / "stage_m3_model.png").stat().st_size > 0
