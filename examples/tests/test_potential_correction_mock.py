"""Observable mock/truth contracts for source–potential reconstruction examples."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.unit
def test_pseudo_jaffe_truth_matches_mass_and_independent_radial_relations():
    from examples.potential_correction.mock import SubhaloTruth
    from TinyLensGpu.PhysicalModel import PseudoJaffeSubhalo

    with jax.experimental.enable_x64():
        truth = SubhaloTruth(0.025, 1.2, 0.1, -0.2)
        x = jnp.asarray([0.4, -0.3, 1.0], dtype=jnp.float64)
        y = jnp.asarray([0.2, 0.5, -0.4], dtype=jnp.float64)
        mass = PseudoJaffeSubhalo(b_sub=0.025, theta_E_host=1.2,
                                  center_x=0.1, center_y=-0.2)
        expected = mass.deriv(x, y)
        np.testing.assert_allclose(truth.deflection(x, y), expected, rtol=2e-13)
        dx, dy = np.asarray(x) - 0.1, np.asarray(y) + 0.2
        radius = np.hypot(dx, dy)
        rt = np.sqrt(0.025 * 1.2)
        convergence = 0.025 / 2 * (1 / radius - 1 / np.hypot(radius, rt))
        np.testing.assert_allclose(truth.convergence(x, y), convergence, rtol=2e-13)
        h = 1e-5
        potential_dx = (truth.potential(x + h, y) - truth.potential(x - h, y)) / (2*h)
        potential_dy = (truth.potential(x, y + h) - truth.potential(x, y - h)) / (2*h)
        np.testing.assert_allclose([potential_dx, potential_dy], expected, rtol=2e-9)


@pytest.mark.boundary
def test_truth_exposes_singular_domain_without_a_cored_replacement():
    from examples.potential_correction.mock import SubhaloTruth
    from TinyLensGpu.PhysicalModel import PseudoJaffeSubhalo

    with jax.experimental.enable_x64():
        x = jnp.asarray([0.0, 1e-10, 1e-8, 1e-6])
        y = jnp.zeros_like(x)
        truth = SubhaloTruth(0.04, 1.44, 0.0, 0.0)
        mass = PseudoJaffeSubhalo(b_sub=0.04, theta_E_host=1.44,
                                  center_x=0.0, center_y=0.0)
        np.testing.assert_allclose(truth.deflection(x, y), mass.deriv(x, y), rtol=1e-14)
        kappa = truth.convergence(x, y)
        assert np.isnan(kappa[:2]).all()
        assert np.isfinite(kappa[2:]).all()
        assert truth.potential(x, y)[0] == 0
        zero = SubhaloTruth(0.0, 1.44, 0.0, 0.0)
        np.testing.assert_array_equal(zero.potential(x, y), np.zeros(4))
        np.testing.assert_array_equal(zero.convergence(x, y), np.zeros(4))
        np.testing.assert_array_equal(zero.deflection(x, y), np.zeros((2, 4)))


@pytest.mark.unit
def test_source_truth_brightness_and_analytic_gradient_support_requested_nodes():
    from examples.potential_correction.mock import SourceTruth

    with jax.experimental.enable_x64():
        source = SourceTruth()
        x = jnp.asarray([0.0, 0.12, -0.2])
        y = jnp.asarray([0.02, -0.1, 0.08])
        gx, gy = source.gradient(x, y)
        h = 1e-6
        numerical = [(source.brightness(x+h, y)-source.brightness(x-h, y))/(2*h),
                     (source.brightness(x, y+h)-source.brightness(x, y-h))/(2*h)]
        np.testing.assert_allclose([gx, gy], numerical, rtol=2e-9, atol=1e-9)
        assert source.brightness(x, y).dtype == jnp.float64
        assert np.isfinite(source.brightness(x, y)).all()


@pytest.mark.integration
def test_mock_pair_reproduces_shared_noise_and_fixed_physical_observation():
    from examples.potential_correction.mock import MockConfig, generate_mock_pair

    config = MockConfig(npix=16, dpix=0.18, nsub=2, psf_size=3, seed=91)
    before = jax.config.x64_enabled
    pair = generate_mock_pair(config)
    again = generate_mock_pair(config)
    assert jax.config.x64_enabled == before
    for observation in (pair.subhalo, pair.control):
        assert observation.image_data.shape == (16, 16)
        assert observation.image_data.dtype == jnp.float64
        assert np.isfinite(observation.image_data).all()
        np.testing.assert_array_equal(observation.mask, observation.sim_config.mask)
        assert not observation.macro.dynamic_params
        assert not observation.macro.lens_light
        np.testing.assert_allclose(observation.psf_kernel.sum(), 1.0, atol=1e-14)
    np.testing.assert_array_equal(pair.subhalo.image_data, again.subhalo.image_data)
    np.testing.assert_array_equal(pair.subhalo.mask, pair.control.mask)
    np.testing.assert_array_equal(pair.subhalo.noise_realization, pair.control.noise_realization)
    np.testing.assert_allclose(np.asarray(pair.subhalo.image_data) - np.asarray(pair.control.image_data),
                               np.asarray(pair.subhalo.model_image) - np.asarray(pair.control.model_image),
                               atol=2e-16)
    assert np.linalg.norm(np.asarray(pair.subhalo.model_image) - np.asarray(pair.control.model_image)) > 0
    assert pair.control.subhalo_truth.b_sub == 0


@pytest.mark.integration
def test_mock_generator_saves_reconstructible_truth_and_provenance(tmp_path):
    import json

    from examples.potential_correction.generate_mock import main
    from examples.potential_correction.mock import MockConfig, generate_mock_pair

    main(["--npix", "12", "--dpix", "0.25", "--nsub", "1", "--psf-size", "3",
          "--seed", "7", "--output", str(tmp_path)])
    metadata = json.loads((tmp_path / "configuration.json").read_text())
    assert metadata["units"]["potential"] == "arcsec^2"
    assert metadata["truth_conventions"]["convergence"] == "point samples; NaN for R < 1e-8 arcsec"
    assert metadata["noise_convention"] == "same independent Gaussian realization in both observations"
    config = MockConfig.from_dict(metadata["configuration"])
    regenerated = generate_mock_pair(config)
    with np.load(tmp_path / "subhalo.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["image_data"], regenerated.subhalo.image_data)
        np.testing.assert_array_equal(saved["mask"], regenerated.subhalo.mask)
        assert saved["true_potential"].shape == (12, 12)
        assert saved["true_convergence_valid"].dtype == bool
    with np.load(tmp_path / "control.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["true_potential"], np.zeros((12, 12)))
        np.testing.assert_array_equal(saved["noise_realization"], regenerated.control.noise_realization)


@pytest.mark.integration
def test_zero_strength_float32_mock_and_delta_psf_keep_mask_convention():
    from examples.potential_correction.mock import MockConfig, generate_mock_pair

    config = MockConfig(npix=12, dpix=0.25, nsub=1, psf_size=1, psf_sigma=0,
                        b_sub=0, dtype="float32")
    pair = generate_mock_pair(config)
    np.testing.assert_array_equal(pair.subhalo.image_data, pair.control.image_data)
    observation = pair.control
    assert observation.image_data.dtype == jnp.float32
    beta_x, beta_y = observation.macro.deflection(observation.sim_config.xgrid,
                                                 observation.sim_config.ygrid)
    direct_brightness = config.source.brightness(beta_x, beta_y)
    expected = np.where(observation.mask, 0, direct_brightness)
    np.testing.assert_allclose(observation.model_image, expected, atol=1e-7)


@pytest.mark.boundary
@pytest.mark.parametrize("kwargs", [{"b_sub": -0.1}, {"noise_sigma": 0},
                                    {"psf_size": 4}, {"dtype": "float16"},
                                    {"nsub": 0}, {"mask_inner": 1.5}])
def test_mock_rejects_malformed_configuration(kwargs):
    from examples.potential_correction.mock import MockConfig

    with pytest.raises(ValueError):
        MockConfig(**kwargs)
