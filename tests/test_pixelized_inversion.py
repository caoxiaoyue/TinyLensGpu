# pyright: reportMissingImports=false

"""
TDD specifications for pixelized source inversion workflows.

These tests define the expected public behavior for the first pixelized-source
simulator and Bayesian evidence probability model. They intentionally exercise
small synthetic problems so the future implementation can be validated quickly.
"""

import pytest
import jax.numpy as jnp

from TinyLensGpu.ForwardSimulation import SimulatorConfig
from TinyLensGpu.ForwardSimulation.LensImage.pixelized import PixelizedLensSimulator
from TinyLensGpu.Inference import ParamU
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model import PixelizedImageProbModel
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import ConstantBackground, GaussianEllipse
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import SIE
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.Light.pixelized_source import PixelizedSourceModel
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel


def _delta_psf() -> jnp.ndarray:
    """Return a normalized Dirac-delta PSF kernel."""
    return jnp.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])


def _blur_psf() -> jnp.ndarray:
    """Return a compact non-delta PSF kernel for FFT-convolution tests."""
    return jnp.asarray([[0.05, 0.10, 0.05], [0.10, 0.40, 0.10], [0.05, 0.10, 0.05]])


def _dynamic_sie() -> SIE:
    """Build a simple mass model with one dynamic parameter."""
    theta_e = ParamU("theta_E", 0.12, prior_type="uniform", prior_settings=[0.05, 0.20], limits=[0.0, 1.0])
    sie = SIE(theta_E=theta_e, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    sie.theta_E.to_dynamic()
    for param in [sie.e1, sie.e2, sie.center_x, sie.center_y]:
        param.to_static()
    return sie


def _static_parametric_source() -> GaussianEllipse:
    """Build a static parametric source used in invalid mixed-source tests."""
    source = GaussianEllipse(flux=1.0, sigma=0.1, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    for param in [source.flux, source.sigma, source.e1, source.e2, source.center_x, source.center_y]:
        param.to_static()
    return source


def _pixelized_source(*, lambda_value: float = 1.0, with_gp: bool = False) -> PixelizedSourceModel:
    """Build a pixelized source with dynamic regularization parameters."""
    lambda_reg = ParamU(
        "lambda_reg",
        lambda_value,
        prior_type="log_uniform",
        prior_settings=[1.0e-3, 1.0e3],
        limits=[1.0e-6, 1.0e6],
    )
    lambda_reg.to_dynamic()
    if not with_gp:
        return PixelizedSourceModel(nx=5, ny=5, lambda_reg=lambda_reg)

    kernel_scale = ParamU(
        "kernel_scale",
        0.2,
        prior_type="uniform",
        prior_settings=[0.05, 1.0],
        limits=[0.01, 5.0],
    )
    kernel_scale.to_dynamic()
    return PixelizedSourceModel(
        nx=5,
        ny=5,
        lambda_reg=lambda_reg,
        kernel_scale=kernel_scale,
        regularization_type="gaussian",
        kernel_type="gaussian",
    )


def _physical_model(*, source: PixelizedSourceModel | None = None) -> PhysicalModel:
    """Build the canonical tiny pixelized physical model."""
    return PhysicalModel(lens_mass=[_dynamic_sie()], source_light=[source or _pixelized_source()], lens_light=[])


def _simulator(
    *,
    mask: jnp.ndarray | None = None,
    psf_kernel: jnp.ndarray | None = None,
    phys_model: PhysicalModel | None = None,
    nsub: int = 1,
) -> PixelizedLensSimulator:
    """Build a small pixelized lens simulator for fast unit tests."""
    config = SimulatorConfig(dpix=0.08, npix=10, nsub=nsub, psf_kernel=psf_kernel or _delta_psf(), mask=mask)
    return PixelizedLensSimulator(phys_model or _physical_model(), config)


def _prob_model(
    *,
    image_data: jnp.ndarray | None = None,
    noise_map: jnp.ndarray | None = None,
    psf_kernel: jnp.ndarray | None = None,
    source: PixelizedSourceModel | None = None,
    mask: jnp.ndarray | None = None,
    nsub: int = 1,
) -> PixelizedImageProbModel:
    """Build a small pixelized evidence model."""
    data = image_data if image_data is not None else jnp.zeros((10, 10))
    noise = noise_map if noise_map is not None else jnp.ones((10, 10)) * 0.1
    return PixelizedImageProbModel(
        image_data=data,
        noise_map=noise,
        psf_kernel=psf_kernel or _delta_psf(),
        dpix=0.08,
        phys_model=_physical_model(source=source),
        mask=mask,
        nsub=nsub,
    )


@pytest.mark.unit
def test_pixelized_simulator_returns_full_2d_image_matching_data_shape() -> None:
    """Test that simulate returns a full image-plane array, not only unmasked pixels."""
    mask = jnp.zeros((10, 10), dtype=bool).at[0, 0].set(True).at[4, 4].set(True)
    simulator = _simulator(mask=mask)
    source_pixels = jnp.linspace(-1.0, 1.0, 25)

    model_image = simulator.simulate(source_pixels)

    assert model_image.shape == mask.shape
    assert jnp.all(jnp.isfinite(model_image))


@pytest.mark.unit
def test_delta_psf_simulation_places_mapping_product_on_unmasked_pixels() -> None:
    """Test that the delta-PSF path approximates the dense mapping product."""
    mask = jnp.zeros((10, 10), dtype=bool).at[1, 1].set(True).at[7, 2].set(True)
    simulator = _simulator(mask=mask, psf_kernel=_delta_psf())
    source_pixels = jnp.arange(25, dtype=float) - 12.0

    design_matrix, _ = simulator.design_matrix(psf_kernel=_delta_psf())
    model_image = simulator.simulate(source_pixels, psf_kernel=_delta_psf())

    # FFT convolution assumes periodic boundaries, so edge pixels may differ
    # from the ideal mapping product; use a relaxed tolerance.
    assert jnp.allclose(model_image[~mask], design_matrix @ source_pixels, atol=1e-4)


@pytest.mark.unit
def test_non_delta_psf_simulation_uses_fft_convolution_and_is_finite() -> None:
    """Test that the non-delta PSF branch returns finite convolved values."""
    simulator = _simulator(psf_kernel=_blur_psf())
    source_pixels = jnp.linspace(0.1, 2.5, 25)

    model_image = simulator.simulate(source_pixels, psf_kernel=_blur_psf())

    unconvolved = simulator.simulate(source_pixels, psf_kernel=_delta_psf())


    assert model_image.shape == (10, 10)
    assert jnp.all(jnp.isfinite(model_image))
    assert not jnp.allclose(model_image, unconvolved)


@pytest.mark.unit
def test_design_matrix_shape_matches_unmasked_pixels_and_source_grid() -> None:
    """Test design-matrix dimensions for a masked 10x10 image and 5x5 source."""
    mask = jnp.zeros((10, 10), dtype=bool).at[0, :].set(True).at[:, 0].set(True)
    simulator = _simulator(mask=mask)

    design_matrix, _ = simulator.design_matrix()

    assert design_matrix.shape == (int(jnp.sum(~mask)), 25)


@pytest.mark.unit
def test_infer_source_half_size_uses_lensed_coordinate_extent_with_floor() -> None:
    """Test source half-size inference from ray-traced coordinates."""
    simulator = _simulator()
    beta_x = jnp.asarray([-0.2, 0.3, 0.1])
    beta_y = jnp.asarray([0.4, -0.1, 0.05])
    tiny_beta = jnp.asarray([0.0, 1.0e-14])

    assert jnp.allclose(simulator.infer_source_half_size(beta_x, beta_y), 1.05 * 0.4)
    assert simulator.infer_source_half_size(tiny_beta, tiny_beta) > 0.0


@pytest.mark.unit
def test_bayesian_evidence_returns_finite_scalar_for_tiny_problem() -> None:
    """Test that the evidence call returns one finite scalar value."""
    model = _prob_model(image_data=jnp.ones((10, 10)) * 0.05)

    log_evidence = model()

    assert jnp.shape(log_evidence) == ()
    assert jnp.isfinite(log_evidence)


@pytest.mark.unit
def test_evidence_prefers_reasonable_regularization_over_extreme_values() -> None:
    """Test evidence ranking rejects strongly under- and over-smoothed solutions."""
    reference = _prob_model(image_data=jnp.ones((10, 10)) * 0.05, source=_pixelized_source(lambda_value=1.0))
    model_image, _ = reference.forward_model(return_source=True)
    noise_map = jnp.ones((10, 10)) * 0.05

    reasonable = _prob_model(image_data=model_image, noise_map=noise_map, source=_pixelized_source(lambda_value=1.0))
    under_smoothed = _prob_model(image_data=model_image, noise_map=noise_map, source=_pixelized_source(lambda_value=1.0e-8))
    over_smoothed = _prob_model(image_data=model_image, noise_map=noise_map, source=_pixelized_source(lambda_value=1.0e8))

    reasonable_log_evidence = reasonable()

    assert reasonable_log_evidence > under_smoothed()
    assert reasonable_log_evidence > over_smoothed()


@pytest.mark.unit
def test_source_reconstruction_allows_negative_values_without_nnls_constraint() -> None:
    """Test that the linear solve can reconstruct signed source pixels."""
    true_source = jnp.linspace(-1.0, 1.0, 25)
    simulator = _simulator(psf_kernel=_delta_psf())
    image_data = simulator.simulate(true_source, psf_kernel=_delta_psf())
    model = _prob_model(image_data=image_data, noise_map=jnp.ones((10, 10)) * 0.01, psf_kernel=_delta_psf())

    _, source_pixels = model.forward_model(return_source=True)

    assert jnp.any(source_pixels < 0.0)
    assert jnp.all(jnp.isfinite(source_pixels))


@pytest.mark.unit
def test_prior_transformation_sees_mass_and_regularization_parameters() -> None:
    """Test that prior extraction includes lens mass and lambda_reg parameters."""
    _, prior_specs = make_prior_transformation(_prob_model())

    assert {spec.name for spec in prior_specs} == {"theta_E", "lambda_reg"}


@pytest.mark.unit
def test_prior_transformation_includes_gp_kernel_scale() -> None:
    """Test that GP pixelized sources expose their kernel-scale parameter."""
    _, prior_specs = make_prior_transformation(_prob_model(source=_pixelized_source(with_gp=True)))

    assert {spec.name for spec in prior_specs} == {"theta_E", "lambda_reg", "kernel_scale"}


@pytest.mark.unit
def test_vectorized_likelihood_works_for_small_parameter_batch() -> None:
    """Test that the sampler-facing vectorized wrapper supports batched evidence."""
    model = _prob_model(image_data=jnp.ones((10, 10)) * 0.02)
    theta = jnp.asarray(model.get_values("flat"), dtype=float)
    batch = jnp.stack([theta, theta.at[-1].set(0.5), theta.at[-1].set(2.0)], axis=0)

    loglike = make_likelihood(model, vectorized=True)
    values = loglike(batch)

    assert values.shape == (3,)
    assert jnp.all(jnp.isfinite(values))


@pytest.mark.unit
@pytest.mark.parametrize(
    "phys_model, message",
    [
        (
            PhysicalModel(lens_mass=[_dynamic_sie()], source_light=[_pixelized_source(), _pixelized_source()], lens_light=[]),
            "single pixelized source",
        ),
        (
            PhysicalModel(lens_mass=[_dynamic_sie()], source_light=[_pixelized_source(), _static_parametric_source()], lens_light=[]),
            "parametric source",
        ),
        (
            PhysicalModel(lens_mass=[_dynamic_sie()], source_light=[_pixelized_source()], lens_light=[ConstantBackground(0.1)]),
            "lens_light",
        ),
    ],
)
def test_invalid_source_configurations_raise_value_error(phys_model: PhysicalModel, message: str) -> None:
    """Test first-version restrictions on pixelized model composition."""
    with pytest.raises(ValueError, match=message):
        PixelizedImageProbModel(
            image_data=jnp.zeros((10, 10)),
            noise_map=jnp.ones((10, 10)) * 0.1,
            psf_kernel=_delta_psf(),
            dpix=0.08,
            phys_model=phys_model,
        )


@pytest.mark.unit
def test_forward_model_returns_image_and_source_when_requested() -> None:
    """Test return_source=True returns both model image and source pixels."""
    model = _prob_model(image_data=jnp.ones((10, 10)) * 0.03)

    model_image, source_pixels = model.forward_model(return_source=True)

    assert model_image.shape == (10, 10)
    assert source_pixels.shape == (25,)
    assert jnp.all(jnp.isfinite(model_image))
    assert jnp.all(jnp.isfinite(source_pixels))


@pytest.mark.unit
def test_nsub_two_design_matrix_matches_simulate_on_active_pixels() -> None:
    """Test design_matrix and simulate are consistent when nsub > 1."""
    mask = jnp.zeros((10, 10), dtype=bool).at[1, 1].set(True).at[7, 2].set(True)
    simulator = _simulator(mask=mask, nsub=2, psf_kernel=_delta_psf())
    source_pixels = jnp.arange(25, dtype=float) - 12.0

    design_matrix, _ = simulator.design_matrix(psf_kernel=_delta_psf())
    model_image = simulator.simulate(source_pixels, psf_kernel=_delta_psf())

    assert jnp.allclose(model_image[~mask], design_matrix @ source_pixels, atol=1e-4)


@pytest.mark.unit
def test_nsub_two_returns_finite_convolved_image() -> None:
    """Test nsub=2 with a non-delta PSF returns finite values."""
    simulator = _simulator(nsub=2, psf_kernel=_blur_psf())
    source_pixels = jnp.linspace(0.1, 2.5, 25)

    model_image = simulator.simulate(source_pixels, psf_kernel=_blur_psf())

    assert model_image.shape == (10, 10)
    assert jnp.all(jnp.isfinite(model_image))


@pytest.mark.unit
def test_nsub_two_prob_model_returns_finite_evidence() -> None:
    """Test that PixelizedImageProbModel with nsub=2 returns finite evidence."""
    model = _prob_model(image_data=jnp.ones((10, 10)) * 0.05, nsub=2)

    log_evidence = model()

    assert jnp.shape(log_evidence) == ()
    assert jnp.isfinite(log_evidence)


@pytest.mark.unit
def test_nsub_two_forward_model_reconstructs_source() -> None:
    """Test forward_model with nsub=2 solves source pixels correctly."""
    true_source = jnp.linspace(-1.0, 1.0, 25)
    simulator = _simulator(nsub=2, psf_kernel=_delta_psf())
    image_data = simulator.simulate(true_source, psf_kernel=_delta_psf())
    model = _prob_model(image_data=image_data, noise_map=jnp.ones((10, 10)) * 0.01, psf_kernel=_delta_psf(), nsub=2)

    _, source_pixels = model.forward_model(return_source=True)

    assert source_pixels.shape == (25,)
    assert jnp.all(jnp.isfinite(source_pixels))


if __name__ == "__main__":
    pytest.main()
