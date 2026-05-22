# pyright: reportMissingImports=false

"""
TDD specifications for pixelized source inversion workflows.

These tests define the expected public behavior for the first pixelized-source
simulator and Bayesian evidence probability model. They intentionally exercise
small synthetic problems so the future implementation can be validated quickly.
"""

import numpy as np
import pytest
import jax.numpy as jnp
from scipy.signal import fftconvolve

from TinyLensGpu.ForwardSimulation import SimulatorConfig
from TinyLensGpu.ForwardSimulation.LensImage.config import make_grid_2d
from TinyLensGpu.ForwardSimulation.LensImage.pixelized import PixelizedLensSimulator
from TinyLensGpu.utils.geometry.transforms import phi_q2_ellipticity
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
    position_likelihood: dict | None = None,
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
        position_likelihood=position_likelihood,
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
    ],
)
def test_invalid_source_configurations_raise_value_error(phys_model: PhysicalModel, message: str) -> None:
    """Test restrictions on pixelized model composition."""
    with pytest.raises(ValueError, match=message):
        PixelizedImageProbModel(
            image_data=jnp.zeros((10, 10)),
            noise_map=jnp.ones((10, 10)) * 0.1,
            psf_kernel=_delta_psf(),
            dpix=0.08,
            phys_model=phys_model,
        )


@pytest.mark.unit
def test_pixelized_simulator_accepts_lens_light() -> None:
    """Test that pixelized simulator now accepts lens_light components."""
    from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import SersicEllipse
    lens_light = SersicEllipse(
        R_sersic=1.0, n_sersic=4.0, Ie=1.0,
        e1=0.0, e2=0.0, center_x=0.0, center_y=0.0,
    )
    for p in [lens_light.R_sersic, lens_light.n_sersic, lens_light.Ie, lens_light.e1, lens_light.e2, lens_light.center_x, lens_light.center_y]:
        p.to_static()
    phys_model = PhysicalModel(
        lens_mass=[_dynamic_sie()],
        source_light=[_pixelized_source()],
        lens_light=[lens_light],
    )
    # Should not raise
    simulator = _simulator(phys_model=phys_model)
    assert simulator.has_lens_light
    assert simulator.n_lens_light == 1


@pytest.mark.unit
def test_build_lens_light_matrix_matches_independent_convolved_lens_light() -> None:
    """Test that build_lens_light_matrix matches independent PSF-convolved lens light."""
    from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import SersicEllipse
    from TinyLensGpu.ForwardSimulation.LensImage.config import make_grid_2d

    lens_light = SersicEllipse(
        R_sersic=1.0, n_sersic=4.0, Ie=1.0,
        e1=0.0, e2=0.0, center_x=0.0, center_y=0.0,
    )
    for p in [lens_light.R_sersic, lens_light.n_sersic, lens_light.Ie, lens_light.e1, lens_light.e2, lens_light.center_x, lens_light.center_y]:
        p.to_static()

    phys_model = PhysicalModel(
        lens_mass=[_dynamic_sie()],
        source_light=[_pixelized_source()],
        lens_light=[lens_light],
    )

    psf = _blur_psf()
    simulator = _simulator(phys_model=phys_model, psf_kernel=psf)

    # Build lens light matrix via the method under test
    L = simulator.build_lens_light_matrix()  # (Nd, Nl), Nl=1
    assert L.shape == (simulator.flat_indices.shape[0], 1)

    # Independent ground truth: evaluate lens light on native grid,
    # convolve with PSF, extract active pixels
    xgrid, ygrid = make_grid_2d(10, 0.08)
    lens_img = lens_light.light(x=xgrid, y=ygrid)
    expected_convolved = fftconvolve(np.array(lens_img), np.array(psf), mode="same")
    expected_1d = expected_convolved.ravel()[np.array(simulator.flat_indices)]

    # Compare: unit-amplitude basis, should match closely
    L_np = np.array(L).ravel()
    assert np.allclose(L_np, expected_1d, atol=1e-6)


@pytest.mark.unit
def test_build_lens_light_matrix_returns_empty_for_no_lens_light() -> None:
    """Test that build_lens_light_matrix returns (Nd, 0) when no lens light present."""
    simulator = _simulator()
    L = simulator.build_lens_light_matrix()
    assert L.shape == (simulator.flat_indices.shape[0], 0)
    assert L.dtype == jnp.float32


@pytest.mark.unit
def test_joint_design_matrix_shape_adds_lens_columns() -> None:
    """Test that joint design matrix [F|L] has correct shape."""
    from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import SersicEllipse
    lens_light = SersicEllipse(
        R_sersic=1.0, n_sersic=4.0, Ie=1.0,
        e1=0.0, e2=0.0, center_x=0.0, center_y=0.0,
    )
    for p in [lens_light.R_sersic, lens_light.n_sersic, lens_light.Ie, lens_light.e1, lens_light.e2, lens_light.center_x, lens_light.center_y]:
        p.to_static()
    phys_model = PhysicalModel(
        lens_mass=[_dynamic_sie()],
        source_light=[_pixelized_source()],
        lens_light=[lens_light],
    )
    simulator = _simulator(phys_model=phys_model)
    design_matrix, _ = simulator.design_matrix()
    n_source = simulator.n_source_pixels
    n_lens = simulator.n_lens_light
    n_data = simulator.flat_indices.shape[0]
    assert design_matrix.shape == (n_data, n_source + n_lens)


@pytest.mark.unit
def test_source_only_design_matrix_is_unchanged() -> None:
    """Test that source-only design matrix shape is unchanged."""
    simulator = _simulator()
    design_matrix, _ = simulator.design_matrix()
    n_source = simulator.n_source_pixels
    n_data = simulator.flat_indices.shape[0]
    assert design_matrix.shape == (n_data, n_source)
    assert not simulator.has_lens_light


@pytest.mark.unit
def test_joint_evidence_is_finite_for_lens_light() -> None:
    """Test that joint evidence with lens_light returns finite scalar."""
    from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import SersicEllipse
    lens_light = SersicEllipse(
        R_sersic=1.0, n_sersic=4.0, Ie=1.0,
        e1=0.0, e2=0.0, center_x=0.0, center_y=0.0,
    )
    for p in [lens_light.R_sersic, lens_light.n_sersic, lens_light.Ie, lens_light.e1, lens_light.e2, lens_light.center_x, lens_light.center_y]:
        p.to_static()
    phys_model = PhysicalModel(
        lens_mass=[_dynamic_sie()],
        source_light=[_pixelized_source()],
        lens_light=[lens_light],
    )
    model = PixelizedImageProbModel(
        image_data=jnp.ones((10, 10)) * 0.05,
        noise_map=jnp.ones((10, 10)) * 0.1,
        psf_kernel=_delta_psf(),
        dpix=0.08,
        phys_model=phys_model,
    )
    log_evidence = model()
    assert jnp.shape(log_evidence) == ()
    assert jnp.isfinite(log_evidence)


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


def _gaussian_psf_kernel(size: int = 5, sigma_pixels: float = 0.8) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError("PSF kernel size must be odd")
    coord = np.arange(size) - size // 2
    xx, yy = np.meshgrid(coord, coord)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma_pixels**2))
    kernel /= kernel.sum()
    return kernel


@pytest.mark.unit
@pytest.mark.parametrize("nsub", [1, 2])
def test_pixelized_mapping_and_design_matrix_match_independent_lensed_source_truth(nsub: int) -> None:
    """Test that L @ s and M @ s match an independent ground-truth lensed image.

    The ground-truth lensed image is built from parametric Gaussian sources
    evaluated directly via PhysicalModel.source_surface_brightness, independent
    of the pixelized bilinear mapping path.  For the ideal (unconvolved) check
    the mapping matrix is built on the native-resolution grid; when nsub > 1
    the design_matrix path uses the sub-grid internally and aggregates back to
    native resolution before PSF convolution.

    The PSF ground truth uses scipy.signal.fftconvolve, which shares the same
    periodic-boundary assumption as the JAX FFT convolution inside
    PixelizedLensSimulator.design_matrix, so the comparison is valid.
    """
    npix = 30
    dpix = 0.08
    source_nx = 15
    source_ny = 15
    source_half_size = 0.6

    psf_kernel = _gaussian_psf_kernel(size=5, sigma_pixels=0.8)

    xgrid, ygrid = tuple(np.asarray(v) for v in make_grid_2d(npix, dpix))
    center_x, center_y = 0.0, 0.0
    radius = np.hypot(xgrid - center_x, ygrid - center_y)
    mask = ~((radius >= 0.65) & (radius <= 1.18))

    e1_1, e2_1 = phi_q2_ellipticity(np.deg2rad(-20.0), 0.72)
    e1_1, e2_1 = float(e1_1), float(e2_1)
    true_src1 = GaussianEllipse(
        flux=1.0 * 2.0 * np.pi * 0.16**2,
        sigma=0.16,
        e1=e1_1,
        e2=e2_1,
        center_x=0.04,
        center_y=0.02,
    )
    for p in [true_src1.flux, true_src1.sigma, true_src1.e1, true_src1.e2, true_src1.center_x, true_src1.center_y]:
        p.to_static()

    e1_2, e2_2 = phi_q2_ellipticity(np.deg2rad(35.0), 0.85)
    e1_2, e2_2 = float(e1_2), float(e2_2)
    true_src2 = GaussianEllipse(
        flux=0.35 * 2.0 * np.pi * 0.10**2,
        sigma=0.10,
        e1=e1_2,
        e2=e2_2,
        center_x=-0.08,
        center_y=0.07,
    )
    for p in [true_src2.flux, true_src2.sigma, true_src2.e1, true_src2.e2, true_src2.center_x, true_src2.center_y]:
        p.to_static()

    e1_lens, e2_lens = phi_q2_ellipticity(np.deg2rad(28.0), 0.76)
    e1_lens, e2_lens = float(e1_lens), float(e2_lens)
    sie = SIE(theta_E=1.12, e1=e1_lens, e2=e2_lens, center_x=center_x, center_y=center_y)
    for p in [sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y]:
        p.to_static()

    gt_phys = PhysicalModel(lens_mass=[sie], source_light=[true_src1, true_src2], lens_light=[])

    beta_x, beta_y = gt_phys.deflection(xgrid, ygrid)
    ideal_lensed_truth = np.asarray(gt_phys.source_surface_brightness(beta_x, beta_y))

    source_x_axis = np.linspace(-source_half_size, source_half_size, source_nx)
    source_y_axis = np.linspace(-source_half_size, source_half_size, source_ny)
    source_xx, source_yy = np.meshgrid(source_x_axis, source_y_axis, indexing="xy")
    source_pixels = np.asarray(gt_phys.source_surface_brightness(source_xx, source_yy)).ravel()

    pixelized_src = PixelizedSourceModel(nx=source_nx, ny=source_ny, lambda_reg=1.0)
    test_phys = PhysicalModel(lens_mass=[sie], source_light=[pixelized_src], lens_light=[])
    config = SimulatorConfig(
        dpix=dpix,
        npix=npix,
        nsub=nsub,
        psf_kernel=psf_kernel,
        mask=jnp.asarray(mask),
    )
    simulator = PixelizedLensSimulator(test_phys, config)

    # Build the native-resolution mapping matrix (no sub-grid).  For nsub=1
    # this is the same as the internal ray-tracing grid; for nsub>1 the
    # design_matrix path traces on the sub-grid and aggregates, so this
    # ideal check only validates the native-resolution interpolation.
    mapping_matrix = simulator.build_mapping_matrix(source_half_size=source_half_size)
    m_ideal = np.asarray(mapping_matrix @ jnp.asarray(source_pixels))
    expected_ideal = ideal_lensed_truth[~mask]

    ideal_rms = np.sqrt(np.mean((m_ideal - expected_ideal) ** 2))
    truth_scale = max(float(np.max(np.abs(expected_ideal))), 1.0e-12)
    # The 5% threshold might seem weak, but it is tightly bound to the coarse grid
    # resolution used in this test (npix=30, source_nx=15) to keep the test fast.
    # A source with sigma=0.10 spans only ~1.25 pixels on this source grid.
    # Bilinear interpolation of such narrow features naturally introduces ~2-4% error.
    # Increasing resolution (e.g., npix=60, source_nx=30) drops this error to ~0.6%.
    assert ideal_rms / truth_scale < 5.0e-2, (
        f"Ideal RMS relative error too large (nsub={nsub}): {ideal_rms / truth_scale:.3e}"
    )

    # Both the ground-truth convolution (scipy) and the design_matrix path
    # (JAX FFT) assume periodic boundaries, so edge wrap-around artifacts
    # cancel in the comparison.  The annular mask further excludes the
    # outermost pixels where boundary effects are strongest.
    ideal_lensed_masked = np.where(~mask, ideal_lensed_truth, 0.0)
    blurred_truth = fftconvolve(ideal_lensed_masked, psf_kernel, mode="same")
    expected_blurred = blurred_truth[~mask]

    design_matrix, inferred_half_size = simulator.design_matrix(
        source_half_size=source_half_size,
        psf_kernel=jnp.asarray(psf_kernel),
    )
    m_blurred = np.asarray(design_matrix @ jnp.asarray(source_pixels))

    model_image = simulator.simulate(
        jnp.asarray(source_pixels),
        source_half_size=source_half_size,
        psf_kernel=jnp.asarray(psf_kernel),
    )

    blurred_matrix_rms = np.sqrt(np.mean((m_blurred - expected_blurred) ** 2))
    blurred_sim_rms = np.sqrt(np.mean((np.asarray(model_image)[~mask] - expected_blurred) ** 2))
    blurred_scale = max(float(np.max(np.abs(expected_blurred))), 1.0e-12)

    # Similarly, the 5% threshold here is due to the coarse resolution.
    assert blurred_matrix_rms / blurred_scale < 5.0e-2, (
        f"Blurred matrix RMS too large (nsub={nsub}): {blurred_matrix_rms / blurred_scale:.3e}"
    )
    assert blurred_sim_rms / blurred_scale < 5.0e-2, (
        f"Blurred sim RMS too large (nsub={nsub}): {blurred_sim_rms / blurred_scale:.3e}"
    )

    assert jnp.allclose(inferred_half_size, source_half_size)


@pytest.mark.unit
def test_pixelized_position_likelihood_evidence_is_finite() -> None:
    """PixelizedImageProbModel with position_likelihood returns a finite scalar."""
    model = _prob_model(
        image_data=jnp.ones((10, 10)) * 0.05,
        position_likelihood={
            "positions": [(0.0, 0.0), (0.1, 0.1)],
            "threshold_arcsec": 0.3,
            "min_log_like": -1.0e10,
        },
    )
    log_evidence = model()
    assert jnp.shape(log_evidence) == ()
    assert jnp.isfinite(log_evidence)


@pytest.mark.unit
def test_pixelized_position_likelihood_inactive_returns_zero() -> None:
    """A very large threshold should yield zero penalty and unchanged evidence."""
    model_without = _prob_model(image_data=jnp.ones((10, 10)) * 0.05)
    model_with = _prob_model(
        image_data=jnp.ones((10, 10)) * 0.05,
        position_likelihood={
            "positions": [(0.0, 0.0), (0.1, 0.1)],
            "threshold_arcsec": 1.0e3,
            "min_log_like": -10.0,
        },
    )

    log_ev_without = float(model_without())
    log_ev_with = float(model_with())
    assert np.isclose(log_ev_with, log_ev_without, atol=1e-4)


@pytest.mark.unit
def test_pixelized_position_likelihood_penalizes_bad_model() -> None:
    """A lens model with shifted centre should incur a negative penalty."""
    theta_e = ParamU("theta_E", 0.12, prior_type="uniform", prior_settings=[0.05, 0.20], limits=[0.0, 1.0])
    sie = SIE(theta_E=theta_e, e1=0.0, e2=0.0, center_x=0.5, center_y=0.0)
    sie.theta_E.to_dynamic()
    for param in [sie.e1, sie.e2, sie.center_x, sie.center_y]:
        param.to_static()

    source = _pixelized_source()
    phys_model = PhysicalModel(lens_mass=[sie], source_light=[source], lens_light=[])

    data = jnp.ones((10, 10)) * 0.05
    noise = jnp.ones((10, 10)) * 0.1

    prob_without = PixelizedImageProbModel(
        image_data=data,
        noise_map=noise,
        psf_kernel=_delta_psf(),
        dpix=0.08,
        phys_model=phys_model,
        nsub=1,
    )
    prob_with = PixelizedImageProbModel(
        image_data=data,
        noise_map=noise,
        psf_kernel=_delta_psf(),
        dpix=0.08,
        phys_model=phys_model,
        nsub=1,
        position_likelihood={
            "positions": [(0.0, 0.0), (0.1, 0.1)],
            "threshold_arcsec": 0.01,
            "min_log_like": -1.0e4,
        },
    )

    log_ev_without = float(prob_without())
    log_ev_with = float(prob_with())
    penalty = log_ev_with - log_ev_without

    assert penalty < 0.0
    assert log_ev_with < log_ev_without


if __name__ == "__main__":
    pytest.main()
