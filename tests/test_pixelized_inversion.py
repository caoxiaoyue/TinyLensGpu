# pyright: reportMissingImports=false

"""
TDD specifications for pixelized source inversion workflows.

These tests define the expected public behavior for the first pixelized-source
simulator and Bayesian evidence probability model. They intentionally exercise
small synthetic problems so the future implementation can be validated quickly.
"""

import jax
import numpy as np
import pytest
import jax.numpy as jnp
import TinyLensGpu.ObservationModel.LensImage.pixelized_image_model as dense_image_model
import caskade as ck
from scipy.linalg import cho_solve
from scipy.signal import fftconvolve

from TinyLensGpu.ForwardSimulation import SimulatorConfig
from TinyLensGpu.ForwardSimulation.LensImage.config import make_grid_2d
from TinyLensGpu.ForwardSimulation.LensImage.pixelized import PixelizedLensSimulator
from TinyLensGpu.utils.geometry.transforms import phi_q2_ellipticity
from TinyLensGpu.utils.inversion.regularization import DenseRegularizationBuilder
from TinyLensGpu.utils.lensing.mapping import infer_source_bbox as infer_square_source_bbox
from TinyLensGpu.Inference import ParamU
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model import PixelizedImageProbModel
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import (
    ConstantBackground,
    GaussianEllipse,
)
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


def _pixelized_source(*, log_lambda_value: float = 0.0, with_gp: bool = False) -> PixelizedSourceModel:
    """Build a pixelized source with dynamic regularization parameters."""
    log_lambda_reg = ParamU(
        "log_lambda_reg",
        log_lambda_value,
        prior_type="uniform",
        prior_settings=[jnp.log(1.0e-3), jnp.log(1.0e3)],
        limits=[jnp.log(1.0e-6), jnp.log(1.0e6)],
    )
    log_lambda_reg.to_dynamic()
    if not with_gp:
        return PixelizedSourceModel(n=5, log_lambda_reg=log_lambda_reg)

    kernel_scale = ParamU(
        "kernel_scale",
        0.2,
        prior_type="uniform",
        prior_settings=[0.05, 1.0],
        limits=[0.01, 5.0],
    )
    kernel_scale.to_dynamic()
    return PixelizedSourceModel(n=5,
        log_lambda_reg=log_lambda_reg,
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
    fixed_source_bbox: tuple[float, float, float, float] | None = None,
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
        fixed_source_bbox=fixed_source_bbox,
    )


def _assert_square_bbox(source_bbox) -> None:
    xmin, xmax, ymin, ymax = source_bbox
    assert jnp.allclose(xmax - xmin, ymax - ymin)


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
def test_infer_source_bbox_uses_lensed_coordinate_extent_with_floor() -> None:
    """Test source bounding-box inference from ray-traced coordinates."""
    simulator = _simulator()
    beta_x = jnp.asarray([-0.2, 0.3, 0.1])
    beta_y = jnp.asarray([0.4, -0.1, 0.05])
    tiny_beta = jnp.asarray([0.0, 1.0e-14])

    xmin, xmax, ymin, ymax = simulator.infer_source_bbox(
        beta_x, beta_y, padding=0.05, outlier_frac=0.0
    )
    span_x, span_y = 0.5, 0.5
    assert jnp.allclose(xmin, -0.2 - 0.05 * span_x)
    assert jnp.allclose(xmax, 0.3 + 0.05 * span_x)
    assert jnp.allclose(ymin, -0.1 - 0.05 * span_y)
    assert jnp.allclose(ymax, 0.4 + 0.05 * span_y)
    # Tiny values still produce a valid (non-degenerate) bbox
    xmin_t, xmax_t, ymin_t, ymax_t = simulator.infer_source_bbox(
        tiny_beta, tiny_beta, padding=0.05, outlier_frac=0.0
    )
    assert xmax_t > xmin_t and ymax_t > ymin_t


@pytest.mark.unit
def test_pixelized_simulator_infers_square_source_bbox_for_asymmetric_betas() -> None:
    """Test dense pixelized simulator expands asymmetric bbox spans to square."""
    simulator = _simulator()
    beta_x = jnp.asarray([0.0, 3.0])
    beta_y = jnp.asarray([-0.2, 0.2])

    source_bbox = simulator.infer_source_bbox(
        beta_x, beta_y, padding=0.0, outlier_frac=0.0
    )

    _assert_square_bbox(source_bbox)
    xmin, xmax, ymin, ymax = source_bbox
    assert jnp.allclose(xmin, 0.0)
    assert jnp.allclose(xmax, 3.0)
    assert jnp.allclose(ymin, -1.5)
    assert jnp.allclose(ymax, 1.5)


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
    reference = _prob_model(image_data=jnp.ones((10, 10)) * 0.05, source=_pixelized_source(log_lambda_value=0.0))
    model_image, _ = reference.forward_model(return_source=True)
    noise_map = jnp.ones((10, 10)) * 0.05

    reasonable = _prob_model(image_data=model_image, noise_map=noise_map, source=_pixelized_source(log_lambda_value=0.0))
    under_smoothed = _prob_model(image_data=model_image, noise_map=noise_map, source=_pixelized_source(log_lambda_value=jnp.log(1.0e-8)))
    over_smoothed = _prob_model(image_data=model_image, noise_map=noise_map, source=_pixelized_source(log_lambda_value=jnp.log(1.0e8)))

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
def test_nnls_solver_enforces_non_negative_source_pixels() -> None:
    """NNLS solver clamps source pixels to >= 0, even when unconstrained solve goes negative."""
    true_source = jnp.linspace(-1.0, 1.0, 25)
    simulator = _simulator(psf_kernel=_delta_psf())
    image_data = simulator.simulate(true_source, psf_kernel=_delta_psf())
    model = _prob_model(image_data=image_data, noise_map=jnp.ones((10, 10)) * 0.01, psf_kernel=_delta_psf())
    model.solver_type = "nnls"

    _, source_pixels = model.forward_model(return_source=True)

    assert jnp.all(source_pixels >= -1e-6)
    assert jnp.all(jnp.isfinite(source_pixels))


@pytest.mark.unit
def test_nnls_solver_reconstructs_positive_source() -> None:
    """NNLS solver faithfully reconstructs a strictly positive source."""
    true_source = jnp.abs(jnp.linspace(-1.0, 1.0, 25))
    simulator = _simulator(psf_kernel=_delta_psf())
    image_data = simulator.simulate(true_source, psf_kernel=_delta_psf())
    model = _prob_model(image_data=image_data, noise_map=jnp.ones((10, 10)) * 0.01, psf_kernel=_delta_psf())
    model.solver_type = "nnls"

    _, source_pixels = model.forward_model(return_source=True)

    assert jnp.all(source_pixels >= -1e-6)
    assert jnp.all(jnp.isfinite(source_pixels))
    assert jnp.any(source_pixels > 0.0)  # not all zeros


@pytest.mark.unit
def test_prior_transformation_sees_mass_and_regularization_parameters() -> None:
    """Test that prior extraction includes lens mass and lambda_reg parameters."""
    _, prior_specs = make_prior_transformation(_prob_model())

    assert {spec.name for spec in prior_specs} == {"theta_E", "log_lambda_reg"}


@pytest.mark.unit
def test_prior_transformation_includes_gp_kernel_scale() -> None:
    """Test that GP pixelized sources expose their kernel-scale parameter."""
    _, prior_specs = make_prior_transformation(_prob_model(source=_pixelized_source(with_gp=True)))

    assert {spec.name for spec in prior_specs} == {"theta_E", "log_lambda_reg", "kernel_scale"}


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
    source_n = 15
    source_bbox_test = (-0.6, 0.6, -0.6, 0.6)

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

    source_x_axis = np.linspace(source_bbox_test[0], source_bbox_test[1], source_n)
    source_y_axis = np.linspace(source_bbox_test[2], source_bbox_test[3], source_n)
    source_xx, source_yy = np.meshgrid(source_x_axis, source_y_axis, indexing="xy")
    source_pixels = np.asarray(gt_phys.source_surface_brightness(source_xx, source_yy)).ravel()

    pixelized_src = PixelizedSourceModel(n=source_n, log_lambda_reg=jnp.log(1.0))
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
    mapping_matrix = simulator.build_mapping_matrix(source_bbox=source_bbox_test)
    m_ideal = np.asarray(mapping_matrix @ jnp.asarray(source_pixels))
    expected_ideal = ideal_lensed_truth[~mask]

    ideal_rms = np.sqrt(np.mean((m_ideal - expected_ideal) ** 2))
    truth_scale = max(float(np.max(np.abs(expected_ideal))), 1.0e-12)
    # The 5% threshold might seem weak, but it is tightly bound to the coarse grid
    # resolution used in this test (npix=30, source_n=15) to keep the test fast.
    # A source with sigma=0.10 spans only ~1.25 pixels on this source grid.
    # Bilinear interpolation of such narrow features naturally introduces ~2-4% error.
    # Increasing resolution (e.g., npix=60, source_n=30) drops this error to ~0.6%.
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

    design_matrix, inferred_bbox = simulator.design_matrix(
        source_bbox=source_bbox_test,
        psf_kernel=jnp.asarray(psf_kernel),
    )
    m_blurred = np.asarray(design_matrix @ jnp.asarray(source_pixels))

    model_image = simulator.simulate(
        jnp.asarray(source_pixels),
        source_bbox=source_bbox_test,
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

    assert jnp.allclose(jnp.array(inferred_bbox), jnp.array(source_bbox_test))


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


@pytest.mark.unit
def test_source_grid_shape_invariant_under_mass_param_changes() -> None:
    """Test that source grid shape remains fixed when mass params change."""
    import jax
    theta_e = ParamU("theta_E", 0.12, prior_type="uniform", prior_settings=[0.05, 0.20], limits=[0.0, 1.0])
    sie = SIE(theta_E=theta_e, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    sie.theta_E.to_dynamic()
    for param in [sie.e1, sie.e2, sie.center_x, sie.center_y]:
        param.to_static()

    source = _pixelized_source()
    phys_model = PhysicalModel(lens_mass=[sie], source_light=[source], lens_light=[])
    simulator = _simulator(phys_model=phys_model)

    design_matrix_1, bbox_1 = simulator.design_matrix()
    shape_1 = design_matrix_1.shape

    sie.theta_E.value = 0.18
    design_matrix_2, bbox_2 = simulator.design_matrix()
    shape_2 = design_matrix_2.shape

    assert shape_1 == shape_2
    assert not jnp.allclose(jnp.array(bbox_1), jnp.array(bbox_2))
    assert not jnp.allclose(design_matrix_1, design_matrix_2)
    # theta_E increased → deflection angles larger → bbox must change
    span_1_x = jnp.array(bbox_1[1] - bbox_1[0])
    span_2_x = jnp.array(bbox_2[1] - bbox_2[0])
    span_1_y = jnp.array(bbox_1[3] - bbox_1[2])
    span_2_y = jnp.array(bbox_2[3] - bbox_2[2])
    assert not jnp.allclose(span_2_x, span_1_x), (
        f"Expected different bbox x-span for different theta_E: {span_2_x} == {span_1_x}"
    )
    assert not jnp.allclose(span_2_y, span_1_y), (
        f"Expected different bbox y-span for different theta_E: {span_2_y} == {span_1_y}"
    )


@pytest.mark.unit
def test_bilinear_interpolation_gradients_flow_through_beta() -> None:
    """Test that build_lens_mapping_matrix is differentiable w.r.t. beta coordinates."""
    import jax
    from TinyLensGpu.utils.lensing.mapping import build_lens_mapping_matrix, build_source_grid

    source_x_axis, source_y_axis, _, _ = build_source_grid(5, -1.0, 1.0, -1.0, 1.0)

    def loss_fn(beta_x, beta_y):
        mapping_matrix = build_lens_mapping_matrix(beta_x, beta_y, source_x_axis, source_y_axis)
        return jnp.sum(mapping_matrix ** 2)

    beta_x = jnp.array([0.1, -0.3, 0.5])
    beta_y = jnp.array([0.2, 0.4, -0.1])

    grad_fn = jax.grad(loss_fn, argnums=(0, 1))
    grad_x, grad_y = grad_fn(beta_x, beta_y)

    assert jnp.all(jnp.isfinite(grad_x))
    assert jnp.all(jnp.isfinite(grad_y))
    # All beta points are in-bounds and should produce non-zero gradients
    assert jnp.all(grad_x != 0.0), "all in-bounds beta points should contribute x-gradients"
    assert jnp.all(grad_y != 0.0), "all in-bounds beta points should contribute y-gradients"


@pytest.mark.unit
def test_infer_source_bbox_asymmetric_offset_betas() -> None:
    """Test bounding-box inference with betas fully offset from the origin."""
    import jax
    from TinyLensGpu.utils.lensing.mapping import infer_source_bbox

    # Betas entirely in the first quadrant, no symmetry around origin
    beta_x = jnp.asarray([0.5, 0.8, 1.2, 1.5])
    beta_y = jnp.asarray([0.3, 0.6, 0.9, 1.1])

    xmin, xmax, ymin, ymax = infer_source_bbox(
        beta_x, beta_y, padding=0.05, outlier_frac=0.0
    )

    # Square bbox: x-span determines both axes
    span_x = 1.5 - 0.5  # = 1.0 (longer span)
    expected_half = 0.5 * span_x * (1.0 + 2.0 * 0.05)  # padded half-span
    x_center = 1.0
    y_center = 0.7
    assert jnp.allclose(xmin, x_center - expected_half)
    assert jnp.allclose(xmax, x_center + expected_half)
    assert jnp.allclose(ymin, y_center - expected_half)
    assert jnp.allclose(ymax, y_center + expected_half)
    # Bbox should be fully offset — min values should be far from origin
    assert xmin > 0.0
    assert ymin > 0.0

    # Single-point source: should produce a valid (non-degenerate) bbox
    single = jnp.asarray([0.7])
    xmin_s, xmax_s, ymin_s, ymax_s = infer_source_bbox(
        single, single, padding=0.05, outlier_frac=0.0
    )
    assert xmax_s > xmin_s
    assert ymax_s > ymin_s


@pytest.mark.unit
def test_infer_source_bbox_custom_padding() -> None:
    """Test infer_source_bbox respects non-default padding values."""
    from TinyLensGpu.utils.lensing.mapping import infer_source_bbox

    beta_x = jnp.asarray([-1.0, 1.0])
    beta_y = jnp.asarray([-1.0, 1.0])

    # 5% padding (explicit)
    xmin_5, xmax_5, ymin_5, ymax_5 = infer_source_bbox(
        beta_x, beta_y, padding=0.05, outlier_frac=0.0
    )
    assert jnp.allclose(xmin_5, -1.0 - 0.05 * 2.0)
    assert jnp.allclose(xmax_5, 1.0 + 0.05 * 2.0)

    # 20% padding
    xmin_20, xmax_20, ymin_20, ymax_20 = infer_source_bbox(
        beta_x, beta_y, padding=0.20, outlier_frac=0.0
    )
    assert jnp.allclose(xmin_20, -1.0 - 0.20 * 2.0)
    assert jnp.allclose(xmax_20, 1.0 + 0.20 * 2.0)

    # Zero padding: bbox == data extent
    xmin_0, xmax_0, ymin_0, ymax_0 = infer_source_bbox(
        beta_x, beta_y, padding=0.0, outlier_frac=0.0
    )
    assert jnp.allclose(xmin_0, -1.0)
    assert jnp.allclose(xmax_0, 1.0)

    # Floor still applies for point-like sources with zero padding
    tiny = jnp.asarray([1.0e-10])
    xmin_t, xmax_t, ymin_t, ymax_t = infer_source_bbox(
        tiny, tiny, padding=0.0, outlier_frac=0.0
    )
    assert xmax_t > xmin_t


@pytest.mark.unit
def test_detach_bbox_false_allows_gradient_through_bbox() -> None:
    """Test that detach_bbox=False produces different gradients than detach_bbox=True.

    When detach_bbox=False, the bounding-box bounds flow through to the grid
    construction and regularization matrix, contributing additional gradient
    components not present when detach_bbox=True.
    """
    import jax

    theta_e = ParamU("theta_E", 0.12, prior_type="uniform",
                     prior_settings=[0.05, 0.20], limits=[0.0, 1.0])
    sie = SIE(theta_E=theta_e, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
    sie.theta_E.to_dynamic()
    for param in [sie.e1, sie.e2, sie.center_x, sie.center_y]:
        param.to_static()

    source = _pixelized_source(log_lambda_value=jnp.log(0.1))
    phys_model = PhysicalModel(lens_mass=[sie], source_light=[source], lens_light=[])

    def make_loss(detach_bbox):
        sim = PixelizedLensSimulator(
            phys_model, SimulatorConfig(
                dpix=0.08, npix=10, psf_kernel=_delta_psf(), nsub=1,
            ),
            detach_bbox=detach_bbox,
        )

        def loss(theta_e_val):
            sie.theta_E.value = theta_e_val
            design_matrix, _ = sim.design_matrix()
            return jnp.sum(design_matrix ** 2)

        return loss

    theta_e_val = jnp.array(0.12)
    grad_detached = jax.grad(make_loss(detach_bbox=True))(theta_e_val)
    grad_attached = jax.grad(make_loss(detach_bbox=False))(theta_e_val)

    # Both should be finite and non-zero
    assert jnp.isfinite(grad_detached).all()
    assert jnp.isfinite(grad_attached).all()
    assert grad_detached != 0.0
    assert grad_attached != 0.0

    # detach_bbox=False includes extra gradient components via bbox bounds,
    # so the gradient magnitude should differ from the detached case
    assert not jnp.allclose(grad_detached, grad_attached), (
        "detach_bbox=True and detach_bbox=False produce identical gradients — "
        "bbox gradient contribution is missing."
    )


@pytest.mark.unit
def test_detach_bbox_gradients_flow_through_mass_params() -> None:
    """Test that detach_bbox=True stops bbox gradients but preserves mass-param gradients.

    With detach_bbox=True (default), the bounding-box bounds (xmin/xmax/ymin/ymax)
    are detached via stop_gradient, but the mapping matrix still depends
    differentiably on beta_x/beta_y which depend on mass parameters. This test
    verifies that gradient flows from the design matrix back through the mass
    model when detach_bbox is active.

    Note: uses e1=0.1 (non-zero ellipticity) to avoid the singular gradient
    of a perfectly circular SIS.
    """
    import jax

    theta_e = ParamU("theta_E", 0.12, prior_type="uniform",
                     prior_settings=[0.05, 0.20], limits=[0.0, 1.0])
    sie = SIE(theta_E=theta_e, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
    sie.theta_E.to_dynamic()
    for param in [sie.e1, sie.e2, sie.center_x, sie.center_y]:
        param.to_static()

    source = _pixelized_source(log_lambda_value=jnp.log(0.1))
    phys_model = PhysicalModel(lens_mass=[sie], source_light=[source], lens_light=[])
    simulator = PixelizedLensSimulator(phys_model, SimulatorConfig(
        dpix=0.08, npix=10, psf_kernel=_delta_psf(), nsub=1,
    ))

    def loss_fn(theta_e_val):
        sie.theta_E.value = theta_e_val
        design_matrix, _ = simulator.design_matrix()
        return jnp.sum(design_matrix ** 2)

    theta_e_val = jnp.array(0.12)
    grad = jax.grad(loss_fn)(theta_e_val)

    assert jnp.isfinite(grad).all()
    assert grad != 0.0, (
        "Gradient through mass params is zero — detach_bbox may be "
        "incorrectly applied or the loss function has no dependence on "
        "the mass parameter through the design matrix."
    )


def _normal_equation_evidence(model: PixelizedImageProbModel, source_bbox: tuple) -> float:
    """Evidence of one explicit source grid, solved outside the probability model."""
    design, _ = model.sim_obj.design_matrix(source_bbox=source_bbox)
    regularizer, covariance_logdet = model._regularization_matrix(source_bbox)
    assert covariance_logdet is None
    matrix = np.asarray(design, dtype=np.float64)
    precision = np.asarray(regularizer, dtype=np.float64)
    noise = np.asarray(model.noise_1d, dtype=np.float64)
    data = np.asarray(model.data_1d, dtype=np.float64)
    log_lambda = float(np.asarray(model.source_model.log_lambda_reg.value))
    lam = np.exp(log_lambda)
    weighted = matrix / noise[:, None]
    curvature = weighted.T @ weighted
    n_source = precision.shape[0]
    curvature[:n_source, :n_source] += lam * precision
    rhs = weighted.T @ (data / noise)
    factor = np.linalg.cholesky(curvature)
    linear = cho_solve((factor, True), rhs)
    source = linear[:n_source]
    residual = data - matrix @ linear
    data_term = 0.5 * np.sum((residual / noise) ** 2)
    source_term = 0.5 * float(source @ precision @ source)
    logdet_curvature = 2.0 * np.sum(np.log(np.diag(factor)))
    sign, logdet_reg = np.linalg.slogdet(precision)
    if sign <= 0.0:
        return -1.0e10
    evidence = (
        -data_term
        - lam * source_term
        - 0.5 * logdet_curvature
        + 0.5 * n_source * log_lambda
        + 0.5 * logdet_reg
        - 0.5 * data.size * np.log(2.0 * np.pi)
        - 0.5 * float(np.asarray(model.logdet_C))
    )
    return float(evidence) if np.isfinite(evidence) else -1.0e10


@pytest.mark.unit
def test_fixed_source_bbox_keeps_evidence_and_reconstruction_on_that_grid() -> None:
    """Mass proposals keep the frozen source grid for the solve and the reconstruction."""
    fixed_bbox = (-0.8, 0.8, -0.8, 0.8)
    model = _prob_model(
        image_data=jnp.ones((10, 10)) * 0.05,
        fixed_source_bbox=fixed_bbox,
    )
    inferred = _prob_model(image_data=jnp.ones((10, 10)) * 0.05).infer_source_bbox()
    assert not np.allclose(inferred, fixed_bbox, atol=1e-3)

    for theta_e in (0.12, 0.45):
        model.phys_model.lens_mass[0].theta_E.value = theta_e
        image, source = model.forward_model(return_source=True)
        fixed_design, _ = model.sim_obj.design_matrix(source_bbox=fixed_bbox)
        inferred_design, inferred_now = model.sim_obj.design_matrix()
        active = np.asarray(image)[np.asarray(model.unmask)]
        fixed_image = np.asarray(fixed_design @ source)
        inferred_image = np.asarray(inferred_design @ source)
        np.testing.assert_allclose(active, fixed_image, rtol=1e-5, atol=1e-5)
        assert not np.allclose(inferred_now, fixed_bbox, atol=1e-3)
        assert not np.allclose(active, inferred_image, rtol=1e-4, atol=1e-4)
        fixed_evidence = _normal_equation_evidence(model, fixed_bbox)
        inferred_evidence = _normal_equation_evidence(model, tuple(inferred_now))
        np.testing.assert_allclose(float(model()), fixed_evidence, rtol=1e-5, atol=1e-5)
        assert abs(float(model()) - inferred_evidence) > 1.0e-2


@pytest.mark.unit
def test_fixed_source_bbox_outside_the_rays_reconstructs_nothing() -> None:
    """Rays that miss a frozen source grid contribute no source light."""
    bbox = (5.0, 5.4, 5.0, 5.4)
    model = _prob_model(
        image_data=jnp.ones((10, 10)) * 0.05,
        fixed_source_bbox=bbox,
    )
    design, _ = model.sim_obj.design_matrix(source_bbox=bbox)
    assert np.all(np.asarray(design) == 0.0)
    image, source = model.forward_model(return_source=True)
    assert np.all(np.asarray(image) == 0.0)
    assert np.all(np.asarray(source) == 0.0)


@pytest.mark.unit
def test_fixed_source_bbox_replaces_nonfinite_evidence_with_penalty() -> None:
    """A non-finite fixed-grid evidence stays the existing penalty value."""
    image = jnp.ones((10, 10)) * 0.05
    image = image.at[4, 4].set(jnp.nan)
    model = _prob_model(image_data=image, fixed_source_bbox=(-0.8, 0.8, -0.8, 0.8))
    assert float(model()) == -1.0e10


@pytest.mark.unit
def test_fixed_source_bbox_still_rejects_adaptive_regularization() -> None:
    """Freezing the grid does not re-enable adaptive source regularization."""
    rho = ParamU(
        "adaptive_reg_rho",
        0.4,
        prior_type="uniform",
        prior_settings=[0.0, 1.0],
        limits=[0.0, 2.0],
    )
    source = PixelizedSourceModel(n=5, log_lambda_reg=0.0, adaptive_reg_rho=rho)
    with pytest.raises(ValueError, match="no longer supports"):
        _prob_model(source=source, fixed_source_bbox=(-0.5, 0.5, -0.5, 0.5))


@pytest.mark.unit
def test_fixed_source_bbox_keeps_position_penalty() -> None:
    """A position constraint still lowers fixed-grid evidence for a shifted lens."""
    theta_e = ParamU("theta_E", 0.12, prior_type="uniform", prior_settings=[0.05, 0.20], limits=[0.0, 1.0])
    sie = SIE(theta_E=theta_e, e1=0.0, e2=0.0, center_x=0.5, center_y=0.0)
    sie.theta_E.to_dynamic()
    for param in [sie.e1, sie.e2, sie.center_x, sie.center_y]:
        param.to_static()
    phys_model = PhysicalModel(
        lens_mass=[sie],
        source_light=[_pixelized_source()],
        lens_light=[],
    )
    data = jnp.ones((10, 10)) * 0.05
    noise = jnp.ones((10, 10)) * 0.1
    shared = dict(
        image_data=data,
        noise_map=noise,
        psf_kernel=_delta_psf(),
        dpix=0.08,
        phys_model=phys_model,
        fixed_source_bbox=(-0.8, 0.8, -0.8, 0.8),
    )
    without = PixelizedImageProbModel(**shared)
    with_penalty = PixelizedImageProbModel(
        **shared,
        position_likelihood={
            "positions": [(0.0, 0.0), (0.1, 0.1)],
            "threshold_arcsec": 0.01,
            "min_log_like": -1.0e4,
        },
    )
    assert float(with_penalty()) < float(without())


@pytest.mark.unit
def test_dense_infer_source_bbox_reports_fixed_or_seed_ray_grid() -> None:
    """The helper reports the grid a second dense model should freeze."""
    free = _prob_model(image_data=jnp.ones((10, 10)) * 0.05)
    beta_x, beta_y = free.phys_model.deflection(
        x=free.sim_obj.image_x_seed,
        y=free.sim_obj.image_y_seed,
    )
    expected = infer_square_source_bbox(beta_x, beta_y, padding=0.05, outlier_frac=0.0)
    np.testing.assert_allclose(
        free.infer_source_bbox(),
        [float(np.asarray(value)) for value in expected],
        atol=1e-6,
    )

    near_square = (0.0, 1.0, 0.0, 1.0 + 5.0e-7)
    frozen = _prob_model(
        image_data=jnp.ones((10, 10)) * 0.05,
        fixed_source_bbox=near_square,
    )
    np.testing.assert_allclose(frozen.infer_source_bbox(), near_square, atol=1e-6)
    frozen.phys_model.lens_mass[0].theta_E.value = 0.5
    np.testing.assert_allclose(frozen.infer_source_bbox(), near_square, atol=1e-6)
    moved = free.infer_source_bbox()
    free.phys_model.lens_mass[0].theta_E.value = 0.5
    assert not np.allclose(moved, free.infer_source_bbox(), atol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("fixed_source_bbox", "message"),
    [
        ((-1.0, 1.0, -0.5, 0.5), "fixed_source_bbox must be square"),
        ((0.0, 1.0, 0.0, 1.0 + 2.0e-6), "fixed_source_bbox must be square"),
        ((-0.3, -0.3, -0.3, 0.3), "xmin < xmax and ymin < ymax"),
        ((0.3, -0.3, -0.3, 0.3), "xmin < xmax and ymin < ymax"),
        ((-0.3, 0.3, float("nan"), 0.3), "fixed_source_bbox values must be finite"),
        ((-0.3, 0.3, -0.3, float("inf")), "fixed_source_bbox values must be finite"),
        ((0.0, 1.0, 0.0), "fixed_source_bbox must be a 4-tuple"),
    ],
)
def test_dense_model_rejects_invalid_fixed_source_bbox(fixed_source_bbox, message) -> None:
    """A fixed pixelized source grid must be a finite square before any solve."""
    with pytest.raises(ValueError, match=message):
        _prob_model(fixed_source_bbox=fixed_source_bbox)


_FIXED_BBOX = (-0.8, 0.8, -0.8, 0.8)


def _static_sie(theta_e: float = 0.12) -> SIE:
    """Build an SIE whose mass parameters are all static."""
    sie = SIE(theta_E=theta_e, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    for param in (sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y):
        param.to_static()
    return sie


def _regularized_source(
    reg_type: str = "second-order",
    *,
    log_lambda: float = 0.0,
    kernel_scale: float = 0.2,
    kernel_dynamic: bool = False,
) -> PixelizedSourceModel:
    """Build a pixelized source with a dynamic regularization strength."""
    log_lambda_reg = ParamU(
        "log_lambda_reg",
        log_lambda,
        prior_type="uniform",
        prior_settings=[jnp.log(1.0e-3), jnp.log(1.0e3)],
        limits=[jnp.log(1.0e-6), jnp.log(1.0e6)],
    )
    log_lambda_reg.to_dynamic()
    if reg_type not in ("exponential", "gaussian", "matern32", "matern52", "matern72"):
        return PixelizedSourceModel(
            n=5,
            log_lambda_reg=log_lambda_reg,
            regularization_type=reg_type,
        )
    scale = ParamU(
        "kernel_scale",
        kernel_scale,
        prior_type="uniform",
        prior_settings=[0.05, 1.0],
        limits=[0.01, 5.0],
    )
    if kernel_dynamic:
        scale.to_dynamic()
    else:
        scale.to_static()
    return PixelizedSourceModel(
        n=5,
        log_lambda_reg=log_lambda_reg,
        regularization_type=reg_type,
        kernel_scale=scale,
    )


def _unit_lens_light(*, sigma_dynamic: bool = False) -> GaussianEllipse:
    """Build a unit-amplitude Gaussian lens-light basis."""
    light = GaussianEllipse(
        flux=1.0, sigma=0.2, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0,
    )
    for param in (light.flux, light.sigma, light.e1, light.e2, light.center_x, light.center_y):
        param.to_static()
    if sigma_dynamic:
        light.sigma.to_dynamic()
    return light


def _dense_model(
    *,
    reg_type: str = "second-order",
    fixed_source_bbox=_FIXED_BBOX,
    dynamic_mass: bool = True,
    lens_mass=None,
    lens_light=None,
    source: PixelizedSourceModel | None = None,
    solver_type: str = "cholesky",
    nsub: int = 1,
    mask=None,
    psf_kernel=None,
    image_data=None,
    position_likelihood=None,
    log_lambda: float = 0.0,
    kernel_scale: float = 0.2,
    kernel_dynamic: bool = False,
) -> PixelizedImageProbModel:
    """Build a small dense evidence model for cache tests."""
    data = jnp.ones((10, 10)) * 0.05 if image_data is None else image_data
    if lens_mass is None:
        lens_mass = [_dynamic_sie()] if dynamic_mass else [_static_sie()]
    if source is None:
        source = _regularized_source(
            reg_type,
            log_lambda=log_lambda,
            kernel_scale=kernel_scale,
            kernel_dynamic=kernel_dynamic,
        )
    phys_model = PhysicalModel(
        lens_mass=list(lens_mass),
        source_light=[source],
        lens_light=list(lens_light or []),
    )
    return PixelizedImageProbModel(
        image_data=data,
        noise_map=jnp.ones((10, 10)) * 0.1,
        psf_kernel=_delta_psf() if psf_kernel is None else psf_kernel,
        dpix=0.08,
        phys_model=phys_model,
        mask=mask,
        nsub=nsub,
        solver_type=solver_type,
        position_likelihood=position_likelihood,
        fixed_source_bbox=fixed_source_bbox,
    )


def _refuse_reconstruction(*_args, **_kwargs):
    raise AssertionError("fixed-grid construction was repeated")


def _clear_dense_cache(model: PixelizedImageProbModel) -> None:
    """Drop the prepared regularization and design terms on a reference model."""
    model._prepared = model._prepared._replace(regularization=None, design=None)


def _assert_cache_agreement(cached: PixelizedImageProbModel, cleared: PixelizedImageProbModel) -> None:
    """Cached and cache-cleared evidence, image, and source agree for both solvers."""
    _clear_dense_cache(cleared)
    for solver in ("cholesky", "nnls"):
        cached.solver_type = solver
        cleared.solver_type = solver
        np.testing.assert_allclose(float(cached()), float(cleared()), rtol=1e-6, atol=1e-6)
        cached_image, cached_source = cached.forward_model(return_source=True)
        cleared_image, cleared_source = cleared.forward_model(return_source=True)
        np.testing.assert_allclose(cached_image, cleared_image, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(cached_source, cleared_source, rtol=1e-6, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("reg_type", ["zero-order", "first-order", "second-order"])
def test_fixed_grid_reuses_finite_difference_regularization(monkeypatch, reg_type) -> None:
    """A fixed grid keeps the finite-difference matrix and its evidence determinant."""
    model = _dense_model(reg_type=reg_type)
    _assert_cache_agreement(model, _dense_model(reg_type=reg_type))
    stored_matrix, covariance_logdet = model._regularization_matrix(model.infer_source_bbox())
    assert covariance_logdet is None
    evidence = float(model())
    image, source = model.forward_model(return_source=True)

    monkeypatch.setattr(dense_image_model.jnp.linalg, "slogdet", _refuse_reconstruction)
    monkeypatch.setattr(model.reg_builder, "matrix", _refuse_reconstruction)
    np.testing.assert_allclose(float(model()), evidence, rtol=1e-6, atol=1e-6)
    cached_image, cached_source = model.forward_model(return_source=True)
    np.testing.assert_allclose(cached_image, image, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cached_source, source, rtol=1e-6, atol=1e-6)
    hit_matrix, hit_logdet = model._regularization_matrix(model._fixed_source_bbox)
    assert hit_logdet is None
    assert hit_matrix is stored_matrix

    calls = []

    def _count_other_bbox(xmin, xmax, ymin, ymax, **kwargs):
        calls.append((xmin, xmax, ymin, ymax))
        builder = DenseRegularizationBuilder(5, reg_type)
        return builder.matrix(xmin, xmax, ymin, ymax, **kwargs)

    monkeypatch.setattr(model.reg_builder, "matrix", _count_other_bbox)
    other_bbox = (0.0, 1.2, 0.0, 1.2)
    other_matrix, other_logdet = model._regularization_matrix(other_bbox)
    assert len(calls) == 1
    assert other_logdet is None
    assert other_matrix is not stored_matrix

    monkeypatch.setattr(model.reg_builder, "matrix", _refuse_reconstruction)
    restored, restored_logdet = model._regularization_matrix(model.infer_source_bbox())
    assert restored_logdet is None
    assert restored is stored_matrix
    np.testing.assert_allclose(float(model()), evidence, rtol=1e-6, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("solver", ["cholesky", "nnls"])
def test_fixed_grid_regularization_strength_still_changes_the_source(solver) -> None:
    """Regularization strength stays outside the cached finite-difference matrix."""
    cached = _dense_model(solver_type=solver, log_lambda=float(jnp.log(0.1)))
    cleared = _dense_model(solver_type=solver, log_lambda=float(jnp.log(0.1)))
    _clear_dense_cache(cleared)
    weak_evidence = float(cached())
    weak_source = np.asarray(cached.forward_model(return_source=True)[1])
    np.testing.assert_allclose(weak_evidence, float(cleared()), rtol=1e-6, atol=1e-6)

    for model in (cached, cleared):
        model.source_model.log_lambda_reg.value = float(jnp.log(10.0))
    strong_evidence = float(cached())
    strong_source = np.asarray(cached.forward_model(return_source=True)[1])
    np.testing.assert_allclose(strong_evidence, float(cleared()), rtol=1e-6, atol=1e-6)
    cached_image, cached_source = cached.forward_model(return_source=True)
    cleared_image, cleared_source = cleared.forward_model(return_source=True)
    np.testing.assert_allclose(cached_image, cleared_image, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cached_source, cleared_source, rtol=1e-6, atol=1e-6)
    assert abs(strong_evidence - weak_evidence) > 1.0e-3
    assert not np.allclose(weak_source, strong_source, rtol=1e-6, atol=1e-6)


@pytest.mark.unit
def test_dynamic_mass_reuses_regularization_and_rebuilds_the_design(monkeypatch) -> None:
    """Sampling lens mass keeps the fixed-grid regularizer and retraces the mapping."""
    model = _dense_model(dynamic_mass=True)
    calls = []
    design_matrix = model.sim_obj.design_matrix

    def _count_design(*args, **kwargs):
        calls.append(kwargs.get("source_bbox"))
        return design_matrix(*args, **kwargs)

    monkeypatch.setattr(model.sim_obj, "design_matrix", _count_design)
    monkeypatch.setattr(model.reg_builder, "matrix", _refuse_reconstruction)
    monkeypatch.setattr(dense_image_model.jnp.linalg, "slogdet", _refuse_reconstruction)
    assert np.isfinite(float(model()))
    image, source = model.forward_model(return_source=True)
    assert jnp.all(jnp.isfinite(image))
    assert source.shape == (25,)
    assert len(calls) == 2
    for bbox in calls:
        np.testing.assert_allclose(bbox, _FIXED_BBOX, atol=1e-6)


@pytest.mark.unit
def test_unfixed_model_still_builds_regularization_and_design(monkeypatch) -> None:
    """Omitting the fixed bbox leaves both constructions on the evaluation path."""
    model = _dense_model(fixed_source_bbox=None)
    calls = {"design": 0, "regularization": 0}
    design_matrix = model.sim_obj.design_matrix
    regularize = model.reg_builder.matrix

    def _count_design(*args, **kwargs):
        calls["design"] += 1
        return design_matrix(*args, **kwargs)

    def _count_regularization(*args, **kwargs):
        calls["regularization"] += 1
        return regularize(*args, **kwargs)

    monkeypatch.setattr(model.sim_obj, "design_matrix", _count_design)
    monkeypatch.setattr(model.reg_builder, "matrix", _count_regularization)
    assert np.isfinite(float(model()))
    assert calls == {"design": 1, "regularization": 1}


@pytest.mark.unit
def test_static_gaussian_process_kernel_is_reused(monkeypatch) -> None:
    """A static kernel scale stores the precision and log|K| for the fixed grid."""
    model = _dense_model(reg_type="gaussian", dynamic_mass=True, kernel_dynamic=False)
    matrix, log_covariance = model._regularization_matrix(model.infer_source_bbox())
    assert log_covariance is not None
    assert np.isfinite(float(np.asarray(log_covariance)))
    evidence = float(model())
    monkeypatch.setattr(model.reg_builder, "matrix", _refuse_reconstruction)
    np.testing.assert_allclose(float(model()), evidence, rtol=1e-6, atol=1e-6)
    hit_matrix, hit_logdet = model._regularization_matrix(model._fixed_source_bbox)
    assert hit_matrix is matrix
    np.testing.assert_allclose(hit_logdet, log_covariance, rtol=1e-6, atol=1e-6)
    _assert_cache_agreement(
        model,
        _dense_model(reg_type="gaussian", dynamic_mass=True, kernel_dynamic=False),
    )


@pytest.mark.unit
@pytest.mark.parametrize("driver", ["dynamic", "pointer", "childless-pointer"])
def test_dynamic_kernel_scale_is_rebuilt_on_the_fixed_bbox(driver) -> None:
    """A kernel scale that can still change is rebuilt on the frozen source grid."""
    source = _regularized_source("gaussian", kernel_dynamic=(driver == "dynamic"))
    if driver == "pointer":
        scale_driver = ParamU(
            "driver", 0.2, prior_type="uniform", prior_settings=[0.05, 1.0], limits=[0.01, 5.0],
        )
        scale_driver.to_dynamic()
        source.kernel_scale = ck.Param("kernel_scale", lambda _: scale_driver.value, link=(scale_driver,))
    elif driver == "childless-pointer":
        source.kernel_scale = ck.Param("kernel_scale", lambda _: jnp.asarray(0.2))
    model = _dense_model(source=source, dynamic_mass=True)
    assert model._prepared.regularization is None
    seen = []
    regularize = model.reg_builder.matrix

    def _record(xmin, xmax, ymin, ymax, **kwargs):
        seen.append((float(xmin), float(xmax), float(ymin), float(ymax), float(kwargs["kernel_scale"])))
        return regularize(xmin, xmax, ymin, ymax, **kwargs)

    model.reg_builder.matrix = _record
    assert np.isfinite(float(model()))
    if driver == "pointer":
        scale_driver.value = 0.55
    elif driver == "dynamic":
        source.kernel_scale.value = 0.55
    assert np.isfinite(float(model()))
    assert len(seen) == 2
    np.testing.assert_allclose(seen[0][:4], _FIXED_BBOX, atol=1e-6)
    np.testing.assert_allclose(seen[1][:4], _FIXED_BBOX, atol=1e-6)
    if driver != "childless-pointer":
        assert seen[0][4] != pytest.approx(seen[1][4])
    assert model._prepared.regularization is None


@pytest.mark.unit
def test_fixed_mapping_reuses_the_joint_design_matrix(monkeypatch) -> None:
    """Fixed mass and lens light keep the design matrix out of later evaluations."""
    joint = dict(dynamic_mass=False, lens_light=[_unit_lens_light()])
    model = _dense_model(**joint)
    cleared = _dense_model(**joint)
    _assert_cache_agreement(model, cleared)

    monkeypatch.setattr(model.sim_obj, "design_matrix", _refuse_reconstruction)
    monkeypatch.setattr(model.reg_builder, "matrix", _refuse_reconstruction)
    assert np.isfinite(float(model()))
    image, source = model.forward_model(return_source=True)
    assert image.shape == (10, 10)
    assert source.shape == (25,)

    empty = _dense_model(lens_mass=[], lens_light=[])
    monkeypatch.setattr(empty.sim_obj, "design_matrix", _refuse_reconstruction)
    assert np.isfinite(float(empty()))
    assert empty.forward_model().shape == (10, 10)


@pytest.mark.unit
def test_subpixel_joint_design_matches_a_cleared_cache() -> None:
    """A cached sub-pixel design matrix matches the matrix rebuilt for the solver."""
    mask = jnp.zeros((10, 10), dtype=bool).at[0, :].set(True)
    kwargs = dict(
        dynamic_mass=False,
        lens_light=[_unit_lens_light()],
        nsub=2,
        mask=mask,
        psf_kernel=_blur_psf(),
    )
    _assert_cache_agreement(_dense_model(**kwargs), _dense_model(**kwargs))


@pytest.mark.unit
def test_cached_design_refits_regularization_under_vmap(monkeypatch) -> None:
    """Two compiled proposals share the design matrix and not the regularization strength."""
    model = _dense_model(dynamic_mass=False, lens_light=[_unit_lens_light()])
    monkeypatch.setattr(model.sim_obj, "design_matrix", _refuse_reconstruction)
    monkeypatch.setattr(model.reg_builder, "matrix", _refuse_reconstruction)
    base = jnp.asarray(model.get_values("flat"))
    proposals = jnp.stack([base.at[0].set(jnp.log(0.05)), base.at[0].set(jnp.log(30.0))])

    @jax.jit
    def one_source(theta):
        return model.forward_model(theta, return_source=True)[1]

    @jax.jit
    def one_evidence(theta):
        return model(theta)

    sources = jax.jit(jax.vmap(one_source))(proposals)
    evidences = jax.jit(jax.vmap(one_evidence))(proposals)
    assert not np.allclose(sources[0], sources[1], rtol=1e-6, atol=1e-6)
    assert abs(float(evidences[0]) - float(evidences[1])) > 1.0e-3


@pytest.mark.unit
@pytest.mark.parametrize("vary", ["mass", "lens_light", "mass_pointer", "childless_pointer"])
def test_dynamic_mapping_skips_the_design_cache(monkeypatch, vary) -> None:
    """A dynamic mass or lens-light basis leaves the design triple unstored."""
    lens_light = [_unit_lens_light(sigma_dynamic=(vary == "lens_light"))]
    if vary == "mass":
        lens_mass = [_dynamic_sie()]
    elif vary == "lens_light":
        lens_mass = [_static_sie()]
    else:
        sie = _static_sie()
        if vary == "mass_pointer":
            driver = ParamU(
                "driver", 0.12, prior_type="uniform", prior_settings=[0.05, 0.2], limits=[0.0, 1.0],
            )
            driver.to_dynamic()
            sie.theta_E = ck.Param("theta_E", lambda _: driver.value, link=(driver,))
        else:
            sie.theta_E = ck.Param("theta_E", lambda _: jnp.asarray(0.12))
        lens_mass = [sie]
    model = _dense_model(lens_mass=lens_mass, lens_light=lens_light, dynamic_mass=False)
    assert model._prepared.design is None
    assert model._prepared.regularization is not None
    calls = []
    design_matrix = model.sim_obj.design_matrix

    def _count_design(*args, **kwargs):
        calls.append(kwargs.get("source_bbox"))
        return design_matrix(*args, **kwargs)

    monkeypatch.setattr(model.sim_obj, "design_matrix", _count_design)
    monkeypatch.setattr(model.reg_builder, "matrix", _refuse_reconstruction)
    assert np.isfinite(float(model()))
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], _FIXED_BBOX, atol=1e-6)


@pytest.mark.unit
def test_static_mass_and_dynamic_kernel_cache_only_the_design(monkeypatch) -> None:
    """A moving kernel scale keeps the design cache and rebuilds the precision."""
    model = _dense_model(reg_type="gaussian", kernel_dynamic=True, dynamic_mass=False)
    assert model._prepared.design is not None
    assert model._prepared.regularization is None
    calls = []
    regularize = model.reg_builder.matrix

    def _count_regularization(*args, **kwargs):
        calls.append(args[:4])
        return regularize(*args, **kwargs)

    monkeypatch.setattr(model.reg_builder, "matrix", _count_regularization)
    monkeypatch.setattr(model.sim_obj, "design_matrix", _refuse_reconstruction)
    assert np.isfinite(float(model()))
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], _FIXED_BBOX, atol=1e-6)


@pytest.mark.unit
def test_simulator_design_matrix_ignores_the_evidence_cache() -> None:
    """The simulator follows the mass and bbox of the call, not the stored design."""
    model = _dense_model(dynamic_mass=False)
    fixed = model._fixed_source_bbox
    original = np.asarray(model.sim_obj.design_matrix(source_bbox=fixed)[0])
    model.phys_model.lens_mass[0].theta_E.value = 0.45
    moved = np.asarray(model.sim_obj.design_matrix(source_bbox=fixed)[0])
    other = np.asarray(model.sim_obj.design_matrix(source_bbox=(0.0, 1.2, 0.0, 1.2))[0])
    assert not np.allclose(original, moved)
    assert not np.allclose(moved, other)
    cached_evidence = float(model())
    model.sim_obj.design_matrix = _refuse_reconstruction
    np.testing.assert_allclose(float(model()), cached_evidence, rtol=1e-6, atol=1e-6)


@pytest.mark.unit
def test_design_cache_keeps_position_and_nonfinite_penalties(monkeypatch) -> None:
    """Position and non-finite penalties stay outside the cached design solve."""
    sie = _static_sie()
    sie.center_x.value = 0.5
    penalty = {
        "positions": [(0.0, 0.0), (0.1, 0.1)],
        "threshold_arcsec": 0.01,
        "min_log_like": -1.0e4,
    }
    plain = _dense_model(lens_mass=[_static_sie()], dynamic_mass=False)
    shifted = _dense_model(
        lens_mass=[sie], dynamic_mass=False, position_likelihood=penalty,
    )
    assert float(shifted()) < float(plain())

    image = jnp.ones((10, 10)) * 0.05
    image = image.at[4, 4].set(jnp.nan)
    nonfinite = _dense_model(dynamic_mass=False, image_data=image)
    monkeypatch.setattr(shifted.sim_obj, "design_matrix", _refuse_reconstruction)
    monkeypatch.setattr(nonfinite.sim_obj, "design_matrix", _refuse_reconstruction)
    assert float(shifted()) < float(plain())
    assert float(nonfinite()) == -1.0e10


@pytest.mark.unit
def test_pointer_with_fixed_children_reuses_the_kernel(monkeypatch) -> None:
    """A pointer is fixed when every parameter it points at is fixed."""
    source = _regularized_source("gaussian", kernel_dynamic=False)
    scale_driver = ParamU("driver", 0.25, prior_type="uniform", prior_settings=[0.05, 1.0], limits=[0.01, 5.0])
    scale_driver.to_static()
    source.kernel_scale = ck.Param("kernel_scale", lambda _: scale_driver.value, link=(scale_driver,))
    model = _dense_model(source=source, dynamic_mass=False, lens_mass=[_static_sie()])
    matrix, log_covariance = model._regularization_matrix(model._fixed_source_bbox)
    assert log_covariance is not None
    monkeypatch.setattr(model.reg_builder, "matrix", _refuse_reconstruction)
    hit_matrix, hit_logdet = model._regularization_matrix(model.infer_source_bbox())
    assert hit_matrix is matrix
    np.testing.assert_allclose(hit_logdet, log_covariance, rtol=1e-6, atol=1e-6)
    assert np.isfinite(float(model()))


if __name__ == "__main__":
    pytest.main()
