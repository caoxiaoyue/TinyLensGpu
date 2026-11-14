"""Operator imaging and inversion with an opt-in Hermite source."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.signal import convolve2d

from TinyLensGpu.ForwardSimulation import SimulatorConfig
from TinyLensGpu.ForwardSimulation.LensImage.pixelized import PixelizedLensSimulator
from TinyLensGpu.ForwardSimulation.LensImage.pixelized_operator import (
    PixelizedLensOperator,
)
from TinyLensGpu.ObservationModel import PixelizedImageProbModel
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import (
    PixelizedImageProbModelOperator,
)
from TinyLensGpu.PhysicalModel import PhysicalModel, PixelizedSourceModel, SersicEllipse


@pytest.fixture
def imaging():
    with jax.experimental.enable_x64():
        size, dpix = 13, 0.1
        mask = np.zeros((size, size), dtype=bool)
        mask[2, 3] = mask[10, 9] = True
        seed_mask = np.ones_like(mask)
        seed_mask[3:10, 3:10] = False
        psf = np.array([[0, 0.04, 0], [0.09, 0.65, 0.12], [0.06, 0.04, 0]])
        config = SimulatorConfig(
            dpix, size, psf, nsub=2, mask=mask, source_seed_mask=seed_mask,
            source_bbox_padding=0.0,
        )
        # Fixed-bbox APIs preserve explicitly typed float64 bounds.
        lower, upper = config.xgrid[0, jnp.array([3, 9])].astype(jnp.float64)
        yield config, (lower, upper, lower, upper)


@pytest.mark.integration
@pytest.mark.parametrize("interpolation", ["bilinear", "cubic_hermite"])
def test_operator_forward_and_adjoint_match_dense(imaging, interpolation):
    config, bbox = imaging
    source_model = PixelizedSourceModel(4, interpolation=interpolation)
    physical = PhysicalModel(lens_mass=[], source_light=[source_model], lens_light=[])
    dense = PixelizedLensSimulator(physical, config)
    operator = PixelizedLensOperator(physical, config)
    design, _ = dense.design_matrix(source_bbox=bbox)
    source = jnp.linspace(0.2, 1.0, 16)
    op_data = operator.precompute_operator_data(*bbox)

    image = operator.forward_model(source, *bbox, op_data=op_data)
    np.testing.assert_allclose(image, design @ source, rtol=1e-12, atol=1e-12)
    if interpolation == "cubic_hermite":
        # All fitted nodes are positive, but the exterior lobe is signed.
        assert jnp.min(image) < -0.02

    probe = jnp.sin(jnp.arange(image.size))
    adjoint = operator.build_rhs(probe, jnp.ones_like(probe), *bbox, op_data=op_data)
    np.testing.assert_allclose(adjoint, design.T @ probe, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(probe @ image, adjoint @ source, rtol=1e-12, atol=1e-12)


@pytest.mark.integration
def test_hermite_preconditioner_data_blocks_match_native_curvature(imaging):
    config, bbox = imaging
    source_model = PixelizedSourceModel(4, interpolation="cubic_hermite")
    source_model.log_lambda_reg.to_static(0.0)
    physical = PhysicalModel(lens_mass=[], source_light=[source_model], lens_light=[])
    operator = PixelizedLensOperator(physical, config)
    mask = np.asarray(config.mask)
    noise = np.linspace(0.03, 0.07, mask.size).reshape(mask.shape)
    weighted = np.where(mask, 0.0, noise**-2)
    psf_squared = np.asarray(config.psf_kernel)[::-1, ::-1] ** 2
    effective_weights = convolve2d(weighted, psf_squared, mode="same").ravel()
    native = np.stack([
        np.asarray(source_model.light(config.xgrid, config.ygrid, column, bbox)).ravel()
        for column in np.eye(4**2)
    ], axis=1)
    curvature = native.T @ (effective_weights[:, None] * native)
    blocks, pixels = operator.build_data_blocks(noise[~mask], *bbox, block_size=2)
    for block, indices in zip(blocks, pixels):
        np.testing.assert_allclose(
            block, curvature[np.ix_(indices, indices)], rtol=1e-12, atol=1e-12
        )


@pytest.mark.integration
@pytest.mark.parametrize("joint", [False, True], ids=["source_only", "joint"])
@pytest.mark.parametrize("solvers", [("cholesky", "pcg"), ("nnls", "pnpg")])
def test_hermite_operator_inversion_matches_dense(imaging, solvers, joint):
    config, bbox = imaging
    dense_solver, operator_solver = solvers
    n, source_strength, lens_strength = 4, 0.5, 0.01
    source_model = PixelizedSourceModel(
        n,
        interpolation="cubic_hermite",
        regularization_type="zero-order",
        log_lambda_reg=np.log(source_strength),
    )
    source_model.log_lambda_reg.to_static()
    lens_light = []
    if joint:
        lens = SersicEllipse(
            R_sersic=0.17, n_sersic=1.0, Ie=1.0, e1=0.05, e2=-0.02,
            center_x=0.47, center_y=0.04,
        )
        lens.Ie.to_static(1.0)
        lens_light.append(lens)
    physical = PhysicalModel(lens_mass=[], source_light=[source_model], lens_light=lens_light)
    simulator = PixelizedLensSimulator(physical, config)
    design, _ = simulator.design_matrix(source_bbox=bbox)
    design = np.asarray(design)
    truth = np.linspace(-0.3, 1.1, n * n)
    prior = np.full(n * n, source_strength)
    if joint:
        truth = np.append(truth, 0.4)
        prior = np.append(prior, lens_strength)
    # Some input nodes are negative so the constrained fit exercises an active set.
    data = design @ truth
    active = ~np.asarray(config.mask)
    image = np.zeros(active.shape)
    image[active] = data
    noise = np.linspace(0.03, 0.07, image.size).reshape(image.shape)
    weight = noise[active] ** -2
    options = dict(
        mask=config.mask, source_seed_mask=config.source_seed_mask,
        nsub=config.nsub, source_bbox_padding=0.0,
        lens_light_regularization=lens_strength,
    )
    model_args = (image, noise, config.psf_kernel, config.dpix, physical)
    dense = PixelizedImageProbModel(
        *model_args, fixed_source_bbox=bbox, solver_type=dense_solver, **options
    ).evaluate()
    np.testing.assert_array_equal(dense.source_bbox, bbox)

    def coefficients(fit):
        if joint:
            return np.concatenate((fit.source_pixels, fit.lens_amplitudes))
        return np.asarray(fit.source_pixels)

    def objective(values):
        residual = design @ values - data
        return 0.5 * (weight @ residual**2 + prior @ values**2)

    evidence = []
    for fixed_bbox in (None, bbox):
        operator = PixelizedImageProbModelOperator(
            *model_args,
            fixed_source_bbox=fixed_bbox,
            block_size=2,
            solver_type=operator_solver,
            pnpg_rtol=1e-9,
            **options,
        )
        operator.pcg_rtol = 1e-11
        result = operator.evaluate()
        assert result.solver_info.converged and not result.solver_info.failed
        np.testing.assert_array_equal(result.source_bbox, bbox)
        np.testing.assert_allclose(result.model_image, dense.model_image, rtol=0, atol=2e-8)
        values = coefficients(result)
        np.testing.assert_allclose(values, coefficients(dense), rtol=0, atol=2e-8)
        np.testing.assert_allclose(
            objective(values), objective(coefficients(dense)), rtol=5e-9, atol=1e-10
        )
        evidence.append(float(result.log_evidence))

        # Evaluate the true residual of the full coupled system, including
        # both regularization terms and the PSF/subpixel data curvature.
        gradient = design.T @ (weight * (design @ values - data)) + prior * values
        rhs_scale = np.linalg.norm(design.T @ (weight * data), ord=np.inf)
        if operator_solver == "pcg":
            assert np.linalg.norm(gradient, ord=np.inf) < 2e-11 * rhs_scale
        else:
            assert np.all(values >= 0.0) and np.any(values == 0.0)
            assert np.min(gradient) >= -2e-9 * rhs_scale
            assert np.linalg.norm(values * gradient, ord=np.inf) < 2e-9 * rhs_scale

    # The cached and uncached paths should also use identical determinant blocks.
    np.testing.assert_allclose(evidence[0], evidence[1], rtol=0, atol=1e-10)


@pytest.mark.integration
def test_saved_node_intensities_can_initialize_a_new_operator_fit(imaging):
    config, bbox = imaging
    strength, sigma = 0.5, 0.05
    source = PixelizedSourceModel(
        4, log_lambda_reg=np.log(strength), regularization_type="zero-order",
        interpolation="cubic_hermite",
    )
    source.log_lambda_reg.to_static()
    physical = PhysicalModel(lens_mass=[], source_light=[source], lens_light=[])
    simulator = PixelizedLensSimulator(physical, config)
    design, _ = simulator.design_matrix(source_bbox=bbox)
    design = np.asarray(design)
    saved_nodes = np.linspace(-0.2, 1.0, 16)
    # Choose a data vector whose regularized solution is the saved source.
    offset = np.linalg.solve(design.T @ design, strength * sigma**2 * saved_nodes)
    image = np.zeros(config.mask.shape)
    image[~np.asarray(config.mask)] = design @ (saved_nodes + offset)
    options = dict(mask=config.mask, nsub=config.nsub, fixed_source_bbox=bbox)
    args = (image, np.full_like(image, sigma), config.psf_kernel, config.dpix, physical)
    cold = PixelizedImageProbModelOperator(*args, **options)
    warm = PixelizedImageProbModelOperator(*args, initial_source=saved_nodes, **options)
    # A zero iteration budget can only converge if the supplied seed is used.
    cold.pcg_max_iter = warm.pcg_max_iter = 0
    assert not cold.evaluate().solver_info.converged
    result = warm.evaluate()
    assert result.solver_info.converged
    np.testing.assert_array_equal(result.source_pixels, saved_nodes)


@pytest.mark.integration
@pytest.mark.parametrize("joint", [False, True], ids=["source", "joint"])
def test_direct_curvature_uses_changed_noise_and_prior_data(imaging, joint):
    """Internal JIT must keep numerical data dynamic at a fixed topology."""
    from TinyLensGpu.utils.inversion.regularization import FiniteDifferenceRegularization

    config, bbox = imaging
    n = 4
    source = PixelizedSourceModel(n, regularization_type="zero-order", interpolation="cubic_hermite")
    source.log_lambda_reg.to_static(0.0)
    lens = SersicEllipse(R_sersic=0.2, n_sersic=1.0, Ie=1.0,
                         e1=0.05, e2=0.0, center_x=0.03, center_y=0.0)
    lens.Ie.to_static(1.0)
    physical = PhysicalModel(lens_mass=[], source_light=[source], lens_light=[lens] if joint else [])
    dense = PixelizedLensSimulator(physical, config)
    simulator = PixelizedLensOperator(physical, config)
    design, _ = dense.design_matrix(source_bbox=bbox)
    mapping = simulator.precompute_operator_data(*bbox)
    assert mapping.n_render == simulator.psf_response.n_render
    assert mapping.n_render > simulator.n_active
    regularizer = FiniteDifferenceRegularization(n, "zero-order")
    reg_data = regularizer.make_reg_data(*bbox)
    vector = jnp.linspace(-0.3, 0.8, n**2 + int(joint), dtype=jnp.float64)
    for scale in (1.0, 1.7):
        noise = scale * jnp.linspace(0.03, 0.07, simulator.n_active, dtype=jnp.float64)
        strength, lens_strength = 0.3 * scale, 0.02 * scale
        if joint:
            operator, _, _ = simulator.build_joint_system(
                noise, jnp.ones_like(noise), *bbox, strength, reg_data, lens_strength,
                op_data=mapping,
            )
            prior = jnp.r_[jnp.full(n**2, strength), lens_strength]
        else:
            operator = simulator.build_source_curvature_operator(
                noise, *bbox, strength, reg_data, op_data=mapping,
            )
            prior = jnp.full(n**2, strength)
        expected = design.T @ ((design @ vector) / noise**2) + prior * vector
        np.testing.assert_allclose(operator.matvec(vector), expected, rtol=1e-12, atol=1e-10)
