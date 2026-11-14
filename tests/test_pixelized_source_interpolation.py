"""Interpolation checks at the mapping, brightness, and dense imaging seams."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from TinyLensGpu.utils.lensing import mapping


@pytest.mark.unit
def test_bilinear_dispatch_preserves_hard_box_and_upper_edge():
    bbox = (0.0, 1.0, 0.0, 1.0)
    rays = jnp.array([[1.0, 0.25], [1.01, 0.25], [-0.01, 0.25], [0.5, 0.75]])
    values = jnp.array([1.0, 2.0, 3.0, 4.0])
    axis = jnp.array([0.0, 1.0])
    explicit = mapping.build_lens_mapping_matrix(
        rays[:, 0], rays[:, 1], axis, axis, interpolation="bilinear"
    )
    default = mapping.build_lens_mapping_matrix(rays[:, 0], rays[:, 1], axis, axis)
    weights, indices, support = mapping.lens_mapping_operator_bilinear_from(
        rays, *bbox, 2
    )
    np.testing.assert_array_equal(default, explicit)
    np.testing.assert_array_equal(explicit @ values, [2.5, 0.0, 0.0, 3.0])
    np.testing.assert_array_equal(jnp.sum(weights * values[indices], axis=1), default @ values)
    np.testing.assert_array_equal(indices[0], [1, 1, 3, 3])
    np.testing.assert_array_equal(support, [True, False, False, True])
    gx, gy = mapping.source_spatial_gradient(
        values, rays, *bbox, 2, interpolation="bilinear"
    )
    np.testing.assert_array_equal(gx, [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(gy, [2.0, 0.0, 0.0, 2.0])


def _zero_extended_reference(values, rays, bbox):
    """Evaluate the Hermite endpoint-value/slope form on an explicit zero grid."""
    n = values.shape[0]
    padded = np.pad(values, 3)
    xmin, xmax, ymin, ymax = bbox
    origin = np.array([xmin, ymin])
    step = (np.array([xmax, ymax]) - origin) / (n - 1)

    def interpolate(window, t):
        left, right = window[..., 1], window[..., 2]
        left_slope = (right - window[..., 0]) / 2
        right_slope = (window[..., 3] - left) / 2
        return (
            (1 + 2 * t) * (1 - t) ** 2 * left
            + t * (1 - t) ** 2 * left_slope
            + t**2 * (3 - 2 * t) * right
            - t**2 * (1 - t) * right_slope
        )

    result = np.zeros(len(rays))
    for i, (u, v) in enumerate((rays - origin) / step):
        if -2 < u < n + 1 and -2 < v < n + 1:
            x, y = np.floor([u, v]).astype(int)
            window = padded[y + 2:y + 6, x + 2:x + 6]
            result[i] = interpolate(interpolate(window, u - x), v - y)
    return result


@pytest.mark.unit
@pytest.mark.parametrize("n", [2, 5])
def test_hermite_mapping_matches_zero_extended_original_node_grid(n):
    bbox = (-1.0, 1.0, 0.5, 2.5)
    x_axis = np.linspace(*bbox[:2], n)
    y_axis = np.linspace(*bbox[2:], n)
    spacing = (x_axis[-1] - x_axis[0]) / (n - 1)
    nodes = np.stack(np.meshgrid(x_axis, y_axis), axis=-1).reshape(-1, 2)
    probes = np.array([
        [x_axis[0] + 0.25 * spacing, y_axis[0] + 0.6 * spacing],
        [x_axis[0] - 0.5 * spacing, y_axis[0]],
        [x_axis[0] - 1.5 * spacing, y_axis[0]],
        [x_axis[-1], y_axis[-1] + 1.5 * spacing],
        [x_axis[0] - 2 * spacing, y_axis[0]],
        [x_axis[0] - 2.1 * spacing, y_axis[0]],
    ])
    rays = np.concatenate((nodes, probes))
    values = np.arange(1, n * n + 1, dtype=np.float64).reshape(n, n)
    with jax.experimental.enable_x64():
        weights, indices, support = mapping.lens_mapping_operator_from(
            rays, *bbox, n, interpolation="cubic_hermite"
        )
        dense = mapping.build_lens_mapping_matrix(
            rays[:, 0], rays[:, 1], x_axis, y_axis, interpolation="cubic_hermite"
        )
        brightness = dense @ values.ravel()

    assert weights.shape == indices.shape == (len(rays), 16)
    assert np.min(indices) >= 0 and np.max(indices) < n * n
    assert dense.shape == (len(rays), n * n)
    np.testing.assert_array_equal(dense[:n * n], np.eye(n * n))
    np.testing.assert_array_equal(support, np.arange(len(rays)) < len(rays) - 1)
    assert brightness[-5] > 0
    assert brightness[-4] < 0 and brightness[-3] < 0
    np.testing.assert_allclose(
        np.sum(np.asarray(weights) * values.ravel()[indices], axis=1), brightness,
        rtol=2e-14, atol=2e-14,
    )
    np.testing.assert_allclose(
        brightness, _zero_extended_reference(values, rays, bbox),
        rtol=2e-14, atol=2e-14,
    )


@pytest.mark.unit
@pytest.mark.parametrize("axis", [0, 1], ids=["x", "y"])
def test_hermite_brightness_and_gradient_are_continuous_at_grid_lines(axis):
    n = 5
    bbox = (-0.625, 1.125, 0.1875, 1.9375)
    values = np.linspace(1.0, 3.0, n * n)
    origin = np.array([bbox[0], bbox[2]])
    spacing = (bbox[1] - bbox[0]) / (n - 1)
    knots = np.tile(origin + 1.43 * spacing, (n + 4, 1))
    knots[:, axis] = origin[axis] + spacing * np.arange(-2, n + 2)
    offsets = np.eye(2)[axis] * np.array([-2e-8, 0.0, 2e-8])[:, None]
    probes = (knots[:, None] + offsets[None, :]).reshape(-1, 2)

    with jax.experimental.enable_x64():
        rays = jnp.asarray(probes)

        def stencil(xy):
            return mapping.lens_mapping_operator_from(
                xy, *bbox, n, interpolation="cubic_hermite"
            )

        def brightness(xy):
            weights, indices, _ = stencil(xy)
            return jnp.sum(weights * jnp.asarray(values)[indices], axis=1)

        predicted = brightness(rays).reshape(-1, 3)
        analytic = jnp.stack(mapping.source_spatial_gradient(
            values, rays, *bbox, n, interpolation="cubic_hermite"
        ), axis=1)
        autodiff = jax.grad(lambda xy: brightness(xy).sum())(rays)
        h = 1e-9  # Smaller than a float32 ULP at these coordinates.
        finite = jnp.stack([
            (brightness(rays + delta) - brightness(rays - delta)) / (2 * h)
            for delta in h * jnp.eye(2)
        ], axis=1)
        weights, _, support = stencil(rays)

    assert predicted.dtype == analytic.dtype == jnp.float64
    np.testing.assert_allclose(analytic, autodiff, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(analytic, finite, rtol=2e-6, atol=2e-6)
    for neighbor in (0, 2):
        np.testing.assert_allclose(predicted[:, neighbor], predicted[:, 1], atol=5e-7)
        np.testing.assert_allclose(
            analytic[neighbor::3], analytic[1::3], rtol=0, atol=2e-6
        )

    # A virtual node line is dark but has a nonzero normal derivative.
    for row in (4, len(rays) - 5):
        assert support[row]
        np.testing.assert_array_equal(weights[row], 0.0)
        assert abs(analytic[row, axis]) > 0.5
        assert analytic[row, 1 - axis] == 0.0

    # Outside and on both final support edges, neither value nor slope survives.
    edge_rows = np.array([0, 1, len(rays) - 2, len(rays) - 1])
    np.testing.assert_array_equal(support[edge_rows], [False, True, True, False])
    np.testing.assert_array_equal(weights[edge_rows], 0.0)
    np.testing.assert_array_equal(analytic[edge_rows], 0.0)


@pytest.mark.unit
@pytest.mark.parametrize("reflect", [False, True], ids=["lower_edge", "upper_edge"])
def test_hermite_support_edges_keep_float64_spacing_under_jit(reflect):
    lower, upper = -0.022404768054233637, 0.3577528867209083
    if reflect:
        lower, upper = -upper, -lower
    n = 11
    spacing = (upper - lower) / (n - 1)
    edges = np.array([lower - 2 * spacing, upper + 2 * spacing])
    probes = np.column_stack((
        np.nextafter(edges, -np.inf), edges, np.nextafter(edges, np.inf)
    )).ravel()
    rays = np.column_stack((probes, np.full_like(probes, (lower + upper) / 2)))
    bbox = (lower, upper, lower, upper)
    expected_support = [False, True, True, True, True, False]
    outside_or_edge = np.array([0, 1, 4, 5])

    with jax.experimental.enable_x64():
        weights, _, support = mapping.lens_mapping_operator_from(
            rays, *bbox, n, interpolation="cubic_hermite"
        )
        gx, gy = mapping.source_spatial_gradient(
            np.ones(n**2), rays, *bbox, n, interpolation="cubic_hermite"
        )
        with jax.disable_jit():
            _, _, eager_support = mapping.lens_mapping_operator_from(
                rays, *bbox, n, interpolation="cubic_hermite"
            )

    np.testing.assert_array_equal(support, expected_support)
    np.testing.assert_array_equal(eager_support, expected_support)
    np.testing.assert_array_equal(weights[outside_or_edge], 0.0)
    np.testing.assert_array_equal(gx[outside_or_edge], 0.0)
    np.testing.assert_array_equal(gy[outside_or_edge], 0.0)


@pytest.mark.unit
def test_source_light_interpolation_is_fixed_at_construction():
    from TinyLensGpu.PhysicalModel import PixelizedSourceModel

    with pytest.raises(ValueError, match="source interpolation"):
        PixelizedSourceModel(2, interpolation="cubic")

    default = PixelizedSourceModel(2, log_lambda_reg=0.0)
    bilinear = PixelizedSourceModel(2, log_lambda_reg=0.0, interpolation="bilinear")
    hermite = PixelizedSourceModel(2, log_lambda_reg=0.0, interpolation="cubic_hermite")
    assert default.interpolation == bilinear.interpolation == "bilinear"
    with pytest.raises(AttributeError):
        hermite.interpolation = "bilinear"

    bbox = (0.0, 1.0, 0.0, 1.0)
    x = jnp.array([1.0, 1.5, -1.5])
    y = jnp.zeros_like(x)
    values = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(default.light(x, y, values, bbox), [2.0, 0.0, 0.0])
    np.testing.assert_array_equal(
        default.light(x, y, values, bbox), bilinear.light(x, y, values, bbox)
    )
    np.testing.assert_allclose(
        hermite.light(x, y, values, bbox),
        _zero_extended_reference(np.asarray(values), np.column_stack((x, y)), bbox),
        atol=1e-7,
    )


def test_dense_jitted_psf_override_can_expand_the_render_footprint():
    from TinyLensGpu.ForwardSimulation import SimulatorConfig
    from TinyLensGpu.ForwardSimulation.LensImage.pixelized import PixelizedLensSimulator
    from TinyLensGpu.PhysicalModel import PhysicalModel, PixelizedSourceModel

    with jax.experimental.enable_x64():
        size = 5
        mask = np.ones((size, size), dtype=bool)
        mask[2, 1] = False
        config = SimulatorConfig(0.125, size, jnp.ones((1, 1)), mask=mask)
        physical = PhysicalModel(lens_mass=[], source_light=[PixelizedSourceModel(size)])
        dense = PixelizedLensSimulator(physical, config)
        bbox = (-0.25, 0.25, -0.25, 0.25)
        source = jnp.zeros((size, size)).at[2, 2].set(1.0).ravel()

        def predict(kernel):
            matrix, _ = dense.design_matrix(source_bbox=bbox, psf_kernel=kernel)
            return (matrix @ source)[0]

        psf = jnp.array([[0.2, 0.7, 0.1]])
        value, gradient = jax.jit(jax.value_and_grad(predict))(psf)
        image = dense.simulate(source, source_bbox=bbox, psf_kernel=psf)

    np.testing.assert_allclose(value, 0.2, atol=2e-15)
    np.testing.assert_allclose(image[2, 1], 0.2, atol=2e-15)
    np.testing.assert_allclose(gradient, [[1, 0, 0]], atol=2e-15)


@pytest.mark.integration
@pytest.mark.parametrize("interpolation", [None, "bilinear", "cubic_hermite"])
def test_dense_reconstruction_matches_masked_subpixel_brightness(interpolation):
    from scipy.signal import convolve2d

    from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
    from TinyLensGpu.ObservationModel import PixelizedImageProbModel
    from TinyLensGpu.PhysicalModel import PhysicalModel, PixelizedSourceModel

    size, n, nsub = 7, 3, 2
    dpix, noise, strength = 0.2, 0.04, 0.15
    mask = np.zeros((size, size), dtype=bool)
    mask[1, 3] = mask[5, 2] = True
    seed_mask = np.ones_like(mask)
    seed_mask[2:5, 2:5] = False
    psf = np.array([[0, 0.04, 0], [0.09, 0.65, 0.12], [0.06, 0.04, 0]])

    with jax.experimental.enable_x64():
        bbox = tuple(jnp.asarray(v, dtype=jnp.float64) for v in (-dpix, dpix, -dpix, dpix))
        mode = {} if interpolation is None else {"interpolation": interpolation}
        source = PixelizedSourceModel(
            n, log_lambda_reg=np.log(strength), regularization_type="zero-order",
            **mode,
        )
        source.log_lambda_reg.to_static()
        physical = PhysicalModel(lens_mass=[], source_light=[source], lens_light=[])
        config = SimulatorConfig(dpix, size, psf, nsub, mask, seed_mask)
        basis = np.eye(n * n)
        direct = np.stack([
            source.light(config.xgrid_sub, config.ygrid_sub, values, bbox) for values in basis
        ])
        native = direct.reshape(n * n, size, nsub, size, nsub).mean(axis=(2, 4))
        expected_design = np.stack(
            [convolve2d(column, psf, mode="same")[~mask] for column in native], axis=1,
        )
        image_data = np.zeros_like(mask, dtype=np.float64)
        image_data[~mask] = expected_design @ np.linspace(0.2, 1.0, n * n)
        reference_source = np.linalg.solve(
            expected_design.T @ expected_design + noise**2 * strength * basis,
            expected_design.T @ image_data[~mask],
        )

        for fixed_bbox in (None, bbox):
            model = PixelizedImageProbModel(
                image_data, np.full_like(image_data, noise), psf, dpix, physical,
                mask=mask, source_seed_mask=seed_mask, nsub=nsub,
                source_bbox_padding=0.0, fixed_source_bbox=fixed_bbox,
                solver_type="cholesky",
            )
            result = model.evaluate()
            subgrid_matrix = model.sim_obj.build_mapping_matrix(
                source_bbox=bbox, use_subgrid=True
            )
            np.testing.assert_allclose(
                subgrid_matrix, direct[:, config.unmask_sub].T, rtol=1e-12, atol=1e-12
            )
            np.testing.assert_array_equal(result.source_bbox, bbox)
            np.testing.assert_allclose(result.source_pixels, reference_source, atol=1e-11)
            np.testing.assert_allclose(
                result.model_image[~mask], expected_design @ reference_source, atol=1e-11
            )
            np.testing.assert_array_equal(result.model_image[mask], 0.0)
            assert result.source_pixels.dtype == result.model_image.dtype == jnp.float64
