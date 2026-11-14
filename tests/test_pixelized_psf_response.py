"""Public PSF response operations preserve fitted/rendered pixel domains."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.signal import convolve2d

from TinyLensGpu.ForwardSimulation import SimulatorConfig
from TinyLensGpu.ForwardSimulation.LensImage._pixelized_psf import PixelizedPSFResponse


def _weights(x, y):
    return jnp.stack((1 + x, y + 0.25, x * y, x**2 - 0.2), axis=-1)


def _config(nsub, kernel):
    mask = np.ones((7, 7), dtype=bool)
    mask[0, 0] = mask[-1, -1] = False
    mask[2:5, 2:5] = False
    mask[3, 3] = True
    return SimulatorConfig(0.125, 7, kernel, nsub=nsub, mask=mask)


@pytest.mark.unit
@pytest.mark.parametrize("nsub", [1, 2, 3])
def test_fit_stencil_and_native_values_follow_only_fitted_rays(nsub):
    with jax.experimental.enable_x64():
        config = _config(nsub, jnp.full((3, 3), 1 / 9))
        response = PixelizedPSFResponse(config.mask, config.psf_kernel, nsub)
        x, y = response.ray_coordinates(config)
        weights = _weights(x, y)
        indices = jnp.broadcast_to(jnp.arange(4), weights.shape)
        fit_weights, fit_indices = response.fit_stencil(weights, indices)
        expected = _weights(config.xgrid_sub_1d, config.ygrid_sub_1d)
        np.testing.assert_array_equal(fit_weights, expected)
        np.testing.assert_array_equal(
            fit_indices, jnp.broadcast_to(jnp.arange(4), expected.shape)
        )

        mask = np.asarray(config.mask)
        values = np.arange(np.count_nonzero(~mask), dtype=np.float64) + 0.5
        native = np.full(mask.shape, np.nan)
        native[~mask] = values
        expanded = np.repeat(np.repeat(native, nsub, axis=0), nsub, axis=1)
        submask = np.repeat(np.repeat(mask, nsub, axis=0), nsub, axis=1)
        np.testing.assert_array_equal(
            response.expand_fit_values(values), expanded[~submask]
        )
        np.testing.assert_array_equal(
            response.integrate_fit_rays(expanded[~submask]), values
        )
        integrated_weights = response.integrate_fit_rays(fit_weights)
        full_weights = np.asarray(_weights(config.xgrid_sub, config.ygrid_sub))
        expected_native = full_weights.reshape(7, nsub, 7, nsub, 4).mean(axis=(1, 3))
        np.testing.assert_allclose(
            integrated_weights, expected_native[~mask], atol=1e-14
        )
        assert response.n_fit == values.size
        assert response.n_render > response.n_fit
        with pytest.raises(ValueError, match="Fitted values must have shape"):
            response.expand_fit_values(values[:-1])
        with pytest.raises(ValueError, match="render subpixels"):
            response.fit_stencil(weights[:-1], indices[:-1])
        with pytest.raises(ValueError, match="fitted subpixel rays"):
            response.integrate_fit_rays(fit_weights[:-1])


@pytest.mark.integration
@pytest.mark.parametrize("nsub", [1, 2, 3])
@pytest.mark.parametrize(
    "kernel",
    [
        np.array([[0.0, 0.1, 0.0], [0.2, 0.4, 0.1], [0.05, 0.15, 0.0]]),
        np.array([[0.1, 0.0, 0.2], [0.4, 0.05, 0.25]]),
    ],
    ids=["odd", "even-asymmetric"],
)
def test_full_and_bounded_columns_match_independent_full_cutout_convolution(
    nsub, kernel
):
    with jax.experimental.enable_x64():
        config = _config(nsub, kernel)
        response = PixelizedPSFResponse(config.mask, config.psf_kernel, nsub)
        x, y = response.ray_coordinates(config)
        weights = _weights(x, y)
        indices = jnp.broadcast_to(jnp.arange(4), weights.shape)
        actual = jax.jit(lambda w: response.convolved_columns(w, indices, 4))(weights)

        full_weights = np.asarray(_weights(config.xgrid_sub, config.ygrid_sub))
        native = full_weights.reshape(7, nsub, 7, nsub, 4).mean(axis=(1, 3))
        expected = np.stack(
            [
                convolve2d(native[..., column], kernel, mode="same")[
                    ~np.asarray(config.mask)
                ]
                for column in range(4)
            ]
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)

        # Deliberately permute the selected source columns. All other nodes use
        # the out-of-window sentinel and must be dropped, never clipped/aliased.
        local_indices = jnp.array([2, 1, 2, 0])[indices]
        bounded = jax.jit(lambda w: response.convolved_columns(w, local_indices, 2))(
            weights
        )
        np.testing.assert_allclose(bounded, expected[[3, 1]], rtol=1e-13, atol=1e-14)
        with pytest.raises(ValueError, match="render subpixels"):
            response.convolved_columns(weights[:-1], indices[:-1], 4)


@pytest.mark.integration
@pytest.mark.parametrize("nsub", [1, 2])
@pytest.mark.parametrize("provide_source", [False, True])
@pytest.mark.parametrize("return_mapping", [False, True])
def test_dense_forward_preserves_native_fitted_mapping(
    nsub, provide_source, return_mapping
):
    from TinyLensGpu.ForwardSimulation.LensImage.pixelized import PixelizedLensSimulator
    from TinyLensGpu.PhysicalModel import PhysicalModel, PixelizedSourceModel

    with jax.experimental.enable_x64():
        kernel = np.array([[0.02, 0.1, 0.03], [0.2, 0.4, 0.1], [0.05, 0.1, 0.0]])
        config = _config(nsub, kernel)
        source = PixelizedSourceModel(4)
        source.log_lambda_reg.to_static(0.0)
        physical = PhysicalModel(lens_mass=[], source_light=[source])
        simulator = PixelizedLensSimulator(physical, config)
        values = jnp.linspace(-0.2, 1.0, 16) if provide_source else None
        result = simulator.forward(source_pixels=values, return_mapping=return_mapping)
        bbox = simulator.infer_source_bbox(*simulator.ray_trace_seed())
        full = np.stack(
            [
                np.asarray(
                    source.light(config.xgrid_sub, config.ygrid_sub, basis, bbox)
                )
                for basis in np.eye(16)
            ]
        )
        native = full.reshape(16, 7, nsub, 7, nsub).mean(axis=(2, 4))
        mask = np.asarray(config.mask)
        if return_mapping or not provide_source:
            np.testing.assert_allclose(
                result.mapping_matrix, native[:, ~mask].T, atol=1e-14
            )
        else:
            assert result.mapping_matrix is None
        expected = np.zeros((7, 7))
        if provide_source:
            ideal = np.einsum("kij,k->ij", native, np.asarray(values))
            expected[~mask] = convolve2d(ideal, kernel, mode="same")[~mask]
        np.testing.assert_allclose(result.model_image, expected, rtol=1e-13, atol=1e-14)
