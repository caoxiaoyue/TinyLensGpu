"""Operator backend tests against explicit matrix construction."""

import numpy as np
import pytest
import jax.numpy as jnp
from numpy.testing import assert_allclose

from TinyLensGpu.utils.inversion.operator_solver import (
    _apply_mapping,
    _apply_mapping_transpose,
    _apply_psf_unmasked_to_unmasked,
)
from TinyLensGpu.utils.interpolation.kernels import get_interpolation_weights
from TinyLensGpu.utils.lensing import lens_mapping_matrix_from, apply_psf_to_mapping_matrix


@pytest.mark.unit
def test_operator_forward_equals_dense_bf():
    np.random.seed(10)
    image_shape = (24, 24)
    h, w = image_shape

    y, x = np.mgrid[:h, :w]
    r = np.sqrt((y - h // 2) ** 2 + (x - w // 2) ** 2)
    mask = r > 9
    unmasked = np.where(~mask)
    unmasked_indices = (jnp.array(unmasked[0]), jnp.array(unmasked[1]))

    psf = np.zeros((7, 7), dtype=np.float32)
    psf[3, 3] = 1.0
    from scipy.ndimage import gaussian_filter
    psf = gaussian_filter(psf, 1.2)
    psf /= psf.sum()
    psf_shape = psf.shape
    fft_shape = (h + psf_shape[0] - 1, w + psf_shape[1] - 1)
    psf_fft = jnp.fft.rfft2(psf, s=fft_shape)

    n_source = 60
    n_data = int(np.sum(~mask))
    source_mesh = jnp.array(np.random.randn(n_source, 2).astype(np.float32))
    data_mesh = jnp.array(np.random.randn(n_data, 2).astype(np.float32))

    weights, indices, _ = get_interpolation_weights(
        points=source_mesh,
        query_points=data_mesh,
        k_neighbors=5,
        kernel='wendland_c4',
        radius_scale=1.5,
    )
    f = lens_mapping_matrix_from(
        source_mesh_beta=source_mesh,
        data_mesh_beta=data_mesh,
        k_neighbors=5,
        kernel='wendland_c4',
        radius_scale=1.5,
    )
    bf = apply_psf_to_mapping_matrix(
        mapping_matrix=f,
        psf_kernel=jnp.array(psf),
        image_shape=image_shape,
        unmasked_indices=unmasked_indices,
        method='fft',
        psf_fft=psf_fft,
        psf_shape=psf_shape,
    )

    x_src = jnp.array(np.random.randn(n_source).astype(np.float32))
    y_op = _apply_psf_unmasked_to_unmasked(
        _apply_mapping(x_src, weights, indices),
        psf_fft,
        image_shape,
        psf_shape,
        unmasked_indices,
        adjoint=False,
    )
    y_dense = bf @ x_src
    assert_allclose(y_op, y_dense, rtol=1e-5, atol=1e-5)


@pytest.mark.unit
def test_operator_adjoint_inner_product_consistency():
    np.random.seed(11)
    image_shape = (22, 22)
    h, w = image_shape
    mask = np.zeros(image_shape, dtype=bool)
    mask[0, :] = True
    mask[:, 0] = True
    unmasked = np.where(~mask)
    unmasked_indices = (jnp.array(unmasked[0]), jnp.array(unmasked[1]))

    psf = np.random.rand(5, 5).astype(np.float32)
    psf /= psf.sum()
    psf_shape = psf.shape
    fft_shape = (h + psf_shape[0] - 1, w + psf_shape[1] - 1)
    psf_fft = jnp.fft.rfft2(psf, s=fft_shape)

    n_source = 45
    n_data = int(np.sum(~mask))
    source_mesh = jnp.array(np.random.randn(n_source, 2).astype(np.float32))
    data_mesh = jnp.array(np.random.randn(n_data, 2).astype(np.float32))
    weights, indices, _ = get_interpolation_weights(
        points=source_mesh,
        query_points=data_mesh,
        k_neighbors=5,
        kernel='wendland_c2',
        radius_scale=1.6,
    )

    x = jnp.array(np.random.randn(n_source).astype(np.float32))
    y = jnp.array(np.random.randn(n_data).astype(np.float32))

    ax = _apply_psf_unmasked_to_unmasked(
        _apply_mapping(x, weights, indices),
        psf_fft,
        image_shape,
        psf_shape,
        unmasked_indices,
        adjoint=False,
    )
    aty = _apply_mapping_transpose(
        _apply_psf_unmasked_to_unmasked(
            y,
            psf_fft,
            image_shape,
            psf_shape,
            unmasked_indices,
            adjoint=True,
        ),
        weights,
        indices,
        n_source,
    )

    lhs = float(np.asarray(jnp.dot(ax, y)))
    rhs = float(np.asarray(jnp.dot(x, aty)))
    assert abs(lhs - rhs) <= 1e-4 * max(1.0, abs(lhs), abs(rhs))

