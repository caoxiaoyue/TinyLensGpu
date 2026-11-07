"""Mapping utilities for lensing reconstruction."""

from functools import partial

import jax
import jax.numpy as jnp

_SOURCE_BOUNDARIES = {
    "bilinear": "hard_zero_outside_bbox",
    "cubic_hermite": "C1_cardinal_bicubic_fixed_zero_exterior_nodes",
}


def validate_source_interpolation(interpolation: str) -> str:
    """Validate the static source interpolation choice."""
    if not isinstance(interpolation, str) or interpolation not in _SOURCE_BOUNDARIES:
        raise ValueError(
            f"Unsupported source interpolation {interpolation!r}. "
            "Use 'bilinear' or 'cubic_hermite'."
        )
    return interpolation


def source_boundary_for_interpolation(interpolation: str) -> str:
    """Return the boundary convention for persisted source metadata."""
    return _SOURCE_BOUNDARIES[validate_source_interpolation(interpolation)]


@partial(jax.jit, static_argnames=("n_source",))
def dense_mapping_from_weights_indices(
    weights: jnp.ndarray,
    indices: jnp.ndarray,
    n_source: int,
) -> jnp.ndarray:
    """
    Construct a dense mapping matrix from sparse interpolation weights and indices.

    Parameters
    ----------
    weights : jnp.ndarray
        Interpolation weights array of shape ``(n_data, k_neighbors)``.
    indices : jnp.ndarray
        Neighbor indices array of shape ``(n_data, k_neighbors)``.
    n_source : int
        Total number of source points (columns in the output matrix).

    Returns
    -------
    jnp.ndarray
        Dense mapping matrix of shape ``(n_data, n_source)``.
    """
    n_data = int(weights.shape[0])
    n_neighbors = int(weights.shape[1])
    row_indices = jnp.repeat(jnp.arange(n_data, dtype=jnp.int32), n_neighbors)
    col_indices = indices.reshape(-1).astype(jnp.int32)
    weights_flat = weights.reshape(-1)
    mapping_matrix = jnp.zeros((n_data, n_source), dtype=weights_flat.dtype)
    return mapping_matrix.at[row_indices, col_indices].add(weights_flat)


def _bilinear_cells(data_mesh_beta, x_min, x_max, y_min, y_max, n):
    """Select the established cells, including duplicate upper-edge nodes."""
    # Keep coordinate precision: downcasting float64 rays quantizes small
    # parameter steps and makes finite differences disagree with autodiff.
    beta = jnp.asarray(data_mesh_beta)

    dx = (x_max - x_min) / (n - 1)
    dy = (y_max - y_min) / (n - 1)

    ux = (beta[:, 0] - x_min) / dx
    uy = (beta[:, 1] - y_min) / dy

    ix0 = jnp.floor(ux).astype(jnp.int32)
    iy0 = jnp.floor(uy).astype(jnp.int32)
    fx = ux - ix0
    fy = uy - iy0

    valid = (ux >= 0) & (ux <= n - 1) & (uy >= 0) & (uy <= n - 1)

    ix0_c = jnp.clip(ix0, 0, n - 1)
    iy0_c = jnp.clip(iy0, 0, n - 1)
    ix1_c = jnp.clip(ix0_c + 1, 0, n - 1)
    iy1_c = jnp.clip(iy0_c + 1, 0, n - 1)

    i00 = iy0_c * n + ix0_c
    i10 = iy0_c * n + ix1_c
    i01 = iy1_c * n + ix0_c
    i11 = iy1_c * n + ix1_c

    return fx, fy, (i00, i10, i01, i11), valid, dx, dy


@partial(jax.jit, static_argnames=("n",))
def lens_mapping_operator_bilinear_from(
    data_mesh_beta: jnp.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    n: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build bilinear interpolation weights/indices on a square source grid."""
    fx, fy, corners, valid, _, _ = _bilinear_cells(
        data_mesh_beta, x_min, x_max, y_min, y_max, n
    )

    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    weights = jnp.where(valid[:, None],
                        jnp.stack([w00, w10, w01, w11], axis=1), 0.0)
    indices = jnp.stack(corners, axis=1).astype(jnp.int32)
    return weights, indices, valid


def _hermite_axis(coordinates, lower, upper, n):
    """Choose the four centered-difference stencil nodes in one dimension."""
    step = (upper - lower) / (n - 1)
    support_lower, support_upper = lower - 2 * step, upper + 2 * step
    support = (coordinates >= support_lower) & (coordinates <= support_upper)
    # At the final edges both brightness and its gradient are exactly zero.
    interior = (coordinates > support_lower) & (coordinates < support_upper)
    u = (jnp.where(interior, coordinates, lower) - lower) / step
    cell = jnp.floor(u).astype(jnp.int32)
    indices = cell[:, None] + jnp.arange(-1, 3, dtype=jnp.int32)
    valid = interior[:, None] & (indices >= 0) & (indices < n)
    return u - cell, jnp.clip(indices, 0, n - 1), valid, support, step


def _hermite_weights(t):
    """Catmull–Rom weights and derivatives on a unit cell."""
    s = 1.0 - t
    weights = jnp.stack(
        (
            -0.5 * t * s**2,
            s * (1.0 + t - 1.5 * t**2),
            t * (0.5 + t * (2.0 - 1.5 * t)),
            -0.5 * t**2 * s,
        ),
        axis=1,
    )
    derivatives = jnp.stack(
        (
            s * (1.5 * t - 0.5),
            t * (4.5 * t - 5.0),
            0.5 + t * (4.0 - 4.5 * t),
            t * (1.5 * t - 1.0),
        ),
        axis=1,
    )
    return weights, derivatives


def _hermite_stencil(data_mesh_beta, x_min, x_max, y_min, y_max, n):
    beta = jnp.asarray(data_mesh_beta)
    tx, xs, valid_x, support_x, dx = _hermite_axis(beta[:, 0], x_min, x_max, n)
    ty, ys, valid_y, support_y, dy = _hermite_axis(beta[:, 1], y_min, y_max, n)
    wx, gx = _hermite_weights(tx)
    wy, gy = _hermite_weights(ty)
    indices = ys[:, :, None] * n + xs[:, None, :]
    valid_nodes = valid_y[:, :, None] & valid_x[:, None, :]
    return wx, wy, gx / dx, gy / dy, indices, valid_nodes, support_x & support_y


@partial(jax.jit, static_argnames=("n",))
def _lens_mapping_operator_hermite_from(
    data_mesh_beta, x_min, x_max, y_min, y_max, n
):
    wx, wy, _, _, indices, valid_nodes, support = _hermite_stencil(
        data_mesh_beta, x_min, x_max, y_min, y_max, n
    )
    weights = jnp.where(valid_nodes, wy[:, :, None] * wx[:, None, :], 0.0)
    return weights.reshape(-1, 16), indices.reshape(-1, 16), support


def lens_mapping_operator_from(
    data_mesh_beta, x_min, x_max, y_min, y_max, n, *, interpolation
):
    """Build weights, original-node indices and geometric support for each ray.

    The interpolation mode and source grid size are static when jitted. In
    Hermite mode the support extends two grid spacings past the fitted nodes.
    A supported ray can have zero brightness weights at a virtual grid line.
    """
    if validate_source_interpolation(interpolation) == "bilinear":
        return lens_mapping_operator_bilinear_from(
            data_mesh_beta, x_min, x_max, y_min, y_max, n
        )
    return _lens_mapping_operator_hermite_from(
        data_mesh_beta, x_min, x_max, y_min, y_max, n
    )


@partial(jax.jit, static_argnames=("n",))
def bilinear_source_spatial_gradient(
    source_values, data_mesh_beta, x_min, x_max, y_min, y_max, n
):
    """Analytically differentiate the selected bilinear brightness branch.

    Interior knots use the right/up cell. Clipped upper-endpoint nodes coincide,
    giving zero derivative along that axis. Both derivatives vanish out of bounds.
    This is a cell derivative, rather than interpolation of node differences.
    """
    fx, fy, corners, valid, dx, dy = _bilinear_cells(
        data_mesh_beta, x_min, x_max, y_min, y_max, n
    )
    source_values = jnp.asarray(source_values)
    s00, s10, s01, s11 = (source_values[index] for index in corners)
    gx = ((1 - fy) * (s10 - s00) + fy * (s11 - s01)) / dx
    gy = ((1 - fx) * (s01 - s00) + fx * (s11 - s10)) / dy
    return jnp.where(valid, gx, 0), jnp.where(valid, gy, 0)


@partial(jax.jit, static_argnames=("n",))
def _hermite_source_spatial_gradient(
    source_values, data_mesh_beta, x_min, x_max, y_min, y_max, n
):
    wx, wy, gx, gy, indices, valid_nodes, _ = _hermite_stencil(
        data_mesh_beta, x_min, x_max, y_min, y_max, n
    )
    values = jnp.where(valid_nodes, jnp.asarray(source_values)[indices], 0.0)
    return (
        jnp.sum(values * wy[:, :, None] * gx[:, None, :], axis=(1, 2)),
        jnp.sum(values * gy[:, :, None] * wx[:, None, :], axis=(1, 2)),
    )


def source_spatial_gradient(
    source_values, data_mesh_beta, x_min, x_max, y_min, y_max, n, *, interpolation
):
    """Differentiate brightness at each ray using the selected interpolant."""
    if validate_source_interpolation(interpolation) == "bilinear":
        return bilinear_source_spatial_gradient(
            source_values, data_mesh_beta, x_min, x_max, y_min, y_max, n
        )
    return _hermite_source_spatial_gradient(
        source_values, data_mesh_beta, x_min, x_max, y_min, y_max, n
    )


def build_source_grid(n, xmin, xmax, ymin, ymax):
    """Build a square source-plane grid spanning [xmin, xmax] x [ymin, ymax].

    Returns (x_axis, y_axis, xgrid, ygrid) with ``jnp.meshgrid(..., indexing='xy')`` layout.
    """
    x_axis = jnp.linspace(xmin, xmax, int(n))
    y_axis = jnp.linspace(ymin, ymax, int(n))
    xgrid, ygrid = jnp.meshgrid(x_axis, y_axis, indexing="xy")
    return x_axis, y_axis, xgrid, ygrid


def make_square_bbox(xmin, xmax, ymin, ymax):
    """Expand bbox bounds to a square while preserving each axis center.

    The shorter span is expanded to match the longer span.  Inputs may be
    Python scalars or JAX scalar arrays; the returned values are JAX-compatible
    scalar arrays.
    """
    xmin = jnp.asarray(xmin)
    xmax = jnp.asarray(xmax)
    ymin = jnp.asarray(ymin)
    ymax = jnp.asarray(ymax)
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    side = jnp.maximum(xmax - xmin, ymax - ymin)
    half = 0.5 * side
    return xmid - half, xmid + half, ymid - half, ymid + half


def source_bbox_from_center_reff(
    center_x,
    center_y,
    effective_radius,
    radius_factor=2.5,
):
    """Build a square source-plane bbox from a source center and half-light radius.

    The box half-size is ``radius_factor * effective_radius``; consequently,
    its full side length is ``2 * radius_factor * effective_radius``.
    ``effective_radius`` and ``radius_factor`` must be positive. Inputs may be
    Python scalars or JAX scalar arrays, and the returned
    ``(xmin, xmax, ymin, ymax)`` values remain JAX compatible.
    """
    center_x = jnp.asarray(center_x)
    center_y = jnp.asarray(center_y)
    effective_radius = jnp.asarray(effective_radius)
    radius_factor = jnp.asarray(radius_factor)
    half_size = radius_factor * effective_radius
    return (
        center_x - half_size,
        center_x + half_size,
        center_y - half_size,
        center_y + half_size,
    )


def infer_source_bbox(beta_x, beta_y, padding=0.05, outlier_frac=0.0):
    """Infer a square source-plane bounding box from ray-traced beta points.

    Computes robust quantile bounds of beta coordinates and adds a
    fractional padding margin on each side (default 0.05 — 5% per side).
    ``outlier_frac`` is trimmed from each tail.  The default of 0 uses the
    absolute min/max; set a positive fraction to enable robust quantile
    trimming.  Ensures a minimum span of 1e-6 in each direction so that
    downstream grid construction is well-defined.  The shorter span is
    always expanded around its center after padding/flooring so the
    returned source-plane bbox has equal x/y extent.
    """
    if not (0.0 <= outlier_frac < 0.5):
        raise ValueError(
            f"outlier_frac must be in [0, 0.5), got {outlier_frac}"
        )

    beta_x = jnp.ravel(jnp.asarray(beta_x))
    beta_y = jnp.ravel(jnp.asarray(beta_y))

    if outlier_frac == 0.0:
        xmin = jnp.min(beta_x)
        xmax = jnp.max(beta_x)
        ymin = jnp.min(beta_y)
        ymax = jnp.max(beta_y)
    else:
        q_low = float(outlier_frac)
        q_high = 1.0 - float(outlier_frac)
        xmin = jnp.quantile(beta_x, q_low)
        xmax = jnp.quantile(beta_x, q_high)
        ymin = jnp.quantile(beta_y, q_low)
        ymax = jnp.quantile(beta_y, q_high)

    span_x = xmax - xmin
    span_y = ymax - ymin
    xmin = xmin - padding * span_x
    xmax = xmax + padding * span_x
    ymin = ymin - padding * span_y
    ymax = ymax + padding * span_y
    # Floor at ±0.5e-6 to keep the grid well-defined for point-like sources.
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    min_half = 0.5e-6
    xmin = jnp.minimum(xmin, xmid - min_half)
    xmax = jnp.maximum(xmax, xmid + min_half)
    ymin = jnp.minimum(ymin, ymid - min_half)
    ymax = jnp.maximum(ymax, ymid + min_half)
    xmin, xmax, ymin, ymax = make_square_bbox(xmin, xmax, ymin, ymax)
    return xmin, xmax, ymin, ymax


def build_lens_mapping_matrix(
    beta_x, beta_y, source_x_axis, source_y_axis, *, interpolation="bilinear"
):
    """Build the dense mapping matrix with one column per fitted source node."""
    beta_x = jnp.ravel(jnp.asarray(beta_x))
    beta_y = jnp.ravel(jnp.asarray(beta_y))
    data_mesh_beta = jnp.stack([beta_x, beta_y], axis=1)

    source_x_axis = jnp.asarray(source_x_axis)
    source_y_axis = jnp.asarray(source_y_axis)
    n = source_x_axis.shape[0]

    weights, indices, _ = lens_mapping_operator_from(
        data_mesh_beta,
        source_x_axis[0],
        source_x_axis[-1],
        source_y_axis[0],
        source_y_axis[-1],
        n,
        interpolation=interpolation,
    )
    return dense_mapping_from_weights_indices(weights, indices, n * n)


__all__ = [
    "validate_source_interpolation",
    "source_boundary_for_interpolation",
    "lens_mapping_operator_from",
    "lens_mapping_operator_bilinear_from",
    "source_spatial_gradient",
    "dense_mapping_from_weights_indices",
    "build_source_grid",
    "build_lens_mapping_matrix",
    "make_square_bbox",
    "source_bbox_from_center_reff",
    "infer_source_bbox",
]
