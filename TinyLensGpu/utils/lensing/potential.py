"""Local regular-grid potential support and physical finite differences.

Topology is prepared on the host once. Numerical gather/scatter operations
remain JAX arrays and can be composed inside compiled reconstruction kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


class PotentialDerivatives(NamedTuple):
    """Physical derivatives at core nodes, ordered as ``core_indices``."""

    deflection_x: Array
    deflection_y: Array
    convergence: Array


@jax.jit
def _core_derivatives(p, indices, spacing):
    center, left, right, lower, upper = (p[indices[:, i]] for i in range(5))
    alpha_x = (right - left) / (2 * spacing)
    alpha_y = (upper - lower) / (2 * spacing)
    kappa = (right + left + upper + lower - 4 * center) / (2 * spacing**2)
    return PotentialDerivatives(alpha_x, alpha_y, kappa)


@jax.jit
def _query_deflection(p, derivative_indices, indices, weights, spacing):
    ax, ay, _ = _core_derivatives(p, derivative_indices, spacing)
    return jnp.sum(weights * ax[indices], axis=1), jnp.sum(
        weights * ay[indices], axis=1
    )


@partial(jax.jit, static_argnames=("n_core", "n_unknown"))
def _query_deflection_adjoint(
    vx,
    vy,
    derivative_indices,
    indices,
    weights,
    spacing,
    *,
    n_core,
    n_unknown,
):
    core_x = jnp.zeros(n_core, dtype=vx.dtype).at[indices.ravel()].add(
        (weights * vx[:, None]).ravel()
    ) / (2 * spacing)
    core_y = jnp.zeros(n_core, dtype=vy.dtype).at[indices.ravel()].add(
        (weights * vy[:, None]).ravel()
    ) / (2 * spacing)
    output = jnp.zeros(n_unknown, dtype=vx.dtype)
    output = output.at[derivative_indices[:, 1]].add(-core_x)
    output = output.at[derivative_indices[:, 2]].add(core_x)
    output = output.at[derivative_indices[:, 3]].add(-core_y)
    return output.at[derivative_indices[:, 4]].add(core_y)


@dataclass(frozen=True)
class PotentialGrid:
    """Potential unknowns around the actual interpolation queries.

    ``node_indices`` stores integer ``(ix, iy)`` pairs in row-major order.
    All nodes, including support-only nodes, are unknown coefficients. Physical
    derivatives are valid only at ``core_indices``. The grid origin is an
    angular ``(x, y)`` pair; spacing is in arcseconds.
    """

    spacing: float
    origin: tuple[float, float]
    node_indices: Array
    x: Array
    y: Array
    core_indices: Array
    derivative_valid: Array
    derivative_indices: Array
    interpolation_indices: Array
    interpolation_weights: Array
    query_x: Array
    query_y: Array

    @property
    def n_unknown(self) -> int:
        return self.x.shape[0]

    @property
    def n_core(self) -> int:
        return self.core_indices.shape[0]

    def _values(self, potential_values):
        values = jnp.asarray(potential_values, dtype=self.x.dtype)
        if values.shape != (self.n_unknown,):
            raise ValueError(f"potential_values must have shape ({self.n_unknown},)")
        return values

    def derivatives(self, potential_values) -> PotentialDerivatives:
        """Centered first/second physical differences on core nodes only."""
        with jax.experimental.enable_x64(self.x.dtype == jnp.float64):
            return _core_derivatives(
                self._values(potential_values),
                self.derivative_indices,
                jnp.asarray(self.spacing, dtype=self.x.dtype),
            )

    def deflection(self, potential_values) -> tuple[Array, Array]:
        """Interpolate the core-node physical deflection to all query positions."""
        with jax.experimental.enable_x64(self.x.dtype == jnp.float64):
            return _query_deflection(
                self._values(potential_values),
                self.derivative_indices,
                self.interpolation_indices,
                self.interpolation_weights,
                jnp.asarray(self.spacing, dtype=self.x.dtype),
            )

    def deflection_adjoint(self, values_x, values_y) -> Array:
        """Transpose of :meth:`deflection`, accumulating into all unknowns."""
        with jax.experimental.enable_x64(self.x.dtype == jnp.float64):
            vx = jnp.asarray(values_x, dtype=self.x.dtype)
            vy = jnp.asarray(values_y, dtype=self.x.dtype)
            if vx.shape != self.query_x.shape or vy.shape != self.query_y.shape:
                raise ValueError("Deflection adjoint values must match query shapes")
            return _query_deflection_adjoint(
                vx,
                vy,
                self.derivative_indices,
                self.interpolation_indices,
                self.interpolation_weights,
                jnp.asarray(self.spacing, dtype=self.x.dtype),
                n_core=self.n_core,
                n_unknown=self.n_unknown,
            )

    @classmethod
    def from_queries(
        cls,
        query_x,
        query_y,
        *,
        spacing,
        origin=(0.0, 0.0),
        node_indices=None,
        dtype="float64",
    ) -> "PotentialGrid":
        """Construct C from four query corners and U from axial ±1/±2 support.

        An optional explicit ``node_indices`` must contain every required node,
        without duplicates; additional supplied nodes remain fitted unknowns.
        Missing support raises an error rather than dropping image queries.
        """
        if dtype not in ("float32", "float64", jnp.float32, jnp.float64):
            raise ValueError("dtype must be float32 or float64")
        dtype = np.dtype(dtype)
        spacing = float(spacing)
        origin_arr = np.asarray(origin, dtype=np.float64)
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("Potential spacing must be finite and positive")
        if origin_arr.shape != (2,) or not np.isfinite(origin_arr).all():
            raise ValueError("Potential origin must contain two finite coordinates")
        qx, qy = np.asarray(query_x), np.asarray(query_y)
        if qx.ndim != 1 or qy.shape != qx.shape or qx.size == 0:
            raise ValueError("Potential queries must be nonempty matching 1D arrays")
        if not np.isfinite(qx).all() or not np.isfinite(qy).all():
            raise ValueError("Potential query coordinates must be finite")
        with np.errstate(over="ignore", invalid="ignore"):
            uv = (np.column_stack((qx, qy)) - origin_arr) / spacing
        index_limits = np.iinfo(np.int32)
        if (
            not np.isfinite(uv).all()
            or np.any(uv < index_limits.min + 3)
            or np.any(uv > index_limits.max - 3)
        ):
            raise ValueError(
                "Potential queries exceed the supported integer index range"
            )
        lower = np.floor(uv).astype(np.int64)
        corners = lower[:, None, :] + np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        core = {tuple(node) for node in corners.reshape(-1, 2)}
        offsets = [
            (0, 0),
            (-1, 0),
            (1, 0),
            (-2, 0),
            (2, 0),
            (0, -1),
            (0, 1),
            (0, -2),
            (0, 2),
        ]
        required = {(ix + dx, iy + dy) for ix, iy in core for dx, dy in offsets}
        if node_indices is None:
            support = required
        else:
            offered = np.asarray(node_indices)
            if (
                offered.ndim != 2
                or offered.shape[1:] != (2,)
                or not np.issubdtype(offered.dtype, np.integer)
            ):
                raise ValueError(
                    "Potential node_indices must have integer shape (N, 2)"
                )
            if np.any(offered < index_limits.min) or np.any(offered > index_limits.max):
                raise ValueError(
                    "Potential node_indices exceed the supported integer index range"
                )
            support = {tuple(node) for node in offered}
            if len(support) != offered.shape[0]:
                raise ValueError("Potential node_indices must be unique")
            missing = required - support
            if missing:
                raise ValueError(
                    f"Explicit potential geometry is missing {len(missing)} required "
                    "core/support nodes; observation queries cannot be discarded"
                )
        ordered = sorted(support, key=lambda node: (node[1], node[0]))
        core_ordered = sorted(core, key=lambda node: (node[1], node[0]))
        lookup = {node: index for index, node in enumerate(ordered)}
        core_lookup = {node: index for index, node in enumerate(core_ordered)}
        core_indices = np.array([lookup[node] for node in core_ordered], dtype=np.int32)
        valid = np.zeros(len(ordered), dtype=bool)
        valid[core_indices] = True
        derivative_indices = np.array(
            [
                [
                    lookup[(x, y)],
                    lookup[(x - 1, y)],
                    lookup[(x + 1, y)],
                    lookup[(x, y - 1)],
                    lookup[(x, y + 1)],
                ]
                for x, y in core_ordered
            ],
            dtype=np.int32,
        )
        interpolation_indices = np.array(
            [[core_lookup[tuple(node)] for node in row] for row in corners],
            dtype=np.int32,
        )
        fx, fy = (uv - lower).T
        weights = np.column_stack(
            ((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy)
        )
        nodes = np.asarray(ordered, dtype=np.int32)
        coordinates = origin_arr + spacing * nodes
        with jax.experimental.enable_x64(dtype == np.dtype("float64")):
            return cls(
                spacing,
                tuple(origin_arr),
                jnp.asarray(nodes),
                jnp.asarray(coordinates[:, 0], dtype=dtype),
                jnp.asarray(coordinates[:, 1], dtype=dtype),
                jnp.asarray(core_indices),
                jnp.asarray(valid),
                jnp.asarray(derivative_indices),
                jnp.asarray(interpolation_indices),
                jnp.asarray(weights, dtype=dtype),
                jnp.asarray(qx, dtype=dtype),
                jnp.asarray(qy, dtype=dtype),
            )
