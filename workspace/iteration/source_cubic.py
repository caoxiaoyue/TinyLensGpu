"""Workspace-only C1 cardinal bicubic source; original node values and prior.

Tensor-product cubic Hermite interpolation with centered nodal derivatives
(Catmull--Rom / Keys a=-1/2). The four weights in each direction interpolate
the *original intensities*, rather than interpreting them as B-spline control
coefficients. Exterior nodes are fixed to zero. Thus values and first spatial
derivatives are continuous at every interior, edge and exterior grid line,
including the final compact-support boundary two cells beyond fitted nodes.

Negative weights and modest overshoot are possible, as for this cubic basis
generally. The existing signed-source convention is preserved. No clipping,
extra fitted derivatives, changes to priors, or custom AD rules are used.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer
from TinyLensGpu.ForwardSimulation.LensImage.potential_correction import SourceMapping


def hermite_weights(t):
    """Cardinal weights for cell neighbors (-1, 0, 1, 2), and d/dt."""
    t2, t3 = t*t, t*t*t
    weights = jnp.stack((-.5*t+t2-.5*t3, 1.-2.5*t2+1.5*t3,
                         .5*t+2.*t2-1.5*t3, -.5*t2+.5*t3), axis=-1)
    derivative = jnp.stack((-.5+2.*t-1.5*t2, -5.*t+4.5*t2,
                            .5+4.*t-4.5*t2, -t+1.5*t2), axis=-1)
    return weights, derivative


def _stencil(beta, bbox, n):
    xmin, xmax, ymin, ymax = bbox
    dx, dy = (xmax-xmin)/(n-1), (ymax-ymin)/(n-1)
    u, v = (beta[:, 0]-xmin)/dx, (beta[:, 1]-ymin)/dy
    x0, y0 = jnp.floor(u).astype(jnp.int32), jnp.floor(v).astype(jnp.int32)
    wx, gx = hermite_weights(u-x0)
    wy, gy = hermite_weights(v-y0)
    offsets = jnp.arange(-1, 3, dtype=jnp.int32)
    xs, ys = x0[:, None]+offsets, y0[:, None]+offsets
    nodes = ((ys[:, :, None] >= 0) & (ys[:, :, None] < n)
             & (xs[:, None, :] >= 0) & (xs[:, None, :] < n))
    indices = jnp.clip(ys[:, :, None], 0, n-1)*n+jnp.clip(xs[:, None, :], 0, n-1)
    support = (u >= -2) & (u <= n+1) & (v >= -2) & (v <= n+1)
    return wx, wy, gx/dx, gy/dy, indices, nodes, support


@partial(jax.jit, static_argnames=("n",))
def cubic_mapping(beta, bbox, n):
    wx, wy, _, _, indices, nodes, support = _stencil(beta, bbox, n)
    weights = jnp.where(nodes, wy[:, :, None]*wx[:, None, :], 0.)
    return weights.reshape(-1, 16), indices.reshape(-1, 16), support


@partial(jax.jit, static_argnames=("n",))
def cubic_gradient(source, beta, bbox, n):
    wx, wy, gx, gy, indices, nodes, _ = _stencil(beta, bbox, n)
    values = jnp.where(nodes, source[indices], 0.)
    dx = jnp.sum(values*wy[:, :, None]*gx[:, None, :], axis=(1, 2))
    dy = jnp.sum(values*gy[:, :, None]*wx[:, None, :], axis=(1, 2))
    return dx, dy


class CubicSourceRenderer(PotentialCorrectionRenderer):
    """C1 source interpolation only; potential and imaging operators inherited."""

    source_boundary = "C1_cardinal_bicubic_fixed_zero_exterior_nodes"

    @property
    def support_bbox(self):
        xmin, xmax, ymin, ymax = np.asarray(self.source_bbox)
        dx, dy = (xmax-xmin)/(self.source_n-1), (ymax-ymin)/(self.source_n-1)
        return xmin-2*dx, xmax+2*dx, ymin-2*dy, ymax+2*dy

    def source_mapping(self, potential_values=None, *, macro_mapping=None):
        rays = self.macro_mapping if macro_mapping is None else macro_mapping
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            bx, by = rays
            if potential_values is not None:
                ax, ay = self.potential_grid.deflection(potential_values)
                bx, by = bx-ax, by-ay
            weights, indices, valid = cubic_mapping(
                jnp.stack((bx, by), axis=1), self.source_bbox, self.source_n)
            return SourceMapping(weights, indices, valid, bx, by)

    def source_spatial_gradient(self, source_values, mapping=None):
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            mapping = self.source_mapping() if mapping is None else mapping
            return cubic_gradient(self._source_values(source_values),
                jnp.stack((mapping.beta_x, mapping.beta_y), axis=1),
                self.source_bbox, self.source_n)
