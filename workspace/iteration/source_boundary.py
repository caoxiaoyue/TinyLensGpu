"""Workspace experiment for a fixed zero-ghost-ring source boundary."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer
from TinyLensGpu.ForwardSimulation.LensImage.potential_correction import SourceMapping


def _cells(beta, bbox, n):
    xmin, xmax, ymin, ymax = bbox
    dx, dy = (xmax-xmin)/(n-1), (ymax-ymin)/(n-1)
    u, v = (beta[:, 0]-xmin)/dx, (beta[:, 1]-ymin)/dy
    x0, y0 = jnp.floor(u).astype(jnp.int32), jnp.floor(v).astype(jnp.int32)
    fx, fy = u-x0, v-y0
    xs = jnp.stack((x0, x0+1, x0, x0+1), axis=1)
    ys = jnp.stack((y0, y0, y0+1, y0+1), axis=1)
    inside_node = (xs >= 0) & (xs < n) & (ys >= 0) & (ys < n)
    indices = jnp.clip(ys, 0, n-1)*n + jnp.clip(xs, 0, n-1)
    in_support = (u >= -1) & (u <= n) & (v >= -1) & (v <= n)
    return fx, fy, indices, inside_node, in_support, dx, dy


@partial(jax.jit, static_argnames=("n",))
def zero_ring_mapping(beta, bbox, n):
    """Bilinear hat basis with zero coefficients at all exterior ghost nodes.

    Original nodes, their row-major ordering, and spacing are unchanged. The
    brightness extends one cell beyond them, continuously reaching zero at
    the ghost ring. Mask each corner separately, rather than cutting a whole
    query to zero at the old bounding box. No ghost coefficient is fitted.
    """
    fx, fy, indices, nodes, support, _, _ = _cells(beta, bbox, n)
    weights = jnp.stack(((1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy), axis=1)
    return jnp.where(nodes, weights, 0), indices, support


@partial(jax.jit, static_argnames=("n",))
def zero_ring_gradient(source, beta, bbox, n):
    """Analytical derivative of the same selected zero-extended bilinear cell."""
    fx, fy, indices, nodes, _, dx, dy = _cells(beta, bbox, n)
    values = jnp.where(nodes, source[indices], 0)
    s00, s10, s01, s11 = (values[:, i] for i in range(4))
    gx = ((1-fy)*(s10-s00) + fy*(s11-s01))/dx
    gy = ((1-fx)*(s01-s00) + fx*(s11-s10))/dy
    return gx, gy


class ZeroRingRenderer(PotentialCorrectionRenderer):
    """One fixed zero ring beyond the original source nodes; no extra unknowns."""

    source_boundary = "bilinear_fixed_zero_ghost_ring"

    @property
    def support_bbox(self):
        xmin, xmax, ymin, ymax = np.asarray(self.source_bbox)
        dx, dy = (xmax-xmin)/(self.source_n-1), (ymax-ymin)/(self.source_n-1)
        return (xmin-dx, xmax+dx, ymin-dy, ymax+dy)

    def source_mapping(self, potential_values=None, *, macro_mapping=None):
        rays = self.macro_mapping if macro_mapping is None else macro_mapping
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            bx, by = rays
            if potential_values is not None:
                ax, ay = self.potential_grid.deflection(potential_values)
                bx, by = bx-ax, by-ay
            weights, indices, valid = zero_ring_mapping(
                jnp.stack((bx, by), axis=1), self.source_bbox, self.source_n)
            return SourceMapping(weights, indices, valid, bx, by)

    def source_spatial_gradient(self, source_values, mapping=None):
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            mapping = self.source_mapping() if mapping is None else mapping
            return zero_ring_gradient(self._source_values(source_values),
                jnp.stack((mapping.beta_x, mapping.beta_y), axis=1),
                self.source_bbox, self.source_n)
