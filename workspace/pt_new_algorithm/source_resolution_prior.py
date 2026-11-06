"""Workspace-only continuum-matched curvature prior for the source-size probe.

The n=49 member is exactly the original uniform second-order prior. Interior
curvature is a 2-D integral, while the existing last-edge gradient and
brightness fallbacks are 1-D boundary integrals. Match all three measures,
not only the derivative powers. The original asymmetric boundary convention
and signed source coefficients are preserved.
"""

import jax.numpy as jnp


class PhysicalCurvaturePrior:
    regularization_type = "second-order"

    def __init__(self, n, reference_n=49):
        self.n, self.reference_n = int(n), int(reference_n)
        if min(self.n, self.reference_n) < 3:
            raise ValueError("Curvature comparison needs at least three nodes per side")
        self.n_pixels = self.n*self.n
        self.spacing_ratio = (self.reference_n-1)/(self.n-1)

    def _coefficients(self, xmin, xmax, ymin, ymax, scale):
        if scale is not None:
            raise ValueError("This experiment matches the original uniform Mock prior only")
        # The renderer validates a positive square bbox. These scalar bounds
        # carry the renderer's Float64 dtype into the stencil coefficient.
        h = (xmax-xmin)/(self.n-1)
        r = self.spacing_ratio
        return r**2/h**4, r**3/h**4, r**5/h**4

    def matvec_free(self, source, xmin, xmax, ymin, ymax, *, scale=None):
        curvature, near, outer = self._coefficients(xmin, xmax, ymin, ymax, scale)
        s = source.reshape(self.n, self.n)
        out = jnp.zeros_like(s)
        cx = curvature*(s[:, :-2]-2*s[:, 1:-1]+s[:, 2:])
        out = out.at[:, :-2].add(cx).at[:, 1:-1].add(-2*cx).at[:, 2:].add(cx)
        cy = curvature*(s[:-2, :]-2*s[1:-1, :]+s[2:, :])
        out = out.at[:-2, :].add(cy).at[1:-1, :].add(-2*cy).at[2:, :].add(cy)
        gx, gy = near*(s[:, -1]-s[:, -2]), near*(s[-1, :]-s[-2, :])
        out = out.at[:, -2].add(-gx).at[:, -1].add(gx)
        out = out.at[-2, :].add(-gy).at[-1, :].add(gy)
        out = out.at[:, -1].add(outer*s[:, -1])
        out = out.at[-1, :].add(outer*s[-1, :])
        return out.ravel()

    def matrix(self, xmin, xmax, ymin, ymax, *, scale=None):
        curvature, near, outer = self._coefficients(xmin, xmax, ymin, ymax, scale)
        eye = jnp.eye(self.n, dtype=jnp.result_type(curvature))
        d2 = eye[:-2]-2*eye[1:-1]+eye[2:]
        d1 = eye[-1]-eye[-2]
        k = curvature*(d2.T@d2) + near*jnp.outer(d1, d1) + outer*jnp.outer(eye[-1], eye[-1])
        return jnp.kron(eye, k)+jnp.kron(k, eye)

    to_dense_free = matrix

    def components(self, source, bbox):
        """Separate quadratic energies for a physical-units sanity check."""
        curvature, near, outer = self._coefficients(*bbox, None)
        s = source.reshape(self.n, self.n)
        sq = lambda v: jnp.vdot(v, v)
        return dict(curvature=curvature*(sq(s[:, :-2]-2*s[:, 1:-1]+s[:, 2:])
                                             + sq(s[:-2, :]-2*s[1:-1, :]+s[2:, :])),
                    boundary_gradient=near*(sq(s[:, -1]-s[:, -2])+sq(s[-1, :]-s[-2, :])),
                    boundary_amplitude=outer*(sq(s[:, -1])+sq(s[-1, :])))
