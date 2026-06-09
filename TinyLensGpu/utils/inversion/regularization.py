"""Dense regularization matrix builders for pixelized source inversion.

This module provides traditional finite-difference penalties and dense
Gaussian-process covariance penalties for source-plane linear inversion.  The
finite-difference operators are precomputed on index space and scaled by the
physical source-grid spacing when a matrix is requested.

For matrix-free (operator) backends, the class also exposes
:meth:`matvec_free`, :meth:`logdet_free`, and :meth:`to_dense_free` for
finite-difference types, which exploit the Kronecker-sum structure of
separable 2-D difference operators to avoid materialising the full Ns x Ns
regularisation matrix.
"""

from __future__ import annotations

# pyright: reportMissingImports=false

from typing import NamedTuple

import jax.numpy as jnp
import jax.scipy.linalg as jsl

VALID_REGULARIZATION_TYPES: frozenset[str] = frozenset({
    "zero-order", "first-order", "second-order",
    "exponential", "gaussian",
    "matern32", "matern52", "matern72",
})
"""All supported regularization type names."""

GP_REGULARIZATION_TYPES: frozenset[str] = frozenset({
    "exponential", "gaussian",
    "matern32", "matern52", "matern72",
})
"""GP-style regularization types that require a kernel_scale parameter."""


class RegData(NamedTuple):
    """Compact regularisation data for matrix-free matvec / logdet.

    Passed through ``A_data[7]`` in the PCG solver so that
    :func:`_A_matvec_jit` can apply the regularisation term without a dense
    ``(Ns, Ns)`` matrix.

    For finite-difference types, ``rx`` / ``ry`` hold the 1-D product
    matrices ``(nx, nx)`` / ``(ny, ny)`` and ``gp_matrix`` is a placeholder
    with a shape compatible with the source vector.
    For GP types, ``is_gp`` is ``True`` and ``gp_matrix`` holds the true
    dense ``(Ns, Ns)`` precision matrix.

    Note: ``nx`` / ``ny`` are *not* stored here; they are passed as static
    arguments to :func:`_A_matvec_jit` via the pre-bound partial.
    """
    rx: "jax.Array"        # (nx, nx)  1-D x-regularisation product
    ry: "jax.Array"        # (ny, ny)  1-D y-regularisation product
    scale_x: "jax.Array"   # scalar  physical pixel-area scale for x
    scale_y: "jax.Array"   # scalar  physical pixel-area scale for y
    is_gp: "jax.Array"     # bool scalar
    gp_matrix: "jax.Array" # (Ns, Ns) dense GP precision, or (1, Ns) placeholder


class DenseRegularizationBuilder:
    """Build dense source-grid regularization matrices.

    Parameters
    ----------
    nx : int
        Number of source pixels along the x-axis.
    ny : int
        Number of source pixels along the y-axis.
    regularization_type : str
        Regularization family. Supported values are ``"zero-order"``,
        ``"first-order"``, ``"second-order"``, ``"exponential"``,
        ``"gaussian"``, ``"matern32"``, ``"matern52"``, and ``"matern72"``.
    jitter : float, optional
        Diagonal jitter added to GP covariance matrices for numerical stability.

    Raises
    ------
    ValueError
        If the grid shape or regularization configuration is invalid.
    """

    _VALID_TYPES = VALID_REGULARIZATION_TYPES

    def __init__(
        self,
        nx: int,
        ny: int,
        regularization_type: str,
        *,
        jitter: float = 1e-6,
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        if self.nx < 2 or self.ny < 2:
            raise ValueError("nx and ny must both be at least 2.")

        self.n_pixels = self.nx * self.ny
        self.regularization_type = regularization_type.lower()
        self.jitter = float(jitter)

        if self.regularization_type not in self._VALID_TYPES:
            raise ValueError(f"Unsupported regularization_type: {regularization_type!r}.")
        if self.jitter < 0.0:
            raise ValueError("jitter must be non-negative.")

        self._identity = jnp.eye(self.n_pixels)
        self._dx_operator, self._dy_operator = self._build_first_difference_operators()
        self._lx_operator, self._ly_operator = self._build_curvature_difference_operators()
        self._unit_coordinates = self._build_unit_coordinates()

        # Precompute H_unit matrices on index space (half_size=1, spacing=2/(n-1)).
        # H(h) = H_unit / dx(h)^2  for first-order,  / dx(h)^4  for second-order.
        # dx(h) = 2h/(nx-1), dy(h) = 2h/(ny-1).
        # Separate x/y contributions so non-square grids (nx != ny) are handled correctly.
        dx_unit = 2.0 / (self.nx - 1)
        dy_unit = 2.0 / (self.ny - 1)
        sdx1 = self._dx_operator / dx_unit
        sdy1 = self._dy_operator / dy_unit
        self._H1_unit_x = sdx1.T @ sdx1   # x contribution at half_size=1
        self._H1_unit_y = sdy1.T @ sdy1   # y contribution at half_size=1
        sdx2 = self._lx_operator / (dx_unit ** 2)
        sdy2 = self._ly_operator / (dy_unit ** 2)
        self._H2_unit_x = sdx2.T @ sdx2
        self._H2_unit_y = sdy2.T @ sdy2

        # ----- 1-D operator products & eigendecompositions for matrix-free path -----
        self._Rx1: "jax.Array | None" = None
        self._Ry1: "jax.Array | None" = None
        self._Rx2: "jax.Array | None" = None
        self._Ry2: "jax.Array | None" = None
        self._Rx_eigvals: "jax.Array | None" = None
        self._Ry_eigvals: "jax.Array | None" = None

        if self.regularization_type not in GP_REGULARIZATION_TYPES:
            if self.regularization_type in ("first-order", "second-order"):
                Dx1d = self._build_1d_first_diff(self.nx)
                Dy1d = self._build_1d_first_diff(self.ny)
                self._Rx1 = Dx1d.T @ Dx1d
                self._Ry1 = Dy1d.T @ Dy1d

            if self.regularization_type == "second-order":
                Lx1d = self._build_1d_curvature_diff(self.nx)
                Ly1d = self._build_1d_curvature_diff(self.ny)
                self._Rx2 = Lx1d.T @ Lx1d
                self._Ry2 = Ly1d.T @ Ly1d

            # Precompute eigenvalues once (used by logdet_free at each evidence eval).
            if self.regularization_type == "first-order":
                self._Rx_eigvals = jnp.linalg.eigvalsh(self._Rx1)  # type: ignore[arg-type]
                self._Ry_eigvals = jnp.linalg.eigvalsh(self._Ry1)  # type: ignore[arg-type]
            elif self.regularization_type == "second-order":
                self._Rx_eigvals = jnp.linalg.eigvalsh(self._Rx2)  # type: ignore[arg-type]
                self._Ry_eigvals = jnp.linalg.eigvalsh(self._Ry2)  # type: ignore[arg-type]
            # zero-order: eigenvalues stay None (logdet=0, matvec=identity)

    # ------------------------------------------------------------------
    # 1-D difference operator builders (used by matrix-free path)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_1d_first_diff(n: int) -> "jax.Array":
        """Build 1-D first-difference operator ``(n, n)``.

        Interior rows use ``[-1, 1]``; the last row is a zero-order
        fallback ``[1]`` on the diagonal (Suyu et al. convention).
        """
        D = jnp.zeros((n, n))
        interior = jnp.arange(n - 1)
        D = D.at[interior, interior].add(-1.0)
        D = D.at[interior, interior + 1].add(1.0)
        D = D.at[n - 1, n - 1].add(1.0)
        return D

    @staticmethod
    def _build_1d_curvature_diff(n: int) -> "jax.Array":
        """Build 1-D curvature (second-difference) operator ``(n, n)``.

        Full-curvature rows use ``[1, -2, 1]``, the penultimate row
        falls back to a first-gradient ``[-1, 1]``, and the outermost
        row is a zero-order diagonal ``[1]``.
        """
        L = jnp.zeros((n, n))
        full = jnp.arange(n - 2)
        L = L.at[full, full].add(1.0)
        L = L.at[full, full + 1].add(-2.0)
        L = L.at[full, full + 2].add(1.0)
        L = L.at[n - 2, n - 2].add(-1.0)
        L = L.at[n - 2, n - 1].add(1.0)
        L = L.at[n - 1, n - 1].add(1.0)
        return L

    def matrix(self, xmin, xmax, ymin, ymax, *, kernel_scale: float | None = None):
        """Return the dense regularization matrix for a rectangular source grid.

        Parameters
        ----------
        xmin, xmax : float
            Source-plane x-axis bounds.
        ymin, ymax : float
            Source-plane y-axis bounds.
        kernel_scale : float, optional
            GP kernel correlation scale. Required for GP regularization and
            ignored for traditional finite-difference penalties.

        Returns
        -------
        tuple[jax.Array, jax.Array | None]
            ``(precision_matrix, logdet_covariance)`` for all regularization types.
            For finite-difference types, ``logdet_covariance`` is ``None`` (callers
            compute ``logdet`` via ``slogdet`` on the returned matrix).
            For GP types, ``logdet_covariance`` is ``log|K|`` extracted from the
            Cholesky factorization — callers compute ``logdet(H) = -logdet_covariance``
            without an extra O(n³) ``slogdet`` call.

        Raises
        ------
        ValueError
            If the bounds produce a non-positive span, or a GP kernel scale is
            missing or non-positive.
        """
        if self.regularization_type == "zero-order":
            return self._identity, None
        if self.regularization_type == "first-order":
            return self._first_order_matrix(xmin, xmax, ymin, ymax), None
        if self.regularization_type == "second-order":
            return self._second_order_matrix(xmin, xmax, ymin, ymax), None

        if kernel_scale is None:
            raise ValueError("kernel_scale must be provided for GP regularization.")
        return self._gp_matrix(xmin, xmax, ymin, ymax, kernel_scale)

    # ------------------------------------------------------------------
    # Matrix-free path  (used by operator / PCG backend)
    # ------------------------------------------------------------------

    def matvec_free(
        self, s: "jax.Array", xmin: float, xmax: float, ymin: float, ymax: float,
    ) -> "jax.Array":
        r"""Matrix-free :math:`R @ s` for finite-difference regularisation.

        Exploits the Kronecker-sum structure

        .. math::
            R = I_{ny} \otimes \frac{R_x}{\Delta x^2}
              + \frac{R_y}{\Delta y^2} \otimes I_{nx}

        to apply :math:`R` in :math:`O(N_s (n_x + n_y))` without
        materialising the dense :math:`(N_s, N_s)` matrix.

        Raises :exc:`ValueError` for GP types (caller should use the dense
        fallback).
        """
        if self.regularization_type in GP_REGULARIZATION_TYPES:
            raise ValueError("matvec_free is not supported for GP regularization types.")
        if self.regularization_type == "zero-order":
            return s

        # Physical pixel scales: dx = (xmax - xmin) / (nx - 1)
        inv_dx = (self.nx - 1) / (xmax - xmin)
        inv_dy = (self.ny - 1) / (ymax - ymin)

        if self.regularization_type == "first-order":
            rx, ry = self._Rx1, self._Ry1
            scale_x = inv_dx ** 2
            scale_y = inv_dy ** 2
        elif self.regularization_type == "second-order":
            rx, ry = self._Rx2, self._Ry2
            scale_x = inv_dx ** 4
            scale_y = inv_dy ** 4
        else:
            raise RuntimeError(f"Unhandled type: {self.regularization_type!r}")

        # s has flat index ``i + j * nx`` (column-major on (nx, ny)).
        # Reshape to (ny, nx), transpose → (nx, ny) where axis 0 = x.
        s_2d = s.reshape(self.ny, self.nx).T  # (nx, ny)
        result_2d = rx @ s_2d * scale_x + s_2d @ ry.T * scale_y
        return result_2d.T.ravel()  # back to flat (Ns,)

    def logdet_free(
        self, xmin: float, xmax: float, ymin: float, ymax: float,
    ) -> "jax.Array":
        r"""Eigenvalue-based :math:`\log\det R` for finite-difference regularisation.

        Uses the Kronecker-sum eigenvalue formula

        .. math::
            \lambda_{ij} = \frac{\mu_i}{\Delta x^2} + \frac{\nu_j}{\Delta y^2}

        where :math:`\mu_i` and :math:`\nu_j` are the eigenvalues of the
        1-D product matrices :math:`R_x` and :math:`R_y`.

        Complexity :math:`O(n_x^3 + n_y^3)` for the initial eigendecomposition
        (done once in ``__init__``) and :math:`O(n_x n_y)` per call.

        Raises :exc:`ValueError` for GP types.
        """
        if self.regularization_type in GP_REGULARIZATION_TYPES:
            raise ValueError("logdet_free is not supported for GP regularization types.")
        if self.regularization_type == "zero-order":
            return jnp.array(0.0, dtype=jnp.float32)

        inv_dx = (self.nx - 1) / (xmax - xmin)
        inv_dy = (self.ny - 1) / (ymax - ymin)

        if self.regularization_type == "first-order":
            scale_x = inv_dx ** 2
            scale_y = inv_dy ** 2
        elif self.regularization_type == "second-order":
            scale_x = inv_dx ** 4
            scale_y = inv_dy ** 4
        else:
            raise RuntimeError(f"Unhandled type: {self.regularization_type!r}")

        # λ_{i,j} = μ_i * scale_x + ν_j * scale_y
        eig_grid = (self._Rx_eigvals[:, None] * scale_x
                    + self._Ry_eigvals[None, :] * scale_y)  # (nx, ny)
        # Guard against numerical zeros from the boundary treatment.
        eig_grid = jnp.maximum(eig_grid, 1e-30)
        return jnp.sum(jnp.log(eig_grid))

    def to_dense_free(
        self, xmin: float, xmax: float, ymin: float, ymax: float,
    ) -> "jax.Array":
        """Materialise the dense ``(N_s, N_s)`` regularisation matrix.

        For finite-difference types this uses the Kronecker representation;
        for GP types it delegates to :meth:`matrix` (identical result).

        Intended for one-shot use in :meth:`~PixelizedLensOperator.build_preconditioner`
        where an explicit ``R`` is still needed.
        """
        if self.regularization_type in GP_REGULARIZATION_TYPES:
            # GP types don't have a meaningful kernel_scale here; callers
            # should go through _regularization_data which supplies one.
            raise ValueError(
                "to_dense_free is not directly supported for GP types. "
                "Use matrix() with a kernel_scale instead."
            )
        if self.regularization_type == "zero-order":
            return self._identity

        inv_dx = (self.nx - 1) / (xmax - xmin)
        inv_dy = (self.ny - 1) / (ymax - ymin)

        if self.regularization_type == "first-order":
            rx, ry = self._Rx1, self._Ry1
            scale_x = inv_dx ** 2
            scale_y = inv_dy ** 2
        elif self.regularization_type == "second-order":
            rx, ry = self._Rx2, self._Ry2
            scale_x = inv_dx ** 4
            scale_y = inv_dy ** 4
        else:
            raise RuntimeError(f"Unhandled type: {self.regularization_type!r}")

        # R = I_ny ⊗ (rx * scale_x) + (ry * scale_y) ⊗ I_nx
        return jnp.kron(jnp.eye(self.ny, dtype=rx.dtype), rx * scale_x) \
             + jnp.kron(ry * scale_y, jnp.eye(self.nx, dtype=ry.dtype))

    def make_reg_data(
        self,
        xmin: float, xmax: float, ymin: float, ymax: float,
        gp_matrix: "jax.Array | None" = None,
    ) -> RegData:
        """Return a compact :class:`RegData` tuple for passing through ``A_data``.

        Parameters
        ----------
        xmin, xmax, ymin, ymax : float
            Source-plane bounds.
        gp_matrix : Array, optional
            Dense ``(N_s, N_s)`` GP precision matrix.  Required when
            ``regularization_type`` is a GP type; ignored otherwise.
        """
        if self.regularization_type in GP_REGULARIZATION_TYPES:
            if gp_matrix is None:
                raise ValueError("gp_matrix is required for GP regularization types.")
            gp_mat = jnp.asarray(gp_matrix)
            # Placeholders — shapes must be valid so the FD branch inside
            # lax.cond still traces successfully.
            rx_ph = jnp.zeros((self.nx, self.nx), dtype=gp_mat.dtype)
            ry_ph = jnp.zeros((self.ny, self.ny), dtype=gp_mat.dtype)
            return RegData(
                rx=rx_ph, ry=ry_ph,
                scale_x=jnp.array(1.0, dtype=gp_mat.dtype),
                scale_y=jnp.array(1.0, dtype=gp_mat.dtype),
                is_gp=jnp.array(True, dtype=bool),
                gp_matrix=gp_mat,
            )

        # Finite-difference types
        inv_dx = (self.nx - 1) / (xmax - xmin)
        inv_dy = (self.ny - 1) / (ymax - ymin)

        if self.regularization_type == "zero-order":
            # R = I  →  I = ½I ⊗ I_x + ½I_y ⊗ I  in the Kronecker form,
            # so that rx @ s_2d + s_2d @ ry^T = ½s_2d + ½s_2d = s_2d.
            rx = 0.5 * jnp.eye(self.nx, dtype=jnp.float32)
            ry = 0.5 * jnp.eye(self.ny, dtype=jnp.float32)
            scl_x = jnp.array(1.0, dtype=jnp.float32)
            scl_y = jnp.array(1.0, dtype=jnp.float32)
        elif self.regularization_type == "first-order":
            rx = self._Rx1
            ry = self._Ry1
            scl_x = jnp.array(inv_dx ** 2, dtype=rx.dtype)
            scl_y = jnp.array(inv_dy ** 2, dtype=ry.dtype)
        elif self.regularization_type == "second-order":
            rx = self._Rx2
            ry = self._Ry2
            scl_x = jnp.array(inv_dx ** 4, dtype=rx.dtype)
            scl_y = jnp.array(inv_dy ** 4, dtype=ry.dtype)
        else:
            raise RuntimeError(f"Unhandled type: {self.regularization_type!r}")

        # Placeholder for gp_matrix — shape (1, Ns) so that gp_matrix @ s traces
        # successfully even when is_gp=False (result is discarded by lax.cond).
        gp_ph = jnp.zeros((1, self.n_pixels), dtype=rx.dtype)

        return RegData(
            rx=rx, ry=ry,
            scale_x=scl_x, scale_y=scl_y,
            is_gp=jnp.array(False, dtype=bool),
            gp_matrix=gp_ph,
        )

    def _build_first_difference_operators(self):
        """Return first-order x/y finite-difference operators on index space.

        Uses vectorized index-based construction instead of per-element
        loops, giving O(1) JAX array operations regardless of grid size.
        Boundary rows use the Suyu et al. zero-order fallback (diagonal 1).
        """
        ix = jnp.arange(self.nx)
        iy = jnp.arange(self.ny)
        gx, gy = jnp.meshgrid(ix, iy, indexing='ij')
        flat_idx = (gy * self.nx + gx).ravel()

        # Interior x-differences: idx -> idx+1
        interior_x = gx < (self.nx - 1)
        interior_rows_x = flat_idx[interior_x.ravel()]
        interior_diag_x = interior_rows_x
        interior_off_x = interior_rows_x + 1

        # Boundary x (last column): diagonal 1
        boundary_x = gx == (self.nx - 1)
        boundary_rows_x = flat_idx[boundary_x.ravel()]

        dx_operator = jnp.zeros((self.n_pixels, self.n_pixels))
        dx_operator = dx_operator.at[interior_diag_x, interior_diag_x].add(-1.0)
        dx_operator = dx_operator.at[interior_rows_x, interior_off_x].add(1.0)
        dx_operator = dx_operator.at[boundary_rows_x, boundary_rows_x].add(1.0)

        # Interior y-differences: idx -> idx+nx
        interior_y = gy < (self.ny - 1)
        interior_rows_y = flat_idx[interior_y.ravel()]
        interior_diag_y = interior_rows_y
        interior_off_y = interior_rows_y + self.nx

        # Boundary y (last row): diagonal 1
        boundary_y = gy == (self.ny - 1)
        boundary_rows_y = flat_idx[boundary_y.ravel()]

        dy_operator = jnp.zeros((self.n_pixels, self.n_pixels))
        dy_operator = dy_operator.at[interior_diag_y, interior_diag_y].add(-1.0)
        dy_operator = dy_operator.at[interior_rows_y, interior_off_y].add(1.0)
        dy_operator = dy_operator.at[boundary_rows_y, boundary_rows_y].add(1.0)

        return dx_operator, dy_operator

    def _build_curvature_difference_operators(self):
        """Return second-order x/y finite-difference operators on index space.

        Uses vectorized index-based construction instead of per-element
        loops, giving O(1) JAX array operations regardless of grid size.
        Near-boundary curvature reduces to first gradient; outer boundary
        uses zero-order fallback (diagonal 1).
        """
        ix = jnp.arange(self.nx)
        iy = jnp.arange(self.ny)
        gx, gy = jnp.meshgrid(ix, iy, indexing='ij')
        flat_idx = (gy * self.nx + gx).ravel()

        # == X curvature operator ==
        # Full curvature (3-point stencil): ix < nx-2
        full_x = gx < (self.nx - 2)
        full_rows_x = flat_idx[full_x.ravel()]

        # Near-boundary (2-point first gradient): ix == nx-2
        near_x = gx == (self.nx - 2)
        near_rows_x = flat_idx[near_x.ravel()]

        # Outer boundary (diagonal): ix == nx-1
        outer_x = gx == (self.nx - 1)
        outer_rows_x = flat_idx[outer_x.ravel()]

        lx_operator = jnp.zeros((self.n_pixels, self.n_pixels))
        lx_operator = lx_operator.at[full_rows_x, full_rows_x].add(1.0)
        lx_operator = lx_operator.at[full_rows_x, full_rows_x + 1].add(-2.0)
        lx_operator = lx_operator.at[full_rows_x, full_rows_x + 2].add(1.0)
        lx_operator = lx_operator.at[near_rows_x, near_rows_x].add(-1.0)
        lx_operator = lx_operator.at[near_rows_x, near_rows_x + 1].add(1.0)
        lx_operator = lx_operator.at[outer_rows_x, outer_rows_x].add(1.0)

        # == Y curvature operator ==
        # Full curvature (3-point stencil): iy < ny-2
        full_y = gy < (self.ny - 2)
        full_rows_y = flat_idx[full_y.ravel()]

        # Near-boundary (2-point first gradient): iy == ny-2
        near_y = gy == (self.ny - 2)
        near_rows_y = flat_idx[near_y.ravel()]

        # Outer boundary (diagonal): iy == ny-1
        outer_y = gy == (self.ny - 1)
        outer_rows_y = flat_idx[outer_y.ravel()]

        ly_operator = jnp.zeros((self.n_pixels, self.n_pixels))
        ly_operator = ly_operator.at[full_rows_y, full_rows_y].add(1.0)
        ly_operator = ly_operator.at[full_rows_y, full_rows_y + self.nx].add(-2.0)
        ly_operator = ly_operator.at[full_rows_y, full_rows_y + 2 * self.nx].add(1.0)
        ly_operator = ly_operator.at[near_rows_y, near_rows_y].add(-1.0)
        ly_operator = ly_operator.at[near_rows_y, near_rows_y + self.nx].add(1.0)
        ly_operator = ly_operator.at[outer_rows_y, outer_rows_y].add(1.0)

        return lx_operator, ly_operator

    def _build_unit_coordinates(self):
        """Return source-grid coordinates for a unit half-size plane."""
        x_axis = jnp.linspace(-1.0, 1.0, self.nx)
        y_axis = jnp.linspace(-1.0, 1.0, self.ny)
        source_x_mesh, source_y_mesh = jnp.meshgrid(x_axis, y_axis, indexing='xy')
        return jnp.stack(
            [source_x_mesh.reshape(-1), source_y_mesh.reshape(-1)],
            axis=1,
        )

    def _first_order_matrix(self, xmin, xmax, ymin, ymax):
        """Return first-order gradient regularization with per-axis pixel scaling.

        H = H1_unit_x / dx² + H1_unit_y / dy²
        where dx = (xmax-xmin)/(nx-1), dy = (ymax-ymin)/(ny-1).
        """
        scale_x = 2.0 / (xmax - xmin)
        scale_y = 2.0 / (ymax - ymin)
        return self._H1_unit_x * (scale_x ** 2) + self._H1_unit_y * (scale_y ** 2)

    def _second_order_matrix(self, xmin, xmax, ymin, ymax):
        """Return second-order curvature regularization with per-axis pixel scaling.

        H = H2_unit_x / dx⁴ + H2_unit_y / dy⁴
        where dx = (xmax-xmin)/(nx-1), dy = (ymax-ymin)/(ny-1).
        """
        scale_x = 2.0 / (xmax - xmin)
        scale_y = 2.0 / (ymax - ymin)
        return self._H2_unit_x * (scale_x ** 4) + self._H2_unit_y * (scale_y ** 4)

    def _gp_matrix(self, xmin, xmax, ymin, ymax, kernel_scale: float):
        """Return (precision, logdet_covariance) for a GP regularization matrix.

        Builds physical pixel coordinates from the source-plane bounds and
        computes pairwise Euclidean distances in physical space.

        Uses Cholesky decomposition to compute both the precision matrix K^{-1}
        and log|K| in a single O(n^3) factorization, avoiding a separate slogdet call.

        Returns
        -------
        tuple[jax.Array, jax.Array]
            ``(precision, logdet_covariance)`` where precision is ``K^{-1}`` of shape
            ``(n_pixels, n_pixels)`` and ``logdet_covariance`` is the scalar ``log|K|``.
        """
        x_scale = (xmax - xmin) / 2.0
        y_scale = (ymax - ymin) / 2.0
        x_shift = (xmax + xmin) / 2.0
        y_shift = (ymax + ymin) / 2.0
        x_phys = self._unit_coordinates[:, 0] * x_scale + x_shift
        y_phys = self._unit_coordinates[:, 1] * y_scale + y_shift
        delta_x = x_phys[:, None] - x_phys[None, :]
        delta_y = y_phys[:, None] - y_phys[None, :]
        distances = jnp.sqrt(delta_x ** 2 + delta_y ** 2)
        r = distances / kernel_scale

        if self.regularization_type == "exponential":
            covariance = jnp.exp(-r)
        elif self.regularization_type == "gaussian":
            covariance = jnp.exp(-0.5 * r ** 2)
        elif self.regularization_type == "matern32":
            sqrt3_r = jnp.sqrt(3.0) * r
            covariance = (1.0 + sqrt3_r) * jnp.exp(-sqrt3_r)
        elif self.regularization_type == "matern52":
            sqrt5_r = jnp.sqrt(5.0) * r
            covariance = (1.0 + sqrt5_r + 5.0 * r ** 2 / 3.0) * jnp.exp(-sqrt5_r)
        elif self.regularization_type == "matern72":
            sqrt7_r = jnp.sqrt(7.0) * r
            covariance = (1.0 + sqrt7_r + 14.0 * r ** 2 / 5.0 + 7.0 * jnp.sqrt(7.0) * r ** 3 / 15.0) * jnp.exp(-sqrt7_r)
        else:
            raise RuntimeError(f"Unhandled GP regularization type: {self.regularization_type!r}")

        stabilized = covariance + self.jitter * self._identity
        # Single Cholesky factorization gives both precision and logdet(K).
        # logdet(K) = 2 * sum(log(diag(L)))  where L = chol(K)
        # K^{-1} via cho_solve avoids a second O(n^3) inv() call.
        chol_k = jnp.linalg.cholesky(stabilized)
        logdet_covariance = 2.0 * jnp.sum(jnp.log(jnp.diag(chol_k)))
        eye = jnp.eye(self.n_pixels, dtype=stabilized.dtype)
        precision = jsl.cho_solve((chol_k, True), eye)
        # Enforce exact symmetry — cho_solve may introduce tiny asymmetry
        # that can break downstream Cholesky factorizations.
        precision = 0.5 * (precision + precision.T)
        return precision, logdet_covariance

__all__ = ["DenseRegularizationBuilder", "RegData", "VALID_REGULARIZATION_TYPES", "GP_REGULARIZATION_TYPES"]
