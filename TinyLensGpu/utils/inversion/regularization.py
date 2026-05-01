"""Dense regularization matrix builders for pixelized source inversion.

This module provides traditional finite-difference penalties and dense
Gaussian-process covariance penalties for source-plane linear inversion.  The
finite-difference operators are precomputed on index space and scaled by the
physical source-grid spacing when a matrix is requested.
"""

from __future__ import annotations

# pyright: reportMissingImports=false

import jax.numpy as jnp


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
        ``"first-order"``, ``"second-order"``, ``"exponential"``, ``"gaussian"``,
        and ``"gp"``.
        The ``"gp"`` alias selects the kernel through ``kernel_type``.
    kernel_type : str, optional
        GP kernel used when ``regularization_type="gp"``. Supported values are
        ``"exponential"`` and ``"gaussian"``.
    jitter : float, optional
        Diagonal jitter added to GP covariance matrices for numerical stability.

    Raises
    ------
    ValueError
        If the grid shape or regularization configuration is invalid.
    """

    _TRADITIONAL_TYPES = {"zero-order", "first-order", "second-order"}
    _GP_TYPES = {"exponential", "gaussian"}
    _VALID_TYPES = {"zero-order", "first-order", "second-order", "exponential", "gaussian", "gp"}

    def __init__(
        self,
        nx: int,
        ny: int,
        regularization_type: str,
        *,
        kernel_type: str = "gaussian",
        jitter: float = 1e-6,
    ) -> None:
        self.nx = int(nx)
        self.ny = int(ny)
        if self.nx < 2 or self.ny < 2:
            raise ValueError("nx and ny must both be at least 2.")

        self.n_pixels = self.nx * self.ny
        self.regularization_type = regularization_type.lower()
        self.kernel_type = kernel_type.lower()
        self.jitter = float(jitter)

        if self.regularization_type not in self._VALID_TYPES:
            raise ValueError(f"Unsupported regularization_type: {regularization_type!r}.")
        if self.kernel_type not in self._GP_TYPES:
            raise ValueError(f"Unsupported kernel_type: {kernel_type!r}.")
        if self.jitter < 0.0:
            raise ValueError("jitter must be non-negative.")

        self._identity = jnp.eye(self.n_pixels)
        self._dx_operator, self._dy_operator = self._build_first_difference_operators()
        self._lx_operator, self._ly_operator = self._build_curvature_difference_operators()
        self._unit_coordinates = self._build_unit_coordinates()

    def matrix(self, half_size: float, *, kernel_scale: float | None = None):
        """Return the dense regularization matrix for a physical grid size.

        Parameters
        ----------
        half_size : float
            Half-width of the square source plane. The physical grid spans
            ``[-half_size, half_size]`` in each axis.
        kernel_scale : float, optional
            GP kernel correlation scale. Required for GP regularization and
            ignored for traditional finite-difference penalties.

        Returns
        -------
        jax.Array
            Dense ``(nx * ny, nx * ny)`` regularization matrix.

        Raises
        ------
        ValueError
            If ``half_size`` is non-positive, or a GP kernel scale is missing or
            non-positive.
        """
        half_size = jnp.asarray(half_size)

        if self.regularization_type == "zero-order":
            return self._identity
        if self.regularization_type == "first-order":
            return self._first_order_matrix(half_size)
        if self.regularization_type == "second-order":
            return self._second_order_matrix(half_size)

        if kernel_scale is None or float(kernel_scale) <= 0.0:
            raise ValueError("kernel_scale must be provided and positive for GP regularization.")
        return self._gp_matrix(half_size, float(kernel_scale))

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
        source_x_mesh, source_y_mesh = jnp.meshgrid(x_axis, y_axis)
        return jnp.stack(
            [source_x_mesh.reshape(-1), source_y_mesh.reshape(-1)],
            axis=1,
        )

    def _first_order_matrix(self, half_size: float):
        """Return spacing-scaled first-order gradient regularization."""
        dx = 2.0 * half_size / (self.nx - 1)
        dy = 2.0 * half_size / (self.ny - 1)
        scaled_dx = self._dx_operator / dx
        scaled_dy = self._dy_operator / dy
        return scaled_dx.T @ scaled_dx + scaled_dy.T @ scaled_dy

    def _second_order_matrix(self, half_size: float):
        """Return spacing-scaled second-order curvature regularization."""
        dx = 2.0 * half_size / (self.nx - 1)
        dy = 2.0 * half_size / (self.ny - 1)
        scaled_lx = self._lx_operator / (dx**2)
        scaled_ly = self._ly_operator / (dy**2)
        return scaled_lx.T @ scaled_lx + scaled_ly.T @ scaled_ly

    def _gp_matrix(self, half_size: float, kernel_scale: float):
        """Return a dense GP precision (inverse covariance) regularization matrix."""
        coordinates = self._unit_coordinates * half_size
        delta = coordinates[:, None, :] - coordinates[None, :, :]
        distances = jnp.sqrt(jnp.sum(delta**2, axis=-1))

        kernel_name = self.kernel_type if self.regularization_type == "gp" else self.regularization_type
        if kernel_name == "exponential":
            covariance = jnp.exp(-distances / kernel_scale)
        else:
            covariance = jnp.exp(-0.5 * (distances / kernel_scale) ** 2)

        stabilized = covariance + self.jitter * self._identity
        return jnp.linalg.inv(stabilized)

    def _index(self, ix: int, iy: int) -> int:
        """Return flattened row-major index for grid coordinates."""
        return iy * self.nx + ix


__all__ = ["DenseRegularizationBuilder"]
