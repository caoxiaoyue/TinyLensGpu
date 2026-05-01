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
        Regularization family. Supported values are ``"zero"``, ``"first"``,
        ``"curvature"``, ``"exponential"``, ``"gaussian"``, and ``"gp"``.
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

    _TRADITIONAL_TYPES = {"zero", "first", "curvature"}
    _GP_TYPES = {"exponential", "gaussian"}
    _VALID_TYPES = _TRADITIONAL_TYPES | _GP_TYPES | {"gp"}

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

        if self.regularization_type == "zero":
            return self._identity
        if self.regularization_type == "first":
            return self._first_matrix(half_size)
        if self.regularization_type == "curvature":
            return self._curvature_matrix(half_size)

        if kernel_scale is None or float(kernel_scale) <= 0.0:
            raise ValueError("kernel_scale must be provided and positive for GP regularization.")
        return self._gp_matrix(half_size, float(kernel_scale))

    def _build_first_difference_operators(self):
        """Return first-order x/y finite-difference operators on index space."""
        dx_operator = jnp.zeros((self.n_pixels, self.n_pixels))
        dy_operator = jnp.zeros((self.n_pixels, self.n_pixels))

        for iy in range(self.ny):
            for ix in range(self.nx):
                idx = self._index(ix, iy)
                if ix < self.nx - 1:
                    dx_operator = dx_operator.at[idx, idx].set(-1.0)
                    dx_operator = dx_operator.at[idx, self._index(ix + 1, iy)].set(1.0)
                else:
                    # Suyu et al. boundary condition: first-order boundary rows
                    # fall back to a zero-order diagonal penalty.
                    dx_operator = dx_operator.at[idx, idx].set(1.0)

                if iy < self.ny - 1:
                    dy_operator = dy_operator.at[idx, idx].set(-1.0)
                    dy_operator = dy_operator.at[idx, self._index(ix, iy + 1)].set(1.0)
                else:
                    dy_operator = dy_operator.at[idx, idx].set(1.0)

        return dx_operator, dy_operator

    def _build_curvature_difference_operators(self):
        """Return second-order x/y finite-difference operators on index space."""
        lx_operator = jnp.zeros((self.n_pixels, self.n_pixels))
        ly_operator = jnp.zeros((self.n_pixels, self.n_pixels))

        for iy in range(self.ny):
            for ix in range(self.nx):
                idx = self._index(ix, iy)
                if ix < self.nx - 2:
                    lx_operator = lx_operator.at[idx, idx].set(1.0)
                    lx_operator = lx_operator.at[idx, self._index(ix + 1, iy)].set(-2.0)
                    lx_operator = lx_operator.at[idx, self._index(ix + 2, iy)].set(1.0)
                elif ix < self.nx - 1:
                    # Near-boundary curvature reduces to a first gradient.
                    lx_operator = lx_operator.at[idx, idx].set(-1.0)
                    lx_operator = lx_operator.at[idx, self._index(ix + 1, iy)].set(1.0)
                else:
                    # Outer boundary falls back to a zero-order penalty.
                    lx_operator = lx_operator.at[idx, idx].set(1.0)

                if iy < self.ny - 2:
                    ly_operator = ly_operator.at[idx, idx].set(1.0)
                    ly_operator = ly_operator.at[idx, self._index(ix, iy + 1)].set(-2.0)
                    ly_operator = ly_operator.at[idx, self._index(ix, iy + 2)].set(1.0)
                elif iy < self.ny - 1:
                    ly_operator = ly_operator.at[idx, idx].set(-1.0)
                    ly_operator = ly_operator.at[idx, self._index(ix, iy + 1)].set(1.0)
                else:
                    ly_operator = ly_operator.at[idx, idx].set(1.0)

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

    def _first_matrix(self, half_size: float):
        """Return spacing-scaled first-gradient regularization."""
        dx = 2.0 * half_size / (self.nx - 1)
        dy = 2.0 * half_size / (self.ny - 1)
        scaled_dx = self._dx_operator / dx
        scaled_dy = self._dy_operator / dy
        return scaled_dx.T @ scaled_dx + scaled_dy.T @ scaled_dy

    def _curvature_matrix(self, half_size: float):
        """Return spacing-scaled curvature regularization."""
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
