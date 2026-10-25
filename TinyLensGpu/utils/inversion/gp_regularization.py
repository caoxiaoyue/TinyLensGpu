"""Standalone Gaussian-process regularization matrix utility.

This module provides dense Gaussian-process covariance penalties for
source-plane linear inversion.  It is a standalone mathematical utility
and is not integrated with the pixelized-source model backends; the model
backends use :class:`FiniteDifferenceRegularization` instead.
"""

from __future__ import annotations

# pyright: reportMissingImports=false

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl

GP_REGULARIZATION_TYPES: frozenset[str] = frozenset({
    "exponential", "gaussian",
    "matern32", "matern52", "matern72",
})
"""GP-style regularization types that require a kernel_scale parameter."""


class GaussianProcessRegularization:
    """Build dense Gaussian-process source-grid regularization matrices.

    Parameters
    ----------
    n : int
        Number of source pixels per side (n x n square grid).
    regularization_type : str
        GP kernel family. Supported values are ``"exponential"``,
        ``"gaussian"``, ``"matern32"``, ``"matern52"``, and ``"matern72"``.
    jitter : float, optional
        Diagonal jitter added to the GP covariance matrix for numerical
        stability.

    Raises
    ------
    ValueError
        If the grid shape or regularization configuration is invalid.
    """

    _VALID_TYPES = GP_REGULARIZATION_TYPES

    def __init__(
        self,
        n: int,
        regularization_type: str,
        *,
        jitter: float = 1e-6,
    ) -> None:
        self.n = int(n)
        if self.n < 2:
            raise ValueError("n must be at least 2.")

        self.n_pixels = self.n * self.n
        self.regularization_type = regularization_type.lower()
        self.jitter = float(jitter)

        if self.regularization_type not in self._VALID_TYPES:
            raise ValueError(f"Unsupported regularization_type: {regularization_type!r}.")
        if self.jitter < 0.0:
            raise ValueError("jitter must be non-negative.")

        self._identity = jnp.eye(self.n_pixels)
        self._unit_coordinates = self._build_unit_coordinates()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_scale(
        self, scale: "jax.Array | None", expected_size: int | None = None,
    ) -> "jax.Array | None":
        """Return validated scale as ``(Ns,)`` or ``None`` for uniform scaling."""
        if scale is None:
            return None

        scale_arr = jnp.asarray(scale)
        expected_size = self.n_pixels if expected_size is None else int(expected_size)
        if scale_arr.shape != (expected_size,):
            raise ValueError(
                "scale must have shape "
                f"({expected_size},), got {scale_arr.shape}."
            )

        valid = jnp.all(jnp.isfinite(scale_arr) & (scale_arr > 0.0))
        try:
            is_valid = bool(valid)
        except jax.errors.TracerBoolConversionError:
            is_valid = True
        if not is_valid:
            raise ValueError("scale values must be finite and strictly positive.")
        return scale_arr

    def _apply_diag_scale(self, matrix: "jax.Array", scale: "jax.Array | None") -> "jax.Array":
        """If *scale* is not None, return ``diag(sqrt(scale)) @ matrix @ diag(sqrt(scale))``.

        Otherwise return *matrix* unchanged.
        """
        scale = self._check_scale(scale, int(matrix.shape[0]))
        if scale is not None:
            sqrt_scale = jnp.sqrt(scale)
            # Explicitly form the rank-1 outer-product scaling matrix first,
            # then apply it with a single element-wise multiply.  This avoids
            # chained broadcast-multiplies that can produce NaN in XLA's GEMM
            # fusion autotuner when the surrounding computation (kron + add)
            # is fused into a single kernel.
            scale_mat = sqrt_scale[:, None] * sqrt_scale[None, :]
            matrix = matrix * scale_mat
        return matrix

    def matrix(self, xmin, xmax, ymin, ymax, *, kernel_scale: float, scale: "jax.Array | None" = None) -> "tuple[jax.Array, jax.Array]":
        """Return GP precision R and logdet(R), optionally scaled per pixel.

        This standalone utility is independent of pixelized-source model backends.
        """
        scale = self._check_scale(scale)
        precision, covariance_logdet = self._gp_matrix(xmin, xmax, ymin, ymax, kernel_scale)
        logdet_precision = -covariance_logdet
        if scale is not None:
            precision = self._apply_diag_scale(precision, scale)
            logdet_precision = logdet_precision + jnp.sum(jnp.log(scale))
        return precision, logdet_precision

    def _build_unit_coordinates(self):
        """Return source-grid coordinates for a unit half-size plane."""
        x_axis = jnp.linspace(-1.0, 1.0, self.n)
        y_axis = jnp.linspace(-1.0, 1.0, self.n)
        source_x_mesh, source_y_mesh = jnp.meshgrid(x_axis, y_axis, indexing='xy')
        return jnp.stack(
            [source_x_mesh.reshape(-1), source_y_mesh.reshape(-1)],
            axis=1,
        )

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


__all__ = ["GaussianProcessRegularization", "GP_REGULARIZATION_TYPES"]
