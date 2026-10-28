"""Compact complete fourth differences on local potential unknowns.

The smoothing differences use pixel units. The positive amplitude precision
is an independent physical prior on every coefficient, including nodes that
do not occur in any complete window. No boundary-order fallback is used.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from TinyLensGpu.utils.lensing.potential import PotentialGrid


class PotentialRegularization:
    """Prepare complete horizontal/vertical five-node windows without matrices.

    ``horizontal_windows`` and ``vertical_windows`` contain unknown-vector
    indices. ``window_coverage`` counts every occurrence of each unknown in
    either direction. Dense storage is created only by :meth:`to_dense`.
    """

    def __init__(self, grid: PotentialGrid):
        self.grid = grid
        self.dtype = grid.x.dtype
        nodes = np.asarray(grid.node_indices)
        lookup = {tuple(node): i for i, node in enumerate(nodes)}
        windows = []
        for dx, dy in ((1, 0), (0, 1)):
            direction = []
            for x, y in nodes:
                window = [(x + step * dx, y + step * dy) for step in range(5)]
                if all(node in lookup for node in window):
                    direction.append([lookup[node] for node in window])
            windows.append(np.asarray(direction, dtype=np.int32).reshape(-1, 5))
        combined = np.concatenate(windows, axis=0)
        self.horizontal_windows = jnp.asarray(windows[0])
        self.vertical_windows = jnp.asarray(windows[1])
        self.windows = jnp.asarray(combined)
        self.window_coverage = jnp.asarray(
            np.bincount(combined.ravel(), minlength=grid.n_unknown), dtype=jnp.int32
        )
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            self.coefficients = jnp.asarray([1, -4, 6, -4, 1], dtype=self.dtype)

    def forward(self, potential_values):
        """Return all pixel-unit fourth differences, horizontal then vertical."""
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            values = self.grid._values(potential_values)
            return jnp.sum(values[self.windows] * self.coefficients, axis=1)

    def adjoint(self, differences):
        """Transpose fourth differences, accumulating on every fitted node."""
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            values = jnp.asarray(differences, dtype=self.dtype)
            if values.shape != (self.windows.shape[0],):
                raise ValueError("differences must match the complete window count")
            return (
                jnp.zeros(self.grid.n_unknown, dtype=self.dtype)
                .at[self.windows.ravel()]
                .add((values[:, None] * self.coefficients).ravel())
            )

    def matvec(self, potential_values, lambda_potential, lambda_amplitude):
        """Apply ``lambda_potential * D4.T D4 + lambda_amplitude * I``."""
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            values = self.grid._values(potential_values)
            return (
                jnp.asarray(lambda_potential, dtype=self.dtype)
                * self.adjoint(self.forward(values))
                + jnp.asarray(lambda_amplitude, dtype=self.dtype) * values
            )

    def to_dense(self, lambda_potential, lambda_amplitude):
        """Explicitly materialize the complete amplitude-augmented precision."""
        with jax.experimental.enable_x64(self.dtype == jnp.float64):
            difference = (
                jnp.zeros(
                    (self.windows.shape[0], self.grid.n_unknown), dtype=self.dtype
                )
                .at[jnp.arange(self.windows.shape[0])[:, None], self.windows]
                .set(jnp.broadcast_to(self.coefficients, self.windows.shape))
            )
            return jnp.asarray(lambda_potential, dtype=self.dtype) * (
                difference.T @ difference
            ) + jnp.asarray(lambda_amplitude, dtype=self.dtype) * jnp.eye(
                self.grid.n_unknown, dtype=self.dtype
            )
