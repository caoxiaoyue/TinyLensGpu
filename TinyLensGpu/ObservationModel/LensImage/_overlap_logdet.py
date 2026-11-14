"""Overlapping principal windows for operator source evidence."""

from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax


class _WindowGroup(NamedTuple):
    pixels: Array
    has_intersection: Array
    selectors: tuple[Array, Array, Array]


def _window_groups(n: int, tile: int) -> tuple[_WindowGroup, ...]:
    """Group maximal windows of equal shape, including the partial edge tiles."""
    edges = (*range(0, n, tile), n)
    if len(edges) <= 3:
        # One maximal window already covers the entire source grid.
        return ()

    nodes = np.arange(n * n, dtype=np.int32).reshape(n, n)
    groups = defaultdict(list)
    for row in range(len(edges) - 2):
        for col in range(len(edges) - 2):
            pixels = nodes[edges[row]:edges[row + 2], edges[col]:edges[col + 2]]
            groups[pixels.shape].append((pixels.ravel(), (row > 0, col > 0)))

    windows = []
    for shape, entries in groups.items():
        pixels, has_intersection = zip(*entries)
        local = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
        selectors = (local[:tile, :], local[:, :tile], local[:tile, :tile])
        windows.append(_WindowGroup(
            jnp.asarray(np.stack(pixels)),
            jnp.asarray(has_intersection),
            tuple(jnp.asarray(selector.ravel()) for selector in selectors),
        ))
    return tuple(windows)


class OverlapLogdet:
    """Compute source log determinant differences from exact principal blocks."""

    def __init__(self, simulator, builder, noise: Array, block_size: int, *, fixed_geometry=None):
        self._simulator = simulator
        self._builder = builder
        self._noise = noise
        self._n_source = simulator.n_source_pixels
        self._windows = _window_groups(simulator.source_n, block_size)
        self._fixed_grams = None
        self._response = simulator.psf_response

        if fixed_geometry is not None:
            bbox, mapping = fixed_geometry
            if self._windows:
                self._fixed_grams = tuple(
                    lax.map(lambda pixels: self._panel_gram(mapping, pixels), group.pixels)
                    for group in self._windows
                )
            else:
                self._fixed_grams = (self._full_gram(bbox, mapping),)

    def _panel_gram(self, mapping, pixels) -> Array:
        """Build and whiten only the image columns selected by this window."""
        width = pixels.size
        local = jnp.full(self._n_source, width, dtype=jnp.int32)
        local = local.at[pixels].set(jnp.arange(width, dtype=local.dtype))
        dtype = jnp.result_type(mapping.weights, self._noise, self._simulator.psf_kernel)
        whitened = self._response.convolved_columns(
            mapping.weights.astype(dtype), local[mapping.indices], width,
        ) / self._noise
        return jnp.matmul(whitened, whitened.T, precision=lax.Precision.HIGHEST)

    def _full_gram(self, bbox, mapping) -> Array:
        """Assemble a full-grid window without storing an image-by-source panel."""
        dtype = jnp.result_type(mapping.weights, self._noise, self._simulator.psf_kernel)

        def column(pixel):
            basis = jax.nn.one_hot(pixel, self._n_source, dtype=dtype)
            image = self._simulator.forward_model(basis, *bbox, op_data=mapping)
            return self._simulator.build_rhs(
                image, self._noise, *bbox, op_data=mapping,
            )

        # A window spanning the entire source needs the complete Gram. Apply
        # the data normal operator to bounded batches of individual basis vectors.
        columns = lax.map(
            jax.checkpoint(column, prevent_cse=False),
            jnp.arange(self._n_source),
            batch_size=min(16, self._n_source - 1),
        )
        return 0.5 * (columns + columns.T)

    @staticmethod
    def _principal_prior(pixels, diagonal, off_diagonals):
        row, col = pixels[:, None], pixels[None, :]
        first = jnp.minimum(row, col)
        distance = jnp.abs(row - col)
        prior = jnp.where(row == col, diagonal[first], 0.0)
        for offset, band in off_diagonals:
            prior = prior + jnp.where(distance == offset, band[first], 0.0)
        return prior

    @staticmethod
    def _difference(prior, posterior):
        prior_diag = jnp.diag(jnp.linalg.cholesky(prior))
        post_diag = jnp.diag(jnp.linalg.cholesky(posterior))
        return 2.0 * jnp.sum(jnp.log(post_diag) - jnp.log(prior_diag))

    def _group_difference(self, group, mapping, diagonal, bands, grams, dtype):
        """Accumulate exact principal-window differences without retaining panels."""
        def term(carry, entry):
            pixels, intersections, gram = entry
            if grams is None:
                gram = self._panel_gram(mapping, pixels)
            prior = self._principal_prior(pixels, diagonal, bands)
            posterior = prior + gram
            ratio = self._difference(prior, posterior)

            # The top and left intersections and their common corner
            # are principal submatrices of this same maximal window.
            top, left = intersections

            def intersection(selector):
                indices = (selector[:, None], selector[None, :])
                return self._difference(prior[indices], posterior[indices])

            for use, sign, selector in zip(
                (top, left, top & left), (-1, -1, 1), group.selectors,
            ):
                ratio += sign * lax.cond(
                    use, intersection, lambda _: jnp.zeros((), dtype=dtype), selector,
                )
            return carry + ratio, None

        return lax.scan(
            jax.checkpoint(term, prevent_cse=False),
            jnp.zeros((), dtype=dtype),
            (group.pixels, group.has_intersection, grams),
        )[0]

    def __call__(self, mapping, bbox, reg_data, strength: Array) -> Array:
        diagonal = strength * self._builder.diag_R(*bbox, scale=reg_data.scale)
        bands = self._builder.scaled_off_diagonal_bands(reg_data, strength)
        if not self._windows:
            gram = (
                self._full_gram(bbox, mapping)
                if self._fixed_grams is None else self._fixed_grams[0]
            )
            prior = self._principal_prior(jnp.arange(self._n_source), diagonal, bands)
            return self._difference(prior, prior + gram)

        dtype = jnp.result_type(
            diagonal, mapping.weights, self._noise, self._simulator.psf_kernel,
        )
        return sum(
            self._group_difference(
                group, mapping, diagonal, bands,
                None if self._fixed_grams is None else self._fixed_grams[index],
                dtype,
            )
            for index, group in enumerate(self._windows)
        )
