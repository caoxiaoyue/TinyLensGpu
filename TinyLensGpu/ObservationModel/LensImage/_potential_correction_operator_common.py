"""Invariant bounded-block preparation for source–potential operator fits.

These helpers have no fixed/current Jacobian or iteration policy. Per-block
math is called inside each adapter's existing scan, so sharing it does not
allocate a complete field basis or keep image panels between blocks.
"""

import hashlib

import jax
import jax.numpy as jnp
import numpy as np


def _positive_integer(value, name):
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if isinstance(value, bool) or integer != value or integer < 1:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _partition(problem, block_size, block_indices):
    block_size = _positive_integer(block_size, "block_size")
    if block_indices is None:
        blocks = []
        for start, size in (
            (0, problem.n_source),
            (problem.n_source, problem.n_potential),
        ):
            # Even a small field is split: preparation never probes a full
            # source/potential/joint basis in one panel.
            width = min(block_size, max(1, (size + 1) // 2))
            blocks.extend(
                tuple(range(offset, min(offset + width, start + size)))
                for offset in range(start, start + size, width)
            )
        layout = "field_separated_contiguous_v1"
    else:
        blocks = []
        try:
            for block in block_indices:
                values = np.asarray(block)
                if (
                    values.ndim != 1
                    or values.size == 0
                    or not np.issubdtype(values.dtype, np.integer)
                ):
                    raise ValueError
                blocks.append(tuple(int(value) for value in values))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "block_indices must contain nonempty integer index blocks"
            ) from exc
        layout = "explicit_joint_partition_v1"
    flat = [index for block in blocks for index in block]
    if sorted(flat) != list(range(problem.n_joint)):
        raise ValueError(
            "block_indices must cover every joint coefficient exactly once"
        )
    if any(len(block) > block_size for block in blocks):
        raise ValueError("Each block must contain at most block_size coefficients")
    if any(
        len(block) == problem.n_joint
        or sum(index >= problem.n_source for index in block) == problem.n_potential
        for block in blocks
    ):
        raise ValueError(
            "Blocks must not contain the complete joint or potential field"
        )
    return tuple(blocks), layout, block_size


def _padded_partition(blocks):
    """Prepare fixed indices/masks and the identical public partition digest."""
    width = max(map(len, blocks))
    indices = np.zeros((len(blocks), width), dtype=np.int32)
    valid = np.zeros_like(indices, dtype=bool)
    for row, block in enumerate(blocks):
        indices[row, : len(block)] = block
        valid[row, : len(block)] = True
    digest = hashlib.sha256(repr(blocks).encode("ascii")).hexdigest()
    return jnp.asarray(indices), jnp.asarray(valid), digest


def _block_basis(problem, indices, valid):
    """One bounded panel with zero rows at padded positions."""
    return (
        jax.nn.one_hot(indices, problem.n_joint, dtype=problem.dtype) * valid[:, None]
    )


def _unit_prior_blocks(problem, basis, indices, valid):
    """Unscaled source/fourth-difference principal blocks for one scan step."""

    def unit_prior(values):
        return jnp.concatenate(
            (
                problem.source_regularizer.matvec_free(
                    values[: problem.n_source],
                    *problem.renderer.source_bbox,
                    scale=problem.source_scale,
                ),
                problem.potential_regularizer.matvec(values[problem.n_source :], 1, 0),
            )
        )

    principal = jax.vmap(unit_prior)(basis)[:, indices]
    principal = principal * (valid[:, None] & valid[None, :])
    source = indices < problem.n_source
    return (
        principal * (source[:, None] & source[None, :]),
        principal * (~source[:, None] & ~source[None, :]),
    )


def _assemble_prior_blocks(
    problem, source, potential, indices, valid, lambda_source, lambda_potential
):
    """Complete physical priors on valid coefficients; identity on padding."""
    identity = jnp.eye(indices.shape[1], dtype=problem.dtype)[None, :, :]
    amplitude = (indices >= problem.n_source) & valid
    return (
        lambda_source * source
        + lambda_potential * potential
        + jnp.asarray(problem.lambda_amplitude, dtype=problem.dtype)
        * amplitude[:, :, None]
        * identity
        + (~valid)[:, :, None] * identity
    )


def _factor_blocks(matrices):
    """Equilibrated Cholesky and validity without a prior or jitter change."""
    scale = jnp.sqrt(jnp.diagonal(matrices, axis1=1, axis2=2))
    normalized = matrices / (scale[:, :, None] * scale[:, None, :])
    chols = jnp.linalg.cholesky(normalized) * scale[:, :, None]
    diagonal = jnp.diagonal(chols, axis1=1, axis2=2)
    logdet = 2 * jnp.sum(jnp.log(diagonal))
    positive = (
        jnp.all(jnp.isfinite(matrices))
        & jnp.all(jnp.isfinite(chols))
        & jnp.all(diagonal > 0)
        & jnp.isfinite(logdet)
    )
    return chols, logdet, positive


def _pcg_controls(problem, max_iter, rtol, atol):
    """Validate the existing shared controls and precision-dependent defaults."""
    max_iter = _positive_integer(max_iter, "pcg_max_iter")
    if rtol is None:
        rtol = min(
            problem.solve_rtol * 0.1, 1e-10 if problem.dtype == jnp.float64 else 1e-5
        )
    if not np.isfinite(rtol) or rtol <= 0:
        raise ValueError("pcg_rtol must be finite and positive")
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("pcg_atol must be finite and nonnegative")
    return max_iter, float(rtol), float(atol)
