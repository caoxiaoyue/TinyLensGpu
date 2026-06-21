"""Preconditioned Conjugate Gradient solver (JAX / matrix-free).

Implements standard PCG with a Cholesky-factorized preconditioner.
Designed to work inside a ``jax.jit`` context (e.g., from ``make_likelihood``).

To avoid recompilation, the A-matrix operator is split into two parts:

* ``A_data`` — a tuple-of-arrays (traced normally by jit).
* ``_A_jit_prebound`` — a ``functools.partial`` (static, created once at init).
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jax import Array, lax


class PCGState(NamedTuple):
    """Carry state for the PCG while_loop."""
    x: Array
    r: Array
    z: Array
    p: Array
    rz_old: Array
    k: Array
    converged: Array
    failed: Array


class PCGInfo(NamedTuple):
    """Convergence diagnostics returned by :func:`pcg_solve`."""
    n_iter: Array
    residual_norm: Array
    converged: Array   # True if residual_norm < tol
    failed: Array      # True iff solver aborted (e.g. non-positive curvature)


@partial(jax.jit, static_argnames=("_A_jit_prebound", "max_iter", "rtol", "atol"))
def pcg_solve(
    A_data: tuple,
    b: Array,
    preconditioner,
    _A_jit_prebound,
    x0: Array | None = None,
    max_iter: int = 200,
    rtol: float = 1e-6,
    atol: float = 1e-12,
) -> tuple[Array, PCGInfo]:
    r"""Solve ``A @ x = b`` with preconditioned conjugate gradient.

    Parameters
    ----------
    A_data : tuple of Arrays
        ``(weights, indices, flat_indices, agg_seg, psf_fft, psf_fft_conj,
          noise_var, reg_data, lambda_reg)``.
        ``reg_data`` is a :class:`~TinyLensGpu.utils.inversion.regularization.RegData`
        tuple — compact descriptor for matrix-free or GP regularisation.
    b : Array, shape ``(N,)``
    preconditioner : Array or tuple[list[Array], list[Array]]
        Either a dense Cholesky lower factor ``P_chol_lower`` of shape
        ``(N, N)`` (legacy), or a block-diagonal tuple
        ``(block_chols, block_masks)`` where ``block_chols[i]`` is the
        Cholesky factor for the i-th block and ``block_masks[i]`` holds
        the global flat source indices belonging to that block.
    _A_jit_prebound : callable (static)
        ``functools.partial`` of ``_A_matvec_jit`` with static ints bound.
        Created once at ``PixelizedLensOperator.__init__`` — its identity
        never changes, so jit caching works across lens-parameter updates.
    x0, max_iter, rtol, atol : optional

    Returns
    -------
    tuple[Array, PCGInfo]
    """
    # Unpack A_data
    (weights, indices, flat_indices, agg_seg,
     psf_fft, psf_fft_conj, noise_var, reg_data, lambda_reg) = A_data

    b = jnp.asarray(b)
    N = b.shape[0]

    x = jnp.zeros(N, dtype=b.dtype) if x0 is None else jnp.asarray(x0, dtype=b.dtype)

    def _A_vec(s: Array) -> Array:
        return _A_jit_prebound(
            s, weights, indices, flat_indices,
            agg_segment_ids=agg_seg, psf_fft=psf_fft,
            psf_fft_conj=psf_fft_conj,
            noise_var=noise_var, reg_data=reg_data, lambda_reg=lambda_reg,
        )

    # Dispatch preconditioner type
    if isinstance(preconditioner, tuple):
        # Block-diagonal: (block_chols, block_masks)
        block_chols, block_masks = preconditioner

        def _preconditioner_solve(r: Array) -> Array:
            z = jnp.zeros_like(r)
            for chol, mask in zip(block_chols, block_masks):
                r_block = r[mask]
                w = jsl.solve_triangular(chol, r_block, lower=True)
                z_block = jsl.solve_triangular(chol.T, w, lower=False)
                z = z.at[mask].set(z_block)
            return z
    else:
        # Dense Cholesky (legacy)
        P_chol_lower = preconditioner

        def _preconditioner_solve(r: Array) -> Array:
            w = jsl.solve_triangular(P_chol_lower, r, lower=True)
            return jsl.solve_triangular(P_chol_lower.T, w, lower=False)

    r = b - _A_vec(x)
    z = _preconditioner_solve(r)
    p = z
    rz_old = jnp.dot(r, z)

    tol = rtol * jnp.linalg.norm(b) + atol

    init_state = PCGState(
        x=x, r=r, z=z, p=p, rz_old=rz_old,
        k=jnp.array(0, dtype=jnp.int32),
        converged=jnp.linalg.norm(r) < tol,
        failed=jnp.array(False, dtype=bool),
    )

    def cond_fn(state: PCGState) -> Array:
        return (~state.converged) & (~state.failed) & (state.k < max_iter)

    def body_fn(state: PCGState) -> PCGState:
        Ap = _A_vec(state.p)
        pAp = jnp.dot(state.p, Ap)

        # Non-positive curvature → system is not SPD (numerical breakdown).
        # Conventional CG theory guarantees pAp > 0 for a true SPD operator;
        # a non-positive value signals that further iterations are futile.
        broken = pAp <= 0

        safe_pAp = jnp.where(pAp > 0, pAp, 1.0)
        alpha = jnp.where(pAp > 0, state.rz_old / safe_pAp, 0.0)

        x_new = state.x + alpha * state.p
        r_new = state.r - alpha * Ap

        z_new = _preconditioner_solve(r_new)
        rz_new = jnp.dot(r_new, z_new)

        beta = jnp.where(state.rz_old > 0, rz_new / state.rz_old, 0.0)
        p_new = z_new + beta * state.p

        r_norm = jnp.linalg.norm(r_new)
        converged = r_norm < tol
        failed = broken

        return PCGState(
            x=x_new, r=r_new, z=z_new, p=p_new, rz_old=rz_new,
            k=state.k + 1, converged=converged, failed=failed,
        )

    final_state = lax.while_loop(cond_fn, body_fn, init_state)

    info = PCGInfo(
        n_iter=final_state.k,
        residual_norm=jnp.linalg.norm(final_state.r),
        converged=final_state.converged,
        failed=final_state.failed,
    )
    return final_state.x, info


__all__ = ["pcg_solve", "PCGInfo", "PCGState"]
