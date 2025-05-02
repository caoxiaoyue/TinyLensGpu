import jax
import jax.numpy as jnp

@jax.jit
def fnnls_jax(Z, x, epsilon=None):
    """
    JAX implementation of the Fast Non-Negative Least Squares (FNNLS) algorithm.

    Parameters
    ----------
    Z: jnp.ndarray
        m x n matrix.
    x: jnp.ndarray
        m vector.
    epsilon: float or None
        Numerical tolerance. If None, uses jnp.finfo(float).eps.

    Returns
    -------
    d: jnp.ndarray
        n vector, solution to min ||x - Zd|| s.t. d >= 0
    res: float
        Residual norm ||x - Zd||
    """
    m, n = Z.shape
    # Ensure Z and x are float arrays
    Z = Z.astype(jnp.float32)
    x = x.astype(jnp.float32)
    if epsilon is None:
        epsilon = jnp.finfo(jnp.float32).eps

    ZTZ = Z.T @ Z
    ZTx = Z.T @ x

    tolerance = epsilon * n
    max_repetitions = 5

    def body_fun(state):
        d, s, P, w, no_update, _ = state
        current_P = P

        # B2 + B3: Move the element in active set with largest value of w into the passive set
        idx = jnp.argmax(jnp.where(~P, w, -jnp.inf))
        P = P.at[idx].set(True)

        # B4: Set s to the least squares solution along the passive set
        def masked_lstsq(ZTZ, ZTx, P):
            # Masked solve: for inactive variables, set ZTZ to identity and ZTx to zero
            mask = P.astype(ZTZ.dtype)
            eye = jnp.eye(ZTZ.shape[0], dtype=ZTZ.dtype)
            ZTZ_masked = ZTZ * (mask[None, :] * mask[:, None]) + eye * (1.0 - mask[None, :] * mask[:, None])
            ZTx_masked = ZTx * mask
            return jnp.linalg.solve(ZTZ_masked, ZTx_masked)
        s = masked_lstsq(ZTZ, ZTx, P)

        # C1: Loop until all s[P] > tolerance
        def c1_cond_fn(carry):
            s, d, P = carry
            return jnp.any(P) & (jnp.min(jnp.where(P, s, jnp.inf)) <= tolerance)

        def c1_body_fn(carry):
            s, d, P = carry
            # C2: Find largest alpha such that d + alpha(s-d) >= 0 for indices where s <= tolerance
            q = P & (s <= tolerance)
            safe = jnp.where(q, d / (d - s + 1e-12), jnp.inf)
            alpha = jnp.min(safe)
            alpha = jnp.where(jnp.isfinite(alpha), alpha, 0.0)
            # C3: Update d
            d = d + alpha * (s - d)
            # C4: Move elements with d <= tolerance to active set
            P = P & (d > tolerance)
            # C5: Update s
            s = masked_lstsq(ZTZ, ZTx, P)
            # C6: Set s[~P] = 0
            s = s * P.astype(s.dtype)
            return (s, d, P)

        s, d, P = jax.lax.while_loop(c1_cond_fn, c1_body_fn, (s, d, P))

        # B5: d = s
        d = s
        # B6: w = ZTx - ZTZ @ d
        w = ZTx - ZTZ @ d

        # Check if there has been a change to the passive set
        no_update = jnp.where(jnp.all(current_P == P), no_update + 1, 0)
        return (d, s, P, w, no_update, 0)

    def cond_fun(state):
        d, s, P, w, no_update, _ = state
        return (~jnp.all(P)) & (jnp.max(jnp.where(~P, w, -jnp.inf)) > tolerance) & (no_update < max_repetitions)

    # Initializations
    P = jnp.zeros(n, dtype=bool)
    d = jnp.zeros(n, dtype=Z.dtype)
    s = jnp.zeros(n, dtype=Z.dtype)
    w = ZTx - ZTZ @ d
    no_update = 0

    # Main loop
    state = (d, s, P, w, no_update, 0)
    d, s, P, w, no_update, _ = jax.lax.while_loop(cond_fun, body_fun, state)

    res = jnp.linalg.norm(x - Z @ d)
    return d, res


@jax.jit
def fnnls_jax_vec(Z, x):
    """
    Vectorized Fast Non-Negative Least Squares solver using JAX's vmap.
    
    This implementation vectorizes the original fnnls function over the batch dimension
    using JAX's vmap functionality.
    
    Args:
        Z: Matrix of shape (m, n, b) where b is the batch size
        x: Vector of shape (m, b)
    Returns:
        Tuple of:
        - Solution vector of shape (n, b)
        - Residual norms of shape (b,)
    """
    # Use vmap to vectorize over the batch dimension
    # in_axes=(-1, -1) means the last axis of both Z and x is the batch dimension
    # out_axes=-1 means the output will have the batch dimension as the last axis
    return jax.vmap(fnnls_jax, in_axes=(-1, -1), out_axes=-1)(Z, x) 


fnnls_vec = fnnls_jax_vec