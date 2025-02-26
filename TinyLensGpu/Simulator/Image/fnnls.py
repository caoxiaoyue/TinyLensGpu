import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Any

@jax.jit
def fix_constraint(ZTZ: jnp.ndarray, 
                  ZTx: jnp.ndarray, 
                  s: jnp.ndarray, 
                  d: jnp.ndarray, 
                  P: jnp.ndarray, 
                  tolerance: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JAX version of fix_constraint function."""
    # Find indices where s <= tolerance in passive set
    q = jnp.logical_and(P, s <= tolerance)
    
    # Calculate alpha
    alpha = jnp.min(jnp.where(q, d / (d - s + 1e-10), jnp.inf))
    
    # Update d
    d = d + alpha * (s - d)
    
    # Update passive set
    P = jnp.logical_and(P, d > tolerance)
    
    # Solve least squares for passive set using masked operations
    def solve_passive(P, ZTZ, ZTx):
        n = ZTZ.shape[0]
        mask = P.astype(jnp.float32)
        # Create masked matrices
        ZTZ_masked = ZTZ * mask.reshape(-1, 1) * mask.reshape(1, -1)
        ZTx_masked = ZTx * mask
        # Add small diagonal term to ensure invertibility when masked
        diag_mask = jnp.eye(n) * (1 - mask) * 1e-10
        ZTZ_masked = ZTZ_masked + diag_mask
        # Solve system
        sol = jnp.linalg.solve(ZTZ_masked + diag_mask, ZTx_masked)
        return sol * mask
    
    s = solve_passive(P, ZTZ, ZTx)
    
    return s, d, P

@partial(jax.jit, static_argnames=('epsilon',))
def fnnls(Z: jnp.ndarray, 
          x: jnp.ndarray, 
          P_initial: jnp.ndarray = jnp.zeros(0, dtype=jnp.int32),
          epsilon: float = jnp.finfo(jnp.float32).eps) -> Tuple[jnp.ndarray, float]:
    """JAX version of Fast Non-negative Least Squares (FNNLS) algorithm."""
    
    m, n = Z.shape
    tolerance = epsilon * n
    max_repetitions = 5
    
    # Initialize arrays
    P = jnp.zeros(n, dtype=bool).at[P_initial].set(True)
    d = jnp.zeros(n)
    s = jnp.zeros(n)
    
    # Precompute matrices
    ZTZ = Z.T @ Z
    ZTx = Z.T @ x
    
    # Initialize if P_initial is not empty
    def init_step(P, s, d):
        mask = P.astype(jnp.float32)
        # Create masked matrices
        ZTZ_masked = ZTZ * mask.reshape(-1, 1) * mask.reshape(1, -1)
        ZTx_masked = ZTx * mask
        # Add small diagonal term for numerical stability
        diag_mask = jnp.eye(n) * (1 - mask) * 1e-10
        # Solve system
        sol = jnp.linalg.solve(ZTZ_masked + diag_mask, ZTx_masked)
        s = sol * mask
        d = jnp.clip(s, 0, None)
        return s, d
    
    s, d = jax.lax.cond(
        P_initial.size > 0,
        lambda x: init_step(*x),
        lambda x: x[1:],
        (P, s, d)
    )
    
    # Main loop
    def cond_fun(state):
        P, _, _, w, no_update, _ = state
        return jnp.logical_and(
            jnp.logical_and(~jnp.all(P), jnp.max(w * ~P) > tolerance),
            no_update < max_repetitions
        )
    
    def body_fun(state):
        P, s, d, w, no_update, _ = state
        
        # Store current P for change detection
        current_P = P
        
        # Update P with argmax of w in active set
        P = P.at[jnp.argmax(w * ~P)].set(True)
        
        # Solve least squares for passive set using masked operations
        mask = P.astype(jnp.float32)
        ZTZ_masked = ZTZ * mask.reshape(-1, 1) * mask.reshape(1, -1)
        ZTx_masked = ZTx * mask
        diag_mask = jnp.eye(n) * (1 - mask) * 1e-10
        sol = jnp.linalg.solve(ZTZ_masked + diag_mask, ZTx_masked)
        s = sol * mask
        
        # Inner loop for constraint fixing
        def inner_cond(state):
            P, s, _, _ = state
            return jnp.logical_and(jnp.any(P), jnp.min(jnp.where(P, s, jnp.inf)) <= tolerance)
        
        def inner_body(state):
            P, s, d, _ = state
            s, d, P = fix_constraint(ZTZ, ZTx, s, d, P, tolerance)
            return P, s, d, None
        
        P, s, d, _ = jax.lax.while_loop(inner_cond, inner_body, (P, s, d, None))
        
        # Update d and w
        d = s
        w = ZTx - ZTZ @ d
        
        # Update no_update counter
        no_update = jax.lax.cond(
            jnp.all(current_P == P),
            lambda x: x + 1,
            lambda x: 0,
            no_update
        )
        
        return P, s, d, w, no_update, None
    
    # Initialize w
    w = ZTx - ZTZ @ d
    
    # Run main loop
    P, s, d, w, _, _ = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (P, s, d, w, 0, None)
    )
    
    # Calculate residual
    res = jnp.linalg.norm(x - Z @ d)
    
    return d, res

@partial(jax.jit, static_argnames=('epsilon',))
def fnnls_vec(Z: jnp.ndarray, 
              x: jnp.ndarray, 
              P_initial: jnp.ndarray = jnp.zeros(0, dtype=jnp.int32),
              epsilon: float = jnp.finfo(jnp.float32).eps) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Vectorized Fast Non-Negative Least Squares solver with fixed number of iterations.
    
    This implementation uses 50 fixed iterations instead of a while loop for better batch processing.
    Each iteration improves the solution, and 50 iterations should be sufficient for most cases.
    
    Args:
        Z: Matrix of shape (m, n, b) where b is the batch size
        x: Vector of shape (m, b)
        P_initial: Initial passive set (optional)
        epsilon: Small number for numerical stability
    
    Returns:
        Tuple of:
        - Solution vector of shape (n, b)
        - Residual norms of shape (b,)
    """
    m, n, b = Z.shape
    ZTZ = jnp.einsum('mnb,mkb->nkb', Z, Z)  # n x n x b
    ZTx = jnp.einsum('mnb,mb->nb', Z, x)    # n x b
    
    def init_step(P, s, d):
        # Convert boolean mask to float for computations
        mask = P.astype(jnp.float32)  # n x b
        
        # Create diagonal mask for numerical stability
        diag_mask = jnp.eye(n)[..., None] * epsilon  # n x n x 1
        
        # Mask the matrices
        ZTZ_masked = ZTZ * mask[None, :, :] * mask[:, None, :]  # n x n x b
        ZTx_masked = ZTx * mask  # n x b
        
        # Solve systems
        ZTZ_reshaped = ZTZ_masked.transpose(2, 0, 1) + diag_mask.transpose(2, 0, 1)  # b x n x n
        ZTx_reshaped = ZTx_masked.T[..., None]  # b x n x 1
        sol = jnp.linalg.solve(ZTZ_reshaped, ZTx_reshaped)  # b x n x 1
        s = sol.squeeze(-1).T * mask  # n x b
        
        return P, s, d, ZTx - jnp.einsum('nkb,kb->nb', ZTZ, d)  # Return w as well

    # Initialize variables
    P = jnp.zeros((n, b), dtype=jnp.int32)  # n x b
    if P_initial.size > 0:
        P = P.at[P_initial].set(1)
    
    s = jnp.zeros((n, b))  # n x b
    d = jnp.zeros((n, b))  # n x b
    
    # Initial setup
    P, s, d, w = init_step(P, s, d)
    
    def iteration_step(state):
        P, s, d, w = state
        
        # Find maximum violating constraint
        w_masked = jnp.where(P == 1, -jnp.inf, w)
        max_idx = jnp.argmax(w_masked, axis=0)  # b
        max_val = jnp.take_along_axis(w_masked, max_idx[None, :], axis=0)[0]  # b
        
        # Update passive set where constraint is violated
        update_mask = (max_val > epsilon)  # b
        P = P.at[max_idx, jnp.arange(b)].set(jnp.where(update_mask, 1, P[max_idx, jnp.arange(b)]))
        
        # Solve the constrained problem
        P, s, d, w = init_step(P, s, d)
        
        # Handle negative components
        negative_mask = (s < 0) & (P == 1)
        alpha = jnp.where(negative_mask, d / (d - s + epsilon), 1.0)
        alpha = jnp.min(jnp.where(negative_mask, alpha, jnp.inf), axis=0)  # b
        alpha = jnp.where(alpha == jnp.inf, 1.0, alpha)
        
        d = d + jnp.expand_dims(alpha, 0) * (s - d)
        
        # Update passive set for negative components
        P = P & (d > epsilon)
        
        # Final solve for this iteration
        P, s, d, w = init_step(P, s, d)
        
        return P, s, d, w
    
    # Run fixed number of iterations
    # This is a hack to get the solution to converge
    # Use the while loop in the original fnnls function will have synchronization issues that stucking the code
    # TODO: Find a better way to do this
    for _ in range(50): #50 looks enough for all my test cases. but is that a good idea to hard code the number of iterations?
        P, s, d, w = iteration_step((P, s, d, w))
    
    # Compute final residual
    residual = jnp.sum((x - jnp.einsum('mnb,nb->mb', Z, d)) ** 2, axis=0) ** 0.5
    
    return d, residual 