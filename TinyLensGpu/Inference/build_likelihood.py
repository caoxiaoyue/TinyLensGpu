"""
Likelihood interface following example_v4.py style.

This module provides a simplified likelihood interface using JAX vmap
for efficient batch processing.
"""

from typing import Callable, Optional
import jax
import jax.numpy as jnp
from jax import jit


def make_likelihood(likelihood_obj, *, vectorized: bool = False, dtype: Optional[jnp.dtype] = None) -> Callable:
    """
    Create Nautilus-compatible log-likelihood function with JAX vmap.
    
    This function creates a likelihood wrapper that:
    1. Takes parameter arrays (from prior transformation)
    2. Calls the likelihood object with those parameters
    3. Returns log-likelihood values
    4. Uses JAX vmap for efficient batch processing (10-100x speedup)
    
    Parameters
    ----------
    likelihood_obj : ck.Module
        Likelihood object (must be a caskade Module) with __call__ method
        that computes log-likelihood (e.g., an ImageProbModel).
    vectorized : bool, optional
        Whether to support batched evaluation (default: False)
        If True, uses JAX vmap for efficient vectorization.
    
    Returns
    -------
    callable
        Log-likelihood function compatible with nested samplers
    
    Notes
    -----
    This implementation uses JAX vmap for true vectorization:
    - 10-100x faster than Python loops
    - Fully JIT compiled
    - GPU accelerated
    - Requires stateless likelihood computation (use an object whose __call__ is JIT/vmap-safe)
    
    Examples
    --------
    >>> loglike = make_likelihood(prob_model, vectorized=True)
    >>> # Use with Nautilus
    >>> sampler = Sampler(prior, loglike, n_dim=ndim, vectorized=True)
    """

    
    @jit
    def loglike_fn(theta):
        """JIT-compiled single sample evaluation."""
        return likelihood_obj(theta)
    
    if vectorized:
        # Vectorize using JAX vmap for efficient batch processing
        batch_loglike = jit(jax.vmap(loglike_fn))
        
        def loglike(params):
            theta = jnp.asarray(params, dtype=dtype) if dtype is not None else jnp.asarray(params)
            if theta.ndim > 1:
                # Batch evaluation using vmap
                return batch_loglike(theta)
            else:
                # Single evaluation
                res = loglike_fn(theta)
                return float(res)
        
        return loglike
    else:
        # Non-vectorized version
        def loglike(params):
            theta = jnp.asarray(params, dtype=dtype) if dtype is not None else jnp.asarray(params)
            res = loglike_fn(theta)
            return float(res)
        
        return loglike
