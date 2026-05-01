"""
This module provides a simplified likelihood interface using JAX vmap
for efficient batch processing.
"""

# pyright: reportMissingImports=false

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
        """
        Evaluate likelihood object on one parameter vector.

        Parameters
        ----------
        theta : array_like
            Parameter vector in physical space.

        Returns
        -------
        jnp.ndarray | float
            Log-likelihood returned by ``likelihood_obj``.
        """
        return likelihood_obj(theta)
    
    batch_loglike = None
    if vectorized:
        # Vectorize using JAX vmap for efficient batch processing
        batch_loglike = jit(jax.vmap(loglike_fn))
    
    def loglike(params):
        """
        Evaluate likelihood for one sample or a batch.

        Parameters
        ----------
        params : array_like
            Single sample ``(ndim,)`` or batch ``(batch, ndim)``.

        Returns
        -------
        float | jnp.ndarray
            Scalar for single input, vector of log-likelihoods for batched input.
        """
        theta = jnp.asarray(params, dtype=dtype) if dtype is not None else jnp.asarray(params)

        # Batch evaluation using vmap if requested and input is 2D
        if vectorized and theta.ndim > 1 and batch_loglike is not None:
            return batch_loglike(theta)
        
        # Single evaluation (fallback for 1D or non-vectorized)
        res = loglike_fn(theta)
        return float(res)
    
    return loglike
