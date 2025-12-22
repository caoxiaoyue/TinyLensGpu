"""
Vectorized likelihood wrapper for batch processing with JAX vmap.

This module provides a stateless likelihood wrapper that can be JIT compiled
and vectorized using JAX vmap, following the example_v4.py approach.
"""

import caskade as ck
import jax.numpy as jnp
from functools import partial
from jax import jit
from typing import Optional

from .image_model import ImageProbModel


class VectorizedLensLikelihood(ck.Module):
    """
    Vectorized likelihood wrapper that supports JAX vmap.
    
    This class inherits from ck.Module and uses caskade's @ck.forward
    mechanism to enable stateless, pure-functional likelihood evaluation
    that can be JIT compiled and vectorized.
    
    Key difference from LensLikelihood:
    - Inherits from ck.Module (enables @ck.forward)
    - Uses caskade's automatic parameter passing
    - No manual param.to_static() calls
    - Can be JIT compiled and vmapped
    
    Parameters
    ----------
    prob_model : ImageProbModel
        Probability model for computing likelihoods
    
    Examples
    --------
    >>> # Build model
    >>> prob_model = build_likelihood(phys_model, image_data, ...)
    >>> likelihood = VectorizedLensLikelihood(prob_model)
    >>> 
    >>> # Make parameters dynamic
    >>> for param in phys_model.dynamic_params:
    ...     param.to_dynamic()
    >>> 
    >>> # Create vectorized likelihood function
    >>> from TinyLensGpu.Models.likelihood import make_likelihood
    >>> loglike = make_likelihood(likelihood, vectorized=True)
    """
    
    def __init__(self, prob_model: ImageProbModel):
        super().__init__("vectorized_lens_likelihood")
        self.prob_model = prob_model
        # Store reference to physical model for parameter access
        self.phys_model = prob_model.sim_obj.phys_model
    
    def get_dynamic_params(self):
        """Get dynamic parameters from physical model."""
        return self.phys_model.dynamic_params
    
    @ck.forward
    @partial(jit, static_argnums=0)
    def __call__(self, theta: Optional[jnp.ndarray] = None):
        """
        Compute log-likelihood in a stateless, pure-functional way.
        
        This method uses caskade's @ck.forward decorator which automatically
        handles parameter passing. When theta is provided, caskade will
        temporarily use those values for dynamic parameters during the
        forward pass, without modifying the module's state.
        
        Parameters
        ----------
        theta : jnp.ndarray, optional
            Parameter array, shape (ndim,)
            If None, uses current parameter values
        
        Returns
        -------
        log_like : float
            Log-likelihood value
        """
        # Run forward model (caskade handles parameter passing)
        image_model, intensity_list = self.prob_model.forward_model()
        
        # Compute chi-square likelihood
        log_like = self.prob_model._likelihood_helper(
            image_model=image_model,
            image_data=self.prob_model.image_data,
            noise_map=self.prob_model.noise_map,
            unmask=self.prob_model.unmask,
        )
        
        return log_like
    
    def __repr__(self):
        return f"VectorizedLensLikelihood(ndim={len(self.get_dynamic_params())})"
