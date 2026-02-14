from abc import ABC, abstractmethod
from typing import Optional, Callable, Any
import numpy as np
import jax
import jax.numpy as jnp

from .build_prior import make_prior_transformation

class AbstractInference(ABC): 
    """
    Represent the `AbstractInference` component in the TinyLensGpu pipeline.
    
    Parameters
    ----------
    prob_model : Any
        Configuration argument consumed during construction of this component.
    ndim : Any
        Configuration argument consumed during construction of this component.
    prior_transform : Any
        Configuration argument consumed during construction of this component.
    
    Notes
    -----
    Instances of this class participate in TinyLensGpu forward modeling and/or
    inference workflows. Keep parameter semantics consistent with neighboring
    modules to ensure predictable numerical behavior.
    """
    def __init__(
        self,
        prob_model: Optional[Any] = None,
        ndim: Optional[int] = None,
        prior_transform: Optional[Callable[[Any], Any]] = None,
    ):
        """
        Initialize a `AbstractInference` instance with validated configuration.
        
        Parameters
        ----------
        prob_model : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        ndim : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        prior_transform : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        None
            This routine updates object state or performs side-effect-free setup only.
        
        """
        self.prob_model = prob_model
        self.ndim = ndim
        self.prior_transform = prior_transform


    def _ensure_prior_transform(self) -> None:
        """
        Internal helper to ensure prior transform.
        
        
        Returns
        -------
        None
            This routine updates object state or performs side-effect-free setup only.
        
        Raises
        ------
        ValueError
            Raised when input validation fails or required runtime state is missing.
        
        """
        if self.prior_transform is not None and self.ndim is not None:
            return
        if self.prob_model is None:
            raise ValueError("prob_model is not set")

        prior_transform, specs = make_prior_transformation(self.prob_model)
        if self.prior_transform is None:
            self.prior_transform = prior_transform
        if self.ndim is None:
            self.ndim = len(specs)


    def loglike_jax(self, theta):
        """
        Compute loglike jax.
        
        Parameters
        ----------
        theta : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        theta = jnp.asarray(theta, dtype=jnp.float32)
        return self.prob_model(theta)


    def likelihood(self, theta):
        """
        Returns the log likelihood of the parameters.
        
        Notes
        -----
        This wrapper returns Python/numpy values for compatibility with
        SciPy and nested samplers that expect numpy/float outputs.
        For JAX vectorization, use `loglike_jax`.
        """
        theta = np.asarray(theta)
        if theta.ndim > 1:
            theta_jax = jnp.asarray(theta, dtype=jnp.float32)
            out = jax.vmap(self.loglike_jax)(theta_jax)
            return np.asarray(out)
        return float(np.asarray(self.loglike_jax(theta)))


    def prior(self, u):
        """
        Compute prior.
        
        Parameters
        ----------
        u : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        self._ensure_prior_transform()
        return np.asarray(self.prior_transform(u))


    @abstractmethod
    def run(self, nlive=1000, **kwargs):
        """
        Compute run.
        
        Parameters
        ----------
        nlive : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        **kwargs : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        None
            This routine updates object state or performs side-effect-free setup only.
        
        """
        pass
