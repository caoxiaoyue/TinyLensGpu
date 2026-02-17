"""
ParamU: caskade Param subclass with prior metadata.

This module provides a parameter class that stores prior information
for automatic transformation in nested sampling and optimization.
"""

from typing import Literal, Sequence, Optional, Any
import caskade as ck


class ParamU(ck.Param):
    """
    ck.Param subclass that stores prior metadata for automatic transforms.
    
    This class extends caskade's Param to include prior distribution information,
    enabling automatic prior transformation for nested sampling and optimization.
    
    Parameters
    ----------
    name : str
        Parameter name
    value : float, optional
        Initial parameter value
    prior_type : {'uniform', 'gaussian', 'log_uniform'}
        Type of prior distribution
    prior_settings : sequence of float, optional
        Prior distribution parameters:
        - uniform: [min, max]
        - gaussian: [mean, std]
        - log_uniform: [min, max]
    limits : sequence of float, optional
        Hard limits [min, max] to clip parameter values
    **kwargs
        Additional arguments passed to ck.Param
    
    Attributes
    ----------
    prior_type : str
        Type of prior distribution
    prior_settings : tuple
        Prior distribution parameters
    limits : tuple or None
        Hard limits for parameter values
    
    Examples
    --------
    >>> # Uniform prior
    >>> param = ParamU("slope", 1.0, prior_type="uniform", 
    ...                prior_settings=[0.0, 2.0])
    >>> 
    >>> # Gaussian prior with limits
    >>> param = ParamU("center_x", 0.0, prior_type="gaussian",
    ...                prior_settings=[0.0, 0.1], limits=[-1.0, 1.0])
    """
    
    def __init__(
        self,
        name: str,
        value: Optional[float] = None,
        *,
        prior_type: Literal["uniform", "gaussian", "log_uniform"] = "uniform",
        prior_settings: Optional[Sequence[float]] = None,
        limits: Optional[Sequence[float]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize parameter with prior metadata.

        Parameters
        ----------
        name : str
            Parameter name.
        value : float, optional
            Initial parameter value.
        prior_type : {'uniform', 'gaussian', 'log_uniform'}, optional
            Prior family used during unit-cube transformation.
        prior_settings : sequence of float, optional
            Prior parameters (``[min, max]`` or ``[mean, std]`` depending on
            ``prior_type``).
        limits : sequence of float, optional
            Hard clipping limits ``[min, max]`` in physical parameter space.
        **kwargs
            Forwarded to ``ck.Param``.
        """
        super().__init__(name, value, **kwargs)
        self.prior_type = prior_type
        self.prior_settings = prior_settings
        self.limits = limits
    
    def __repr__(self) -> str:
        """
        Return debug-friendly parameter summary.

        Returns
        -------
        str
            String containing name, value, prior configuration, and hard limits.
        """
        return (f"ParamU(name={self.name}, value={self.value}, "
                f"prior_type={self.prior_type}, "
                f"prior_settings={self.prior_settings}, "
                f"limits={self.limits})")
