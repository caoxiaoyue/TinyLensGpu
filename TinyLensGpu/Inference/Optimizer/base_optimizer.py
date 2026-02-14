from TinyLensGpu.Inference.base import AbstractInference
import numpy as np
from abc import abstractmethod

class BaseOptimizer(AbstractInference):
    """
    Represent the `BaseOptimizer` component in the TinyLensGpu pipeline.
    
    Parameters
    ----------
    prob_model : Any
        Configuration argument consumed during construction of this component.
    ndim : Any
        Configuration argument consumed during construction of this component.
    
    Notes
    -----
    Instances of this class participate in TinyLensGpu forward modeling and/or
    inference workflows. Keep parameter semantics consistent with neighboring
    modules to ensure predictable numerical behavior.
    """
    def __init__(self, prob_model=None, ndim=None):
        """
        Initialize a `BaseOptimizer` instance with validated configuration.
        
        Parameters
        ----------
        prob_model : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        ndim : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        None
            This routine updates object state or performs side-effect-free setup only.
        
        """
        super().__init__(prob_model=prob_model, ndim=ndim)
        self.best_params = None
        self.best_value = None
        self.optimization_result = None
        self.iteration = 0
        self.best_values_history = []

    def _progress_callback(self, xk=None, convergence=None):
        """
        Internal helper to progress callback.
        
        Parameters
        ----------
        xk : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        convergence : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        self.iteration += 1
        if self.best_value is not None:
            self.best_values_history.append(self.best_value)
            print(f"Iteration {self.iteration}: Best merit = {self.best_value:.6f}")
        return True

    def objective(self, x):
        """
        Compute objective.
        
        Parameters
        ----------
        x : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        value : Any
            Computed output produced by this routine. For array outputs, shape follows
            the input mesh/matrix conventions used by the corresponding pipeline stage.
        
        """
        log_like = float(self.likelihood(x))
        return -log_like  # negative because we want to minimize

    @abstractmethod
    def run(self, **kwargs):
        """
        Compute run.
        
        Parameters
        ----------
        **kwargs : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        None
            This routine updates object state or performs side-effect-free setup only.
        
        """
        pass 