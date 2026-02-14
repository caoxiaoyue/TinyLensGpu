from scipy.optimize import direct
import numpy as np
from TinyLensGpu.Inference.Optimizer.base_optimizer import BaseOptimizer

class DirectOptimizer(BaseOptimizer):
    """
    Represent the `DirectOptimizer` component in the TinyLensGpu pipeline.
    
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
        Initialize a `DirectOptimizer` instance with validated configuration.
        
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

    def run(self, bounds=None, maxiter=1000, maxfun=None, eps=1e-4, 
            locally_biased=True, callback=None, **kwargs):
        """
        Run DIRECT optimization
        
        Args:
            bounds (sequence): Bounds for variables [(x1_min, x1_max), (x2_min, x2_max), ...]
            maxiter (int): Maximum number of iterations
            maxfun (int): Maximum number of function evaluations
            eps (float): Desired relative accuracy in the optimization
            locally_biased (bool): Use the locally biased variant of the algorithm
            callback (callable): User callback function
            **kwargs: Additional arguments passed to direct (unsupported args will be filtered)
            
        Returns:
            dict: Optimization results
        """
        if bounds is None:
            raise ValueError("bounds must be provided for DIRECT optimization")

        # Reset iteration counter
        self.iteration = 0
        self.best_values_history = []

        # Convert bounds to the format expected by DIRECT
        bounds = np.array(bounds)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("bounds must be a sequence of (min, max) pairs")

        # Filter kwargs to only include supported parameters
        supported_params = {'maxiter', 'maxfun', 'eps', 'locally_biased', 'callback'}
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}

        # Update with explicitly defined parameters
        if maxiter is not None:
            filtered_kwargs['maxiter'] = maxiter
        if maxfun is not None:
            filtered_kwargs['maxfun'] = maxfun
        if eps is not None:
            filtered_kwargs['eps'] = eps
        filtered_kwargs['locally_biased'] = locally_biased

        # Create a wrapper function that includes bounds and tracks progress
        def objective_with_tracking(x):
            """
            Compute objective with tracking.
            
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
            f = self.objective(x)
            self.best_value = f
            self._progress_callback(x)
            if callback is not None:
                callback(x, f)
            return f

        result = direct(
            objective_with_tracking,
            bounds.tolist(),  # Pass bounds as a single argument
            **filtered_kwargs
        )

        self.optimization_result = result
        self.best_params = result.x
        self.best_value = result.fun

        return {
            'x': result.x,  # Best parameter set found
            'fun': result.fun,  # Value of objective at minimum
            'nfev': result.nfev,  # Number of function evaluations
            'nit': result.nit,  # Number of iterations
            'success': result.success,  # Whether optimization was successful
            'message': result.message,  # Description of the cause of termination
            'result': result  # Full optimization result object
        } 