from scipy.optimize import basinhopping
import numpy as np
from TinyLensGpu.Inference.Optimizer.base_optimizer import BaseOptimizer

class BasinHoppingOptimizer(BaseOptimizer):
    def __init__(self, prob_model=None, ndim=None):
        """
        Initialize the BasinHoppingOptimizer

        Args:
            prob_model: The probability model to optimize
            ndim (int): Number of dimensions in the parameter space
        """
        super().__init__(prob_model=prob_model, ndim=ndim)

    def run(self, x0, niter=100, T=1.0, stepsize=0.5, minimizer_kwargs=None, 
            take_step=None, accept_test=None, callback=None, seed=None, ftol=3e-09, **kwargs):
        """
        Run basin-hopping optimization
        
        Args:
            x0 (array_like): Initial guess
            niter (int): Number of basin-hopping iterations
            T (float): Temperature parameter for accept/reject criterion
            stepsize (float): Maximum step size for use in the random displacement
            minimizer_kwargs (dict): Extra arguments to be passed to the local minimizer
            take_step (callable): Custom step-taking routine
            accept_test (callable): Custom acceptance test routine
            callback (callable): Called after each minimization iteration
            seed (int): Random seed
            ftol (float): The function value must be lower than ftol before succesful termination
            **kwargs: Additional arguments passed to basinhopping
            
        Returns:
            dict: Optimization results
        """
        if minimizer_kwargs is None:
            minimizer_kwargs = {}
        
        # Reset iteration counter
        self.iteration = 0
        self.best_values_history = []

        # Ensure we're using L-BFGS-B method with proper ftol
        minimizer_kwargs.update({
            'method': 'L-BFGS-B',
            'options': {'ftol': float(ftol)}
        })

        # Create a callback that updates both our progress and user's callback if provided
        def combined_callback(x, f, accept):
            self.best_value = f
            self._progress_callback(x)
            if callback is not None:
                callback(x, f, accept)
            return False

        result = basinhopping(
            self.objective,
            x0,
            niter=niter,
            T=T,
            stepsize=stepsize,
            minimizer_kwargs=minimizer_kwargs,
            take_step=take_step,
            accept_test=accept_test,
            callback=combined_callback,
            seed=seed,
            **kwargs
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