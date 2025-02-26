from scipy.optimize import differential_evolution
import numpy as np
from TinyLensGpu.Inference.Optimizer.base_optimizer import BaseOptimizer

class DifferentialEvolutionOptimizer(BaseOptimizer):
    def __init__(self, prob_model=None, ndim=None):
        """
        Initialize the DifferentialEvolutionOptimizer

        Args:
            prob_model: The probability model to optimize
            ndim (int): Number of dimensions in the parameter space
        """
        super().__init__(prob_model=prob_model, ndim=ndim)

    def run(self, bounds=None, strategy='best1bin', maxiter=1000, popsize=15, tol=0.01, 
            mutation=(0.5, 1), recombination=0.7, seed=None, callback=None, **kwargs):
        """
        Run differential evolution optimization
        
        Args:
            bounds (sequence): Bounds for variables [(x1_min, x1_max), (x2_min, x2_max), ...]
            strategy (str): The differential evolution strategy to use
            maxiter (int): Maximum number of generations
            popsize (int): Multiplier for setting population size
            tol (float): Relative tolerance for convergence
            mutation (float or tuple): Mutation constant
            recombination (float): Recombination constant
            seed (int): Random seed
            callback (callable): User callback function
            **kwargs: Additional arguments passed to differential_evolution
            
        Returns:
            dict: Optimization results
        """
        if bounds is None:
            raise ValueError("bounds must be provided for differential evolution")

        # Reset iteration counter
        self.iteration = 0
        self.best_values_history = []

        # Create a callback that updates both our progress and user's callback if provided
        def combined_callback(xk, convergence=None):
            self.best_value = self.objective(xk)
            self._progress_callback(xk, convergence)
            if callback is not None:
                callback(xk, convergence)
            return False

        result = differential_evolution(
            self.objective,
            bounds,
            strategy=strategy,
            maxiter=maxiter,
            popsize=popsize,
            tol=tol,
            mutation=mutation,
            recombination=recombination,
            seed=seed,
            callback=combined_callback,
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