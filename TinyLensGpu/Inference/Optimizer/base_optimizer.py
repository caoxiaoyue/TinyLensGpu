from TinyLensGpu.Inference.base import AbstractInference
import numpy as np
from abc import abstractmethod

class BaseOptimizer(AbstractInference):
    def __init__(self, prob_model=None, ndim=None):
        """
        Initialize the BaseOptimizer

        Args:
            prob_model: The probability model to optimize
            ndim (int): Number of dimensions in the parameter space
        """
        super().__init__(prob_model=prob_model, ndim=ndim)
        self.best_params = None
        self.best_value = None
        self.optimization_result = None
        self.iteration = 0
        self.best_values_history = []

    def _progress_callback(self, xk=None, convergence=None):
        """
        Callback to track optimization progress
        
        Args:
            xk: Current parameter values
            convergence: Convergence value if available
        """
        self.iteration += 1
        if self.best_value is not None:
            self.best_values_history.append(self.best_value)
            print(f"Iteration {self.iteration}: Best merit = {self.best_value:.6f}")
        return True

    def objective(self, x):
        """
        The objective function to minimize (negative log likelihood + negative log prior)
        
        Args:
            x (array): Parameter array
            
        Returns:
            float: Negative log probability (likelihood + prior)
        """
        log_like = float(self.likelihood(x))
        return -log_like  # negative because we want to minimize

    @abstractmethod
    def run(self, **kwargs):
        """
        Runs the optimizer
        
        Args:
            **kwargs: Optimizer-specific arguments
        """
        pass 