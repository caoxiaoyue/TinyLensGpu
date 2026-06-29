from TinyLensGpu.Inference.base import AbstractInference
from abc import abstractmethod

class BaseOptimizer(AbstractInference):
    """
    Base class for optimization-based inference backends.

    Parameters
    ----------
    prob_model : Any, optional
        Probability model callable.
    ndim : int, optional
        Number of free parameters.
    """
    def __init__(self, prob_model=None, ndim=None):
        """
        Initialize optimizer bookkeeping state.

        Parameters
        ----------
        prob_model : Any, optional
            Probability model callable.
        ndim : int, optional
            Number of parameters.
        """
        super().__init__(prob_model=prob_model, ndim=ndim)
        self.best_params = None
        self.best_value = None
        self.optimization_result = None
        self.iteration = 0
        self.best_values_history = []

    def _progress_callback(self, xk=None, convergence=None):
        """
        Generic progress callback used by SciPy optimizers.

        Parameters
        ----------
        xk : array_like, optional
            Current optimizer iterate.
        convergence : float, optional
            Convergence metric provided by optimizer backends.

        Returns
        -------
        bool
            ``True`` to continue optimization.
        """
        self.iteration += 1
        if self.best_value is not None:
            self.best_values_history.append(self.best_value)
            print(f"Iteration {self.iteration}: Best merit = {self.best_value:.6f}")
        return True

    def objective(self, x):
        """
        Objective function minimized by deterministic optimizers.

        Parameters
        ----------
        x : array_like
            Model parameters in physical space.

        Returns
        -------
        float
            Negative log-likelihood.
        """
        log_like = float(self.likelihood(x))
        return -log_like  # negative because we want to minimize

    @abstractmethod
    def run(self, **kwargs):
        """
        Run optimizer backend.

        Parameters
        ----------
        **kwargs
            Backend-specific optimizer options.
        """
        pass 
