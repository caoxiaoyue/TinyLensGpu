from abc import ABC, abstractmethod
import jax.numpy as jnp

class AbstractInference(ABC): 
    def __init__(self, prob_model=None, ndim=None):
        self.prob_model = prob_model
        self.ndim = ndim


    @abstractmethod
    def params_array2kargs(self, array):
        """
        Converts an array of parameters into a dictionary of keyword arguments
        """
        pass


    @abstractmethod
    def params_kargs2array(self, kargs):
        """
        Converts a dictionary of keyword arguments into an array of parameters
        """
        pass


    def likelihood(self, array):
        """
        Returns the log likelihood of the parameters.
        
        For batch processing, use JAX vmap to vectorize this function.
        """
        kargs = self.params_array2kargs(array)
        return self.prob_model.likelihood(kargs)


    @abstractmethod
    def prior(self, array):
        """
        Returns the prior probability of the parameters
        """
        pass


    @abstractmethod
    def run(self, nlive=1000, **kwargs):
        """
        Runs the inference
        """
        pass
