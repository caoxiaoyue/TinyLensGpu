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
        Returns the log likelihood of the parameters
        """
        if array.ndim == 1:
            bs =1
        else:
            bs = array.shape[0]
        kargs = self.params_array2kargs(array)
        return self.prob_model.likelihood(kargs, bs)


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
