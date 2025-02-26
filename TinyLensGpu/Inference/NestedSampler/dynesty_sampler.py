from dynesty import NestedSampler
from dynesty import utils as dyfunc
from TinyLensGpu.Inference.base import AbstractInference
import numpy as np

class DynestySampler(AbstractInference):
    def __init__(self, prob_model=None, ndim=None):
        """
        Initialize the DynestySampler

        Args:
            prob_model: The probability model to sample from
            ndim (int): Number of dimensions in the parameter space
        """
        super().__init__(prob_model=prob_model, ndim=ndim)

    def _wrap_likelihood(self, x):
        """Wrapper for likelihood function to handle array dimensions"""
        # print("x shape", x.shape)
        like = self.likelihood(x)
        # print("like shape", like.shape)
        # if x.ndim == 1:
        #     x = x.reshape(1, -1)
        return float(like)  # dynesty expects scalar output

    def _wrap_prior(self, x):
        """Wrapper for prior transform to handle array dimensions"""
        # print("hahaha", x.shape)
        # if x.ndim == 1:
        #     x = x.reshape(1, -1)
        result = self.prior(x)
        # print("result shape", result.shape)
        return result.squeeze()  # dynesty expects 1D array output

    def run(self, nlive=1000, dlogz=None, bound='multi', sample='auto', **kwargs):
        """
        Runs the sampler using dynesty's NestedSampler

        Args:
            nlive (int): Number of live points
            dlogz (float, optional): Target evidence tolerance. Defaults to None.
            bound (str): Method used to bound the prior volume. Options include 
                        'none', 'single', 'multi', 'balls', 'cubes'. Defaults to 'multi'.
            sample (str): Method used to sample uniformly within the likelihood constraint.
                         Options include 'auto', 'unif', 'rwalk', 'slice', 'rslice'.
                         Defaults to 'auto'.
            **kwargs: Additional arguments passed to dynesty.NestedSampler
        """
        sampler = NestedSampler(
            self._wrap_likelihood,
            self._wrap_prior,
            self.ndim,
            nlive=nlive,
            bound=bound,
            sample=sample,
            **kwargs
        )
        
        sampler.run_nested(dlogz=dlogz)
        results = sampler.results
        
        # Store results
        self.samples = results.samples  # samples from the posterior [nsamps, ndim]
        self.weights = np.exp(results.logwt - results.logz[-1])  # posterior weights
        self.log_z = results.logz[-1]  # log evidence
        self.log_z_err = results.logzerr[-1]  # error in log evidence
        self.log_l = results.logl  # log likelihood values
        self.log_l_max = results.logl.max()  # maximum log likelihood

        # Calculate quantiles for each parameter
        self.quantiles = [dyfunc.quantile(samps, [0.16, 0.5, 0.84], weights=self.weights)
            for samps in self.samples.T]  # self.samples.T: [ndim, 3] 