from dynesty import NestedSampler
from dynesty import utils as dyfunc
from TinyLensGpu.Inference.base import AbstractInference
import numpy as np

class DynestySampler(AbstractInference):
    """
    Dynesty nested-sampling adapter.

    Parameters
    ----------
    prob_model : Any, optional
        Probability model callable.
    ndim : int, optional
        Number of free parameters.
    """
    def __init__(self, prob_model=None, ndim=None):
        """
        Initialize the DynestySampler.

        Parameters
        ----------
        prob_model : Any
            The probability model providing `likelihood` and `prior` methods.
        ndim : int
            Number of dimensions for the sampling parameter space.
        """
        super().__init__(prob_model=prob_model, ndim=ndim)

    def _wrap_likelihood(self, x):
        """
        Wrap the probability model's likelihood function for Dynesty.

        Dynesty expects the likelihood function to return a scalar float.

        Parameters
        ----------
        x : np.ndarray
            Parameter vector.

        Returns
        -------
        float
            Log-likelihood value.
        """
        # print("x shape", x.shape)
        like = self.likelihood(x)
        # print("like shape", like.shape)
        # if x.ndim == 1:
        #     x = x.reshape(1, -1)
        return float(like)  # dynesty expects scalar output

    def _wrap_prior(self, x):
        """
        Wrap prior transform for Dynesty API.

        Parameters
        ----------
        x : np.ndarray
            Unit-cube sample with shape ``(ndim,)``.

        Returns
        -------
        np.ndarray
            Physical-space parameter vector with shape ``(ndim,)``.
        """
        # print("hahaha", x.shape)
        # if x.ndim == 1:
        #     x = x.reshape(1, -1)
        result = self.prior(x)
        # print("result shape", result.shape)
        return result.squeeze()  # dynesty expects 1D array output

    def run(self, nlive=1000, dlogz=None, bound='multi', sample='auto', **kwargs):
        """
        Run the Nested Sampling process using dynesty.NestedSampler.

        Parameters
        ----------
        nlive : int, optional
            Number of live points, by default 1000.
        dlogz : float, optional
            Target evidence tolerance for stopping, by default None.
        bound : str, optional
            Method used to bound the prior volume ('none', 'single', 'multi', 'balls', 'cubes'), by default 'multi'.
        sample : str, optional
            Method used to sample uniformly within the likelihood constraint ('auto', 'unif', 'rwalk', 'slice', 'rslice'), by default 'auto'.
        **kwargs
            Additional arguments passed to `dynesty.NestedSampler`.
        """
        # Ensure ndim and prior_transform are initialized from prob_model
        self._ensure_prior_transform()

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
