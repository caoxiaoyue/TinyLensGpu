from nautilus import Sampler, Prior
from dynesty import utils as dyfunc
from TinyLensGpu.Inference.base import AbstractInference
import numpy as np

class NautilusSampler(AbstractInference): 
    def run(self, nlive=1000, bs=1, **kwargs):
        """
        Runs the sampler
        """
        # Extract run-specific parameters that shouldn't go to Sampler constructor
        verbose = kwargs.pop('verbose', True)  # Default to True for backward compatibility
        
        if bs>1:
            sampler = Sampler(self.prior, self.likelihood, n_dim=self.ndim, n_live=nlive, n_batch=bs, vectorized=True, **kwargs)
        else:
            sampler = Sampler(self.prior, self.likelihood, n_dim=self.ndim, n_live=nlive, n_batch=None, vectorized=False, **kwargs)
                     
        sampler.run(verbose=verbose, n_eff=500)
        self.samples, self.log_w, self.log_l = sampler.posterior() ##samples: [nsamps, ndim], log_w: [n_samps]
        # self.samples = np.squeeze(self.samples)
        self.weights = np.exp(self.log_w)
        self.log_z = sampler.log_z
        self.log_l_max = self.log_l.max()

        self.quantiles = [dyfunc.quantile(samps, [0.16, 0.5, 0.84], weights=self.weights)
            for samps in self.samples.T] #self.samples.T: [ndim, 3]