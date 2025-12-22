from nautilus import Sampler, Prior
from dynesty import utils as dyfunc
from TinyLensGpu.Inference.base import AbstractInference
import numpy as np
import jax

class NautilusSampler(AbstractInference): 
    def run(self, nlive=1000, vectorized=False, n_batch=None, **kwargs):
        """
        Runs the sampler.
        
        Parameters
        ----------
        nlive : int
            Number of live points
        vectorized : bool
            Whether to use vectorized likelihood evaluation (recommended)
        n_batch : int, optional
            Batch size for vectorized evaluation (only used if vectorized=True)
        **kwargs
            Additional arguments passed to Nautilus Sampler
        """
        # Extract run-specific parameters that shouldn't go to Sampler constructor
        verbose = kwargs.pop('verbose', True)  # Default to True for backward compatibility
        
        if vectorized:
            # Create vectorized likelihood using JAX vmap
            likelihood_vec = jax.jit(jax.vmap(self.likelihood))
            sampler = Sampler(self.prior, likelihood_vec, n_dim=self.ndim, n_live=nlive, 
                            n_batch=n_batch or nlive, vectorized=True, **kwargs)
        else:
            sampler = Sampler(self.prior, self.likelihood, n_dim=self.ndim, n_live=nlive, 
                            n_batch=None, vectorized=False, **kwargs)
                     
        sampler.run(verbose=verbose, n_eff=500)
        self.samples, self.log_w, self.log_l = sampler.posterior() ##samples: [nsamps, ndim], log_w: [n_samps]
        # self.samples = np.squeeze(self.samples)
        self.weights = np.exp(self.log_w)
        self.log_z = sampler.log_z
        self.log_l_max = self.log_l.max()

        self.quantiles = [dyfunc.quantile(samps, [0.16, 0.5, 0.84], weights=self.weights)
            for samps in self.samples.T] #self.samples.T: [ndim, 3]