from ultranest import ReactiveNestedSampler
from TinyLensGpu.Inference.base import AbstractInference

class UltraNestSampler(AbstractInference): 
    """
    Represent the `UltraNestSampler` component in the TinyLensGpu pipeline.
    
    Notes
    -----
    Instances of this class participate in TinyLensGpu forward modeling and/or
    inference workflows. Keep parameter semantics consistent with neighboring
    modules to ensure predictable numerical behavior.
    """
    def run(self, log_dir='logs', resume=True, vectorized=True):
        """
        Compute run.
        
        Parameters
        ----------
        log_dir : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        resume : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        vectorized : Any
            Input argument used by this routine. Shapes/units follow the surrounding
            simulation or inference convention in the calling context.
        
        Returns
        -------
        None
            This routine updates object state or performs side-effect-free setup only.
        
        """
        # Ensure ndim and prior_transform are initialized from prob_model
        self._ensure_prior_transform()

        paramnames = ['param%d' % (i+1) for i in range(self.ndim)]
        sampler = ReactiveNestedSampler(
            paramnames, 
            self.likelihood, 
            transform=self.prior, 
            log_dir=log_dir, 
            resume=resume, 
            vectorized=vectorized
        )
        result = sampler.run()
        sampler.print_results()
        sampler.plot()