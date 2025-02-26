from ultranest import ReactiveNestedSampler
from TinyLensGpu.Inference.base import AbstractInference

class UltraNestSampler(AbstractInference): 
    def run(self, log_dir='logs', resume=True, vectorized=True):
        """
        Runs the sampler
        """
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