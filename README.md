# TinyLensGpu

`TinyLensGpu` is a GPU-accelerated software for galaxy-galaxy strong gravitational lens modeling, built using JAX. It is designed to process the vast influx of lensing data from upcoming space telescopes such as Euclid, CSST, and Roman.

On a consumer-grade RTX 4060 Ti GPU, TinyLensGpu can model a typical 200×200-pixel lensing image in approximately 100–200 seconds. This performance is comparable to that of the previous [gigalens](https://github.com/giga-lens/gigalens) software, which requires four H100 super GPUs to achieve similar speeds—demonstrating the efficiency of `TinyLensGpu` on standard hardware.

We applied `TinyLensGpu` to uniformly model 1,000 mock lenses and 63 Hubble Space Telescope lenses, achieving strong performance in automated lens analysis. The fraction of catastrophic outliers, where automated modeling fails, is approximately 5–10%.

Currently, `TinyLensGpu` can model the light distribution of both the lens and source galaxy using parametric models such as Sérsic, Gaussian, and multi-Gaussian expansion models. In future updates, we plan to incorporate a pixelated source model to enhance its capabilities.

## 🆕 Integration (v2.0)

TinyLensGpu now features a ** implementation** that provides:
- **Modular architecture**: All physical components (SIE, Shear, Sersic, Gaussian) are implemented as `caskade.Module` objects
- **Automatic parameter management**: Parameters automatically support dynamic (sampling), static (fixed), linear (NNLS), and pointer (linked) modes
- **Improved batch processing**: Seamless handling of large batch sizes for nested sampling
- **Backward compatibility**: Existing YAML configurations work without modification
- **Enhanced maintainability**: Cleaner code structure with `@ck.forward` decorators

See [CASKADE_GUIDE.md](CASKADE_GUIDE.md) for detailed usage instructions and migration guide.

## Installation

```bash
conda create -n tinylens_gpu python=3.11 #create a new conda environment
sudo pacman -S cuda cudnn #for arch linux, install cuda and cudnn
conda activate tinylens_gpu #activate the conda environment
pip install -U "jax[cuda12]" #install jax with cuda 12 support
pip install numba #install numba
pip install nautilus-sampler dynesty
pip install astropy matplotlib corner pyyaml
conda install jupyter
pip install "caskade[jax]"  # Required for  implementation
git clone https://github.com/caoxiaoyue/TinyLensGpu #clone the TinyLensGpu repository, suppose you place it in the current directory
conda develop TinyLensGpu #install TinyLensGpu in the conda environment
```

## Testing

TinyLensGpu includes a comprehensive test suite with **90+ tests** covering all major functionality:

```bash
# Run all tests
pytest

# Run specific test suites
pytest tests/test_image_models.py      # Test caskade model implementations
pytest tests/test_config_parser.py        # Test configuration parsing
pytest tests/test_lens_simulator.py       # Test forward simulation
pytest tests/test_caskade_inference.py    # Test inference system
pytest tests/test_demo_lens_src.py        # Test full demo workflow
```

## Usage
Use the YAML file to configure the model settings. For example, the following YAML file named `model_config.yaml` fits only the lens light distribution using a Sérsic profile.

```yaml
dataset:
  data_path: "data/image.fits" #the path to the lensing image
  noise_path: "data/noise.fits" #the path to the noise image
  psf_path: "data/psf.fits" #the path to the PSF image
  pixel_scale: 0.074 #the pixel scale of the image

model_components:
  lens_mass_list: [] #the list of lens mass models

  source_light_list: [] #the list of source light models

  lens_light_list: #the list of lens light models
    - type: "Sersic" #the type of the lens light model
      params: #the parameters of the lens light model
        R_sersic: #the effective radius of the lens light model
          prior_type: "uniform" #the prior type of the parameter
          prior_settings: [0.001, 2.001] #the prior settings of the parameter, for uniform prior, it is the range of the parameter
          limits: [0.0, 5.0] #the limits of the parameter
          fixed: false #whether the parameter is fixed
        n_sersic: #the Sersic index of the lens light model
          prior_type: "gaussian" #the prior type of the parameter
          prior_settings: [4.0, 0.5] #the prior settings of the parameter, for gaussian prior, it is the mean and standard deviation of the parameter
          limits: [0.3, 6.0] #the limits of the parameter
          fixed: false #whether the parameter is fixed
        e1: #the ellipticity of the lens light model
          prior_type: "gaussian" #the prior type of the parameter
          prior_settings: [0.0, 0.3] 
          limits: [-1.0, 1.0] 
          fixed: false 
        e2: #the ellipticity of the lens light model
          prior_type: "gaussian" 
          prior_settings: [0.0, 0.3]
          limits: [-1.0, 1.0]
          fixed: false
        center_x: #the x-coordinate of the center of the lens light model
          fixed: true
          fixed_value: 0.0
        center_y: #the y-coordinate of the center of the lens light model
          fixed: true
          fixed_value: 0.0
        Ie: #the intensity at the effective radius of the lens light model
          use_linear: true #whether to use the linear solving method

inference:
  type: "sampler" #the type of the inference method
  method: "nautilus" #the method of the inference
  settings:
    nlive: 200 #the number of live points
    batch_size: 200

output:
  path: "output" #the path to the output directory
  figures:
    results: "model_results.png" #the path to the results figure
    corner: "model_corner.png" #the path to the triangle plot figure
  tables:
    samples: "result_samples.csv" #the path to the samples table
    summary: "result_summary.csv" #the path to the summary table of modeling results
  datasets:
    subplot: "dataset_subplot.png" 
```
The following script named `run_model.py` can use the above YAML file (`model_config.yaml`) to fit the model.
```python
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# If you want to run on CPU, uncomment the following lines
# os.environ["NPROC"] = "1" #https://github.com/jax-ml/jax/issues/743
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
# os.environ["JAX_PLATFORMS"] = "cpu"

from TinyLensGpu.LinearSolver.runner import RunCaskadeLensModel
config_path = 'model_config.yaml'


lens_model = RunCaskadeLensModel(config_path)
lens_model.run() 
``` 
Finally, run `python run_model.py` in the terminal to do the lens modeling.

For additional examples, refer to the scripts in the `paper/demo` folder.

## Citation
If you find this work useful, please cite Cao et al. (2025). The BibTeX entry is provided below for your convenience.
```
@ARTICLE{2025MNRAS.540.3121C,
       author = {{Cao}, Xiaoyue and {Li}, Ran and {Li}, Nan and {Chen}, Yun and {Li}, Rui and {Shan}, Huanyuan and {Li}, Tian},
        title = "{CSST strong lensing preparation: fast modelling of galaxy{\textendash}galaxy strong lenses in the big data era}",
      journal = {\mnras},
     keywords = {gravitational lensing: strong, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2025,
        month = jul,
       volume = {540},
       number = {4},
        pages = {3121-3134},
          doi = {10.1093/mnras/staf891},
archivePrefix = {arXiv},
       eprint = {2503.08586},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2025MNRAS.540.3121C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
Additionally, TinyLensGpu has benefited from several other open-source lens modeling projects. Please consider crediting them in your work as well:
- [gigalens](https://github.com/giga-lens/gigalens)
- [PyAutoLens](https://github.com/Jammy2211/PyAutoLens)
- [herculens](https://github.com/Herculens/herculens)
- [lenstronomy](https://github.com/lenstronomy/lenstronomy)