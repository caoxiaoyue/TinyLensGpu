# TinyLensGpu

`TinyLensGpu` is a GPU-accelerated software for galaxy-galaxy strong gravitational lens modeling, built using JAX. It is designed to process the vast influx of lensing data from upcoming space telescopes such as Euclid, CSST, and Roman.

On a consumer-grade RTX 4060 Ti GPU, TinyLensGpu can model a typical 200×200-pixel lensing image in approximately 100–200 seconds. This performance is comparable to that of the previous [gigalens](https://github.com/giga-lens/gigalens) software, which requires four H100 super GPUs to achieve similar speeds—demonstrating the efficiency of `TinyLensGpu` on standard hardware.

We applied `TinyLensGpu` to uniformly model 1,000 mock lenses and 63 Hubble Space Telescope lenses, achieving strong performance in automated lens analysis. The fraction of catastrophic outliers, where automated modeling fails, is approximately 5–10%.

Currently, `TinyLensGpu` can model the light distribution of both the lens and source galaxy using:
- **Parametric models**: Sérsic, Gaussian, and multi-Gaussian expansion (MGE) models
- **Pixelized source models**: Discrete pixel reconstruction with Gaussian Process regularization (NEW in v2.1)

## 🆕 Programmatic API (v2.0)

TinyLensGpu now ships with a fully programmatic modeling API (see `paper/demo/*/run_model.py`) that provides:

- **ParamU-powered components** – All physical components (SIE, Shear, Sérsic, Gaussian, MGE) expose priors, bounds, and modes (dynamic/static/linear/pointer) through `ParamU`.
- **Direct Python configs** – Define complete models in Python; no YAML is required for new workflows.
- **Vectorized likelihoods** – `ImageProbModel` + JAX `vmap` deliver 10–100× throughput for batched nested sampling.
- **Sampler-ready outputs** – `make_prior_transformation` and `make_likelihood` return Nautilus/Dynesty-compatible callables.
- **Type hints & IDE support** – All builders expose precise signatures for faster iteration.

See [CASKADE_GUIDE.md](CASKADE_GUIDE.md) and the demos in `paper/demo` for detailed usage patterns and migration notes.

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
pytest tests/test_caskade_models.py     # Caskade model implementations
pytest tests/test_integration.py         # End-to-end integration
pytest tests/test_operator_solver.py     # Operator backend inversion
pytest tests/test_pixelized_source.py    # Pixelized source modeling
pytest tests/test_mass_profile.py        # Parametric mass models
```

## Usage (Programmatic API)

Every demo under `paper/demo/*` contains a `run_model.py` that follows the same recipe:

1. **Load data** – `load_lens_data` wraps FITS image/noise/PSF loading and basic masking.
2. **Define components** – Instantiate `ParamU` parameters inside mass/light models (e.g., `SIE`, `Shear`, `SersicEllipse`, `GaussianEllipse`).
3. **Select dynamic/static parameters** – Call `.to_dynamic()`, `.to_static(value)`, or rely on `.to_linear()` defaults for flux-like parameters.
4. **Build physics + likelihood** – assemble `PhysicalModel(...)`, then construct `ImageProbModel(...)` (or `PixelizedImageProbModel(...)`) with `dpix`, `nsub`, solver, and optional position likelihood.
5. **Vectorize and sample** – Use `prob_model` directly as the likelihood object, then create `prior, prior_specs = make_prior_transformation(prob_model)` and `loglike = make_likelihood(prob_model, ...)`. Feed both into Nautilus/Dynesty.

### Minimal example

```python
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from TinyLensGpu.Inference import ParamU
from TinyLensGpu.PhysicalModel import PhysicalModel, SersicEllipse, SIE
from TinyLensGpu.ObservationModel import ImageProbModel
from TinyLensGpu.utils import load_lens_data
from TinyLensGpu.Inference.build_prior import make_prior_transformation
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from nautilus import Sampler

image_data, noise_map, psf_kernel, mask = load_lens_data(
    image_path="data/image.fits",
    noise_path="data/noise.fits",
    psf_path="data/psf.fits",
)

sie = SIE(theta_E=ParamU("theta_E", 1.5, prior_type="uniform",
                         prior_settings=[0.001, 3.001], limits=[0.0, 10.0]))
source = SersicEllipse(
    R_sersic=ParamU("R_sersic_src", 1.0, prior_type="uniform",
                    prior_settings=[0.001, 2.001], limits=[0.0, 5.0]),
    n_sersic=ParamU("n_sersic_src", 1.0, prior_type="uniform",
                    prior_settings=[0.3, 2.3], limits=[0.3, 6.0]),
    Ie=ParamU("Ie_src", 1.0),  # solved linearly
)

sie.theta_E.to_dynamic()
source.R_sersic.to_dynamic()
source.n_sersic.to_dynamic()

phys_model = PhysicalModel(lens_mass=[sie], source_light=[source], lens_light=[])
prob_model = ImageProbModel(
    image_data=image_data,
    noise_map=noise_map,
    psf_kernel=psf_kernel,
    dpix=0.074,
    nsub=4,
    phys_model=phys_model,
    use_linear=True,
    solver_type="nnls",
    mask=mask,
)

prior, prior_specs = make_prior_transformation(prob_model)
loglike = make_likelihood(prob_model, vectorized=True)

sampler = Sampler(prior, loglike, n_dim=len(prior_specs), n_live=200, vectorized=True, n_batch=200)
sampler.run(verbose=True, n_eff=800)
```

### Running the demos

```bash
cd TinyLensGpu/paper/demo/lens_src
python run_model.py          # lens + source parametric example

cd ../lens_src_mge
python run_model.py          # MGE lens + source example

cd ../src_only_pix_src
python demo_pix_src.py       # pixelized source reconstruction example
```

Each demo writes results to `output/` (`result_samples.csv`, `result_summary.csv`, `results.pkl.gz`). Modify the scripts directly to experiment with priors, components, likelihood options, or sampler settings.

### Pixelized Source Modeling (NEW)

TinyLensGpu now supports pixelized source reconstruction as an alternative to parametric source models:

```python
from TinyLensGpu.PhysicalModel import (
    PhysicalModel,
    PixelizedSourceModel,
    PixelizedSourceConfig,
    IrregularGridConfig,
    MappingConfig,
    RegularizationConfig,
    SIE,
)
from TinyLensGpu.ObservationModel import PixelizedImageProbModel

# Create mass model
sie = SIE(theta_E=1.5, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
pix_src = PixelizedSourceModel(
    config=PixelizedSourceConfig(
        grid=IrregularGridConfig(n_source_points=1500, mesh_alpha=1.5),
        mapping=MappingConfig(k_neighbors=5, interp_kernel="wendland_c4", radius_scale=1.5),
        regularization=RegularizationConfig(mode="dense_gp", gp_kernel="exp"),
    ),
    reg_scale=0.05,
    reg_coefficient=1.0,
)
phys_model = PhysicalModel(lens_mass=[sie], source_light=[pix_src])

# Create probability model
prob_model = PixelizedImageProbModel(
    image_data=image,
    noise_map=noise,
    psf_kernel=psf,
    dpix=0.05,
    phys_model=phys_model,
    mask=mask,
)

# Compute log evidence (analogous to log likelihood)
log_ev = prob_model.log_evidence()

# Reconstruct source (via simulator)
data_vector = prob_model.image_data[~prob_model.mask]
noise_variance = prob_model.noise_map[~prob_model.mask] ** 2
source_intensities, source_mesh_beta, model_image, _ = prob_model.simulator.reconstruct_source(
    data_vector=data_vector,
    noise_variance=noise_variance,
    reg_scale=prob_model.pix_src_model.reg_scale.value,
    reg_coefficient=prob_model.pix_src_model.reg_coefficient.value,
)
```

**Key Features**:
- Bayesian evidence calculation for hyperparameter optimization
- Multiple regularization kernels (exponential, Gaussian, Matern-3/2, Matern-5/2)
- Adaptive source mesh generation
- Compatible with nested sampling for joint mass + hyperparameter inference

See [Pixelized Source Guide](doc/pixelized_source_guide.md) for detailed documentation.


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
