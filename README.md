# TinyLensGpu

`TinyLensGpu` is a GPU-accelerated software for galaxy-galaxy strong gravitational lens modeling, built using JAX. It is designed to process the vast influx of lensing data from upcoming space telescopes such as Euclid, CSST, and Roman.

On a consumer-grade RTX 4060 Ti GPU, TinyLensGpu can model a typical 200×200-pixel lensing image in approximately 100–200 seconds. This performance is comparable to that of the previous [gigalens](https://github.com/giga-lens/gigalens) software, which requires four H100 super GPUs to achieve similar speeds—demonstrating the efficiency of `TinyLensGpu` on standard hardware.

We applied `TinyLensGpu` to uniformly model 1,000 mock lenses and 63 Hubble Space Telescope lenses, achieving strong performance in automated lens analysis. The fraction of catastrophic outliers, where automated modeling fails, is approximately 5–10%.

## Capabilities

TinyLensGpu supports a range of modeling approaches, from classical parametric fitting to advanced pixel-based and basis-function reconstructions:

| Category | Public API | Demos | Description |
|---|---|---|---|
| **Parametric image modeling** | `ImageProbModel`, `PhysicalModel`, `SIE`, `Shear`, `SersicEllipse`, `GaussianEllipse` | `lens_only/`, `lens_src/`, `lens_only_plus_sky/`, `src_only/` | Standard lens/source image fitting with Sérsic, Gaussian, and MGE-style components, linear intensity solving, and vectorized likelihoods |
| **Point source position modeling** | `PointSourceProbModel` | `point_source/` | Fits lensed image positions by solving the lens equation and matching predicted/observed image positions with permutation invariance |
| **Multi-band fitting** | `MultiBandImageProbModel`, `BandImageData` | `lens_src_multiband/`, `lens_src_multiband_galfitm/` | Simultaneous `g/r/i` image fitting with shared physical parameters and band-specific data/PSFs/alignment |
| **GALFITM-style wavelength evolution** | `MultiBandImageProbModel`, Chebyshev utilities | `lens_src_multiband_galfitm/` | Multi-band fitting where Sérsic radius and index evolve with wavelength via Chebyshev polynomials |
| **Pixelated source modeling** | `PixelizedSourceModel`, `PixelizedImageProbModel` | `pix_src/` | Grid-based source reconstruction (e.g. 40×40) with Bayesian evidence and regularization (first-order, Matern32, Gaussian, etc.) |
| **Shapelet source modeling** | `ShapeletBasisFunction`, `build_shapelet_set()` | `shapelet_src/` | Refregier (2003) shapelet basis reconstruction with analytically solved linear amplitudes; supports source-only or joint with lens light |

## Programmatic API

TinyLensGpu ships with a fully programmatic modeling API (see `examples/*/run_model.py`) that provides:

- **ParamU-powered components** – All physical components (SIE, Shear, Sérsic, Gaussian, MGE, Shapelet, Pixelized) expose priors, bounds, and modes (dynamic/static/linear/pointer) through `ParamU`.
- **Direct Python configs** – Define complete models in Python; no YAML is required for new workflows.
- **Vectorized likelihoods** – `ImageProbModel` + JAX `vmap` deliver 10–100× throughput for batched nested sampling.
- **Sampler-ready outputs** – `make_prior_transformation` and `make_likelihood` return Nautilus/Dynesty-compatible callables.
- **Type hints & IDE support** – All builders expose precise signatures for faster iteration.

See the [user guide](docs/guides/guide.md) and the demos in `examples` for detailed usage patterns and migration notes.

## Installation

It is recommended to install `TinyLensGpu` in an isolated environment (e.g., Conda). The software requires Python 3.10 or newer.

```bash
# 1. Create and activate a new conda environment
conda create -n tinylens_gpu python=3.11
conda activate tinylens_gpu

# 2. Clone the repository
git clone https://github.com/caoxiaoyue/TinyLensGpu.git
cd TinyLensGpu

# 3. Install runtime dependencies and the editable package
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` includes the `jaxnnls` dependency used by the NNLS linear solver in `TinyLensGpu.utils.linear_solver`.

If you plan to contribute to the codebase or run tests, you can install the development dependencies:

```bash
pip install -r requirements-dev.txt
pip install -e ".[dev]"
```

*Note: For GPU acceleration, the default installation automatically pulls `jax[cuda12]`. If you encounter issues with JAX and CUDA versions, please refer to the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).*

## Quickstart

Every demo under `examples/*` follows the same recipe:

1. **Load data** – `load_lens_data` wraps FITS image/noise/PSF loading and basic masking.
2. **Define components** – Instantiate `ParamU` parameters inside mass/light models (e.g., `SIE`, `Shear`, `SersicEllipse`, `GaussianEllipse`).
3. **Select dynamic/static parameters** – Call `.to_dynamic()`, `.to_static(value)`.
4. **Build physics + likelihood** – Assemble `PhysicalModel(...)`, then construct `ImageProbModel(...)` with `dpix`, `nsub`, solver, and optional position likelihood.
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

## Running the Demos

All demos live under `examples/`. Sampling-based demos write results to `output/` (`result_samples.csv`, `result_summary.csv`, `results.pkl.gz`). Inversion-based demos (pixelated/shapelet) write figures and JSON files.

### Parametric lens light

```bash
cd TinyLensGpu/examples/lens_only
python run_model.py           # lens-only Sérsic fitting

cd ../lens_only_mge
python run_model.py           # lens-only MGE fitting

cd ../lens_only_plus_sky
python run_model.py           # lens light plus sky background

cd ../lens_only_no_batch
python run_model.py           # non-vectorized baseline
```

### Parametric lens + source

```bash
cd TinyLensGpu/examples/lens_src
python run_model.py           # SIE + Sérsic source

cd ../lens_src_mge
python run_model.py           # lens/source fitting with MGE
```

### Source-only

```bash
cd TinyLensGpu/examples/src_only
python run_model.py           # source-only image likelihood

cd ../src_only_poslike
python run_model.py           # source-only with position likelihood constraint
```

### Point source position modeling

```bash
cd TinyLensGpu/examples/point_source
python sim_data.py && python run_model.py
```

### Multi-band fitting

```bash
cd TinyLensGpu/examples/lens_src_multiband
python sim_data.py && python run_model.py   # joint g/r/i fitting

cd ../lens_src_multiband_galfitm
python sim_data.py && python run_model.py   # Chebyshev wavelength evolution
```

### Pixelated source modeling

```bash
cd TinyLensGpu/examples/pix_src
python sim_data.py && python fit_lens_src.py
```

### Shapelet source modeling

```bash
cd TinyLensGpu/examples/shapelet_src/src_light_only
python sim_data.py && python single_step_inversion.py

cd ../src_lens_light_joint
python sim_data.py && python single_step_inversion.py          # joint MGE + shapelet
python sim_data.py && python single_step_inversion_sersic_shapelet.py
python sim_data.py && python single_step_inversion_mge_shapelet.py
```

## Testing

TinyLensGpu includes a comprehensive test suite with **90+ tests** covering all major functionality:

```bash
# Run all tests
pytest

# Run specific test suites
pytest tests/test_caskade_models.py     # Caskade model implementations
pytest tests/test_integration.py         # End-to-end integration
pytest tests/test_mass_profile.py        # Parametric mass models
pytest tests/test_multiband_parametric.py   # Multi-band image models
pytest tests/test_pixelized_inversion.py    # Pixelated source reconstruction
pytest tests/test_point_source_model.py     # Point source position modeling
```

## Documentation

- **[User guide](docs/guides/guide.md)** – Installation, quickstart, tests, and troubleshooting.
- **[Point-source modeling guide](docs/guides/point-source-model.md)** – Position-only likelihood setup and solver configuration.
- **examples/** – Runnable examples; each subdirectory contains a `run_model.py` or `single_step_inversion.py` entry point.

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
