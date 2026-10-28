# Potential-correction mock inputs

`generate_mock.py` creates single-band, lens-light-subtracted Pseudo-Jaffe
subhalo observations and a matched no-subhalo control. It does not fit a
correction. Use the `tinylens_gpu` environment; PyAuto packages are unnecessary.

```bash
source ~/anaconda3/bin/activate && conda activate tinylens_gpu
XLA_PYTHON_CLIENT_PREALLOCATE=false python examples/potential_correction/generate_mock.py \
  --output output/potential_correction_mock
```

The default image is 200×200 at 0.02 arcsec/pixel, with two midpoint integration
samples per axis, a fixed true SIS+external-shear macro, two Gaussian source
clumps, and `PseudoJaffeSubhalo(b_sub=0.01, theta_E_host=1.0, center=(0.9,0.1))`.
The component's adopted tidal-radius relation is `rt=sqrt(b_sub*theta_E_host)`;
it is separate from the cored `PseudoJaffe` profile.

A small invocation suitable for inspection is:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python examples/potential_correction/generate_mock.py \
  --npix 16 --dpix 0.18 --nsub 2 --psf-size 3 --seed 91 \
  --output output/potential_correction_small
```

`--config` accepts JSON fields from `MockConfig`, including nested `source`
parameters, or the saved `configuration.json`. Command-line fields override
that configuration. Image sampling, source brightness, macro and subhalo
physical parameters, noise, PSF, mask radii, dtype, and output directory can
therefore all be configured. `--nsub` controls simulated-image integration;
subsequent inversions configure their own integration independently.

`subhalo.npz` and `control.npz` contain `image_data`, `noise_map`,
`noise_realization`, `model_image`, `psf_kernel`, `mask`, coordinate grids,
`true_potential`, `true_deflection_x/y`, `true_convergence`, and
`true_convergence_valid`. Arrays require no pickle. `configuration.json`
records physical values, noise seed/generator, units, integration, initialization
provenance, and truth conventions. Regenerate the identical inputs with:

```bash
python examples/potential_correction/generate_mock.py \
  --config output/potential_correction_small/configuration.json \
  --output output/potential_correction_small_repeated
```

The observations share the same independent Gaussian noise draw. The default
annular observation mask has `True` for excluded pixels. The generator follows
the current pixelized imaging convention: excluded native pixels contribute
zero brightness before native-grid circular FFT PSF convolution. It does not
add a PSF halo. No lens light is included. Source brightness is evaluated
analytically at subpixels before averaging; it is not first sampled on the
inversion source grid.

The Python interface supports truth-derived initialization directly:

```python
from examples.potential_correction.mock import MockConfig, generate_mock_pair

pair = generate_mock_pair(MockConfig())
observation = pair.subhalo
# observation.macro is the fixed true macro PhysicalModel.
# observation.sim_config is the corresponding SimulatorConfig.
# gx, gy = observation.source_truth.gradient(beta_x, beta_y)
# initial_source = observation.source_truth.brightness(source_x, source_y)
# true_potential = observation.subhalo_truth.potential(potential_x, potential_y)
```

`SubhaloTruth` uses the analytic singular-profile potential with zero at the
subhalo center. It reports physical point-sampled convergence, with `NaN` for
radii below the mass component's 1e-8 arcsec numerical radius convention.
Deflection follows that existing component convention, including a zero vector
at the exact center; no new physical core is introduced. Potential derivatives
and the analytic convergence/deflection relations should be compared outside
that tiny numerical domain. At zero strength every product is zero, including
at the center. Stored truth maps are point samples, not pixel averages. Later
coarse-grid convergence comparisons must separately state their finite-
difference/sampling convention.

Generation scopes the requested float32/float64 setting and preserves the
process's prior JAX precision configuration. Callers doing additional JAX
arithmetic on float64 outputs must keep an appropriate x64 context enabled;
otherwise JAX itself may downcast subsequent operations. These mock products
establish reproducible inputs and analytic truth, with no claim of recovery,
formal detection significance, mass precision, or exact correspondence to the
reference repository's simulation.
