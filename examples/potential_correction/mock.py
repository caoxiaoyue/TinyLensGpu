"""Reproducible Pseudo-Jaffe mocks and independent analytic truth products.

Coordinates and deflections use arcseconds; lens potential uses arcsec squared.
These helpers generate observations and initialization, without fitting a model.
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class SubhaloTruth:
    """Singular Pseudo-Jaffe truth with ``rt = sqrt(b_sub * theta_E_host)``.

    Potential is the physical singular profile with its additive constant set
    by psi(center)=0. Deflection follows the existing component's numerical
    radius floor of 1e-8 arcsec, including zero vector at the exact center.
    Convergence is a point sample, NaN inside that numerical radius domain;
    no physical core or pixel averaging is introduced. At zero strength all
    products are identically zero, including at the center.
    """

    b_sub: float
    theta_E_host: float
    center_x: float
    center_y: float

    def __post_init__(self):
        if not np.isfinite([self.b_sub, self.theta_E_host,
                            self.center_x, self.center_y]).all():
            raise ValueError("Subhalo parameters must be finite")
        if self.b_sub < 0 or self.theta_E_host <= 0:
            raise ValueError("b_sub must be nonnegative and theta_E_host positive")

    @property
    def tidal_radius(self):
        return np.sqrt(self.b_sub * self.theta_E_host)

    def potential(self, x, y):
        """Return analytic potential, referenced to zero at the center."""
        radius = jnp.hypot(x - self.center_x, y - self.center_y)
        if self.b_sub == 0:
            return jnp.zeros_like(radius)
        rt = self.tidal_radius
        root = jnp.hypot(radius, rt)
        root_minus_rt = radius**2 / (root + rt)
        return self.b_sub * (
            radius - root_minus_rt + rt * jnp.log1p(root_minus_rt / (2 * rt))
        )

    def deflection(self, x, y):
        """Return the component-compatible vector deflection (alpha_x, alpha_y)."""
        dx, dy = x - self.center_x, y - self.center_y
        radius = jnp.maximum(jnp.hypot(dx, dy), 1e-8)
        rt = self.tidal_radius
        alpha_r = 2 * self.b_sub * rt / (radius + rt + jnp.hypot(radius, rt))
        return alpha_r * dx / radius, alpha_r * dy / radius

    def convergence(self, x, y):
        """Return point-sampled kappa; NaN for R<1e-8 when strength is nonzero."""
        radius = jnp.hypot(x - self.center_x, y - self.center_y)
        if self.b_sub == 0:
            return jnp.zeros_like(radius)
        rt = self.tidal_radius
        safe_radius = jnp.maximum(radius, 1e-8)
        value = self.b_sub / 2 * (1 / safe_radius - 1 / jnp.hypot(safe_radius, rt))
        return jnp.where(radius >= 1e-8, value, jnp.nan)


@dataclass(frozen=True)
class SourceTruth:
    """Two analytic Gaussian brightness clumps and exact spatial derivatives."""

    center_x: float = 0.05
    center_y: float = 0.03
    sigma_x: float = 0.12
    sigma_y: float = 0.08
    amplitude: float = 1.0
    clump_fraction: float = 0.35
    clump_dx: float = -0.13
    clump_dy: float = 0.1
    clump_scale: float = 0.5

    def __post_init__(self):
        from dataclasses import asdict

        if not np.isfinite(list(asdict(self).values())).all():
            raise ValueError("Source parameters must be finite")
        if min(self.sigma_x, self.sigma_y, self.clump_scale) <= 0:
            raise ValueError("Source widths and clump_scale must be positive")
        if min(self.amplitude, self.clump_fraction) < 0:
            raise ValueError("Source amplitudes must be nonnegative")

    def _components(self, x, y):
        for dx, dy, scale, fraction in (
            (0.0, 0.0, 1.0, 1.0),
            (self.clump_dx, self.clump_dy, self.clump_scale, self.clump_fraction),
        ):
            u, v = x - self.center_x - dx, y - self.center_y - dy
            sx, sy = self.sigma_x * scale, self.sigma_y * scale
            brightness = self.amplitude * fraction * jnp.exp(
                -0.5 * ((u / sx)**2 + (v / sy)**2)
            )
            yield brightness, u / sx**2, v / sy**2

    def brightness(self, x, y):
        """Evaluate true source intensity at arbitrary source-plane coordinates."""
        return sum(value for value, _, _ in self._components(x, y))

    def gradient(self, x, y):
        """Evaluate dI/dx and dI/dy analytically, without a sampled source grid."""
        components = tuple(self._components(x, y))
        return (sum(-value * gx for value, gx, _ in components),
                sum(-value * gy for value, _, gy in components))


@dataclass(frozen=True)
class MockConfig:
    """Physical and numerical inputs for a matched subhalo/control pair."""

    npix: int = 200
    dpix: float = 0.02
    nsub: int = 2
    theta_E: float = 1.0
    macro_center_x: float = 0.0
    macro_center_y: float = 0.0
    gamma1: float = 0.05
    gamma2: float = 0.02
    b_sub: float = 0.01
    subhalo_x: float = 0.9
    subhalo_y: float = 0.1
    noise_sigma: float = 0.01
    seed: int = 17
    psf_sigma: float = 0.04
    psf_size: int = 9
    mask_inner: float = 0.6
    mask_outer: float = 1.4
    dtype: str = "float64"
    source: SourceTruth = SourceTruth()

    @classmethod
    def from_dict(cls, values):
        """Reconstruct the complete input from configuration.json or plain JSON."""
        values = dict(values)
        if isinstance(values.get("source"), dict):
            values["source"] = SourceTruth(**values["source"])
        return cls(**values)

    def __post_init__(self):
        from dataclasses import asdict

        for name in ("npix", "nsub", "psf_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.psf_size > self.npix or self.psf_size % 2 != 1:
            raise ValueError("psf_size must be odd and no larger than npix")
        if self.dtype not in ("float64", "float32"):
            raise ValueError("dtype must be 'float64' or 'float32'")
        numeric = {k: v for k, v in asdict(self).items()
                   if k not in ("source", "dtype")}
        if not np.isfinite(list(numeric.values())).all():
            raise ValueError("Mock numerical parameters must be finite")
        if min(self.dpix, self.theta_E, self.noise_sigma) <= 0:
            raise ValueError("dpix, theta_E, and noise_sigma must be positive")
        if min(self.b_sub, self.psf_sigma, self.mask_inner) < 0:
            raise ValueError("b_sub, psf_sigma, and mask_inner must be nonnegative")
        if self.mask_outer <= self.mask_inner:
            raise ValueError("mask_outer must exceed mask_inner")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(self.source, SourceTruth):
            raise TypeError("source must be SourceTruth")


@dataclass(frozen=True)
class MockObservation:
    """One observation and sufficient callable truth to initialize later fits."""

    image_data: object
    noise_map: object
    psf_kernel: object
    mask: object
    model_image: object
    noise_realization: object
    sim_config: object
    macro: object
    source_truth: SourceTruth
    subhalo_truth: SubhaloTruth
    config: MockConfig
    has_subhalo: bool


@dataclass(frozen=True)
class MockPair:
    """Subhalo observation and control sharing exactly the same noise draw."""

    subhalo: MockObservation
    control: MockObservation


def generate_mock_pair(config=MockConfig()):
    """Generate a deterministic pair with fixed true SIS+shear macro parameters.

    Analytic source brightness is integrated at ``nsub`` midpoint samples per
    native axis. Native pixels excluded by the observation mask contribute zero
    before the existing pixelized native-grid circular PSF convolution. There
    is no lens light. The same independent Gaussian noise realization is added
    to both noiseless images, making their difference a controlled perturbation.
    Generation requests its dtype inside a scoped JAX precision context and
    leaves process-wide precision unchanged.
    """
    import jax

    from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
    from TinyLensGpu.ForwardSimulation.LensImage.parametric import bin_image_general
    from TinyLensGpu.ForwardSimulation.LensImage.pixelized_operator import (
        _psf_convolve_full_jit,
    )
    from TinyLensGpu.PhysicalModel import PhysicalModel, PseudoJaffeSubhalo, SIS, Shear

    with jax.experimental.enable_x64(config.dtype == "float64"):
        dtype = jnp.dtype(config.dtype)
        axis = (jnp.arange(config.npix, dtype=dtype) - config.npix / 2 + 0.5) * config.dpix
        x, y = jnp.meshgrid(axis, axis)
        radius = jnp.hypot(x, y)
        mask = (radius < config.mask_inner) | (radius > config.mask_outer)
        if np.asarray(mask).all():
            raise ValueError("The configured observation mask selects no pixels")
        psf_axis = (jnp.arange(config.psf_size, dtype=dtype) - config.psf_size//2) * config.dpix
        px, py = jnp.meshgrid(psf_axis, psf_axis)
        if config.psf_sigma == 0:
            psf = jnp.zeros_like(px).at[config.psf_size//2, config.psf_size//2].set(1)
        else:
            psf = jnp.exp(-(px**2 + py**2)/(2 * config.psf_sigma**2))
            psf = psf / jnp.sum(psf)
        sim = SimulatorConfig(config.dpix, config.npix, psf, config.nsub, mask)
        scalar = lambda value: jnp.asarray(value, dtype=dtype)
        macro = PhysicalModel(
            lens_mass=[SIS(theta_E=scalar(config.theta_E),
                           center_x=scalar(config.macro_center_x),
                           center_y=scalar(config.macro_center_y)),
                       Shear(gamma1=scalar(config.gamma1), gamma2=scalar(config.gamma2))],
            source_light=[], lens_light=[],
        )
        beta_x, beta_y = macro.deflection(sim.xgrid_sub, sim.ygrid_sub)
        kernel_pad = jnp.zeros((config.npix, config.npix), dtype=dtype)
        kernel_pad = kernel_pad.at[:config.psf_size, :config.psf_size].set(psf)
        kernel_pad = jnp.roll(kernel_pad, (-(config.psf_size//2),
                                         -(config.psf_size//2)), axis=(0, 1))
        psf_fft = jnp.fft.rfft2(kernel_pad)
        rng = np.random.default_rng(config.seed)
        noise = jnp.asarray(rng.standard_normal((config.npix, config.npix)) * config.noise_sigma,
                            dtype=dtype)
        noise_map = jnp.full_like(noise, config.noise_sigma)
        observations = []
        for strength in (config.b_sub, 0.0):
            truth = SubhaloTruth(strength, config.theta_E, config.subhalo_x, config.subhalo_y)
            mass = PseudoJaffeSubhalo(b_sub=scalar(strength), theta_E_host=scalar(config.theta_E),
                                     center_x=scalar(config.subhalo_x), center_y=scalar(config.subhalo_y))
            alpha_x, alpha_y = mass.deriv(sim.xgrid_sub, sim.ygrid_sub)
            brightness_sub = config.source.brightness(beta_x - alpha_x, beta_y - alpha_y)
            brightness = bin_image_general(brightness_sub, config.nsub)
            brightness = jnp.where(mask, 0, brightness)
            model = _psf_convolve_full_jit(brightness.ravel(), psf_fft,
                                            config.npix, config.npix).reshape(config.npix, config.npix)
            observations.append(MockObservation(
                image_data=model + noise, noise_map=noise_map, psf_kernel=psf, mask=mask,
                model_image=model, noise_realization=noise, sim_config=sim, macro=macro,
                source_truth=config.source, subhalo_truth=truth, config=config,
                has_subhalo=strength > 0,
            ))
        return MockPair(*observations)



def save_mock_pair(pair, output):
    """Write safe NPZ observations/truth and JSON reproducibility metadata."""
    import json
    from dataclasses import asdict
    from pathlib import Path

    import jax

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = pair.subhalo.config
    metadata = {
        "configuration": asdict(config),
        "units": {"coordinates": "arcsec", "deflection": "arcsec",
                  "potential": "arcsec^2", "convergence": "dimensionless",
                  "brightness": "arbitrary intensity units"},
        "macro": "fixed true SIS plus external shear; no lens light",
        "subhalo": "PseudoJaffeSubhalo; rt=sqrt(b_sub*theta_E_host)",
        "noise_convention": "same independent Gaussian realization in both observations",
        "integration": {"mock_nsub": config.nsub,
                        "rule": "midpoint average per native pixel",
                        "psf": "existing pixelized native-grid circular FFT",
                        "masked_input_light": "zero before PSF; no halo expansion"},
        "truth_conventions": {
            "potential": "analytic singular profile; psi(center)=0",
            "deflection": "existing component numerical radius=max(R,1e-8 arcsec)",
            "convergence": "point samples; NaN for R < 1e-8 arcsec",
            "zero_strength": "all products zero, including at the center",
            "source": "analytic Gaussian mixture; gradients evaluated directly",
        },
        "initialization": {"single_step": "source_truth.gradient at macro rays",
                           "iterative": "source_truth.brightness at fixed source nodes"},
        "seed_generator": "numpy.random.default_rng PCG64",
        "software": {"jax": jax.__version__, "numpy": np.__version__},
    }
    (output / "configuration.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with jax.experimental.enable_x64(config.dtype == "float64"):
        for name, observation in (("subhalo", pair.subhalo), ("control", pair.control)):
            x, y = observation.sim_config.xgrid, observation.sim_config.ygrid
            truth = observation.subhalo_truth
            alpha_x, alpha_y = truth.deflection(x, y)
            kappa = truth.convergence(x, y)
            np.savez_compressed(
                output / f"{name}.npz", image_data=np.asarray(observation.image_data),
                noise_map=np.asarray(observation.noise_map),
                noise_realization=np.asarray(observation.noise_realization),
                model_image=np.asarray(observation.model_image),
                psf_kernel=np.asarray(observation.psf_kernel), mask=np.asarray(observation.mask),
                xgrid=np.asarray(x), ygrid=np.asarray(y),
                true_potential=np.asarray(truth.potential(x, y)),
                true_deflection_x=np.asarray(alpha_x), true_deflection_y=np.asarray(alpha_y),
                true_convergence=np.asarray(kappa), true_convergence_valid=np.isfinite(np.asarray(kappa)),
            )
    return output
