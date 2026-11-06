"""Small, fixed-strength fixtures for an expendable optimizer experiment."""

from dataclasses import asdict

import jax.numpy as jnp
import numpy as np

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer, SimulatorConfig
from TinyLensGpu.ObservationModel import IterativePotentialCorrection
from TinyLensGpu.PhysicalModel import PhysicalModel, SIS, Shear
from examples.potential_correction.mock import (
    MockConfig,
    SourceTruth,
    SubhaloTruth,
    generate_mock_pair,
)

CASES = ("ridge", "coupled", "nodal", "analytic_subhalo", "analytic_control")


def make_problem(case):
    """Caller enables x64 before building *any* inputs. No library edits."""
    if case.startswith("analytic_"):
        mock = MockConfig(
            npix=32,
            dpix=0.08,
            nsub=2,
            theta_E=0.8,
            b_sub=0.04,
            subhalo_x=0.75,
            subhalo_y=0.1,
            psf_sigma=0.05,
            psf_size=5,
            mask_inner=0.45,
            mask_outer=1.1,
        )
        pair = generate_mock_pair(mock)
        observation = getattr(pair, case.removeprefix("analytic_"))
        renderer = PotentialCorrectionRenderer(
            observation.macro,
            observation.sim_config,
            11,
            (-0.65, 0.65, -0.65, 0.65),
            potential_spacing=0.16,
            dtype="float64",
        )
        sx, sy = jnp.meshgrid(renderer.source_x_axis, renderer.source_y_axis)
        source = observation.source_truth.brightness(sx, sy).ravel()
        problem = IterativePotentialCorrection(
            renderer,
            observation.image_data,
            observation.noise_map,
            source,
            lambda_amplitude=1e4,
        )
        strengths = (6e-4, 1.6e7 if observation.has_subhalo else 1e11)
        metadata = dict(
            generation="independent analytic source + analytic Pseudo-Jaffe deflection + Gaussian noise",
            mock=asdict(mock),
            truth_initialization=True,
            truth_is_in_reconstruction_family=False,
        )
        return problem, strengths, metadata

    npix, dpix, source_n, bbox = (
        (12, 0.12, 5, (-0.6, 0.6, -0.6, 0.6))
        if case == "nodal"
        else (4, 0.1, 2, (-0.5, 0.5, -0.5, 0.5))
    )
    axis = (np.arange(npix) - (npix - 1) / 2) * dpix
    x, y = np.meshgrid(axis, axis)
    radius = np.hypot(x, y)
    mask = ((radius < 0.18) | (radius > 0.65)) if case == "nodal" else None
    macro = PhysicalModel(
        lens_mass=(
            [
                SIS(theta_E=0.4, center_x=0.0, center_y=0.0),
                Shear(gamma1=0.03, gamma2=0.01),
            ]
            if case == "nodal"
            else [Shear(gamma1=0.0, gamma2=0.0)]
        ),
        source_light=[],
        lens_light=[],
    )
    renderer = PotentialCorrectionRenderer(
        macro,
        SimulatorConfig(dpix, npix, jnp.ones((1, 1)), mask=mask),
        source_n,
        bbox,
        dtype="float64",
    )
    if case == "ridge":
        source, image, noise = np.full(4, 5.0), np.full((4, 4), 2.0), np.ones((4, 4))
        amplitude, strengths, regularization = 1e8, (3.0, 0.4), "zero-order"
        metadata = dict(
            generation="constant-source analytic ridge control", exact_source=8 / 7
        )
    elif case == "coupled":
        source = jnp.array([0.4, 1.1, 0.2, 1.4])
        grid = renderer.potential_grid
        potential = 0.003 * jnp.exp(
            -((grid.x - 0.07) ** 2 + (grid.y + 0.03) ** 2) / 0.02
        )
        image = renderer.render(source, potential).model_image
        image = image + 0.002 * jnp.sin(jnp.arange(16)).reshape((4, 4))
        noise = jnp.full_like(image, 0.07)
        amplitude, strengths, regularization = 20.0, (0.3, 0.02), "zero-order"
        metadata = dict(
            generation="same nodal coupled problem as the library nonlinear regression"
        )
    elif case == "nodal":
        sx, sy = jnp.meshgrid(renderer.source_x_axis, renderer.source_y_axis)
        source = (
            SourceTruth(sigma_x=0.22, sigma_y=0.17, clump_scale=0.8)
            .brightness(sx, sy)
            .ravel()
        )
        grid = renderer.potential_grid
        potential = SubhaloTruth(0.01, 0.4, 0.38, 0.07).potential(grid.x, grid.y)
        potential = potential - jnp.mean(potential)
        image = renderer.render(source, potential).model_image
        noise = jnp.full_like(image, 0.01)
        amplitude, strengths, regularization = 100.0, (1e-4, 100.0), "second-order"
        metadata = dict(
            generation="representable nodal SIS+shear/two-clump source/Pseudo-Jaffe mock"
        )
    else:
        raise ValueError(case)
    problem = IterativePotentialCorrection(
        renderer,
        image,
        noise,
        source,
        lambda_amplitude=amplitude,
        source_regularization=regularization,
    )
    return problem, strengths, metadata
