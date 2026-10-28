"""Explicit source-grid precision at the existing reconstruction boundary."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
from TinyLensGpu.ForwardSimulation.LensImage.pixelized import PixelizedLensSimulator
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model import (
    PixelizedImageProbModel,
)
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import (
    PixelizedImageProbModelOperator,
)
from TinyLensGpu.PhysicalModel import PhysicalModel, PixelizedSourceModel


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["dense", "operator"])
def test_reconstruction_keeps_explicit_double_precision_source_geometry(backend):
    with jax.experimental.enable_x64():
        bbox = tuple(
            jnp.asarray(v, dtype=jnp.float64)
            for v in (-0.600000000123, 0.600000000321, -0.600000000123, 0.600000000321)
        )
        source = PixelizedSourceModel(n=3, log_lambda_reg=-3.0)
        source.log_lambda_reg.to_static()
        physical = PhysicalModel(lens_mass=[], source_light=[source], lens_light=[])
        psf = jnp.asarray([[0.02, 0.08, 0.02], [0.08, 0.60, 0.08], [0.02, 0.08, 0.02]])
        config = SimulatorConfig(dpix=0.2, npix=5, nsub=2, psf_kernel=psf)
        simulator = PixelizedLensSimulator(physical, config)
        truth = jnp.linspace(0.2, 1.0, 9)
        design, _ = simulator.design_matrix(source_bbox=bbox)
        image = (design @ truth).reshape(5, 5)
        kwargs = dict(
            image_data=image,
            noise_map=jnp.full_like(image, 0.05),
            psf_kernel=psf,
            dpix=0.2,
            phys_model=physical,
            nsub=2,
            fixed_source_bbox=bbox,
        )
        if backend == "dense":
            model = PixelizedImageProbModel(**kwargs, solver_type="cholesky")
        else:
            model = PixelizedImageProbModelOperator(
                **kwargs, solver_type="pcg", block_size=3
            )
            model.pcg_rtol = 1e-11
        result = model.evaluate()
        assert all(bound.dtype == jnp.float64 for bound in result.source_bbox)
        np.testing.assert_array_equal(np.asarray(result.source_bbox), np.asarray(bbox))
        np.testing.assert_array_equal(model.infer_source_bbox(), np.asarray(bbox))
        assert result.source_pixels.dtype == result.model_image.dtype == jnp.float64
        assert np.isfinite(result.log_evidence)
        # Independent Gaussian reference using the specified curvature rows,
        # including the existing first-/zero-order source-boundary fallback.
        difference = np.array([[1.0, -2.0, 1.0], [0.0, -1.0, 1.0], [0.0, 0.0, 1.0]])
        dx = float(bbox[1] - bbox[0]) / 2.0
        horizontal = np.kron(np.eye(3), difference)
        vertical = np.kron(difference, np.eye(3))
        prior = (
            np.exp(-3.0) * (horizontal.T @ horizontal + vertical.T @ vertical) / dx**4
        )
        mapping = np.asarray(design)
        curvature = mapping.T @ mapping / 0.05**2 + prior
        expected_source = np.linalg.solve(
            curvature, mapping.T @ np.asarray(image).ravel() / 0.05**2
        )
        expected_image = mapping @ expected_source
        np.testing.assert_allclose(
            result.source_pixels, expected_source, rtol=1e-9, atol=1e-9
        )
        np.testing.assert_allclose(
            result.model_image.ravel(), expected_image, rtol=1e-9, atol=1e-9
        )
        if backend == "dense":
            chi2 = np.sum(((np.asarray(image).ravel() - expected_image) / 0.05) ** 2)
            expected_logz = -0.5 * (
                chi2
                + expected_source @ prior @ expected_source
                + np.linalg.slogdet(curvature)[1]
                - np.linalg.slogdet(prior)[1]
                + image.size * np.log(2 * np.pi * 0.05**2)
            )
            np.testing.assert_allclose(
                result.log_evidence, expected_logz, rtol=1e-12, atol=1e-12
            )
        if result.solver_info is not None:
            assert result.solver_info.converged
