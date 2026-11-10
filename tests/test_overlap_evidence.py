"""Independent exact-window references for pixelized operator evidence."""

from itertools import combinations
from math import prod
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from TinyLensGpu.Inference.param_u import ParamU
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import (
    PixelizedImageProbModelOperator,
)
from TinyLensGpu.PhysicalModel import PhysicalModel, PixelizedSourceModel
from tests.test_adapt_image import _Translation, _reference_projection
from tests.test_pixelized_interpolation_evidence import _evidence, _logdet


@pytest.mark.integration
def test_single_overlap_window_scores_the_complete_small_source():
    with jax.experimental.enable_x64():
        n, dpix, nsub, strength = 2, 0.25, 2, 0.4
        bbox = tuple(np.float64(v) for v in (-0.5, 0.5, -0.5, 0.5))
        y, x = np.indices((5, 5))
        image = 0.3 + 0.07 * np.cos(x + 2 * y)
        noise = 0.06 + 0.004 * x + 0.003 * y
        mask = (x == 2) & (y == 1)
        psf = np.array([[0.0, 0.04, 0.0], [0.09, 0.65, 0.12], [0.06, 0.04, 0.0]])
        _, _, design = _reference_projection(
            image, mask, image, psf, dpix, nsub, bbox, n, (0.0, 0.0),
        )
        # On the smallest grid second order reduces to first-difference and
        # outer zero-order boundary terms along each axis.
        difference = np.array([[-1.0, 1.0], [0.0, 1.0]])
        axis_precision = difference.T @ difference
        prior = strength * (
            np.kron(np.eye(n), axis_precision) + np.kron(axis_precision, np.eye(n))
        )
        data, sigma = image[~mask], noise[~mask]
        white_design = design / sigma[:, None]
        posterior = white_design.T @ white_design + prior
        expected_source = np.linalg.solve(posterior, white_design.T @ (data / sigma))
        prediction = design @ expected_source
        loss = 0.5 * (
            np.sum(((data - prediction) / sigma) ** 2) + expected_source @ prior @ expected_source
        )

        source = PixelizedSourceModel(
            n, log_lambda_reg=np.log(strength), regularization_type="second-order",
        )
        physical = PhysicalModel(lens_mass=[], source_light=[source])
        fit = PixelizedImageProbModelOperator(
            image, noise, psf, dpix, physical, mask=mask, nsub=nsub,
            fixed_source_bbox=bbox, block_size=1, evidence_logdet="overlap",
        )
        fit.pcg_rtol = 1e-12
        result = fit.evaluate()

        assert fit.evidence_block_size == 10
        assert result.solver_info.converged
        np.testing.assert_allclose(result.source_pixels, expected_source, atol=1e-10, rtol=0)
        np.testing.assert_allclose(
            result.log_evidence, _evidence(loss, sigma, _logdet(prior), _logdet(posterior)),
            atol=1e-10, rtol=0,
        )


def _finite_difference_prior(scale, spacing, order):
    """Form a complete precision by applying every global difference row."""
    n = scale.shape[0]
    if order == 0:
        return np.diag(scale.ravel())
    nodes = np.arange(n**2).reshape(n, n)
    unit = np.eye(n**2)
    differences = []
    weights = []
    for axis_rows in (nodes, nodes.T):
        for row in axis_rows:
            for index in range(n):
                support = row[index:index + order + 1]
                differences.append(np.diff(unit[support], n=len(support) - 1, axis=0)[0])
                weights.append(
                    np.prod(scale.ravel()[support]) ** (1 / len(support))
                    if len(support) > 1 else 1
                )
    differences = np.asarray(differences) / spacing**order
    return differences.T @ (np.asarray(weights)[:, None] * differences)


def _reference_overlap_ratio(prior, posterior, n, tile):
    """Apply inclusion-exclusion directly to intersections of maximal windows."""
    nodes = np.arange(n**2).reshape(n, n)
    starts = range(0, n - tile, tile)
    windows = [
        set(nodes[y:y + 2 * tile, x:x + 2 * tile].ravel())
        for y in starts for x in starts
    ]
    ratio = 0.0
    for count in range(1, len(windows) + 1):
        for subset in combinations(windows, count):
            overlap = sorted(set.intersection(*subset))
            if overlap:
                indices = np.ix_(overlap, overlap)
                ratio += (-1)**(count + 1) * (
                    _logdet(posterior[indices]) - _logdet(prior[indices])
                )
    return ratio


@pytest.mark.integration
@pytest.mark.parametrize("order,interpolation", [
    (0, "bilinear"), (1, "cubic_hermite"), (2, "cubic_hermite"),
])
def test_overlap_score_uses_exact_partial_windows_on_the_block_solution(order, interpolation):
    with jax.experimental.enable_x64():
        n, tile, dpix, nsub, strength, rho = 5, 2, 0.25, 2, 0.4, 1.1
        bbox = tuple(np.float64(v) for v in (-1.5, 1.5, -1.5, 1.5))
        y, x = np.indices((11, 11))
        image = 0.3 + 0.07 * np.cos(x + 2 * y)
        reference = 0.2 + 0.6 * np.cos(x / 3 + 0.7 * y)
        noise = 0.06 + 0.004 * x + 0.003 * y
        # Preserve periodic convolution along the full-height axis while
        # leaving room for the native FFT workspace to crop the other axis.
        mask = (x < 3) | (x > 7) | ((x == 4) & (y == 5))
        psf = np.array([[0.0, 0.04, 0.0], [0.09, 0.65, 0.12], [0.06, 0.04, 0.0]])
        brightness, _, design = _reference_projection(
            image, mask, reference, psf, dpix, nsub, bbox, n, (0.0, 0.0), interpolation,
        )
        scale = np.exp(rho * (1 - brightness)).reshape(n, n)
        prior = strength * _finite_difference_prior(scale, 0.75, order)
        data, sigma = image[~mask], noise[~mask]
        white_design = design / sigma[:, None]
        posterior = white_design.T @ white_design + prior
        ratio = _reference_overlap_ratio(prior, posterior, n, tile)
        assert abs(ratio - (_logdet(posterior) - _logdet(prior))) > 0.01

        def fit(mode):
            source = PixelizedSourceModel(
                n, log_lambda_reg=np.log(strength),
                regularization_type=("zero-order", "first-order", "second-order")[order],
                interpolation=interpolation, adaptive_reg_rho=rho,
            )
            model = PixelizedImageProbModelOperator(
                image, noise, psf, dpix, PhysicalModel(lens_mass=[], source_light=[source]),
                mask=mask, nsub=nsub, fixed_source_bbox=bbox, adapt_image=reference,
                block_size=3, evidence_logdet=mode, evidence_block_size=tile,
            )
            model.pcg_rtol = 1e-12
            return model.evaluate()

        overlap, block = fit("overlap"), fit("block")
        assert overlap.solver_info.converged and block.solver_info.converged
        np.testing.assert_allclose(overlap.source_pixels, block.source_pixels, atol=1e-12, rtol=0)
        np.testing.assert_allclose(overlap.model_image, block.model_image, atol=1e-12, rtol=0)
        pixels = np.asarray(overlap.source_pixels)
        prediction = design @ pixels
        np.testing.assert_allclose(overlap.model_image[~mask], prediction, atol=1e-12, rtol=0)
        loss = 0.5 * (np.sum(((data - prediction) / sigma) ** 2) + pixels @ prior @ pixels)
        np.testing.assert_allclose(
            overlap.log_evidence, _evidence(loss, sigma, 0, ratio), atol=1e-10, rtol=0,
        )


@pytest.mark.integration
def test_overlap_compiled_score_keeps_only_window_sized_design_panels():
    n, tile, size = 11, 2, 13
    y, x = np.indices((size, size))
    image = np.full((size, size), 0.3, dtype=np.float32)
    noise = np.full_like(image, 0.08)
    mask = (x < 3) | (x > 8) | (y < 3) | (y > 9) | ((x == 5) & (y == 5))
    psf = np.array([[0.0, 0.04, 0.0], [0.09, 0.65, 0.12], [0.06, 0.04, 0.0]], dtype=np.float32)
    shift = ParamU("shift_x", 0.01)
    shift.to_dynamic()
    physical = PhysicalModel(
        lens_mass=[_Translation(shift, 0.0)],
        source_light=[PixelizedSourceModel(n, log_lambda_reg=-1.3, interpolation="cubic_hermite")],
    )
    model = PixelizedImageProbModelOperator(
        image, noise, psf, 0.125, physical, mask=mask, nsub=2,
        fixed_source_bbox=(-0.625, 0.625, -0.625, 0.625),
        block_size=3, evidence_logdet="overlap", evidence_block_size=tile,
    )
    theta = jnp.array([0.013], dtype=jnp.float32)
    lowered = jax.jit(model).lower(theta)
    score = float(lowered.compile()(theta))
    assert np.isfinite(score) and score > -1e10

    # Audit the public score's allocation shapes, including flattened forms.
    native, active, source = image.size, int(np.count_nonzero(~mask)), n**2
    forbidden = {
        (native, native), (active, active), (source, source),
        tuple(sorted((active, source))), tuple(sorted((native, source))),
    }
    forbidden_sizes = {prod(shape) for shape in forbidden}
    ir = lowered.as_text()
    shapes = {
        tuple(int(axis) for axis in dims.rstrip("x").split("x"))
        for dims in re.findall(r"tensor<((?:[1-9]\d*x)+)(?:f\d+|complex<)", ir)
    }
    assert any(shape[-2:] == ((2 * tile)**2, (2 * tile)**2) for shape in shapes)
    assert not any(
        prod(shape) in forbidden_sizes
        or any(tuple(sorted(pair)) in forbidden for pair in combinations(shape, 2))
        for shape in shapes
    )
