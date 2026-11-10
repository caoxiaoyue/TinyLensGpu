"""Independent exact-window references for pixelized operator evidence."""

from itertools import combinations
from math import prod
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import nnls
from scipy.signal import convolve2d

from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.Inference.param_u import ParamU
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import (
    PixelizedImageProbModelOperator,
)
from TinyLensGpu.PhysicalModel import GaussianEllipse, PhysicalModel, PixelizedSourceModel
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


@pytest.fixture
def overlap_workspace_problem():
    n, tile, size = 11, 3, 13
    y, x = np.indices((size, size))
    image = np.full((size, size), 0.3, dtype=np.float32)
    noise = np.full_like(image, 0.08)
    mask = (x < 3) | (x > 8) | (y < 3) | (y > 9) | ((x == 5) & (y == 5))
    psf = np.array([[0.0, 0.04, 0.0], [0.09, 0.65, 0.12], [0.06, 0.04, 0.0]], dtype=np.float32)
    reference = (0.2 + 0.6 * np.cos(x / 3 + 0.7 * y)).astype(image.dtype)
    shift = ParamU("shift_x", 0.01)
    shift.to_dynamic()
    physical = PhysicalModel(
        lens_mass=[_Translation(shift, 0.0)],
        source_light=[PixelizedSourceModel(
            n, log_lambda_reg=-1.3, adaptive_reg_rho=0.7, interpolation="cubic_hermite",
        )],
    )
    model = PixelizedImageProbModelOperator(
        image, noise, psf, 0.125, physical, mask=mask, nsub=2,
        fixed_source_bbox=(-0.625, 0.625, -0.625, 0.625), adapt_image=reference,
        block_size=3, evidence_logdet="overlap", evidence_block_size=tile,
    )
    return model, image, mask, n, tile


def _real_ir_shapes(lowered):
    return {
        tuple(int(axis) for axis in dims.rstrip("x").split("x"))
        for dims in re.findall(r"tensor<((?:[1-9]\d*x)+)(?:f\d+|complex<)", lowered.as_text())
    }


@pytest.mark.integration
@pytest.mark.parametrize("differentiate", [False, True], ids=["score", "score-and-gradient"])
def test_overlap_compiled_score_keeps_only_window_sized_design_panels(
    overlap_workspace_problem, differentiate,
):
    model, image, mask, n, tile = overlap_workspace_problem
    theta = jnp.array([0.013], dtype=jnp.float32)
    evaluate = jax.value_and_grad(model) if differentiate else model
    lowered = jax.jit(evaluate).lower(theta)
    output = lowered.compile()(theta)
    if differentiate:
        score, gradient = output
        assert np.all(np.isfinite(gradient))
    else:
        score = output
    assert np.isfinite(score) and score > -1e10

    # Audit the public score's allocation shapes, including flattened forms.
    native, active, source = image.size, int(np.count_nonzero(~mask)), n**2
    forbidden = {
        (native, native), (active, active), (source, source),
        tuple(sorted((active, source))), tuple(sorted((native, source))),
    }
    forbidden_sizes = {prod(shape) for shape in forbidden}
    shapes = _real_ir_shapes(lowered)
    window = (2 * tile)**2
    assert max(prod(shape[:-2]) for shape in shapes if shape[-2:] == (window, window)) == 1
    assert not any(
        prod(shape) in forbidden_sizes
        or any(tuple(sorted(pair)) in forbidden for pair in combinations(shape, 2))
        for shape in shapes
    )


@pytest.mark.integration
@pytest.mark.parametrize("chunk_size,width", [(None, 1), (2, 2)])
def test_overlap_sampler_preserves_batch_shape_with_bounded_chunks(
    overlap_workspace_problem, chunk_size, width,
):
    model, _, _, _, tile = overlap_workspace_problem
    likelihood = make_likelihood(model, vectorized=True, vectorized_chunk_size=chunk_size)
    points = jnp.linspace(0.007, 0.019, 5, dtype=jnp.float32)[:, None]
    lowered = jax.jit(likelihood).lower(points)

    # Count how many maximal window curvatures are assembled at once. A
    # vectorized call over the whole sampler batch would use five here.
    window = (2 * tile)**2
    assert max(
        prod(shape[:-2]) for shape in _real_ir_shapes(lowered) if shape[-2:] == (window, window)
    ) == width

    actual = np.asarray(lowered.compile()(points))
    expected = [likelihood(point) for point in points]
    assert actual.shape == (5,)
    assert all(isinstance(score, float) for score in expected)
    assert np.all(actual > -1e10)
    np.testing.assert_allclose(actual, expected, atol=3e-4, rtol=0)


def _joint_problem(
    order=1, solver="pcg", mode="overlap", *, n=3, tile=10, warm_start=False,
):
    size, dpix, nsub = 7, 0.125, 2
    strength, rho, lens_strength = 0.025, 1.2, 0.001
    bbox = tuple(np.float64(value) for value in (-0.375, 0.375, -0.375, 0.375))
    y, x = np.indices((size, size))
    mask = ((x == 0) & (y == 0)) | ((x == 3) & (y == 5))
    psf = np.array([[0., 0.04, 0.], [0.09, 0.65, 0.12], [0.06, 0.04, 0.]])
    image = np.zeros_like(x, dtype=np.float64)
    noise = 0.05 + 0.003 * x + 0.002 * y
    reference = 0.2 + 0.6 * np.cos(x / 3 + 0.7 * y)
    brightness, _, source_design = _reference_projection(
        image, mask, reference, psf, dpix, nsub, bbox, n, (0.0, 0.0), "cubic_hermite",
    )
    source_prior = strength * _finite_difference_prior(
        np.exp(rho * (1 - brightness)).reshape(n, n), 0.75 / (n - 1), order,
    )

    y_sub, x_sub = (np.indices((size * nsub, size * nsub)) + 0.5) * dpix / nsub - size * dpix / 2
    lens_bases, lens_columns = [], []
    for width, cx, cy in ((0.18, 0.015, -0.025), (0.24, -0.035, 0.04)):
        lens = GaussianEllipse(flux=1.0, sigma=width, e1=0., e2=0., center_x=cx, center_y=cy)
        lens.flux.to_static(1.0)
        lens_bases.append(lens)
        profile = np.exp(-((x_sub - cx)**2 + (y_sub - cy)**2) / (2 * width**2))
        native = profile.reshape(size, nsub, size, nsub).mean(axis=(1, 3))
        lens_columns.append(convolve2d(native, psf, mode="same")[~mask] / (2 * np.pi * width**2))

    design = np.column_stack((source_design, *lens_columns))
    prior = np.zeros((design.shape[1], design.shape[1]))
    prior[:n**2, :n**2] = source_prior
    prior[n**2:, n**2:] = lens_strength * np.eye(len(lens_bases))
    source = PixelizedSourceModel(
        n, log_lambda_reg=np.log(strength), adaptive_reg_rho=rho, interpolation="cubic_hermite",
        regularization_type=("zero-order", "first-order", "second-order")[order],
    )
    initial = position = None
    if warm_start:
        # Construct an observation with a known, nonzero MAP initial point.
        # The primary solve can start converged even with zero iterations.
        initial = np.linspace(0.1, 0.6, n**2)
        white = design / noise[~mask, None]
        gram = white.T @ white
        rhs = (gram + prior) @ np.r_[initial, 0., 0.]
        image[~mask] = noise[~mask] * (white @ np.linalg.solve(gram, rhs))
        source.log_lambda_reg.to_dynamic()
        position = dict(
            positions=[(-0.1, 0.), (0.1, 0.)],
            threshold_arcsec=0.05,
            min_log_like=-100.,
        )
    else:
        image[~mask] = design @ np.r_[np.linspace(-0.3, 0.8, n**2), 0.18, 0.43]
    physical = PhysicalModel(lens_mass=[], source_light=[source], lens_light=lens_bases)
    model = PixelizedImageProbModelOperator(
        image, noise, psf, dpix, physical, mask=mask, nsub=nsub,
        fixed_source_bbox=bbox, adapt_image=reference, block_size=1,
        lens_light_regularization=lens_strength, solver_type=solver, evidence_logdet=mode,
        pnpg_max_iter=3000, pnpg_rtol=1e-8, initial_source=initial,
        evidence_block_size=tile, position_likelihood=position,
    )
    model.pcg_rtol = 1e-12
    return model, prior, design, image[~mask], noise[~mask], mask


@pytest.mark.integration
@pytest.mark.parametrize(
    "order,n,tile,solver", [(0, 2, 10, "pcg"), (1, 3, 10, "pcg"), (2, 3, 1, "pnpg")],
)
def test_joint_overlap_scores_physical_curvature_on_the_primary_solution(order, n, tile, solver):
    with jax.experimental.enable_x64():
        model, prior, design, data, sigma, mask = _joint_problem(order, solver, n=n, tile=tile)
        result, diagnostics = model.evaluate_with_diagnostics()
        assert result.solver_info.converged and diagnostics.valid
        assert diagnostics.evidence_valid and diagnostics.evidence_invalid_reason is None
        assert diagnostics.evidence_logdet == "overlap"
        coefficients = np.r_[result.source_pixels, result.lens_amplitudes]
        white = design / sigma[:, None]
        posterior = white.T @ white + prior
        if solver == "pnpg":
            augmented = np.vstack((white, np.linalg.cholesky(prior).T))
            expected_coefficients, _ = nnls(augmented, np.r_[data / sigma, np.zeros(len(prior))])
            assert np.any(coefficients == 0)
        else:
            expected_coefficients = np.linalg.solve(posterior, white.T @ (data / sigma))
        np.testing.assert_allclose(
            coefficients, expected_coefficients, atol=2e-6 if solver == "pnpg" else 1e-9, rtol=0,
        )

        block_model, *_ = _joint_problem(order, solver, "block", n=n, tile=tile)
        block = block_model.evaluate()
        np.testing.assert_allclose(
            coefficients, np.r_[block.source_pixels, block.lens_amplitudes], atol=2e-10, rtol=0,
        )
        prediction = design @ coefficients
        np.testing.assert_allclose(result.model_image[~mask], prediction, atol=1e-10, rtol=0)
        loss = 0.5 * (np.sum(((data - prediction) / sigma) ** 2) + coefficients @ prior @ coefficients)
        if n <= 2 * tile:
            ratio = _logdet(posterior) - _logdet(prior)
        else:
            split = n**2
            source_posterior = posterior[:split, :split]
            cross = posterior[:split, split:]
            schur = posterior[split:, split:] - cross.T @ np.linalg.solve(source_posterior, cross)
            ratio = (
                _reference_overlap_ratio(prior[:split, :split], source_posterior, n, tile)
                + _logdet(schur) - _logdet(prior[split:, split:])
            )
        np.testing.assert_allclose(
            result.log_evidence, _evidence(loss, sigma, 0, ratio), atol=1e-9, rtol=0,
        )


@pytest.mark.integration
def test_joint_overlap_distinguishes_auxiliary_failure_from_primary_failure():
    with jax.experimental.enable_x64():
        model, *_ = _joint_problem(warm_start=True)
        model.pcg_max_iter = 0
        theta = model.get_values("flat")
        result, diagnostics = model.evaluate_with_diagnostics(theta, include_components=True)
        assert result.solver_info.converged and diagnostics.valid
        assert diagnostics.invalid_reason is None
        assert diagnostics.evidence_valid is False
        assert "auxiliary source-curvature solve" in diagnostics.evidence_invalid_reason
        assert result.log_evidence == -1e10
        assert diagnostics.position.penalty < 0
        assert np.any(result.model_image != 0)
        np.testing.assert_allclose(
            result.model_image, diagnostics.source_image + diagnostics.lens_light_image,
            atol=1e-12, rtol=0,
        )
        value_and_grad = jax.jit(jax.value_and_grad(model))
        score, gradient = value_and_grad(theta)
        assert score == -1e10
        assert np.all(np.isnan(gradient))

        result, diagnostics = model.evaluate_with_diagnostics(theta + 0.1)
        assert not result.solver_info.converged and not diagnostics.valid
        assert diagnostics.evidence_valid is False
        assert "primary reconstruction" in diagnostics.evidence_invalid_reason
        assert result.log_evidence == -1e10 + diagnostics.position.penalty
        assert np.all(result.model_image == 0)
        assert np.all(result.source_pixels == 0)
        assert np.all(result.lens_amplitudes == 0)
        score, gradient = value_and_grad(theta + 0.1)
        assert score == result.log_evidence
        assert np.all(np.isnan(gradient))


@pytest.mark.integration
def test_joint_overlap_reconstruction_uses_one_primary_solve_and_skips_score_work(monkeypatch):
    with jax.experimental.enable_x64():
        model, *_ = _joint_problem()
        solve = model._solve_source
        calls = []

        def counted_solve(*args, **kwargs):
            calls.append(None)
            return solve(*args, **kwargs)

        monkeypatch.setattr(model, "_solve_source", counted_solve)
        result, diagnostics = model.evaluate_with_diagnostics(include_components=True)
        assert len(calls) == 1
        assert diagnostics.valid and diagnostics.evidence_valid

        def unexpected(*args, **kwargs):
            raise AssertionError("reconstruction requested score-only work")

        monkeypatch.setattr(model, "_overlap_logdet", unexpected)
        monkeypatch.setattr(model, "_logdet_lens_schur", unexpected)
        image, source, lens = model.forward_model(return_components=True)
        assert len(calls) == 2
        np.testing.assert_allclose(image, result.model_image, atol=2e-10, rtol=0)
        np.testing.assert_allclose(source, result.source_pixels, atol=2e-10, rtol=0)
        np.testing.assert_allclose(lens, result.lens_amplitudes, atol=2e-10, rtol=0)
