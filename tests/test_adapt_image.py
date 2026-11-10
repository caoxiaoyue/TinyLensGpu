"""Public inversion checks for reference-image source adaptation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.Inference.param_u import ParamU
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model import (
    PixelizedImageProbModel,
)
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import (
    PixelizedImageProbModelOperator,
)
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.Light import PixelizedSourceModel
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
import caskade as ck


class _Translation(ck.Module):
    def __init__(self, shift_x, shift_y):
        super().__init__()
        self.shift_x = shift_x
        self.shift_y = shift_y

    @ck.forward
    def deriv(self, x, y):
        return jnp.full_like(x, self.shift_x.value), jnp.full_like(y, self.shift_y)


def _hermite_kernel(distance, array_module=np):
    """Cardinal cubic kernel for a node at a given distance in grid units."""
    distance = array_module.abs(distance)
    return array_module.where(
        distance < 1,
        1 + (1.5 * distance - 2.5) * distance**2,
        array_module.where(distance < 2, -0.5 * (distance - 1) * (distance - 2)**2, 0),
    )


def _reference_projection(image, mask, reference, psf, dpix, nsub, bbox, n, shift,
                          interpolation="bilinear"):
    """Independently form the geometric coverage and periodic blurred design."""
    xmin, xmax, ymin, ymax = bbox
    positive = np.maximum(reference, 0)
    normalized = np.minimum(positive / max(np.percentile(positive[~mask], 99.5), 1e-10), 1)
    mapping = np.zeros((*image.shape, n**2))
    hermite = interpolation == "cubic_hermite"
    projected = np.zeros_like(mapping) if hermite else mapping
    nodes = np.arange(n)
    for iy, ix in np.argwhere(~mask):
        for sy, sx in np.ndindex(nsub, nsub):
            x = dpix * (ix + (sx + 0.5) / nsub - image.shape[1] / 2) - shift[0]
            y = dpix * (iy + (sy + 0.5) / nsub - image.shape[0] / 2) - shift[1]
            gx = (x - xmin) / ((xmax - xmin) / (n - 1))
            gy = (y - ymin) / ((ymax - ymin) / (n - 1))
            if hermite:
                weights = np.outer(_hermite_kernel(gy - nodes), _hermite_kernel(gx - nodes))
                projected[iy, ix] += weights.ravel()**2 / nsub**2
            elif 0 <= gx <= n - 1 and 0 <= gy <= n - 1:
                weights = np.outer(np.maximum(0, 1 - abs(gy - nodes)),
                                   np.maximum(0, 1 - abs(gx - nodes)))
            else:
                continue
            mapping[iy, ix] += weights.ravel() / nsub**2

    coverage = projected.sum(axis=(0, 1))
    signal = (projected * normalized[..., None]).sum(axis=(0, 1))
    if hermite:
        brightness = signal / (coverage + 1e-4)
    else:
        brightness = np.divide(signal, coverage, out=np.zeros_like(signal), where=coverage > 1e-8)
    convolved = sum(
        psf[jy, jx] * np.roll(
            mapping, (jy - psf.shape[0] // 2, jx - psf.shape[1] // 2), axis=(0, 1)
        )
        for jy, jx in np.ndindex(psf.shape)
    )
    return brightness, coverage, convolved[~mask]


def _fixture():
    image = np.linspace(0.3, 2, 16).reshape(4, 4)
    noise = np.linspace(0.1, 0.2, 16).reshape(4, 4)
    reference = np.array([
        [1e6, 2, 1, 0],
        [0.2, -4, 8, 0.5],
        [5, 4, 3, 0.1],
        [0.3, 1, 0.5, 2],
    ])
    mask = np.zeros(image.shape, dtype=bool)
    mask[0, 0] = mask[1, 2] = True
    psf = np.array([[0.0, 0.1, 0.0], [0.05, 0.6, 0.2], [0.0, 0.05, 0.0]])
    return image, noise, reference, mask, psf


def _model(model_type, image, noise, psf, rho, shift, *, interpolation="bilinear", **kwargs):
    rho_is_pointer = isinstance(rho, ck.Param) and rho.pointer
    source = PixelizedSourceModel(
        n=5, log_lambda_reg=float(np.log(0.7)), regularization_type="zero-order",
        adaptive_reg_rho=0.0 if rho_is_pointer else rho, interpolation=interpolation,
    )
    if rho_is_pointer:
        del source.adaptive_reg_rho
        source.adaptive_reg_rho = rho
    shift_x = shift[0] if isinstance(shift[0], ck.Param) else ParamU("shift_x", shift[0])
    mass = _Translation(shift_x, shift[1])
    physical = PhysicalModel(lens_mass=[mass], source_light=[source])
    model = model_type(image, noise, psf, np.float64(0.2), physical, **kwargs)
    if isinstance(model, PixelizedImageProbModelOperator):
        model.pcg_rtol = 1e-9
    return model


@pytest.mark.unit
@pytest.mark.parametrize("interpolation", ["bilinear", "cubic_hermite"])
def test_masked_adapt_image_matches_independent_inversion(interpolation):
    image, noise, reference, mask, psf = _fixture()
    hermite = interpolation == "cubic_hermite"
    limit = np.float64(0.4 if hermite else 0.49)
    bbox = (-limit, limit, -limit, limit)
    shift = (0.4 if hermite else 0.055, -0.065)
    rho = 1.4
    brightness, coverage, design = _reference_projection(
        image, mask, reference, psf, 0.2, 2, bbox, 5, shift, interpolation,
    )
    assert np.any(coverage == 0)
    if hermite:
        assert np.min(design) < -0.02
    scale = np.exp(rho * (1 - brightness))
    whitened = design / noise[~mask, None]
    prior = 0.7 * np.diag(scale)
    curvature = whitened.T @ whitened + prior
    source = np.linalg.solve(curvature, whitened.T @ (image[~mask] / noise[~mask]))
    predicted = design @ source
    chi2 = np.sum(((image[~mask] - predicted) / noise[~mask]) ** 2)
    dense_score = -0.5 * (
        chi2 + source @ prior @ source
        + np.linalg.slogdet(curvature)[1] - np.linalg.slogdet(prior)[1]
        + np.log(2 * np.pi * noise[~mask] ** 2).sum()
    )

    with jax.experimental.enable_x64():
        for model_type in (PixelizedImageProbModel, PixelizedImageProbModelOperator):
            model = _model(
                model_type, image, noise, psf, rho, shift, mask=mask, nsub=2,
                fixed_source_bbox=bbox, adapt_image=reference, interpolation=interpolation,
            )
            result, diagnostics = model.evaluate_with_diagnostics(include_components=True)
            assert diagnostics.valid
            assert diagnostics.regularization_scale.dtype == np.float64
            np.testing.assert_allclose(diagnostics.regularization_scale.ravel(), scale, rtol=1e-9)
            np.testing.assert_allclose(result.source_pixels, source, rtol=1e-7, atol=2e-8)
            np.testing.assert_allclose(result.model_image[~mask], predicted, rtol=1e-7)
            np.testing.assert_allclose(diagnostics.source_image, result.model_image, rtol=1e-10)
            np.testing.assert_allclose(diagnostics.chi2, chi2, rtol=1e-7)
            np.testing.assert_allclose(model.evaluate().log_evidence, result.log_evidence)
            forward_image, forward_source = model.forward_model(return_source=True)
            np.testing.assert_allclose(forward_image, result.model_image, rtol=1e-10)
            np.testing.assert_allclose(forward_source, result.source_pixels, rtol=1e-10)
            if model_type is PixelizedImageProbModel:
                np.testing.assert_allclose(result.log_evidence, dense_score, rtol=1e-7)


_MODEL_TYPES = (PixelizedImageProbModel, PixelizedImageProbModelOperator)


@pytest.mark.unit
@pytest.mark.parametrize("model_type", _MODEL_TYPES)
def test_adapt_image_requires_valid_native_reference_and_fixed_grid(model_type):
    image, noise, reference, mask, psf = _fixture()
    shift = (0.055, -0.065)
    bbox = (-0.49, 0.49, -0.49, 0.49)
    rho = ParamU("adaptive_reg_rho", 0.0, prior_settings=[0.0, 3.0], limits=[0.0, 3.0])
    rho.to_dynamic()

    def build(**kwargs):
        return _model(model_type, image, noise, psf, rho, shift, mask=mask, **kwargs)

    with pytest.raises(ValueError, match="adapt_image"):
        build(fixed_source_bbox=bbox)
    with pytest.raises(ValueError, match="fixed_source_bbox"):
        build(adapt_image=reference)
    with pytest.raises(ValueError, match="fixed_source_bbox must be square"):
        build(adapt_image=reference, fixed_source_bbox=(-1.0, 1.0, -0.49, 0.49))
    with pytest.raises(ValueError, match="adapt_image must have shape"):
        build(fixed_source_bbox=bbox, adapt_image=reference.ravel())
    for value in (np.nan, np.inf):
        invalid = reference.copy()
        invalid[0, 0] = value  # Masked pixels must still be finite in the input.
        with pytest.raises(ValueError, match="adapt_image values must be finite"):
            build(fixed_source_bbox=bbox, adapt_image=invalid)
    for removed_keyword in ("fixed_reg_template", "fixed_reg_scale"):
        with pytest.raises(TypeError, match=removed_keyword):
            build(fixed_source_bbox=bbox, adapt_image=reference, **{removed_keyword: np.ones(25)})

    uniform = _model(model_type, image, noise, psf, 1e-12, shift, mask=mask)
    _, diagnostics = uniform.evaluate_with_diagnostics()
    assert diagnostics.valid
    np.testing.assert_array_equal(diagnostics.regularization_scale, np.ones((5, 5)))


@pytest.mark.unit
@pytest.mark.parametrize("model_type", _MODEL_TYPES)
@pytest.mark.parametrize("reference_value", [-1.0, 0.0])
def test_dark_adapt_image_uses_finite_faint_node_scale(model_type, reference_value):
    image, noise, _, mask, psf = _fixture()
    rho = 1.4
    model = _model(
        model_type, image, noise, psf, rho, (0.055, -0.065), mask=mask,
        fixed_source_bbox=(-0.49, 0.49, -0.49, 0.49),
        adapt_image=np.full_like(image, reference_value),
    )
    _, diagnostics = model.evaluate_with_diagnostics()

    assert diagnostics.valid
    np.testing.assert_allclose(diagnostics.regularization_scale, np.exp(rho), rtol=1e-6)


@pytest.mark.unit
def test_bilinear_adapt_image_discards_weak_nonzero_coverage():
    image, noise, reference, _, _ = _fixture()
    mask = np.ones(image.shape, dtype=bool)
    mask[1, 1] = False
    reference[1, 1] = 7.0
    psf = np.ones((1, 1))
    rho = 1.4
    bbox = tuple(np.float64(v) for v in (-0.4, 0.4, -0.4, 0.4))

    with jax.experimental.enable_x64():
        for fractional_x in (0.5e-8, 2e-8):
            shift = (-0.1 - 0.2 * fractional_x, -0.14)
            brightness, coverage, _ = _reference_projection(
                image, mask, reference, psf, 0.2, 1, bbox, 5, shift,
            )
            assert np.any((coverage > 0) & (coverage <= 1e-8))
            assert brightness[2 * 5 + 3] == (0.0 if fractional_x < 1e-8 else 1.0)
            for model_type in _MODEL_TYPES:
                model = _model(
                    model_type, image, noise, psf, rho, shift, mask=mask,
                    fixed_source_bbox=bbox, adapt_image=reference,
                )
                _, diagnostics = model.evaluate_with_diagnostics()
                assert diagnostics.valid
                np.testing.assert_allclose(
                    diagnostics.regularization_scale.ravel(),
                    np.exp(rho * (1 - brightness)), rtol=1e-9,
                )


@pytest.mark.unit
@pytest.mark.parametrize("model_type", _MODEL_TYPES)
@pytest.mark.parametrize("vary", ["mass_pointer", "rho_pointer"])
@pytest.mark.parametrize("interpolation", ["bilinear", "cubic_hermite"])
def test_adapt_image_updates_registered_dependencies_across_batches(model_type, vary, interpolation):
    image, noise, reference, mask, psf = _fixture()
    bbox = tuple(np.float64(v) for v in (-0.49, 0.49, -0.49, 0.49))
    shift_y = -0.065
    fixed_shift_x = 0.055
    fixed_rho = 1.4

    with jax.experimental.enable_x64():
        if vary == "mass_pointer":
            proposals = jnp.array([[0.055], [0.097], [0.055]])
            driver = ParamU("shift_driver", fixed_shift_x)
            driver.to_dynamic()
            shift_x = ck.Param("shift_x", lambda _: driver.value, link=(driver,))
            rho = fixed_rho
        else:
            proposals = jnp.array([[0.0], [1.8], [0.0]])
            driver = ParamU("rho_driver", 0.0, prior_settings=[0.0, 3.0], limits=[0.0, 3.0])
            driver.to_dynamic()
            rho = ck.Param("adaptive_reg_rho", lambda _: driver.value, link=(driver,))
            shift_x = fixed_shift_x

        options = dict(
            mask=mask, nsub=2, fixed_source_bbox=bbox, adapt_image=reference,
            interpolation=interpolation,
        )
        model = _model(model_type, image, noise, psf, rho, (shift_x, shift_y), **options)
        assert [parameter.name for parameter in model.get_dynamic_params()] == [driver.name]
        compiled = jax.jit(jax.vmap(model.evaluate))(proposals)
        loglike = make_likelihood(model, vectorized=True)
        np.testing.assert_allclose(loglike(proposals), compiled.log_evidence, rtol=1e-9)

        for index, proposal in enumerate(proposals):
            value = float(proposal[0])
            current_x = value if vary == "mass_pointer" else fixed_shift_x
            current_rho = value if vary == "rho_pointer" else fixed_rho
            brightness, _, _ = _reference_projection(
                image, mask, reference, psf, 0.2, 2, bbox, 5, (current_x, shift_y),
                interpolation,
            )
            expected_scale = np.exp(current_rho * (1 - brightness))
            _, diagnostics = model.evaluate_with_diagnostics(proposal)
            assert diagnostics.valid
            np.testing.assert_allclose(
                diagnostics.regularization_scale.ravel(), expected_scale, rtol=1e-9,
            )
            fixed = _model(
                model_type, image, noise, psf, current_rho, (current_x, shift_y), **options,
            ).evaluate()
            np.testing.assert_allclose(
                compiled.source_pixels[index], fixed.source_pixels, rtol=1e-7, atol=2e-8,
            )
            np.testing.assert_allclose(compiled.log_evidence[index], fixed.log_evidence, rtol=1e-8)
        assert not np.allclose(compiled.source_pixels[0], compiled.source_pixels[1])


@pytest.mark.unit
@pytest.mark.parametrize("column,shifts", [
    (2, (-0.15002, -0.15, -0.14998)),
    (1, (-0.33,)),
], ids=["knot_crossing", "weak_coverage"])
def test_hermite_conditional_prior_derivative(column, shifts):
    image, noise, reference, _, _ = _fixture()
    mask = np.ones(image.shape, dtype=bool)
    mask[1, 1] = False
    reference[1, 1] = 7.0
    psf = np.ones((1, 1))
    bbox = tuple(np.float64(v) for v in (-0.4, 0.4, -0.4, 0.4))
    node = 2 * 5 + column
    rho, shift_y = 1.4, -0.1
    interpolation = "cubic_hermite"

    def expected_energy(shift_x):
        brightness, coverage, _ = _reference_projection(
            image, mask, reference, psf, 0.2, 2, bbox, 5, (shift_x, shift_y), interpolation,
        )
        if column == 1:
            assert 0 < coverage[node] < 1e-4
        return 0.5 * 0.7 * np.exp(rho * (1 - brightness[node]))

    with jax.experimental.enable_x64():
        shift_x = ParamU("shift_x", shifts[0])
        shift_x.to_dynamic()
        model = _model(
            PixelizedImageProbModel, image, noise, psf, rho, (shift_x, shift_y),
            mask=mask, nsub=2, fixed_source_bbox=bbox, adapt_image=reference,
            interpolation=interpolation,
        )
        assert model.get_dynamic_params() == (shift_x,)

        def conditional_energy(theta):
            model_image, source = model.forward_model(theta, return_source=True)
            # Recover the diagonal prior precision from the normal equation.
            # The conditional energy uses a fixed unit coefficient at this node.
            dx = (jnp.array([-0.15, -0.05]) - theta[0]) / 0.2 + 2 - column
            dy = jnp.array([-0.25, 0.25])
            response = _hermite_kernel(dx, jnp).mean() * _hermite_kernel(dy, jnp).mean()
            residual = (image[1, 1] - model_image[1, 1]) / noise[1, 1]**2
            return 0.5 * response * residual / source[node]

        value_and_grad = jax.jit(jax.value_and_grad(conditional_energy))
        for shift in shifts:
            value, gradient = value_and_grad(jnp.array([shift], dtype=jnp.float64))
            step = 1e-6
            finite_difference = (
                expected_energy(shift + step) - expected_energy(shift - step)
            ) / (2 * step)
            assert abs(finite_difference) > 1e-5
            np.testing.assert_allclose(value, expected_energy(shift), rtol=2e-10)
            np.testing.assert_allclose(gradient[0], finite_difference, rtol=2e-5, atol=1e-7)


@pytest.mark.unit
def test_dense_invalid_exact_evidence_preserves_reconstruction_and_rejects_gradients():
    with jax.experimental.enable_x64(False):
        data = np.linspace(0.25, 1.0, 16, dtype=np.float32).reshape(4, 4)
        rho = ParamU("adaptive_reg_rho", 1.0, prior_settings=[0.0, 60.0], limits=[0.0, 60.0])
        rho.to_dynamic()
        source = PixelizedSourceModel(
            n=2, log_lambda_reg=-3.0, adaptive_reg_rho=rho,
            regularization_type="first-order",
        )
        source.log_lambda_reg.to_dynamic()
        model = PixelizedImageProbModel(
            data, np.full_like(data, 0.1), np.ones((1, 1), dtype=np.float32), 0.2,
            PhysicalModel(source_light=[source]), fixed_source_bbox=(-0.4, 0.4, -0.4, 0.4),
            adapt_image=np.zeros_like(data),
            position_likelihood=dict(
                positions=[[-0.2, 0.0], [0.2, 0.0]], threshold_arcsec=0.1,
                min_log_like=-1.0e4,
            ),
        )
        # The Float32 prior loses its small boundary eigenvalue before scaling,
        # while the design keeps the posterior curvature invertible.
        values = {"log_lambda_reg": -40.0, "adaptive_reg_rho": 40.0}
        point = jnp.asarray([values[parameter.name] for parameter in model.get_dynamic_params()])
        result, diagnostics = model.evaluate_with_diagnostics(point, include_components=True)
        score, gradient = jax.jit(jax.value_and_grad(model))(point)

        assert len(result) == 6
        assert diagnostics.valid and diagnostics.position.penalty < 0.0
        assert np.isfinite(result.source_pixels).all() and np.any(result.model_image != 0.0)
        assert diagnostics.evidence_logdet == "exact"
        assert diagnostics.evidence_valid is False
        assert "nonfinite" in diagnostics.evidence_invalid_reason
        assert score == result.log_evidence == -1e10
        assert np.isnan(gradient).all()
        np.testing.assert_allclose(model.forward_model(point), result.model_image, atol=1e-6)
        np.testing.assert_array_equal(diagnostics.source_image, result.model_image)

        valid_point = model.get_values("flat")
        _, diagnostics = model.evaluate_with_diagnostics(valid_point)
        assert diagnostics.evidence_valid and diagnostics.evidence_invalid_reason is None
        valid_score, directional = jax.jvp(model, (valid_point,), (jnp.ones_like(valid_point),))
        assert np.isfinite(valid_score) and np.isfinite(directional)


@pytest.mark.unit
@pytest.mark.parametrize("model_type", _MODEL_TYPES)
def test_uninitialized_rho_requires_adapt_inputs_and_accepts_proposals(model_type):
    image, noise, reference, mask, psf = _fixture()
    rho = ParamU("adaptive_reg_rho", prior_settings=[0.0, 2.0], limits=[0.0, 2.0])
    assert rho.dynamic and rho.value is None
    bbox = (-0.49, 0.49, -0.49, 0.49)
    shift = (0.055, -0.065)

    def build(**kwargs):
        return _model(model_type, image, noise, psf, rho, shift, mask=mask, **kwargs)

    with pytest.raises(ValueError, match="fixed_source_bbox"):
        build(adapt_image=reference)
    with pytest.raises(ValueError, match="adapt_image"):
        build(fixed_source_bbox=bbox)
    model = build(adapt_image=reference, fixed_source_bbox=bbox)
    assert [param.name for param in model.get_dynamic_params()] == ["adaptive_reg_rho"]
    for strength in (0.0, 1.4):
        result, diagnostics = model.evaluate_with_diagnostics(jnp.asarray([strength]))
        assert diagnostics.valid and diagnostics.evidence_valid
        assert np.isfinite(result.log_evidence)
        if strength == 0.0:
            np.testing.assert_array_equal(diagnostics.regularization_scale, np.ones((5, 5)))
        else:
            assert np.ptp(diagnostics.regularization_scale) > 0.0


@pytest.mark.integration
@pytest.mark.parametrize("model_type", _MODEL_TYPES)
def test_adapt_image_uses_mixed_fit_precision(model_type):
    with jax.experimental.enable_x64():
        image, _, reference, mask, psf = _fixture()
        image = image.astype(np.float32)
        # Power-of-two noise keeps whitening exact in both precisions.
        noise = np.full_like(image, 0.5)
        bbox = tuple(np.float64(v) for v in (-0.49, 0.49, -0.49, 0.49))
        shift, rho = (0.055, -0.065), 1.4
        brightness, _, design = _reference_projection(
            image, mask, reference, psf, 0.2, 2, bbox, 5, shift, "cubic_hermite",
        )
        scale = np.exp(rho * (1 - brightness))
        sigma = noise[~mask].astype(np.float64)
        whitened = design / sigma[:, None]
        expected = np.linalg.solve(
            whitened.T @ whitened + 0.7 * np.diag(scale),
            whitened.T @ (image[~mask] / sigma),
        )
        model = _model(
            model_type, image, noise, psf, rho, shift, mask=mask, nsub=2,
            fixed_source_bbox=bbox, adapt_image=reference, interpolation="cubic_hermite",
        )
        result, diagnostics = model.evaluate_with_diagnostics()

        assert diagnostics.valid and diagnostics.evidence_valid
        assert result.source_pixels.dtype == diagnostics.regularization_scale.dtype == np.float64
        np.testing.assert_allclose(diagnostics.regularization_scale.ravel(), scale, rtol=1e-9)
        tolerance = 2e-10 if model_type is PixelizedImageProbModel else 2e-7
        np.testing.assert_allclose(result.source_pixels, expected, rtol=tolerance, atol=tolerance)
