"""Independent block determinants and fixed-strength reconstruction checks."""

import json

import jax
import numpy as np
import pytest
from scipy.signal import convolve2d

from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer, SimulatorConfig
from TinyLensGpu.ForwardSimulation.LensImage.pixelized import PixelizedLensSimulator
from TinyLensGpu.ObservationModel import (
    IterativePotentialCorrection,
    PixelizedImageProbModel,
)
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import (
    PixelizedImageProbModelOperator,
)
from TinyLensGpu.PhysicalModel import PhysicalModel, PixelizedSourceModel
from TinyLensGpu.utils.inversion.regularization import FiniteDifferenceRegularization
from tests.test_potential_correction_iterative import _coupled_problem
from tests.test_potential_correction_operator import _problem


def _logdet(matrix):
    sign, value = np.linalg.slogdet(matrix)
    assert sign == 1
    return value


def _block_logdet(matrix, blocks, *, jitter_floor=None):
    values = []
    for indices in blocks:
        block = matrix[np.ix_(indices, indices)]
        if jitter_floor is not None:
            mean = np.mean(np.abs(np.diag(block)))
            block = block + 1e-6 * max(mean, jitter_floor) * np.eye(len(indices))
        values.append(_logdet(block))
    return sum(values)


def _quality(values, prediction, data, noise, design, prior):
    residual = prediction - data
    loss = 0.5 * (np.sum((residual / noise) ** 2) + values @ prior @ values)
    gradient = design.T @ (residual / noise**2) + prior @ values
    scale = np.linalg.norm(design.T @ (data / noise**2))
    return loss, np.linalg.norm(gradient) / scale


def _evidence(loss, noise, prior_logdet, posterior_logdet):
    normalizer = 2 * np.log(noise).sum() + len(noise) * np.log(2 * np.pi)
    return -loss - 0.5 * (posterior_logdet - prior_logdet + normalizer)


@pytest.mark.integration
@pytest.mark.parametrize("interpolation", ["bilinear", "cubic_hermite"])
def test_ordinary_evidence_uses_native_data_and_prior_tile_determinants(
    interpolation, record_property
):
    with jax.experimental.enable_x64():
        n, dpix, nsub, strength = 4, 0.125, 2, 0.7
        bbox = tuple(np.float64(v) for v in (-0.375, 0.375, -0.375, 0.375))
        y, x = np.indices((9, 9))
        mask = (x == 0) | (x == 8) | (y == 0) | (y == 8) | ((x == 4) & (y == 3))
        psf = np.array([[0., 0.04, 0.], [0.09, 0.65, 0.12], [0.06, 0.04, 0.]])
        image = 0.25 + 0.12 * np.cos((x + 2 * y) / 3)
        noise = 0.04 + 0.006 * x + 0.003 * y
        source = PixelizedSourceModel(
            n, regularization_type="first-order", interpolation=interpolation,
        )
        source.log_lambda_reg.to_static(np.log(strength))
        physical = PhysicalModel(lens_mass=[], source_light=[source], lens_light=[])
        simulator = PixelizedLensSimulator(
            physical, SimulatorConfig(dpix, 9, psf, nsub=nsub, mask=mask)
        )
        design, _ = simulator.design_matrix(source_bbox=bbox)
        design = np.asarray(design)
        native = np.asarray(simulator.build_mapping_matrix(source_bbox=bbox))

        # Forward differences include a final edge to zero on each upper bound.
        spacing = 0.25
        nodes = np.arange(n * n).reshape(n, n)
        basis = np.eye(n * n)[nodes]
        dx = np.diff(basis, axis=1, append=np.zeros_like(basis[:, :1, :]))
        dy = np.diff(basis, axis=0, append=np.zeros_like(basis[:1, :, :]))
        difference = np.concatenate((dx.reshape(n * n, -1), dy.reshape(n * n, -1)))
        difference /= spacing
        regularization = difference.T @ difference
        prior = strength * regularization
        data, sigma = image[~mask], noise[~mask]
        posterior = design.T @ (design / sigma[:, None] ** 2) + prior
        pixel_weights = np.where(mask, 0, noise**-2)
        effective_weights = convolve2d(pixel_weights, psf[::-1, ::-1] ** 2, mode="same")
        native_posterior = native.T @ (effective_weights[~mask, None] * native) + prior

        args = image, noise, psf, dpix, physical
        options = dict(mask=mask, nsub=nsub, fixed_source_bbox=bbox)
        dense, diagnostics = PixelizedImageProbModel(
            *args, **options, solver_type="cholesky"
        ).evaluate_with_diagnostics()
        assert diagnostics.valid, diagnostics.invalid_reason
        dense_pixels = np.asarray(dense.source_pixels)
        dense_image = np.asarray(dense.model_image)[~mask]
        dense_loss, dense_residual = _quality(
            dense_pixels, dense_image, data, sigma, design, prior
        )
        exact = _evidence(dense_loss, sigma, _logdet(prior), _logdet(posterior))
        np.testing.assert_allclose(dense.log_evidence, exact, atol=1e-10, rtol=0)
        regularizer = FiniteDifferenceRegularization(n, "first-order")

        rows = []
        for width in (2, 4):
            blocks = [
                nodes[y : y + width, x : x + width].ravel()
                for y in range(0, n, width)
                for x in range(0, n, width)
            ]
            reference_prior = n**2 * np.log(strength) + _block_logdet(
                regularization, blocks, jitter_floor=1.0
            )
            reference_posterior = _block_logdet(
                native_posterior, blocks, jitter_floor=1e-8
            )
            model = PixelizedImageProbModelOperator(
                *args, **options, block_size=width, solver_type="pcg", evidence_logdet="block"
            )
            model.pcg_rtol = 1e-12
            fit = model.evaluate()
            assert fit.solver_info.converged and not fit.solver_info.failed
            chols, _ = model.sim_obj.build_block_diag_preconditioner(
                sigma, *bbox, strength, regularizer, block_size=width
            )
            prior_det = n**2 * np.log(strength) + float(
                regularizer.logdet_free(*bbox, block_size=width)
            )
            posterior_det = float(model.sim_obj.logdet_block_diag(chols))
            np.testing.assert_allclose(prior_det, reference_prior, atol=1e-10, rtol=0)
            np.testing.assert_allclose(
                posterior_det, reference_posterior, atol=1e-10, rtol=0
            )

            pixels = np.asarray(fit.source_pixels)
            prediction = np.asarray(fit.model_image)[~mask]
            loss, residual = _quality(pixels, prediction, data, sigma, design, prior)
            expected = _evidence(loss, sigma, reference_prior, reference_posterior)
            np.testing.assert_allclose(fit.log_evidence, expected, atol=1e-10, rtol=0)
            np.testing.assert_allclose(loss, dense_loss, atol=1e-10, rtol=0)
            np.testing.assert_allclose(pixels, dense_pixels, atol=1e-10, rtol=0)
            np.testing.assert_allclose(prediction, dense_image, atol=1e-10, rtol=0)
            assert max(dense_residual, residual) < 2e-12

            rows.append(dict(
                model="ordinary_source_only", interpolation=interpolation,
                partition=f"{width}x{width}_source_tiles",
                prior_logdet_error=prior_det - reference_prior,
                posterior_logdet_error=posterior_det - reference_posterior,
                dense_log_evidence=float(dense.log_evidence),
                operator_log_evidence=float(fit.log_evidence),
                dense_objective=dense_loss, operator_objective=loss,
                max_source_difference=np.max(np.abs(pixels - dense_pixels)),
                max_image_difference=np.max(np.abs(prediction - dense_image)),
                dense_relative_normal_residual=dense_residual,
                operator_relative_normal_residual=residual,
                dense_status="valid", operator_status="converged",
                operator_iterations=int(fit.solver_info.n_iter),
            ))
    record_property("comparisons", json.dumps(rows, allow_nan=False))


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", ["single_step", "iterative"])
@pytest.mark.parametrize("interpolation", ["bilinear", "cubic_hermite"])
def test_correction_evidence_uses_exact_curvature_principal_blocks(
    interpolation, algorithm, record_property
):
    with jax.experimental.enable_x64():
        if algorithm == "single_step":
            problem = _problem(source_interpolation=interpolation)
            ls, lp = 1.3, 0.3
            design = np.asarray(problem.dense_mapping())

            def jacobian(_):
                return design
        else:
            # Use one observed image for both modes, even though the simulated
            # control image was generated with the default bilinear source.
            control = _coupled_problem()
            base = control.renderer
            renderer = PotentialCorrectionRenderer(
                base.macro_model, base.sim_config, base.source_n, base.source_bbox,
                potential_grid=base.potential_grid, source_interpolation=interpolation,
            )
            problem = IterativePotentialCorrection(
                renderer, control.image_data, control.noise_map,
                control.initial_values[: control.n_source],
                lambda_amplitude=control.lambda_amplitude,
                source_regularization="zero-order",
            )
            ls, lp = 0.3, 0.02
            jacobian = jax.jit(jax.jacfwd(problem.forward))

        prior = np.asarray(problem.dense_prior(ls, lp))
        data, noise = np.asarray(problem.data_active), np.asarray(problem.noise_active)
        dense = problem.fit(ls, lp)
        rows = []
        width = 8
        indices = np.arange(problem.n_joint)
        partitions = {
            "contiguous_8": indices,
            "even_then_odd_8": np.concatenate((indices[::2], indices[1::2])),
        }
        for partition, order in partitions.items():
            blocks = tuple(
                tuple(order[i : i + width]) for i in range(0, order.size, width)
            )
            fit = problem.fit(
                ls, lp, backend="operator", block_size=width,
                block_indices=blocks, pcg_rtol=1e-12,
            )
            fit_metrics = {}
            for result, layout in ((dense, (indices,)), (fit, blocks)):
                assert result.success, result.termination_reason
                values = np.r_[result.source_values, result.potential_values]
                image = np.asarray(problem.forward(values))
                jac = np.asarray(jacobian(values))
                posterior = jac.T @ (jac / noise[:, None] ** 2) + prior
                loss, residual = _quality(values, image, data, noise, jac, prior)
                p0, p1 = _block_logdet(prior, layout), _block_logdet(posterior, layout)
                expected = _evidence(loss, noise, p0, p1)
                np.testing.assert_allclose(result.objective, loss, atol=1e-12, rtol=0)
                np.testing.assert_allclose(result.prior_logdet, p0, atol=1e-10, rtol=0)
                np.testing.assert_allclose(
                    result.posterior_logdet, p1, atol=1e-10, rtol=0
                )
                np.testing.assert_allclose(result.log_evidence, expected, atol=1e-10, rtol=0)
                assert result.solver_info.converged and not result.solver_info.failed
                if algorithm == "single_step":
                    inner_residual = result.solver_info.relative_residual
                    np.testing.assert_allclose(inner_residual, residual, atol=2e-14, rtol=0)
                else:
                    inner_residual = max(
                        row["solver_info"]["relative_residual"]
                        for row in result.history
                        if row["accepted"]
                    )
                    assert residual < 2e-6
                assert inner_residual < 2e-12
                fit_metrics.update({
                    f"{result.backend}_relative_normal_residual": residual,
                    f"{result.backend}_max_inner_relative_residual": inner_residual,
                    f"{result.backend}_status": result.termination_reason,
                    f"{result.backend}_objective": float(result.objective),
                    f"{result.backend}_log_evidence": float(result.log_evidence),
                })
                if result is fit:
                    fit_metrics.update(
                        prior_logdet_error=float(result.prior_logdet) - p0,
                        posterior_logdet_error=float(result.posterior_logdet) - p1,
                    )

            np.testing.assert_allclose(fit.objective, dense.objective, atol=1e-10, rtol=0)
            differences = {}
            for key in ("source_values", "potential_values", "nonlinear_image_active"):
                operator_values = np.asarray(getattr(fit, key))
                dense_values = np.asarray(getattr(dense, key))
                np.testing.assert_allclose(operator_values, dense_values, atol=2e-8, rtol=0)
                differences[f"max_{key}_difference"] = np.max(
                    np.abs(operator_values - dense_values)
                )
            assert fit.configuration["block_indices"] == blocks

            rows.append(dict(
                model=algorithm, interpolation=interpolation, partition=partition,
                n_source=problem.n_source, n_potential=problem.n_potential,
                lambda_source=ls, lambda_potential=lp,
                lambda_amplitude=problem.lambda_amplitude,
                **fit_metrics,
                **differences,
            ))
    record_property("comparisons", json.dumps(rows, allow_nan=False))
