"""Tests for the operator-based (matrix-free) pixelized source backend."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
from TinyLensGpu.ForwardSimulation.LensImage.pixelized import PixelizedLensSimulator
from TinyLensGpu.ForwardSimulation.LensImage.pixelized_operator import (
    JointCurvatureData,
    PixelizedLensOperator,
    SourceCurvatureData,
)
from TinyLensGpu.Inference import ParamU
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.ObservationModel.LensImage.inversion_result import (
    InversionResult,
)
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model import (
    PixelizedImageProbModel,
)
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import (
    PixelizedImageProbModelOperator,
)
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import EPL, SIE
from TinyLensGpu.PhysicalModel.LensImage.Pixelized.Light.pixelized_source import (
    PixelizedSourceModel,
)
from TinyLensGpu.PhysicalModel.LensImage.composite import PhysicalModel
from TinyLensGpu.utils.cg_solver import (
    pcg_solve,
    apply_preconditioner,
    preconditioner_diagonal,
    BlockSchurPreconditioner,
)
from TinyLensGpu.utils.curvature_operator import CurvatureOperator
from TinyLensGpu.utils.pnpg_solver import pnpg_nnls_solve
from TinyLensGpu.utils.inversion.regularization import (
    FiniteDifferenceRegularization,
    source_template_scale_map,
)
import caskade as ck

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def _delta_psf():
    return jnp.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])


def _blur_psf():
    return jnp.asarray(
        [[0.05, 0.10, 0.05], [0.10, 0.40, 0.10], [0.05, 0.10, 0.05]]
    )


def _static_sie():
    sie = SIE(theta_E=0.12, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    for p in [sie.theta_E, sie.e1, sie.e2, sie.center_x, sie.center_y]:
        p.to_static()
    return sie


def _pix_src(log_lambda_val=0.0, adaptive_reg_rho=0.0):
    lam = ParamU("log_lambda_reg", log_lambda_val,
                 prior_type="uniform",
                 prior_settings=[jnp.log(1e-3), jnp.log(1e3)])
    lam.to_dynamic()
    return PixelizedSourceModel(n=5,
        log_lambda_reg=lam,
        regularization_type="first-order",
        adaptive_reg_rho=adaptive_reg_rho,
    )


def _pix_src_dynamic_adaptive():
    lam = ParamU("log_lambda_reg", 0.0,
                 prior_type="uniform",
                 prior_settings=[jnp.log(1e-3), jnp.log(1e3)],
                 limits=[jnp.log(1e-3), jnp.log(1e3)])
    rho = ParamU("adaptive_reg_rho", 1.0,
                 prior_type="uniform",
                 prior_settings=[0.0, 3.0],
                 limits=[0.0, 3.0])
    for p in (lam, rho):
        p.to_dynamic()
    return PixelizedSourceModel(n=5,
        log_lambda_reg=lam,
        regularization_type="first-order",
        adaptive_reg_rho=rho,
    )


def _fixed_bbox():
    return (-0.3, 0.3, -0.3, 0.3)


def _fixed_scale(n=5, rho=1.0):
    s0 = jnp.abs(jnp.linspace(-1.0, 1.0, n * n))
    return source_template_scale_map(s0, n, rho=rho)


def _fixed_template(n=5):
    return jnp.abs(jnp.linspace(-1.0, 1.0, n * n))


def _phys_model(source=None):
    return PhysicalModel(
        lens_mass=[_static_sie()],
        source_light=[source or _pix_src()],
        lens_light=[],
    )


def _sim_config(psf=None, mask=None, nsub=1):
    return SimulatorConfig(
        dpix=0.08, npix=10, nsub=nsub,
        psf_kernel=psf if psf is not None else _delta_psf(),
        mask=mask,
    )


def _make_test_data(psf=None, mask=None, nsub=1):
    """Create mock data using the matrix backend for ground-truth comparison."""
    phys = _phys_model()
    config = _sim_config(psf=psf, mask=mask, nsub=nsub)
    sim = PixelizedLensSimulator(phys, config)
    true_src = jnp.abs(jnp.linspace(-1.0, 1.0, 25))
    mock_image = sim.simulate(true_src, psf_kernel=psf if psf is not None else _delta_psf())
    noise = jnp.ones((10, 10)) * 0.05
    return mock_image, noise, phys, config


def _make_joint_test_data():
    """Create a positive source + lens-light problem for constrained solvers."""
    from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import SersicEllipse

    lens_light = SersicEllipse(
        R_sersic=0.35,
        n_sersic=2.0,
        Ie=1.0,
        e1=0.05,
        e2=-0.02,
        center_x=0.0,
        center_y=0.0,
    )
    for parameter in (
        lens_light.R_sersic,
        lens_light.n_sersic,
        lens_light.Ie,
        lens_light.e1,
        lens_light.e2,
        lens_light.center_x,
        lens_light.center_y,
    ):
        parameter.to_static()
    phys = PhysicalModel(
        lens_mass=[_static_sie()],
        source_light=[_pix_src()],
        lens_light=[lens_light],
    )
    config = _sim_config()
    simulator = PixelizedLensSimulator(phys, config)
    true_source = jnp.linspace(0.05, 1.0, 25)
    image = simulator.simulate(
        true_source,
        lens_light_amplitudes=jnp.asarray([0.7]),
    )
    noise = jnp.ones_like(image) * 0.05
    return image, noise, phys


def _assert_square_bbox(source_bbox):
    xmin, xmax, ymin, ymax = source_bbox
    assert jnp.allclose(xmax - xmin, ymax - ymin)


@pytest.mark.unit
@pytest.mark.parametrize("model_cls,solver", [
    (PixelizedImageProbModel, "cholesky"),
    (PixelizedImageProbModel, "nnls"),
    (PixelizedImageProbModelOperator, "pcg"),
    (PixelizedImageProbModelOperator, "pnpg"),
])
@pytest.mark.parametrize("joint", [False, True])
def test_evaluate_shares_one_solve_and_preserves_legacy_outputs(monkeypatch, model_cls, solver, joint):
    if joint:
        image, noise, phys = _make_joint_test_data()
    else:
        image, noise, phys, _ = _make_test_data()
    model = model_cls(
        image, noise, _delta_psf(), 0.08, phys, solver_type=solver,
        fixed_source_bbox=_fixed_bbox(),
        position_likelihood=dict(positions=[[0.2, 0.0], [-0.2, 0.0]],
                                 threshold_arcsec=0.01, min_log_like=-20.0),
    )
    solve = model._solve_source
    calls = []

    def counted_solve(*args, **kwargs):
        calls.append(None)
        return solve(*args, **kwargs)

    monkeypatch.setattr(model, "_solve_source", counted_solve)
    theta = jnp.asarray([0.0])
    result = model.evaluate(theta)
    assert isinstance(result, InversionResult)
    assert len(calls) == 1
    if model_cls is PixelizedImageProbModelOperator:
        assert result.solver_info.converged
    else:
        assert result.solver_info is None
    np.testing.assert_allclose(result.log_evidence, model(theta),
                               rtol=1e-5, atol=1e-4)
    image_old, source_old = model.forward_model(theta, return_source=True)
    np.testing.assert_allclose(result.model_image, image_old, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(result.source_pixels, source_old, rtol=1e-5, atol=1e-5)
    assert (result.lens_amplitudes is not None) == joint
    assert tuple(result.source_bbox) == tuple(model._fixed_source_bbox)
    # Source-only models return the image alone even with return_components.
    if not joint:
        assert model.forward_model(theta, return_components=True).shape == (10, 10)
    if joint:
        _, _, lens_old = model.forward_model(theta, return_components=True)
        np.testing.assert_allclose(result.lens_amplitudes, lens_old, rtol=1e-5, atol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("n,block_size,reg_type", [
    (5, 3, "zero-order"), (8, 4, "first-order"), (5, 3, "second-order"),
])
def test_fixed_cache_bypasses_preparation_and_matches_dynamic_path(
    monkeypatch, n, block_size, reg_type,
):
    psf = jnp.asarray([[0.0, 0.15, 0.05], [0.0, 0.5, 0.2], [0.0, 0.1, 0.0]])
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, :] = True
    image, noise, _, _ = _make_test_data(psf=psf, mask=mask, nsub=4)
    source = PixelizedSourceModel(
        n=n, log_lambda_reg=_pix_src().log_lambda_reg,
        regularization_type=reg_type, adaptive_reg_rho=1.0,
    )
    phys = _phys_model(source)
    kwargs = dict(mask=mask, nsub=4, block_size=block_size,
                  fixed_source_bbox=_fixed_bbox(), fixed_reg_template=_fixed_template(n))
    cached = PixelizedImageProbModelOperator(image, noise, psf, 0.08, phys, **kwargs)
    reference = PixelizedImageProbModelOperator(image, noise, psf, 0.08, phys, **kwargs)
    reference._prepared = reference._prepared._replace(regularization=None, mapping=None)
    # Tighten the solve so early stopping does not dominate the cache comparison.
    cached.pcg_rtol = reference.pcg_rtol = 1e-7
    proposals = jnp.asarray([[0.0], [1.0]])
    expected = jax.jit(jax.vmap(reference.evaluate))(proposals)

    def unexpected(*args, **kwargs):
        raise AssertionError("fixed preparation was repeated")

    for name in ("precompute_operator_data", "build_rhs", "build_data_blocks"):
        monkeypatch.setattr(cached.sim_obj, name, unexpected)
    monkeypatch.setattr(cached.reg_builder, "block_matrices", unexpected)
    monkeypatch.setattr(cached, "_regularization_logdet", unexpected)
    actual = jax.jit(jax.vmap(cached.evaluate))(proposals)
    assert jnp.all(actual.solver_info.converged)
    for field in ("log_evidence", "model_image", "source_pixels"):
        np.testing.assert_allclose(getattr(actual, field), getattr(expected, field),
                                   rtol=3e-5, atol=3e-5)
    assert not jnp.allclose(actual.source_pixels[0], actual.source_pixels[1])


@pytest.mark.unit
@pytest.mark.parametrize("vary", ["mass_pointer", "rho", "fixed_rho"])
def test_cache_tracks_transitive_and_actual_regularization_dependencies(vary):
    source = _pix_src_dynamic_adaptive() if vary == "rho" else _pix_src(adaptive_reg_rho=1.0)
    mass = _static_sie()
    if vary == "mass_pointer":
        driver = ParamU("driver", 0.12)
        driver.to_dynamic()
        mass.theta_E = ck.Param("theta_E", lambda _: driver.value, link=(driver,))
    phys = PhysicalModel(lens_mass=[mass], source_light=[source])
    image, noise, _, _ = _make_test_data()
    kwargs = dict(fixed_source_bbox=_fixed_bbox(), fixed_reg_template=_fixed_template())
    model = PixelizedImageProbModelOperator(image, noise, _blur_psf(), 0.08, phys, **kwargs)
    assert (model._prepared.mapping is None) == (vary == "mass_pointer")
    assert (model._prepared.regularization is None) == (vary == "rho")
    changing, values = {
        "mass_pointer": ("driver", [0.1, 0.15]),
        "rho": ("adaptive_reg_rho", [0.5, 2.0]),
        "fixed_rho": ("log_lambda_reg", [0.0, 1.0]),
    }[vary]
    proposals = jnp.asarray([
        [value if p.name == changing else 0.0 for p in model.dynamic_params]
        for value in values
    ])
    result = jax.jit(jax.vmap(model.evaluate))(proposals)
    assert jnp.all(result.solver_info.converged)
    assert not jnp.allclose(result.source_pixels[0], result.source_pixels[1]), changing


@pytest.mark.unit
def test_reconstruction_skips_evidence_determinants(monkeypatch):
    image, noise, _, _ = _make_test_data()
    model = PixelizedImageProbModelOperator(
        image, noise, _delta_psf(), 0.08, _phys_model(_pix_src_dynamic_adaptive()),
        fixed_source_bbox=_fixed_bbox(), fixed_reg_template=_fixed_template(),
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("evidence determinant requested")

    monkeypatch.setattr(model, "_regularization_logdet", unexpected)
    image, source = model.forward_model(jnp.asarray([0.0, 1.0]), return_source=True)
    assert jnp.all(jnp.isfinite(image)) and jnp.any(source != 0)
    with pytest.raises(AssertionError, match="evidence determinant"):
        model.evaluate(jnp.asarray([0.0, 1.0]))


@pytest.mark.unit
@pytest.mark.parametrize("solver", ["pcg", "pnpg"])
def test_evaluate_failure_returns_status_penalty_and_zero_arrays(solver):
    image, noise, phys, _ = _make_test_data()
    model = PixelizedImageProbModelOperator(
        image, noise, _delta_psf(), 0.08, phys, solver_type=solver,
        fixed_source_bbox=_fixed_bbox(),
    )
    setattr(model, solver + "_max_iter", 0)
    setattr(model, solver + "_rtol", 0.0)
    result = model.evaluate(jnp.asarray([0.0]))
    assert not result.solver_info.converged
    assert result.log_evidence == -1e10
    assert jnp.all(result.model_image == 0) and jnp.all(result.source_pixels == 0)


@pytest.mark.unit
def test_operator_infers_square_source_bbox_for_asymmetric_betas():
    """Test operator bbox helper applies default padding and makes a square."""
    config = _sim_config()
    config.source_bbox_outlier_frac = 0.0
    operator = PixelizedLensOperator(_phys_model(), config)
    beta_x = jnp.asarray([0.0, 3.0])
    beta_y = jnp.asarray([-0.2, 0.2])

    source_bbox = operator._infer_and_fix_bbox(beta_x, beta_y)

    _assert_square_bbox(source_bbox)
    xmin, xmax, ymin, ymax = source_bbox
    assert jnp.allclose(xmin, -0.15)
    assert jnp.allclose(xmax, 3.15)
    assert jnp.allclose(ymin, -1.65)
    assert jnp.allclose(ymax, 1.65)


# ------------------------------------------------------------------
# PCG solver tests
# ------------------------------------------------------------------

def _dense_curvature_kernel(coefficients, matrix, _spec):
    return matrix @ coefficients


def _dense_curvature_operator(matrix):
    return CurvatureOperator(
        data=matrix,
        kernel=_dense_curvature_kernel,
        spec=None,
        size=int(matrix.shape[0]),
    )


@pytest.mark.unit
def test_pcg_solves_small_spd_system():
    """PCG should solve a well-conditioned SPD system to high accuracy."""
    n = 20
    import jax.random as jrandom
    key = jrandom.PRNGKey(42)
    B = jrandom.normal(key, (n, n))
    A_dense = B.T @ B + 0.1 * jnp.eye(n)  # SPD
    b = jnp.linspace(-1.0, 1.0, n)

    operator = _dense_curvature_operator(A_dense)
    P_chol = jnp.eye(n)

    x, info = pcg_solve(operator, b, P_chol, max_iter=100, rtol=1e-10)

    expected = jnp.linalg.solve(A_dense, b)
    assert jnp.allclose(x, expected, atol=1e-4)
    assert info.converged


@pytest.mark.unit
def test_pcg_uses_preconditioner():
    """PCG with a good preconditioner converges in fewer iterations."""
    n = 30
    diag = jnp.logspace(-2, 2, n)
    A_dense = jnp.diag(diag)
    b = jnp.ones(n)
    operator = _dense_curvature_operator(A_dense)

    # No preconditioner (identity)
    P_chol_id = jnp.eye(n)
    _, info_id = pcg_solve(operator, b, P_chol_id, max_iter=200, rtol=1e-8)

    # Good preconditioner (Cholesky of the exact A)
    P_chol_good = jnp.diag(jnp.sqrt(diag))
    _, info_good = pcg_solve(operator, b, P_chol_good, max_iter=200, rtol=1e-8)

    # Good preconditioner should converge faster than identity
    assert info_good.n_iter < info_id.n_iter
    assert info_good.converged


@pytest.mark.unit
def test_pcg_detects_non_spd_failure():
    """PCG should set failed=True and converged=False for a non-SPD system."""
    n = 4
    import jax.random as jrandom

    key = jrandom.PRNGKey(7)
    B = jrandom.normal(key, (n, n))
    # Negative-definite matrix → pAp < 0 on the first iteration.
    A_dense = -(B.T @ B + 0.1 * jnp.eye(n))
    b = jnp.ones(n)

    operator = _dense_curvature_operator(A_dense)
    P_chol = jnp.eye(n)

    x, info = pcg_solve(operator, b, P_chol, max_iter=50, rtol=1e-10)

    assert info.failed, f"Expected failed=True for non-SPD system, got failed={info.failed}"
    assert not info.converged, "Expected converged=False when breakdown occurs"
    assert jnp.all(jnp.isfinite(x)), "Returned x should be finite even on failure"


# ------------------------------------------------------------------
# PNPG NNLS solver tests
# ------------------------------------------------------------------

@pytest.mark.unit
def test_pnpg_nnls_solves_positive_quadratic():
    """PNPG should solve a small non-negative SPD quadratic."""
    A_dense = jnp.diag(jnp.asarray([2.0, 4.0, 8.0], dtype=jnp.float32))
    b = jnp.asarray([2.0, 8.0, 4.0], dtype=jnp.float32)
    operator = _dense_curvature_operator(A_dense)

    x, info = pnpg_nnls_solve(
        operator, b, jnp.linalg.cholesky(A_dense),
        max_iter=300, rtol=1e-5, power_iter=8,
    )

    expected = jnp.asarray([1.0, 2.0, 0.5], dtype=jnp.float32)
    np.testing.assert_allclose(np.array(x), np.array(expected), rtol=1e-3, atol=1e-3)
    assert jnp.all(x >= 0.0)
    assert jnp.isfinite(info.convergence_metric)
    assert info.converged


@pytest.mark.unit
def test_pnpg_nnls_projects_negative_solution_to_zero():
    """PNPG projection should enforce x >= 0 for an all-negative drive."""
    A_dense = jnp.eye(4, dtype=jnp.float32)
    b = -jnp.ones(4, dtype=jnp.float32)
    operator = _dense_curvature_operator(A_dense)

    x, info = pnpg_nnls_solve(
        operator, b, jnp.linalg.cholesky(A_dense),
        max_iter=50, rtol=1e-5, power_iter=4,
    )

    np.testing.assert_allclose(np.array(x), np.zeros(4), atol=1e-6)
    assert info.converged
    assert not info.failed


@pytest.mark.unit
def test_pnpg_nnls_resolves_ill_conditioned_active_set():
    """PNPG should resolve low-curvature entries after equilibration."""
    n = 64
    eigenvalues = jnp.logspace(0, 8, n, dtype=jnp.float32)
    A_dense = jnp.diag(eigenvalues)
    expected = jnp.where(jnp.arange(n) % 2 == 0, 1.0, 0.0)
    # Positive dual multipliers on zero-valued entries make ``expected`` the
    # exact KKT solution of the bound-constrained quadratic.
    b = jnp.where(expected > 0.0, eigenvalues, -jnp.ones_like(eigenvalues))
    operator = _dense_curvature_operator(A_dense)

    x, info = pnpg_nnls_solve(
        operator,
        b,
        jnp.diag(jnp.sqrt(eigenvalues)),
        max_iter=100,
        rtol=1e-3,
    )

    relative_error = jnp.linalg.norm(x - expected) / jnp.linalg.norm(expected)
    assert info.converged
    assert not info.failed
    assert info.convergence_metric <= 1e-3
    assert relative_error < 1e-3
    assert jnp.all(x >= 0.0)


@pytest.mark.unit
def test_pnpg_nnls_solves_coupled_quadratic_with_changing_active_set():
    """PNPG should identify coupled variables that belong on the bound."""
    factor = jnp.asarray(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.8, 1.5, 0.0, 0.0],
            [-0.4, 0.3, 1.2, 0.0],
            [0.2, -0.5, 0.7, 1.0],
        ],
        dtype=jnp.float32,
    )
    A_dense = factor @ factor.T + 0.1 * jnp.eye(4, dtype=jnp.float32)
    expected = jnp.asarray([1.5, 0.0, 0.7, 0.0], dtype=jnp.float32)
    dual = jnp.asarray([0.0, 0.4, 0.0, 0.2], dtype=jnp.float32)
    b = A_dense @ expected - dual
    operator = _dense_curvature_operator(A_dense)

    x, info = pnpg_nnls_solve(
        operator,
        b,
        jnp.linalg.cholesky(A_dense),
        x0=jnp.ones(4, dtype=jnp.float32),
        max_iter=1000,
        rtol=1e-5,
    )

    np.testing.assert_allclose(
        np.asarray(x), np.asarray(expected), rtol=2e-4, atol=2e-4
    )
    assert info.converged
    assert not info.failed


@pytest.mark.unit
def test_pnpg_backtracks_when_power_iteration_underestimates_curvature():
    """The line search must repair an unsafe initial Rayleigh step."""
    size = 8
    index = jnp.arange(size, dtype=jnp.float32)
    power_vector = jnp.sin((index + 1.0) * 1.61803398875)
    power_vector = power_vector / jnp.linalg.norm(power_vector)
    dominant = jnp.eye(size, dtype=jnp.float32)[0]
    dominant = dominant - jnp.dot(dominant, power_vector) * power_vector
    dominant = dominant / jnp.linalg.norm(dominant)
    A_dense = (
        jnp.eye(size, dtype=jnp.float32)
        + 1.0e4 * jnp.outer(dominant, dominant)
    )
    operator = _dense_curvature_operator(A_dense)

    x, info = pnpg_nnls_solve(
        operator,
        jnp.ones(size, dtype=jnp.float32),
        jnp.eye(size, dtype=jnp.float32),
        max_iter=20,
        power_iter=1,
        rtol=1.0,
    )

    assert jnp.all(jnp.isfinite(x))
    assert not info.failed
    assert info.step_size < 1.0e-3


@pytest.mark.unit
def test_pnpg_line_search_failure_cannot_report_convergence():
    """A failed backtracking gate must override a loose KKT tolerance."""
    A_dense = jnp.eye(3, dtype=jnp.float32)
    operator = _dense_curvature_operator(A_dense)

    _, info = pnpg_nnls_solve(
        operator,
        jnp.ones(3, dtype=jnp.float32),
        jnp.eye(3, dtype=jnp.float32),
        max_iter=1,
        rtol=1.0,
        max_backtracking=0,
    )

    assert info.failed
    assert not info.converged


# ------------------------------------------------------------------
# PixelizedLensOperator tests
# ------------------------------------------------------------------

@pytest.mark.unit
def test_operator_forward_model_matches_matrix():
    """Operator forward_model should match the matrix design_matrix product."""
    mock, noise, phys, config = _make_test_data(psf=_delta_psf())

    sim_mat = PixelizedLensSimulator(phys, config)
    sim_op = PixelizedLensOperator(phys, config)

    F, bbox = sim_mat.design_matrix(psf_kernel=_delta_psf())
    # We need consistent bbox — get from operator
    _, _, bx, by = sim_op._get_beta_sub_and_seed()
    xmi, xma, ymi, yma = sim_op._infer_and_fix_bbox(bx, by)

    s = jnp.linspace(-1.0, 1.0, 25)

    model_mat = F @ s
    model_op = sim_op.forward_model(s, xmi, xma, ymi, yma)

    # Results should match closely
    np.testing.assert_allclose(np.array(model_mat), np.array(model_op), atol=1e-4)


@pytest.mark.unit
def test_A_matvec_matches_explicit():
    """A_matvec(s) should match the explicit A @ s computed from F, within float32 precision."""
    mock, noise, phys, config = _make_test_data(psf=_delta_psf())

    sim_op = PixelizedLensOperator(phys, config)

    # Get bbox from operator
    _, _, bx, by = sim_op._get_beta_sub_and_seed()
    xmi, xma, ymi, yma = sim_op._infer_and_fix_bbox(bx, by)

    n_1d = noise[~config.mask].ravel()
    lam = jnp.asarray(1.0)

    builder = FiniteDifferenceRegularization(5, "first-order")
    xmif, xmaf, ymif, ymaf = float(xmi), float(xma), float(ymi), float(yma)
    reg_dense = builder.matrix(xmif, xmaf, ymif, ymaf)
    reg_data = builder.make_reg_data(xmif, xmaf, ymif, ymaf)

    # Build explicit F from operator's design_matrix (uses same bbox)
    F_op, bbox_op = sim_op.design_matrix()

    # Explicit A
    wF = F_op / n_1d[:, None]
    A_explicit = wF.T @ wF + lam * reg_dense

    s = jnp.linspace(-1.0, 1.0, 25)

    # Operator A — uses compact reg_data
    operator = sim_op.build_source_curvature_operator(
        n_1d, xmi, xma, ymi, yma, lam, reg_data,
    )
    op_result = operator.matvec(s)
    explicit_result = A_explicit @ s

    # Float32 operator chain vs explicit dense
    np.testing.assert_allclose(
        np.array(op_result), np.array(explicit_result), rtol=1e-3, atol=1e-3
    )


@pytest.mark.unit
def test_source_and_joint_curvature_operators_match_jit_and_dense():
    """Typed source and joint adapters preserve explicit curvature semantics."""
    image, noise, phys = _make_joint_test_data()
    config = _sim_config()
    simulator = PixelizedLensOperator(phys, config)
    _, _, beta_x, beta_y = simulator._get_beta_sub_and_seed()
    xmin, xmax, ymin, ymax = simulator._infer_and_fix_bbox(beta_x, beta_y)
    builder = FiniteDifferenceRegularization(5, "first-order")
    reg_data = builder.make_reg_data(
        float(xmin), float(xmax), float(ymin), float(ymax),
    )
    noise_1d = noise[~config.mask].ravel()
    image_1d = image[~config.mask].ravel()
    lambda_reg = jnp.asarray(1.0)
    lens_regularization = jnp.asarray(1.0e-6)
    op_data = simulator.precompute_operator_data(xmin, xmax, ymin, ymax)

    source_operator = simulator.build_source_curvature_operator(
        noise_1d, xmin, xmax, ymin, ymax, lambda_reg, reg_data,
        op_data=op_data,
    )
    joint_operator, rhs, lens_matrix = simulator.build_joint_system(
        noise_1d,
        image_1d,
        xmin,
        xmax,
        ymin,
        ymax,
        lambda_reg,
        reg_data,
        lens_regularization,
        op_data=op_data,
    )

    assert isinstance(source_operator, CurvatureOperator)
    assert isinstance(source_operator.data, SourceCurvatureData)
    assert isinstance(joint_operator, CurvatureOperator)
    assert isinstance(joint_operator.data, JointCurvatureData)
    assert isinstance(joint_operator.data.source, SourceCurvatureData)
    assert source_operator.size == 25
    assert joint_operator.size == 25 + lens_matrix.shape[1]
    assert rhs.shape == (joint_operator.size,)

    apply_jit = jax.jit(lambda operator, values: operator.matvec(values))
    source_values = jnp.linspace(-0.5, 0.5, source_operator.size)
    joint_values = jnp.linspace(-0.5, 0.5, joint_operator.size)
    source_result = source_operator.matvec(source_values)
    joint_result = joint_operator.matvec(joint_values)
    np.testing.assert_allclose(
        apply_jit(source_operator, source_values), source_result,
        rtol=1.0e-5, atol=5.0e-5,
    )
    np.testing.assert_allclose(
        apply_jit(joint_operator, joint_values), joint_result,
        rtol=1.0e-5, atol=5.0e-5,
    )

    source_design = jax.vmap(
        lambda basis: simulator.forward_model(
            basis, xmin, xmax, ymin, ymax, op_data=op_data,
        ),
        in_axes=1,
        out_axes=1,
    )(jnp.eye(source_operator.size))
    regularization = builder.matrix(
        float(xmin), float(xmax), float(ymin), float(ymax),
    )
    noise_variance = noise_1d ** 2
    source_block = (
        source_design.T @ (source_design / noise_variance[:, None])
        + lambda_reg * regularization
    )
    cross_block = source_design.T @ (
        lens_matrix / noise_variance[:, None]
    )
    lens_block = lens_matrix.T @ (
        lens_matrix / noise_variance[:, None]
    ) + lens_regularization * jnp.eye(lens_matrix.shape[1])
    explicit_joint = jnp.block([
        [source_block, cross_block],
        [cross_block.T, lens_block],
    ])
    np.testing.assert_allclose(
        joint_result,
        explicit_joint @ joint_values,
        rtol=1.0e-3,
        atol=1.0e-1,
    )


@pytest.mark.unit
def test_operator_design_matrix_matches_matrix():
    """Operator's explicit design_matrix (for testing) matches matrix backend."""
    mock, noise, phys, config = _make_test_data(psf=_delta_psf())

    sim_mat = PixelizedLensSimulator(phys, config)
    sim_op = PixelizedLensOperator(phys, config)

    F_mat, bbox_mat = sim_mat.design_matrix(psf_kernel=_delta_psf())
    F_op, bbox_op = sim_op.design_matrix()

    np.testing.assert_allclose(np.array(F_mat), np.array(F_op), atol=1e-4)
    for a, b in zip(bbox_mat, bbox_op):
        np.testing.assert_allclose(float(a), float(b), atol=1e-5)


@pytest.mark.unit
def test_preconditioner_is_spd():
    """Each block of the block-diagonal preconditioner should be SPD."""
    mock, noise, phys, config = _make_test_data(psf=_delta_psf())

    sim_op = PixelizedLensOperator(phys, config)
    _, _, bx, by = sim_op._get_beta_sub_and_seed()
    xmi, xma, ymi, yma = sim_op._infer_and_fix_bbox(bx, by)

    n_1d = noise[~config.mask].ravel()
    lam = jnp.asarray(1.0)
    builder = FiniteDifferenceRegularization(5, "first-order")

    block_chols, block_masks = sim_op.build_block_diag_preconditioner(
        n_1d, xmi, xma, ymi, yma, lam, builder, block_size=3,
    )

    # Each block should be SPD
    for chol, mask in zip(block_chols, block_masks):
        P_block = chol @ chol.T
        # Check symmetry
        np.testing.assert_allclose(np.array(P_block), np.array(P_block.T), atol=1e-6)
        # Check positive eigenvalues
        eigvals = jnp.linalg.eigvalsh(P_block)
        assert jnp.all(eigvals > 1e-3), f"Block has non-positive eigenvalues: min={jnp.min(eigvals):.2e}"

    # Verify all source pixels are covered exactly once
    all_masked = jnp.concatenate([jnp.asarray(m) for m in block_masks])
    all_sorted = jnp.sort(all_masked)
    np.testing.assert_equal(np.array(all_sorted), np.arange(25))


@pytest.mark.unit
def test_equilibrated_cholesky_repairs_float32_roundoff_indefiniteness():
    """Redundant MGE-like Gram blocks should yield a finite SPD factor."""
    matrix = jnp.asarray(
        [[1.0, 1.0], [1.0, 1.0 - 2.0e-7]], dtype=jnp.float32
    )

    chol, stabilized = PixelizedImageProbModelOperator._equilibrated_cholesky(
        matrix
    )

    assert jnp.all(jnp.isfinite(chol))
    assert jnp.all(jnp.diag(chol) > 0.0)
    assert jnp.min(jnp.linalg.eigvalsh(stabilized)) > 0.0


@pytest.mark.unit
def test_preconditioner_stacks_equal_sized_blocks():
    """Equal-sized block preconditioners should be stacked for vmapped solves."""
    mock, noise, phys, config = _make_test_data(psf=_delta_psf())

    sim_op = PixelizedLensOperator(phys, config)
    _, _, bx, by = sim_op._get_beta_sub_and_seed()
    xmi, xma, ymi, yma = sim_op._infer_and_fix_bbox(bx, by)

    n_1d = noise[~config.mask].ravel()
    lam = jnp.asarray(1.0)
    builder = FiniteDifferenceRegularization(5, "first-order")

    block_chols, block_masks = sim_op.build_block_diag_preconditioner(
        n_1d, xmi, xma, ymi, yma, lam, builder, block_size=5,
    )

    assert isinstance(block_chols, jnp.ndarray)
    assert isinstance(block_masks, jnp.ndarray)
    assert block_chols.shape == (1, 25, 25)
    assert block_masks.shape == (1, 25)


@pytest.mark.unit
@pytest.mark.parametrize("block_size", [3, 5])
def test_operator_forward_model_is_warning_free_with_x64_enabled(block_size):
    """Both ragged and uniform block paths should handle x64 inputs safely."""
    with jax.experimental.enable_x64():
        mock, noise, phys, config = _make_test_data(psf=_delta_psf())
        model = PixelizedImageProbModelOperator(
            mock,
            noise,
            _delta_psf(),
            0.08,
            phys,
            mask=config.mask,
            fixed_source_bbox=_fixed_bbox(),
            solver_type="pnpg",
            block_size=block_size,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            model.forward_model()


# ------------------------------------------------------------------
# PixelizedImageProbModelOperator tests
# ------------------------------------------------------------------

@pytest.mark.unit
def test_operator_prob_model_returns_finite_evidence():
    """Operator evidence model returns a finite scalar."""
    mock, noise, phys, config = _make_test_data(psf=_delta_psf())

    model = PixelizedImageProbModelOperator(
        mock, noise, _delta_psf(), 0.08, phys,
        mask=config.mask,
    )

    log_ev = model()
    assert jnp.shape(log_ev) == ()
    assert jnp.isfinite(log_ev)


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_cls", [PixelizedImageProbModel, PixelizedImageProbModelOperator],
)
def test_position_likelihood_sanitizes_non_finite_epl_deflection(model_cls):
    """A non-finite position penalty must reject, not poison, a proposal."""
    # A negative Einstein radius with a non-integer slope is outside the EPL
    # domain and produces a genuinely non-finite deflection.
    mass = EPL(
        theta_E=-1.0,
        gamma=1.9857056,
        e1=0.0,
        e2=0.0,
        center_x=0.0,
        center_y=0.0,
    )
    for parameter in (
        mass.theta_E,
        mass.gamma,
        mass.e1,
        mass.e2,
        mass.center_x,
        mass.center_y,
    ):
        parameter.to_static()
    physical_model = PhysicalModel(
        lens_mass=[mass], source_light=[_pix_src()], lens_light=[],
    )
    model = model_cls(
        jnp.zeros((10, 10)),
        jnp.ones((10, 10)) * 0.1,
        _delta_psf(),
        0.08,
        physical_model,
        position_likelihood={
            "positions": [(0.1, 0.0), (0.0, 0.1)],
            "threshold_arcsec": 0.1,
            "min_log_like": -1.0e10,
        },
    )

    log_evidence = model()

    assert jnp.shape(log_evidence) == ()
    assert jnp.isfinite(log_evidence)
    assert log_evidence == -1.0e10


@pytest.mark.unit
def test_operator_pnpg_returns_nonnegative_source_for_negative_data():
    """PNPG source solve should enforce hard non-negative source pixels."""
    mock, noise, phys, config = _make_test_data(psf=_delta_psf())
    mock = -jnp.abs(mock)

    model = PixelizedImageProbModelOperator(
        mock, noise, _delta_psf(), 0.08, phys,
        mask=config.mask,
        solver_type="pnpg",
    )

    log_ev = model()
    _, source = model.forward_model(return_source=True)

    assert jnp.isfinite(log_ev)
    assert source.shape == (25,)
    assert jnp.all(source >= -1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("solver_type", ["bad", "fista"])
def test_operator_rejects_invalid_solver_type(solver_type):
    """Operator model should reject unsupported source solver names."""
    mock, noise, phys, config = _make_test_data(psf=_delta_psf())

    with pytest.raises(ValueError, match="solver_type"):
        PixelizedImageProbModelOperator(
            mock, noise, _delta_psf(), 0.08, phys,
            mask=config.mask,
            solver_type=solver_type,
        )


@pytest.mark.unit
def test_inversion_result_is_one_shared_type():
    """Both model modules and both package levels re-export one type."""
    import TinyLensGpu.ObservationModel as observation_model
    import TinyLensGpu.ObservationModel.LensImage as lens_image

    from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model import (
        InversionResult as DenseInversionResult,
    )
    from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import (
        InversionResult as OperatorInversionResult,
    )

    assert InversionResult is DenseInversionResult
    assert InversionResult is OperatorInversionResult
    assert InversionResult is lens_image.InversionResult
    assert InversionResult is observation_model.InversionResult


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_cls", [PixelizedImageProbModel, PixelizedImageProbModelOperator],
)
def test_source_bbox_padding_expands_inferred_bbox(model_cls):
    """Both models forward bbox padding into source-plane bbox inference."""
    mock, noise, phys, config = _make_test_data(psf=_delta_psf())

    model_plain = model_cls(
        mock, noise, _delta_psf(), 0.08, phys,
        mask=config.mask,
        source_bbox_padding=0.0,
    )
    model_padded = model_cls(
        mock, noise, _delta_psf(), 0.08, phys,
        mask=config.mask,
        source_bbox_padding=0.25,
    )

    xmin0, xmax0, ymin0, ymax0 = model_plain.infer_source_bbox()
    xmin1, xmax1, ymin1, ymax1 = model_padded.infer_source_bbox()

    assert xmax1 - xmin1 > xmax0 - xmin0
    assert ymax1 - ymin1 > ymax0 - ymin0


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_cls", [PixelizedImageProbModel, PixelizedImageProbModelOperator],
)
@pytest.mark.parametrize(
    "padding,outlier_frac", [(0.0, 0.0), (0.25, 0.0), (0.05, 0.1)],
)
def test_inferred_source_bbox_matches_utility_on_both_backends(
    model_cls, padding, outlier_frac,
):
    """Inferred bboxes match the shared utility identically on both backends."""
    from TinyLensGpu.utils.lensing.mapping import (
        infer_source_bbox as infer_square_source_bbox,
    )

    mock, noise, phys, config = _make_test_data()
    seed_mask = np.zeros((10, 10), dtype=bool)
    seed_mask[4:6, 4:6] = True
    model = model_cls(
        mock, noise, _delta_psf(), 0.08, phys,
        mask=config.mask,
        source_seed_mask=seed_mask,
        source_bbox_padding=padding,
        source_bbox_outlier_frac=outlier_frac,
    )
    beta_x, beta_y = model.sim_obj.ray_trace_seed()
    expected = infer_square_source_bbox(
        beta_x, beta_y, padding=padding, outlier_frac=outlier_frac,
    )
    np.testing.assert_allclose(
        model.infer_source_bbox(),
        [float(np.asarray(v)) for v in expected],
        rtol=1e-6,
        atol=1e-7,
    )
    if outlier_frac > 0.0:
        untrimmed = infer_square_source_bbox(
            beta_x, beta_y, padding=padding, outlier_frac=0.0,
        )
        assert not np.allclose(
            model.infer_source_bbox(),
            [float(np.asarray(v)) for v in untrimmed],
            atol=1e-7,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_cls", [PixelizedImageProbModel, PixelizedImageProbModelOperator],
)
def test_fixed_bbox_overrides_inference_settings(model_cls):
    """A fixed bbox wins over padding/outlier settings everywhere it appears."""
    mock, noise, phys, config = _make_test_data()
    model = model_cls(
        mock, noise, _delta_psf(), 0.08, phys,
        mask=config.mask,
        fixed_source_bbox=_fixed_bbox(),
        source_bbox_padding=0.4,
        source_bbox_outlier_frac=0.2,
    )
    result = model.evaluate(jnp.asarray(model.get_values("flat")))
    np.testing.assert_allclose(
        [float(np.asarray(v)) for v in result.source_bbox],
        _fixed_bbox(),
        atol=1e-7,
    )
    np.testing.assert_allclose(model.infer_source_bbox(), _fixed_bbox(), atol=1e-7)
    _, _, helper_bbox = model.reconstruct_component_images(
        result.source_pixels, result.lens_amplitudes,
    )
    np.testing.assert_allclose(
        [float(np.asarray(v)) for v in helper_bbox], _fixed_bbox(), atol=1e-7,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_cls", [PixelizedImageProbModel, PixelizedImageProbModelOperator],
)
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("nsub", [1, 2])
def test_reconstruct_component_images_matches_model_image(
    monkeypatch, model_cls, joint, nsub,
):
    """Component helper rebuilds the model image without solving or evaluating."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, :] = True
    mask[4, 4] = True
    if joint:
        _, _, phys = _make_joint_test_data()
    else:
        _, _, phys, _ = _make_test_data()
    config = SimulatorConfig(
        dpix=0.08, npix=10, psf_kernel=_blur_psf(), mask=mask, nsub=nsub,
    )
    reference_sim = PixelizedLensSimulator(phys, config)
    image = reference_sim.simulate(
        jnp.linspace(0.05, 1.0, 25), source_bbox=_fixed_bbox(),
        lens_light_amplitudes=jnp.asarray([0.7]) if joint else None,
    )
    noise = jnp.ones_like(image) * 0.05
    model = model_cls(
        image, noise, _blur_psf(), 0.08, phys,
        mask=mask, nsub=nsub, fixed_source_bbox=_fixed_bbox(),
    )
    theta = jnp.asarray(model.get_values("flat"))
    result = model.evaluate(theta)

    def refuse(*args, **kwargs):
        raise AssertionError("reconstruction repeated work")

    monkeypatch.setattr(model, "_solve_source", refuse)
    if model_cls is PixelizedImageProbModelOperator:
        monkeypatch.setattr(model, "_regularization_logdet", refuse)
    else:
        monkeypatch.setattr(model, "_evidence_logdet", refuse)

    source_image, lens_image, bbox = model.reconstruct_component_images(
        result.source_pixels, result.lens_amplitudes,
    )
    assert source_image.shape == (10, 10) and lens_image.shape == (10, 10)
    np.testing.assert_allclose(
        [float(np.asarray(v)) for v in bbox], _fixed_bbox(), atol=1e-7,
    )
    np.testing.assert_allclose(
        np.asarray(source_image + lens_image),
        np.asarray(result.model_image),
        rtol=2e-5,
        atol=2e-5,
    )
    assert np.all(np.asarray(source_image)[mask] == 0.0)
    assert np.all(np.asarray(lens_image)[mask] == 0.0)
    if not joint:
        assert np.all(np.asarray(lens_image) == 0.0)
        # An empty lens amplitude vector is equivalent to None.
        empty_source, empty_lens, _ = model.reconstruct_component_images(
            result.source_pixels, jnp.empty((0,)),
        )
        np.testing.assert_allclose(
            np.asarray(empty_source), np.asarray(source_image),
            rtol=2e-5, atol=2e-5,
        )
        np.testing.assert_allclose(
            np.asarray(empty_lens), np.asarray(lens_image),
            rtol=2e-5, atol=2e-5,
        )

    # Independent check: same columns of an explicitly built design matrix.
    design, _ = reference_sim.design_matrix(source_bbox=_fixed_bbox())
    n_source = reference_sim.n_source_pixels
    source_ref = design[:, :n_source] @ result.source_pixels
    lens_ref = (
        design[:, n_source:] @ result.lens_amplitudes
        if joint else jnp.zeros_like(source_ref)
    )
    np.testing.assert_allclose(
        np.asarray(source_image)[~mask], np.asarray(source_ref),
        rtol=2e-5, atol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(lens_image)[~mask], np.asarray(lens_ref),
        rtol=2e-5, atol=2e-5,
    )
    # The model's own design matrix is only needed on a cache miss: patch it
    # out and confirm the helper output is unchanged.
    monkeypatch.setattr(model.sim_obj, "design_matrix", refuse)
    cached_source, cached_lens, _ = model.reconstruct_component_images(
        result.source_pixels, result.lens_amplitudes,
    )
    np.testing.assert_allclose(
        np.asarray(cached_source), np.asarray(source_image),
        rtol=2e-5, atol=2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(cached_lens), np.asarray(lens_image),
        rtol=2e-5, atol=2e-5,
    )

    if joint:
        with pytest.raises(ValueError, match="lens_light_intensities"):
            model.reconstruct_component_images(result.source_pixels, None)


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_cls", [PixelizedImageProbModel, PixelizedImageProbModelOperator],
)
@pytest.mark.parametrize("fixed", [True, False])
def test_component_images_follow_current_parameter_context(
    model_cls, fixed,
):
    """The component helper uses current dynamic values, not a last result."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, :] = True
    theta_e = ParamU(
        "theta_E", 0.12, prior_type="uniform",
        prior_settings=[0.05, 0.2], limits=[0.0, 1.0],
    )
    theta_e.to_dynamic()
    sie = SIE(theta_E=theta_e, e1=0.1, e2=0.0, center_x=0.0, center_y=0.0)
    for p in (sie.e1, sie.e2, sie.center_x, sie.center_y):
        p.to_static()
    lam = ParamU("log_lambda_reg", 0.0, prior_type="uniform",
                 prior_settings=[jnp.log(1e-3), jnp.log(1e3)])
    lam.to_dynamic()
    source = PixelizedSourceModel(
        n=5, log_lambda_reg=lam, regularization_type="first-order",
    )
    phys = PhysicalModel(lens_mass=[sie], source_light=[source], lens_light=[])
    mock, noise, _, _ = _make_test_data(psf=_blur_psf(), nsub=2)
    kwargs = dict(mask=mask, nsub=2)
    if fixed:
        kwargs["fixed_source_bbox"] = _fixed_bbox()
    model = model_cls(mock, noise, _blur_psf(), 0.08, phys, **kwargs)
    named_theta = jnp.asarray([
        0.14 if p.name == "theta_E" else p.value
        for p in model.get_dynamic_params()
    ])
    with ck.ActiveContext(model):
        model.fill_params(named_theta)
        result = model.evaluate()
        source_image, lens_image, bbox = model.reconstruct_component_images(
            result.source_pixels, result.lens_amplitudes,
        )
        expected_bbox = model.infer_source_bbox() if not fixed else _fixed_bbox()
    np.testing.assert_allclose(
        np.asarray(source_image + lens_image),
        np.asarray(result.model_image),
        rtol=2e-5,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        [float(np.asarray(v)) for v in bbox],
        [float(np.asarray(v)) for v in expected_bbox],
        atol=1e-7,
    )


@pytest.mark.unit
def test_operator_forward_model_returns_correct_shapes():
    """Forward model returns image and source of correct shape."""
    mock, noise, phys, config = _make_test_data(psf=_delta_psf())

    model = PixelizedImageProbModelOperator(
        mock, noise, _delta_psf(), 0.08, phys,
        mask=config.mask,
    )

    model_image, source = model.forward_model(return_source=True)
    assert model_image.shape == (10, 10)
    assert source.shape == (25,)
    assert jnp.all(jnp.isfinite(model_image))
    assert jnp.all(jnp.isfinite(source))


@pytest.mark.unit
def test_operator_pnpg_forward_model_returns_nonnegative_source():
    """PNPG forward_model returns finite image and non-negative source."""
    mock, noise, phys, config = _make_test_data(psf=_delta_psf())

    model = PixelizedImageProbModelOperator(
        mock, noise, _delta_psf(), 0.08, phys,
        mask=config.mask,
        solver_type="pnpg",
    )

    model_image, source = model.forward_model(return_source=True)
    assert model_image.shape == (10, 10)
    assert source.shape == (25,)
    assert jnp.all(jnp.isfinite(model_image))
    assert jnp.all(jnp.isfinite(source))
    assert jnp.all(source >= -1e-6)


@pytest.mark.unit
def test_operator_pnpg_solver_controls_are_configurable():
    """Constructor-level PNPG controls should be stored on the model."""
    mock, noise, phys, config = _make_test_data(psf=_delta_psf())

    model = PixelizedImageProbModelOperator(
        mock,
        noise,
        _delta_psf(),
        0.08,
        phys,
        mask=config.mask,
        solver_type="pnpg",
        pnpg_max_iter=17,
        pnpg_rtol=2.0e-5,
        pnpg_power_iter=5,
    )

    assert model.pnpg_max_iter == 17
    assert model.pnpg_rtol == 2.0e-5
    assert model.pnpg_power_iter == 5


@pytest.mark.unit
def test_operator_pnpg_nonconvergence_penalizes_evidence():
    """PNPG gating should penalize an explicitly non-converged solve.

    ``max_iter=0`` exercises the no-iteration boundary: the solver returns the
    projected PCG initial source, reports non-convergence at zero KKT
    tolerance, and the evidence layer applies the same large penalty used for
    failed iterative solves.
    """
    mock, noise, phys, config = _make_test_data(psf=_delta_psf())

    model = PixelizedImageProbModelOperator(
        mock, noise, _delta_psf(), 0.08, phys,
        mask=config.mask,
        solver_type="pnpg",
    )
    # Boundary case for evidence gating rather than a realistic hard problem.
    model.pnpg_max_iter = 0
    model.pnpg_rtol = 0.0

    log_ev = model()
    assert log_ev < -1.0e9


@pytest.mark.unit
@pytest.mark.boundary
def test_operator_pnpg_joint_nonconvergence_penalizes_evidence():
    """A non-converged PNPG joint solve should return the exact penalty."""
    image, noise, phys = _make_joint_test_data()
    model = PixelizedImageProbModelOperator(
        image,
        noise,
        _delta_psf(),
        0.08,
        phys,
        solver_type="pnpg",
        pnpg_max_iter=0,
        pnpg_rtol=0.0,
    )

    assert model() == -1.0e10


def _dense_objective(design_matrix, data_1d, noise_1d, reg_matrix, lambda_reg, source):
    resid = data_1d - design_matrix @ source
    e_d = 0.5 * jnp.sum((resid / noise_1d) ** 2)
    e_s = 0.5 * jnp.dot(source, reg_matrix @ source)
    return e_d + lambda_reg * e_s


def _joint_objective(
    model_image,
    data,
    noise,
    reg_matrix,
    lambda_reg,
    source,
    lens_intensities,
    lens_light_regularization,
):
    resid = data - model_image
    e_d = 0.5 * jnp.sum((resid / noise) ** 2)
    e_source = 0.5 * lambda_reg * jnp.dot(source, reg_matrix @ source)
    e_lens = 0.5 * lens_light_regularization * jnp.sum(lens_intensities**2)
    return e_d + e_source + e_lens


@pytest.mark.unit
def test_operator_pnpg_matches_dense_nnls_objective_small_grid():
    """Operator PNPG should match dense NNLS on a real lens operator."""
    mock, noise, phys, config = _make_test_data(psf=_delta_psf())
    source_bbox = _fixed_bbox()
    lambda_reg = jnp.exp(jnp.asarray(phys.source_light[0].log_lambda_reg.value))

    prob_dense = PixelizedImageProbModel(
        mock, noise, _delta_psf(), 0.08, phys,
        mask=config.mask,
        solver_type="nnls",
    )
    design_matrix, _ = prob_dense.sim_obj.design_matrix(source_bbox=source_bbox)
    reg_matrix = prob_dense._regularization_matrix(source_bbox)
    src_dense, _, _ = prob_dense._solve_source(design_matrix, reg_matrix, lambda_reg)

    prob_op = PixelizedImageProbModelOperator(
        mock, noise, _delta_psf(), 0.08, phys,
        mask=config.mask,
        fixed_source_bbox=source_bbox,
        solver_type="pnpg",
        pnpg_max_iter=1000,
        pnpg_rtol=1e-4,
    )
    _, src_op = prob_op.forward_model(return_source=True)

    obj_dense = _dense_objective(
        design_matrix, prob_dense.data_1d, prob_dense.noise_1d,
        reg_matrix, lambda_reg, src_dense,
    )
    obj_op = _dense_objective(
        design_matrix, prob_dense.data_1d, prob_dense.noise_1d,
        reg_matrix, lambda_reg, src_op,
    )

    assert jnp.all(src_op >= 0.0)
    np.testing.assert_allclose(float(obj_op), float(obj_dense), rtol=5e-3, atol=5e-3)


@pytest.mark.unit
def test_operator_pnpg_joint_matches_dense_nnls_objective():
    """PNPG joint MAP should match dense NNLS without collapsing components."""
    image, noise, phys = _make_joint_test_data()
    lens_reg = 1.0e-6
    lambda_reg = jnp.exp(jnp.asarray(phys.source_light[0].log_lambda_reg.value))
    dense = PixelizedImageProbModel(
        image,
        noise,
        _delta_psf(),
        0.08,
        phys,
        solver_type="nnls",
        lens_light_regularization=lens_reg,
    )
    operator = PixelizedImageProbModelOperator(
        image,
        noise,
        _delta_psf(),
        0.08,
        phys,
        solver_type="pnpg",
        pnpg_max_iter=1000,
        pnpg_rtol=1.0e-4,
        lens_light_regularization=lens_reg,
    )

    dense_image, dense_source, dense_lens = dense.forward_model(
        return_components=True
    )
    operator_image, operator_source, operator_lens = operator.forward_model(
        return_components=True
    )
    _, source_bbox = dense.sim_obj.design_matrix()
    reg_matrix = FiniteDifferenceRegularization(
        5, "first-order"
    ).matrix(*source_bbox)
    dense_objective = _joint_objective(
        dense_image,
        image,
        noise,
        reg_matrix,
        lambda_reg,
        dense_source,
        dense_lens,
        lens_reg,
    )
    operator_objective = _joint_objective(
        operator_image,
        image,
        noise,
        reg_matrix,
        lambda_reg,
        operator_source,
        operator_lens,
        lens_reg,
    )

    assert jnp.all(operator_source >= 0.0)
    assert jnp.all(operator_lens >= 0.0)
    assert jnp.any(operator_source > 1.0e-6)
    assert jnp.any(operator_lens > 1.0e-6)
    np.testing.assert_allclose(
        float(operator_objective),
        float(dense_objective),
        rtol=5.0e-3,
        atol=5.0e-3,
    )


@pytest.mark.unit
@pytest.mark.parametrize("reg_type", ["zero-order", "first-order", "second-order"])
@pytest.mark.parametrize("dense_solver,op_solver", [("cholesky", "pcg"), ("nnls", "pnpg")])
def test_adaptive_template_map_matches_across_backends(reg_type, dense_solver, op_solver):
    """Dense and operator backends solve the same S0-weighted MAP objective."""
    lam = ParamU("log_lambda_reg", 0.0,
                 prior_type="uniform",
                 prior_settings=[jnp.log(1e-3), jnp.log(1e3)])
    lam.to_dynamic()
    source = PixelizedSourceModel(n=5,
        log_lambda_reg=lam,
        regularization_type=reg_type,
        adaptive_reg_rho=1.0,
    )
    phys = _phys_model(source=source)
    config = _sim_config()
    simulator = PixelizedLensSimulator(phys, config)
    true_src = jnp.abs(jnp.linspace(-1.0, 1.0, 25))
    mock = simulator.simulate(true_src, psf_kernel=_delta_psf())
    noise = jnp.ones((10, 10)) * 0.05

    fixed = dict(
        mask=config.mask,
        fixed_source_bbox=_fixed_bbox(),
        fixed_reg_template=_fixed_template(),
    )
    dense = PixelizedImageProbModel(
        mock, noise, _delta_psf(), 0.08, phys, solver_type=dense_solver, **fixed,
    )
    operator = PixelizedImageProbModelOperator(
        mock, noise, _delta_psf(), 0.08, phys, solver_type=op_solver,
        pnpg_max_iter=1000, pnpg_rtol=1e-4, **fixed,
    )

    _, src_dense = dense.forward_model(return_source=True)
    _, src_op = operator.forward_model(return_source=True)

    design_matrix, _ = dense.sim_obj.design_matrix(source_bbox=_fixed_bbox())
    scale = source_template_scale_map(_fixed_template(), 5, rho=1.0)
    reg_matrix = FiniteDifferenceRegularization(5, reg_type).matrix(
        *_fixed_bbox(), scale=scale,
    )
    lambda_reg = jnp.exp(jnp.asarray(source.log_lambda_reg.value))

    obj_dense = _dense_objective(
        design_matrix, dense.data_1d, dense.noise_1d,
        reg_matrix, lambda_reg, src_dense,
    )
    obj_op = _dense_objective(
        design_matrix, dense.data_1d, dense.noise_1d,
        reg_matrix, lambda_reg, src_op,
    )

    assert jnp.all(jnp.isfinite(src_dense)) and jnp.any(src_dense != 0.0)
    assert jnp.all(jnp.isfinite(src_op)) and jnp.any(src_op != 0.0)
    if op_solver == "pnpg":
        assert jnp.all(src_op >= 0.0)
        assert jnp.all(src_dense >= 0.0)
    np.testing.assert_allclose(
        float(obj_op), float(obj_dense), rtol=5e-3, atol=5e-3,
    )


@pytest.mark.integration
def test_operator_pnpg_joint_forward_model_jits_end_to_end():
    """The constrained joint forward path should compile and return components."""
    image, noise, phys = _make_joint_test_data()
    model = PixelizedImageProbModelOperator(
        image,
        noise,
        _delta_psf(),
        0.08,
        phys,
        solver_type="pnpg",
    )

    model_image, source, lens = jax.jit(
        lambda: model.forward_model(return_components=True)
    )()

    assert jnp.all(jnp.isfinite(model_image))
    assert jnp.any(model_image != 0.0)
    assert jnp.all(source >= 0.0)
    assert jnp.any(source > 1.0e-6)
    assert jnp.all(lens >= 0.0)
    assert jnp.any(lens > 1.0e-6)


@pytest.mark.unit
def test_matrix_vs_operator_source_consistency():
    """Source reconstruction from both backends should be nearly identical."""
    mock, noise, phys, config = _make_test_data(psf=_delta_psf())

    prob_mat = PixelizedImageProbModel(
        mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
    )
    prob_op = PixelizedImageProbModelOperator(
        mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
    )

    _, src_mat = prob_mat.forward_model(return_source=True)
    _, src_op = prob_op.forward_model(return_source=True)

    rms_rel = float(
        jnp.sqrt(jnp.mean((src_mat - src_op) ** 2))
        / jnp.sqrt(jnp.mean(src_mat**2))
    )
    assert rms_rel < 1e-3, f"Source RMS relative error {rms_rel:.2e} exceeds 1e-3"


@pytest.mark.unit
def test_operator_joint_inversion_returns_source_and_lens_light_intensities():
    """Operator backend jointly reconstructs source and lens-light intensities."""
    from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import SersicEllipse

    lens_light = SersicEllipse(
        R_sersic=1.0, n_sersic=4.0, Ie=1.0,
        e1=0.0, e2=0.0, center_x=0.0, center_y=0.0,
    )
    for p in [lens_light.R_sersic, lens_light.n_sersic, lens_light.Ie,
              lens_light.e1, lens_light.e2, lens_light.center_x, lens_light.center_y]:
        p.to_static()

    phys = PhysicalModel(
        lens_mass=[_static_sie()],
        source_light=[_pix_src()],
        lens_light=[lens_light],
    )

    model = PixelizedImageProbModelOperator(
        jnp.ones((10, 10)) * 0.05, jnp.ones((10, 10)) * 0.1,
        _delta_psf(), 0.08, phys,
    )

    model_image, source_pixels, lens_intensities = model.forward_model(
        return_components=True
    )

    assert model_image.shape == (10, 10)
    assert source_pixels.shape == (25,)
    assert lens_intensities.shape == (1,)
    assert jnp.all(jnp.isfinite(model_image))
    assert jnp.all(jnp.isfinite(source_pixels))
    assert jnp.all(jnp.isfinite(lens_intensities))


@pytest.mark.unit
@pytest.mark.boundary
@pytest.mark.parametrize("solver_type", ["pcg", "pnpg"])
def test_operator_joint_inversion_supports_ragged_source_blocks(solver_type):
    """Joint inversion supports source grids not divisible by block size."""
    from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import SersicEllipse

    lens_light = SersicEllipse(
        R_sersic=0.4, n_sersic=2.0, Ie=1.0,
        e1=0.0, e2=0.0, center_x=0.0, center_y=0.0,
    )
    for parameter in (
        lens_light.R_sersic, lens_light.n_sersic, lens_light.Ie,
        lens_light.e1, lens_light.e2,
        lens_light.center_x, lens_light.center_y,
    ):
        parameter.to_static()
    source = PixelizedSourceModel(
        n=12,
        regularization_type="first-order",
        log_lambda_reg=ParamU("log_lambda_reg_ragged", 0.0),
    )
    source.log_lambda_reg.to_static()
    phys = PhysicalModel(
        lens_mass=[_static_sie()],
        source_light=[source],
        lens_light=[lens_light],
    )
    model = PixelizedImageProbModelOperator(
        jnp.ones((10, 10)) * 0.1,
        jnp.ones((10, 10)) * 0.1,
        _delta_psf(),
        0.08,
        phys,
        block_size=10,
        solver_type=solver_type,
    )

    log_evidence = model()
    jitted_log_evidence = jax.jit(lambda: model())()
    model_image, source_pixels, lens_intensities = model.forward_model(
        return_components=True
    )

    assert log_evidence > -1.0e9
    assert jitted_log_evidence > -1.0e9
    assert jnp.any(model_image != 0.0)
    assert jnp.all(jnp.isfinite(source_pixels))
    assert jnp.all(jnp.isfinite(lens_intensities))


@pytest.mark.unit
@pytest.mark.parametrize("nsub", [1, 2])
@pytest.mark.parametrize("rho", [0.0, 1.0])
def test_operator_joint_map_matches_dense_backend(nsub, rho):
    """Joint source/lens-light MAP agrees with the dense reference backend."""
    from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import SersicEllipse

    lens_light = SersicEllipse(
        R_sersic=0.35, n_sersic=2.0, Ie=1.0,
        e1=0.05, e2=-0.02, center_x=0.0, center_y=0.0,
    )
    for parameter in (
        lens_light.R_sersic, lens_light.n_sersic, lens_light.Ie,
        lens_light.e1, lens_light.e2,
        lens_light.center_x, lens_light.center_y,
    ):
        parameter.to_static()
    phys = PhysicalModel(
        lens_mass=[_static_sie()], source_light=[_pix_src(adaptive_reg_rho=rho)],
        lens_light=[lens_light],
    )
    config = _sim_config(nsub=nsub)
    simulator = PixelizedLensSimulator(phys, config)
    true_source = jnp.abs(jnp.linspace(-0.5, 1.0, 25))
    image = simulator.simulate(
        true_source, lens_light_amplitudes=jnp.asarray([0.7]),
    )
    noise = jnp.ones_like(image) * 0.05
    fixed = (
        dict(fixed_source_bbox=_fixed_bbox(), fixed_reg_template=_fixed_template())
        if rho > 0 else {}
    )
    dense = PixelizedImageProbModel(
        image, noise, _delta_psf(), 0.08, phys, nsub=nsub, **fixed,
    )
    operator = PixelizedImageProbModelOperator(
        image, noise, _delta_psf(), 0.08, phys, nsub=nsub, **fixed,
    )

    _, dense_source, dense_lens = dense.forward_model(return_components=True)
    _, operator_source, operator_lens = operator.forward_model(
        return_components=True
    )

    source_rms = jnp.linalg.norm(operator_source - dense_source) / jnp.linalg.norm(
        dense_source
    )
    assert source_rms < 5e-3
    assert jnp.allclose(operator_lens, dense_lens, rtol=5e-3, atol=5e-4)


@pytest.mark.unit
def test_pnpg_joint_inversion_constrains_source_and_lens_light_nonnegative():
    """PNPG projects both joint linear-parameter groups to non-negative values."""
    from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import SersicEllipse

    lens_light = SersicEllipse(
        R_sersic=0.4, n_sersic=2.0, Ie=1.0,
        e1=0.0, e2=0.0, center_x=0.0, center_y=0.0,
    )
    for parameter in (
        lens_light.R_sersic, lens_light.n_sersic, lens_light.Ie,
        lens_light.e1, lens_light.e2,
        lens_light.center_x, lens_light.center_y,
    ):
        parameter.to_static()
    phys = PhysicalModel(
        lens_mass=[_static_sie()], source_light=[_pix_src()],
        lens_light=[lens_light],
    )
    model = PixelizedImageProbModelOperator(
        jnp.ones((10, 10)) * 0.1, jnp.ones((10, 10)) * 0.1,
        _delta_psf(), 0.08, phys, solver_type="pnpg",
        pnpg_max_iter=1000, pnpg_rtol=1e-4,
    )

    _, source, lens = model.forward_model(return_components=True)

    assert jnp.all(source >= 0.0)
    assert jnp.all(lens >= 0.0)


@pytest.mark.unit
@pytest.mark.parametrize("intensity,dynamic", [(2.0, False), (1.0, True)])
def test_joint_inversion_rejects_non_unit_or_dynamic_lens_basis(
    intensity, dynamic,
):
    """Joint inversion requires a static unit-amplitude lens-light basis."""
    from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import SersicEllipse

    lens_light = SersicEllipse(
        R_sersic=1.0, n_sersic=4.0, Ie=intensity,
        e1=0.0, e2=0.0, center_x=0.0, center_y=0.0,
    )
    for parameter in (
        lens_light.R_sersic, lens_light.n_sersic, lens_light.Ie,
        lens_light.e1, lens_light.e2,
        lens_light.center_x, lens_light.center_y,
    ):
        parameter.to_static()
    if dynamic:
        lens_light.Ie.to_dynamic()
    phys = PhysicalModel(
        lens_mass=[_static_sie()], source_light=[_pix_src()],
        lens_light=[lens_light],
    )

    with pytest.raises(ValueError, match="unit-amplitude"):
        PixelizedImageProbModelOperator(
            jnp.zeros((10, 10)), jnp.ones((10, 10)) * 0.1,
            _delta_psf(), 0.08, phys,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_cls", [PixelizedImageProbModel, PixelizedImageProbModelOperator],
)
@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_joint_inversion_rejects_invalid_lens_light_regularization(model_cls, value):
    """Both pixelized backends reject invalid lens-light regularization."""
    with pytest.raises(ValueError, match="finite and positive"):
        model_cls(
            jnp.zeros((10, 10)), jnp.ones((10, 10)) * 0.1,
            _delta_psf(), 0.08, _phys_model(),
            lens_light_regularization=value,
        )


@pytest.mark.unit
def test_block_schur_logdet_matches_explicit_preconditioner():
    """Fast block-Schur logdet equals an explicit arrowhead reference."""
    source_blocks = jnp.asarray([
        [[2.0, 0.0], [0.3, 1.5]],
        [[1.2, 0.0], [-0.1, 1.8]],
    ])
    source_masks = jnp.asarray([[0, 1], [2, 3]])
    cross = jnp.asarray([
        [0.10, -0.03], [0.02, 0.04],
        [-0.05, 0.01], [0.03, 0.06],
    ])
    schur_chol = jnp.asarray([[1.4, 0.0], [0.2, 1.1]])
    preconditioner = BlockSchurPreconditioner(
        source_blocks, source_masks, cross, schur_chol,
    )
    source_precision = jnp.zeros((4, 4))
    for chol, mask in zip(source_blocks, source_masks):
        source_precision = source_precision.at[jnp.ix_(mask, mask)].set(
            chol @ chol.T
        )
    source_inverse_cross = jnp.linalg.solve(source_precision, cross)
    schur = schur_chol @ schur_chol.T
    lens_precision = schur + cross.T @ source_inverse_cross
    explicit = jnp.block([
        [source_precision, cross], [cross.T, lens_precision],
    ])
    _, expected = jnp.linalg.slogdet(explicit)

    actual = PixelizedImageProbModelOperator._logdet_block_schur(
        preconditioner
    )

    assert jnp.allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.unit
@pytest.mark.boundary
def test_ragged_block_schur_matches_explicit_preconditioner():
    """Ragged block-Schur inverse and diagonal match a dense reference."""
    source_chols = [
        jnp.asarray([[2.0, 0.0], [0.3, 1.5]]),
        jnp.asarray([[1.2]]),
    ]
    source_masks = [jnp.asarray([0, 1]), jnp.asarray([2])]
    cross = jnp.asarray([
        [0.10, -0.03],
        [0.02, 0.04],
        [-0.05, 0.01],
    ])
    schur_chol = jnp.asarray([[1.4, 0.0], [0.2, 1.1]])
    preconditioner = BlockSchurPreconditioner(
        source_chols, source_masks, cross, schur_chol,
    )

    source_precision = jnp.zeros((3, 3))
    for chol, mask in zip(source_chols, source_masks):
        source_precision = source_precision.at[jnp.ix_(mask, mask)].set(
            chol @ chol.T
        )
    source_inverse_cross = jnp.linalg.solve(source_precision, cross)
    lens_precision = (
        schur_chol @ schur_chol.T
        + cross.T @ source_inverse_cross
    )
    explicit = jnp.block([
        [source_precision, cross],
        [cross.T, lens_precision],
    ])
    rhs = jnp.asarray([0.4, -0.2, 0.7, 0.3, -0.5])

    expected_solution = jnp.linalg.solve(explicit, rhs)
    expected_diagonal = jnp.diag(explicit)
    eager_solution = apply_preconditioner(preconditioner, rhs)
    eager_diagonal = preconditioner_diagonal(preconditioner, rhs.size)
    jitted_solution = jax.jit(
        lambda value: apply_preconditioner(preconditioner, value)
    )(rhs)
    jitted_diagonal = jax.jit(
        lambda: preconditioner_diagonal(preconditioner, rhs.size)
    )()

    np.testing.assert_allclose(eager_solution, expected_solution, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(jitted_solution, expected_solution, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(eager_diagonal, expected_diagonal, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(jitted_diagonal, expected_diagonal, rtol=1e-6, atol=1e-6)


@pytest.mark.unit
def test_nonconverged_joint_solve_returns_exact_penalty_and_zero_components():
    """A failed joint solve has deterministic likelihood and forward outputs."""
    from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import SersicEllipse

    lens_light = SersicEllipse(
        R_sersic=0.4, n_sersic=2.0, Ie=1.0,
        e1=0.0, e2=0.0, center_x=0.0, center_y=0.0,
    )
    for parameter in (
        lens_light.R_sersic, lens_light.n_sersic, lens_light.Ie,
        lens_light.e1, lens_light.e2,
        lens_light.center_x, lens_light.center_y,
    ):
        parameter.to_static()
    phys = PhysicalModel(
        lens_mass=[_static_sie()], source_light=[_pix_src()],
        lens_light=[lens_light],
    )
    model = PixelizedImageProbModelOperator(
        jnp.ones((10, 10)), jnp.ones((10, 10)) * 0.1,
        _delta_psf(), 0.08, phys,
    )
    model.pcg_max_iter = 0

    assert model() == -1.0e10
    image, source, lens = model.forward_model(return_components=True)
    assert jnp.all(image == 0.0)
    assert jnp.all(source == 0.0)
    assert jnp.all(lens == 0.0)


@pytest.mark.unit
def test_weak_regularization_stabilizes_duplicate_lens_light_bases():
    """Duplicate lens-light columns remain a finite regularized joint system."""
    from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import GaussianEllipse

    lens_light = []
    for _ in range(2):
        component = GaussianEllipse(
            flux=1.0, sigma=0.3, e1=0.0, e2=0.0,
            center_x=0.0, center_y=0.0,
        )
        for parameter in (
            component.flux, component.sigma, component.e1, component.e2,
            component.center_x, component.center_y,
        ):
            parameter.to_static()
        lens_light.append(component)
    phys = PhysicalModel(
        lens_mass=[_static_sie()], source_light=[_pix_src()],
        lens_light=lens_light,
    )
    model = PixelizedImageProbModelOperator(
        jnp.ones((10, 10)) * 0.1, jnp.ones((10, 10)) * 0.1,
        _delta_psf(), 0.08, phys,
    )

    log_evidence = model()
    _, _, intensities = model.forward_model(return_components=True)

    assert jnp.isfinite(log_evidence)
    assert jnp.all(jnp.isfinite(intensities))


@pytest.mark.unit
def test_joint_pcg_converges_when_source_regularization_is_weak():
    """Block-Schur preconditioning remains SPD in the weak-prior regime."""
    from TinyLensGpu.PhysicalModel.LensImage.Parametric.Light import SersicEllipse

    lens_light = SersicEllipse(
        R_sersic=0.4, n_sersic=2.0, Ie=1.0,
        e1=0.0, e2=0.0, center_x=0.0, center_y=0.0,
    )
    for parameter in (
        lens_light.R_sersic, lens_light.n_sersic, lens_light.Ie,
        lens_light.e1, lens_light.e2,
        lens_light.center_x, lens_light.center_y,
    ):
        parameter.to_static()
    log_lambda = ParamU(
        "log_lambda_reg", -10.0,
        prior_type="uniform", prior_settings=[-14.0, 7.0],
    )
    log_lambda.to_dynamic()
    source = PixelizedSourceModel(
        n=20, log_lambda_reg=log_lambda,
        regularization_type="first-order",
    )
    phys = PhysicalModel(
        lens_mass=[_static_sie()], source_light=[source],
        lens_light=[lens_light],
    )
    model = PixelizedImageProbModelOperator(
        jnp.ones((10, 10)) * 0.1, jnp.ones((10, 10)) * 0.1,
        _delta_psf(), 0.08, phys, block_size=10,
    )

    model_image, source_pixels, lens_intensities = model.forward_model(
        return_components=True
    )

    assert jnp.any(model_image != 0.0)
    assert jnp.all(jnp.isfinite(source_pixels))
    assert jnp.all(jnp.isfinite(lens_intensities))


@pytest.mark.unit
def test_matrix_vs_operator_source_consistency_nsub2():
    """Source reconstruction parity with nsub=2 sub-grid sampling."""
    mock, noise, phys, config = _make_test_data(psf=_delta_psf(), nsub=2)

    prob_mat = PixelizedImageProbModel(
        mock, noise, _delta_psf(), 0.08, phys, mask=config.mask, nsub=2,
    )
    prob_op = PixelizedImageProbModelOperator(
        mock, noise, _delta_psf(), 0.08, phys, mask=config.mask, nsub=2,
    )

    _, src_mat = prob_mat.forward_model(return_source=True)
    _, src_op = prob_op.forward_model(return_source=True)

    rms_rel = float(
        jnp.sqrt(jnp.mean((src_mat - src_op) ** 2))
        / jnp.sqrt(jnp.mean(src_mat**2))
    )
    assert rms_rel < 1e-3, f"Source RMS relative error {rms_rel:.2e} exceeds 1e-3"


@pytest.mark.unit
def test_matrix_vs_operator_source_consistency_blur_psf():
    """Source reconstruction parity with a symmetric non-delta PSF."""
    psf = _blur_psf()
    mock, noise, phys, config = _make_test_data(psf=psf)

    prob_mat = PixelizedImageProbModel(
        mock, noise, psf, 0.08, phys, mask=config.mask,
    )
    prob_op = PixelizedImageProbModelOperator(
        mock, noise, psf, 0.08, phys, mask=config.mask,
    )

    _, src_mat = prob_mat.forward_model(return_source=True)
    _, src_op = prob_op.forward_model(return_source=True)

    rms_rel = float(
        jnp.sqrt(jnp.mean((src_mat - src_op) ** 2))
        / jnp.sqrt(jnp.mean(src_mat**2))
    )
    assert rms_rel < 1e-3, f"Source RMS relative error {rms_rel:.2e} exceeds 1e-3"


@pytest.mark.unit
def test_operator_forward_model_converges_pcg():
    """Operator forward model should produce a converged PCG solve with block-diag preconditioner."""
    mock, noise, phys, config = _make_test_data(psf=_delta_psf())

    prob_op = PixelizedImageProbModelOperator(
        mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
    )

    # Access internal _solve_source to inspect PCG convergence
    lambda_reg = jnp.exp(jnp.asarray(prob_op.source_model.log_lambda_reg.value))
    (xmin, xmax, ymin, ymax, _bx_sub, _by_sub,
     _bx_seed, _by_seed) = prob_op._get_bbox()
    reg_data = prob_op._regularization_data(xmin, xmax, ymin, ymax)

    block_chols, block_masks = prob_op.sim_obj.build_block_diag_preconditioner(
        prob_op.noise_1d, xmin, xmax, ymin, ymax, lambda_reg,
        prob_op.reg_builder, block_size=prob_op.block_size,
    )
    preconditioner = (block_chols, block_masks)
    source_pixels, pcg_info = prob_op._solve_source(
        xmin, xmax, ymin, ymax, lambda_reg, reg_data, preconditioner,
    )

    assert pcg_info.converged, (
        f"PCG did not converge: residual={pcg_info.residual_norm:.2e}, "
        f"iterations={pcg_info.n_iter}"
    )
    assert pcg_info.n_iter < prob_op.pcg_max_iter


@pytest.mark.unit
def test_dense_vectorized_likelihood_jit_with_uniform_reg():
    """Vectorized dense likelihood should compile for uniform regularization."""
    source = _pix_src(adaptive_reg_rho=0.0)
    phys = _phys_model(source=source)
    config = _sim_config()
    sim = PixelizedLensSimulator(phys, config)
    true_src = jnp.abs(jnp.linspace(-1.0, 1.0, 25))
    mock = sim.simulate(true_src, psf_kernel=_delta_psf())
    noise = jnp.ones((10, 10)) * 0.05

    prob_dense = PixelizedImageProbModel(
        mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
    )
    loglike = make_likelihood(prob_dense, vectorized=True)
    values = loglike(jnp.asarray([[0.0], [1.0]], dtype=jnp.float32))

    assert values.shape == (2,)
    assert jnp.all(jnp.isfinite(values))


@pytest.mark.unit
def test_dense_vectorized_likelihood_jit_with_template_adaptive_reg():
    """Vectorized dense likelihood compiles with a fixed S0 template and rho."""
    source = _pix_src(adaptive_reg_rho=1.0)
    phys = _phys_model(source=source)
    config = _sim_config()
    sim = PixelizedLensSimulator(phys, config)
    true_src = jnp.abs(jnp.linspace(-1.0, 1.0, 25))
    mock = sim.simulate(true_src, psf_kernel=_delta_psf())
    noise = jnp.ones((10, 10)) * 0.05

    prob_dense = PixelizedImageProbModel(
        mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
        fixed_source_bbox=_fixed_bbox(),
        fixed_reg_template=_fixed_template(),
    )
    loglike = make_likelihood(prob_dense, vectorized=True)
    values = loglike(jnp.asarray([[0.0], [1.0]], dtype=jnp.float32))

    assert values.shape == (2,)
    assert jnp.all(jnp.isfinite(values))


@pytest.mark.unit
@pytest.mark.parametrize("adaptive_reg_rho", [0.0, 1.0])
def test_operator_vectorized_likelihood_jit_with_adaptive_reg(adaptive_reg_rho):
    """Vectorized operator likelihood should compile for uniform and adaptive reg."""
    source = _pix_src(adaptive_reg_rho=adaptive_reg_rho)
    phys = _phys_model(source=source)
    config = _sim_config()
    sim = PixelizedLensSimulator(phys, config)
    true_src = jnp.abs(jnp.linspace(-1.0, 1.0, 25))
    mock = sim.simulate(true_src, psf_kernel=_delta_psf())
    noise = jnp.ones((10, 10)) * 0.05

    prob_op = PixelizedImageProbModelOperator(
        mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
        fixed_source_bbox=_fixed_bbox() if adaptive_reg_rho > 0 else None,
        fixed_reg_template=_fixed_template() if adaptive_reg_rho > 0 else None,
    )
    loglike = make_likelihood(prob_op, vectorized=True)
    values = loglike(jnp.asarray([[0.0], [1.0]], dtype=jnp.float32))

    assert values.shape == (2,)
    assert jnp.all(jnp.isfinite(values))


@pytest.mark.unit
def test_operator_chunked_likelihood_accepts_nautilus_batch_of_200():
    """The operator path preserves a sampler batch while bounding vmap memory."""
    phys = _phys_model()
    config = _sim_config()
    sim = PixelizedLensSimulator(phys, config)
    true_src = jnp.abs(jnp.linspace(-1.0, 1.0, 25))
    mock = sim.simulate(true_src, psf_kernel=_delta_psf())
    noise = jnp.ones((10, 10)) * 0.05
    prob_op = PixelizedImageProbModelOperator(
        mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
        fixed_source_bbox=_fixed_bbox(),
    )
    loglike = make_likelihood(
        prob_op, vectorized=True, vectorized_chunk_size=50
    )

    values = loglike(jnp.zeros((200, 1), dtype=jnp.float32))

    assert values.shape == (200,)
    assert jnp.all(jnp.isfinite(values))


@pytest.mark.unit
def test_operator_zero_order_adaptive_reg_jit_scan_path():
    """Zero-order adaptive regularization should compile on the scan preconditioner path."""
    lam = ParamU("log_lambda_reg", 0.0,
                 prior_type="uniform",
                 prior_settings=[jnp.log(1e-3), jnp.log(1e3)])
    lam.to_dynamic()
    source = PixelizedSourceModel(n=8,
        log_lambda_reg=lam,
        regularization_type="zero-order",
        adaptive_reg_rho=1.0,
    )
    phys = _phys_model(source=source)
    config = _sim_config()
    sim = PixelizedLensSimulator(phys, config)
    true_src = jnp.abs(jnp.linspace(-1.0, 1.0, 64))
    mock = sim.simulate(true_src, psf_kernel=_delta_psf())
    noise = jnp.ones((10, 10)) * 0.05

    prob_op = PixelizedImageProbModelOperator(
        mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
        block_size=4,
        fixed_source_bbox=_fixed_bbox(),
        fixed_reg_template=_fixed_template(n=8),
    )
    loglike = make_likelihood(prob_op, vectorized=True)
    values = loglike(jnp.asarray([[0.0], [1.0]], dtype=jnp.float32))

    assert values.shape == (2,)
    assert jnp.all(jnp.isfinite(values))


@pytest.mark.unit
def test_operator_fixed_bbox_overrides_seed_inference():
    """A configured source bbox should be used even though seed betas are computed."""
    source = _pix_src(adaptive_reg_rho=1.0)
    phys = _phys_model(source=source)
    mock, noise, _, config = _make_test_data()
    prob_op = PixelizedImageProbModelOperator(
        mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
        fixed_source_bbox=_fixed_bbox(),
        fixed_reg_template=_fixed_template(),
    )
    xmin, xmax, ymin, ymax, beta_x_sub, beta_y_sub, beta_x_seed, beta_y_seed = prob_op._get_bbox()

    np.testing.assert_allclose(
        np.asarray([xmin, xmax, ymin, ymax]), np.asarray(_fixed_bbox()), atol=1e-7
    )
    assert beta_x_sub.size > 0
    assert beta_y_sub.size > 0
    assert beta_x_seed.size > 0
    assert beta_y_seed.size > 0


@pytest.mark.unit
def test_operator_can_export_reference_bbox_for_fixed_sampling_grid():
    """The public bbox helper should round-trip into a fixed-grid model."""
    phys = _phys_model()
    reference = PixelizedImageProbModelOperator(
        image_data=jnp.ones((5, 5)),
        noise_map=jnp.ones((5, 5)),
        psf_kernel=_delta_psf(),
        dpix=0.1,
        phys_model=phys,
        source_bbox_padding=0.2,
    )

    bbox = reference.infer_source_bbox()
    fixed = PixelizedImageProbModelOperator(
        image_data=jnp.ones((5, 5)),
        noise_map=jnp.ones((5, 5)),
        psf_kernel=_delta_psf(),
        dpix=0.1,
        phys_model=phys,
        fixed_source_bbox=bbox,
    )

    np.testing.assert_allclose(fixed.infer_source_bbox(), bbox, atol=1e-7)


@pytest.mark.unit
def test_operator_rejects_rectangular_fixed_source_bbox():
    """Fixed source bboxes must be square for pixelized source grids."""
    source = _pix_src(adaptive_reg_rho=1.0)
    phys = _phys_model(source=source)
    mock, noise, _, config = _make_test_data()

    with pytest.raises(ValueError, match="fixed_source_bbox must be square"):
        PixelizedImageProbModelOperator(
            mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
            fixed_source_bbox=(-1.0, 1.0, -0.5, 0.5),
            fixed_reg_template=_fixed_template(),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_cls", [PixelizedImageProbModel, PixelizedImageProbModelOperator],
)
def test_adaptive_requires_fixed_source_bbox(model_cls):
    """Adaptive runs require a fixed source bbox from S0 on both backends."""
    source = _pix_src(adaptive_reg_rho=1.0)
    phys = _phys_model(source=source)
    mock, noise, _, config = _make_test_data()
    with pytest.raises(ValueError, match="fixed_source_bbox"):
        model_cls(
            mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
            fixed_reg_template=_fixed_template(),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_cls", [PixelizedImageProbModel, PixelizedImageProbModelOperator],
)
def test_adaptive_requires_fixed_reg_template(model_cls):
    """Adaptive runs require an S0 source template on both backends."""
    source = _pix_src(adaptive_reg_rho=1.0)
    phys = _phys_model(source=source)
    mock, noise, _, config = _make_test_data()
    with pytest.raises(ValueError, match="fixed_reg_template"):
        model_cls(
            mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
            fixed_source_bbox=_fixed_bbox(),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_cls", [PixelizedImageProbModel, PixelizedImageProbModelOperator],
)
@pytest.mark.parametrize(
    "template,message",
    [
        (jnp.ones(24), "fixed_reg_template must have shape"),
        (jnp.ones((5, 6)), "fixed_reg_template must have shape"),
        (jnp.concatenate([jnp.ones(24), jnp.asarray([jnp.nan])]),
         "fixed_reg_template values must be finite"),
        (jnp.concatenate([jnp.ones(24), jnp.asarray([jnp.inf])]),
         "fixed_reg_template values must be finite"),
    ],
)
def test_rejects_invalid_fixed_reg_template(model_cls, template, message):
    """Fixed templates must match the source grid and stay finite."""
    source = _pix_src(adaptive_reg_rho=1.0)
    phys = _phys_model(source=source)
    mock, noise, _, config = _make_test_data()
    with pytest.raises(ValueError, match=message):
        model_cls(
            mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
            fixed_source_bbox=_fixed_bbox(),
            fixed_reg_template=template,
        )


@pytest.mark.unit
def test_operator_rejects_removed_fixed_reg_scale_keyword():
    """The removed ``fixed_reg_scale`` keyword is a hard TypeError."""
    source = _pix_src(adaptive_reg_rho=1.0)
    phys = _phys_model(source=source)
    mock, noise, _, config = _make_test_data()
    with pytest.raises(TypeError):
        PixelizedImageProbModelOperator(
            mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
            fixed_source_bbox=_fixed_bbox(),
            fixed_reg_scale=jnp.ones(25),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_cls", [PixelizedImageProbModel, PixelizedImageProbModelOperator],
)
@pytest.mark.parametrize("flat", [True, False])
def test_fixed_template_scale_matches_source_template_scale_map(
    model_cls, flat,
):
    """Both backends evaluate the same scale map as the standalone utility."""
    source = _pix_src(adaptive_reg_rho=1.0)
    phys = _phys_model(source=source)
    mock, noise, _, config = _make_test_data()
    template = _fixed_template()
    model = model_cls(
        mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
        fixed_source_bbox=_fixed_bbox(),
        fixed_reg_template=template if flat else template.reshape(5, 5),
    )
    expected = source_template_scale_map(template, 5, rho=1.0)
    np.testing.assert_allclose(
        np.asarray(model._get_reg_scale()), np.asarray(expected),
        rtol=1e-6, atol=1e-6,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_cls", [PixelizedImageProbModel, PixelizedImageProbModelOperator],
)
@pytest.mark.parametrize(
    "template",
    [-jnp.abs(_fixed_template()), jnp.zeros(25)],
)
def test_negative_and_all_dark_templates_give_finite_scales(model_cls, template):
    """Negative pixels clip and all-dark templates normalize to finite weights."""
    source = _pix_src(adaptive_reg_rho=1.0)
    phys = _phys_model(source=source)
    mock, noise, _, config = _make_test_data()
    model = model_cls(
        mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
        fixed_source_bbox=_fixed_bbox(),
        fixed_reg_template=template,
    )
    scale = model._get_reg_scale()
    expected = source_template_scale_map(template, 5, rho=1.0)
    assert jnp.all(jnp.isfinite(scale))
    np.testing.assert_allclose(
        np.asarray(scale), np.asarray(expected), rtol=1e-6, atol=1e-6,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_cls", [PixelizedImageProbModel, PixelizedImageProbModelOperator],
)
def test_static_zero_rho_needs_no_template_or_bbox(model_cls):
    """A fixed near-zero rho disables adaptation; both inputs are optional."""
    source = _pix_src(adaptive_reg_rho=0.0)
    phys = _phys_model(source=source)
    mock, noise, _, config = _make_test_data()
    model = model_cls(
        mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
    )
    assert model._get_reg_scale() is None
    assert jnp.isfinite(model())


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_cls", [PixelizedImageProbModel, PixelizedImageProbModelOperator],
)
def test_dynamic_rho_initialized_zero_still_requires_template_inputs(model_cls):
    """A dynamic rho requires bbox and template even when it starts at zero."""
    lam = ParamU("log_lambda_reg", 0.0,
                 prior_type="uniform",
                 prior_settings=[jnp.log(1e-3), jnp.log(1e3)],
                 limits=[jnp.log(1e-3), jnp.log(1e3)])
    rho = ParamU("adaptive_reg_rho", 0.0,
                 prior_type="uniform",
                 prior_settings=[0.0, 3.0],
                 limits=[0.0, 3.0])
    for p in (lam, rho):
        p.to_dynamic()
    source = PixelizedSourceModel(n=5,
        log_lambda_reg=lam,
        regularization_type="first-order",
        adaptive_reg_rho=rho,
    )
    phys = _phys_model(source=source)
    mock, noise, _, config = _make_test_data()
    with pytest.raises(ValueError, match="fixed_source_bbox"):
        model_cls(
            mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
            fixed_reg_template=_fixed_template(),
        )
    with pytest.raises(ValueError, match="fixed_reg_template"):
        model_cls(
            mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
            fixed_source_bbox=_fixed_bbox(),
        )


@pytest.mark.unit
def test_operator_fixed_template_scale_reproducible():
    """_get_reg_scale() returns the template-derived S0 scale on every call."""
    source = _pix_src(adaptive_reg_rho=1.0)
    phys = _phys_model(source=source)
    mock, noise, _, config = _make_test_data()
    prob_op = PixelizedImageProbModelOperator(
        mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
        fixed_source_bbox=_fixed_bbox(),
        fixed_reg_template=_fixed_template(),
    )
    expected = _fixed_scale()
    scale_a = prob_op._get_reg_scale()
    scale_b = prob_op._get_reg_scale()
    np.testing.assert_allclose(
        np.asarray(scale_a), np.asarray(expected), rtol=1e-6, atol=1e-6,
    )
    assert jnp.array_equal(scale_a, scale_b)


@pytest.mark.unit
def test_operator_fixed_template_dynamic_scale_changes_with_hyperparams():
    """_get_reg_scale() can generate scale from S0 template and current params."""
    source = _pix_src_dynamic_adaptive()
    phys = _phys_model(source=source)
    mock, noise, _, config = _make_test_data()
    template = _fixed_template()
    prob_op = PixelizedImageProbModelOperator(
        mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
        fixed_source_bbox=_fixed_bbox(),
        fixed_reg_template=template,
    )

    with ck.ActiveContext(prob_op):
        prob_op.fill_params(jnp.asarray([0.0, 0.5]))
        scale_low = prob_op._get_reg_scale()
        prob_op.fill_params(jnp.asarray([0.0, 3.0]))
        scale_high = prob_op._get_reg_scale()

    assert scale_low.shape == (25,)
    assert scale_high.shape == (25,)
    assert jnp.all(jnp.isfinite(scale_low))
    assert jnp.all(jnp.isfinite(scale_high))
    assert float(scale_high[12]) > float(scale_low[12])


@pytest.mark.unit
def test_operator_dynamic_adaptive_requires_template():
    """Dynamic adaptive params require a fixed S0 source template."""
    source = _pix_src_dynamic_adaptive()
    phys = _phys_model(source=source)
    mock, noise, _, config = _make_test_data()
    with pytest.raises(ValueError, match="fixed_reg_template"):
        PixelizedImageProbModelOperator(
            mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
            fixed_source_bbox=_fixed_bbox(),
        )


@pytest.mark.unit
def test_operator_rejects_invalid_fixed_template_shape():
    """Fixed template shape must match the source-grid pixel count."""
    source = _pix_src(adaptive_reg_rho=1.0)
    phys = _phys_model(source=source)
    mock, noise, _, config = _make_test_data()
    with pytest.raises(ValueError, match="fixed_reg_template must have shape"):
        PixelizedImageProbModelOperator(
            mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
            fixed_source_bbox=_fixed_bbox(),
            fixed_reg_template=jnp.ones(24),
        )


@pytest.mark.unit
def test_operator_vectorized_likelihood_jit_with_template_adaptive_reg():
    """Vectorized operator likelihood should compile with dynamic S0-template scale."""
    source = _pix_src_dynamic_adaptive()
    phys = _phys_model(source=source)
    config = _sim_config()
    sim = PixelizedLensSimulator(phys, config)
    true_src = jnp.abs(jnp.linspace(-1.0, 1.0, 25))
    mock = sim.simulate(true_src, psf_kernel=_delta_psf())
    noise = jnp.ones((10, 10)) * 0.05

    prob_op = PixelizedImageProbModelOperator(
        mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
        fixed_source_bbox=_fixed_bbox(),
        fixed_reg_template=_fixed_template(),
    )
    loglike = make_likelihood(prob_op, vectorized=True)
    values = loglike(jnp.asarray([
        [0.0, 0.5],
        [1.0, 3.0],
    ], dtype=jnp.float32))

    assert values.shape == (2,)
    assert jnp.all(jnp.isfinite(values))


@pytest.mark.unit
def test_nonsymmetric_psf_A_matvec_matches_explicit():
    """Operator A_matvec must match explicit A for a non-centrosymmetric PSF.

    This verifies that the adjoint convolution correctly uses conj(FFT(PSF))
    rather than FFT(PSF), which would only be equivalent for symmetric PSFs.
    """
    # Non-centrosymmetric 3x3 PSF
    psf_asym = jnp.asarray(
        [[0.02, 0.05, 0.10],
         [0.08, 0.50, 0.15],
         [0.03, 0.05, 0.02]]
    )
    mock, noise, phys, config = _make_test_data(psf=psf_asym)

    sim_op = PixelizedLensOperator(phys, config)

    _, _, bx, by = sim_op._get_beta_sub_and_seed()
    xmi, xma, ymi, yma = sim_op._infer_and_fix_bbox(bx, by)

    n_1d = noise[~config.mask].ravel()
    lam = jnp.asarray(1.0)

    builder = FiniteDifferenceRegularization(5, "first-order")
    xmif, xmaf, ymif, ymaf = float(xmi), float(xma), float(ymi), float(yma)
    reg_dense = builder.matrix(xmif, xmaf, ymif, ymaf)
    reg_data = builder.make_reg_data(xmif, xmaf, ymif, ymaf)

    # Explicit F from operator's design_matrix (uses same bbox and PSF)
    F_op, _ = sim_op.design_matrix()
    wF = F_op / n_1d[:, None]
    A_explicit = wF.T @ wF + lam * reg_dense

    s = jnp.linspace(-1.0, 1.0, 25)

    operator = sim_op.build_source_curvature_operator(
        n_1d, xmi, xma, ymi, yma, lam, reg_data,
    )
    op_result = operator.matvec(s)
    explicit_result = A_explicit @ s

    np.testing.assert_allclose(
        np.array(op_result), np.array(explicit_result), rtol=1e-2, atol=0.2
    )


@pytest.mark.unit
def test_nonsymmetric_psf_source_consistency():
    """Matrix vs operator source reconstruction with a non-centrosymmetric PSF."""
    psf_asym = jnp.asarray(
        [[0.02, 0.05, 0.10],
         [0.08, 0.50, 0.15],
         [0.03, 0.05, 0.02]]
    )
    mock, noise, phys, config = _make_test_data(psf=psf_asym)

    prob_mat = PixelizedImageProbModel(
        mock, noise, psf_asym, 0.08, phys, mask=config.mask,
    )
    prob_op = PixelizedImageProbModelOperator(
        mock, noise, psf_asym, 0.08, phys, mask=config.mask,
    )

    _, src_mat = prob_mat.forward_model(return_source=True)
    _, src_op = prob_op.forward_model(return_source=True)

    rms_rel = float(
        jnp.sqrt(jnp.mean((src_mat - src_op) ** 2))
        / jnp.sqrt(jnp.mean(src_mat ** 2))
    )
    assert rms_rel < 1e-2, (
        f"Non-symmetric PSF source RMS relative error {rms_rel:.2e} exceeds 1e-2"
    )


# ------------------------------------------------------------------
# Matrix-free regularisation (matvec_free / logdet_free) tests
# ------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.parametrize("reg_type", ["zero-order", "first-order", "second-order"])
def test_matvec_free_matches_dense(reg_type):
    """matvec_free should give identical result to dense R @ s."""
    n = 7  # test different grid sizes
    builder = FiniteDifferenceRegularization(n, reg_type)
    xmin, xmax, ymin, ymax = -1.0, 2.0, -0.5, 1.5

    reg_dense = builder.matrix(xmin, xmax, ymin, ymax)
    s = jnp.linspace(-1.0, 1.0, n * n)

    result_free = builder.matvec_free(s, xmin, xmax, ymin, ymax)
    result_dense = reg_dense @ s

    np.testing.assert_allclose(
        np.array(result_free), np.array(result_dense), rtol=1e-4, atol=1e-4,
    )


@pytest.mark.unit
@pytest.mark.parametrize("reg_type", ["zero-order", "first-order", "second-order"])
def test_logdet_free_matches_dense(reg_type):
    """logdet_free should match slogdet of dense R."""
    n = 5
    builder = FiniteDifferenceRegularization(n, reg_type)
    xmin, xmax, ymin, ymax = -2.0, 2.0, -1.0, 3.0

    reg_dense = builder.matrix(xmin, xmax, ymin, ymax)
    _, logdet_expected = jnp.linalg.slogdet(reg_dense)
    logdet_computed = builder.logdet_free(xmin, xmax, ymin, ymax)

    np.testing.assert_allclose(
        float(logdet_computed), float(logdet_expected), rtol=1e-4, atol=1e-3,
    )


@pytest.mark.unit
@pytest.mark.parametrize("reg_type", ["first-order", "second-order"])
def test_edge_weighted_constant_source_has_zero_internal_energy(reg_type):
    """Edge-weighted FD regularisation must vanish on constant source pixels
    that are not on the global boundary (boundary fallbacks are numerical
    stabilisers and are excluded from this check).
    """
    n = 5
    builder = FiniteDifferenceRegularization(n, reg_type)
    xmin, xmax, ymin, ymax = -1.0, 1.0, -1.0, 1.0
    scale = jnp.linspace(0.5, 1.5, n * n)  # deliberately non-uniform
    s = jnp.ones(n * n)

    Rs = builder.matvec_free(s, xmin, xmax, ymin, ymax, scale=scale)
    Rs_2d = np.array(Rs).reshape(n, n)

    # Internal pixels (not on last row or last column) must be zero
    internal = Rs_2d[:-1, :-1]
    np.testing.assert_allclose(internal, 0.0, atol=1e-5)


@pytest.mark.unit
def test_to_dense_free_matches_matrix():
    """to_dense_free should match matrix() for FD types."""
    n = 6
    for reg_type in ("zero-order", "first-order", "second-order"):
        builder = FiniteDifferenceRegularization(n, reg_type)
        xmin, xmax, ymin, ymax = -1.5, 1.5, -0.5, 2.0
        expected = builder.matrix(xmin, xmax, ymin, ymax)
        computed = builder.to_dense_free(xmin, xmax, ymin, ymax)
        np.testing.assert_allclose(
            np.array(computed), np.array(expected), rtol=1e-4, atol=1e-4,
        ), f"Failed for {reg_type}"


@pytest.mark.unit
@pytest.mark.parametrize("reg_type", ["first-order", "second-order"])
def test_diag_R_matches_dense_matrix_diagonal(reg_type):
    """diag_R should match jnp.diag(matrix()) for non-square grids with
    non-uniform adaptive scale — this catches coefficient errors and
    scale_x/scale_y cross-contamination."""
    n = 7  # square grid
    builder = FiniteDifferenceRegularization(n, reg_type)
    xmin, xmax, ymin, ymax = -2.0, 3.0, -1.0, 4.0  # asymmetric, dx != dy
    scale = jnp.linspace(0.3, 2.0, n * n)  # non-uniform

    # Reference: extract diagonal from the full dense matrix.
    R_dense = builder.matrix(xmin, xmax, ymin, ymax, scale=scale)
    expected = jnp.diag(R_dense)

    computed = builder.diag_R(xmin, xmax, ymin, ymax, scale=scale)

    # diag_R (analytic stencil) vs jnp.diag(matrix()) (dense matmul) differ only
    # by float32 accumulation noise (~2.5e-4 relative at n=35); the analytic
    # formula is verified bit-accurate against the dense path in float64.
    np.testing.assert_allclose(
        np.array(computed), np.array(expected), rtol=1e-3, atol=1e-3,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_scale",
    [
        jnp.ones((25, 1)),
        jnp.concatenate([jnp.ones(24), jnp.asarray([0.0])]),
        jnp.concatenate([jnp.ones(24), jnp.asarray([-1.0])]),
    ],
)
def test_regularization_scale_validation_rejects_invalid_scale(bad_scale):
    """Invalid adaptive scale should fail before sqrt/logdet paths diverge."""
    builder = FiniteDifferenceRegularization(5, "first-order")
    with pytest.raises(ValueError, match="scale"):
        builder.matrix(-1.0, 1.0, -1.0, 1.0, scale=bad_scale)


@pytest.mark.unit
def test_make_reg_data_shapes():
    """make_reg_data should return scale and physical factors."""
    n = 8
    builder = FiniteDifferenceRegularization(n, "first-order")
    rd = builder.make_reg_data(-1.0, 1.0, -1.0, 1.0)

    assert rd.scale is None
    assert rd.scale_factor.shape == ()
    assert rd.scale_factor.shape == ()


@pytest.mark.unit
def test_make_reg_data_propagates_scale():
    """make_reg_data should propagate a provided scale array into RegData."""
    n = 8
    builder = FiniteDifferenceRegularization(n, "first-order")
    scale = jnp.linspace(0.5, 1.5, n * n)
    rd = builder.make_reg_data(-1.0, 1.0, -1.0, 1.0, scale=scale)
    assert rd.scale is not None
    assert rd.scale.shape == (n * n,)


@pytest.mark.unit
@pytest.mark.parametrize("reg_type", ["first-order", "second-order"])
def test_matvec_free_nonuniform_scale_matches_dense(reg_type):
    """matvec_free with non-uniform scale must match dense R(scale) @ s.

    This directly validates the edge-weighting (geometric-mean) math, which
    the uniform-scale matvec test cannot exercise: a bug in _geom_mean /
    _geom_mean3 or in the scale_x/scale_y off-diagonal scaling would only
    surface with a non-constant source and non-uniform scale.
    """
    n = 7  # test different grid sizes so dx != dy
    builder = FiniteDifferenceRegularization(n, reg_type)
    xmin, xmax, ymin, ymax = -1.0, 2.0, -0.5, 1.5
    scale = jnp.linspace(0.3, 2.0, n * n)  # non-uniform
    s = jnp.linspace(-1.0, 1.0, n * n)     # non-constant

    reg_dense = builder.matrix(xmin, xmax, ymin, ymax, scale=scale)
    result_free = builder.matvec_free(s, xmin, xmax, ymin, ymax, scale=scale)
    result_dense = reg_dense @ s

    # Tolerance accommodates float32 accumulation noise from the different
    # operation orderings of the matrix-free stencil vs the dense matmul
    # (~3e-3 at n=35); a real weighting bug would produce O(1) errors.
    np.testing.assert_allclose(
        np.array(result_free), np.array(result_dense), rtol=5e-3, atol=5e-3,
    )


@pytest.mark.unit
@pytest.mark.parametrize("reg_type", ["first-order", "second-order"])
def test_logdet_free_block_diag_approximation(reg_type):
    """logdet_free with block_size < grid_dim is a block-diagonal approximation.

    The block-diagonal approximation uses principal submatrices ``R_ii`` of the
    full R, dropping only the cross-block off-diagonal couplings.  By the
    Hadamard-Fischer inequality, ``det(R) <= prod_i det(R_ii)``, so the
    approximation (without jitter) is ``>=`` the exact ``slogdet(R)``
    (biased high).  The per-block Cholesky jitter
    (``1e-6 * max(diag_mean, 1.0) * I``) further increases each block's
    determinant.  When ``block_size >= grid_dim`` the whole grid fits in one
    block and the approximation is exact (up to jitter).

    ``exact=True`` must match ``slogdet`` of the full R to float precision.
    """
    n = 9  # smaller than default block_size=10
    builder = FiniteDifferenceRegularization(n, reg_type)
    xmin, xmax, ymin, ymax = -2.0, 2.0, -1.0, 3.0
    scale = jnp.linspace(0.3, 2.0, n * n)

    reg_dense = builder.matrix(xmin, xmax, ymin, ymax, scale=scale)
    logdet_slogdet = float(jnp.linalg.slogdet(reg_dense)[1])

    # exact=True must match slogdet of the full R matrix.
    logdet_exact = float(builder.logdet_free(
        xmin, xmax, ymin, ymax, scale=scale, exact=True,
    ))
    np.testing.assert_allclose(logdet_exact, logdet_slogdet, rtol=1e-4, atol=1e-3)

    # Multi-block: 9x9 grid with block_size=4 -> 3x3 = 9 blocks (approximate).
    logdet_multiblock = float(builder.logdet_free(
        xmin, xmax, ymin, ymax, scale=scale, block_size=4,
    ))
    # Single-block: block_size >= grid_dim -> exact (up to jitter).
    logdet_single = float(builder.logdet_free(
        xmin, xmax, ymin, ymax, scale=scale, block_size=16,
    ))

    # Single-block must match exact slogdet (jitter ~1e-6 * diag).
    np.testing.assert_allclose(logdet_single, logdet_slogdet, rtol=1e-3, atol=1e-1)

    # Multi-block must be >= exact: by Hadamard-Fischer, prod det(R_ii) >=
    # det(R), and the per-block jitter only increases it further.
    assert jnp.isfinite(logdet_multiblock)
    assert logdet_multiblock >= logdet_slogdet - 1e-1, (
        f"Multi-block logdet {logdet_multiblock:.4f} should be >= exact "
        f"{logdet_slogdet:.4f} for {reg_type} (Hadamard-Fischer)"
    )
    rel_err = abs(logdet_multiblock - logdet_slogdet) / max(abs(logdet_slogdet), 1.0)
    assert rel_err < 0.2, (
        f"Multi-block logdet {logdet_multiblock:.4f} deviates from exact "
        f"{logdet_slogdet:.4f} by {rel_err:.2%} for {reg_type}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("n", [2, 5])
@pytest.mark.parametrize("reg_type", ["first-order", "second-order"])
def test_matvec_free_degenerate_grid_matches_dense(n, reg_type):
    """matvec_free must match dense R @ s on degenerate (thin) grids.

    For source_n=2 the second-order full-curvature stencil (source_n > 2) is skipped and
    only the near-boundary first-gradient fallback runs.  This guards the
    ``if self.n > 2`` / ``if self.n > 2`` boundary branches.
    """
    builder = FiniteDifferenceRegularization(n, reg_type)
    xmin, xmax, ymin, ymax = -1.0, 1.0, -0.5, 1.5

    reg_dense = builder.matrix(xmin, xmax, ymin, ymax)
    s = jnp.linspace(-1.0, 1.0, n * n)

    result_free = builder.matvec_free(s, xmin, xmax, ymin, ymax)
    result_dense = reg_dense @ s

    np.testing.assert_allclose(
        np.array(result_free), np.array(result_dense), rtol=1e-4, atol=1e-4,
    )


@pytest.mark.unit
@pytest.mark.parametrize("reg_type", ["first-order", "second-order"])
@pytest.mark.parametrize("use_scale", [False, True])
def test_block_diag_R_is_principal_submatrix(reg_type, use_scale):
    """block_diag_R must equal the principal submatrix R[bf, bf] of the full R.

    This guards against the regression where cross-block stencil rows (edges
    or curvatures straddling the block boundary) were dropped, missing their
    diagonal contributions to in-block pixels.  The principal submatrix
    includes those contributions by construction.
    """
    n = 10  # square grid with block_size=4
    builder = FiniteDifferenceRegularization(n, reg_type)
    xmin, xmax, ymin, ymax = -2.0, 2.0, -1.0, 3.0
    scale = jnp.linspace(0.3, 2.0, n * n) if use_scale else None

    R_full = builder.matrix(xmin, xmax, ymin, ymax, scale=scale)

    # Check several blocks: interior, edge, and corner blocks
    for x_s, x_e, y_s, y_e in [(2, 6, 2, 6), (0, 4, 0, 4), (6, 10, 6, 10)]:
        bf = jnp.asarray(
            [x + y * n for y in range(y_s, y_e) for x in range(x_s, x_e)],
            dtype=jnp.int32,
        )
        R_principal = R_full[jnp.ix_(bf, bf)]
        R_block = builder.block_diag_R(
            x_s, x_e, y_s, y_e, xmin, xmax, ymin, ymax, scale=scale,
        )
        # Tolerance accommodates float32 accumulation noise from the different
        # operation orderings of the stencil scatter vs the dense matmul.
        # Uniform scale is exact (rtol=0); adaptive scale differs only by
        # rounding (~0.04 for second-order at n=16 block).
        np.testing.assert_allclose(
            np.array(R_block), np.array(R_principal), rtol=1e-3, atol=0.1,
            err_msg=f"block ({x_s}:{x_e}, {y_s}:{y_e}) mismatch for "
                    f"{reg_type}, use_scale={use_scale}",
        )


# ------------------------------------------------------------------
# evaluate_with_diagnostics
# ------------------------------------------------------------------

def _operator_model(image, noise, phys, solver="pcg", **kwargs):
    """Build a small operator evidence model on a fixed source grid."""
    kwargs.setdefault("fixed_source_bbox", _fixed_bbox())
    return PixelizedImageProbModelOperator(
        image, noise, _delta_psf(), 0.08, phys, solver_type=solver, **kwargs
    )


@pytest.mark.unit
@pytest.mark.parametrize("solver", ["pcg", "pnpg"])
@pytest.mark.parametrize("joint", [False, True])
def test_operator_diagnostics_match_evaluate_and_unrequested_none(solver, joint):
    """Operator diagnostics pair the evaluate result with basic products."""
    from TinyLensGpu.ObservationModel import InversionDiagnostics

    if joint:
        image, noise, phys = _make_joint_test_data()
    else:
        image, noise, phys, _ = _make_test_data()
    model = _operator_model(image, noise, phys, solver=solver)
    theta = jnp.asarray(model.get_values("flat"))
    result, diagnostics = model.evaluate_with_diagnostics(theta)
    reference = model.evaluate(theta)

    assert isinstance(diagnostics, InversionDiagnostics)
    np.testing.assert_allclose(
        float(result.log_evidence), float(reference.log_evidence), rtol=1e-5, atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(result.model_image), np.asarray(reference.model_image), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(result.source_pixels), np.asarray(reference.source_pixels), rtol=1e-5, atol=1e-5
    )
    if joint:
        np.testing.assert_allclose(
            np.asarray(result.lens_amplitudes), np.asarray(reference.lens_amplitudes),
            rtol=1e-5, atol=1e-5,
        )
    assert diagnostics.valid and diagnostics.invalid_reason is None
    assert bool(result.solver_info.converged)
    assert diagnostics.effective_parameters is None
    assert diagnostics.residual_dof is None
    assert diagnostics.reduced_chi2 is None
    assert diagnostics.effective_dof_method is None
    assert diagnostics.effective_dof_reason is None
    assert diagnostics.source_image is None
    assert diagnostics.lens_light_image is None
    assert diagnostics.position is None

    expected_resid = (np.asarray(model.image_data) - np.asarray(result.model_image)) / np.asarray(
        model.noise_map
    )
    np.testing.assert_allclose(
        np.asarray(diagnostics.normalized_residuals), expected_resid, rtol=1e-5, atol=1e-5
    )
    expected_chi2 = float(np.sum(expected_resid[np.asarray(model.unmask)] ** 2))
    np.testing.assert_allclose(diagnostics.chi2, expected_chi2, rtol=1e-5, atol=1e-5)
    n = int(model.source_model.n)
    assert diagnostics.regularization_scale.shape == (n, n)
    np.testing.assert_allclose(
        np.asarray(diagnostics.regularization_scale), 1.0, rtol=1e-6, atol=1e-6
    )


@pytest.mark.unit
def test_operator_diagnostics_adaptive_regularization_scale():
    """Adaptive regularization reports the public S0-template scale map."""
    image, noise, phys, _ = _make_test_data()
    phys = _phys_model(_pix_src(adaptive_reg_rho=1.0))
    model = _operator_model(
        image, noise, phys, fixed_reg_template=_fixed_template()
    )
    theta = jnp.asarray(model.get_values("flat"))
    _, diagnostics = model.evaluate_with_diagnostics(theta)
    assert diagnostics.valid
    n = int(model.source_model.n)
    expected = np.asarray(
        source_template_scale_map(_fixed_template(), n, 1.0)
    ).reshape(n, n)
    np.testing.assert_allclose(
        np.asarray(diagnostics.regularization_scale), expected, rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(diagnostics.adaptive_reg_rho, 1.0, atol=1e-6)


def _operator_effective_dof_reference(model, bbox, block_size, rho, n):
    """Independent NumPy trace over the block-diagonal preconditioner."""
    from scipy.linalg import cho_solve as scipy_cho_solve

    lam = float(np.exp(np.asarray(model.source_model.log_lambda_reg.value)))
    scale = (
        source_template_scale_map(_fixed_template(n), n, rho) if rho > 0.0 else None
    )
    block_chols, _ = model.sim_obj.build_block_diag_preconditioner(
        model.noise_1d,
        *bbox,
        lam,
        model.reg_builder,
        block_size=block_size,
        scale=scale,
    )
    reg_blocks = model.reg_builder.block_matrices(
        *bbox, scale=scale, block_size=block_size
    )
    trace = 0.0
    for chol, rblock in zip(block_chols, reg_blocks):
        trace += float(
            np.trace(scipy_cho_solve(
                (np.asarray(chol), True), np.asarray(rblock)
            ))
        )
    effective = int(model.sim_obj.n_source_pixels) - lam * trace
    return effective, int(model.data_1d.size) - effective


@pytest.mark.unit
@pytest.mark.parametrize("solver", ["pcg", "pnpg"])
@pytest.mark.parametrize("block_size", [2, 5])
@pytest.mark.parametrize("rho", [0.0, 1.0])
def test_operator_diagnostics_effective_dof_matches_block_reference(solver, block_size, rho):
    """Source-only effective parameters match an independent block trace."""
    image, noise, phys, _ = _make_test_data()
    phys = _phys_model(_pix_src(adaptive_reg_rho=rho))
    kwargs = {"fixed_reg_template": _fixed_template()} if rho > 0.0 else {}
    model = _operator_model(
        image, noise, phys, solver=solver, block_size=block_size, **kwargs
    )
    theta = jnp.asarray(model.get_values("flat"))
    _, diagnostics = model.evaluate_with_diagnostics(
        theta, include_effective_dof=True
    )
    assert diagnostics.valid
    assert diagnostics.effective_dof_method == "operator_source_block_preconditioner"
    effective, dof = _operator_effective_dof_reference(
        model, _fixed_bbox(), block_size, rho, int(model.source_model.n)
    )
    np.testing.assert_allclose(
        diagnostics.effective_parameters, effective, rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(diagnostics.residual_dof, dof, rtol=1e-4, atol=1e-4)
    expected_reduced = diagnostics.chi2 / dof if dof > 0 else 0.0
    np.testing.assert_allclose(
        diagnostics.reduced_chi2, expected_reduced, rtol=1e-4, atol=1e-4
    )


@pytest.mark.unit
def test_operator_diagnostics_joint_effective_dof_unsupported():
    """Joint effective-parameter requests report unsupported, not invalid."""
    image, noise, phys = _make_joint_test_data()
    model = _operator_model(image, noise, phys)
    theta = jnp.asarray(model.get_values("flat"))
    result, diagnostics = model.evaluate_with_diagnostics(
        theta, include_effective_dof=True, include_components=True
    )
    assert diagnostics.valid and diagnostics.invalid_reason is None
    assert diagnostics.effective_parameters is None
    assert diagnostics.residual_dof is None
    assert diagnostics.reduced_chi2 is None
    assert diagnostics.effective_dof_method == "operator_joint_unsupported"
    assert "unsupported" in diagnostics.effective_dof_reason
    # Raw chi-square and image products remain available.
    assert np.isfinite(diagnostics.chi2)
    active = np.asarray(model.unmask)
    total = np.asarray(diagnostics.source_image) + np.asarray(diagnostics.lens_light_image)
    np.testing.assert_allclose(
        total[active], np.asarray(result.model_image)[active], rtol=1e-4, atol=1e-4
    )


@pytest.mark.unit
def test_operator_diagnostics_components_sum_to_model_image():
    """Requested source/lens components sum to the model on active pixels."""
    image, noise, phys = _make_joint_test_data()
    model = _operator_model(image, noise, phys)
    theta = jnp.asarray(model.get_values("flat"))
    result, diagnostics = model.evaluate_with_diagnostics(
        theta, include_components=True
    )
    assert diagnostics.valid
    active = np.asarray(model.unmask)
    total = np.asarray(diagnostics.source_image) + np.asarray(diagnostics.lens_light_image)
    np.testing.assert_allclose(
        total[active], np.asarray(result.model_image)[active], rtol=1e-4, atol=1e-4
    )
    source_ref, lens_ref, _ = model.reconstruct_component_images(
        result.source_pixels, result.lens_amplitudes
    )
    np.testing.assert_allclose(
        np.asarray(diagnostics.source_image), np.asarray(source_ref), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(diagnostics.lens_light_image), np.asarray(lens_ref), rtol=1e-4, atol=1e-4
    )


@pytest.mark.unit
def test_operator_diagnostics_source_only_lens_image_zeros():
    """A source-only model returns a zero lens-light component image."""
    image, noise, phys, _ = _make_test_data()
    model = _operator_model(image, noise, phys)
    theta = jnp.asarray(model.get_values("flat"))
    result, diagnostics = model.evaluate_with_diagnostics(
        theta, include_components=True
    )
    assert diagnostics.valid
    np.testing.assert_allclose(np.asarray(diagnostics.lens_light_image), 0.0)
    active = np.asarray(model.unmask)
    np.testing.assert_allclose(
        np.asarray(diagnostics.source_image)[active],
        np.asarray(result.model_image)[active],
        rtol=1e-4, atol=1e-4,
    )


@pytest.mark.unit
def test_operator_diagnostics_position_likelihood_fields():
    """Enabled position constraints report penalty and mapped positions."""
    positions = [[0.2, 0.0], [-0.2, 0.0]]
    image, noise, phys, _ = _make_test_data()
    model = _operator_model(
        image, noise, phys,
        position_likelihood=dict(
            positions=positions, threshold_arcsec=0.01, min_log_like=-20.0,
        ),
    )
    theta = jnp.asarray(model.get_values("flat"))
    result, diagnostics = model.evaluate_with_diagnostics(theta)
    assert diagnostics.valid and diagnostics.position is not None
    positions_arr = np.asarray(positions, dtype=np.float32)
    with ck.ActiveContext(model):
        model.fill_params(theta)
        bx, by = model.phys_model.deflection(
            jnp.asarray(positions_arr[:, 0]), jnp.asarray(positions_arr[:, 1])
        )
    dist = np.sqrt(
        (np.asarray(bx)[:, None] - np.asarray(bx)[None, :]) ** 2
        + (np.asarray(by)[:, None] - np.asarray(by)[None, :]) ** 2
    )
    max_sep = float(dist.max())
    penalty = -20.0 * (1.0 - np.exp(-max(0.0, max_sep - 0.01) / 0.01))
    np.testing.assert_allclose(
        diagnostics.position.penalty, penalty, rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(diagnostics.position.image_positions), positions_arr,
        rtol=1e-6, atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(diagnostics.position.source_positions),
        np.stack([np.asarray(bx), np.asarray(by)], axis=-1),
        rtol=1e-4, atol=1e-4,
    )
    np.testing.assert_allclose(
        diagnostics.position.max_separation, max_sep, rtol=1e-4, atol=1e-4
    )


@pytest.mark.unit
def test_operator_diagnostics_unconverged_solve_is_invalid():
    """Explicit non-convergence marks the zero reconstruction invalid."""
    image, noise, phys, _ = _make_test_data()
    model = _operator_model(image, noise, phys, solver="pcg")
    model.pcg_max_iter = 0
    theta = jnp.asarray(model.get_values("flat"))
    result, diagnostics = model.evaluate_with_diagnostics(
        theta, include_effective_dof=True
    )
    assert not bool(result.solver_info.converged)
    np.testing.assert_allclose(np.asarray(result.model_image), 0.0)
    np.testing.assert_allclose(np.asarray(result.source_pixels), 0.0)
    assert not diagnostics.valid
    assert "converge" in diagnostics.invalid_reason
    assert float(result.log_evidence) == -1.0e10
    assert diagnostics.effective_parameters is None
    assert diagnostics.effective_dof_reason == diagnostics.invalid_reason


@pytest.mark.unit
def test_operator_diagnostics_nonfinite_observation_is_invalid():
    """A nonfinite observation reports invalid diagnostics without raising."""
    image, noise, phys, _ = _make_test_data()
    model = _operator_model(
        np.full((10, 10), np.nan), noise, phys, fixed_source_bbox=None
    )
    theta = jnp.asarray(model.get_values("flat"))
    result, diagnostics = model.evaluate_with_diagnostics(
        theta, include_effective_dof=True
    )
    assert not diagnostics.valid
    assert diagnostics.invalid_reason is not None
    assert float(result.log_evidence) == -1.0e10
    assert diagnostics.effective_dof_reason == diagnostics.invalid_reason


@pytest.mark.unit
def test_operator_diagnostics_uses_one_source_solve(monkeypatch):
    """One diagnostic request performs exactly one inversion."""
    image, noise, phys, _ = _make_test_data()
    model = _operator_model(image, noise, phys)
    theta = jnp.asarray(model.get_values("flat"))
    solve = model._solve_source
    calls = []

    def counted_solve(*args, **kwargs):
        calls.append(None)
        return solve(*args, **kwargs)

    monkeypatch.setattr(model, "_solve_source", counted_solve)
    result, diagnostics = model.evaluate_with_diagnostics(
        theta, include_effective_dof=True, include_components=True
    )
    assert len(calls) == 1
    assert diagnostics.valid


@pytest.mark.unit
def test_operator_likelihood_paths_do_no_diagnostic_work(monkeypatch):
    """evaluate/__call__/forward_model never assemble diagnostics."""
    import TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator as op_model

    image, noise, phys, _ = _make_test_data()
    model = _operator_model(image, noise, phys)

    def refuse(*args, **kwargs):
        raise AssertionError("likelihood path assembled diagnostics")

    monkeypatch.setattr(op_model, "reconstruction_diagnostics", refuse)
    theta = jnp.asarray(model.get_values("flat"))
    result = model.evaluate(theta)
    assert np.isfinite(float(result.log_evidence))
    assert np.isfinite(float(model(theta)))
    model.forward_model(theta, return_source=True)


if __name__ == "__main__":
    pytest.main()
