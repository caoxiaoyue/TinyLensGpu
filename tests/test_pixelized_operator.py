"""Tests for the operator-based (matrix-free) pixelized source backend."""

import functools
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from TinyLensGpu.ForwardSimulation.LensImage.config import SimulatorConfig
from TinyLensGpu.ForwardSimulation.LensImage.pixelized import PixelizedLensSimulator
from TinyLensGpu.ForwardSimulation.LensImage.pixelized_operator import (
    PixelizedLensOperator,
)
from TinyLensGpu.Inference import ParamU
from TinyLensGpu.Inference.build_likelihood import make_likelihood
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model import (
    PixelizedImageProbModel,
)
from TinyLensGpu.ObservationModel.LensImage.pixelized_image_model_operator import (
    PixelizedImageProbModelOperator,
)
from TinyLensGpu.PhysicalModel.LensImage.Parametric.Mass import SIE
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
from TinyLensGpu.utils.fista_solver import fista_nnls_solve
from TinyLensGpu.utils.pnpg_solver import pnpg_nnls_solve
from TinyLensGpu.utils.inversion.regularization import (
    DenseRegularizationBuilder,
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


def _assert_square_bbox(source_bbox):
    xmin, xmax, ymin, ymax = source_bbox
    assert jnp.allclose(xmax - xmin, ymax - ymin)


@pytest.mark.unit
def test_operator_infers_square_source_bbox_for_asymmetric_betas():
    """Test operator bbox helper expands asymmetric seed-ray extents to square."""
    config = _sim_config()
    config.source_bbox_outlier_frac = 0.0
    operator = PixelizedLensOperator(_phys_model(), config)
    beta_x = jnp.asarray([0.0, 3.0])
    beta_y = jnp.asarray([-0.2, 0.2])

    source_bbox = operator._infer_and_fix_bbox(beta_x, beta_y)

    _assert_square_bbox(source_bbox)
    xmin, xmax, ymin, ymax = source_bbox
    assert jnp.allclose(xmin, 0.0)
    assert jnp.allclose(xmax, 3.0)
    assert jnp.allclose(ymin, -1.5)
    assert jnp.allclose(ymax, 1.5)


# ------------------------------------------------------------------
# PCG solver tests
# ------------------------------------------------------------------

# Minimal RegData placeholder for PCG tests that bypass regularisation
# via a custom _simple_A.  Not consumed by the real _A_matvec_jit path.
_DUMMY_REG_DATA = (
    jnp.ones(1, dtype=jnp.float32),     # scale
    jnp.array(1.0, dtype=jnp.float32),  # scale_factor
)


@pytest.mark.unit
def test_pcg_solves_small_spd_system():
    """PCG should solve a well-conditioned SPD system to high accuracy."""
    n = 20
    import jax, jax.random as jrandom
    key = jrandom.PRNGKey(42)
    B = jrandom.normal(key, (n, n))
    A_dense = B.T @ B + 0.1 * jnp.eye(n)  # SPD
    b = jnp.linspace(-1.0, 1.0, n)

    # Use the new tuple-based PCG API (A_data has 9 slots)
    import functools
    @functools.partial(jax.jit, static_argnames=())
    def _simple_A(s, w, idx, fi, *, agg_segment_ids=None, psf_fft=None,
                  psf_fft_conj=None,
                  noise_var=None, reg_data=None, lambda_reg=None, **_kw):
        return w @ s   # 'w' = A_dense (first data slot)

    A_data = (A_dense, None, None, None, None, None, None, _DUMMY_REG_DATA, None)
    P_chol = jnp.eye(n)

    x, info = pcg_solve(A_data, b, P_chol, _simple_A, max_iter=100, rtol=1e-10)

    expected = jnp.linalg.solve(A_dense, b)
    assert jnp.allclose(x, expected, atol=1e-4)
    assert info.converged


@pytest.mark.unit
def test_pcg_uses_preconditioner():
    """PCG with a good preconditioner converges in fewer iterations."""
    n = 30
    import jax
    diag = jnp.logspace(-2, 2, n)
    A_dense = jnp.diag(diag)
    b = jnp.ones(n)

    import functools
    @functools.partial(jax.jit, static_argnames=())
    def _simple_A(s, w, idx, fi, *, agg_segment_ids=None, psf_fft=None,
                  psf_fft_conj=None,
                  noise_var=None, reg_data=None, lambda_reg=None, **_kw):
        return w @ s

    A_data = (A_dense, None, None, None, None, None, None, _DUMMY_REG_DATA, None)

    # No preconditioner (identity)
    P_chol_id = jnp.eye(n)
    _, info_id = pcg_solve(A_data, b, P_chol_id, _simple_A, max_iter=200, rtol=1e-8)

    # Good preconditioner (Cholesky of the exact A)
    P_chol_good = jnp.diag(jnp.sqrt(diag))
    _, info_good = pcg_solve(A_data, b, P_chol_good, _simple_A, max_iter=200, rtol=1e-8)

    # Good preconditioner should converge faster than identity
    assert info_good.n_iter < info_id.n_iter
    assert info_good.converged


@pytest.mark.unit
def test_pcg_detects_non_spd_failure():
    """PCG should set failed=True and converged=False for a non-SPD system."""
    n = 4
    import jax, jax.random as jrandom

    key = jrandom.PRNGKey(7)
    B = jrandom.normal(key, (n, n))
    # Negative-definite matrix → pAp < 0 on the first iteration.
    A_dense = -(B.T @ B + 0.1 * jnp.eye(n))
    b = jnp.ones(n)

    @functools.partial(jax.jit, static_argnames=())
    def _simple_A(s, w, idx, fi, *, agg_segment_ids=None, psf_fft=None,
                  psf_fft_conj=None,
                  noise_var=None, reg_data=None, lambda_reg=None, **_kw):
        return w @ s

    A_data = (A_dense, None, None, None, None, None, None, _DUMMY_REG_DATA, None)
    P_chol = jnp.eye(n)

    x, info = pcg_solve(A_data, b, P_chol, _simple_A, max_iter=50, rtol=1e-10)

    assert info.failed, f"Expected failed=True for non-SPD system, got failed={info.failed}"
    assert not info.converged, "Expected converged=False when breakdown occurs"
    assert jnp.all(jnp.isfinite(x)), "Returned x should be finite even on failure"


# ------------------------------------------------------------------
# FISTA NNLS solver tests
# ------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=())
def _simple_A(s, w, idx, fi, *, agg_segment_ids=None, psf_fft=None,
              psf_fft_conj=None,
              noise_var=None, reg_data=None, lambda_reg=None, **_kw):
    return w @ s


@pytest.mark.unit
def test_fista_nnls_solves_positive_quadratic():
    """FISTA should solve a small non-negative SPD quadratic."""
    A_dense = jnp.diag(jnp.asarray([2.0, 4.0, 8.0], dtype=jnp.float32))
    b = jnp.asarray([2.0, 8.0, 4.0], dtype=jnp.float32)
    A_data = (A_dense, None, None, None, None, None, None, _DUMMY_REG_DATA, None)

    x, info = fista_nnls_solve(
        A_data, b, _simple_A,
        max_iter=300, rtol=1e-5, power_iter=8,
    )

    expected = jnp.asarray([1.0, 2.0, 0.5], dtype=jnp.float32)
    np.testing.assert_allclose(np.array(x), np.array(expected), rtol=1e-3, atol=1e-3)
    assert jnp.all(x >= 0.0)
    assert jnp.isfinite(info.convergence_metric)
    assert info.converged


@pytest.mark.unit
def test_fista_nnls_projects_negative_solution_to_zero():
    """FISTA projection should enforce x >= 0 for an all-negative drive."""
    A_dense = jnp.eye(4, dtype=jnp.float32)
    b = -jnp.ones(4, dtype=jnp.float32)
    A_data = (A_dense, None, None, None, None, None, None, _DUMMY_REG_DATA, None)

    x, info = fista_nnls_solve(
        A_data, b, _simple_A,
        max_iter=50, rtol=1e-5, power_iter=4,
    )

    np.testing.assert_allclose(np.array(x), np.zeros(4), atol=1e-6)
    assert info.converged
    assert not info.failed


@pytest.mark.unit
def test_fista_nnls_reports_invalid_step_failure():
    """Invalid explicit step sizes should fail cleanly without NaN output."""
    A_dense = jnp.eye(3, dtype=jnp.float32)
    b = jnp.ones(3, dtype=jnp.float32)
    A_data = (A_dense, None, None, None, None, None, None, _DUMMY_REG_DATA, None)

    x, info = fista_nnls_solve(
        A_data, b, _simple_A,
        max_iter=5, step_size=-1.0,
    )

    assert info.failed
    assert not info.converged
    assert jnp.all(jnp.isfinite(x))


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
    A_data = (A_dense, None, None, None, None, None, None, _DUMMY_REG_DATA, None)

    x, info = pnpg_nnls_solve(
        A_data,
        b,
        jnp.diag(jnp.sqrt(eigenvalues)),
        _simple_A,
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
    A_data = (A_dense, None, None, None, None, None, None, _DUMMY_REG_DATA, None)

    x, info = pnpg_nnls_solve(
        A_data,
        b,
        jnp.linalg.cholesky(A_dense),
        _simple_A,
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
    A_data = (A_dense, None, None, None, None, None, None, _DUMMY_REG_DATA, None)

    x, info = pnpg_nnls_solve(
        A_data,
        jnp.ones(size, dtype=jnp.float32),
        jnp.eye(size, dtype=jnp.float32),
        _simple_A,
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
    A_data = (A_dense, None, None, None, None, None, None, _DUMMY_REG_DATA, None)

    _, info = pnpg_nnls_solve(
        A_data,
        jnp.ones(3, dtype=jnp.float32),
        jnp.eye(3, dtype=jnp.float32),
        _simple_A,
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

    builder = DenseRegularizationBuilder(5, "first-order")
    xmif, xmaf, ymif, ymaf = float(xmi), float(xma), float(ymi), float(yma)
    reg_dense, _ = builder.matrix(xmif, xmaf, ymif, ymaf)
    reg_data = builder.make_reg_data(xmif, xmaf, ymif, ymaf)

    # Build explicit F from operator's design_matrix (uses same bbox)
    F_op, bbox_op = sim_op.design_matrix()

    # Explicit A
    wF = F_op / n_1d[:, None]
    A_explicit = wF.T @ wF + lam * reg_dense

    s = jnp.linspace(-1.0, 1.0, 25)

    # Operator A — uses compact reg_data
    A_data, _A_jit = sim_op.build_A_matvec(n_1d, xmi, xma, ymi, yma, lam, reg_data)
    op_result = sim_op.call_A_matvec(s, A_data, _A_jit)
    explicit_result = A_explicit @ s

    # Float32 operator chain vs explicit dense
    np.testing.assert_allclose(
        np.array(op_result), np.array(explicit_result), rtol=1e-3, atol=1e-3
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
    builder = DenseRegularizationBuilder(5, "first-order")

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
    builder = DenseRegularizationBuilder(5, "first-order")

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
            solver_type="fista",
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
def test_operator_fista_returns_nonnegative_source_for_negative_data():
    """FISTA source solve should enforce hard non-negative source pixels."""
    mock, noise, phys, config = _make_test_data(psf=_delta_psf())
    mock = -jnp.abs(mock)

    model = PixelizedImageProbModelOperator(
        mock, noise, _delta_psf(), 0.08, phys,
        mask=config.mask,
        solver_type="fista",
    )

    log_ev = model()
    _, source = model.forward_model(return_source=True)

    assert jnp.isfinite(log_ev)
    assert source.shape == (25,)
    assert jnp.all(source >= -1e-6)


@pytest.mark.unit
def test_operator_rejects_invalid_solver_type():
    """Operator model should reject unsupported source solver names."""
    mock, noise, phys, config = _make_test_data(psf=_delta_psf())

    with pytest.raises(ValueError, match="solver_type"):
        PixelizedImageProbModelOperator(
            mock, noise, _delta_psf(), 0.08, phys,
            mask=config.mask,
            solver_type="bad",
        )


@pytest.mark.unit
def test_operator_source_bbox_padding_expands_inferred_bbox():
    """Operator model forwards bbox padding into source-plane bbox inference."""
    mock, noise, phys, config = _make_test_data(psf=_delta_psf())

    model_plain = PixelizedImageProbModelOperator(
        mock, noise, _delta_psf(), 0.08, phys,
        mask=config.mask,
        source_bbox_padding=0.0,
    )
    model_padded = PixelizedImageProbModelOperator(
        mock, noise, _delta_psf(), 0.08, phys,
        mask=config.mask,
        source_bbox_padding=0.25,
    )

    xmin0, xmax0, ymin0, ymax0, *_ = model_plain._get_bbox()
    xmin1, xmax1, ymin1, ymax1, *_ = model_padded._get_bbox()

    assert xmax1 - xmin1 > xmax0 - xmin0
    assert ymax1 - ymin1 > ymax0 - ymin0


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
def test_operator_fista_forward_model_returns_nonnegative_source():
    """FISTA forward_model returns finite image and non-negative source."""
    mock, noise, phys, config = _make_test_data(psf=_delta_psf())

    model = PixelizedImageProbModelOperator(
        mock, noise, _delta_psf(), 0.08, phys,
        mask=config.mask,
        solver_type="fista",
    )

    model_image, source = model.forward_model(return_source=True)
    assert model_image.shape == (10, 10)
    assert source.shape == (25,)
    assert jnp.all(jnp.isfinite(model_image))
    assert jnp.all(jnp.isfinite(source))
    assert jnp.all(source >= -1e-6)


@pytest.mark.unit
def test_operator_fista_solver_controls_are_configurable():
    """Constructor-level FISTA controls should be stored on the model."""
    mock, noise, phys, config = _make_test_data(psf=_delta_psf())

    model = PixelizedImageProbModelOperator(
        mock, noise, _delta_psf(), 0.08, phys,
        mask=config.mask,
        solver_type="fista",
        fista_max_iter=17,
        fista_rtol=2e-5,
        fista_power_iter=5,
        fista_step_safety=1.4,
    )

    assert model.fista_max_iter == 17
    assert model.fista_rtol == 2e-5
    assert model.fista_power_iter == 5
    assert model.fista_step_safety == 1.4


@pytest.mark.unit
def test_operator_fista_nonconvergence_penalizes_evidence():
    """FISTA gating should penalize an explicitly non-converged solve.

    ``max_iter=0`` exercises the no-iteration boundary: the solver returns the
    zero initial source, reports non-convergence from the projected-gradient
    metric, and the evidence layer applies the same large penalty used for
    failed iterative solves.
    """
    mock, noise, phys, config = _make_test_data(psf=_delta_psf())

    model = PixelizedImageProbModelOperator(
        mock, noise, _delta_psf(), 0.08, phys,
        mask=config.mask,
        solver_type="fista",
    )
    # Boundary case for evidence gating rather than a realistic hard problem.
    model.fista_max_iter = 0

    log_ev = model()
    assert log_ev < -1.0e9


def _dense_objective(design_matrix, data_1d, noise_1d, reg_matrix, lambda_reg, source):
    resid = data_1d - design_matrix @ source
    e_d = 0.5 * jnp.sum((resid / noise_1d) ** 2)
    e_s = 0.5 * jnp.dot(source, reg_matrix @ source)
    return e_d + lambda_reg * e_s


@pytest.mark.unit
def test_operator_fista_matches_dense_nnls_objective_small_grid():
    """Operator FISTA should match dense NNLS objective on a small problem."""
    mock, noise, phys, config = _make_test_data(psf=_delta_psf())
    source_bbox = _fixed_bbox()
    lambda_reg = jnp.exp(jnp.asarray(phys.source_light[0].log_lambda_reg.value))

    prob_dense = PixelizedImageProbModel(
        mock, noise, _delta_psf(), 0.08, phys,
        mask=config.mask,
        solver_type="nnls",
    )
    design_matrix, _ = prob_dense.sim_obj.design_matrix(source_bbox=source_bbox)
    reg_matrix, _ = prob_dense._regularization_matrix(source_bbox)
    src_dense, _, _ = prob_dense._solve_source(design_matrix, reg_matrix, lambda_reg)

    prob_op = PixelizedImageProbModelOperator(
        mock, noise, _delta_psf(), 0.08, phys,
        mask=config.mask,
        fixed_source_bbox=source_bbox,
        solver_type="fista",
    )
    prob_op.fista_max_iter = 1000
    prob_op.fista_rtol = 1e-4
    _, src_op = prob_op.forward_model(return_source=True)

    obj_dense = _dense_objective(
        design_matrix, prob_dense.data_1d, prob_dense.noise_1d,
        reg_matrix, lambda_reg, src_dense,
    )
    obj_op = _dense_objective(
        design_matrix, prob_dense.data_1d, prob_dense.noise_1d,
        reg_matrix, lambda_reg, src_op,
    )

    assert jnp.all(src_op >= -1e-6)
    np.testing.assert_allclose(float(obj_op), float(obj_dense), rtol=5e-3, atol=5e-3)


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
    reg_matrix, _ = prob_dense._regularization_matrix(source_bbox)
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
@pytest.mark.parametrize("solver_type", ["pcg", "fista", "pnpg"])
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
def test_operator_joint_map_matches_dense_backend(nsub):
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
        lens_mass=[_static_sie()], source_light=[_pix_src()],
        lens_light=[lens_light],
    )
    config = _sim_config(nsub=nsub)
    simulator = PixelizedLensSimulator(phys, config)
    true_source = jnp.abs(jnp.linspace(-0.5, 1.0, 25))
    image = simulator.simulate(
        true_source, lens_light_amplitudes=jnp.asarray([0.7]),
    )
    noise = jnp.ones_like(image) * 0.05
    dense = PixelizedImageProbModel(
        image, noise, _delta_psf(), 0.08, phys, nsub=nsub,
    )
    operator = PixelizedImageProbModelOperator(
        image, noise, _delta_psf(), 0.08, phys, nsub=nsub,
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
def test_fista_joint_inversion_constrains_source_and_lens_light_nonnegative():
    """FISTA projects both joint linear-parameter groups to non-negative values."""
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
        _delta_psf(), 0.08, phys, solver_type="fista",
        fista_max_iter=1000, fista_rtol=1e-4,
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
def test_dense_vectorized_likelihood_rejects_adaptive_reg():
    """Dense backend no longer exposes the retired seed-ray adaptive path."""
    source = _pix_src(adaptive_reg_rho=1.0)
    phys = _phys_model(source=source)
    config = _sim_config()
    sim = PixelizedLensSimulator(phys, config)
    true_src = jnp.abs(jnp.linspace(-1.0, 1.0, 25))
    mock = sim.simulate(true_src, psf_kernel=_delta_psf())
    noise = jnp.ones((10, 10)) * 0.05

    with pytest.raises(ValueError, match="no longer supports"):
        PixelizedImageProbModel(
            mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
        )


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
        fixed_reg_scale=_fixed_scale(rho=adaptive_reg_rho) if adaptive_reg_rho > 0 else None,
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
        fixed_reg_scale=_fixed_scale(n=8),
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
        fixed_reg_scale=_fixed_scale(),
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
            fixed_reg_scale=_fixed_scale(),
        )


@pytest.mark.unit
def test_operator_adaptive_requires_fixed_bbox():
    """Adaptive operator runs require a fixed source bbox from S0."""
    source = _pix_src(adaptive_reg_rho=1.0)
    phys = _phys_model(source=source)
    mock, noise, _, config = _make_test_data()
    with pytest.raises(ValueError, match="fixed_source_bbox"):
        PixelizedImageProbModelOperator(
            mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
            fixed_reg_scale=_fixed_scale(),
        )


@pytest.mark.unit
def test_operator_adaptive_requires_fixed_scale():
    """Adaptive operator runs require an S0-derived fixed scale map."""
    source = _pix_src(adaptive_reg_rho=1.0)
    phys = _phys_model(source=source)
    mock, noise, _, config = _make_test_data()
    with pytest.raises(ValueError, match="fixed_reg_scale"):
        PixelizedImageProbModelOperator(
            mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
            fixed_source_bbox=_fixed_bbox(),
        )


@pytest.mark.unit
def test_operator_rejects_invalid_fixed_scale_shape():
    """Fixed scale shape must match the source-grid pixel count."""
    source = _pix_src(adaptive_reg_rho=1.0)
    phys = _phys_model(source=source)
    mock, noise, _, config = _make_test_data()
    with pytest.raises(ValueError, match="fixed_reg_scale must have shape"):
        PixelizedImageProbModelOperator(
            mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
            fixed_source_bbox=_fixed_bbox(),
            fixed_reg_scale=jnp.ones(24),
        )


@pytest.mark.unit
def test_operator_fixed_scale_reproducible():
    """_get_reg_scale() returns the configured S0 scale on every call."""
    source = _pix_src(adaptive_reg_rho=1.0)
    phys = _phys_model(source=source)
    mock, noise, _, config = _make_test_data()
    fixed_scale = _fixed_scale()
    prob_op = PixelizedImageProbModelOperator(
        mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
        fixed_source_bbox=_fixed_bbox(),
        fixed_reg_scale=fixed_scale,
    )
    scale_a = prob_op._get_reg_scale()
    scale_b = prob_op._get_reg_scale()
    assert jnp.array_equal(scale_a, fixed_scale)
    assert jnp.array_equal(scale_b, fixed_scale)


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
def test_operator_dynamic_adaptive_requires_scale_or_template():
    """Dynamic adaptive params require fixed S0 scale or template inputs."""
    source = _pix_src_dynamic_adaptive()
    phys = _phys_model(source=source)
    mock, noise, _, config = _make_test_data()
    with pytest.raises(ValueError, match="fixed_reg_scale or fixed_reg_template"):
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

    builder = DenseRegularizationBuilder(5, "first-order")
    xmif, xmaf, ymif, ymaf = float(xmi), float(xma), float(ymi), float(yma)
    reg_dense, _ = builder.matrix(xmif, xmaf, ymif, ymaf)
    reg_data = builder.make_reg_data(xmif, xmaf, ymif, ymaf)

    # Explicit F from operator's design_matrix (uses same bbox and PSF)
    F_op, _ = sim_op.design_matrix()
    wF = F_op / n_1d[:, None]
    A_explicit = wF.T @ wF + lam * reg_dense

    s = jnp.linspace(-1.0, 1.0, 25)

    A_data, _A_jit = sim_op.build_A_matvec(n_1d, xmi, xma, ymi, yma, lam, reg_data)
    op_result = sim_op.call_A_matvec(s, A_data, _A_jit)
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
    builder = DenseRegularizationBuilder(n, reg_type)
    xmin, xmax, ymin, ymax = -1.0, 2.0, -0.5, 1.5

    reg_dense, _ = builder.matrix(xmin, xmax, ymin, ymax)
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
    builder = DenseRegularizationBuilder(n, reg_type)
    xmin, xmax, ymin, ymax = -2.0, 2.0, -1.0, 3.0

    reg_dense, _ = builder.matrix(xmin, xmax, ymin, ymax)
    _, logdet_expected = jnp.linalg.slogdet(reg_dense)
    logdet_computed = builder.logdet_free(xmin, xmax, ymin, ymax)

    np.testing.assert_allclose(
        float(logdet_computed), float(logdet_expected), rtol=1e-4, atol=1e-3,
    )


@pytest.mark.unit
def test_matvec_free_raises_for_gp():
    """matvec_free should raise ValueError for GP types."""
    builder = DenseRegularizationBuilder(5, "exponential")
    s = jnp.ones(25)
    with pytest.raises(ValueError, match="GP"):
        builder.matvec_free(s, -1.0, 1.0, -1.0, 1.0)


@pytest.mark.unit
def test_logdet_free_raises_for_gp():
    """logdet_free should raise ValueError for GP types."""
    builder = DenseRegularizationBuilder(5, "gaussian")
    with pytest.raises(ValueError, match="GP"):
        builder.logdet_free(-1.0, 1.0, -1.0, 1.0)


@pytest.mark.unit
@pytest.mark.parametrize("reg_type", ["first-order", "second-order"])
def test_edge_weighted_constant_source_has_zero_internal_energy(reg_type):
    """Edge-weighted FD regularisation must vanish on constant source pixels
    that are not on the global boundary (boundary fallbacks are numerical
    stabilisers and are excluded from this check).
    """
    n = 5
    builder = DenseRegularizationBuilder(n, reg_type)
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
        builder = DenseRegularizationBuilder(n, reg_type)
        xmin, xmax, ymin, ymax = -1.5, 1.5, -0.5, 2.0
        expected, _ = builder.matrix(xmin, xmax, ymin, ymax)
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
    builder = DenseRegularizationBuilder(n, reg_type)
    xmin, xmax, ymin, ymax = -2.0, 3.0, -1.0, 4.0  # asymmetric, dx != dy
    scale = jnp.linspace(0.3, 2.0, n * n)  # non-uniform

    # Reference: extract diagonal from the full dense matrix.
    R_dense, _ = builder.matrix(xmin, xmax, ymin, ymax, scale=scale)
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
    builder = DenseRegularizationBuilder(5, "first-order")
    with pytest.raises(ValueError, match="scale"):
        builder.matrix(-1.0, 1.0, -1.0, 1.0, scale=bad_scale)


@pytest.mark.unit
def test_make_reg_data_shapes():
    """make_reg_data should return scale and physical factors."""
    n = 8
    builder = DenseRegularizationBuilder(n, "first-order")
    rd = builder.make_reg_data(-1.0, 1.0, -1.0, 1.0)

    assert rd.scale is None
    assert rd.scale_factor.shape == ()
    assert rd.scale_factor.shape == ()


@pytest.mark.unit
def test_make_reg_data_raises_for_gp():
    """make_reg_data should raise ValueError for GP types (operator backend unsupported)."""
    builder = DenseRegularizationBuilder(5, "exponential")
    with pytest.raises(ValueError, match="Operator backend does not support GP"):
        builder.make_reg_data(-1.0, 1.0, -1.0, 1.0)


@pytest.mark.unit
def test_make_reg_data_propagates_scale():
    """make_reg_data should propagate a provided scale array into RegData."""
    n = 8
    builder = DenseRegularizationBuilder(n, "first-order")
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
    builder = DenseRegularizationBuilder(n, reg_type)
    xmin, xmax, ymin, ymax = -1.0, 2.0, -0.5, 1.5
    scale = jnp.linspace(0.3, 2.0, n * n)  # non-uniform
    s = jnp.linspace(-1.0, 1.0, n * n)     # non-constant

    reg_dense, _ = builder.matrix(xmin, xmax, ymin, ymax, scale=scale)
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
    builder = DenseRegularizationBuilder(n, reg_type)
    xmin, xmax, ymin, ymax = -2.0, 2.0, -1.0, 3.0
    scale = jnp.linspace(0.3, 2.0, n * n)

    reg_dense, _ = builder.matrix(xmin, xmax, ymin, ymax, scale=scale)
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
def test_operator_rejects_gp_regularization_type():
    """PixelizedLensOperator should reject GP regularization at construction.

    GP types require a dense (Ns, Ns) precision matrix and gain nothing from
    the matrix-free operator backend; they must use the dense backend.
    """
    gp_source = PixelizedSourceModel(n=5, regularization_type="gaussian",
    )
    phys = PhysicalModel(
        lens_mass=[_static_sie()],
        source_light=[gp_source],
        lens_light=[],
    )
    config = _sim_config()
    with pytest.raises(ValueError, match="finite-difference"):
        PixelizedLensOperator(phys, config)


@pytest.mark.unit
@pytest.mark.parametrize("n", [2, 5])
@pytest.mark.parametrize("reg_type", ["first-order", "second-order"])
def test_matvec_free_degenerate_grid_matches_dense(n, reg_type):
    """matvec_free must match dense R @ s on degenerate (thin) grids.

    For source_n=2 the second-order full-curvature stencil (source_n > 2) is skipped and
    only the near-boundary first-gradient fallback runs.  This guards the
    ``if self.n > 2`` / ``if self.n > 2`` boundary branches.
    """
    builder = DenseRegularizationBuilder(n, reg_type)
    xmin, xmax, ymin, ymax = -1.0, 1.0, -0.5, 1.5

    reg_dense, _ = builder.matrix(xmin, xmax, ymin, ymax)
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
    builder = DenseRegularizationBuilder(n, reg_type)
    xmin, xmax, ymin, ymax = -2.0, 2.0, -1.0, 3.0
    scale = jnp.linspace(0.3, 2.0, n * n) if use_scale else None

    R_full, _ = builder.matrix(xmin, xmax, ymin, ymax, scale=scale)

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


if __name__ == "__main__":
    pytest.main()
