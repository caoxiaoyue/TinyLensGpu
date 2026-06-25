"""Tests for the operator-based (matrix-free) pixelized source backend."""

import functools

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
from TinyLensGpu.utils.cg_solver import pcg_solve
from TinyLensGpu.utils.inversion.regularization import DenseRegularizationBuilder

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


def _pix_src(log_lambda_val=0.0, adaptive_reg_alpha=0.0):
    lam = ParamU("log_lambda_reg", log_lambda_val,
                 prior_type="uniform",
                 prior_settings=[jnp.log(1e-3), jnp.log(1e3)])
    lam.to_dynamic()
    return PixelizedSourceModel(
        nx=5,
        ny=5,
        log_lambda_reg=lam,
        regularization_type="first-order",
        adaptive_reg_alpha=adaptive_reg_alpha,
    )


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


# ------------------------------------------------------------------
# PCG solver tests
# ------------------------------------------------------------------

# Minimal 3-tuple RegData placeholder for PCG tests that bypass regularisation
# via a custom _simple_A.  Not consumed by the real _A_matvec_jit path.
_DUMMY_REG_DATA = (
    jnp.ones(1, dtype=jnp.float32),     # scale
    jnp.array(1.0, dtype=jnp.float32),  # scale_x
    jnp.array(1.0, dtype=jnp.float32),  # scale_y
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

    builder = DenseRegularizationBuilder(5, 5, "first-order")
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
    builder = DenseRegularizationBuilder(5, 5, "first-order")

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
def test_preconditioner_stacks_equal_sized_blocks():
    """Equal-sized block preconditioners should be stacked for vmapped solves."""
    mock, noise, phys, config = _make_test_data(psf=_delta_psf())

    sim_op = PixelizedLensOperator(phys, config)
    _, _, bx, by = sim_op._get_beta_sub_and_seed()
    xmi, xma, ymi, yma = sim_op._infer_and_fix_bbox(bx, by)

    n_1d = noise[~config.mask].ravel()
    lam = jnp.asarray(1.0)
    builder = DenseRegularizationBuilder(5, 5, "first-order")

    block_chols, block_masks = sim_op.build_block_diag_preconditioner(
        n_1d, xmi, xma, ymi, yma, lam, builder, block_size=5,
    )

    assert isinstance(block_chols, jnp.ndarray)
    assert isinstance(block_masks, jnp.ndarray)
    assert block_chols.shape == (1, 25, 25)
    assert block_masks.shape == (1, 25)


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
def test_lens_light_raises_not_implemented():
    """Operator backend should raise NotImplementedError with lens light."""
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

    with pytest.raises(NotImplementedError, match="Lens-light"):
        PixelizedImageProbModelOperator(
            jnp.zeros((10, 10)), jnp.ones((10, 10)) * 0.1,
            _delta_psf(), 0.08, phys,
        )


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
@pytest.mark.parametrize("adaptive_reg_alpha", [0.0, 1.0])
def test_dense_vectorized_likelihood_jit_with_adaptive_reg(adaptive_reg_alpha):
    """Vectorized dense likelihood should compile for uniform and adaptive reg."""
    source = _pix_src(adaptive_reg_alpha=adaptive_reg_alpha)
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
@pytest.mark.parametrize("adaptive_reg_alpha", [0.0, 1.0])
def test_operator_vectorized_likelihood_jit_with_adaptive_reg(adaptive_reg_alpha):
    """Vectorized operator likelihood should compile for uniform and adaptive reg."""
    source = _pix_src(adaptive_reg_alpha=adaptive_reg_alpha)
    phys = _phys_model(source=source)
    config = _sim_config()
    sim = PixelizedLensSimulator(phys, config)
    true_src = jnp.abs(jnp.linspace(-1.0, 1.0, 25))
    mock = sim.simulate(true_src, psf_kernel=_delta_psf())
    noise = jnp.ones((10, 10)) * 0.05

    prob_op = PixelizedImageProbModelOperator(
        mock, noise, _delta_psf(), 0.08, phys, mask=config.mask,
    )
    loglike = make_likelihood(prob_op, vectorized=True)
    values = loglike(jnp.asarray([[0.0], [1.0]], dtype=jnp.float32))

    assert values.shape == (2,)
    assert jnp.all(jnp.isfinite(values))


@pytest.mark.unit
def test_operator_zero_order_adaptive_reg_jit_scan_path():
    """Zero-order adaptive regularization should compile on the scan preconditioner path."""
    lam = ParamU("log_lambda_reg", 0.0,
                 prior_type="uniform",
                 prior_settings=[jnp.log(1e-3), jnp.log(1e3)])
    lam.to_dynamic()
    source = PixelizedSourceModel(
        nx=8,
        ny=8,
        log_lambda_reg=lam,
        regularization_type="zero-order",
        adaptive_reg_alpha=1.0,
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
    )
    loglike = make_likelihood(prob_op, vectorized=True)
    values = loglike(jnp.asarray([[0.0], [1.0]], dtype=jnp.float32))

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

    builder = DenseRegularizationBuilder(5, 5, "first-order")
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
    nx, ny = 5, 7  # non-square
    builder = DenseRegularizationBuilder(nx, ny, reg_type)
    xmin, xmax, ymin, ymax = -1.0, 2.0, -0.5, 1.5

    reg_dense, _ = builder.matrix(xmin, xmax, ymin, ymax)
    s = jnp.linspace(-1.0, 1.0, nx * ny)

    result_free = builder.matvec_free(s, xmin, xmax, ymin, ymax)
    result_dense = reg_dense @ s

    np.testing.assert_allclose(
        np.array(result_free), np.array(result_dense), rtol=1e-4, atol=1e-4,
    )


@pytest.mark.unit
@pytest.mark.parametrize("reg_type", ["zero-order", "first-order", "second-order"])
def test_logdet_free_matches_dense(reg_type):
    """logdet_free should match slogdet of dense R."""
    nx, ny = 5, 5
    builder = DenseRegularizationBuilder(nx, ny, reg_type)
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
    builder = DenseRegularizationBuilder(5, 5, "exponential")
    s = jnp.ones(25)
    with pytest.raises(ValueError, match="GP"):
        builder.matvec_free(s, -1.0, 1.0, -1.0, 1.0)


@pytest.mark.unit
def test_logdet_free_raises_for_gp():
    """logdet_free should raise ValueError for GP types."""
    builder = DenseRegularizationBuilder(5, 5, "gaussian")
    with pytest.raises(ValueError, match="GP"):
        builder.logdet_free(-1.0, 1.0, -1.0, 1.0)


@pytest.mark.unit
@pytest.mark.parametrize("reg_type", ["first-order", "second-order"])
def test_edge_weighted_constant_source_has_zero_internal_energy(reg_type):
    """Edge-weighted FD regularisation must vanish on constant source pixels
    that are not on the global boundary (boundary fallbacks are numerical
    stabilisers and are excluded from this check).
    """
    nx, ny = 5, 5
    builder = DenseRegularizationBuilder(nx, ny, reg_type)
    xmin, xmax, ymin, ymax = -1.0, 1.0, -1.0, 1.0
    scale = jnp.linspace(0.5, 1.5, nx * ny)  # deliberately non-uniform
    s = jnp.ones(nx * ny)

    Rs = builder.matvec_free(s, xmin, xmax, ymin, ymax, scale=scale)
    Rs_2d = np.array(Rs).reshape(ny, nx)

    # Internal pixels (not on last row or last column) must be zero
    internal = Rs_2d[:-1, :-1]
    np.testing.assert_allclose(internal, 0.0, atol=1e-5)


@pytest.mark.unit
def test_to_dense_free_matches_matrix():
    """to_dense_free should match matrix() for FD types."""
    nx, ny = 4, 6
    for reg_type in ("zero-order", "first-order", "second-order"):
        builder = DenseRegularizationBuilder(nx, ny, reg_type)
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
    nx, ny = 7, 5  # deliberately non-square so dx != dy
    builder = DenseRegularizationBuilder(nx, ny, reg_type)
    xmin, xmax, ymin, ymax = -2.0, 3.0, -1.0, 4.0  # asymmetric, dx != dy
    scale = jnp.linspace(0.3, 2.0, nx * ny)  # non-uniform

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
    builder = DenseRegularizationBuilder(5, 5, "first-order")
    with pytest.raises(ValueError, match="scale"):
        builder.matrix(-1.0, 1.0, -1.0, 1.0, scale=bad_scale)


@pytest.mark.unit
def test_make_reg_data_shapes():
    """make_reg_data should return scale and physical factors."""
    nx, ny = 5, 8
    builder = DenseRegularizationBuilder(nx, ny, "first-order")
    rd = builder.make_reg_data(-1.0, 1.0, -1.0, 1.0)

    assert rd.scale is None
    assert rd.scale_x.shape == ()
    assert rd.scale_y.shape == ()


@pytest.mark.unit
def test_make_reg_data_raises_for_gp():
    """make_reg_data should raise ValueError for GP types (operator backend unsupported)."""
    builder = DenseRegularizationBuilder(5, 5, "exponential")
    with pytest.raises(ValueError, match="Operator backend does not support GP"):
        builder.make_reg_data(-1.0, 1.0, -1.0, 1.0)


@pytest.mark.unit
def test_make_reg_data_propagates_scale():
    """make_reg_data should propagate a provided scale array into RegData."""
    nx, ny = 5, 8
    builder = DenseRegularizationBuilder(nx, ny, "first-order")
    scale = jnp.linspace(0.5, 1.5, nx * ny)
    rd = builder.make_reg_data(-1.0, 1.0, -1.0, 1.0, scale=scale)
    assert rd.scale is not None
    assert rd.scale.shape == (nx * ny,)


@pytest.mark.unit
@pytest.mark.parametrize("reg_type", ["first-order", "second-order"])
def test_matvec_free_nonuniform_scale_matches_dense(reg_type):
    """matvec_free with non-uniform scale must match dense R(scale) @ s.

    This directly validates the edge-weighting (geometric-mean) math, which
    the uniform-scale matvec test cannot exercise: a bug in _geom_mean /
    _geom_mean3 or in the scale_x/scale_y off-diagonal scaling would only
    surface with a non-constant source and non-uniform scale.
    """
    nx, ny = 5, 7  # non-square so dx != dy
    builder = DenseRegularizationBuilder(nx, ny, reg_type)
    xmin, xmax, ymin, ymax = -1.0, 2.0, -0.5, 1.5
    scale = jnp.linspace(0.3, 2.0, nx * ny)  # non-uniform
    s = jnp.linspace(-1.0, 1.0, nx * ny)     # non-constant

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
    nx = ny = 9  # smaller than default block_size=10
    builder = DenseRegularizationBuilder(nx, ny, reg_type)
    xmin, xmax, ymin, ymax = -2.0, 2.0, -1.0, 3.0
    scale = jnp.linspace(0.3, 2.0, nx * ny)

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
    gp_source = PixelizedSourceModel(
        nx=5, ny=5, regularization_type="gaussian",
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
@pytest.mark.parametrize("nx,ny", [(2, 5), (5, 2)])
@pytest.mark.parametrize("reg_type", ["first-order", "second-order"])
def test_matvec_free_degenerate_grid_matches_dense(nx, ny, reg_type):
    """matvec_free must match dense R @ s on degenerate (thin) grids.

    For nx=2 the second-order full-curvature stencil (nx > 2) is skipped and
    only the near-boundary first-gradient fallback runs.  This guards the
    ``if self.nx > 2`` / ``if self.ny > 2`` boundary branches.
    """
    builder = DenseRegularizationBuilder(nx, ny, reg_type)
    xmin, xmax, ymin, ymax = -1.0, 1.0, -0.5, 1.5

    reg_dense, _ = builder.matrix(xmin, xmax, ymin, ymax)
    s = jnp.linspace(-1.0, 1.0, nx * ny)

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
    nx, ny = 10, 9  # non-square, multiple blocks with block_size=4
    builder = DenseRegularizationBuilder(nx, ny, reg_type)
    xmin, xmax, ymin, ymax = -2.0, 2.0, -1.0, 3.0
    scale = jnp.linspace(0.3, 2.0, nx * ny) if use_scale else None

    R_full, _ = builder.matrix(xmin, xmax, ymin, ymax, scale=scale)

    # Check several blocks: interior, edge, and corner blocks
    for x_s, x_e, y_s, y_e in [(2, 6, 2, 6), (0, 4, 0, 4), (6, 10, 5, 9)]:
        bf = jnp.asarray(
            [x + y * nx for y in range(y_s, y_e) for x in range(x_s, x_e)],
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
