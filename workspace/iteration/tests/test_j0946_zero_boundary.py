"""Boundary continuity and operator/evidence regressions for J0946 experiments."""

import importlib.util
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

from TinyLensGpu.ForwardSimulation import SimulatorConfig
from TinyLensGpu.ForwardSimulation import PotentialCorrectionRenderer
from TinyLensGpu.ObservationModel import IterativePotentialCorrection
from TinyLensGpu.PhysicalModel import PhysicalModel, Shear
from TinyLensGpu.utils.lensing.mapping import lens_mapping_operator_bilinear_from


HERE = Path(__file__).resolve().parents[1]


def load_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_real_renderer_is_continuous_across_old_source_edge():
    boundary = load_module("source_boundary.py", "zero_boundary_continuity_test")
    with jax.experimental.enable_x64(True):
        config = SimulatorConfig(.1, 1, jnp.ones((1, 1)), nsub=1,
                                 mask=np.zeros((1, 1), dtype=bool))
        renderer = boundary.ZeroRingRenderer(
            PhysicalModel(lens_mass=[Shear(gamma1=0., gamma2=0.)],
                          source_light=[], lens_light=[]),
            config, 3, (-.105, -.005, -.05, .05), dtype="float64")
        source = jnp.full(9, .04)

        def image(position):
            potential = -position * renderer.potential_grid.x
            return float(renderer.source_forward(source, renderer.source_mapping(potential))[0])

        differences = [abs(image(-.005-h) - image(-.005+h)) for h in (1e-5, 1e-7)]
        assert differences[-1] < 1e-6, f"Brightness jumps across old edge: {differences}"
        assert differences[-1] < differences[0] / 50
        assert renderer.n_source == 9


def test_zero_ring_matches_explicit_fixed_padding_and_preserves_interior():
    boundary = load_module("source_boundary.py", "zero_boundary_padding_test")
    with jax.experimental.enable_x64(True):
        rng = np.random.default_rng(43)
        source = jnp.asarray(rng.normal(size=(5, 5)))
        points = jnp.asarray(np.r_[rng.uniform(-.39, .39, (70, 2)),
                                    [[-.3, 0], [.3, 0], [0, -.3], [0, .3]]])
        bbox = tuple(jnp.asarray(x) for x in (-.2, .2, -.2, .2))
        weights, indices, _ = boundary.zero_ring_mapping(points, bbox, 5)
        actual = jnp.sum(weights * source.ravel()[indices], axis=1)
        padded = jnp.pad(source, 1).ravel()
        w, idx, _ = lens_mapping_operator_bilinear_from(points, -.3, .3, -.3, .3, 7)
        expected = jnp.sum(w * padded[idx], axis=1)
        np.testing.assert_allclose(actual, expected, atol=2e-14)
        interior = jnp.asarray(rng.uniform(-.19, .19, (40, 2)))
        w0, i0, _ = lens_mapping_operator_bilinear_from(interior, *bbox, 5)
        w1, i1, _ = boundary.zero_ring_mapping(interior, bbox, 5)
        np.testing.assert_allclose(jnp.sum(w0 * source.ravel()[i0], axis=1),
                                   jnp.sum(w1 * source.ravel()[i1], axis=1), atol=1e-14)


def test_zero_ring_analytic_gradient_matches_ad_in_buffer_and_interior():
    boundary = load_module("source_boundary.py", "zero_boundary_gradient_test")
    with jax.experimental.enable_x64(True):
        source = jnp.asarray(np.random.default_rng(12).normal(size=25))
        points = jnp.array([[-.245, .031], [.243, -.087], [.034, .259],
                            [-.064, -.227], [.031, -.067], [.36, .04]])
        bbox = tuple(jnp.asarray(x) for x in (-.2, .2, -.2, .2))

        def light(beta):
            w, idx, _ = boundary.zero_ring_mapping(beta, bbox, 5)
            return jnp.sum(w * source[idx])

        ad = jax.grad(light)(points)
        gx, gy = boundary.zero_ring_gradient(source, points, bbox, 5)
        np.testing.assert_allclose(ad, jnp.stack((gx, gy), axis=1), atol=2e-13)


def tiny_pair():
    boundary = load_module("source_boundary.py", "zero_boundary_operator_test")
    kernel = jnp.array([[0., .1, 0.], [.1, .6, .1], [0., .1, 0.]])
    config = SimulatorConfig(.06, 3, kernel, nsub=1, mask=np.zeros((3, 3), bool))
    macro = PhysicalModel(lens_mass=[Shear(gamma1=0., gamma2=0.)],
                          source_light=[], lens_light=[])
    hard = PotentialCorrectionRenderer(macro, config, 3, (-.05, .05, -.05, .05), dtype="float64")
    zero = boundary.ZeroRingRenderer(macro, config, 3, hard.source_bbox,
                                     potential_grid=hard.potential_grid, dtype="float64")
    source = np.linspace(.01, .08, 9)
    data = np.arange(9).reshape(3, 3)*.005
    problems = []
    for renderer in (hard, zero):
        problem = IterativePotentialCorrection(renderer, data, np.full((3, 3), .01),
            source, lambda_amplitude=5., source_regularization="first-order")
        problem.source_scale = jnp.linspace(1., 2., 9)
        problems.append(problem)
    return problems


def test_joint_analytic_gradient_and_source_transpose_use_same_zero_boundary():
    with jax.experimental.enable_x64(True):
        _, problem = tiny_pair()
        rng = np.random.default_rng(39)
        q = problem.initial_values + jnp.asarray(rng.normal(size=problem.n_joint))*1e-6
        point = problem._evaluate_kernel(q, .2, 3.)

        def objective(x):
            residual = (problem.forward(x)-problem.data_active)/problem.noise_active
            return .5*(jnp.vdot(residual, residual) + jnp.vdot(x, problem.prior_matvec(x, .2, 3.)))

        np.testing.assert_allclose(jax.grad(objective)(q), -point.negative_gradient,
                                   rtol=1e-12, atol=1e-9)
        mapping = problem.linearize(q).mapping
        source = jnp.asarray(rng.normal(size=problem.n_source))
        image = jnp.asarray(rng.normal(size=problem.renderer.n_active))
        lhs = jnp.vdot(problem.renderer.source_forward(source, mapping), image)
        rhs = jnp.vdot(source, problem.renderer.source_adjoint(image, mapping))
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)


def test_shared_metric_explicit_start_and_endpoint_evidence_match_dense_reference():
    engine_module = load_module("lbfgs.py", "zero_boundary_engine_test")
    with jax.experimental.enable_x64(True):
        hard, zero = tiny_pair()
        reference = engine_module.EvidenceEngine(hard, maxiter=1)
        engine = engine_module.EvidenceEngine(zero, maxiter=1, reference_engine=reference)
        log_strengths = np.log10([.2, 3., 5.])
        strengths = jnp.asarray(10.**log_strengths)
        coordinates = reference.prepare_coordinates(strengths)
        initial = zero.initial_values + jnp.arange(zero.n_joint)*1e-7
        row, q = engine.fit(log_strengths, initial_values=initial, audit=True,
            initialization_label="explicit_test", prepared_coordinates=(strengths, coordinates))
        np.testing.assert_array_equal(engine.last_initial_values, initial)
        assert row["source_initialization"] == "explicit_test"
        assert row["shared_preconditioner_reference"]
        assert row["algebra_audit"]["whitening_pass"]
        assert row["algebra_audit"]["determinant_pass"]
        prior = zero.dense_prior(float(strengths[0]), float(strengths[1]))
        jacobian = jax.jacfwd(zero.forward)(jnp.asarray(q))/zero.noise_active[:, None]
        hessian = np.asarray(prior+jacobian.T@jacobian)
        ratio = np.linalg.slogdet(hessian)[1] - np.linalg.slogdet(np.asarray(prior))[1]
        expected = -row["objective"]-.5*(ratio+engine.noise_normalization)
        np.testing.assert_allclose(row["diagnostic_log_evidence"], expected, atol=1e-8)
        # The shared metric must not replace the physical source-only solve.
        jac = jax.jacfwd(zero.forward)(zero.initial_values)[:, :zero.n_source]/zero.noise_active[:, None]
        rhs = jac.T@(zero.data_active/zero.noise_active)
        expected_source = np.linalg.solve(np.asarray(prior[:9, :9]+jac.T@jac), np.asarray(rhs))
        np.testing.assert_allclose(engine.conditional_source(strengths[0])[:9], expected_source,
                                   rtol=1e-10, atol=1e-10)
